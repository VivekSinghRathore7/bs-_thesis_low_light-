"""
Master script for LOL-v2 Real.
"""

import os, sys, json, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from retinex_utils import decompose_retinex, get_all_enhanced_illuminations, enhance_clahe
from models_p1_p3_unet import UNetGenerator, IllumNet, ReflecNet, RefineBlock, PatchDiscriminator
from losses_all_pipelines import CombinedLoss, ReflectanceLoss, ReconstructionConsistencyLoss

print(f"Working dir: {os.getcwd()}", flush=True)
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

# ============================================================
# LOL-v2 Datasets (adapted from dataset.py for LOL-v2 folder structure)
# ============================================================

class BaseLOLv2Dataset(Dataset):
    def __init__(self, root_dir, split="Train", img_size=256):
        self.low_dir = os.path.join(root_dir, split, "Input")
        self.high_dir = os.path.join(root_dir, split, "GT")
        self.img_size = img_size
        filenames = sorted([f for f in os.listdir(self.low_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        brightness_list = []
        for f in filenames:
            path = os.path.join(self.low_dir, f)
            img = cv2.imread(path)
            brightness = img.mean() if img is not None else 0
            brightness_list.append((f, brightness))
        brightness_list.sort(key=lambda x: x[1], reverse=True)
        self.all_filenames = [x[0] for x in brightness_list]
        self.filenames = list(self.all_filenames)

    def set_curriculum_ratio(self, ratio):
        self.filenames = self.all_filenames[:max(1, int(len(self.all_filenames) * max(0.01, min(1.0, ratio))))]

    def __len__(self): return len(self.filenames)
    def _load_and_resize(self, path):
        img = cv2.resize(cv2.imread(path), (self.img_size, self.img_size))
        return img.astype(np.float32) / 255.0
    def _to_tensor(self, img):
        img = img[np.newaxis, :] if img.ndim == 2 else img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

class P1LOLv2Dataset(BaseLOLv2Dataset):
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))
        _, I_low = decompose_retinex(low)
        I_enh = enhance_clahe((I_low * 255).astype(np.uint8)).astype(np.float32) / 255.0
        inp = np.concatenate([low, I_enh[:, :, np.newaxis]], axis=2)
        return self._to_tensor(inp), self._to_tensor(high)

class P2LOLv2Dataset(BaseLOLv2Dataset):
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))
        R_low, I_low = decompose_retinex(low)
        R_high, I_high = decompose_retinex(high)
        return self._to_tensor(R_low), self._to_tensor(I_low), self._to_tensor(R_high), self._to_tensor(I_high), self._to_tensor(high)

class P3LOLv2Dataset(BaseLOLv2Dataset):
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))
        R_low, I_low = decompose_retinex(low)
        enhanced_maps = get_all_enhanced_illuminations(I_low)
        candidates = [np.clip(R_low * enhanced_maps[k][:, :, np.newaxis], 0, 1) for k in ['clahe', 'agc', 'glagc', 'log']]
        inp = np.concatenate([low] + candidates, axis=2)
        return self._to_tensor(inp), self._to_tensor(high)


# ============================================================
# Metrics
# ============================================================

def calc_psnr(pred, gt):
    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / (mse + 1e-10))

def calc_ssim(pred, gt, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = pred.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(gt, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])

class LPIPS_VGG:
    def __init__(self, device):
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]]).eval().to(device)
        for p in self.slices.parameters():
            p.requires_grad = False
        self.device = device

    def __call__(self, pred, gt):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        pred_n, gt_n = (pred - mean) / std, (gt - mean) / std
        diffs = []
        x_p, x_g = pred_n, gt_n
        for s in self.slices:
            x_p, x_g = s(x_p), s(x_g)
            x_p_n = x_p / (x_p.norm(dim=1, keepdim=True) + 1e-10)
            x_g_n = x_g / (x_g.norm(dim=1, keepdim=True) + 1e-10)
            diffs.append((x_p_n - x_g_n).pow(2).mean(dim=[1, 2, 3]))
        return torch.stack(diffs).mean(dim=0)


# ============================================================
# Pipeline 1: Dual-Input Conditional GAN
# ============================================================

def train_p1(data_root, save_dir, epochs=200, batch_size=4, lr=2e-4, img_size=256):
    print(f"\n{'='*60}", flush=True)
    print(f"  PIPELINE 1: Dual-Input cGAN on LOL-v2 Real", flush=True)
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    os.makedirs(save_dir, exist_ok=True)

    train_ds = P1LOLv2Dataset(data_root, split="Train", img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    gen = UNetGenerator(in_ch=4, out_ch=3, base_filters=64).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = CombinedLoss().to(device)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        gen.train(); disc.train()
        g_sum, d_sum = 0.0, 0.0
        for inp, gt in train_loader:
            inp, gt = inp.to(device), gt.to(device)
            fake = gen(inp).detach()
            d_loss = 0.5 * (F.mse_loss(disc(gt), torch.ones_like(disc(gt))) +
                            F.mse_loss(disc(fake), torch.zeros_like(disc(fake))))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            fake = gen(inp)
            g_loss = criterion(fake, gt, disc(fake))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            g_sum += g_loss.item(); d_sum += d_loss.item()

        n = len(train_loader)
        if epoch % 20 == 0 or epoch == epochs:
            elapsed = (time.time() - t0) / 60
            print(f"  P1 Ep {epoch}/{epochs}: G={g_sum/n:.4f} D={d_sum/n:.4f} [{elapsed:.1f}min]", flush=True)

    torch.save(gen.state_dict(), os.path.join(save_dir, "gen_final.pth"))
    print(f"  P1 Done in {(time.time()-t0)/60:.1f} min. Saved to {save_dir}", flush=True)


# ============================================================
# Pipeline 2: Disentangled GAN
# ============================================================

def train_p2(data_root, save_dir, epochs=200, batch_size=4, lr=2e-4, img_size=256):
    print(f"\n{'='*60}", flush=True)
    print(f"  PIPELINE 2: Disentangled GAN on LOL-v2 Real", flush=True)
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    os.makedirs(save_dir, exist_ok=True)
    WARMUP = 20

    train_ds = P2LOLv2Dataset(data_root, split="Train", img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)

    gen_params = list(illum_net.parameters()) + list(reflec_net.parameters()) + list(refine.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr * 0.5, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-7)

    criterion = CombinedLoss().to(device)
    criterion_reflec = ReflectanceLoss(alpha=2.0, lambda_grad=0.5).to(device)
    criterion_recon = ReconstructionConsistencyLoss().to(device)
    l1_loss = nn.L1Loss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        is_warmup = epoch <= WARMUP
        curriculum_ratio = min(1.0, 0.2 + 0.8 * ((epoch - 1) / max(1, int(epochs * 0.6))))
        train_ds.set_curriculum_ratio(curriculum_ratio)

        illum_net.train(); reflec_net.train(); refine.train(); disc.train()
        g_sum, d_sum = 0.0, 0.0

        for R_low, I_low, R_high, I_high, gt in train_loader:
            R_low, I_low = R_low.to(device), I_low.to(device)
            R_high, I_high = R_high.to(device), I_high.to(device)
            gt = gt.to(device)

            I_pred = illum_net(I_low)
            R_pred = reflec_net(R_low)
            recombined = R_pred * I_pred
            output = refine(recombined)

            d_loss_val = 0.0
            if not is_warmup:
                d_real, d_fake = disc(gt), disc(output.detach())
                d_loss = 0.5 * (F.mse_loss(d_real, torch.ones_like(d_real)) +
                                F.mse_loss(d_fake, torch.zeros_like(d_fake)))
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()
                d_loss_val = d_loss.item()

            loss_I = l1_loss(I_pred, I_high)
            loss_R, _ = criterion_reflec(R_pred, R_high, R_low)
            loss_recon = criterion_recon(recombined, gt)

            if is_warmup:
                g_loss = l1_loss(output, gt) + 0.5 * loss_I + loss_R + 0.5 * loss_recon
            else:
                d_fake = disc(output)
                loss_final = criterion(output, gt, d_fake)
                g_loss = loss_final + 0.5 * loss_I + loss_R + 0.5 * loss_recon

            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            g_sum += g_loss.item(); d_sum += d_loss_val

        sched_g.step(); sched_d.step()
        n = len(train_loader)
        if epoch % 20 == 0 or epoch == epochs:
            elapsed = (time.time() - t0) / 60
            phase = "WARM" if is_warmup else "GAN"
            print(f"  P2 [{phase}] Ep {epoch}/{epochs}: G={g_sum/n:.4f} D={d_sum/n:.4f} [{elapsed:.1f}min]", flush=True)

    torch.save({
        'illum_net': illum_net.state_dict(),
        'reflec_net': reflec_net.state_dict(),
        'refine': refine.state_dict(),
    }, os.path.join(save_dir, "gen_final.pth"))
    print(f"  P2 Done in {(time.time()-t0)/60:.1f} min. Saved to {save_dir}", flush=True)


# ============================================================
# Pipeline 3: Ensemble GAN
# ============================================================

def train_p3(data_root, save_dir, epochs=200, batch_size=4, lr=2e-4, img_size=256):
    print(f"\n{'='*60}", flush=True)
    print(f"  PIPELINE 3: Ensemble GAN on LOL-v2 Real", flush=True)
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    os.makedirs(save_dir, exist_ok=True)

    train_ds = P3LOLv2Dataset(data_root, split="Train", img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    gen = UNetGenerator(in_ch=15, out_ch=3, base_filters=64).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = CombinedLoss().to(device)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        gen.train(); disc.train()
        g_sum, d_sum = 0.0, 0.0
        for inp, gt in train_loader:
            inp, gt = inp.to(device), gt.to(device)
            fake = gen(inp).detach()
            d_loss = 0.5 * (F.mse_loss(disc(gt), torch.ones_like(disc(gt))) +
                            F.mse_loss(disc(fake), torch.zeros_like(disc(fake))))
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            fake = gen(inp)
            g_loss = criterion(fake, gt, disc(fake))
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()
            g_sum += g_loss.item(); d_sum += d_loss.item()

        n = len(train_loader)
        if epoch % 20 == 0 or epoch == epochs:
            elapsed = (time.time() - t0) / 60
            print(f"  P3 Ep {epoch}/{epochs}: G={g_sum/n:.4f} D={d_sum/n:.4f} [{elapsed:.1f}min]", flush=True)

    torch.save(gen.state_dict(), os.path.join(save_dir, "gen_final.pth"))
    print(f"  P3 Done in {(time.time()-t0)/60:.1f} min. Saved to {save_dir}", flush=True)


# ============================================================
# Evaluation (all 3 pipelines on LOL-v2 Test)
# ============================================================

@torch.no_grad()
def evaluate_all(data_root, ckpt_base, output_dir, img_size=256):
    print(f"\n{'='*60}", flush=True)
    print(f"  EVALUATING ALL PIPELINES on LOL-v2 Real Test", flush=True)
    print(f"{'='*60}", flush=True)

    device = torch.device('cuda')
    os.makedirs(output_dir, exist_ok=True)
    lpips_fn = LPIPS_VGG(device)

    all_results = {}

    # --- P1 ---
    print("\n  Evaluating P1...", flush=True)
    gen1 = UNetGenerator(in_ch=4, out_ch=3).to(device)
    gen1.load_state_dict(torch.load(os.path.join(ckpt_base, "p1_lolv2/gen_final.pth"), map_location=device, weights_only=True))
    gen1.eval()
    ds1 = P1LOLv2Dataset(data_root, split="Test", img_size=img_size)
    psnrs, ssims, lpipss = [], [], []
    for i in range(len(ds1)):
        inp, gt = ds1[i]
        inp, gt = inp.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
        pred = gen1(inp).clamp(0, 1)
        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())
    all_results['p1'] = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}
    print(f"  P1: PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f} LPIPS={np.mean(lpipss):.4f}", flush=True)

    # --- P2 ---
    print("  Evaluating P2...", flush=True)
    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)
    ckpt = torch.load(os.path.join(ckpt_base, "p2_lolv2/gen_final.pth"), map_location=device, weights_only=True)
    illum_net.load_state_dict(ckpt['illum_net'])
    reflec_net.load_state_dict(ckpt['reflec_net'])
    refine.load_state_dict(ckpt['refine'])
    illum_net.eval(); reflec_net.eval(); refine.eval()
    ds2 = P2LOLv2Dataset(data_root, split="Test", img_size=img_size)
    psnrs, ssims, lpipss = [], [], []
    for i in range(len(ds2)):
        R_low, I_low, R_high, I_high, gt = ds2[i]
        R_low = R_low.unsqueeze(0).to(device)
        I_low = I_low.unsqueeze(0).to(device)
        gt = gt.unsqueeze(0).to(device)
        I_pred = illum_net(I_low)
        R_pred = reflec_net(R_low)
        pred = refine(R_pred * I_pred).clamp(0, 1)
        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())
    all_results['p2'] = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}
    print(f"  P2: PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f} LPIPS={np.mean(lpipss):.4f}", flush=True)

    # --- P3 ---
    print("  Evaluating P3...", flush=True)
    gen3 = UNetGenerator(in_ch=15, out_ch=3).to(device)
    gen3.load_state_dict(torch.load(os.path.join(ckpt_base, "p3_lolv2/gen_final.pth"), map_location=device, weights_only=True))
    gen3.eval()
    ds3 = P3LOLv2Dataset(data_root, split="Test", img_size=img_size)
    psnrs, ssims, lpipss = [], [], []
    for i in range(len(ds3)):
        inp, gt = ds3[i]
        inp, gt = inp.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)
        pred = gen3(inp).clamp(0, 1)
        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())
    all_results['p3'] = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}
    print(f"  P3: PSNR={np.mean(psnrs):.2f} SSIM={np.mean(ssims):.4f} LPIPS={np.mean(lpipss):.4f}", flush=True)

    # Save results
    with open(os.path.join(output_dir, "metrics_lolv2_gan.json"), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*65}", flush=True)
    print(f"  LOL-v2 Real Test Set — GAN Pipeline Results", flush=True)
    print(f"{'='*65}", flush=True)
    print(f"  {'Pipeline':<25} {'PSNR↑':>10} {'SSIM↑':>10} {'LPIPS↓':>10}", flush=True)
    print(f"  {'-'*55}", flush=True)
    names = ['P1 (Dual-Input)', 'P2 (Disentangled)', 'P3 (Ensemble)']
    for i, name in enumerate(names):
        key = f'p{i+1}'
        p = np.mean(all_results[key]['psnr'])
        s = np.mean(all_results[key]['ssim'])
        l = np.mean(all_results[key]['lpips'])
        print(f"  {name:<25} {p:>10.2f} {s:>10.4f} {l:>10.4f}", flush=True)
    print(f"  {'-'*55}", flush=True)
    print(f"  RetinexRestormer       {'20.50':>10} {'0.8767':>10} {'0.1325':>10}  (already evaluated)", flush=True)
    print(f"{'='*65}", flush=True)

    return all_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    DATA_ROOT = "datasets/LOL_v2_real"
    CKPT_BASE = "checkpoints"
    OUTPUT_DIR = "results/lolv2/metrics"
    EPOCHS = 200
    BATCH_SIZE = 4
    IMG_SIZE = 256

    total_start = time.time()

    # Train all 3 pipelines
    train_p1(DATA_ROOT, os.path.join(CKPT_BASE, "p1_lolv2"), epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    train_p2(DATA_ROOT, os.path.join(CKPT_BASE, "p2_lolv2"), epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    train_p3(DATA_ROOT, os.path.join(CKPT_BASE, "p3_lolv2"), epochs=EPOCHS, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # Evaluate all
    evaluate_all(DATA_ROOT, CKPT_BASE, OUTPUT_DIR, img_size=IMG_SIZE)

    total_time = (time.time() - total_start) / 60
    print(f"\n✅ ALL DONE! Total time: {total_time:.1f} min ({total_time/60:.1f} hours)", flush=True)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
