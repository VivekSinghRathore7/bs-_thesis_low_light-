"""
Stage 1: Patch-based Retinex training — retinex_patch/train.py

Architecture  : original P2 (IllumNet 1.86M + ReflecNet 4.19M + RefineBlock 0.11M = 6.16M)
Loss          : Charbonnier + SSIM only  (NO GAN — discriminator overfits 485 imgs)
Training data : 128×128 patches, 200/image → 97,000 samples/epoch
Expected PSNR : 23–25 dB (vs 22.29 current best)

Usage (from project root):
    conda run -n viv python src/retinex_patch/train.py
"""

import os, sys, json, math, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)

from models import IllumNet, ReflecNet, RefineBlock
from src.retinex_patch.dataset import PatchLOLDataset, FullLOLDataset

# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    data_root   = "datasets/LOL_dataset",
    save_dir    = "checkpoints/retinex_patch",
    log_dir     = "experiments/retinex_patch",

    patch       = 128,
    n_patches   = 200,         # patches per image per epoch
    batch_size  = 32,          # 97,000/32 = ~3,031 batches/epoch
    epochs      = 500,
    lr          = 2e-4,
    lr_min      = 1e-7,
    weight_decay= 1e-4,
    workers     = 8,
    save_every  = 50,
    eval_every  = 10,
    ema_decay   = 0.9999,

    # Loss weights (simple = stable on small datasets)
    lam_charb   = 1.0,
    lam_ssim    = 0.5,
    lam_vgg     = 0.1,         # light perceptual — prevents blur
    lam_ret     = 0.5,         # R×I consistency
)


# ── Losses ────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__(); self.eps = eps**2
    def forward(self, p, g):
        return torch.mean(torch.sqrt((p - g)**2 + self.eps))


class SSIMLoss(nn.Module):
    def __init__(self, ws=11, ch=3):
        super().__init__(); self.ws = ws; self.ch = ch
    def forward(self, p, g):
        C1, C2 = 0.01**2, 0.03**2
        d = torch.arange(self.ws, dtype=torch.float32, device=p.device) - self.ws//2
        g1d = torch.exp(-d**2/4.5); g1d /= g1d.sum()
        win = (g1d.unsqueeze(1)*g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(self.ch,1,1,1)
        pad = self.ws//2
        mu1 = F.conv2d(p, win, padding=pad, groups=self.ch)
        mu2 = F.conv2d(g, win, padding=pad, groups=self.ch)
        s1  = F.conv2d(p*p, win, padding=pad, groups=self.ch) - mu1**2
        s2  = F.conv2d(g*g, win, padding=pad, groups=self.ch) - mu2**2
        s12 = F.conv2d(p*g, win, padding=pad, groups=self.ch) - mu1*mu2
        return 1 - (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        vgg = M.vgg19(weights=M.VGG19_Weights.DEFAULT).features
        self.s = nn.Sequential(*list(vgg[:18])).eval()
        for p in self.parameters(): p.requires_grad=False
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self, p, g):
        p=(p-self.mean)/self.std; g=(g-self.mean)/self.std
        return F.l1_loss(self.s(p), self.s(g))


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for em, mp in zip(self.model.parameters(), model.parameters()):
            em.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)
        for eb, mb in zip(self.model.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1.0/(mse+1e-10))).mean().item()

def cosine_lr(opt, ep, total, lr_max, lr_min):
    lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*ep/total))
    for pg in opt.param_groups: pg['lr'] = lr


# ── Inference (full image, TTA optional) ──────────────────────────────────────

@torch.no_grad()
def run_model(illum, reflec, refine, inp7):
    """inp7: (B,7,H,W) → enhanced (B,3,H,W)"""
    I_low = inp7[:, :3]
    R_tv  = inp7[:, 3:6]
    L_tv  = inp7[:, 6:7]

    I_pred = illum(L_tv)
    R_pred = reflec(R_tv)
    recomb = (R_pred * I_pred).clamp(0,1)
    return refine(recomb)

@torch.no_grad()
def eval_psnr_ssim(illum, reflec, refine, loader, device):
    illum.eval(); reflec.eval(); refine.eval()
    psnrs = []
    ssim_fn = SSIMLoss()
    ssims = []
    for inp7, gt, _, _ in loader:
        inp7, gt = inp7.to(device), gt.to(device)
        pred = run_model(illum, reflec, refine, inp7).clamp(0,1)
        psnrs.append(psnr_fn(pred, gt))
        ssims.append(1 - ssim_fn(pred, gt).item())
    illum.train(); reflec.train(); refine.train()
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(ROOT)
    os.makedirs(CFG['save_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'],  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type=='cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── data ──────────────────────────────────────────────────────────────────
    train_ds = PatchLOLDataset(CFG['data_root'], split="our485",
                                patch=CFG['patch'], n_patches=CFG['n_patches'],
                                augment=True)
    eval_ds  = FullLOLDataset(CFG['data_root'],  split="eval15")

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                               shuffle=True, num_workers=CFG['workers'],
                               pin_memory=True, drop_last=True)
    eval_loader  = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    n_batches = len(train_loader)
    print(f"Train  : {len(train_ds):,} patches → {n_batches} batches/epoch")
    print(f"Eval   : {len(eval_ds)} full images")

    # ── models (original V1 architecture — 6.16M params total) ───────────────
    illum  = IllumNet(base_filters=32).to(device)
    reflec = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)

    n_params = sum(p.numel() for m in [illum,reflec,refine]
                   for p in m.parameters())/1e6
    print(f"Params : {n_params:.2f}M  (V1 original — correct ratio for 485 imgs)")

    ema_illum  = EMA(illum,  CFG['ema_decay'])
    ema_reflec = EMA(reflec, CFG['ema_decay'])
    ema_refine = EMA(refine, CFG['ema_decay'])

    # ── optimiser (single group for all three) ────────────────────────────────
    params_all = list(illum.parameters()) + list(reflec.parameters()) + list(refine.parameters())
    opt = torch.optim.AdamW(params_all, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    # ── losses ────────────────────────────────────────────────────────────────
    charb = CharbonnierLoss().to(device)
    ssim  = SSIMLoss().to(device)
    vgg   = VGGLoss().to(device)

    history = {'loss':[], 'psnr_eval':[], 'ssim_eval':[], 'lr':[]}
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(1, CFG['epochs']+1):
        cosine_lr(opt, epoch-1, CFG['epochs'], CFG['lr'], CFG['lr_min'])
        loss_sum = psnr_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{CFG['epochs']}", leave=False)
        for inp7, gt, R_tv, L_tv in pbar:
            inp7 = inp7.to(device); gt    = gt.to(device)
            R_tv = R_tv.to(device); L_tv  = L_tv.to(device)
            L_ch  = inp7[:, 6:7]   # illumination channel

            # Forward
            I_pred = illum(L_ch)
            R_pred = reflec(R_tv)
            recomb = (R_pred * I_pred).clamp(0, 1)
            pred   = refine(recomb)

            # Loss: Charbonnier + SSIM + light VGG + Retinex consistency
            l_charb = charb(pred, gt)
            l_ssim  = ssim(pred,  gt)
            l_vgg   = vgg(pred,   gt)
            l_ret   = charb(recomb, gt)  # physics consistency
            loss = (CFG['lam_charb'] * l_charb +
                    CFG['lam_ssim']  * l_ssim  +
                    CFG['lam_vgg']   * l_vgg   +
                    CFG['lam_ret']   * l_ret)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

            for e, m in [(ema_illum,illum),(ema_reflec,reflec),(ema_refine,refine)]:
                e.update(m)

            loss_sum += loss.item()
            psnr_sum += psnr_fn(pred.detach().clamp(0,1), gt)
            pbar.set_postfix(L=f"{loss.item():.4f}",
                             PSNR=f"{psnr_fn(pred.detach().clamp(0,1),gt):.2f}")

        ep_loss = loss_sum / n_batches
        ep_psnr = psnr_sum / n_batches
        cur_lr  = opt.param_groups[0]['lr']

        ep_eval_psnr, ep_eval_ssim = 0.0, 0.0
        if epoch % CFG['eval_every'] == 0:
            # Eval with EMA models
            ep_eval_psnr, ep_eval_ssim = eval_psnr_ssim(
                ema_illum.model, ema_reflec.model, ema_refine.model,
                eval_loader, device
            )
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | "
                  f"train_PSNR={ep_psnr:.2f} | "
                  f"eval_PSNR={ep_eval_psnr:.2f} dB | "
                  f"eval_SSIM={ep_eval_ssim:.4f} | lr={cur_lr:.2e}")

            if ep_eval_psnr > best_psnr:
                best_psnr = ep_eval_psnr
                _save(illum, reflec, refine,
                      ema_illum, ema_reflec, ema_refine,
                      os.path.join(CFG['save_dir'], 'best_psnr.pth'), epoch)
                print(f"  ★  New best PSNR: {best_psnr:.2f} dB")

            if ep_eval_ssim > best_ssim:
                best_ssim = ep_eval_ssim
                _save(illum, reflec, refine,
                      ema_illum, ema_reflec, ema_refine,
                      os.path.join(CFG['save_dir'], 'best_ssim.pth'), epoch)
        else:
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | train_PSNR={ep_psnr:.2f} | lr={cur_lr:.2e}")

        history['loss'].append(ep_loss)
        history['psnr_eval'].append(ep_eval_psnr)
        history['ssim_eval'].append(ep_eval_ssim)
        history['lr'].append(cur_lr)

        if epoch % CFG['save_every'] == 0:
            _save(illum, reflec, refine, ema_illum, ema_reflec, ema_refine,
                  os.path.join(CFG['save_dir'], f'epoch_{epoch:03d}.pth'), epoch)

        _save(illum, reflec, refine, ema_illum, ema_reflec, ema_refine,
              os.path.join(CFG['save_dir'], 'latest.pth'), epoch)

    with open(os.path.join(CFG['log_dir'], 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅  Stage 1 complete. Best eval PSNR: {best_psnr:.2f} dB")


def _save(illum, reflec, refine, ema_i, ema_r, ema_rb, path, epoch):
    torch.save(dict(
        epoch=epoch,
        illum=illum.state_dict(),
        reflec=reflec.state_dict(),
        refine=refine.state_dict(),
        ema_illum=ema_i.model.state_dict(),
        ema_reflec=ema_r.model.state_dict(),
        ema_refine=ema_rb.model.state_dict(),
    ), path)


if __name__ == "__main__":
    main()
