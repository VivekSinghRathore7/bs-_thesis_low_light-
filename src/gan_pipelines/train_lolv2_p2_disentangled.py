"""
Advanced training script for P2 on LOL-v2.
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from tqdm import tqdm

# Use ORIGINAL models (6.2M params - right-sized for 485 images)
from models_p1_p3_unet import IllumNet, ReflecNet, RefineBlock, PatchDiscriminator, CBAM
from losses_all_pipelines import SSIMLoss, VGGPerceptualLoss, Sobel, ReconstructionConsistencyLoss
from retinex_utils import decompose_retinex


# ============================================================
# Charbonnier Loss (replaces L1 - smoother gradients)
# ============================================================

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


# ============================================================
# Augmented P2 Dataset (adds flips+rotations to reduce overfitting)
# ============================================================

class AugP2Dataset(Dataset):
    def __init__(self, root_dir, split="our485", img_size=256, augment=True):
        self.low_dir = os.path.join(root_dir, split, "low")
        self.high_dir = os.path.join(root_dir, split, "high")
        self.img_size = img_size
        self.augment = augment

        filenames = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
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
        ratio = max(0.01, min(1.0, ratio))
        n = max(1, int(len(self.all_filenames) * ratio))
        self.filenames = self.all_filenames[:n]

    def __len__(self):
        return len(self.filenames)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img.astype(np.float32) / 255.0

    def _augment_pair(self, low, high):
        """Synchronized augmentation on low/high pair."""
        if not self.augment:
            return low, high
        # Random horizontal flip
        if random.random() > 0.5:
            low = np.fliplr(low).copy()
            high = np.fliplr(high).copy()
        # Random vertical flip
        if random.random() > 0.5:
            low = np.flipud(low).copy()
            high = np.flipud(high).copy()
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k > 0:
            low = np.rot90(low, k).copy()
            high = np.rot90(high, k).copy()
        return low, high

    def _to_tensor(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        else:
            img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load(os.path.join(self.low_dir, fname))
        high = self._load(os.path.join(self.high_dir, fname))

        low, high = self._augment_pair(low, high)

        R_low, I_low = decompose_retinex(low)
        R_high, I_high = decompose_retinex(high)

        return (
            self._to_tensor(R_low),
            self._to_tensor(I_low),
            self._to_tensor(R_high),
            self._to_tensor(I_high),
            self._to_tensor(high),
        )


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                    (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# ============================================================
# Reflectance Loss V2 (improved from original, same complexity)
# ============================================================

class ReflectanceLossV2(nn.Module):
    """
    Same 5 losses as original but with Charbonnier instead of L1:
    1. Noise-aware Charbonnier (was noise-aware L1)
    2. Gradient-guided texture loss (unchanged)
    3. Channel-decoupled Charbonnier (was L1)
    4. Frequency-domain loss (unchanged)
    5. Perceptual loss (unchanged)
    """
    def __init__(self, alpha=2.0, lambda_grad=0.5, w_b=1.2, w_g=1.0, w_r=1.5,
                 lambda_freq=0.1, lambda_percp=0.1):
        super().__init__()
        self.alpha = alpha
        self.lambda_grad = lambda_grad
        self.w_b, self.w_g, self.w_r = w_b, w_g, w_r
        self.lambda_freq = lambda_freq
        self.lambda_percp = lambda_percp
        self.charb = CharbonnierLoss()
        self.sobel = Sobel()
        self.vgg = VGGPerceptualLoss()

    def forward(self, R_pred, R_high, R_low=None):
        loss_dict = {}
        total = 0.0

        # 1. Noise-aware Charbonnier
        if R_low is not None:
            noise_weight = 1 + self.alpha * (1 - R_low)
            diff = torch.sqrt((R_pred - R_high) ** 2 + 1e-6)
            l_noise = torch.mean(noise_weight * diff)
        else:
            l_noise = self.charb(R_pred, R_high)
        loss_dict['noise_charb'] = l_noise
        total += l_noise

        # 2. Gradient-guided texture
        l_grad = F.l1_loss(self.sobel(R_pred), self.sobel(R_high))
        loss_dict['grad'] = l_grad
        total += self.lambda_grad * l_grad

        # 3. Channel-decoupled Charbonnier
        eps2 = 1e-6
        l_ch = (self.w_b * torch.mean(torch.sqrt((R_pred[:,0]-R_high[:,0])**2 + eps2)) +
                self.w_g * torch.mean(torch.sqrt((R_pred[:,1]-R_high[:,1])**2 + eps2)) +
                self.w_r * torch.mean(torch.sqrt((R_pred[:,2]-R_high[:,2])**2 + eps2)))
        loss_dict['ch_decoupled'] = l_ch
        total += l_ch

        # 4. Frequency-domain
        fft_pred = torch.fft.fft2(R_pred)
        fft_high = torch.fft.fft2(R_high)
        l_freq = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_high))
        loss_dict['freq'] = l_freq
        total += self.lambda_freq * l_freq

        # 5. Perceptual
        l_percp = self.vgg(R_pred, R_high)
        loss_dict['perceptual'] = l_percp
        total += self.lambda_percp * l_percp

        loss_dict['total'] = total
        return total, loss_dict


# ============================================================
# Combined Loss V2 (L1 -> Charbonnier, SSIM weight increased)
# ============================================================

class CombinedLossV2(nn.Module):
    """
    Charbonnier + SSIM(1.5) + VGG + Adversarial (replaced L1 with Charbonnier, increased SSIM weight)
    """
    def __init__(self, lambda_charb=1.0, lambda_ssim=1.5, lambda_vgg=0.1, lambda_adv=0.01):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_ssim = lambda_ssim
        self.lambda_vgg = lambda_vgg
        self.lambda_adv = lambda_adv
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.vgg = VGGPerceptualLoss()

    def forward(self, pred, target, disc_fake=None):
        loss = self.lambda_charb * self.charb(pred, target)
        loss += self.lambda_ssim * self.ssim(pred, target)
        loss += self.lambda_vgg * self.vgg(pred, target)
        if disc_fake is not None:
            adv_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake))
            loss += self.lambda_adv * adv_loss
        return loss


# ============================================================
# R1 gradient penalty
# ============================================================

def r1_gradient_penalty(disc, real, lambda_r1=10.0):
    real = real.requires_grad_(True)
    d_real = disc(real)
    grads = torch.autograd.grad(d_real.sum(), real, create_graph=True)[0]
    penalty = grads.view(grads.size(0), -1).norm(2, dim=1).pow(2).mean()
    return lambda_r1 * penalty


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- SAME original models (6.2M params, right-sized for 485 images) ----
    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)

    n_gen = sum(p.numel() for m in [illum_net, reflec_net, refine] for p in m.parameters())
    print(f"Generator params: {n_gen:,} (same as original)")

    # ---- EMA ----
    ema_illum = EMA(illum_net, decay=0.999)
    ema_reflec = EMA(reflec_net, decay=0.999)
    ema_refine = EMA(refine, decay=0.999)

    # ---- Improved losses ----
    criterion = CombinedLossV2(
        lambda_charb=1.0, lambda_ssim=1.5, lambda_vgg=0.1, lambda_adv=0.01
    ).to(device)
    criterion_reflec = ReflectanceLossV2(
        alpha=args.alpha_noise, lambda_grad=args.lambda_grad,
    ).to(device)
    criterion_recon = ReconstructionConsistencyLoss(lambda_l1=1.0, lambda_ssim=0.5).to(device)
    charb_loss = CharbonnierLoss().to(device)
    ssim_illum = SSIMLoss(window_size=11, channels=1).to(device)

    # ---- Optimizers with weight decay (regularization) ----
    gen_params = (list(illum_net.parameters()) +
                  list(reflec_net.parameters()) +
                  list(refine.parameters()))
    opt_g = torch.optim.AdamW(gen_params, lr=args.lr, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999),
                              weight_decay=1e-4)

    sched_g = CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-7)

    # ---- Dataset with augmentation ----
    train_ds = AugP2Dataset(args.data_root, split="our485",
                            img_size=args.img_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)

    # ---- Checkpoints ----
    ckpt_dir = os.path.join(args.checkpoint_dir, "pipeline2_v2")
    os.makedirs(ckpt_dir, exist_ok=True)

    WARMUP = args.warmup_epochs
    LR_WARMUP = 10

    history = {'g_loss': [], 'd_loss': [], 'illum_loss': [], 'reflec_loss': []}
    best_g_loss = float('inf')
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  Pipeline 2 V2 Training (Anti-Overfitting)")
    print(f"  Resolution: {args.img_size}x{args.img_size}")
    print(f"  Epochs: {args.epochs} | Warm-up: {WARMUP}")
    print(f"  Batch size: {args.batch_size} | Augment: YES")
    print(f"  LR: {args.lr} | Weight decay: 1e-4")
    print(f"  EMA: YES (decay=0.999)")
    print(f"  Losses: Charbonnier + SSIM(1.5) + VGG + Reflec(5-comp)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        # LR warm-up
        if epoch <= LR_WARMUP:
            wf = epoch / LR_WARMUP
            for pg in opt_g.param_groups:
                pg['lr'] = args.lr * wf
            for pg in opt_d.param_groups:
                pg['lr'] = args.lr * 0.5 * wf

        is_warmup = epoch <= WARMUP

        # Curriculum
        curriculum_cap = max(1, int(args.epochs * 0.6))
        ratio = min(1.0, 0.2 + 0.8 * ((epoch - 1) / curriculum_cap))
        train_ds.set_curriculum_ratio(ratio)

        illum_net.train(); reflec_net.train(); refine.train(); disc.train()
        g_sum, d_sum, il_sum, rl_sum = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        phase = "WARM" if is_warmup else "GAN"
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{args.epochs} [{phase}]")

        for R_low, I_low, R_high, I_high, gt in pbar:
            R_low = R_low.to(device)
            I_low = I_low.to(device)
            R_high = R_high.to(device)
            I_high = I_high.to(device)
            gt = gt.to(device)

            # Forward
            I_pred = illum_net(I_low)
            R_pred = reflec_net(R_low)
            recombined = R_pred * I_pred
            output = refine(recombined)

            # ---- Discriminator (skip warm-up) ----
            d_loss_val = 0.0
            if not is_warmup:
                d_real = disc(gt)
                d_fake = disc(output.detach())
                d_loss = 0.5 * (
                    F.mse_loss(d_real, torch.ones_like(d_real)) +
                    F.mse_loss(d_fake, torch.zeros_like(d_fake))
                )
                if torch.rand(1).item() < 0.0625:
                    d_loss = d_loss + r1_gradient_penalty(disc, gt.detach())
                opt_d.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                opt_d.step()
                d_loss_val = d_loss.item()

            # ---- Generator ----
            # Illumination loss: Charbonnier + SSIM
            loss_I = charb_loss(I_pred, I_high) + 0.5 * ssim_illum(I_pred, I_high)

            # Reflectance loss (5-component)
            loss_R, _ = criterion_reflec(R_pred, R_high, R_low)

            # Reconstruction consistency
            loss_recon = criterion_recon(recombined, gt)

            if is_warmup:
                loss_final = charb_loss(output, gt)
                g_loss = loss_final + 0.5 * loss_I + loss_R + 0.5 * loss_recon
            else:
                d_fake = disc(output)
                loss_final = criterion(output, gt, d_fake)
                g_loss = loss_final + 0.5 * loss_I + loss_R + 0.5 * loss_recon

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
            opt_g.step()

            # Update EMA
            ema_illum.update(illum_net)
            ema_reflec.update(reflec_net)
            ema_refine.update(refine)

            g_sum += g_loss.item()
            d_sum += d_loss_val
            il_sum += loss_I.item()
            rl_sum += loss_R.item()
            n_batches += 1

            pbar.set_postfix(G=f'{g_loss.item():.4f}', D=f'{d_loss_val:.4f}')

        # Step schedulers
        if epoch > LR_WARMUP:
            sched_g.step()
            sched_d.step()

        avg_g = g_sum / max(n_batches, 1)
        avg_d = d_sum / max(n_batches, 1)
        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)
        history['illum_loss'].append(il_sum / max(n_batches, 1))
        history['reflec_loss'].append(rl_sum / max(n_batches, 1))

        elapsed = (time.time() - start_time) / 60
        print(f"  Ep {epoch}: G={avg_g:.4f} D={avg_d:.4f} "
              f"I={il_sum/max(n_batches,1):.4f} R={rl_sum/max(n_batches,1):.4f} "
              f"[{elapsed:.1f}min]")

        # Save best + periodic
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            _save(illum_net, reflec_net, refine, disc,
                  ema_illum, ema_reflec, ema_refine, ckpt_dir, "best")

        if epoch % 50 == 0 or epoch == args.epochs:
            _save(illum_net, reflec_net, refine, disc,
                  ema_illum, ema_reflec, ema_refine, ckpt_dir, f"ep{epoch}")

    # Save final
    _save(illum_net, reflec_net, refine, disc,
          ema_illum, ema_reflec, ema_refine, ckpt_dir, "final")

    with open(os.path.join(ckpt_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training complete! {total_time:.1f} min")
    print(f"   Checkpoints: {ckpt_dir}/")


def _save(illum_net, reflec_net, refine, disc,
          ema_illum, ema_reflec, ema_refine, ckpt_dir, tag):
    # Regular
    torch.save({
        'illum_net': illum_net.state_dict(),
        'reflec_net': reflec_net.state_dict(),
        'refine': refine.state_dict(),
        'disc': disc.state_dict(),
    }, os.path.join(ckpt_dir, f"gen_{tag}.pth"))

    # EMA
    ema_illum.apply_shadow(illum_net)
    ema_reflec.apply_shadow(reflec_net)
    ema_refine.apply_shadow(refine)
    torch.save({
        'illum_net': illum_net.state_dict(),
        'reflec_net': reflec_net.state_dict(),
        'refine': refine.state_dict(),
    }, os.path.join(ckpt_dir, f"gen_{tag}_ema.pth"))
    ema_illum.restore(illum_net)
    ema_reflec.restore(reflec_net)
    ema_refine.restore(refine)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P2 V2")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--alpha_noise', type=float, default=2.0)
    parser.add_argument('--lambda_grad', type=float, default=0.5)
    args = parser.parse_args()
    train(args)
