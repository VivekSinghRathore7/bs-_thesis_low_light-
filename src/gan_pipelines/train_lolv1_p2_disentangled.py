"""
train_p2_improved.py
Improved Pipeline 2 training script with:
- Deeper architectures (IllumNetV2, ReflecNetV2, MultiScaleRefineBlock)
- EMA weights
- Advanced losses (Charbonnier, LPIPS, MS-SSIM, Color consistency)
- Data augmentation (flips, rotations, random crops)
- Curriculum learning + extended warm-up
- 384x384 resolution
- Dual-scale conditional discriminator

Usage:
    conda run -n viv python train_p2_improved.py \
        --data_root datasets/LOL_dataset \
        --epochs 400 --img_size 384 --batch_size 4
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

# Local imports
from retinex_utils import decompose_retinex
from models_p2_disentangled import (
    IllumNetV2, ReflecNetV2, MultiScaleRefineBlock,
    ConditionalPatchDiscriminator, DualScaleDiscriminator, EMA
)
from losses_all_pipelines import (
    CharbonnierLoss, SSIMLoss, MultiScaleSSIMLoss,
    VGGPerceptualLoss, LPIPSLoss, ColorConsistencyLoss,
    ReflectanceLossV2, CombinedLossV2, ReconstructionConsistencyLoss,
    Sobel
)


# ============================================================
# Augmented Dataset
# ============================================================

class AugmentedPipeline2Dataset(Dataset):
    """
    Pipeline 2 dataset with augmentation:
    - Random horizontal/vertical flips
    - Random 90° rotations
    - Random crops from original resolution (if larger than img_size)
    """

    def __init__(self, root_dir, split="our485", img_size=384, augment=True):
        self.low_dir = os.path.join(root_dir, split, "low")
        self.high_dir = os.path.join(root_dir, split, "high")
        self.img_size = img_size
        self.augment = augment

        filenames = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Curriculum: sort by brightness
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
        return img.astype(np.float32) / 255.0

    def _augment_pair(self, low, high):
        """Apply synchronized augmentations to low/high pair."""
        if not self.augment:
            low = cv2.resize(low, (self.img_size, self.img_size))
            high = cv2.resize(high, (self.img_size, self.img_size))
            return low, high

        h, w = low.shape[:2]

        # Random crop if image is larger than target
        if h >= self.img_size and w >= self.img_size:
            top = random.randint(0, h - self.img_size)
            left = random.randint(0, w - self.img_size)
            low = low[top:top+self.img_size, left:left+self.img_size]
            high = high[top:top+self.img_size, left:left+self.img_size]
        else:
            low = cv2.resize(low, (self.img_size, self.img_size))
            high = cv2.resize(high, (self.img_size, self.img_size))

        # Random horizontal flip
        if random.random() > 0.5:
            low = np.fliplr(low).copy()
            high = np.fliplr(high).copy()

        # Random vertical flip
        if random.random() > 0.5:
            low = np.flipud(low).copy()
            high = np.flipud(high).copy()

        # Random 90° rotation
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
            self._to_tensor(low),  # Also return low image for discriminator
        )


# ============================================================
# Training
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_pipeline2_improved(args):
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Models ---- #
    illum_net = IllumNetV2(base_filters=32).to(device)
    reflec_net = ReflecNetV2(base_filters=64).to(device)
    refine = MultiScaleRefineBlock(base_filters=64).to(device)
    disc = DualScaleDiscriminator(in_ch=6, base_filters=64).to(device)

    # ---- EMA ---- #
    ema_illum = EMA(illum_net, decay=0.999)
    ema_reflec = EMA(reflec_net, decay=0.999)
    ema_refine = EMA(refine, decay=0.999)

    # ---- Losses ---- #
    illum_loss_fn = CharbonnierLoss().to(device)
    illum_ssim_fn = SSIMLoss(window_size=11, channels=1).to(device)
    reflec_loss_fn = ReflectanceLossV2(
        alpha=args.alpha_noise,
        lambda_grad=args.lambda_grad,
        lambda_percp=0.15,
    ).to(device)
    recon_loss_fn = ReconstructionConsistencyLoss().to(device)
    combined_loss_fn = CombinedLossV2(
        lambda_charb=1.0,
        lambda_msssim=1.5,
        lambda_vgg=0.1,
        lambda_lpips=0.5,
        lambda_color=0.3,
        lambda_adv=args.lambda_adv,
    ).to(device)

    # ---- Optimizers ---- #
    gen_params = (list(illum_net.parameters()) +
                  list(reflec_net.parameters()) +
                  list(refine.parameters()))
    opt_G = torch.optim.AdamW(gen_params, lr=args.lr_g, betas=(0.5, 0.999),
                              weight_decay=1e-4)
    opt_D = torch.optim.AdamW(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999),
                              weight_decay=1e-4)

    # ---- Schedulers ---- #
    sched_G = CosineAnnealingLR(opt_G, T_max=args.epochs, eta_min=1e-6)
    sched_D = CosineAnnealingLR(opt_D, T_max=args.epochs, eta_min=1e-6)

    # ---- Dataset ---- #
    train_ds = AugmentedPipeline2Dataset(
        args.data_root, split="our485", img_size=args.img_size, augment=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)

    # ---- Checkpoint dir ---- #
    ckpt_dir = os.path.join(args.checkpoint_dir, "pipeline2_improved")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Training history ---- #
    history = {
        'g_loss': [], 'd_loss': [], 'psnr_train': [],
        'illum_loss': [], 'reflec_loss': [], 'recon_loss': [],
    }

    print(f"\n{'='*60}")
    print(f"  Improved Pipeline 2 Training")
    print(f"  Resolution: {args.img_size}×{args.img_size}")
    print(f"  Epochs: {args.epochs} | Warm-up: {args.warmup_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR (G): {args.lr_g} | LR (D): {args.lr_d}")
    print(f"  Grad accumulation: {args.grad_accum} steps")
    print(f"{'='*60}\n")

    best_g_loss = float('inf')
    start_time = time.time()

    for epoch in range(args.epochs):
        illum_net.train()
        reflec_net.train()
        refine.train()
        disc.train()

        # ---- Curriculum learning ---- #
        curriculum_ratio = min(1.0, 0.3 + 0.7 * (epoch / (args.epochs * 0.6)))
        train_ds.set_curriculum_ratio(curriculum_ratio)

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_illum = 0.0
        epoch_reflec = 0.0
        epoch_recon = 0.0
        n_batches = 0

        is_warmup = epoch < args.warmup_epochs
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs}"
                    f"{' [WARM]' if is_warmup else ''}")

        for batch_idx, (R_low, I_low, R_high, I_high, gt, low_img) in enumerate(pbar):
            R_low = R_low.to(device)
            I_low = I_low.to(device)
            R_high = R_high.to(device)
            I_high = I_high.to(device)
            gt = gt.to(device)
            low_img = low_img.to(device)

            # ============================================
            # Generator forward
            # ============================================
            I_pred = illum_net(I_low)
            R_pred = reflec_net(R_low)
            recombined = R_pred * I_pred
            enhanced = refine(recombined)

            # ---- Illumination loss ---- #
            l_illum = illum_loss_fn(I_pred, I_high) + 0.5 * illum_ssim_fn(I_pred, I_high)

            # ---- Reflectance loss ---- #
            l_reflec, reflec_dict = reflec_loss_fn(R_pred, R_high, R_low)

            # ---- Reconstruction consistency ---- #
            l_recon = recon_loss_fn(recombined, gt)

            # ---- Final output loss ---- #
            if is_warmup:
                # During warm-up: no adversarial, just supervised
                l_final, final_dict = combined_loss_fn(enhanced, gt, disc_fake=None)
                g_loss = l_illum + l_reflec + l_recon + l_final
                d_loss_val = 0.0
            else:
                # Full training: with adversarial
                # Generator step
                d_full_fake, d_half_fake = disc(low_img, enhanced)
                l_final, final_dict = combined_loss_fn(enhanced, gt,
                    disc_fake=d_full_fake)
                # Additional half-scale adversarial
                l_adv_half = F.mse_loss(d_half_fake, torch.ones_like(d_half_fake))
                l_final_total = l_final + args.lambda_adv * l_adv_half

                g_loss = l_illum + l_reflec + l_recon + l_final_total

            # Gradient accumulation
            g_loss_scaled = g_loss / args.grad_accum
            g_loss_scaled.backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(gen_params, max_norm=1.0)
                opt_G.step()
                opt_G.zero_grad()

                # Update EMA
                ema_illum.update(illum_net)
                ema_reflec.update(reflec_net)
                ema_refine.update(refine)

            # ============================================
            # Discriminator step (skip during warm-up)
            # ============================================
            if not is_warmup and (batch_idx + 1) % args.grad_accum == 0:
                d_full_real, d_half_real = disc(low_img, gt)
                d_full_fake, d_half_fake = disc(low_img, enhanced.detach())

                d_loss_real = (F.mse_loss(d_full_real, torch.ones_like(d_full_real)) +
                               F.mse_loss(d_half_real, torch.ones_like(d_half_real)))
                d_loss_fake = (F.mse_loss(d_full_fake, torch.zeros_like(d_full_fake)) +
                               F.mse_loss(d_half_fake, torch.zeros_like(d_half_fake)))
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                # R1 gradient penalty
                if epoch % 4 == 0:
                    gt.requires_grad_(True)
                    d_r1, _ = disc(low_img, gt)
                    r1_grads = torch.autograd.grad(
                        d_r1.sum(), gt, create_graph=True
                    )[0]
                    r1_penalty = r1_grads.pow(2).sum(dim=[1, 2, 3]).mean()
                    d_loss = d_loss + 10.0 * r1_penalty
                    gt.requires_grad_(False)

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                opt_D.step()
                opt_D.zero_grad()
                d_loss_val = d_loss.item()
            else:
                d_loss_val = 0.0

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss_val
            epoch_illum += l_illum.item()
            epoch_reflec += l_reflec.item()
            epoch_recon += l_recon.item()
            n_batches += 1

            pbar.set_postfix({
                'G': f'{g_loss.item():.4f}',
                'D': f'{d_loss_val:.4f}',
                'cur': f'{curriculum_ratio:.0%}',
            })

        # ---- Step schedulers ---- #
        sched_G.step()
        sched_D.step()

        avg_g = epoch_g_loss / max(n_batches, 1)
        avg_d = epoch_d_loss / max(n_batches, 1)
        avg_illum = epoch_illum / max(n_batches, 1)
        avg_reflec = epoch_reflec / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)

        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)
        history['illum_loss'].append(avg_illum)
        history['reflec_loss'].append(avg_reflec)
        history['recon_loss'].append(avg_recon)

        elapsed = (time.time() - start_time) / 60
        print(f"  Ep {epoch+1}: G={avg_g:.4f} D={avg_d:.4f} "
              f"Illum={avg_illum:.4f} Reflec={avg_reflec:.4f} "
              f"Recon={avg_recon:.4f} [{elapsed:.1f}min]")

        # Save best + periodic checkpoints
        if avg_g < best_g_loss:
            best_g_loss = avg_g
            save_checkpoint(illum_net, reflec_net, refine, disc,
                           ema_illum, ema_reflec, ema_refine,
                           ckpt_dir, "best")

        if (epoch + 1) % 50 == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(illum_net, reflec_net, refine, disc,
                           ema_illum, ema_reflec, ema_refine,
                           ckpt_dir, f"ep{epoch+1}")

    # Save final
    save_checkpoint(illum_net, reflec_net, refine, disc,
                   ema_illum, ema_reflec, ema_refine,
                   ckpt_dir, "final")

    # Save history
    with open(os.path.join(ckpt_dir, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training complete! Total time: {total_time:.1f} min")
    print(f"   Checkpoints saved to {ckpt_dir}/")


def save_checkpoint(illum_net, reflec_net, refine, disc,
                    ema_illum, ema_reflec, ema_refine,
                    ckpt_dir, tag):
    """Save both regular and EMA checkpoints."""
    # Regular checkpoint
    torch.save({
        'illum_net': illum_net.state_dict(),
        'reflec_net': reflec_net.state_dict(),
        'refine': refine.state_dict(),
        'disc': disc.state_dict(),
    }, os.path.join(ckpt_dir, f"gen_{tag}.pth"))

    # EMA checkpoint
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Improved Pipeline 2")
    parser.add_argument('--data_root', type=str, default='datasets/LOL_dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--lr_g', type=float, default=2e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lambda_adv', type=float, default=0.01)
    parser.add_argument('--lambda_grad', type=float, default=0.5)
    parser.add_argument('--alpha_noise', type=float, default=2.0)
    parser.add_argument('--grad_accum', type=int, default=4)
    args = parser.parse_args()

    train_pipeline2_improved(args)
