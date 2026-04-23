"""
train.py
Train all 3 pipelines on the LOL dataset.

Usage:
    python train.py --pipeline 1 --epochs 100 --data_root datasets/LOL_dataset
    python train.py --pipeline 2 --epochs 100 --data_root datasets/LOL_dataset
    python train.py --pipeline 3 --epochs 100 --data_root datasets/LOL_dataset
    python train.py --pipeline all --epochs 100 --data_root datasets/LOL_dataset
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Pipeline1Dataset, Pipeline2Dataset, Pipeline3Dataset
from models import (
    UNetGenerator, IllumNet, ReflecNet, RefineBlock, PatchDiscriminator
)
from losses import CombinedLoss, ReflectanceLoss, ReconstructionConsistencyLoss


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================
# Pipeline 1: Dual-Input Conditional GAN
# ============================================================

def train_pipeline1(args):
    print("\n" + "="*60)
    print("TRAINING PIPELINE 1: Dual-Input Conditional GAN")
    print("="*60)

    device = get_device()
    save_dir = os.path.join(args.save_dir, "pipeline1")
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds = Pipeline1Dataset(args.data_root, split="our485", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Models
    gen = UNetGenerator(in_ch=4, out_ch=3, base_filters=64).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss
    criterion = CombinedLoss().to(device)

    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(1, args.epochs + 1):
        gen.train(); disc.train()
        g_loss_sum, d_loss_sum = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"P1 Epoch {epoch}/{args.epochs}")
        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)

            # --- Train Discriminator ---
            fake = gen(inp).detach()
            d_real = disc(gt)
            d_fake = disc(fake)
            d_loss = 0.5 * (
                F.mse_loss(d_real, torch.ones_like(d_real)) +
                F.mse_loss(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # --- Train Generator ---
            fake = gen(inp)
            d_fake = disc(fake)
            g_loss = criterion(fake, gt, d_fake)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            pbar.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        n = len(train_loader)
        history['g_loss'].append(g_loss_sum / n)
        history['d_loss'].append(d_loss_sum / n)
        print(f"  Epoch {epoch}: G_loss={g_loss_sum/n:.4f}, D_loss={d_loss_sum/n:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(gen.state_dict(), os.path.join(save_dir, f"gen_epoch{epoch}.pth"))
            torch.save(disc.state_dict(), os.path.join(save_dir, f"disc_epoch{epoch}.pth"))

    # Save final + history
    torch.save(gen.state_dict(), os.path.join(save_dir, "gen_final.pth"))
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f)
    print(f"Pipeline 1 training complete. Saved to {save_dir}")


# ============================================================
# Pipeline 2: Reflectance-Illumination Disentangled GAN
# ============================================================

def r1_gradient_penalty(disc, real, lambda_r1=10.0):
    """R1 gradient penalty for discriminator regularization."""
    real = real.requires_grad_(True)
    d_real = disc(real)
    grads = torch.autograd.grad(
        outputs=d_real.sum(), inputs=real, create_graph=True
    )[0]
    penalty = grads.view(grads.size(0), -1).norm(2, dim=1).pow(2).mean()
    return lambda_r1 * penalty


def train_pipeline2(args):
    print("\n" + "="*60)
    print("TRAINING PIPELINE 2: Disentangled GAN (U-Net + CBAM)")
    print("="*60)

    device = get_device()
    save_dir = os.path.join(args.save_dir, "pipeline2")
    os.makedirs(save_dir, exist_ok=True)

    WARMUP_EPOCHS = 20  # Train without GAN loss for stability
    LR_WARMUP_EPOCHS = 10  # Linear LR warm-up

    # Data
    train_ds = Pipeline2Dataset(args.data_root, split="our485", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Models (upgraded U-Net architectures)
    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)

    # Print model sizes
    n_illum = sum(p.numel() for p in illum_net.parameters())
    n_reflec = sum(p.numel() for p in reflec_net.parameters())
    n_refine = sum(p.numel() for p in refine.parameters())
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"  IllumNet:  {n_illum:>10,} params")
    print(f"  ReflecNet: {n_reflec:>10,} params")
    print(f"  RefineBlock: {n_refine:>8,} params")
    print(f"  Disc:      {n_disc:>10,} params")
    print(f"  Total Gen: {n_illum+n_reflec+n_refine:>10,} params")

    # Optimizers
    gen_params = list(illum_net.parameters()) + list(reflec_net.parameters()) + list(refine.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))

    # Learning rate scheduler
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-7)

    # Losses
    criterion = CombinedLoss().to(device)
    criterion_reflec = ReflectanceLoss(
        alpha=args.alpha_noise, 
        lambda_grad=args.lambda_grad,
        w_b=args.w_b, w_g=args.w_g, w_r=args.w_r,
        lambda_freq=args.lambda_freq,
        lambda_percp=args.lambda_percp
    ).to(device)
    criterion_recon = ReconstructionConsistencyLoss(lambda_l1=1.0, lambda_ssim=0.5).to(device)
    l1_loss = nn.L1Loss()

    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(1, args.epochs + 1):
        # LR warm-up: linear ramp for first LR_WARMUP_EPOCHS
        if epoch <= LR_WARMUP_EPOCHS:
            warmup_factor = epoch / LR_WARMUP_EPOCHS
            for pg in opt_g.param_groups:
                pg['lr'] = args.lr * warmup_factor
            for pg in opt_d.param_groups:
                pg['lr'] = args.lr * 0.5 * warmup_factor

        is_warmup = epoch <= WARMUP_EPOCHS

        # Curriculum strategy
        curriculum_cap_epoch = max(1, int(args.epochs * 0.6))
        ratio = min(1.0, 0.2 + 0.8 * ((epoch - 1) / curriculum_cap_epoch))
        train_ds.set_curriculum_ratio(ratio)

        illum_net.train(); reflec_net.train(); refine.train(); disc.train()
        g_loss_sum, d_loss_sum = 0.0, 0.0

        phase = "WARM" if is_warmup else "GAN"
        pbar = tqdm(train_loader, desc=f"P2 [{phase}] Epoch {epoch}/{args.epochs}")
        for R_low, I_low, R_high, I_high, gt in pbar:
            R_low, I_low = R_low.to(device), I_low.to(device)
            R_high, I_high = R_high.to(device), I_high.to(device)
            gt = gt.to(device)

            # --- Forward pass ---
            I_pred = illum_net(I_low)         # 1ch
            R_pred = reflec_net(R_low)        # 3ch
            recombined = R_pred * I_pred      # 3ch (Retinex recombination)
            output = refine(recombined)       # 3ch

            # --- Train Discriminator (skip during warm-up) ---
            d_loss = torch.tensor(0.0, device=device)
            if not is_warmup:
                d_real = disc(gt)
                d_fake = disc(output.detach())
                d_loss = 0.5 * (
                    F.mse_loss(d_real, torch.ones_like(d_real)) +
                    F.mse_loss(d_fake, torch.zeros_like(d_fake))
                )
                # R1 gradient penalty every 16 steps
                if torch.rand(1).item() < 0.0625:
                    d_loss = d_loss + r1_gradient_penalty(disc, gt.detach(), lambda_r1=10.0)
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # --- Train Generators ---
            # Component-level losses
            loss_I = l1_loss(I_pred, I_high)
            loss_R_total, loss_R_dict = criterion_reflec(R_pred, R_high, R_low)

            # Reconstruction consistency: R * I should approximate GT
            loss_recon = criterion_recon(recombined, gt)

            if is_warmup:
                # Warm-up: only supervised losses, no GAN
                loss_final = l1_loss(output, gt)
                g_loss = loss_final + 0.5 * loss_I + loss_R_total + 0.5 * loss_recon
            else:
                # Full training with adversarial
                d_fake = disc(output)
                loss_final = criterion(output, gt, d_fake)
                g_loss = loss_final + 0.5 * loss_I + loss_R_total + 0.5 * loss_recon

            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            pbar.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        # Step LR schedulers (after warm-up period)
        if epoch > LR_WARMUP_EPOCHS:
            sched_g.step()
            sched_d.step()

        n = len(train_loader)
        history['g_loss'].append(g_loss_sum / n)
        history['d_loss'].append(d_loss_sum / n)
        lr_g = opt_g.param_groups[0]['lr']
        print(f"  Epoch {epoch} [{phase}]: G={g_loss_sum/n:.4f}, D={d_loss_sum/n:.4f}, lr={lr_g:.2e}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            state = {
                'illum_net': illum_net.state_dict(),
                'reflec_net': reflec_net.state_dict(),
                'refine': refine.state_dict(),
                'disc': disc.state_dict(),
            }
            torch.save(state, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth"))

    # Save final
    torch.save({
        'illum_net': illum_net.state_dict(),
        'reflec_net': reflec_net.state_dict(),
        'refine': refine.state_dict(),
    }, os.path.join(save_dir, "gen_final.pth"))
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f)
    print(f"Pipeline 2 training complete. Saved to {save_dir}")


# ============================================================
# Pipeline 3: Multi-Illumination Ensemble GAN
# ============================================================

def train_pipeline3(args):
    print("\n" + "="*60)
    print("TRAINING PIPELINE 3: Multi-Illumination Ensemble GAN")
    print("="*60)

    device = get_device()
    save_dir = os.path.join(args.save_dir, "pipeline3")
    os.makedirs(save_dir, exist_ok=True)

    # Data
    train_ds = Pipeline3Dataset(args.data_root, split="our485", img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    # Models
    gen = UNetGenerator(in_ch=15, out_ch=3, base_filters=64).to(device)
    disc = PatchDiscriminator(in_ch=3).to(device)

    # Optimizers
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Loss
    criterion = CombinedLoss().to(device)

    history = {'g_loss': [], 'd_loss': []}

    for epoch in range(1, args.epochs + 1):
        gen.train(); disc.train()
        g_loss_sum, d_loss_sum = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"P3 Epoch {epoch}/{args.epochs}")
        for inp, gt in pbar:
            inp, gt = inp.to(device), gt.to(device)

            # --- Train Discriminator ---
            fake = gen(inp).detach()
            d_real = disc(gt)
            d_fake = disc(fake)
            d_loss = 0.5 * (
                F.mse_loss(d_real, torch.ones_like(d_real)) +
                F.mse_loss(d_fake, torch.zeros_like(d_fake))
            )
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            # --- Train Generator ---
            fake = gen(inp)
            d_fake = disc(fake)
            g_loss = criterion(fake, gt, d_fake)
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            pbar.set_postfix(G=f"{g_loss.item():.4f}", D=f"{d_loss.item():.4f}")

        n = len(train_loader)
        history['g_loss'].append(g_loss_sum / n)
        history['d_loss'].append(d_loss_sum / n)
        print(f"  Epoch {epoch}: G_loss={g_loss_sum/n:.4f}, D_loss={d_loss_sum/n:.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(gen.state_dict(), os.path.join(save_dir, f"gen_epoch{epoch}.pth"))
            torch.save(disc.state_dict(), os.path.join(save_dir, f"disc_epoch{epoch}.pth"))

    torch.save(gen.state_dict(), os.path.join(save_dir, "gen_final.pth"))
    with open(os.path.join(save_dir, "history.json"), 'w') as f:
        json.dump(history, f)
    print(f"Pipeline 3 training complete. Saved to {save_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLIE Pipelines")
    parser.add_argument('--pipeline', type=str, default='all', choices=['1', '2', '3', 'all'])
    parser.add_argument('--data_root', type=str, default='datasets/LOL_dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=20)
    
    # Reflectance Fine-Tuning Hyperparameters
    parser.add_argument('--alpha_noise', type=float, default=2.0)
    parser.add_argument('--lambda_grad', type=float, default=0.5)
    parser.add_argument('--w_b', type=float, default=1.2)
    parser.add_argument('--w_g', type=float, default=1.0)
    parser.add_argument('--w_r', type=float, default=1.5)
    parser.add_argument('--lambda_freq', type=float, default=0.1)
    parser.add_argument('--lambda_percp', type=float, default=0.1)
    
    args = parser.parse_args()

    print(f"Device: {get_device()}")
    print(f"Dataset: {args.data_root}")

    if args.pipeline in ['1', 'all']:
        train_pipeline1(args)
    if args.pipeline in ['2', 'all']:
        train_pipeline2(args)
    if args.pipeline in ['3', 'all']:
        train_pipeline3(args)

    print("\n✅ All requested training complete!")
