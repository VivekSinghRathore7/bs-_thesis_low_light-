"""
train_rain.py
=============
Training script for RAIN.

Usage:
    conda run -n thesis python3 src/training/train_rain.py

Config is set via the CONFIG dict at the top of this file.
Checkpoints + logs are saved to experiments/rain/.
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rain_model   import RAIN
from datasets.lol_dataset import LOLDataset
from training.losses     import RAINLoss

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = dict(
    # Data
    train_low  = os.path.join(BASE_DIR, "datasets/LOL_dataset/our485/low"),
    train_high = os.path.join(BASE_DIR, "datasets/LOL_dataset/our485/high"),
    val_low    = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low"),
    val_high   = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/high"),
    best_dir   = os.path.join(BASE_DIR, "results/reflectance_finetuned"),

    # Model
    base_c     = 64,
    mhsa_heads = 8,

    # Training
    epochs     = 200,
    batch_size = 8,
    patch_size = 256,
    lr         = 1e-4,
    lr_min     = 1e-6,
    weight_decay = 1e-4,

    # Loss weights
    lambda_l1   = 1.0,
    lambda_ssim = 1.0,
    lambda_perc = 0.04,
    lambda_illum= 0.5,

    # Logging
    print_every  = 20,     # iterations
    val_every    = 1,      # epochs
    save_every   = 10,     # epochs
    exp_dir      = os.path.join(BASE_DIR, "experiments/rain"),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def illum_from_rgb(img_low_np):
    """Compute I_low (1-ch) from BGR float32 numpy image for val."""
    import cv2
    I = np.max(img_low_np, axis=2)
    I_u8 = (I * 255).astype(np.uint8)
    I_s  = cv2.bilateralFilter(I_u8, d=15, sigmaColor=75, sigmaSpace=75)
    return (I_s.astype(np.float32) / 255.0)


def tensor_to_np01(t):
    """(C,H,W) tanh tensor [−1,1] → (H,W,C) numpy [0,1]."""
    x = (t.clamp(-1, 1) + 1.0) * 0.5
    return x.permute(1, 2, 0).cpu().numpy()


def evaluate(model, val_loader, device):
    model.eval()
    psnrs, ssims = [], []
    with torch.no_grad():
        for batch in val_loader:
            img_low  = batch["img_low"].to(device)
            r_low    = batch["r_low"].to(device)
            i_low    = batch["i_low"].to(device)
            img_high = batch["img_high"]          # keep on CPU for metrics

            enh, _ = model(img_low, r_low, i_low)

            for b in range(enh.shape[0]):
                pred = tensor_to_np01(enh[b])
                gt   = img_high[b].permute(1, 2, 0).numpy()
                pred_u8 = (pred * 255).astype(np.uint8)
                gt_u8   = (gt   * 255).astype(np.uint8)
                psnrs.append(calc_psnr(gt_u8, pred_u8, data_range=255))
                ssims.append(calc_ssim(gt_u8, pred_u8, channel_axis=2, data_range=255))

    model.train()
    return np.mean(psnrs), np.mean(ssims)


def save_checkpoint(model, optimizer, scheduler, epoch, best_psnr, path):
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_psnr": best_psnr,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_psnr"]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = CONFIG
    os.makedirs(cfg["exp_dir"], exist_ok=True)
    with open(os.path.join(cfg["exp_dir"], "config.json"), "w") as f:
        json.dump({k: v for k, v in cfg.items() if isinstance(v, (int, float, str))}, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = LOLDataset(
        low_dir    = cfg["train_low"],
        high_dir   = cfg["train_high"],
        best_dir   = cfg["best_dir"] if os.path.exists(cfg["best_dir"]) else None,
        patch_size = cfg["patch_size"],
        augment    = True,
    )
    val_ds = LOLDataset(
        low_dir    = cfg["val_low"],
        high_dir   = cfg["val_high"],
        best_dir   = cfg["best_dir"] if os.path.exists(cfg["best_dir"]) else None,
        patch_size = None,    # full-resolution validation
        augment    = False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train  : {len(train_ds)} images  |  Val : {len(val_ds)} images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RAIN(base_c=cfg["base_c"], mhsa_heads=cfg["mhsa_heads"]).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params : {total_params:.2f} M")

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    criterion = RAINLoss(
        lambda_l1   = cfg["lambda_l1"],
        lambda_ssim = cfg["lambda_ssim"],
        lambda_perc = cfg["lambda_perc"],
        lambda_illum= cfg["lambda_illum"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr_min"]
    )

    # ── Resume if checkpoint exists ───────────────────────────────────────────
    start_epoch = 0
    best_psnr   = 0.0
    ckpt_latest = os.path.join(cfg["exp_dir"], "latest.pth")
    if os.path.exists(ckpt_latest):
        start_epoch, best_psnr = load_checkpoint(
            ckpt_latest, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}  (best PSNR {best_psnr:.3f})")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = os.path.join(cfg["exp_dir"], "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,iter,loss_total,loss_l1,loss_ssim,loss_perc,loss_illum,"
                    "val_psnr,val_ssim\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    scaler = None   # fp32 training — avoids NaN in attention with fp16

    print(f"\n{'='*65}")
    print(f"  Starting RAIN training  ({cfg['epochs']} epochs)")
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            img_low  = batch["img_low"].to(device)
            r_low    = batch["r_low"].to(device)
            i_low    = batch["i_low"].to(device)
            img_high = batch["img_high"].to(device)
            i_high   = batch["i_low"].to(device)   # use low illumination as aux target

            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    enhanced, i_pred = model(img_low, r_low, i_low)
                    loss, loss_dict  = criterion(enhanced, i_pred, img_high, i_high)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                enhanced, i_pred = model(img_low, r_low, i_low)
                loss, loss_dict  = criterion(enhanced, i_pred, img_high, i_high)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss += loss_dict["total"]

            if (it + 1) % cfg["print_every"] == 0:
                print(f"  Ep {epoch+1:03d} [{it+1:4d}/{len(train_loader)}]  "
                      f"loss={loss_dict['total']:.4f}  "
                      f"l1={loss_dict['l1']:.4f}  "
                      f"ssim={loss_dict['ssim']:.4f}  "
                      f"perc={loss_dict['perc']:.4f}  "
                      f"illum={loss_dict['illum']:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0

        # Validate
        val_psnr = val_ssim = 0.0
        if (epoch + 1) % cfg["val_every"] == 0:
            val_psnr, val_ssim = evaluate(model, val_loader, device)
            print(f"\n  [Val] Ep {epoch+1:03d}  PSNR={val_psnr:.3f} dB  "
                  f"SSIM={val_ssim:.4f}  avg_loss={avg_loss:.4f}  "
                  f"time={elapsed:.1f}s\n")

        # Log
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{(epoch+1)*len(train_loader)},"
                    f"{avg_loss:.6f},{loss_dict['l1']:.6f},"
                    f"{loss_dict['ssim']:.6f},{loss_dict['perc']:.6f},"
                    f"{loss_dict['illum']:.6f},{val_psnr:.4f},{val_ssim:.4f}\n")

        # Save checkpoints
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_psnr, ckpt_latest)

        if (epoch + 1) % cfg["save_every"] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_psnr,
                            os.path.join(cfg["exp_dir"], f"epoch_{epoch+1:03d}.pth"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_psnr,
                            os.path.join(cfg["exp_dir"], "best.pth"))
            print(f"  ★ New best PSNR: {best_psnr:.3f} dB  (saved best.pth)")

    print(f"\n{'='*65}")
    print(f"  Training complete.  Best Val PSNR = {best_psnr:.3f} dB")
    print(f"  Checkpoints in: {cfg['exp_dir']}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
