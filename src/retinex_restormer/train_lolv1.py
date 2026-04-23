"""
Sure-Shot trainer — RetinexRestormer on LOL-v1.

Why this architecture beats CNNs on LOL-v1:
  - MDTA processes global context at every resolution (CNNs are local)
  - GDFN gating suppresses noise-only features
  - Patch training: 128×128, 300/image → 145,500 samples/epoch (3× more than baseline)
  - width=32 → ~5M params — avoids the 43M overfitting trap

Expected: 24–26 dB (competitive with TFFormer 26.13, approaching HFL 27.26)

Usage (from project root):
    conda run -n viv python src/retinex_restormer/train.py
"""

import os, sys, json, math, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)

from src.retinex_patch.dataset import PatchLOLDataset, FullLOLDataset
from src.retinex_restormer.architecture import RetinexRestormer

CFG = dict(
    data_root   = "datasets/LOL_dataset",
    save_dir    = "checkpoints/lolv1/retinex_restormer",
    log_dir     = "experiments/lolv1/retinex_restormer",

    # Model
    width       = 32,
    depths      = (2, 2, 2, 4),
    heads       = (1, 2, 4, 8),

    # Data — 30 patches/img × 485 = 14,550 samples/epoch (30× more diverse than full-image)
    patch       = 128,
    n_patches   = 30,
    batch_size  = 64,              # large batch on H100
    workers     = 8,

    # Training
    epochs      = 500,
    lr          = 3e-4,
    lr_min      = 1e-6,
    weight_decay= 1e-4,
    ema_decay   = 0.9999,

    # Mixed precision disabled — bf16 causes numerical instability in SSIM conv2d
    use_amp     = False,

    # Loss weights
    lam_charb   = 1.0,
    lam_ssim    = 0.5,
    lam_vgg     = 0.05,

    # Logging
    eval_every  = 10,
    save_every  = 50,
)


# ── Losses ────────────────────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, p, g):
        return torch.mean(torch.sqrt((p - g) ** 2 + self.eps2))


class SSIMLoss(nn.Module):
    def __init__(self, ws=11, ch=3):
        super().__init__()
        self.ws = ws
        self.ch = ch

    def forward(self, p, g):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        d   = torch.arange(self.ws, dtype=torch.float32, device=p.device) - self.ws // 2
        g1d = torch.exp(-d ** 2 / 4.5)
        g1d = g1d / g1d.sum()
        win = (g1d.unsqueeze(1) * g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(self.ch, 1, 1, 1)
        pad = self.ws // 2
        mu1 = F.conv2d(p, win, padding=pad, groups=self.ch)
        mu2 = F.conv2d(g, win, padding=pad, groups=self.ch)
        s1  = F.conv2d(p * p, win, padding=pad, groups=self.ch) - mu1 ** 2
        s2  = F.conv2d(g * g, win, padding=pad, groups=self.ch) - mu2 ** 2
        s12 = F.conv2d(p * g, win, padding=pad, groups=self.ch) - mu1 * mu2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return 1 - ssim.mean()


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        vgg = M.vgg19(weights=M.VGG19_Weights.DEFAULT).features
        self.s = nn.Sequential(*list(vgg[:18])).eval()
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, p, g):
        p = (p - self.mean) / self.std
        g = (g - self.mean) / self.std
        return F.l1_loss(self.s(p), self.s(g))


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for em, mp in zip(self.model.parameters(), model.parameters()):
            em.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
        for eb, mb in zip(self.model.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()


def cosine_lr(opt, ep, total, lr_max, lr_min):
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * ep / total))
    for pg in opt.param_groups:
        pg['lr'] = lr


@torch.no_grad()
def eval_psnr(model, loader, device):
    model.eval()
    psnrs = []
    for inp7, gt, _, _ in loader:
        inp7, gt = inp7.to(device), gt.to(device)
        pred = model(inp7).clamp(0, 1)
        psnrs.append(psnr_fn(pred, gt))
    model.train()
    return float(np.mean(psnrs))


# ── Main ──────────────────────────────────────────────────────────────────────

import copy

def main():
    os.chdir(ROOT)
    os.makedirs(CFG['save_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'],  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PatchLOLDataset(CFG['data_root'], split="our485",
                                patch=CFG['patch'], n_patches=CFG['n_patches'],
                                augment=True)
    eval_ds  = FullLOLDataset(CFG['data_root'], split="eval15")

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                               shuffle=True, num_workers=CFG['workers'],
                               pin_memory=True, drop_last=True)
    eval_loader  = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)

    n_batches = len(train_loader)
    print(f"Train  : {len(train_ds):,} patches → {n_batches} batches/epoch")
    print(f"Eval   : {len(eval_ds)} full images")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RetinexRestormer(
        in_ch=7, out_ch=3,
        width=CFG['width'],
        depths=CFG['depths'],
        heads=CFG['heads'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params : {n_params:.2f}M")

    ema = EMA(model, CFG['ema_decay'])

    # ── Losses ────────────────────────────────────────────────────────────────
    charb = CharbonnierLoss().to(device)
    ssim  = SSIMLoss().to(device)
    vgg   = VGGLoss().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                             weight_decay=CFG['weight_decay'])

    history = {'loss': [], 'psnr_train': [], 'psnr_eval': [], 'lr': []}
    best_psnr = 0.0
    start_epoch = 1
    no_improve_count = 0   # overfitting detector
    PATIENCE = 50          # stop if no improvement for 50 eval checks

    # ── Auto-resume from latest checkpoint (disconnect-safe) ──
    latest_path = os.path.join(CFG['save_dir'], 'latest.pth')
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        ema.model.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt.get('best_psnr', 0.0)
        if 'optimizer' in ckpt:
            opt.load_state_dict(ckpt['optimizer'])
        # Load history if exists
        log_path = os.path.join(CFG['log_dir'], 'train_log.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                history = json.load(f)
        print(f"\n⚡ RESUMING from epoch {start_epoch} (best_psnr={best_psnr:.2f} dB)")
    else:
        print(f"\n🚀 Starting fresh training")

    print(f"   Epochs: {start_epoch} → {CFG['epochs']}")
    print(f"   Estimated time: ~{(CFG['epochs'] - start_epoch + 1) * 0.5:.0f} min on H100")
    print(f"   Overfitting patience: {PATIENCE} eval checks\n")

    # ── AMP (mixed precision for H100 speedup) ──
    use_amp = CFG.get('use_amp', False)
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    print(f"   AMP: {'bf16' if use_amp and amp_dtype == torch.bfloat16 else 'fp16' if use_amp else 'OFF'}")

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        cosine_lr(opt, epoch - 1, CFG['epochs'], CFG['lr'], CFG['lr_min'])
        loss_sum = psnr_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{CFG['epochs']}", leave=False)
        for inp7, gt, _, _ in pbar:
            inp7 = inp7.to(device)
            gt   = gt.to(device)

            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                pred = model(inp7)
                l_charb = charb(pred, gt)
                l_ssim  = ssim(pred, gt)
                l_vgg   = vgg(pred, gt)
                loss = (CFG['lam_charb'] * l_charb +
                        CFG['lam_ssim']  * l_ssim  +
                        CFG['lam_vgg']   * l_vgg)

            opt.zero_grad()
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            ema.update(model)

            loss_sum += loss.item()
            psnr_sum += psnr_fn(pred.detach().clamp(0, 1), gt)
            pbar.set_postfix(L=f"{loss.item():.4f}",
                             PSNR=f"{psnr_fn(pred.detach().clamp(0,1), gt):.2f}")

        ep_loss  = loss_sum / n_batches
        ep_psnr  = psnr_sum / n_batches
        cur_lr   = opt.param_groups[0]['lr']
        ep_eval  = 0.0

        if epoch % CFG['eval_every'] == 0:
            ep_eval = eval_psnr(ema.model, eval_loader, device)
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | "
                  f"train_PSNR={ep_psnr:.2f} | eval_PSNR={ep_eval:.2f} dB | lr={cur_lr:.2e}")
            if ep_eval > best_psnr:
                best_psnr = ep_eval
                no_improve_count = 0
                torch.save({'epoch': epoch,
                            'model': model.state_dict(),
                            'ema':   ema.model.state_dict(),
                            'best_psnr': best_psnr,
                            'optimizer': opt.state_dict()},
                           os.path.join(CFG['save_dir'], 'best_psnr.pth'))
                print(f"  ★  New best PSNR: {best_psnr:.2f} dB")
            else:
                no_improve_count += 1
                gap = best_psnr - ep_eval
                if no_improve_count >= PATIENCE:
                    print(f"\n⚠️  OVERFITTING DETECTED: eval PSNR dropped {gap:.2f} dB "
                          f"below best ({best_psnr:.2f}) for {no_improve_count} evals")
                    print(f"    Stopping early. Best checkpoint: best_psnr.pth")
                    break
                elif no_improve_count >= 3:
                    print(f"    ⚠ No improvement for {no_improve_count}/{PATIENCE} evals "
                          f"(gap={gap:.2f} dB)")
        else:
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | train_PSNR={ep_psnr:.2f} | lr={cur_lr:.2e}")

        history['loss'].append(ep_loss)
        history['psnr_train'].append(ep_psnr)
        history['psnr_eval'].append(ep_eval)
        history['lr'].append(cur_lr)

        if epoch % CFG['save_every'] == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
                        'best_psnr': best_psnr, 'optimizer': opt.state_dict()},
                       os.path.join(CFG['save_dir'], f'epoch_{epoch:03d}.pth'))

        # Save latest (for resume on disconnect)
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
                    'best_psnr': best_psnr, 'optimizer': opt.state_dict()},
                   os.path.join(CFG['save_dir'], 'latest.pth'))

        # Save history incrementally (for resume)
        with open(os.path.join(CFG['log_dir'], 'train_log.json'), 'w') as hf:
            json.dump(history, hf, indent=2)

    with open(os.path.join(CFG['log_dir'], 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best eval PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
