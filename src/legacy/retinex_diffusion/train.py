"""
Stage 2: DDPM training — retinex_diffusion/train.py

Trains the denoising U-Net to refine Stage 1 output.
Condition = [stage1_pred(3) | I_low(3) | R_cnn(3) | L_tv(1)] = 10ch

Prerequisite: run src/retinex_patch/train.py first → best_psnr.pth

Usage (from project root):
    conda run -n viv python src/retinex_diffusion/train.py
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
from src.retinex_diffusion.model import DenoisingUNet, DDPMSchedule

CFG = dict(
    data_root       = "datasets/LOL_dataset",
    stage1_ckpt     = "checkpoints/retinex_patch/best_psnr.pth",
    save_dir        = "checkpoints/retinex_diffusion",
    log_dir         = "experiments/retinex_diffusion",

    patch           = 128,
    n_patches       = 200,
    batch_size      = 32,
    epochs          = 300,
    lr              = 1e-4,
    lr_min          = 1e-7,
    weight_decay    = 1e-4,
    workers         = 8,
    save_every      = 50,
    eval_every      = 20,
    ema_decay       = 0.9999,

    T               = 1000,   # diffusion timesteps
    ddim_steps      = 20,     # inference steps
)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for em,mp in zip(self.model.parameters(), model.parameters()):
            em.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)
        for eb,mb in zip(self.model.buffers(), model.buffers()):
            eb.data.copy_(mb.data)


def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1./(mse+1e-10))).mean().item()


def cosine_lr(opt, ep, total, lr_max, lr_min):
    lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*ep/total))
    for pg in opt.param_groups: pg['lr'] = lr


def load_stage1(ckpt_path, device):
    """Load frozen Stage 1 generator."""
    illum  = IllumNet(base_filters=32).to(device).eval()
    reflec = ReflecNet(base_filters=48).to(device).eval()
    refine = RefineBlock(base_filters=48).to(device).eval()
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=True)
    illum.load_state_dict(ckpt['ema_illum'])
    reflec.load_state_dict(ckpt['ema_reflec'])
    refine.load_state_dict(ckpt['ema_refine'])
    for m in [illum, reflec, refine]:
        for p in m.parameters(): p.requires_grad_(False)
    return illum, reflec, refine


@torch.no_grad()
def get_stage1_output(illum, reflec, refine, inp7):
    L_ch  = inp7[:, 6:7]
    R_tv  = inp7[:, 3:6]
    I_pred = illum(L_ch)
    R_pred = reflec(R_tv)
    recomb = (R_pred * I_pred).clamp(0,1)
    return refine(recomb).clamp(0,1)


@torch.no_grad()
def eval_psnr(denoiser_ema, schedule, illum, reflec, refine, loader, device):
    denoiser_ema.eval()
    psnrs = []
    for inp7, gt, _, _ in loader:
        inp7, gt = inp7.to(device), gt.to(device)
        s1 = get_stage1_output(illum, reflec, refine, inp7)

        # Build condition
        I_low = inp7[:, :3]
        R_cnn = inp7[:, 3:6]
        L_tv  = inp7[:, 6:7]
        cond  = torch.cat([s1, I_low, R_cnn, L_tv], 1)  # (B,10,H,W)

        # DDIM sample
        pred_norm = schedule.ddim_sample(denoiser_ema, cond, CFG['ddim_steps'], eta=0.0)
        pred = (pred_norm + 1) / 2   # [-1,1] → [0,1]
        psnrs.append(psnr_fn(pred.clamp(0,1), gt))
    denoiser_ema.train()
    return float(np.mean(psnrs))


def main():
    os.chdir(ROOT)
    os.makedirs(CFG['save_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'],  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Stage 1 (frozen) ──────────────────────────────────────────────────────
    if not os.path.exists(CFG['stage1_ckpt']):
        print(f"ERROR: Stage 1 checkpoint not found: {CFG['stage1_ckpt']}")
        print("Run src/retinex_patch/train.py first!")
        return
    illum, reflec, refine = load_stage1(CFG['stage1_ckpt'], device)
    print(f"Stage 1 loaded from: {CFG['stage1_ckpt']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = PatchLOLDataset(CFG['data_root'], split="our485",
                                patch=CFG['patch'], n_patches=CFG['n_patches'])
    eval_ds  = FullLOLDataset(CFG['data_root'],  split="eval15")
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                               shuffle=True, num_workers=CFG['workers'],
                               pin_memory=True, drop_last=True)
    eval_loader  = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds):,} patches | Eval: {len(eval_ds)} imgs")

    # ── Diffusion model ───────────────────────────────────────────────────────
    # in_ch = 3(x_t) + 3(s1) + 3(I_low) + 3(R_cnn) + 1(L_tv) = 13
    model = DenoisingUNet(in_ch=13, out_ch=3, width=64, depths=(2,2,2,2)).to(device)
    n_params = sum(p.numel() for p in model.parameters())/1e6
    print(f"Denoiser params: {n_params:.2f}M")

    ema   = EMA(model, CFG['ema_decay'])
    sched = DDPMSchedule(T=CFG['T'], device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'],
                             weight_decay=CFG['weight_decay'])

    history = {'loss':[], 'psnr_eval':[], 'lr':[]}
    best_psnr = 0.0

    for epoch in range(1, CFG['epochs']+1):
        cosine_lr(opt, epoch-1, CFG['epochs'], CFG['lr'], CFG['lr_min'])
        loss_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{CFG['epochs']}", leave=False)
        for inp7, gt, R_tv, L_tv in pbar:
            inp7 = inp7.to(device); gt   = gt.to(device)
            R_tv = R_tv.to(device); L_tv = L_tv.to(device)

            # Stage 1 output (frozen)
            s1   = get_stage1_output(illum, reflec, refine, inp7)
            I_low= inp7[:, :3]
            R_cnn= inp7[:, 3:6]
            L_ch = inp7[:, 6:7]
            cond = torch.cat([s1, I_low, R_cnn, L_ch], 1)  # (B,10,H,W)

            # Convert gt to [-1,1] for DDPM
            gt_norm = gt * 2 - 1

            # Sample random timesteps
            t = torch.randint(0, CFG['T'], (inp7.size(0),), device=device)
            x_t, noise = sched.q_sample(gt_norm, t)

            # Predict noise
            pred_noise = model(x_t, t, cond)
            loss = F.mse_loss(pred_noise, noise)

            # Optional x0 auxiliary loss (helps with PSNR)
            ab = sched.alpha_bar[t].view(-1,1,1,1)
            x0_pred = (x_t - (1-ab).sqrt()*pred_noise) / ab.sqrt()
            loss = loss + 0.1 * F.l1_loss(x0_pred, gt_norm)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            loss_sum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        ep_loss = loss_sum / len(train_loader)
        cur_lr  = opt.param_groups[0]['lr']

        ep_eval_psnr = 0.0
        if epoch % CFG['eval_every'] == 0:
            ep_eval_psnr = eval_psnr(ema.model, sched, illum, reflec, refine,
                                     eval_loader, device)
            print(f"Ep {epoch:03d} | loss={ep_loss:.5f} | "
                  f"eval_PSNR={ep_eval_psnr:.2f} dB | lr={cur_lr:.2e}")
            if ep_eval_psnr > best_psnr:
                best_psnr = ep_eval_psnr
                torch.save({'epoch':epoch,'model':model.state_dict(),
                             'ema':ema.model.state_dict()},
                           os.path.join(CFG['save_dir'], 'best.pth'))
                print(f"  ★  New best PSNR: {best_psnr:.2f} dB")
        else:
            print(f"Ep {epoch:03d} | loss={ep_loss:.5f} | lr={cur_lr:.2e}")

        history['loss'].append(ep_loss)
        history['psnr_eval'].append(ep_eval_psnr)
        history['lr'].append(cur_lr)

        if epoch % CFG['save_every'] == 0:
            torch.save({'epoch':epoch,'model':model.state_dict(),'ema':ema.model.state_dict()},
                       os.path.join(CFG['save_dir'], f'epoch_{epoch:03d}.pth'))
        torch.save({'epoch':epoch,'model':model.state_dict(),'ema':ema.model.state_dict()},
                   os.path.join(CFG['save_dir'], 'latest.pth'))

    with open(os.path.join(CFG['log_dir'], 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅  Stage 2 complete. Best eval PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
