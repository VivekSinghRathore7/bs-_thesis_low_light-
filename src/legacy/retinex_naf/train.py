"""
RetinexNAF-MS — train.py

Usage (from project root):
    python src/retinex_naf/train.py

Key features:
  - 400 epochs, cosine annealing lr 3e-4 → 5e-7
  - EMA (exponential moving average) model updated every step
  - AdamW + weight decay 1e-4
  - Multi-scale auxiliary supervision (decoder levels 1,2,3)
  - Periodic eval PSNR on eval15 (every 10 epochs)
  - Best EMA checkpoint saved separately
"""

import os, sys, json, math, copy
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.retinex_naf.dataset import RetinexNAFDataset
from src.retinex_naf.models  import RetinexNAF
from src.retinex_naf.losses  import TotalLoss

# ── Config ────────────────────────────────────────────────────────────────────

CFG = dict(
    data_root      = "datasets/LOL_dataset",
    save_dir       = "checkpoints/retinex_naf",
    log_dir        = "experiments/retinex_naf",
    refine_ckpt    = "results/reflectance_cnn/checkpoints/best_model.pth",

    # H100 NVL 100GB — use full LOL resolution + large model
    img_size       = 400,        # native LOL resolution (was 256)
    epochs         = 400,
    batch_size     = 16,         # 4× larger (H100 has 100GB)
    lr             = 2e-4,
    lr_min         = 1e-7,
    weight_decay   = 1e-4,
    workers        = 8,
    save_every     = 20,
    eval_every     = 10,

    # NAFNet-Large: width=128, 5 levels (was 64, 4 levels)
    width          = 128,
    depths         = (2, 2, 4, 8, 8),

    # EMA
    ema_decay      = 0.9999,     # tighter EMA for larger batch
)

# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9995):
        self.model  = copy.deepcopy(model).eval()
        self.decay  = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for em, mp in zip(self.model.parameters(), model.parameters()):
            em.data.mul_(self.decay).add_(mp.data, alpha=1-self.decay)
        for em, mp in zip(self.model.buffers(), model.buffers()):
            em.data.copy_(mp.data)

    def state_dict(self):
        return self.model.state_dict()


# ── helpers ───────────────────────────────────────────────────────────────────

def psnr(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1.0/(mse+1e-10))).mean().item()


def cosine_lr(opt, epoch, total, lr_max, lr_min):
    lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*epoch/total))
    for pg in opt.param_groups: pg['lr'] = lr


@torch.no_grad()
def eval_psnr(model, loader, device):
    model.eval()
    psnrs = []
    for inp9, gt, R_tv, L_tv in loader:
        inp9, gt = inp9.to(device), gt.to(device)
        out, _, _, _ = model(inp9)
        psnrs.append(psnr(out.clamp(0,1), gt))
    model.train()
    return float(np.mean(psnrs))


# ── main ──────────────────────────────────────────────────────────────────────

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
    train_ds = RetinexNAFDataset(CFG['data_root'], split="our485",
                                  size=CFG['img_size'], augment=True, cache=True)
    eval_ds  = RetinexNAFDataset(CFG['data_root'], split="eval15",
                                  size=CFG['img_size'], augment=False, cache=True)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                               shuffle=True,  num_workers=CFG['workers'],
                               pin_memory=(device.type=='cuda'), drop_last=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=1,
                               shuffle=False, num_workers=2)

    print(f"Train  : {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── model ─────────────────────────────────────────────────────────────────
    refine_ckpt = CFG['refine_ckpt'] if os.path.exists(CFG['refine_ckpt']) else None
    if refine_ckpt is None:
        print("⚠  LightRefineNet checkpoint not found — proceeding without it")

    model = RetinexNAF(in_ch=9, out_ch=3,
                       width=CFG['width'], depths=CFG['depths'],
                       refine_ckpt=refine_ckpt).to(device)
    ema = EMA(model.to(device), decay=CFG['ema_decay'])

    nG = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
    print(f"Params : {nG:.2f}M (trainable)")

    # ── optimiser ─────────────────────────────────────────────────────────────
    # Separate param groups: frozen refiner gets lr=0
    frozen_ids = {id(p) for p in model.ret_enc.refiner.parameters()}
    g1 = [p for p in model.parameters() if id(p) not in frozen_ids and p.requires_grad]
    opt = torch.optim.AdamW(g1, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    criterion = TotalLoss().to(device)

    history = {'loss':[], 'psnr_train':[], 'psnr_eval':[], 'lr':[]}
    best_eval_psnr = 0.0

    for epoch in range(1, CFG['epochs']+1):
        model.train()
        cosine_lr(opt, epoch-1, CFG['epochs'], CFG['lr'], CFG['lr_min'])

        loss_sum = psnr_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{CFG['epochs']}", leave=False)

        for inp9, gt, R_tv, L_tv in pbar:
            inp9 = inp9.to(device); gt    = gt.to(device)
            R_tv = R_tv.to(device); L_tv  = L_tv.to(device)

            pred, aux_outs, R_cnn, L_enh = model(inp9)

            total_loss, loss_dict = criterion(
                pred, gt, aux_outs, R_cnn, L_enh, L_tv
            )

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(g1, max_norm=1.0)   # gradient clipping
            opt.step()
            ema.update(model)

            loss_sum += total_loss.item()
            psnr_sum += psnr(pred.detach().clamp(0,1), gt)
            pbar.set_postfix(
                loss=f"{total_loss.item():.4f}",
                PSNR=f"{psnr(pred.detach().clamp(0,1),gt):.2f}",
            )

        n = len(train_loader)
        ep_loss  = loss_sum / n
        ep_psnr  = psnr_sum / n
        cur_lr   = opt.param_groups[0]['lr']

        # ── eval ──────────────────────────────────────────────────────────────
        ep_eval_psnr = 0.0
        if epoch % CFG['eval_every'] == 0:
            ep_eval_psnr = eval_psnr(ema.model, eval_loader, device)
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | "
                  f"train_PSNR={ep_psnr:.2f} | eval_PSNR={ep_eval_psnr:.2f} dB | "
                  f"lr={cur_lr:.2e}")
            if ep_eval_psnr > best_eval_psnr:
                best_eval_psnr = ep_eval_psnr
                torch.save({'epoch': epoch, 'psnr': ep_eval_psnr,
                            'model': model.state_dict(),
                            'ema': ema.state_dict()},
                           os.path.join(CFG['save_dir'], 'best_ema.pth'))
                print(f"  ★  New best eval-PSNR: {best_eval_psnr:.2f} dB")
        else:
            print(f"Ep {epoch:03d} | loss={ep_loss:.4f} | "
                  f"train_PSNR={ep_psnr:.2f} dB | lr={cur_lr:.2e}")

        history['loss'].append(ep_loss)
        history['psnr_train'].append(ep_psnr)
        history['psnr_eval'].append(ep_eval_psnr)
        history['lr'].append(cur_lr)

        if epoch % CFG['save_every'] == 0:
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'ema': ema.state_dict(), 'opt': opt.state_dict()},
                       os.path.join(CFG['save_dir'], f'epoch_{epoch:03d}.pth'))

        # rolling latest
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'ema': ema.state_dict(), 'opt': opt.state_dict()},
                   os.path.join(CFG['save_dir'], 'latest.pth'))

    # ── save final ────────────────────────────────────────────────────────────
    torch.save(ema.state_dict(), os.path.join(CFG['save_dir'], 'gen_final_ema.pth'))
    with open(os.path.join(CFG['log_dir'], 'train_log.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✅  Training complete.")
    print(f"    Best eval PSNR : {best_eval_psnr:.2f} dB")
    print(f"    Weights        : {CFG['save_dir']}/gen_final_ema.pth")


if __name__ == "__main__":
    main()
