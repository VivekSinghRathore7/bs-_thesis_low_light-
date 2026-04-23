"""
RetinexPix2Pix — train.py

Usage (from project root):
    cd src/pix2pix_retinex
    python train.py

Key hyperparameters (tuned for LOL-v1, 256×256, single GPU):
    epochs=300, batch=4, lr=2e-4 → 1e-6 cosine
    Discriminator delayed 5 epochs, GAN weight ramps up over 50 epochs
    Checkpoint saved every 20 epochs + best PSNR
"""

import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── path setup so we can import from project root ────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.pix2pix_retinex.dataset import RetinexPix2PixDataset
from src.pix2pix_retinex.models  import RetinexUNet, MultiScaleDisc
from src.pix2pix_retinex.losses  import GeneratorLoss, gan_loss_real, gan_loss_fake

# ── config ────────────────────────────────────────────────────────────────────

CFG = dict(
    data_root   = "datasets/LOL_dataset",
    save_dir    = "checkpoints/rp2p",
    log_dir     = "experiments/rp2p",
    img_size    = 256,
    epochs      = 300,
    batch_size  = 4,
    lr          = 2e-4,
    lr_min      = 1e-6,
    workers     = 4,
    save_every  = 20,
    disc_delay  = 5,        # start updating D after this many epochs
    adv_warmup  = 50,       # epochs to ramp λ_adv from 0→1
    base_g      = 64,
    base_d      = 64,
    n_scales    = 3,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def psnr_batch(pred, gt):
    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=[1,2,3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()


def cosine_lr(optimizer, epoch, total_epochs, lr_max, lr_min):
    lr = lr_min + 0.5*(lr_max-lr_min)*(1+math.cos(math.pi*epoch/total_epochs))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def adv_ramp(epoch, delay, warmup):
    if epoch < delay:
        return 0.0
    return min(1.0, (epoch - delay) / max(warmup - delay, 1))


def save_ckpt(path, G, D, opt_G, opt_D, epoch, history, psnr):
    torch.save(dict(
        epoch=epoch, psnr=psnr, history=history,
        G=G.state_dict(), D=D.state_dict(),
        opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(),
    ), path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(ROOT)
    os.makedirs(CFG['save_dir'], exist_ok=True)
    os.makedirs(CFG['log_dir'],  exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── data ──────────────────────────────────────────────────────────────────
    train_ds = RetinexPix2PixDataset(
        CFG['data_root'], split="our485",
        img_size=CFG['img_size'], augment=True, cache=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=CFG['batch_size'],
        shuffle=True, num_workers=CFG['workers'],
        pin_memory=(device.type=='cuda'), drop_last=True
    )
    print(f"Train  : {len(train_ds)} images  →  {len(train_loader)} batches/epoch")

    # ── models ────────────────────────────────────────────────────────────────
    G = RetinexUNet(in_ch=8, out_ch=3, base=CFG['base_g']).to(device)
    D = MultiScaleDisc(in_ch=6, ndf=CFG['base_d'], n_scales=CFG['n_scales']).to(device)

    n_G = sum(p.numel() for p in G.parameters())/1e6
    n_D = sum(p.numel() for p in D.parameters())/1e6
    print(f"G params: {n_G:.2f}M   D params: {n_D:.2f}M")

    # ── optimisers ────────────────────────────────────────────────────────────
    opt_G = torch.optim.Adam(G.parameters(), lr=CFG['lr'], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=CFG['lr'], betas=(0.5, 0.999))

    # ── loss ──────────────────────────────────────────────────────────────────
    crit_G = GeneratorLoss().to(device)

    history = {'g_loss':[], 'd_loss':[], 'psnr':[], 'lr':[]}
    best_psnr = 0.0

    for epoch in range(1, CFG['epochs']+1):
        G.train(); D.train()
        cosine_lr(opt_G, epoch-1, CFG['epochs'], CFG['lr'], CFG['lr_min'])
        cosine_lr(opt_D, epoch-1, CFG['epochs'], CFG['lr'], CFG['lr_min'])
        aw = adv_ramp(epoch, CFG['disc_delay'], CFG['adv_warmup'])

        g_sum = d_sum = psnr_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Ep {epoch:03d}/{CFG['epochs']}", leave=False)

        for inp, gt, R_tv, L_tv in pbar:
            inp, gt = inp.to(device), gt.to(device)
            R_tv   = R_tv.to(device)
            L_tv   = L_tv.to(device)
            low_rgb = inp[:, :3]          # first 3 channels are I_low

            # ── forward G ────────────────────────────────────────────────────
            fake = G(inp)                 # (B,3,H,W) in [0,1]

            # For Retinex consistency we need predicted R and I from fake output
            # Approximate: R_pred = fake / (max_ch(fake)+ε), I_pred = max_ch(fake)
            I_pred = fake.max(dim=1, keepdim=True)[0].clamp(1e-4, 1.0)
            R_pred = (fake / (I_pred + 1e-6)).clamp(0, 1)

            # ── train D ──────────────────────────────────────────────────────
            if epoch >= CFG['disc_delay']:
                cond_real = torch.cat([low_rgb, gt],   dim=1)   # (B,6,H,W)
                cond_fake = torch.cat([low_rgb, fake.detach()], dim=1)
                d_real_outs = D(cond_real)
                d_fake_outs = D(cond_fake)
                d_loss = 0.5 * (gan_loss_real(d_real_outs) + gan_loss_fake(d_fake_outs))
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
                d_sum += d_loss.item()
            else:
                d_loss = torch.tensor(0.0)

            # ── train G ──────────────────────────────────────────────────────
            cond_real = torch.cat([low_rgb, gt],   dim=1)
            cond_fake = torch.cat([low_rgb, fake],  dim=1)
            d_fake_outs  = D(cond_fake)
            real_feats   = D.get_all_features(cond_real)
            fake_feats   = D.get_all_features(cond_fake)

            g_loss, g_dict = crit_G(
                pred=fake, gt=gt,
                disc_fake_outs=d_fake_outs,
                disc_real_feats=real_feats,
                disc_fake_feats=fake_feats,
                R_pred=R_pred, I_pred=I_pred, I_low=L_tv,
                adv_weight=aw,
            )
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            g_sum    += g_loss.item()
            psnr_sum += psnr_batch(fake.detach(), gt)

            pbar.set_postfix(
                G=f"{g_loss.item():.3f}",
                D=f"{d_loss.item():.3f}" if epoch >= CFG['disc_delay'] else "—",
                PSNR=f"{psnr_batch(fake.detach(), gt):.2f}",
            )

        n = len(train_loader)
        ep_psnr = psnr_sum / n
        ep_g    = g_sum / n
        ep_d    = d_sum / n if epoch >= CFG['disc_delay'] else 0.0
        cur_lr  = opt_G.param_groups[0]['lr']

        history['g_loss'].append(ep_g)
        history['d_loss'].append(ep_d)
        history['psnr'].append(ep_psnr)
        history['lr'].append(cur_lr)

        print(f"Ep {epoch:03d} | G={ep_g:.4f} D={ep_d:.4f} | "
              f"PSNR={ep_psnr:.2f} dB | adv_w={aw:.2f} | lr={cur_lr:.2e}")

        # ── save checkpoints ─────────────────────────────────────────────────
        if epoch % CFG['save_every'] == 0:
            save_ckpt(
                os.path.join(CFG['save_dir'], f"epoch_{epoch:03d}.pth"),
                G, D, opt_G, opt_D, epoch, history, ep_psnr
            )

        if ep_psnr > best_psnr:
            best_psnr = ep_psnr
            save_ckpt(
                os.path.join(CFG['save_dir'], "best.pth"),
                G, D, opt_G, opt_D, epoch, history, ep_psnr
            )
            print(f"  ★  New best train-PSNR: {best_psnr:.2f} dB  (epoch {epoch})")

        # rolling save
        save_ckpt(
            os.path.join(CFG['save_dir'], "latest.pth"),
            G, D, opt_G, opt_D, epoch, history, ep_psnr
        )

    # ── save final weights + history ──────────────────────────────────────────
    torch.save(G.state_dict(), os.path.join(CFG['save_dir'], "gen_final.pth"))
    with open(os.path.join(CFG['log_dir'], "train_log.json"), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅  Training complete. Best train-PSNR: {best_psnr:.2f} dB")
    print(f"    Weights → {CFG['save_dir']}/gen_final.pth")


if __name__ == "__main__":
    main()
