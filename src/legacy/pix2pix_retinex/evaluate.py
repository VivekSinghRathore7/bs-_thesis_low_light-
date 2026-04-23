"""
RetinexPix2Pix — evaluate.py

Evaluates on LOL eval15 with optional Test-Time Augmentation (TTA ×8).
TTA averages predictions over all 8 flip/rotation combinations.

Usage (from project root):
    cd src/pix2pix_retinex
    python evaluate.py                           # TTA on, uses best.pth
    python evaluate.py --tta 0                   # no TTA
    python evaluate.py --ckpt checkpoints/rp2p/epoch_200.pth
"""

import os, sys, argparse, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.pix2pix_retinex.dataset import RetinexPix2PixDataset
from src.pix2pix_retinex.models  import RetinexUNet


# ─────────────────────────────────────────────────
# Metric functions
# ─────────────────────────────────────────────────

def calc_psnr(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1.0/(mse+1e-10))).item()


def calc_ssim(p, g, ws=11):
    C1, C2, ch = 0.01**2, 0.03**2, 3
    sigma = 1.5
    coords = torch.arange(ws, dtype=torch.float32, device=p.device) - ws//2
    g1d = torch.exp(-coords**2/(2*sigma**2)); g1d /= g1d.sum()
    win = (g1d.unsqueeze(1)*g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(ch,1,1,1)
    pad = ws//2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1  = F.conv2d(p*p, win, padding=pad, groups=ch) - mu1**2
    s2  = F.conv2d(g*g, win, padding=pad, groups=ch) - mu2**2
    s12 = F.conv2d(p*g, win, padding=pad, groups=ch) - mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()


class LPIPSMetric(nn.Module):
    """Custom lightweight LPIPS (VGG16 feature distance) — consistent with existing code."""
    def __init__(self, device):
        super().__init__()
        import torchvision.models as M
        vgg = M.vgg16(weights=M.VGG16_Weights.DEFAULT).features.eval().to(device)
        self.slices = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]]).eval()
        for p in self.parameters(): p.requires_grad = False
        self.register_buffer('mean', torch.tensor([.485,.456,.406],device=device).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([.229,.224,.225],device=device).view(1,3,1,1))

    def forward(self, p, g):
        p = (p-self.mean)/self.std
        g = (g-self.mean)/self.std
        diffs = []
        for s in self.slices:
            p = s(p); g = s(g)
            pn = p/(p.norm(dim=1,keepdim=True)+1e-10)
            gn = g/(g.norm(dim=1,keepdim=True)+1e-10)
            diffs.append((pn-gn).pow(2).mean(dim=[1,2,3]))
        return torch.stack(diffs).mean(dim=0).item()


# ─────────────────────────────────────────────────
# TTA: 8-fold (4 rotations × 2 flips)
# ─────────────────────────────────────────────────

def tta_predict(G, inp, device):
    """Average prediction over 8 augmentations."""
    preds = []
    for k in range(4):
        for flip in [False, True]:
            x = inp.clone()
            x = torch.rot90(x, k, dims=[2, 3])
            if flip:
                x = torch.flip(x, dims=[3])
            with torch.no_grad():
                pred = G(x.to(device))
            if flip:
                pred = torch.flip(pred, dims=[3])
            pred = torch.rot90(pred, -k, dims=[2, 3])
            preds.append(pred.cpu())
    return torch.stack(preds).mean(0)


# ─────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────

def evaluate(args):
    os.chdir(ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── load model ────────────────────────────────────────────────────────────
    G = RetinexUNet(in_ch=8, out_ch=3, base=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ckpt.get('G', ckpt)   # handle both full ckpt and bare state_dict
    G.load_state_dict(state)
    G.eval()
    print(f"Loaded: {args.ckpt}")

    # ── dataset ───────────────────────────────────────────────────────────────
    ds = RetinexPix2PixDataset(
        args.data_root, split="eval15",
        img_size=args.img_size, augment=False, cache=True
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPSMetric(device)
    psnrs, ssims, lpipss = [], [], []
    os.makedirs(args.out_dir, exist_ok=True)

    for i, (inp, gt, R_tv, L_tv) in enumerate(tqdm(loader, desc="Eval")):
        gt = gt.to(device)

        if args.tta:
            pred = tta_predict(G, inp, device).to(device)
        else:
            with torch.no_grad():
                pred = G(inp.to(device))

        pred = pred.clamp(0, 1)
        psnrs.append(calc_psnr(pred, gt))
        ssims.append(calc_ssim(pred, gt))
        lpipss.append(lpips_fn(pred, gt))

        img = pred[0].cpu().numpy().transpose(1,2,0)*255
        cv2.imwrite(os.path.join(args.out_dir, f"{i:04d}.png"),
                    img.astype(np.uint8))

    # ── report ────────────────────────────────────────────────────────────────
    p_mean = np.mean(psnrs)
    s_mean = np.mean(ssims)
    l_mean = np.mean(lpipss)

    print("\n" + "="*55)
    print(f"{'Metric':<20} {'Ours':>10}  {'P3 (old best)':>13}")
    print("="*55)
    print(f"{'PSNR (dB) ↑':<20} {p_mean:>10.4f}  {'21.36':>13}")
    print(f"{'SSIM ↑':<20} {s_mean:>10.4f}  {'0.8661':>13}")
    print(f"{'LPIPS ↓':<20} {l_mean:>10.5f}  {'0.00234':>13}")
    print("="*55)
    tta_str = "  (TTA ×8)" if args.tta else ""
    print(f"TTA: {'ON' if args.tta else 'OFF'}{tta_str}")

    results = {"psnr": psnrs, "ssim": ssims, "lpips": lpipss,
               "mean_psnr": p_mean, "mean_ssim": s_mean, "mean_lpips": l_mean,
               "tta": args.tta}
    out_json = os.path.join(args.out_dir, "metrics.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved images + metrics → {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="checkpoints/rp2p/best.pth")
    parser.add_argument("--data_root", default="datasets/LOL_dataset")
    parser.add_argument("--out_dir",   default="results/rp2p_eval")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--tta",       type=int, default=1,
                        help="1=TTA×8 (default), 0=no TTA")
    args = parser.parse_args()
    evaluate(args)
