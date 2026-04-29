"""
eval_p2_improved.py
Evaluate the improved Pipeline 2 (uses EMA weights) and compare with old P2.

Usage:
    conda run -n viv python eval_p2_improved.py \
        --data_root datasets/LOL_dataset \
        --checkpoint_dir checkpoints
"""

import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_all_pipelines import Pipeline2Dataset
from models_p2_disentangled import IllumNetV2, ReflecNetV2, MultiScaleRefineBlock
import torchvision.models as models


# ============================================================
# Metrics
# ============================================================

def calc_psnr(pred, gt):
    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


def calc_ssim(pred, gt, window_size=11):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    channels = pred.shape[1]
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(gt, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=pad, groups=channels) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3])


class LPIPS:
    """Lightweight LPIPS using VGG16 features."""
    def __init__(self, device):
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
        self.slices = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23],
        ]).eval().to(device)
        for p in self.slices.parameters():
            p.requires_grad = False
        self.device = device

    def __call__(self, pred, gt):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        pred_n = (pred - mean) / std
        gt_n = (gt - mean) / std
        diffs = []
        x_p, x_g = pred_n, gt_n
        for s in self.slices:
            x_p = s(x_p)
            x_g = s(x_g)
            x_p_norm = x_p / (x_p.norm(dim=1, keepdim=True) + 1e-10)
            x_g_norm = x_g / (x_g.norm(dim=1, keepdim=True) + 1e-10)
            diffs.append((x_p_norm - x_g_norm).pow(2).mean(dim=[1, 2, 3]))
        return torch.stack(diffs).mean(dim=0)


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(args, device):
    print(f"\n{'='*60}")
    print(f"  Evaluating Improved Pipeline 2 (EMA weights)")
    print(f"  Image size: {args.img_size}")
    print(f"{'='*60}\n")

    # Load models
    illum_net = IllumNetV2(base_filters=32).to(device)
    reflec_net = ReflecNetV2(base_filters=64).to(device)
    refine = MultiScaleRefineBlock(base_filters=64).to(device)

    ckpt_dir = os.path.join(args.checkpoint_dir, "pipeline2_improved")

    # Try EMA weights first, fall back to regular
    ema_path = os.path.join(ckpt_dir, "gen_final_ema.pth")
    regular_path = os.path.join(ckpt_dir, "gen_final.pth")
    best_ema_path = os.path.join(ckpt_dir, "gen_best_ema.pth")

    if os.path.exists(ema_path):
        print(f"Loading EMA checkpoint: {ema_path}")
        ckpt = torch.load(ema_path, map_location=device, weights_only=True)
    elif os.path.exists(best_ema_path):
        print(f"Loading best EMA checkpoint: {best_ema_path}")
        ckpt = torch.load(best_ema_path, map_location=device, weights_only=True)
    elif os.path.exists(regular_path):
        print(f"Loading regular checkpoint: {regular_path}")
        ckpt = torch.load(regular_path, map_location=device, weights_only=True)
    else:
        print(f"ERROR: No checkpoint found in {ckpt_dir}")
        return None

    illum_net.load_state_dict(ckpt['illum_net'])
    reflec_net.load_state_dict(ckpt['reflec_net'])
    refine.load_state_dict(ckpt['refine'])
    illum_net.eval()
    reflec_net.eval()
    refine.eval()

    # Dataset (for eval, use original Pipeline2Dataset at specified size)
    ds = Pipeline2Dataset(args.data_root, split="eval15", img_size=args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPS(device)
    psnrs, ssims, lpipss = [], [], []

    out_dir = os.path.join(args.output_dir, "pipeline2_improved_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Also save intermediate outputs for pix2pix preparation
    retinex_dir = os.path.join(args.output_dir, "pipeline2_retinex_outputs")
    os.makedirs(retinex_dir, exist_ok=True)

    for i, (R_low, I_low, R_high, I_high, gt) in enumerate(tqdm(loader, desc="P2 Improved Eval")):
        R_low, I_low, gt = R_low.to(device), I_low.to(device), gt.to(device)

        I_pred = illum_net(I_low)
        R_pred = reflec_net(R_low)
        recombined = R_pred * I_pred
        pred = refine(recombined).clamp(0, 1)

        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())

        # Save enhanced output
        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img.astype(np.uint8))

        # Save reflectance and illumination for later pix2pix use
        r_img = R_pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        i_img = I_pred[0].cpu().numpy().squeeze() * 255
        recom_img = recombined[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(retinex_dir, f"{i:04d}_reflectance.png"), r_img.astype(np.uint8))
        cv2.imwrite(os.path.join(retinex_dir, f"{i:04d}_illumination.png"), i_img.astype(np.uint8))
        cv2.imwrite(os.path.join(retinex_dir, f"{i:04d}_recombined.png"), recom_img.astype(np.uint8))

    results = {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}

    # Print results
    print(f"\n{'='*60}")
    print(f"  Improved P2 Results (eval15)")
    print(f"{'='*60}")
    print(f"  PSNR : {np.mean(psnrs):.4f} dB  (±{np.std(psnrs):.2f})")
    print(f"  SSIM : {np.mean(ssims):.4f}     (±{np.std(ssims):.4f})")
    print(f"  LPIPS: {np.mean(lpipss):.5f}   (±{np.std(lpipss):.5f})")
    print(f"{'='*60}")

    # Compare with old P2
    old_metrics_path = os.path.join(args.output_dir, "metrics.json")
    if os.path.exists(old_metrics_path):
        with open(old_metrics_path) as f:
            old = json.load(f)
        old_p2 = old.get('p2', {})
        if old_p2:
            old_psnr = np.mean(old_p2['psnr'])
            old_ssim = np.mean(old_p2['ssim'])
            old_lpips = np.mean(old_p2['lpips'])
            print(f"\n  Comparison with Old P2:")
            print(f"  PSNR : {old_psnr:.4f} → {np.mean(psnrs):.4f}  "
                  f"({'+' if np.mean(psnrs) > old_psnr else ''}"
                  f"{np.mean(psnrs) - old_psnr:.2f} dB)")
            print(f"  SSIM : {old_ssim:.4f} → {np.mean(ssims):.4f}  "
                  f"({'+' if np.mean(ssims) > old_ssim else ''}"
                  f"{np.mean(ssims) - old_ssim:.4f})")
            print(f"  LPIPS: {old_lpips:.5f} → {np.mean(lpipss):.5f}  "
                  f"({'+' if np.mean(lpipss) < old_lpips else ''}"
                  f"{np.mean(lpipss) - old_lpips:.5f})")

    # Save results
    results_path = os.path.join(args.output_dir, "metrics_p2_improved.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print(f"  Retinex outputs saved to {retinex_dir}/")
    print(f"  (Ready for pix2pix input!)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Improved Pipeline 2")
    parser.add_argument('--data_root', type=str, default='datasets/LOL_dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate(args, device)
