"""
Evaluation and benchmarking for LOL-v1.
"""

import os
import argparse
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets_all_pipelines import Pipeline1Dataset, Pipeline2Dataset, Pipeline3Dataset
from models_p1_p3_unet import UNetGenerator, IllumNet, ReflecNet, RefineBlock

# ============================================================
# Metrics
# ============================================================

def calc_psnr(pred, gt):
    """PSNR between two tensors [0,1], shape (B,C,H,W)."""
    mse = F.mse_loss(pred, gt, reduction='none').mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-10))
    return psnr


def calc_ssim(pred, gt, window_size=11):
    """SSIM between two tensors [0,1], shape (B,C,H,W)."""
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
    """Lightweight LPIPS using VGG16 features (no external package needed)."""

    def __init__(self, device):
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
        self.slices = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23],# relu4_3
        ]).eval().to(device)
        for p in self.slices.parameters():
            p.requires_grad = False
        self.device = device

    def __call__(self, pred, gt):
        """Lower = more similar. Returns per-sample scores."""
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        pred_n = (pred - mean) / std
        gt_n = (gt - mean) / std

        diffs = []
        x_p, x_g = pred_n, gt_n
        for s in self.slices:
            x_p = s(x_p)
            x_g = s(x_g)
            # Normalize features
            x_p_norm = x_p / (x_p.norm(dim=1, keepdim=True) + 1e-10)
            x_g_norm = x_g / (x_g.norm(dim=1, keepdim=True) + 1e-10)
            diffs.append((x_p_norm - x_g_norm).pow(2).mean(dim=[1, 2, 3]))

        return torch.stack(diffs).mean(dim=0)


import torch.nn as nn


# ============================================================
# Inference helpers
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@torch.no_grad()
def evaluate_pipeline1(args, device):
    print("\nEvaluating Pipeline 1...")
    gen = UNetGenerator(in_ch=4, out_ch=3).to(device)
    ckpt_path = os.path.join(args.checkpoint_dir, "pipeline1", "gen_final.pth")
    gen.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    gen.eval()

    ds = Pipeline1Dataset(args.data_root, split="eval15", img_size=args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPS(device)
    psnrs, ssims, lpipss = [], [], []

    out_dir = os.path.join(args.output_dir, "pipeline1_outputs")
    os.makedirs(out_dir, exist_ok=True)

    for i, (inp, gt) in enumerate(tqdm(loader, desc="P1 Eval")):
        inp, gt = inp.to(device), gt.to(device)
        pred = gen(inp)
        pred = pred.clamp(0, 1)

        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())

        # Save output image
        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    return {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}


@torch.no_grad()
def evaluate_pipeline2(args, device):
    print("\nEvaluating Pipeline 2...")
    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)

    ckpt_path = os.path.join(args.checkpoint_dir, "pipeline2", "gen_final.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    illum_net.load_state_dict(ckpt['illum_net'])
    reflec_net.load_state_dict(ckpt['reflec_net'])
    refine.load_state_dict(ckpt['refine'])
    illum_net.eval(); reflec_net.eval(); refine.eval()

    ds = Pipeline2Dataset(args.data_root, split="eval15", img_size=args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPS(device)
    psnrs, ssims, lpipss = [], [], []

    out_dir = os.path.join(args.output_dir, "pipeline2_outputs")
    os.makedirs(out_dir, exist_ok=True)

    for i, (R_low, I_low, R_high, I_high, gt) in enumerate(tqdm(loader, desc="P2 Eval")):
        R_low, I_low, gt = R_low.to(device), I_low.to(device), gt.to(device)

        I_pred = illum_net(I_low)
        R_pred = reflec_net(R_low)
        recombined = R_pred * I_pred
        pred = refine(recombined).clamp(0, 1)

        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())

        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    return {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}


@torch.no_grad()
def evaluate_pipeline3(args, device):
    print("\nEvaluating Pipeline 3...")
    gen = UNetGenerator(in_ch=15, out_ch=3).to(device)
    ckpt_path = os.path.join(args.checkpoint_dir, "pipeline3", "gen_final.pth")
    gen.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    gen.eval()

    ds = Pipeline3Dataset(args.data_root, split="eval15", img_size=args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPS(device)
    psnrs, ssims, lpipss = [], [], []

    out_dir = os.path.join(args.output_dir, "pipeline3_outputs")
    os.makedirs(out_dir, exist_ok=True)

    for i, (inp, gt) in enumerate(tqdm(loader, desc="P3 Eval")):
        inp, gt = inp.to(device), gt.to(device)
        pred = gen(inp).clamp(0, 1)

        psnrs.append(calc_psnr(pred, gt).item())
        ssims.append(calc_ssim(pred, gt).item())
        lpipss.append(lpips_fn(pred, gt).item())

        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    return {'psnr': psnrs, 'ssim': ssims, 'lpips': lpipss}


# ============================================================
# Comparison graphs
# ============================================================

def generate_graphs(all_results, output_dir):
    """Generate comparison bar charts and per-image plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    pipeline_names = ['Pipeline 1\n(Dual-Input)', 'Pipeline 2\n(Disentangled)', 'Pipeline 3\n(Ensemble)']
    colors = ['#3b82f6', '#22c55e', '#f59e0b']

    # --- Compute means ---
    means = {}
    for metric in ['psnr', 'ssim', 'lpips']:
        means[metric] = [np.mean(all_results[f'p{i+1}'][metric]) for i in range(3)]

    # ============================================================
    # 1. Bar chart: Average PSNR
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1724')
    ax.set_facecolor('#0f1724')
    bars = ax.bar(pipeline_names, means['psnr'], color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means['psnr']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f} dB', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='white')
    ax.set_ylabel('PSNR (dB) ↑', fontsize=12, color='white')
    ax.set_title('Average PSNR Comparison', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compare_psnr.png'), dpi=150, facecolor='#0f1724')
    plt.close()

    # ============================================================
    # 2. Bar chart: Average SSIM
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1724')
    ax.set_facecolor('#0f1724')
    bars = ax.bar(pipeline_names, means['ssim'], color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means['ssim']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='white')
    ax.set_ylabel('SSIM ↑', fontsize=12, color='white')
    ax.set_title('Average SSIM Comparison', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compare_ssim.png'), dpi=150, facecolor='#0f1724')
    plt.close()

    # ============================================================
    # 3. Bar chart: Average LPIPS (lower is better)
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0f1724')
    ax.set_facecolor('#0f1724')
    bars = ax.bar(pipeline_names, means['lpips'], color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, means['lpips']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12,
                fontweight='bold', color='white')
    ax.set_ylabel('LPIPS ↓', fontsize=12, color='white')
    ax.set_title('Average LPIPS Comparison (lower = better)', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compare_lpips.png'), dpi=150, facecolor='#0f1724')
    plt.close()

    # ============================================================
    # 4. Per-image line plots
    # ============================================================
    n_images = len(all_results['p1']['psnr'])

    for metric, ylabel, title, higher_better in [
        ('psnr', 'PSNR (dB)', 'Per-Image PSNR', True),
        ('ssim', 'SSIM', 'Per-Image SSIM', True),
        ('lpips', 'LPIPS', 'Per-Image LPIPS', False),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0f1724')
        ax.set_facecolor('#0f1724')
        x = range(n_images)
        for i, (name, c) in enumerate(zip(['P1', 'P2', 'P3'], colors)):
            ax.plot(x, all_results[f'p{i+1}'][metric], color=c, marker='o',
                    markersize=4, label=name, linewidth=1.5, alpha=0.85)
        ax.set_xlabel('Image Index', color='white', fontsize=11)
        ax.set_ylabel(ylabel, color='white', fontsize=11)
        arrow = '↑' if higher_better else '↓'
        ax.set_title(f'{title} ({arrow} better)', fontsize=13, fontweight='bold', color='white')
        ax.legend(fontsize=10, facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_image_{metric}.png'), dpi=150, facecolor='#0f1724')
        plt.close()

    # ============================================================
    # 5. Combined radar/summary chart
    # ============================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1724')
    ax.set_facecolor('#0f1724')

    metrics_labels = ['PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓']
    x_pos = np.arange(3)
    width = 0.22

    for i, (name, c) in enumerate(zip(['P1', 'P2', 'P3'], colors)):
        vals = [means['psnr'][i], means['ssim'][i] * 100, means['lpips'][i] * 100]
        bars = ax.bar(x_pos + i * width, vals, width, label=name, color=c,
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, v, m in zip(bars, vals, ['psnr', 'ssim', 'lpips']):
            real_v = means[m][i]
            fmt = f'{real_v:.2f}' if m == 'psnr' else f'{real_v:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    fmt, ha='center', va='bottom', fontsize=9, color='white')

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(metrics_labels, fontsize=11, color='white')
    ax.set_title('Pipeline Comparison Summary', fontsize=14, fontweight='bold', color='white')
    ax.legend(fontsize=10, facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
    ax.set_ylabel('Score (SSIM & LPIPS ×100)', color='white', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_comparison.png'), dpi=150, facecolor='#0f1724')
    plt.close()

    # ============================================================
    # 6. Training loss curves (if history files exist)
    # ============================================================
    histories = {}
    for i in range(1, 4):
        hist_path = os.path.join(args.checkpoint_dir, f"pipeline{i}", "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[f'p{i}'] = json.load(f)

    if histories:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f1724')
        ax.set_facecolor('#0f1724')
        for i, c in enumerate(colors):
            key = f'p{i+1}'
            if key in histories:
                ax.plot(histories[key]['g_loss'], color=c, label=f'P{i+1} G_loss',
                        linewidth=1.5, alpha=0.85)
        ax.set_xlabel('Epoch', color='white', fontsize=11)
        ax.set_ylabel('Generator Loss', color='white', fontsize=11)
        ax.set_title('Training Loss Curves', fontsize=13, fontweight='bold', color='white')
        ax.legend(fontsize=10, facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_color('#334155')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150, facecolor='#0f1724')
        plt.close()

    print(f"\n📊 All graphs saved to {output_dir}/")


# ============================================================
# Print summary table
# ============================================================

def print_summary(all_results):
    print("\n" + "="*65)
    print(f"{'Pipeline':<25} {'PSNR↑':>10} {'SSIM↑':>10} {'LPIPS↓':>10}")
    print("="*65)
    names = [
        'P1 (Dual-Input)',
        'P2 (Disentangled)',
        'P3 (Ensemble)',
    ]
    for i, name in enumerate(names):
        key = f'p{i+1}'
        p = np.mean(all_results[key]['psnr'])
        s = np.mean(all_results[key]['ssim'])
        l = np.mean(all_results[key]['lpips'])
        print(f"{name:<25} {p:>10.2f} {s:>10.4f} {l:>10.4f}")
    print("="*65)

    # Find best per metric
    best_psnr = np.argmax([np.mean(all_results[f'p{i+1}']['psnr']) for i in range(3)])
    best_ssim = np.argmax([np.mean(all_results[f'p{i+1}']['ssim']) for i in range(3)])
    best_lpips = np.argmin([np.mean(all_results[f'p{i+1}']['lpips']) for i in range(3)])
    print(f"\n🏆 Best PSNR:  Pipeline {best_psnr+1}")
    print(f"🏆 Best SSIM:  Pipeline {best_ssim+1}")
    print(f"🏆 Best LPIPS: Pipeline {best_lpips+1}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLIE Pipelines")
    parser.add_argument('--data_root', type=str, default='datasets/LOL_dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--img_size', type=int, default=256)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    all_results = {}
    all_results['p1'] = evaluate_pipeline1(args, device)
    all_results['p2'] = evaluate_pipeline2(args, device)
    all_results['p3'] = evaluate_pipeline3(args, device)

    # Save raw results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print_summary(all_results)
    generate_graphs(all_results, os.path.join(args.output_dir, "graphs"))
    print("\n✅ Evaluation complete!")
