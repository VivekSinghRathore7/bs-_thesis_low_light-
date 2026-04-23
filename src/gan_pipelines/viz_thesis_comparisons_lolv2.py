"""
viz_thesis_comparisons_lolv2.py
Creates a finalized thesis-grade comparison figure for LOL-v2 Real test set.
Resolves cross-indexing since RetinexRestormer saved outputs alphabetically (0000.png ... 0099.png), 
while GANs saved outputs using original filenames sorted by brightness.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import cv2

# ─── Paths ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE)

DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "LOL_v2_real", "Test")
LOW_DIR = os.path.join(DATA_ROOT, "Input")
HIGH_DIR = os.path.join(DATA_ROOT, "GT")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "lolv2")
P1_DIR = os.path.join(RESULTS_DIR, "gan_p1_images")
P2_DIR = os.path.join(RESULTS_DIR, "gan_p2_images")
P3_DIR = os.path.join(RESULTS_DIR, "gan_p3_images")
RR_DIR = os.path.join(RESULTS_DIR, "retinex_restormer_images")

METRICS_P123 = os.path.join(RESULTS_DIR, "metrics", "metrics_lolv2_gan.json")
METRICS_RR = os.path.join(RR_DIR, "metrics_lolv2.json")

OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "lolv2_comparisons")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Map metrics to absolute filenames ───────────────────────────────
def get_rr_mapping():
    # RetinexRestormer evaluates sorted alphabetically
    all_fnames = sorted([f for f in os.listdir(LOW_DIR) if f.endswith(('.png', '.jpg'))])
    with open(METRICS_RR) as f: mrr = json.load(f)
    mapping = {}
    for i, fname in enumerate(all_fnames):
        mapping[fname] = {
            'psnr': mrr['psnr'][i],
            'ssim': mrr['ssim'][i],
            'img_path': os.path.join(RR_DIR, f"{i:04d}.png")
        }
    return mapping

def get_gan_mapping():
    # GAN evaluates sorted by brightness
    all_fnames = sorted([f for f in os.listdir(LOW_DIR) if f.endswith(('.png', '.jpg'))])
    brightness_list = []
    for f in all_fnames:
        img = cv2.imread(os.path.join(LOW_DIR, f))
        brightness = img.mean() if img is not None else 0
        brightness_list.append((f, brightness))
    brightness_list.sort(key=lambda x: x[1], reverse=True)
    sorted_fnames = [x[0] for x in brightness_list]

    with open(METRICS_P123) as f: m123 = json.load(f)
    mapping = {}
    for i, fname in enumerate(sorted_fnames):
        mapping[fname] = {
            'P1': {'psnr': m123['p1']['psnr'][i], 'ssim': m123['p1']['ssim'][i], 'img_path': os.path.join(P1_DIR, fname)},
            'P2': {'psnr': m123['p2']['psnr'][i], 'ssim': m123['p2']['ssim'][i], 'img_path': os.path.join(P2_DIR, fname)},
            'P3': {'psnr': m123['p3']['psnr'][i], 'ssim': m123['p3']['ssim'][i], 'img_path': os.path.join(P3_DIR, fname)}
        }
    return mapping

# ─── Main Generation ─────────────────────────────────────────────────
def generate_lolv2_comparison():
    rr_map = get_rr_mapping()
    gan_map = get_gan_mapping()
    
    # Calculate average PSNR across all methods to select the best 4 images
    avg_psnr_list = []
    for fname in rr_map.keys():
        avg = np.mean([
            gan_map[fname]['P1']['psnr'],
            gan_map[fname]['P2']['psnr'],
            gan_map[fname]['P3']['psnr'],
            rr_map[fname]['psnr']
        ])
        avg_psnr_list.append((avg, fname))
    
    avg_psnr_list.sort(reverse=True)
    best_4_fnames = [fname for _, fname in avg_psnr_list[:4]]

    print("=" * 70)
    print("  Finalized Output Comparison (LOL-v2) — Best 4 Images")
    print("=" * 70)

    n_rows = 4
    n_cols = 6  # Low, GT, P1, P2, P3, RetinexRestormer

    col_labels = [
        'Low-Light Input', 'Ground Truth', 'Pipeline 1\n(Dual-Input cGAN)',
        'Pipeline 2\n(Disentangled GAN)', 'Pipeline 3\n(Ensemble GAN)', 'RetinexRestormer\n(Ours)'
    ]
    col_colors = ['#495057','#2E7D32','#1565C0','#E65100','#2E7D32','#C62828']

    plt.rcParams.update({'font.family':'serif', 'font.size':11, 'figure.dpi':300})
    fig = plt.figure(figsize=(26, 19), facecolor='white')

    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[0.12] + [1.0] * n_rows, width_ratios=[1] * n_cols,
        hspace=0.06, wspace=0.03, left=0.06, right=0.98, top=0.92, bottom=0.06,
    )

    # Headers
    for col in range(n_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.08), 0.94, 0.84, boxstyle="round,pad=0.06",
            facecolor=col_colors[col], edgecolor='white', linewidth=2.5, alpha=0.95,
        ))
        ax.text(0.5, 0.5, col_labels[col], ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', linespacing=1.3, transform=ax.transAxes)

    # Grid
    for row_i, fname in enumerate(best_4_fnames):
        img_id = os.path.splitext(fname)[0]

        low = cv2.imread(os.path.join(LOW_DIR, fname))
        gt = cv2.imread(os.path.join(HIGH_DIR, fname))
        p1 = cv2.imread(gan_map[fname]['P1']['img_path'])
        p2 = cv2.imread(gan_map[fname]['P2']['img_path'])
        p3 = cv2.imread(gan_map[fname]['P3']['img_path'])
        rr = cv2.imread(rr_map[fname]['img_path'])

        images = [low, gt, p1, p2, p3, rr]
        # Metadata mapped to column indices
        metadata = [
            None, None, 
            {'psnr': gan_map[fname]['P1']['psnr'], 'ssim': gan_map[fname]['P1']['ssim']},
            {'psnr': gan_map[fname]['P2']['psnr'], 'ssim': gan_map[fname]['P2']['ssim']},
            {'psnr': gan_map[fname]['P3']['psnr'], 'ssim': gan_map[fname]['P3']['ssim']},
            {'psnr': rr_map[fname]['psnr'], 'ssim': rr_map[fname]['ssim']}
        ]

        print(f"\n  Image {fname}:")
        for col_i, (img, meta) in enumerate(zip(images, metadata)):
            ax = fig.add_subplot(gs[row_i + 1, col_i])

            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb, aspect='auto')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='red', transform=ax.transAxes)

            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(col_colors[col_i]); spine.set_linewidth(2.5)

            if meta is not None:
                print(f"    Col {col_i}: PSNR={meta['psnr']:.2f}, SSIM={meta['ssim']:.4f}")
                ax.text(
                    0.5, 0.03, f"PSNR: {meta['psnr']:.2f}  |  SSIM: {meta['ssim']:.4f}",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='white', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.75, edgecolor=col_colors[col_i], linewidth=1.5),
                )

            if col_i == 0:
                ax.set_ylabel(f"Image #{img_id}", fontsize=12, fontweight='bold', color='#2C3E50', rotation=90, labelpad=10)

    # Title & Global Mean
    fig.suptitle('Qualitative Comparison of Low-Light Image Enhancement Methods\non LOL-v2 Real Dataset (Best 4 Images by Average PSNR)',
                 fontsize=18, fontweight='bold', y=0.97, color='#1A1A2E', linespacing=1.4)

    out_path = os.path.join(OUT_DIR, "final_comparison_best4_lolv2.png")
    fig.savefig(out_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✅ Final comparison saved → {out_path}")

if __name__ == "__main__":
    generate_lolv2_comparison()
