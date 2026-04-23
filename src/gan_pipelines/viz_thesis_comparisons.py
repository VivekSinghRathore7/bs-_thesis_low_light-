"""
generate_final_comparison.py
Creates a finalized thesis-grade comparison figure:
  Low-Light | Ground Truth | Pipeline 1 | Pipeline 2 | Pipeline 3 | RetinexRestormer

Selects the best 4 images by average PSNR across all methods.
Annotates each cell with per-image PSNR/SSIM.
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
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, "..", "..", ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "LOL_dataset", "eval15")
LOW_DIR = os.path.join(DATA_ROOT, "low")
HIGH_DIR = os.path.join(DATA_ROOT, "high")

RESULTS_DIR = os.path.join(BASE, "results")
P1_DIR = os.path.join(RESULTS_DIR, "pipeline1_outputs")
P2_DIR = os.path.join(RESULTS_DIR, "pipeline2_outputs")
P3_DIR = os.path.join(RESULTS_DIR, "pipeline3_outputs")
RR_DIR = os.path.join(PROJECT_ROOT, "results", "sure_shot_eval")

METRICS_P123 = os.path.join(RESULTS_DIR, "metrics.json")
METRICS_RR = os.path.join(RR_DIR, "metrics.json")

OUT_DIR = os.path.join(RESULTS_DIR, "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)


# ─── Load metrics ────────────────────────────────────────────────────
def load_all_metrics():
    with open(METRICS_P123) as f:
        m123 = json.load(f)
    with open(METRICS_RR) as f:
        mrr = json.load(f)
    return m123, mrr


# ─── Select best 4 images by average PSNR ────────────────────────────
def select_best_images(m123, mrr, top_k=4):
    n = len(m123['p1']['psnr'])
    avg_psnr = []
    for i in range(n):
        avg = np.mean([
            m123['p1']['psnr'][i],
            m123['p2']['psnr'][i],
            m123['p3']['psnr'][i],
            mrr['psnr'][i],
        ])
        avg_psnr.append((avg, i))
    avg_psnr.sort(reverse=True)
    return [idx for _, idx in avg_psnr[:top_k]]


# ─── Main figure generation ──────────────────────────────────────────
def generate_comparison():
    m123, mrr = load_all_metrics()
    best_indices = select_best_images(m123, mrr, top_k=4)

    # Get sorted low-light filenames to map index → filename
    filenames = sorted([
        f for f in os.listdir(LOW_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print("=" * 70)
    print("  Finalized Output Comparison — Best 4 Images")
    print("=" * 70)

    # Collect per-image metrics for annotation
    all_psnr = {
        'P1': m123['p1']['psnr'], 'P2': m123['p2']['psnr'],
        'P3': m123['p3']['psnr'], 'RR': mrr['psnr'],
    }
    all_ssim = {
        'P1': m123['p1']['ssim'], 'P2': m123['p2']['ssim'],
        'P3': m123['p3']['ssim'], 'RR': mrr['ssim'],
    }

    n_rows = len(best_indices)
    n_cols = 6  # Low, GT, P1, P2, P3, RetinexRestormer

    col_labels = [
        'Low-Light Input',
        'Ground Truth',
        'Pipeline 1\n(Dual-Input cGAN)',
        'Pipeline 2\n(Disentangled GAN)',
        'Pipeline 3\n(Ensemble GAN)',
        'RetinexRestormer\n(Ours)',
    ]

    # Color palette for column headers
    col_colors = [
        '#495057',   # dark gray for input
        '#2E7D32',   # deep green for GT
        '#1565C0',   # deep blue for P1
        '#E65100',   # deep orange for P2
        '#2E7D32',   # deep green for P3 (differentiated by label)
        '#C62828',   # deep red for RetinexRestormer
    ]
    col_bg_light = [
        '#F8F9FA',   # light gray
        '#E8F5E9',   # light green
        '#E3F2FD',   # light blue
        '#FFF3E0',   # light orange
        '#E8F5E9',   # light green
        '#FFEBEE',   # light red
    ]

    # Publication font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })

    fig = plt.figure(figsize=(26, 19), facecolor='white')

    # Create gridspec with proper spacing
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[0.12] + [1.0] * n_rows,
        width_ratios=[1] * n_cols,
        hspace=0.06,
        wspace=0.03,
        left=0.06, right=0.98,
        top=0.92, bottom=0.06,
    )

    # ─── Column headers ──────────────────────────────────────────────
    for col in range(n_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Rounded header box
        ax.add_patch(FancyBboxPatch(
            (0.03, 0.08), 0.94, 0.84,
            boxstyle="round,pad=0.06",
            facecolor=col_colors[col],
            edgecolor='white',
            linewidth=2.5,
            alpha=0.95,
        ))
        ax.text(0.5, 0.5, col_labels[col],
                ha='center', va='center',
                fontsize=11, fontweight='bold',
                color='white',
                linespacing=1.3,
                transform=ax.transAxes)

    # ─── Image grid ──────────────────────────────────────────────────
    for row_i, img_idx in enumerate(best_indices):
        fname = filenames[img_idx]
        img_id = os.path.splitext(fname)[0]  # e.g., "23", "778"

        # Load all versions
        low = cv2.imread(os.path.join(LOW_DIR, fname))
        gt = cv2.imread(os.path.join(HIGH_DIR, fname))
        p1 = cv2.imread(os.path.join(P1_DIR, f"{img_idx:04d}.png"))
        p2 = cv2.imread(os.path.join(P2_DIR, f"{img_idx:04d}.png"))
        p3 = cv2.imread(os.path.join(P3_DIR, f"{img_idx:04d}.png"))
        rr = cv2.imread(os.path.join(RR_DIR, f"{img_idx:04d}.png"))

        images = [low, gt, p1, p2, p3, rr]
        method_keys = [None, None, 'P1', 'P2', 'P3', 'RR']

        print(f"\n  Image {img_idx} ({fname}):")

        for col_i, (img, mkey) in enumerate(zip(images, method_keys)):
            ax = fig.add_subplot(gs[row_i + 1, col_i])

            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb, aspect='auto')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=16, color='red', transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # Colored border per column
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(col_colors[col_i])
                spine.set_linewidth(2.5)

            # Annotate with PSNR/SSIM for model outputs
            if mkey is not None:
                psnr_val = all_psnr[mkey][img_idx]
                ssim_val = all_ssim[mkey][img_idx]
                print(f"    {mkey}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}")

                # Semi-transparent label at bottom
                ax.text(
                    0.5, 0.03,
                    f"PSNR: {psnr_val:.2f}  |  SSIM: {ssim_val:.4f}",
                    ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold',
                    color='white',
                    transform=ax.transAxes,
                    bbox=dict(
                        boxstyle='round,pad=0.25',
                        facecolor='black',
                        alpha=0.75,
                        edgecolor=col_colors[col_i],
                        linewidth=1.5,
                    ),
                )

            # Image ID label on left side
            if col_i == 0:
                ax.set_ylabel(
                    f"Image #{img_id}",
                    fontsize=12, fontweight='bold',
                    color='#2C3E50', rotation=90,
                    labelpad=10,
                )

    # ─── Title ────────────────────────────────────────────────────────
    fig.suptitle(
        'Qualitative Comparison of Low-Light Image Enhancement Methods\non LOL eval15 Dataset (Best 4 Images by Average PSNR)',
        fontsize=18, fontweight='bold', y=0.97,
        color='#1A1A2E', linespacing=1.4,
    )

    # ─── Mean metrics footer ──────────────────────────────────────────
    footer_parts = []
    for mkey, label, color in [
        ('P1', 'Pipeline 1', col_colors[2]),
        ('P2', 'Pipeline 2', col_colors[3]),
        ('P3', 'Pipeline 3', col_colors[4]),
        ('RR', 'RetinexRestormer', col_colors[5]),
    ]:
        if mkey == 'RR':
            mean_psnr = mrr['mean']['psnr']
            mean_ssim = mrr['mean']['ssim']
        else:
            pvals = m123[mkey.lower()]['psnr']
            svals = m123[mkey.lower()]['ssim']
            mean_psnr = np.mean(pvals)
            mean_ssim = np.mean(svals)
        footer_parts.append(f"{label}: PSNR={mean_psnr:.2f}, SSIM={mean_ssim:.4f}")

    footer_text = "   │   ".join(footer_parts)
    fig.text(0.5, 0.02, f"Mean Metrics (eval15 — 15 images):  {footer_text}",
             ha='center', va='bottom', fontsize=10.5,
             color='#333333', style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#CCCCCC', alpha=0.9))

    # ─── Save ─────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "final_comparison_best4.png")
    fig.savefig(out_path, facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✅ Final comparison saved → {out_path}")
    print(f"   File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    return out_path


if __name__ == "__main__":
    generate_comparison()
