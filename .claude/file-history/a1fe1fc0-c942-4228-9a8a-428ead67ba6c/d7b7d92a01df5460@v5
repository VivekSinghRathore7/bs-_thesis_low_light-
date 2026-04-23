"""
compare_r_final.py
==================
Simple side-by-side comparison of:
  1. R_low   — original reflectance from low-light image
  2. R_best  — fine-tuned reflectance (best variant)
  3. R_high  — ground-truth reflectance from the high-light image
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity  as calc_ssim

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
decom_dir = os.path.join(BASE_DIR, "results/decomposition")
best_dir  = os.path.join(BASE_DIR, "results/reflectance_finetuned")
out_dir   = os.path.join(BASE_DIR, "results/reflectance_final_comparison")
os.makedirs(out_dir, exist_ok=True)


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Missing: {path}")
    return img.astype(np.float32) / 255.0


def to_rgb(arr):
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)


def psnr_ssim(est, ref):
    a = (np.clip(est, 0, 1) * 255).astype(np.uint8)
    b = (np.clip(ref, 0, 1) * 255).astype(np.uint8)
    p = calc_psnr(b, a, data_range=255)
    s = calc_ssim(b, a, channel_axis=2, data_range=255)
    return p, s


img_names = sorted(
    f for f in os.listdir(decom_dir) if f.endswith("_R_low.png")
)

for fname in img_names:
    base = fname.replace("_R_low.png", "")

    R_low  = load(os.path.join(decom_dir, f"{base}_R_low.png"))
    R_high = load(os.path.join(decom_dir, f"{base}_R_high.png"))
    R_best = load(os.path.join(best_dir,  f"{base}_R_best.png"))

    R_low_inv  = 1.0 - R_low
    R_best_inv = 1.0 - R_best
    R_high_inv = 1.0 - R_high
    diff_low   = np.abs(R_low  - R_high)
    diff_best  = np.abs(R_best - R_high)

    p_low,  s_low  = psnr_ssim(R_low,  R_high)
    p_best, s_best = psnr_ssim(R_best, R_high)

    rows = [
        [
            (to_rgb(R_low),      "R_low\n(original)"),
            (to_rgb(R_best),     "R_best\n(fine-tuned)"),
            (to_rgb(R_high),     "R_high\n(ground truth)"),
        ],
        [
            (to_rgb(R_low_inv),  "R_low Inverse"),
            (to_rgb(R_best_inv), "R_best Inverse"),
            (to_rgb(R_high_inv), "R_high Inverse"),
        ],
        [
            (to_rgb(diff_low),   "|R_low − R_high|"),
            (to_rgb(diff_best),  "|R_best − R_high|"),
            None,
        ],
    ]

    fig = plt.figure(figsize=(18, 21))
    fig.patch.set_facecolor("#111122")

    # 4 rows: 3 image rows + 1 metrics text row
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[1, 1, 1, 0.22],
                           hspace=0.25, wspace=0.06,
                           left=0.03, right=0.97,
                           top=0.95, bottom=0.02)

    TKW = dict(color="white", fontsize=12, fontweight="bold", pad=8)

    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            ax = fig.add_subplot(gs[r, c])
            ax.set_facecolor("#111122")
            ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor("#333")
            if panel is None:
                continue
            img, title = panel
            ax.imshow(img)
            ax.set_title(title, **TKW)

    # ── Metrics row (spans all 3 columns) ────────────────────────────────────
    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.set_facecolor("#0d0d1e")
    ax_txt.axis("off")
    for sp in ax_txt.spines.values():
        sp.set_edgecolor("#444")

    dp = p_best - p_low
    ds = s_best - s_low
    arrow_p = "▲" if dp >= 0 else "▼"
    arrow_s = "▲" if ds >= 0 else "▼"

    table_txt = (
        f"  {'':30}  {'R_low  vs  R_high':^26}    {'R_best  vs  R_high':^26}    {'Change':^20}\n"
        f"  {'─'*110}\n"
        f"  {'PSNR (dB)  ↑  higher is better':30}  {p_low:^26.4f}    {p_best:^26.4f}    "
        f"{arrow_p} {abs(dp):.4f}\n"
        f"  {'SSIM       ↑  higher is better':30}  {s_low:^26.4f}    {s_best:^26.4f}    "
        f"{arrow_s} {abs(ds):.4f}"
    )

    ax_txt.text(0.01, 0.85, table_txt,
                transform=ax_txt.transAxes,
                color="#e8e8e8", fontsize=10.5, fontfamily="monospace",
                va="top", ha="left",
                bbox=dict(fc="#0a0a1a", ec="#445566", pad=8, boxstyle="round"))

    fig.suptitle(f"{base}.png", color="white", fontsize=15, fontweight="bold")
    plt.savefig(os.path.join(out_dir, f"{base}_R_compare_final.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {base}_R_compare_final.png")

print(f"\nAll comparisons saved to:\n  {out_dir}")
