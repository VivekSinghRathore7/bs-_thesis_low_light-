"""
Detailed comparison: Old R_low vs Fine-tuned R_low vs R_enh

Fine-tuning pipeline explained:
  Step 1 — Better Illumination Estimation
          Old: Gaussian blur on max-channel I  → blurs across edges → halos in R
          New: Bilateral filter on max-channel I → edge-preserving smoothing → sharper I boundaries
          Effect: R = S/I has less halo leakage around object edges

  Step 2 — Reflectance Denoising (refine_reflectance)
          R = S/I amplifies noise in dark pixels (I ~ 0 → R spikes)
          New: Bilateral filter on R itself (sigma_color=25, sigma_space=10)
          Effect: Smooths noise in flat/dark regions while keeping edges & textures intact

  Result: R_enh = decompose(R_low × I_high) — final reconstruction feeds cleaner R,
          giving a sharper, less-noisy enhanced output
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

low_dir     = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
high_dir    = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/high")
decom_dir   = os.path.join(BASE_DIR, "results/decomposition")
enhanced_dir= os.path.join(BASE_DIR, "results/enhanced")
out_dir     = os.path.join(BASE_DIR, "results/r_comparison_detailed")
os.makedirs(out_dir, exist_ok=True)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def decompose_old(img, blur_ksize=15):
    """Original: Gaussian illumination, no refinement."""
    I = np.max(img, axis=2)
    I = cv2.GaussianBlur(I, (blur_ksize, blur_ksize), 0)
    I = np.expand_dims(I, axis=2)
    R = np.clip(img / (I + 1e-6), 0, 1)
    return R, I.squeeze()

def decompose_new(img, blur_ksize=15):
    """Fine-tuned: Bilateral illumination + bilateral reflectance refinement."""
    I = np.max(img, axis=2)
    I_u8 = (I * 255).astype(np.uint8)
    I_smooth = cv2.bilateralFilter(I_u8, d=blur_ksize, sigmaColor=75, sigmaSpace=75)
    I = np.expand_dims(I_smooth.astype(np.float32) / 255.0, axis=2)
    R = np.clip(img / (I + 1e-6), 0, 1)
    R_u8 = (R * 255).astype(np.uint8)
    R = cv2.bilateralFilter(R_u8, d=9, sigmaColor=25, sigmaSpace=10).astype(np.float32) / 255.0
    return R, I.squeeze()

def noise_level(R):
    """Estimate noise as std of Laplacian (higher = more noise)."""
    gray = cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

def edge_strength(R):
    """Mean gradient magnitude — proxy for sharpness."""
    gray = cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx**2 + gy**2))

def to_display(arr, bgr=True):
    """Float [0,1] BGR → uint8 RGB for matplotlib."""
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB) if bgr else u8

def add_crop_box(ax, crop, color="yellow", lw=2):
    x, y, w, h = crop
    rect = patches.Rectangle((x, y), w, h, linewidth=lw, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

def get_crop(img, crop):
    x, y, w, h = crop
    return img[y:y+h, x:x+w]


# ── Per-image comparison ───────────────────────────────────────────────────────

img_names = sorted(f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg')))

for img_name in img_names:
    base = os.path.splitext(img_name)[0]

    low  = cv2.imread(os.path.join(low_dir,  img_name)).astype(np.float32) / 255.0
    high = cv2.imread(os.path.join(high_dir, img_name)).astype(np.float32) / 255.0
    enh  = cv2.imread(os.path.join(enhanced_dir, f"{base}_enhanced.png"))
    if enh is not None:
        enh = enh.astype(np.float32) / 255.0

    R_old, I_old = decompose_old(low)
    R_new, I_new = decompose_new(low)

    # R_enh: reflectance extracted from the enhanced reconstruction
    if enh is not None:
        R_enh, _ = decompose_new(enh)
    else:
        R_enh = R_new  # fallback

    # Metrics
    nl_old  = noise_level(R_old)
    nl_new  = noise_level(R_new)
    nl_enh  = noise_level(R_enh)
    es_old  = edge_strength(R_old)
    es_new  = edge_strength(R_new)
    es_enh  = edge_strength(R_enh)
    diff_map = np.abs(R_new - R_old).mean(axis=2)

    # Choose a meaningful crop (centre-left quarter, avoids black borders)
    H, W = low.shape[:2]
    cx, cy = W // 6, H // 6
    cw, ch = W // 3, H // 3
    crop = (cx, cy, cw, ch)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 14))
    fig.patch.set_facecolor("#1a1a2e")

    gs = GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.08,
                  left=0.03, right=0.97, top=0.88, bottom=0.04)

    def styled_ax(row, col, span_col=1):
        ax = fig.add_subplot(gs[row, col:col+span_col])
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        return ax

    TITLE_KW  = dict(color="white",    fontsize=10, fontweight="bold", pad=5)
    LABEL_KW  = dict(color="#cccccc",  fontsize=8)
    METRIC_KW = dict(color="#f0f0a0",  fontsize=8)

    # Row 0: full images
    panels_full = [
        (to_display(low),        "Input — Low-light",                    None),
        (to_display(R_old),      "R_low  [OLD: Gaussian]",               nl_old),
        (to_display(R_new),      "R_low  [NEW: Bilateral refined]",      nl_new),
        (to_display(R_enh),      "R_enh  [from enhanced S = R×I_high]",  nl_enh),
        (diff_map,               "Diff |R_new − R_old| (×3)",            None),
    ]
    for col, (im, title, nl) in enumerate(panels_full):
        ax = styled_ax(0, col)
        if col == 4:  # diff map
            ax.imshow(np.clip(im * 3, 0, 1), cmap="inferno")
        else:
            cmap = None if im.ndim == 3 else "gray"
            ax.imshow(im, cmap=cmap)
            add_crop_box(ax, crop)
        ax.set_title(title, **TITLE_KW)
        ax.axis("off")
        if nl is not None:
            ax.text(0.02, 0.03, f"Noise: {nl:.1f}  |  Edges: {edge_strength(R_old if col==1 else (R_new if col==2 else R_enh)):.1f}",
                    transform=ax.transAxes, **METRIC_KW,
                    bbox=dict(fc="#000000aa", ec="none", pad=2))

    # Row 1: zoomed crops
    crops = [
        get_crop(to_display(low),   crop),
        get_crop(to_display(R_old), crop),
        get_crop(to_display(R_new), crop),
        get_crop(to_display(R_enh), crop),
        get_crop(diff_map,          crop),
    ]
    for col, (im, title, _) in enumerate(panels_full):
        ax = styled_ax(1, col)
        if col == 4:
            ax.imshow(np.clip(crops[col] * 3, 0, 1), cmap="inferno")
        else:
            cmap = None if crops[col].ndim == 3 else "gray"
            ax.imshow(crops[col], cmap=cmap)
        ax.set_title(f"[Zoomed crop] {title.split('[')[0].strip()}", **TITLE_KW)
        ax.axis("off")

    # Row 2: noise histograms + explanation text
    ax_hist = styled_ax(2, 0, span_col=2)
    for R, label, color in [
        (R_old, "Old R_low (Gaussian)",   "#ff6b6b"),
        (R_new, "New R_low (Bilateral)",  "#6bcb77"),
        (R_enh, "R_enh",                  "#4d96ff"),
    ]:
        gray = cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        ax_hist.hist(gray.flatten(), bins=64, range=(0, 255),
                     alpha=0.6, label=label, color=color, density=True)
    ax_hist.set_title("Reflectance pixel distribution", **TITLE_KW)
    ax_hist.set_xlabel("Pixel intensity", **LABEL_KW)
    ax_hist.set_ylabel("Density", **LABEL_KW)
    ax_hist.legend(fontsize=8, facecolor="#2a2a4a", labelcolor="white")
    ax_hist.set_facecolor("#12122a")
    ax_hist.tick_params(colors="#aaa")
    for sp in ax_hist.spines.values(): sp.set_edgecolor("#444")

    # Metrics bar chart
    ax_bar = styled_ax(2, 2, span_col=1)
    labels = ["Old R_low", "New R_low", "R_enh"]
    noises = [nl_old, nl_new, nl_enh]
    edges  = [es_old, es_new, es_enh]
    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax_bar.bar(x - w/2, noises, w, label="Noise (Laplacian var)", color="#ff6b6b", alpha=0.85)
    bars2 = ax_bar.bar(x + w/2, edges,  w, label="Edge strength",         color="#6bcb77", alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, color="#ccc", fontsize=8)
    ax_bar.set_title("Noise vs Edge Strength", **TITLE_KW)
    ax_bar.legend(fontsize=7, facecolor="#2a2a4a", labelcolor="white")
    ax_bar.set_facecolor("#12122a")
    ax_bar.tick_params(colors="#aaa")
    for sp in ax_bar.spines.values(): sp.set_edgecolor("#444")

    # Explanation text panel
    ax_txt = styled_ax(2, 3, span_col=2)
    ax_txt.axis("off")
    explanation = (
        "HOW THE REFLECTANCE WAS FINE-TUNED\n"
        "─────────────────────────────────────────────────────────────────\n\n"
        "STEP 1 — Better Illumination Estimation\n"
        "  Old: GaussianBlur(I)   → bleeds across edges → halos leak into R\n"
        "  New: BilateralFilter(I) → preserves edge boundaries → cleaner I\n"
        "  Why it matters: R = S / I, so a sharper I gives a sharper R.\n\n"
        "STEP 2 — Reflectance Denoising\n"
        "  Problem: In dark pixels I≈0, so S/I amplifies sensor noise.\n"
        "  Fix: BilateralFilter(R, d=9, σ_color=25, σ_space=10)\n"
        "       → smooths noise in flat/dark regions\n"
        "       → keeps textures & edges because bilateral is edge-aware.\n\n"
        "RESULT — R_enh\n"
        "  Enhanced image = R_low (clean) × I_high\n"
        "  Feeding a denoised R into reconstruction reduces noise in final S.\n"
        "  Expect: ↓ Noise, ≈ Edge Strength (structure preserved)."
    )
    ax_txt.text(0.02, 0.97, explanation,
                transform=ax_txt.transAxes,
                color="#e0e0e0", fontsize=8.2,
                fontfamily="monospace",
                va="top", ha="left",
                bbox=dict(fc="#12122a", ec="#444", pad=8, boxstyle="round"))

    noise_delta = ((nl_old - nl_new) / nl_old * 100) if nl_old > 0 else 0
    fig.suptitle(
        f"Reflectance Fine-tuning — {img_name}    "
        f"[Noise reduction: {noise_delta:.1f}%]",
        color="white", fontsize=13, fontweight="bold", y=0.95
    )

    out_path = os.path.join(out_dir, f"{base}_R_detailed.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [{base}]  Noise: {nl_old:.1f} → {nl_new:.1f} ({noise_delta:+.1f}%)  |  "
          f"Edges: {es_old:.1f} → {es_new:.1f}")

print(f"\nDetailed comparisons saved to:\n  {out_dir}")
