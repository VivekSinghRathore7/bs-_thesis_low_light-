"""
fine_tune_reflectance.py
========================
Systematically compare multiple reflectance fine-tuning strategies and select
the combination that yields R_low closest to R_high.

Strategies tested
-----------------
V0  Baseline            Gaussian illumination, no R refinement
V1  BilateralIllum      Bilateral illumination, no R refinement
V2  BilateralIllum+Bil  Bilateral illumination + bilateral R denoising   (current best)
V3  BilateralIllum+NLM  Bilateral illumination + NL-Means R denoising
V4  BilateralIllum+EPF  Bilateral illumination + edge-preserving R filter
V5  V2+CLAHE            V2  + per-channel CLAHE on R
V6  V3+CLAHE            V3  + per-channel CLAHE on R
V7  GuidedIllum+Bil     Guided-filter illumination + bilateral R denoising
V8  GuidedIllum+NLM     Guided-filter illumination + NL-Means R denoising
V9  GuidedIllum+NLM+CLAHE  V8  + per-channel CLAHE  ← typically best

Metrics
-------
PSNR and SSIM between each R variant and R_high (ground-truth reflectance).
Best variant per image is highlighted; global winner shown in the summary.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
low_dir     = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
high_dir    = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/high")
out_dir     = os.path.join(BASE_DIR, "results/reflectance_finetuned")
os.makedirs(out_dir, exist_ok=True)

VARIANT_NAMES = [
    "V0 Gaussian-Illum",
    "V1 Bilateral-Illum",
    "V2 BilIllum+Bilateral-R",
    "V3 BilIllum+NLM-R",
    "V4 BilIllum+EPF-R",
    "V5 V2+CLAHE",
    "V6 V3+CLAHE",
    "V7 GuidedIllum+Bilateral-R",
    "V8 GuidedIllum+NLM-R",
    "V9 GuidedIllum+NLM+CLAHE",
]


# ── Illumination estimators ───────────────────────────────────────────────────

def illum_gaussian(img, ksize=15):
    """Original: Gaussian blur on max-channel."""
    I = np.max(img, axis=2)
    I = cv2.GaussianBlur(I, (ksize, ksize), 0)
    return np.expand_dims(I, 2)


def illum_bilateral(img, ksize=15):
    """Bilateral filter on max-channel — edge-preserving, fewer halos."""
    I = np.max(img, axis=2)
    I_u8 = (I * 255).astype(np.uint8)
    I_s   = cv2.bilateralFilter(I_u8, d=ksize, sigmaColor=75, sigmaSpace=75)
    return np.expand_dims(I_s.astype(np.float32) / 255.0, 2)


def _box_filter(img, r):
    """Integral-image based box filter (used by guided filter)."""
    return cv2.blur(img, (2 * r + 1, 2 * r + 1))


def illum_guided(img, r=16, eps=1e-3):
    """
    Self-guided filter on max-channel illumination.
    Guide = gray version of the input; input = max-channel map.
    Sharper edges than bilateral while still being globally smooth.
    """
    I   = np.max(img, axis=2).astype(np.float32)          # illumination seed
    guide = cv2.cvtColor((img * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    mean_I  = _box_filter(I, r)
    mean_g  = _box_filter(guide, r)
    mean_Ig = _box_filter(I * guide, r)
    mean_gg = _box_filter(guide * guide, r)

    cov_Ig  = mean_Ig - mean_I * mean_g
    var_g   = mean_gg - mean_g * mean_g

    a = cov_Ig / (var_g + eps)
    b = mean_I - a * mean_g

    mean_a = _box_filter(a, r)
    mean_b = _box_filter(b, r)

    I_guided = mean_a * guide + mean_b
    I_guided = np.clip(I_guided, 1e-4, 1.0)
    return np.expand_dims(I_guided, 2)


# ── Reflectance refiners ──────────────────────────────────────────────────────

def refine_bilateral(R, d=9, sc=25, ss=10):
    R_u8 = (R * 255).astype(np.uint8)
    R_ref = cv2.bilateralFilter(R_u8, d=d, sigmaColor=sc, sigmaSpace=ss)
    return R_ref.astype(np.float32) / 255.0


def refine_nlm(R, h=6, hColor=6, tws=7, sws=21):
    """NL-Means denoising — better for heavy sensor noise."""
    R_u8 = (R * 255).astype(np.uint8)
    R_den = cv2.fastNlMeansDenoisingColored(R_u8, None, h, hColor, tws, sws)
    return R_den.astype(np.float32) / 255.0


def refine_epf(R, sigma_s=10, sigma_r=0.15):
    """Edge-preserving filter (recursive domain filter)."""
    R_u8 = (R * 255).astype(np.uint8)
    R_ep  = cv2.edgePreservingFilter(R_u8, flags=1,
                                     sigma_s=sigma_s, sigma_r=sigma_r)
    return R_ep.astype(np.float32) / 255.0


def apply_clahe(R, clip=2.0, tile=(8, 8)):
    """
    Per-channel CLAHE on reflectance in LAB space.
    Boosts local micro-contrast lost during denoising.
    """
    R_u8  = (R * 255).astype(np.uint8)
    lab   = cv2.cvtColor(R_u8, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    L_eq  = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, A, B])
    R_eq  = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return R_eq.astype(np.float32) / 255.0


# ── Core decompose ────────────────────────────────────────────────────────────

def get_R(img, illum_fn, refine_fn=None, clahe=False):
    I   = illum_fn(img)
    R   = np.clip(img / (I + 1e-6), 0, 1)
    if refine_fn is not None:
        R = refine_fn(R)
    if clahe:
        R = apply_clahe(R)
    return R


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(R_est, R_ref):
    """PSNR and SSIM of R_est vs R_ref (both float32 BGR [0,1])."""
    a = (R_est * 255).astype(np.uint8)
    b = (R_ref * 255).astype(np.uint8)
    p = calc_psnr(b, a, data_range=255)
    s = calc_ssim(b, a, channel_axis=2, data_range=255)
    return p, s


# ── Variant factory ───────────────────────────────────────────────────────────

def build_variants(img):
    """Return list of (name, R) for all pipeline variants."""
    bil  = lambda R: refine_bilateral(R)
    nlm  = lambda R: refine_nlm(R)
    epf  = lambda R: refine_epf(R)
    return [
        (VARIANT_NAMES[0], get_R(img, illum_gaussian)),
        (VARIANT_NAMES[1], get_R(img, illum_bilateral)),
        (VARIANT_NAMES[2], get_R(img, illum_bilateral, bil)),
        (VARIANT_NAMES[3], get_R(img, illum_bilateral, nlm)),
        (VARIANT_NAMES[4], get_R(img, illum_bilateral, epf)),
        (VARIANT_NAMES[5], get_R(img, illum_bilateral, bil,  clahe=True)),
        (VARIANT_NAMES[6], get_R(img, illum_bilateral, nlm,  clahe=True)),
        (VARIANT_NAMES[7], get_R(img, illum_guided,    bil)),
        (VARIANT_NAMES[8], get_R(img, illum_guided,    nlm)),
        (VARIANT_NAMES[9], get_R(img, illum_guided,    nlm,  clahe=True)),
    ]


# ── Visualisation ─────────────────────────────────────────────────────────────

def to_rgb(arr):
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)


def save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx):
    n   = len(variants)
    fig = plt.figure(figsize=(28, 16))
    fig.patch.set_facecolor("#111122")

    ncols = n + 2          # low + R_high + all variants
    gs    = gridspec.GridSpec(3, ncols, figure=fig,
                              hspace=0.38, wspace=0.05,
                              left=0.02, right=0.98,
                              top=0.90, bottom=0.04)

    TKW = dict(color="white",   fontsize=7.5, fontweight="bold", pad=4)
    MKW = dict(color="#ffe066", fontsize=7,   va="top", ha="left")

    # Row 0: full images
    def make_ax(row, col):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#111122")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.axis("off")
        return ax

    # Col 0: original low-light
    ax = make_ax(0, 0)
    ax.imshow(to_rgb(low))
    ax.set_title("Low-light input", **TKW)

    # Col 1: R_high (ground truth)
    ax = make_ax(0, 1)
    ax.imshow(to_rgb(R_high))
    ax.set_title("R_high  [GT]", **TKW)

    # Cols 2..n+1: variants
    for j, (vname, R) in enumerate(variants):
        p, s = all_psnr[j], all_ssim[j]
        highlight = (j == best_idx)
        ax = make_ax(0, j + 2)
        ax.imshow(to_rgb(R))
        border_color = "#00ff88" if highlight else "#333"
        for sp in ax.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(2.5 if highlight else 0.8)
        short = vname.split(" ", 1)[1]   # drop V0/V1 prefix
        star  = " ★" if highlight else ""
        ax.set_title(f"{short}{star}", **TKW)
        ax.text(0.01, 0.99, f"PSNR {p:.2f}\nSSIM {s:.4f}",
                transform=ax.transAxes, **MKW)

    # Row 1: zoomed crops (centre quarter)
    H, W = low.shape[:2]
    cx, cy, cw, ch = W // 6, H // 6, W // 3, H // 3

    def crop(img):
        return img[cy:cy+ch, cx:cx+cw]

    ax = make_ax(1, 0);  ax.imshow(to_rgb(crop(low)));    ax.set_title("Low crop", **TKW)
    ax = make_ax(1, 1);  ax.imshow(to_rgb(crop(R_high))); ax.set_title("R_high crop", **TKW)
    for j, (vname, R) in enumerate(variants):
        ax = make_ax(1, j + 2)
        ax.imshow(to_rgb(crop(R)))
        ax.set_title(f"[crop] {vname.split(' ',1)[1]}", **TKW)
        if j == best_idx:
            for sp in ax.spines.values():
                sp.set_edgecolor("#00ff88"); sp.set_linewidth(2.5)

    # Row 2: PSNR & SSIM bar charts + diff maps side by side
    ax_p = fig.add_subplot(gs[2, :n//2 + 1])
    ax_s = fig.add_subplot(gs[2, n//2 + 1:n + 1])
    for ax in [ax_p, ax_s]:
        ax.set_facecolor("#0d0d1e")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")

    short_names = [v.split(" ", 1)[1] for v, _ in variants]
    colors_bar  = ["#00ff88" if i == best_idx else "#5599cc" for i in range(n)]

    ax_p.bar(range(n), all_psnr, color=colors_bar, alpha=0.85)
    ax_p.set_xticks(range(n)); ax_p.set_xticklabels(short_names, rotation=35,
                                                      ha="right", fontsize=7, color="#ccc")
    ax_p.set_ylabel("PSNR (dB) ↑", color="#ccc"); ax_p.set_title("PSNR vs R_high", **TKW)
    for i, v in enumerate(all_psnr):
        ax_p.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=6, color="#fff")

    ax_s.bar(range(n), all_ssim, color=colors_bar, alpha=0.85)
    ax_s.set_xticks(range(n)); ax_s.set_xticklabels(short_names, rotation=35,
                                                      ha="right", fontsize=7, color="#ccc")
    ax_s.set_ylabel("SSIM ↑", color="#ccc"); ax_s.set_title("SSIM vs R_high", **TKW)
    ax_s.set_ylim(0, 1.05)
    for i, v in enumerate(all_ssim):
        ax_s.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=6, color="#fff")

    # Diff map of best variant vs R_high
    _, R_best = variants[best_idx]
    diff = np.abs(R_best - R_high).mean(axis=2)
    ax_d = fig.add_subplot(gs[2, n + 1])
    ax_d.set_facecolor("#111122"); ax_d.axis("off")
    ax_d.imshow(np.clip(diff * 5, 0, 1), cmap="inferno")
    ax_d.set_title(f"Diff(best, R_high)×5", color="white", fontsize=7.5, pad=4)

    best_name = VARIANT_NAMES[best_idx].split(" ", 1)[1]
    fig.suptitle(
        f"Reflectance Fine-tuning — {img_name}    "
        f"Best: {best_name}  "
        f"[PSNR {all_psnr[best_idx]:.2f} dB | SSIM {all_ssim[best_idx]:.4f}]",
        color="white", fontsize=11, fontweight="bold", y=0.96
    )

    base = os.path.splitext(img_name)[0]
    plt.savefig(os.path.join(out_dir, f"{base}_Rfine.png"),
                dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ── Main loop ─────────────────────────────────────────────────────────────────

img_names = sorted(f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg')))

# Accumulate global metrics
global_psnr = np.zeros(len(VARIANT_NAMES))
global_ssim = np.zeros(len(VARIANT_NAMES))
count        = 0

print(f"Processing {len(img_names)} images …\n")
print(f"{'Image':<12}", end="")
for name in VARIANT_NAMES:
    short = name.split(" ", 1)[1][:22]
    print(f"  {short:<22}", end="")
print()
print("-" * (12 + 24 * len(VARIANT_NAMES)))

for img_name in img_names:
    base = os.path.splitext(img_name)[0]

    low  = cv2.imread(os.path.join(low_dir,  img_name)).astype(np.float32) / 255.0
    high = cv2.imread(os.path.join(high_dir, img_name)).astype(np.float32) / 255.0

    # Ground-truth reflectance: decompose R_high using the *same* best illumination
    # method so the comparison is fair.  We use guided-filter illumination for R_high.
    R_high = get_R(high, illum_guided, refine_nlm)

    variants    = build_variants(low)
    all_psnr    = []
    all_ssim    = []
    for _, R in variants:
        p, s = metrics(R, R_high)
        all_psnr.append(p)
        all_ssim.append(s)

    best_idx = int(np.argmax(all_psnr))   # rank by PSNR (SSIM usually agrees)

    global_psnr += np.array(all_psnr)
    global_ssim += np.array(all_ssim)
    count += 1

    # Save best R_low variant
    _, R_best = variants[best_idx]
    out_name = f"{base}_R_best.png"
    cv2.imwrite(os.path.join(out_dir, out_name), (R_best * 255).astype(np.uint8))

    # Save per-image comparison figure
    save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx)

    # Console row
    print(f"{base:<12}", end="")
    for j, (p, s) in enumerate(zip(all_psnr, all_ssim)):
        marker = "★" if j == best_idx else " "
        print(f"  {marker}PSNR={p:.2f} SSIM={s:.3f}", end="")
    print()

# ── Summary ───────────────────────────────────────────────────────────────────
avg_psnr = global_psnr / count
avg_ssim = global_ssim / count
overall_best = int(np.argmax(avg_psnr))

print(f"\n{'='*70}")
print(f"  GLOBAL AVERAGE METRICS  (n={count} images)")
print(f"{'='*70}")
print(f"  {'Variant':<40}  {'PSNR':>8}  {'SSIM':>8}")
print(f"  {'-'*60}")
for j, name in enumerate(VARIANT_NAMES):
    star = " ★" if j == overall_best else "  "
    print(f"  {star}{name:<38}  {avg_psnr[j]:>8.3f}  {avg_ssim[j]:>8.4f}")
print(f"\n  Best overall: {VARIANT_NAMES[overall_best]}")
print(f"  Avg PSNR = {avg_psnr[overall_best]:.3f} dB  |  Avg SSIM = {avg_ssim[overall_best]:.4f}")
print(f"{'='*70}")

# ── Summary bar-chart figure ──────────────────────────────────────────────────
short_names = [v.split(" ", 1)[1] for v in VARIANT_NAMES]
colors_bar  = ["#00ff88" if i == overall_best else "#5599cc" for i in range(len(VARIANT_NAMES))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor("#111122")

for ax, vals, ylabel, title in [
    (ax1, avg_psnr, "PSNR (dB) ↑", "Average PSNR vs R_high across all images"),
    (ax2, avg_ssim, "SSIM ↑",       "Average SSIM vs R_high across all images"),
]:
    ax.set_facecolor("#0d0d1e")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa")
    ax.bar(range(len(VARIANT_NAMES)), vals, color=colors_bar, alpha=0.88)
    ax.set_xticks(range(len(VARIANT_NAMES)))
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8, color="#ccc")
    ax.set_ylabel(ylabel, color="#ccc")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.005, f"{v:.3f}", ha="center",
                fontsize=7, color="#fff")

fig.suptitle(f"Reflectance Fine-tuning — Global Summary  (★ = best: {VARIANT_NAMES[overall_best].split(' ',1)[1]})",
             color="white", fontsize=11, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "global_summary.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nAll outputs saved to:\n  {out_dir}")
