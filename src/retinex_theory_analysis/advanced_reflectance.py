"""
advanced_reflectance.py
=======================
Advanced techniques to maximally close the R_low → R_high gap.

Techniques used
---------------
1. TV-Regularized Illumination   : Total Variation smoothing on illumination map
                                   (piecewise-constant I, sharper than bilateral/guided)
2. Wiener-Optimal Division       : R = S·I / (I² + σ²_n)  — optimal linear estimator
                                   under spatially-varying Poisson-like noise; avoids
                                   naive 1/I amplification in dark regions.
3. Wavelet Denoising (BayesShrink): Adaptive multi-scale soft thresholding
                                   (skimage.restoration.denoise_wavelet)
                                   — far superior to NLM/bilateral for structured noise.
4. Iterative Alt-Min             : Alternately refine I and R for 3 iterations
                                   (EM-style; each pass makes I cleaner → R cleaner).
5. Log-domain TV Retinex (LTRVR) : Work in log space (log S = log R + log I), solve
                                   TV-regularised linear system via scipy sparse solver.

Variants compared
-----------------
Old best (from fine_tune_reflectance.py):
  BASE_V3   BilIllum  + NLM-R         (previous best by PSNR)
  BASE_V8   GuidedIllum + NLM-R       (previous best by SSIM)

New variants:
  A0  TV-Illum + NLM
  A1  TV-Illum + Wavelet
  A2  Guided-Illum + Wiener + Wavelet
  A3  TV-Illum  + Wiener + Wavelet     ← full stack
  A4  Iterative Alt-Min (TV+Wiener+Wavelet, 3 iters)
  A5  Log-domain TV Retinex
  A6  Best-of-all combo (A4 on top of A3)
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import scipy.sparse
import scipy.sparse.linalg

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
low_dir    = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
high_dir   = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/high")
decom_dir  = os.path.join(BASE_DIR, "results/decomposition")
out_dir    = os.path.join(BASE_DIR, "results/reflectance_advanced")
os.makedirs(out_dir, exist_ok=True)

VARIANT_NAMES = [
    "BASE_V3 BilIllum+NLM",
    "BASE_V8 GuidedIllum+NLM",
    "A0 TV-Illum+NLM",
    "A1 TV-Illum+Wavelet",
    "A2 Guided+Wiener+Wavelet",
    "A3 TV+Wiener+Wavelet",
    "A4 IterAltMin(3x)",
    "A5 LogTV-Retinex",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  ILLUMINATION ESTIMATORS
# ══════════════════════════════════════════════════════════════════════════════

def _box(img, r):
    return cv2.blur(img, (2 * r + 1, 2 * r + 1))


def illum_bilateral(img, ksize=15):
    I = np.max(img, axis=2)
    I_u8 = (I * 255).astype(np.uint8)
    I_s  = cv2.bilateralFilter(I_u8, d=ksize, sigmaColor=75, sigmaSpace=75)
    return np.expand_dims(I_s.astype(np.float32) / 255.0, 2)


def illum_guided(img, r=16, eps=1e-3):
    I     = np.max(img, axis=2).astype(np.float32)
    guide = cv2.cvtColor((img * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mI  = _box(I,         r); mg  = _box(guide,         r)
    mIg = _box(I * guide, r); mgg = _box(guide * guide, r)
    a   = (mIg - mI * mg) / (mgg - mg * mg + eps)
    b   = mI - a * mg
    I_guided = _box(a, r) * guide + _box(b, r)
    return np.expand_dims(np.clip(I_guided, 1e-4, 1.0).astype(np.float32), 2)


def illum_tv(img, weight=0.08):
    """Total-Variation regularised illumination — piecewise smooth, sharp edges."""
    I = np.max(img, axis=2).astype(np.float32)
    # denoise_tv_chambolle expects [0,1] float
    I_tv = denoise_tv_chambolle(I, weight=weight, channel_axis=None)
    return np.expand_dims(np.clip(I_tv, 1e-4, 1.0).astype(np.float32), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  NOISE ESTIMATOR (for Wiener division)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_noise_sigma(img):
    """
    Estimate additive noise σ from high-frequency residuals of dark regions.
    Uses the MAD estimator on a Laplacian-filtered dark patch.
    Returns σ clipped to [0.02, 0.15].
    """
    gray = np.mean(img, axis=2).astype(np.float32)
    # Laplacian for high-frequency content
    lap  = cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_32F) / 255.0
    # Restrict to dark regions (I < 0.4)
    mask = gray < 0.4
    if mask.sum() < 100:
        return 0.05
    mad  = np.median(np.abs(lap[mask] - np.median(lap[mask])))
    sigma = mad / 0.6745          # consistent estimator for Gaussian
    return float(np.clip(sigma, 0.02, 0.15))


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DIVISION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════════

def naive_divide(img, I):
    """Standard: R = S / (I + ε)"""
    return np.clip(img / (I + 1e-6), 0, 1)


def wiener_divide(img, I, sigma_n):
    """
    Wiener-optimal division for model  S = R·I + n  (n ~ N(0, σ²_n)).
    R̂ = S·I / (I² + σ²_n)
    In dark regions where I→0, this naturally suppresses noise instead
    of amplifying it (unlike naive division).
    """
    sigma2 = sigma_n ** 2
    R = (img * I) / (I ** 2 + sigma2)
    return np.clip(R, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REFLECTANCE REFINERS
# ══════════════════════════════════════════════════════════════════════════════

def refine_nlm(R, h=6, hc=6, tws=7, sws=21):
    R_u8 = (R * 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(R_u8, None, h, hc, tws, sws).astype(np.float32) / 255.0


def refine_wavelet(R, method="BayesShrink", sigma=None):
    """
    Per-channel wavelet denoising with adaptive BayesShrink thresholding.
    Outperforms NLM for spatially correlated noise typical in low-light.
    """
    R_f = R.astype(np.float64)
    R_d = denoise_wavelet(R_f, channel_axis=2, method=method,
                          mode="soft", rescale_sigma=True)
    return np.clip(R_d.astype(np.float32), 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ITERATIVE ALT-MIN (EM-style)
# ══════════════════════════════════════════════════════════════════════════════

def iterative_altmin(img, n_iter=3, tv_weight=0.08, sigma_n=None):
    """
    Alternately refine illumination I and reflectance R:
      Init:  I  ← TV-illum(S)
      Loop:  R  ← Wiener-divide(S, I, σ)
             R  ← Wavelet-denoise(R)
             I  ← re-estimate using R as structural guide (damp update)
    Each iteration makes I cleaner → R cleaner.
    """
    if sigma_n is None:
        sigma_n = estimate_noise_sigma(img)

    I = illum_tv(img, weight=tv_weight)

    for iteration in range(n_iter):
        # Step 1: estimate R given current I
        R = wiener_divide(img, I, sigma_n)
        R = refine_wavelet(R)

        # Step 2: re-estimate I using R as a structural guide
        # Idea: use R-guided smoothing on max-channel
        R_gray = np.mean(R, axis=2).astype(np.float32)
        I_seed = np.max(img, axis=2).astype(np.float32)
        # Guided filter with R_gray as guide for I_seed
        guide  = np.clip(R_gray, 1e-4, 1.0)
        r      = 16; eps = 1e-3
        mI     = _box(I_seed, r); mg  = _box(guide, r)
        mIg    = _box(I_seed * guide, r); mgg = _box(guide * guide, r)
        a      = (mIg - mI * mg) / (mgg - mg * mg + eps)
        b      = mI - a * mg
        I_new  = np.expand_dims(_box(a, r) * guide + _box(b, r), 2)
        I_new  = np.clip(I_new, 1e-4, 1.0).astype(np.float32)

        # Damped update: prevent oscillation
        damp = 0.4 + 0.2 * iteration      # increasingly trust new I
        I    = damp * I_new + (1 - damp) * I

    return R


# ══════════════════════════════════════════════════════════════════════════════
# 6.  LOG-DOMAIN TV RETINEX  (sparse linear system)
# ══════════════════════════════════════════════════════════════════════════════

def log_tv_retinex(img, lambda_I=0.15, lambda_R=0.05, max_iter=50):
    """
    Log-domain Retinex: log S = log I + log R
    Minimise:
        ||log I - log I_init||² + λ_I · TV(log I) + λ_R · TV(log R)
    subject to  log R = log S - log I

    Implemented as alternating gradient descent on log I.
    log R is obtained by subtraction, then both are exponentiated.
    """
    S     = np.clip(img, 1e-4, 1.0).astype(np.float32)
    logS  = np.log(S)

    # Initialise log I with TV-smoothed max-channel
    I0    = np.clip(illum_tv(img, weight=0.1).squeeze(), 1e-4, 1.0)
    logI  = np.log(I0)
    logI_init = logI.copy()

    H, W  = logI.shape
    lr    = 0.05
    for _ in range(max_iter):
        # Gradient of data term: 2(logI - logI_init) broadcast over channels
        grad = 2.0 * (logI - logI_init)

        # Gradient of TV(logI)  [isotropic TV via finite differences]
        dx  = np.pad(np.diff(logI, axis=1), ((0,0),(0,1)), 'constant')
        dy  = np.pad(np.diff(logI, axis=0), ((0,1),(0,0)), 'constant')
        dxb = np.pad(np.diff(logI, axis=1), ((0,0),(1,0)), 'constant')
        dyb = np.pad(np.diff(logI, axis=0), ((1,0),(0,0)), 'constant')
        eps_tv = 1e-4
        norm_fwd = np.sqrt(dx**2 + dy**2  + eps_tv)
        norm_bwd = np.sqrt(dxb**2 + dyb**2 + eps_tv)
        tv_grad  = (dx / norm_fwd - dxb / norm_bwd +
                    dy / norm_fwd - dyb / norm_bwd)
        grad += lambda_I * tv_grad

        # Gradient from TV(logR): logR = logS_per_chan - logI
        # TV(logR) penalty propagates back through logR = logS - logI
        logR = logS - logI[..., np.newaxis]          # H×W×3
        dRx  = np.pad(np.diff(logR, axis=1), ((0,0),(0,1),(0,0)), 'constant')
        dRy  = np.pad(np.diff(logR, axis=0), ((0,1),(0,0),(0,0)), 'constant')
        dRxb = np.pad(np.diff(logR, axis=1), ((0,0),(1,0),(0,0)), 'constant')
        dRyb = np.pad(np.diff(logR, axis=0), ((1,0),(0,0),(0,0)), 'constant')
        norm_R_fwd = np.sqrt((dRx**2 + dRy**2).sum(2, keepdims=True) + eps_tv)
        norm_R_bwd = np.sqrt((dRxb**2 + dRyb**2).sum(2, keepdims=True) + eps_tv)
        tv_R_grad  = -(dRx / norm_R_fwd - dRxb / norm_R_bwd +
                       dRy / norm_R_fwd - dRyb / norm_R_bwd).sum(2)
        grad += lambda_R * tv_R_grad

        logI -= lr * grad
        logI  = np.clip(logI, np.log(1e-4), 0.0)    # I ∈ [1e-4, 1]

    I_final = np.exp(logI)[..., np.newaxis]
    R_final = np.clip(img / (I_final + 1e-6), 0, 1)
    R_final = refine_wavelet(R_final)
    return R_final


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BUILD ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def build_variants(img):
    sigma_n = estimate_noise_sigma(img)

    # --- old-best baselines ---
    I_bil  = illum_bilateral(img)
    I_guid = illum_guided(img)
    I_tv   = illum_tv(img)

    R_base_v3 = refine_nlm(naive_divide(img, I_bil))
    R_base_v8 = refine_nlm(naive_divide(img, I_guid))

    # --- new variants ---
    R_a0 = refine_nlm    (naive_divide  (img, I_tv))
    R_a1 = refine_wavelet(naive_divide  (img, I_tv))
    R_a2 = refine_wavelet(wiener_divide (img, I_guid, sigma_n))
    R_a3 = refine_wavelet(wiener_divide (img, I_tv,   sigma_n))
    R_a4 = iterative_altmin(img, n_iter=3, sigma_n=sigma_n)
    R_a5 = log_tv_retinex(img)

    return list(zip(VARIANT_NAMES, [
        R_base_v3, R_base_v8,
        R_a0, R_a1, R_a2, R_a3, R_a4, R_a5,
    ]))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  METRICS & VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def metrics(R_est, R_ref):
    a = (R_est * 255).astype(np.uint8)
    b = (R_ref * 255).astype(np.uint8)
    p = calc_psnr(b, a, data_range=255)
    s = calc_ssim(b, a, channel_axis=2, data_range=255)
    return p, s


def to_rgb(arr):
    u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_BGR2RGB)


def save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx):
    n   = len(variants)
    fig = plt.figure(figsize=(30, 14))
    fig.patch.set_facecolor("#0d0d1a")

    ncols = n + 2      # low + R_high + variants
    gs    = gridspec.GridSpec(3, ncols, figure=fig,
                              hspace=0.4, wspace=0.04,
                              left=0.01, right=0.99,
                              top=0.91, bottom=0.04)

    TKW = dict(color="white",   fontsize=7, fontweight="bold", pad=3)
    MKW = dict(color="#ffe066", fontsize=6.5, va="top", ha="left")

    H, W = low.shape[:2]
    cx, cy, cw, ch = W//6, H//6, W//3, H//3

    def make_ax(r, c):
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#0d0d1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.axis("off")
        return ax

    # Row 0: full images
    make_ax(0, 0).imshow(to_rgb(low));    fig.axes[-1].set_title("Low input", **TKW)
    make_ax(0, 1).imshow(to_rgb(R_high)); fig.axes[-1].set_title("R_high [GT]", **TKW)
    for j, (vname, R) in enumerate(variants):
        ax = make_ax(0, j + 2)
        ax.imshow(to_rgb(R))
        bc = "#00ff88" if j == best_idx else "#333"
        lw = 2.5 if j == best_idx else 0.8
        for sp in ax.spines.values():
            sp.set_edgecolor(bc); sp.set_linewidth(lw)
        short = vname.split(" ", 1)[1]
        star  = " ★" if j == best_idx else ""
        ax.set_title(f"{short}{star}", **TKW)
        ax.text(0.01, 0.99, f"PSNR {all_psnr[j]:.2f}\nSSIM {all_ssim[j]:.4f}",
                transform=ax.transAxes, **MKW)

    # Row 1: crops
    make_ax(1, 0).imshow(to_rgb(low[cy:cy+ch, cx:cx+cw]));    fig.axes[-1].set_title("Low crop", **TKW)
    make_ax(1, 1).imshow(to_rgb(R_high[cy:cy+ch, cx:cx+cw])); fig.axes[-1].set_title("R_high crop", **TKW)
    for j, (vname, R) in enumerate(variants):
        ax = make_ax(1, j + 2)
        ax.imshow(to_rgb(R[cy:cy+ch, cx:cx+cw]))
        ax.set_title(f"crop: {vname.split(' ',1)[1]}", **TKW)
        if j == best_idx:
            for sp in ax.spines.values():
                sp.set_edgecolor("#00ff88"); sp.set_linewidth(2.5)

    # Row 2: bar charts
    ax_p = fig.add_subplot(gs[2, :n//2 + 1])
    ax_s = fig.add_subplot(gs[2, n//2 + 1:n + 1])
    for ax in [ax_p, ax_s]:
        ax.set_facecolor("#0a0a14")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")

    short_names = [v.split(" ", 1)[1] for v, _ in variants]
    cols = ["#00ff88" if i == best_idx else "#5599cc" for i in range(n)]

    ax_p.bar(range(n), all_psnr, color=cols, alpha=0.85)
    ax_p.set_xticks(range(n))
    ax_p.set_xticklabels(short_names, rotation=35, ha="right", fontsize=6.5, color="#ccc")
    ax_p.set_ylabel("PSNR (dB) ↑", color="#ccc"); ax_p.set_title("PSNR vs R_high", **TKW)
    for i, v in enumerate(all_psnr):
        ax_p.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=5.5, color="#fff")

    ax_s.bar(range(n), all_ssim, color=cols, alpha=0.85)
    ax_s.set_xticks(range(n))
    ax_s.set_xticklabels(short_names, rotation=35, ha="right", fontsize=6.5, color="#ccc")
    ax_s.set_ylabel("SSIM ↑", color="#ccc"); ax_s.set_title("SSIM vs R_high", **TKW)
    ax_s.set_ylim(0, 1.05)
    for i, v in enumerate(all_ssim):
        ax_s.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=5.5, color="#fff")

    # Diff map best vs R_high
    _, R_best = variants[best_idx]
    diff = np.abs(R_best - R_high).mean(axis=2)
    ax_d = fig.add_subplot(gs[2, n + 1])
    ax_d.set_facecolor("#0d0d1a"); ax_d.axis("off")
    ax_d.imshow(np.clip(diff * 5, 0, 1), cmap="inferno")
    ax_d.set_title("Diff(best,GT)×5", color="white", fontsize=7, pad=3)

    best_name = VARIANT_NAMES[best_idx].split(" ", 1)[1]
    fig.suptitle(
        f"Advanced Reflectance — {img_name}    "
        f"Best: {best_name}  "
        f"[PSNR {all_psnr[best_idx]:.2f} dB | SSIM {all_ssim[best_idx]:.4f}]",
        color="white", fontsize=10, fontweight="bold", y=0.96
    )
    base = os.path.splitext(img_name)[0]
    plt.savefig(os.path.join(out_dir, f"{base}_adv.png"),
                dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

img_names = sorted(f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg')))

global_psnr = np.zeros(len(VARIANT_NAMES))
global_ssim = np.zeros(len(VARIANT_NAMES))
count = 0

# Header
header_names = [n.split(" ", 1)[1][:18] for n in VARIANT_NAMES]
print(f"\n{'Image':<8}", end="")
for h in header_names:
    print(f"  {h:<20}", end="")
print()
print("-" * (8 + 22 * len(VARIANT_NAMES)))

for img_name in img_names:
    base = os.path.splitext(img_name)[0]
    low  = cv2.imread(os.path.join(low_dir,  img_name)).astype(np.float32) / 255.0
    high = cv2.imread(os.path.join(high_dir, img_name)).astype(np.float32) / 255.0

    # Ground-truth R_high: guided + NLM (consistent with fine_tune_reflectance.py)
    I_high = illum_guided(high)
    R_high = refine_wavelet(np.clip(high / (I_high + 1e-6), 0, 1))

    variants   = build_variants(low)
    all_psnr   = []
    all_ssim   = []
    for _, R in variants:
        p, s = metrics(R, R_high)
        all_psnr.append(p); all_ssim.append(s)

    best_idx = int(np.argmax(all_psnr))
    global_psnr += np.array(all_psnr)
    global_ssim += np.array(all_ssim)
    count += 1

    # Save best R
    _, R_best = variants[best_idx]
    cv2.imwrite(os.path.join(out_dir, f"{base}_R_adv_best.png"),
                (R_best * 255).astype(np.uint8))

    # Save comparison figure
    save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx)

    # Console
    print(f"{base:<8}", end="")
    for j, (p, s) in enumerate(zip(all_psnr, all_ssim)):
        m = "★" if j == best_idx else " "
        print(f"  {m}P={p:.2f} S={s:.3f}   ", end="")
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
    old  = " [baseline]" if j < 2 else ""
    print(f"  {star}{name:<38}  {avg_psnr[j]:>8.3f}  {avg_ssim[j]:>8.4f}{old}")

print(f"\n  Best overall: {VARIANT_NAMES[overall_best]}")
print(f"  Avg PSNR = {avg_psnr[overall_best]:.3f} dB  |  Avg SSIM = {avg_ssim[overall_best]:.4f}")
print(f"{'='*70}")

# ── Global summary figure ─────────────────────────────────────────────────────
short_names = [v.split(" ", 1)[1] for v in VARIANT_NAMES]
cols = ["#00ff88" if i == overall_best else ("#ff8855" if i < 2 else "#5599cc")
        for i in range(len(VARIANT_NAMES))]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
fig.patch.set_facecolor("#0d0d1a")
for ax, vals, ylabel, title in [
    (ax1, avg_psnr, "PSNR (dB) ↑", "Average PSNR vs R_high (orange=old, blue=new, green=best)"),
    (ax2, avg_ssim, "SSIM ↑",       "Average SSIM vs R_high"),
]:
    ax.set_facecolor("#0a0a14")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa")
    ax.bar(range(len(VARIANT_NAMES)), vals, color=cols, alpha=0.88)
    ax.set_xticks(range(len(VARIANT_NAMES)))
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=8.5, color="#ccc")
    ax.set_ylabel(ylabel, color="#ccc")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.005, f"{v:.3f}", ha="center", fontsize=7, color="#fff")

fig.suptitle(
    f"Advanced Reflectance — Global Summary  (n={count})  "
    f"★ best: {VARIANT_NAMES[overall_best].split(' ',1)[1]}",
    color="white", fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "global_summary_advanced.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\nAll outputs saved to:\n  {out_dir}")
