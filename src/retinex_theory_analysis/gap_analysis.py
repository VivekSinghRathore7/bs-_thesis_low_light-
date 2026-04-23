"""
gap_analysis.py
===============
Quantify the gap between fine-tuned reflectance images and ground-truth
reflectance (R_high) using PSNR, SSIM, MAE, RMSE and pixel-error histogram.
Compares: R_low (raw), V3 old-best, A5 new-best.
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim

BASE  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
decom = os.path.join(BASE, "results/decomposition")
adv   = os.path.join(BASE, "results/reflectance_advanced")
finet = os.path.join(BASE, "results/reflectance_finetuned")
out   = os.path.join(BASE, "results/gap_analysis")
os.makedirs(out, exist_ok=True)


def load(p):
    img = cv2.imread(p)
    assert img is not None, f"Missing: {p}"
    return img.astype(np.float32) / 255.0


def metrics(est, ref):
    u8  = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    p   = calc_psnr(u8(ref), u8(est), data_range=255)
    s   = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    mae = float(np.mean(np.abs(est - ref)))
    rmse= float(np.sqrt(np.mean((est - ref) ** 2)))
    return p, s, mae, rmse


names = sorted(f for f in os.listdir(decom) if f.endswith("_R_low.png"))

acc = {"rl": [0,0,0,0], "ob": [0,0,0,0], "nb": [0,0,0,0]}
n   = 0

rows = []

# ── Header ──────────────────────────────────────────────────────────────────
H1 = f"{'':10}  {'--- R_low (raw) ---':^36}  {'--- V3 Old Best ---':^36}  {'--- A5 New Best ---':^36}"
H2 = f"{'Image':<10}  {'PSNR':>6} {'SSIM':>6} {'MAE':>6} {'RMSE':>6}   {'PSNR':>6} {'SSIM':>6} {'MAE':>6} {'RMSE':>6}   {'PSNR':>6} {'SSIM':>6} {'MAE':>6} {'RMSE':>6}"
SEP = "-" * 112
print(H1); print(H2); print(SEP)

for fn in names:
    base = fn.replace("_R_low.png", "")
    rl   = load(os.path.join(decom,  f"{base}_R_low.png"))
    rh   = load(os.path.join(decom,  f"{base}_R_high.png"))
    rob  = load(os.path.join(finet,  f"{base}_R_best.png"))
    rnb  = load(os.path.join(adv,    f"{base}_R_adv_best.png"))

    ml = metrics(rl,  rh)
    mb = metrics(rob, rh)
    mn = metrics(rnb, rh)

    rows.append((base, ml, mb, mn, rl, rh, rob, rnb))

    print(f"{base:<10}  "
          f"{ml[0]:>6.2f} {ml[1]:>6.3f} {ml[2]:>6.4f} {ml[3]:>6.4f}   "
          f"{mb[0]:>6.2f} {mb[1]:>6.3f} {mb[2]:>6.4f} {mb[3]:>6.4f}   "
          f"{mn[0]:>6.2f} {mn[1]:>6.3f} {mn[2]:>6.4f} {mn[3]:>6.4f}")

    for i, v in enumerate(ml): acc["rl"][i] += v
    for i, v in enumerate(mb): acc["ob"][i] += v
    for i, v in enumerate(mn): acc["nb"][i] += v
    n += 1

print(SEP)
avgs = {k: [v/n for v in vals] for k, vals in acc.items()}

for key, label in [("rl","R_low (raw)"), ("ob","V3 Old Best"), ("nb","A5 New Best")]:
    a = avgs[key]
    print(f"AVG  {label:<12}  {a[0]:>6.3f} {a[1]:>6.4f} {a[2]:>6.4f} {a[3]:>6.4f}")

print()
print("=" * 70)
print("  GAP TO PERFECT (R_high = ground truth)")
print("=" * 70)
print(f"  {'Method':<14}  {'SSIM gap':>10}  {'MAE (%)':>10}  {'RMSE':>8}  {'PSNR (dB)':>10}")
print(f"  {'-'*58}")
for key, label in [("rl","R_low (raw)"), ("ob","V3 Old Best"), ("nb","A5 New Best")]:
    a = avgs[key]
    print(f"  {label:<14}  {1.0-a[1]:>10.4f}  {a[2]*100:>9.2f}%  {a[3]:>8.4f}  {a[0]:>10.3f}")
print(f"\n  Perfect match: SSIM gap=0.0000, MAE=0.00%, RMSE=0.0000, PSNR=inf")
print("=" * 70)

# ── Improvement summary ──────────────────────────────────────────────────────
al, ab, an = avgs["rl"], avgs["ob"], avgs["nb"]
print(f"\n  Improvement  A5 vs raw R_low:  PSNR {an[0]-al[0]:+.3f} dB  |  SSIM {an[1]-al[1]:+.4f}  |  MAE {(an[2]-al[2])*100:+.2f}%")
print(f"  Improvement  A5 vs V3 old:     PSNR {an[0]-ab[0]:+.3f} dB  |  SSIM {an[1]-ab[1]:+.4f}  |  MAE {(an[2]-ab[2])*100:+.2f}%")

# ── Per-image bar chart ──────────────────────────────────────────────────────
bases    = [r[0] for r in rows]
psnr_rl  = [r[1][0] for r in rows]
psnr_ob  = [r[2][0] for r in rows]
psnr_nb  = [r[3][0] for r in rows]
ssim_rl  = [r[1][1] for r in rows]
ssim_ob  = [r[2][1] for r in rows]
ssim_nb  = [r[3][1] for r in rows]
mae_rl   = [r[1][2]*100 for r in rows]
mae_ob   = [r[2][2]*100 for r in rows]
mae_nb   = [r[3][2]*100 for r in rows]

x     = np.arange(len(bases))
width = 0.26

fig, axes = plt.subplots(3, 1, figsize=(18, 14))
fig.patch.set_facecolor("#0d0d1a")

configs = [
    (axes[0], psnr_rl, psnr_ob, psnr_nb, "PSNR (dB)  -- higher = closer to GT", "PSNR (dB)"),
    (axes[1], ssim_rl, ssim_ob, ssim_nb, "SSIM  -- higher = closer to GT",       "SSIM"),
    (axes[2], mae_rl,  mae_ob,  mae_nb,  "MAE (%)  -- lower = closer to GT",     "MAE (%)"),
]

for ax, vl, vb, vn, title, ylabel in configs:
    ax.set_facecolor("#0a0a14")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa", labelsize=9)
    b1 = ax.bar(x - width, vl, width, label="R_low raw",    color="#888888", alpha=0.85)
    b2 = ax.bar(x,          vb, width, label="V3 old best", color="#ff8855", alpha=0.85)
    b3 = ax.bar(x + width,  vn, width, label="A5 new best", color="#00ff88", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bases, rotation=30, ha="right", color="#ccc")
    ax.set_ylabel(ylabel, color="#ccc")
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white",
              edgecolor="#555", loc="upper right")
    # Value labels on new-best bars only
    for rect, v in zip(b3, vn):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01*max(vn),
                f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#ddd")

fig.suptitle(
    "Reflectance Gap Analysis: R_low / V3 Old-Best / A5 New-Best  vs  R_high (Ground Truth)\n"
    f"Avg PSNR: raw={al[0]:.2f}  old={ab[0]:.2f}  new={an[0]:.2f} dB  |  "
    f"Avg SSIM: raw={al[1]:.3f}  old={ab[1]:.3f}  new={an[1]:.3f}  |  "
    f"Avg MAE: raw={al[2]*100:.2f}%  old={ab[2]*100:.2f}%  new={an[2]*100:.2f}%",
    color="white", fontsize=10, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig(os.path.join(out, "gap_analysis_bar.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nBar chart saved: {out}/gap_analysis_bar.png")

# ── Error distribution histogram ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0d0d1a")
titles_h = ["R_low (raw) vs R_high", "V3 Old Best vs R_high", "A5 New Best vs R_high"]
colors_h = ["#888888", "#ff8855", "#00ff88"]

for ax, (_, ml, mb, mn, rl, rh, rob, rnb), title_h, color_h, est in zip(
        axes,
        [(None, None, None, None, rows[0][4], rows[0][5], rows[0][6], rows[0][7])],  # placeholder
        titles_h, colors_h,
        ["rl", "ob", "nb"]):
    pass  # replaced below

# Aggregate pixel errors across all images
errs = {"rl": [], "ob": [], "nb": []}
for _, ml, mb, mn, rl, rh, rob, rnb in rows:
    errs["rl"].append(np.abs(rl  - rh).ravel())
    errs["ob"].append(np.abs(rob - rh).ravel())
    errs["nb"].append(np.abs(rnb - rh).ravel())

for key in errs:
    errs[key] = np.concatenate(errs[key])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor("#0d0d1a")

for ax, (key, label, color) in zip(axes, [
        ("rl", f"R_low (raw)\nMAE={avgs['rl'][2]*100:.2f}%  PSNR={avgs['rl'][0]:.2f}dB  SSIM={avgs['rl'][1]:.3f}", "#888888"),
        ("ob", f"V3 Old Best\nMAE={avgs['ob'][2]*100:.2f}%  PSNR={avgs['ob'][0]:.2f}dB  SSIM={avgs['ob'][1]:.3f}", "#ff8855"),
        ("nb", f"A5 New Best\nMAE={avgs['nb'][2]*100:.2f}%  PSNR={avgs['nb'][0]:.2f}dB  SSIM={avgs['nb'][1]:.3f}", "#00ff88"),
]):
    ax.set_facecolor("#0a0a14")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa")
    ax.hist(errs[key], bins=100, range=(0, 0.5), color=color, alpha=0.85, density=True)
    ax.axvline(avgs[key][2], color="white", linewidth=1.5, linestyle="--",
               label=f"MAE = {avgs[key][2]*100:.2f}%")
    ax.set_xlabel("Pixel absolute error (0-1 scale)", color="#ccc")
    ax.set_ylabel("Density", color="#ccc")
    ax.set_title(label, color="white", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")

    # Shade the gap region (error > 0.1 = 10% off)
    ax.axvspan(0.1, 0.5, alpha=0.1, color="red", label=">10% error")
    pct_bad = float(np.mean(errs[key] > 0.1)) * 100
    ax.text(0.15, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1,
            f"{pct_bad:.1f}% pixels\n>10% error",
            color="#ff6666", fontsize=8, ha="left")

fig.suptitle("Pixel Error Distribution vs Ground-Truth Reflectance (all 15 images pooled)",
             color="white", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(out, "gap_analysis_histogram.png"),
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Histogram saved: {out}/gap_analysis_histogram.png")
print(f"\nAll outputs: {out}/")
