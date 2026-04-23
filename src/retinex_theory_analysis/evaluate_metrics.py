import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- Paths ---
low_dir = "../../datasets/LOL_dataset/eval15/low"
high_dir = "../../datasets/LOL_dataset/eval15/high"
enhanced_dir = "../../results/enhanced"
output_dir = "../../results/metrics"
os.makedirs(output_dir, exist_ok=True)

# --- Collect metrics ---
names = []
psnr_low, psnr_enh = [], []
ssim_low, ssim_enh = [], []

for img_name in sorted(os.listdir(low_dir)):
    base = os.path.splitext(img_name)[0]

    gt  = cv2.imread(os.path.join(high_dir, img_name))
    low = cv2.imread(os.path.join(low_dir, img_name))
    enh = cv2.imread(os.path.join(enhanced_dir, f"{base}_enhanced.png"))

    if gt is None or low is None or enh is None:
        print(f"  Skipping {img_name} (file not found)")
        continue

    # Resize enhanced to match ground truth if needed
    if enh.shape != gt.shape:
        enh = cv2.resize(enh, (gt.shape[1], gt.shape[0]))

    names.append(base)

    # PSNR (higher = better)
    psnr_low.append(psnr(gt, low))
    psnr_enh.append(psnr(gt, enh))

    # SSIM (higher = better, closer to 1 = more similar)
    ssim_low.append(ssim(gt, low, channel_axis=2))
    ssim_enh.append(ssim(gt, enh, channel_axis=2))

    print(f"  {base}: PSNR(low)={psnr_low[-1]:.2f} PSNR(enh)={psnr_enh[-1]:.2f}  |  SSIM(low)={ssim_low[-1]:.4f} SSIM(enh)={ssim_enh[-1]:.4f}")

# Convert to arrays
psnr_low = np.array(psnr_low)
psnr_enh = np.array(psnr_enh)
ssim_low = np.array(ssim_low)
ssim_enh = np.array(ssim_enh)

# =========================================================
# Plot 1: PSNR Comparison (grouped bar chart)
# =========================================================
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(names))
width = 0.35

bars1 = ax.bar(x - width/2, psnr_low, width, label="Low-Light vs GT", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, psnr_enh, width, label="Enhanced vs GT", color="#2ecc71", alpha=0.85)

ax.set_xlabel("Image", fontsize=12)
ax.set_ylabel("PSNR (dB) ↑", fontsize=12)
ax.set_title("PSNR Comparison: Low-Light vs Enhanced (relative to Ground Truth)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{output_dir}/psnr_comparison.png", dpi=150)
plt.close()

# =========================================================
# Plot 2: SSIM Comparison (grouped bar chart)
# =========================================================
fig, ax = plt.subplots(figsize=(14, 6))

bars1 = ax.bar(x - width/2, ssim_low, width, label="Low-Light vs GT", color="#e74c3c", alpha=0.85)
bars2 = ax.bar(x + width/2, ssim_enh, width, label="Enhanced vs GT", color="#2ecc71", alpha=0.85)

ax.set_xlabel("Image", fontsize=12)
ax.set_ylabel("SSIM ↑", fontsize=12)
ax.set_title("SSIM Comparison: Low-Light vs Enhanced (relative to Ground Truth)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.legend(fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig(f"{output_dir}/ssim_comparison.png", dpi=150)
plt.close()

# =========================================================
# Plot 3: Average metrics summary
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Average PSNR
metrics = ["Low-Light", "Enhanced"]
avg_psnr = [psnr_low.mean(), psnr_enh.mean()]
colors = ["#e74c3c", "#2ecc71"]
bars = axes[0].bar(metrics, avg_psnr, color=colors, width=0.5, alpha=0.85, edgecolor="black")
axes[0].set_title("Average PSNR (dB) ↑", fontsize=13, fontweight="bold")
axes[0].set_ylabel("PSNR (dB)")
axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, avg_psnr):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

# Average SSIM
avg_ssim = [ssim_low.mean(), ssim_enh.mean()]
bars = axes[1].bar(metrics, avg_ssim, color=colors, width=0.5, alpha=0.85, edgecolor="black")
axes[1].set_title("Average SSIM ↑", fontsize=13, fontweight="bold")
axes[1].set_ylabel("SSIM")
axes[1].set_ylim(0, 1.05)
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, avg_ssim):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

fig.suptitle("Quality Metrics: Enhanced vs Low-Light (relative to Ground Truth)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/average_metrics.png", dpi=150, bbox_inches="tight")
plt.close()

# Print summary
print(f"\n{'='*50}")
print(f"  AVERAGE METRICS SUMMARY")
print(f"{'='*50}")
print(f"  PSNR  | Low-Light: {psnr_low.mean():.2f} dB  | Enhanced: {psnr_enh.mean():.2f} dB")
print(f"  SSIM  | Low-Light: {ssim_low.mean():.4f}     | Enhanced: {ssim_enh.mean():.4f}")
print(f"{'='*50}")
print(f"\n  Plots saved to: {output_dir}/")
