import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Paths ---
high_dir = "../../datasets/LOL_dataset/eval15/high"
enhanced_dir = "../../results/enhanced"
output_dir = "../../results/metrics"
os.makedirs(output_dir, exist_ok=True)

# --- Collect properties ---
names = []
brightness_gt, brightness_enh = [], []
contrast_gt, contrast_enh = [], []
colorfulness_gt, colorfulness_enh = [], []

def calc_brightness(img):
    return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def calc_contrast(img):
    return np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def calc_colorfulness(img):
    B, G, R = img[:,:,0].astype(float), img[:,:,1].astype(float), img[:,:,2].astype(float)
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    return np.sqrt(np.mean(rg)**2 + np.mean(yb)**2) + 0.3*np.sqrt(np.std(rg)**2 + np.std(yb)**2)

for img_name in sorted(os.listdir(high_dir)):
    base = os.path.splitext(img_name)[0]
    gt  = cv2.imread(os.path.join(high_dir, img_name))
    enh = cv2.imread(os.path.join(enhanced_dir, f"{base}_enhanced.png"))
    if gt is None or enh is None:
        continue
    if enh.shape != gt.shape:
        enh = cv2.resize(enh, (gt.shape[1], gt.shape[0]))

    names.append(base)
    brightness_gt.append(calc_brightness(gt))
    brightness_enh.append(calc_brightness(enh))
    contrast_gt.append(calc_contrast(gt))
    contrast_enh.append(calc_contrast(enh))
    colorfulness_gt.append(calc_colorfulness(gt))
    colorfulness_enh.append(calc_colorfulness(enh))

x = np.arange(len(names))
width = 0.35

# =============================================
# Plot 1: Brightness Comparison
# =============================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, brightness_gt, width, label="Ground Truth", color="#2ecc71", edgecolor="black", linewidth=0.5)
ax.bar(x + width/2, brightness_enh, width, label="Enhanced", color="#3498db", edgecolor="black", linewidth=0.5)
for i in x:
    ax.text(i - width/2, brightness_gt[i] + 1, f"{brightness_gt[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + width/2, brightness_enh[i] + 1, f"{brightness_enh[i]:.0f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Image", fontsize=12)
ax.set_ylabel("Mean Brightness", fontsize=12)
ax.set_title("Brightness: Ground Truth vs Enhanced", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/brightness_comparison.png", dpi=150)
plt.close()
print("Saved: brightness_comparison.png")

# =============================================
# Plot 2: Contrast Comparison
# =============================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, contrast_gt, width, label="Ground Truth", color="#2ecc71", edgecolor="black", linewidth=0.5)
ax.bar(x + width/2, contrast_enh, width, label="Enhanced", color="#3498db", edgecolor="black", linewidth=0.5)
for i in x:
    ax.text(i - width/2, contrast_gt[i] + 0.5, f"{contrast_gt[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + width/2, contrast_enh[i] + 0.5, f"{contrast_enh[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Image", fontsize=12)
ax.set_ylabel("Contrast (Std Dev)", fontsize=12)
ax.set_title("Contrast: Ground Truth vs Enhanced", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/contrast_comparison.png", dpi=150)
plt.close()
print("Saved: contrast_comparison.png")

# =============================================
# Plot 3: Colorfulness Comparison
# =============================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - width/2, colorfulness_gt, width, label="Ground Truth", color="#2ecc71", edgecolor="black", linewidth=0.5)
ax.bar(x + width/2, colorfulness_enh, width, label="Enhanced", color="#3498db", edgecolor="black", linewidth=0.5)
for i in x:
    ax.text(i - width/2, colorfulness_gt[i] + 0.5, f"{colorfulness_gt[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.text(i + width/2, colorfulness_enh[i] + 0.5, f"{colorfulness_enh[i]:.1f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xlabel("Image", fontsize=12)
ax.set_ylabel("Colorfulness Score", fontsize=12)
ax.set_title("Colorfulness: Ground Truth vs Enhanced", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=45, ha="right")
ax.legend(fontsize=12)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_dir}/colorfulness_comparison.png", dpi=150)
plt.close()
print("Saved: colorfulness_comparison.png")

# =============================================
# Plot 4: All metrics summary
# =============================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
labels = ["Ground Truth", "Enhanced"]
colors = ["#2ecc71", "#3498db"]

for ax, title, gt_vals, enh_vals in zip(
    axes,
    ["Avg Brightness", "Avg Contrast", "Avg Colorfulness"],
    [np.mean(brightness_gt), np.mean(contrast_gt), np.mean(colorfulness_gt)],
    [np.mean(brightness_enh), np.mean(contrast_enh), np.mean(colorfulness_enh)],
):
    vals = [gt_vals, enh_vals]
    bars = ax.bar(labels, vals, color=colors, width=0.5, edgecolor="black")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", fontsize=12, fontweight="bold")

fig.suptitle("Image Quality Summary: Ground Truth (Green) vs Enhanced (Blue)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{output_dir}/summary_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: summary_comparison.png")

print(f"\nAll plots saved to: {output_dir}/")
