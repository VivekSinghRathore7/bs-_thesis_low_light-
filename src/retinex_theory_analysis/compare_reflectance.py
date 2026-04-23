import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

low_dir        = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
decom_dir      = os.path.join(BASE_DIR, "results/decomposition")   # new (bilateral) results
compare_dir    = os.path.join(BASE_DIR, "results/reflectance_comparison")
os.makedirs(compare_dir, exist_ok=True)


# ------------------------------------------------------------------
# Old pipeline (Gaussian blur, no reflectance refinement)
# ------------------------------------------------------------------
def estimate_illumination_old(img, blur_ksize=15):
    I = np.max(img, axis=2)
    I = cv2.GaussianBlur(I, (blur_ksize, blur_ksize), 0)
    return np.expand_dims(I, axis=2)

def decompose_old(img):
    I = estimate_illumination_old(img)
    R = np.clip(img / (I + 1e-6), 0, 1)
    return R


# ------------------------------------------------------------------
# New pipeline (bilateral illumination + bilateral reflectance refine)
# ------------------------------------------------------------------
def estimate_illumination_new(img, blur_ksize=15):
    I = np.max(img, axis=2)
    I_u8 = (I * 255).astype(np.uint8)
    I_smooth = cv2.bilateralFilter(I_u8, d=blur_ksize, sigmaColor=75, sigmaSpace=75)
    return np.expand_dims(I_smooth.astype(np.float32) / 255.0, axis=2)

def refine_reflectance(R, d=9, sigma_color=25, sigma_space=10):
    R_u8 = (R * 255).astype(np.uint8)
    R_refined = cv2.bilateralFilter(R_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return R_refined.astype(np.float32) / 255.0

def decompose_new(img):
    I = estimate_illumination_new(img)
    R = np.clip(img / (I + 1e-6), 0, 1)
    R = refine_reflectance(R)
    return R


# ------------------------------------------------------------------
# Main comparison loop
# ------------------------------------------------------------------
img_names = sorted(f for f in os.listdir(low_dir) if f.endswith(('.png', '.jpg')))

for img_name in img_names:
    base = os.path.splitext(img_name)[0]

    low = cv2.imread(os.path.join(low_dir, img_name))
    low_rgb = cv2.cvtColor(low, cv2.COLOR_BGR2RGB)
    low_f   = low.astype(np.float32) / 255.0

    R_old = decompose_old(low_f)
    R_new = decompose_new(low_f)

    # Difference map (amplified for visibility)
    diff = np.abs(R_new - R_old)
    diff_amplified = np.clip(diff * 5, 0, 1)

    # Convert to display format
    def to_rgb(R):
        return cv2.cvtColor((R * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    R_old_disp = to_rgb(R_old)
    R_new_disp = to_rgb(R_new)
    diff_disp  = (diff_amplified * 255).astype(np.uint8)

    # ------ Figure ------
    fig = plt.figure(figsize=(20, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)

    titles = ["Input (low-light)", "R_low  [Old: Gaussian]",
              "R_low  [New: Bilateral]", "Difference × 5"]
    imgs   = [low_rgb, R_old_disp, R_new_disp, diff_disp]

    for col, (title, im) in enumerate(zip(titles, imgs)):
        ax = fig.add_subplot(gs[col])
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    fig.suptitle(f"Reflectance Comparison — {img_name}", fontsize=13, y=1.01)
    out_path = os.path.join(compare_dir, f"{base}_R_compare.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

print(f"\nAll comparisons saved to: {compare_dir}")
