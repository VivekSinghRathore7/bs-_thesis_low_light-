import cv2
import numpy as np
import os

# --- Paths ---
low_dir = "../../datasets/LOL_dataset/eval15/low"
high_dir = "../../datasets/LOL_dataset/eval15/high"
output_dir = "../../results/decomposition"
enhanced_dir = "../../results/enhanced"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(enhanced_dir, exist_ok=True)


def estimate_illumination(img, method="max_channel", blur_ksize=15):
    """
    Estimate the illumination map from an image.

    Methods:
      - 'max_channel': max(R, G, B) per pixel (RetinexNet-style)
      - 'grayscale': simple grayscale conversion
      - 'hsv_v': V channel from HSV color space

    Uses bilateral filter instead of Gaussian blur to preserve illumination
    boundaries, which reduces halo artifacts in the reflectance map.
    """
    if method == "max_channel":
        I = np.max(img, axis=2)
    elif method == "grayscale":
        I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif method == "hsv_v":
        I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bilateral filter: edge-preserving smoothing avoids halo artifacts in R
    I_u8 = (I * 255).astype(np.uint8)
    I_smooth = cv2.bilateralFilter(I_u8, d=blur_ksize, sigmaColor=75, sigmaSpace=75)
    I = np.expand_dims(I_smooth.astype(np.float32) / 255.0, axis=2)
    return I


def refine_reflectance(R, d=9, sigma_color=25, sigma_space=10):
    """
    Denoise the reflectance map using bilateral filtering.

    The R = S/I division amplifies noise in dark regions. Bilateral filtering
    smooths this noise while preserving texture and edge content in R.
    """
    R_u8 = (R * 255).astype(np.uint8)
    R_refined = cv2.bilateralFilter(R_u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return R_refined.astype(np.float32) / 255.0


def decompose_retinex(img, method="max_channel", refine=True):
    """Decompose image S into Reflectance R and Illumination I."""
    I = estimate_illumination(img, method=method)
    R = img / (I + 1e-6)
    R = np.clip(R, 0, 1)
    if refine:
        R = refine_reflectance(R)
    return R, I


# --- Main Loop ---
for img_name in sorted(os.listdir(low_dir)):
    base_name = os.path.splitext(img_name)[0]

    low = cv2.imread(os.path.join(low_dir, img_name)).astype(np.float32) / 255.0
    high = cv2.imread(os.path.join(high_dir, img_name)).astype(np.float32) / 255.0

    R_low, I_low = decompose_retinex(low)
    R_high, I_high = decompose_retinex(high)

    # Convert to uint8 for saving
    R_low_u8 = (R_low * 255).astype(np.uint8)
    I_low_u8 = (I_low.squeeze() * 255).astype(np.uint8)
    R_high_u8 = (R_high * 255).astype(np.uint8)
    I_high_u8 = (I_high.squeeze() * 255).astype(np.uint8)

    # --- Save original decomposition ---
    cv2.imwrite(f"{output_dir}/{base_name}_R_low.png", R_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_low.png", I_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_R_high.png", R_high_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_high.png", I_high_u8)

    # --- Save inverse images (255 - image) ---
    cv2.imwrite(f"{output_dir}/{base_name}_R_low_inv.png", 255 - R_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_low_inv.png", 255 - I_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_R_high_inv.png", 255 - R_high_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_high_inv.png", 255 - I_high_u8)

    # --- Save difference maps |high - low| ---
    I_diff = cv2.absdiff(I_high_u8, I_low_u8)
    R_diff = cv2.absdiff(R_high_u8, R_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_diff.png", I_diff)
    cv2.imwrite(f"{output_dir}/{base_name}_R_diff.png", R_diff)

    # --- Enhanced image: I_high x R_low ---
    enhanced = I_high * R_low
    enhanced = np.clip(enhanced, 0, 1)
    enhanced_u8 = (enhanced * 255).astype(np.uint8)
    cv2.imwrite(f"{enhanced_dir}/{base_name}_enhanced.png", enhanced_u8)

    # --- Decompose enhanced image ---
    R_enh, I_enh = decompose_retinex(enhanced)
    R_enh_u8 = (R_enh * 255).astype(np.uint8)
    I_enh_u8 = (I_enh.squeeze() * 255).astype(np.uint8)

    cv2.imwrite(f"{output_dir}/{base_name}_R_enh.png", R_enh_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_enh.png", I_enh_u8)

    # --- Inverse of enhanced decomposition ---
    cv2.imwrite(f"{output_dir}/{base_name}_R_enh_inv.png", 255 - R_enh_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_enh_inv.png", 255 - I_enh_u8)

    # --- Difference: |enhanced - low| ---
    I_diff_enh = cv2.absdiff(I_enh_u8, I_low_u8)
    R_diff_enh = cv2.absdiff(R_enh_u8, R_low_u8)
    cv2.imwrite(f"{output_dir}/{base_name}_I_diff_enh.png", I_diff_enh)
    cv2.imwrite(f"{output_dir}/{base_name}_R_diff_enh.png", R_diff_enh)

    print(f"  Processed: {img_name}")

print("\nRetinex decomposition + enhancement completed.")
