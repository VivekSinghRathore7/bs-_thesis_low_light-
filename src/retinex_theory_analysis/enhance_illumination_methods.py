import cv2
import numpy as np
import os
import sys

# Get BASE_DIR robustly so it works from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths relative to base dir
low_dir = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
output_dir = os.path.join(BASE_DIR, "results/illumination_enhancement")
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------
# Helper function: Estimate base Illumination (Il0)
# -----------------------------------------------------
def estimate_illumination(img, method="max_channel", blur_ksize=15):
    if method == "max_channel":
        I = np.max(img, axis=2)
    elif method == "grayscale":
        I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unknown method")
    I = cv2.GaussianBlur(I, (blur_ksize, blur_ksize), 0)
    return I

# -----------------------------------------------------
# Category 1: Histogram Equalization
# -----------------------------------------------------
def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_bbhe(img):
    mean_val = int(np.mean(img))
    lower, upper = img[img <= mean_val], img[img > mean_val]
    
    hist_l, _ = np.histogram(lower, bins=np.arange(0, mean_val + 2))
    hist_u, _ = np.histogram(upper, bins=np.arange(mean_val + 1, 257))
    
    pdf_l = hist_l / (len(lower) + 1e-6)
    pdf_u = hist_u / (len(upper) + 1e-6)
    
    cdf_l, cdf_u = np.cumsum(pdf_l), np.cumsum(pdf_u)
    
    map_l = np.round(cdf_l * mean_val).astype(np.uint8)
    map_u = np.round((cdf_u * (255 - (mean_val + 1))) + (mean_val + 1)).astype(np.uint8)
    
    full_map = np.zeros(256, dtype=np.uint8)
    if len(map_l) > 0: full_map[:mean_val+1] = map_l
    if len(map_u) > 0: full_map[mean_val+1:] = map_u
        
    return cv2.LUT(img, full_map)

def apply_rmshe(img, depth=2):
    def get_segments(data, d):
        if d == 0 or len(data) == 0:
            return [(data.min() if len(data) > 0 else 0, data.max() if len(data) > 0 else 255, data)]
        mean_val = np.mean(data)
        lower, upper = data[data <= mean_val], data[data > mean_val]
        return get_segments(lower, d-1) + get_segments(upper, d-1)

    full_map = np.arange(256, dtype=np.uint8)
    segments = get_segments(img.flatten(), depth)
    
    for (min_val, max_val, data) in segments:
        if len(data) == 0: continue
        hist, _ = np.histogram(data, bins=np.arange(int(min_val), int(max_val) + 2))
        pdf = hist / len(data)
        cdf = np.cumsum(pdf)
        mapped = np.round(cdf * (max_val - min_val) + min_val).astype(np.uint8)
        
        idx = 0
        for val in range(int(min_val), int(max_val) + 1):
            if val < 256 and idx < len(mapped):
                full_map[val] = mapped[idx]
                idx += 1
                
    return cv2.LUT(img, full_map)

# -----------------------------------------------------
# Category 2: Gamma Correction
# -----------------------------------------------------
def apply_agc(img):
    img_norm = img.astype(np.float32) / 255.0
    mean_val = np.mean(img_norm)
    gamma = np.log(0.5) / np.log(mean_val + 1e-6)
    out = np.power(img_norm, gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

def apply_agcwhd(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0,256])
    pdf = hist / (img.shape[0] * img.shape[1])
    
    pdf_max, pdf_min = np.max(pdf), np.min(pdf)
    alpha = 0.5
    pdf_w = pdf_max * (((pdf - pdf_min) / (pdf_max - pdf_min + 1e-6)) ** alpha)
    cdf_w = np.cumsum(pdf_w) / (np.sum(pdf_w) + 1e-6)
    
    img_norm = img.astype(np.float32) / 255.0
    out_img = np.zeros_like(img_norm)
    
    for i in range(256):
        mask = (img == i)
        gamma = 1 - cdf_w[i]
        out_img[mask] = 255.0 * np.power(img_norm[mask], np.maximum(gamma, 0.1))
        
    return np.clip(out_img, 0, 255).astype(np.uint8)

def apply_glagc(img):
    img_norm = img.astype(np.float32) / 255.0
    mean_global = np.mean(img_norm)
    gamma_g = np.log(0.5) / np.log(mean_global + 1e-6)
    
    local_mean = cv2.GaussianBlur(img_norm, (15, 15), 0)
    gamma_l = np.log(0.5) / np.log(np.clip(local_mean, 1e-6, 1.0))
    
    gamma_combined = 0.5 * gamma_g + 0.5 * gamma_l
    out = np.power(img_norm, gamma_combined)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)

# -----------------------------------------------------
# Category 3: Contrast Stretching
# -----------------------------------------------------
def apply_linear_stretch(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def apply_pwlc(img):
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)
    mean_val = np.mean(img_float)
    
    out = np.zeros_like(img_float)
    mask_low = img_float <= mean_val
    if mean_val > min_val:
        out[mask_low] = 127 * (img_float[mask_low] - min_val) / (mean_val - min_val)
        
    mask_high = img_float > mean_val
    if max_val > mean_val:
        out[mask_high] = 127 + 128 * (img_float[mask_high] - mean_val) / (max_val - mean_val)
        
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_log_transform(img):
    c = 255.0 / np.log(256.0)
    log_image = c * np.log(1.0 + img.astype(np.float32))
    return np.clip(log_image, 0, 255).astype(np.uint8)

# -----------------------------------------------------
# Main Loop
# -----------------------------------------------------
def main():
    print("Starting Illumination Enhancement...")
    
    if not os.path.exists(low_dir):
        print(f"Error: Low directory not found at {low_dir}")
        sys.exit(1)

    for img_name in sorted(os.listdir(low_dir)):
        if not img_name.endswith(('.png', '.jpg')): 
            continue
            
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(low_dir, img_name)
        
        # Read low light image
        low = cv2.imread(img_path)
        if low is None:
            continue
            
        low_norm = low.astype(np.float32) / 255.0
        
        # Il0: Original formulation I_low
        I_low = estimate_illumination(low_norm, method="max_channel")
        Il0 = (I_low * 255).astype(np.uint8)
        
        # Generate Il1 to Il9
        Il1 = apply_clahe(Il0)
        Il2 = apply_bbhe(Il0)
        Il3 = apply_rmshe(Il0, depth=2)
        
        Il4 = apply_agc(Il0)
        Il5 = apply_agcwhd(Il0)
        Il6 = apply_glagc(Il0)
        
        Il7 = apply_linear_stretch(Il0)
        Il8 = apply_pwlc(Il0)
        Il9 = apply_log_transform(Il0)
        
        # Save all outputs
        cv2.imwrite(f"{output_dir}/{base_name}_Il0.png", Il0)
        cv2.imwrite(f"{output_dir}/{base_name}_Il1_CLAHE.png", Il1)
        cv2.imwrite(f"{output_dir}/{base_name}_Il2_BBHE.png", Il2)
        cv2.imwrite(f"{output_dir}/{base_name}_Il3_RMSHE.png", Il3)
        cv2.imwrite(f"{output_dir}/{base_name}_Il4_AGC.png", Il4)
        cv2.imwrite(f"{output_dir}/{base_name}_Il5_AGCWHD.png", Il5)
        cv2.imwrite(f"{output_dir}/{base_name}_Il6_GLAGC.png", Il6)
        cv2.imwrite(f"{output_dir}/{base_name}_Il7_MinMax.png", Il7)
        cv2.imwrite(f"{output_dir}/{base_name}_Il8_PWLC.png", Il8)
        cv2.imwrite(f"{output_dir}/{base_name}_Il9_Log.png", Il9)
        
        print(f"Processed: {img_name}")

    print(f"\nCompleted! Enhanced illumination maps saved in: {output_dir}")

if __name__ == "__main__":
    main()
