"""
Retinex and classical processing logic.
"""

import cv2
import numpy as np


def estimate_illumination(img_float, method="max_channel", ksize=15):
    """Estimate illumination from float32 BGR image [0,1]."""
    if method == "max_channel":
        I = np.max(img_float, axis=2)
    elif method == "grayscale":
        I = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown method: {method}")
    I_u8 = (I * 255).astype(np.uint8)
    I_smooth = cv2.bilateralFilter(I_u8, d=ksize, sigmaColor=75, sigmaSpace=75)
    return I_smooth.astype(np.float32) / 255.0


def decompose_retinex(img_float):
    """Decompose float32 BGR image [0,1] into reflectance R and illumination I."""
    I = estimate_illumination(img_float)
    I_3ch = np.expand_dims(I, axis=2)
    R = img_float / (I_3ch + 1e-6)
    R = np.clip(R, 0, 1)
    # Denoise reflectance
    R_u8 = (R * 255).astype(np.uint8)
    R_u8 = cv2.bilateralFilter(R_u8, d=9, sigmaColor=25, sigmaSpace=10)
    R = R_u8.astype(np.float32) / 255.0
    return R, I


# --- Classical illumination enhancement methods ---

def enhance_clahe(I_u8):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(I_u8)


def enhance_agc(I_u8):
    I_norm = I_u8.astype(np.float32) / 255.0
    mean_val = np.mean(I_norm)
    gamma = np.log(0.5) / np.log(mean_val + 1e-6)
    out = np.power(I_norm, gamma)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def enhance_glagc(I_u8):
    I_norm = I_u8.astype(np.float32) / 255.0
    mean_g = np.mean(I_norm)
    gamma_g = np.log(0.5) / np.log(mean_g + 1e-6)
    local_mean = cv2.GaussianBlur(I_norm, (15, 15), 0)
    gamma_l = np.log(0.5) / np.log(np.clip(local_mean, 1e-6, 1.0))
    gamma = 0.5 * gamma_g + 0.5 * gamma_l
    out = np.power(I_norm, gamma)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def enhance_log(I_u8):
    c = 255.0 / np.log(256.0)
    out = c * np.log(1.0 + I_u8.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def get_all_enhanced_illuminations(I_float):
    """Return dict of enhanced illumination maps from float [0,1] illumination."""
    I_u8 = (I_float * 255).astype(np.uint8)
    return {
        'clahe': enhance_clahe(I_u8).astype(np.float32) / 255.0,
        'agc': enhance_agc(I_u8).astype(np.float32) / 255.0,
        'glagc': enhance_glagc(I_u8).astype(np.float32) / 255.0,
        'log': enhance_log(I_u8).astype(np.float32) / 255.0,
    }
