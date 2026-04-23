"""
RetinexPix2Pix — dataset.py

Builds 8-channel Retinex-guided input:
  [I_low(3) | R_tv(3) | L_tv(1) | L_hint(1)]  →  GT(3)

TV decomposition results are cached to disk (first run is slow, subsequent runs fast).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.restoration import denoise_tv_chambolle


CACHE_DIR_NAME = ".retinex_cache"


# ─────────────────────────────────────────────────
# Retinex decomposition (TV-based, fast)
# ─────────────────────────────────────────────────

def tv_decompose(img_f32, weight=0.08):
    """
    img_f32: HxWx3 float32 [0,1] BGR
    returns: R (3ch), L (1ch), L_hint (1ch)
    """
    L = denoise_tv_chambolle(
        np.max(img_f32, axis=2).astype(np.float32),
        weight=weight, channel_axis=None
    ).astype(np.float32)
    L = np.clip(L, 1e-4, 1.0)
    R = np.clip(img_f32 / (L[:,:,None] + 1e-6), 0, 1).astype(np.float32)

    # Illumination boost hint: how much does L need to increase to reach 0.6?
    L_target = 0.6
    L_hint = np.clip((L_target - L) / (L_target + 1e-6), 0, 1).astype(np.float32)
    return R, L, L_hint


def _cache_path(cache_root, fname, suffix):
    return os.path.join(cache_root, fname.replace(os.sep, '_') + suffix)


# ─────────────────────────────────────────────────
# Augmentation helpers
# ─────────────────────────────────────────────────

def random_augment(inp, gt):
    """Random flip + rotation on np arrays HxWxC."""
    k = np.random.randint(0, 4)
    inp = np.rot90(inp, k).copy()
    gt  = np.rot90(gt,  k).copy()
    if np.random.rand() > 0.5:
        inp = inp[:, ::-1, :].copy()
        gt  = gt[:, ::-1, :].copy()
    if np.random.rand() > 0.5:
        inp = inp[::-1, :, :].copy()
        gt  = gt[::-1, :, :].copy()
    return inp, gt


def random_color_jitter(img_f32, hue=0.02, sat=0.15, val=0.15):
    """Light HSV jitter for colour augmentation."""
    img_u8 = np.clip(img_f32 * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = np.clip(hsv[:,:,0] + np.random.uniform(-hue*180, hue*180), 0, 179)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * np.random.uniform(1-sat, 1+sat), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * np.random.uniform(1-val, 1+val), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0


# ─────────────────────────────────────────────────
# Main dataset
# ─────────────────────────────────────────────────

class RetinexPix2PixDataset(Dataset):
    """
    Returns:
        inp  : (8, H, W)  [I_low(3) | R_tv(3) | L_tv(1) | L_hint(1)]
        gt   : (3, H, W)
        R_tv : (3, H, W)  for RetinexConsistencyLoss
        L_tv : (1, H, W)  for IllumMonotonicityLoss
    """

    def __init__(self, root_dir, split="our485", img_size=256,
                 augment=True, cache=True):
        self.low_dir  = os.path.join(root_dir, split, "low")
        self.high_dir = os.path.join(root_dir, split, "high")
        self.img_size = img_size
        self.augment  = augment
        self.cache    = cache
        self.cache_dir = os.path.join(root_dir, CACHE_DIR_NAME, split)
        if cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.files = sorted(
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def _get_decomp(self, fname, img_f32):
        key = os.path.splitext(fname)[0]
        cp_R = _cache_path(self.cache_dir, key, '_R.npy')
        cp_L = _cache_path(self.cache_dir, key, '_L.npy')
        cp_H = _cache_path(self.cache_dir, key, '_Lhint.npy')
        if self.cache and os.path.exists(cp_R):
            R = np.load(cp_R); L = np.load(cp_L); Lh = np.load(cp_H)
        else:
            R, L, Lh = tv_decompose(img_f32)
            if self.cache:
                np.save(cp_R, R); np.save(cp_L, L); np.save(cp_H, Lh)
        return R, L, Lh

    def __getitem__(self, idx):
        fname = self.files[idx]
        low  = self._load(os.path.join(self.low_dir,  fname))
        high = self._load(os.path.join(self.high_dir, fname))

        R_tv, L_tv, L_hint = self._get_decomp(fname, low)

        # Build 8-channel input: HxWx8
        inp = np.concatenate([
            low,                          # 3ch BGR
            R_tv,                         # 3ch
            L_tv[:,:,None],               # 1ch
            L_hint[:,:,None],             # 1ch
        ], axis=2)                        # HxWx8

        if self.augment:
            # Augment inp and high jointly (same spatial transform)
            inp_gt = np.concatenate([inp, high], axis=2)  # HxWx11
            inp_gt, _ = random_augment(inp_gt, inp_gt)    # only spatial
            inp  = inp_gt[:,:,:8]
            high = inp_gt[:,:,8:]
            # Colour jitter on low-light channel only
            if np.random.rand() > 0.5:
                inp[:,:,:3] = random_color_jitter(inp[:,:,:3])

        def to_t(arr):
            if arr.ndim == 2:
                return torch.from_numpy(arr[None]).float()
            return torch.from_numpy(arr.transpose(2,0,1)).float()

        inp_t  = to_t(inp)           # (8, H, W)
        gt_t   = to_t(high)          # (3, H, W)
        R_t    = to_t(inp[:,:,3:6])  # (3, H, W)
        L_t    = to_t(inp[:,:,6])    # (1, H, W)
        return inp_t, gt_t, R_t, L_t
