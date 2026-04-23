"""
RetinexNAF-MS — dataset.py

Builds 9-channel Retinex input:
  [I_low(3) | R_tv(3) | L_tv(1) | L_hint(1) | zeros(1)]

TV decomposition cached per-image. Aggressive augmentation for overfitting prevention.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.restoration import denoise_tv_chambolle

CACHE = ".retinex_naf_cache"


def tv_decompose(img, weight=0.08):
    """Fast TV Retinex decomp. Returns R(3ch), L(1ch), L_hint(1ch)."""
    L = denoise_tv_chambolle(
        np.max(img, axis=2).astype(np.float32), weight=weight, channel_axis=None
    ).clip(1e-4, 1.0).astype(np.float32)
    R = (img / (L[:,:,None] + 1e-6)).clip(0, 1).astype(np.float32)
    # illumination boost hint: how far is L from target 0.6
    hint = ((0.6 - L) / (0.6 + 1e-6)).clip(0, 1).astype(np.float32)
    return R, L, hint


class RetinexNAFDataset(Dataset):
    """
    Returns:
      inp9  : (9, H, W)  [I_low(3)|R_tv(3)|L_tv(1)|L_hint(1)|zeros(1)]
      gt    : (3, H, W)
      R_tv  : (3, H, W)  for Retinex consistency loss
      L_tv  : (1, H, W)  for illumination monotonicity
    """

    def __init__(self, root, split="our485", size=256, augment=True, cache=True):
        self.low  = os.path.join(root, split, "low")
        self.high = os.path.join(root, split, "high")
        self.size = size
        self.augment = augment
        self.cache   = cache
        self.cdir    = os.path.join(root, CACHE, split)
        if cache:
            os.makedirs(self.cdir, exist_ok=True)

        self.files = sorted(
            f for f in os.listdir(self.low)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        )

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        img = cv2.imread(path)
        if img is None: raise FileNotFoundError(path)
        return cv2.resize(img, (self.size, self.size),
                          interpolation=cv2.INTER_AREA).astype(np.float32)/255.0

    def _decomp(self, fname, img):
        key = os.path.splitext(fname)[0]
        pR = os.path.join(self.cdir, key+'_R.npy')
        pL = os.path.join(self.cdir, key+'_L.npy')
        pH = os.path.join(self.cdir, key+'_H.npy')
        if self.cache and os.path.exists(pR):
            return np.load(pR), np.load(pL), np.load(pH)
        R, L, H = tv_decompose(img)
        if self.cache:
            np.save(pR,R); np.save(pL,L); np.save(pH,H)
        return R, L, H

    # ── augmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def _spatial_aug(a, b):
        k = np.random.randint(4)
        a = np.rot90(a, k).copy(); b = np.rot90(b, k).copy()
        if np.random.rand()>.5: a=a[:,::-1].copy(); b=b[:,::-1].copy()
        if np.random.rand()>.5: a=a[::-1].copy();   b=b[::-1].copy()
        return a, b

    @staticmethod
    def _color_jitter(img):
        h = np.clip(img*255, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(h, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = np.clip(hsv[:,:,0]+np.random.uniform(-5,5),0,179)
        hsv[:,:,1] = np.clip(hsv[:,:,1]*np.random.uniform(0.85,1.15),0,255)
        hsv[:,:,2] = np.clip(hsv[:,:,2]*np.random.uniform(0.85,1.15),0,255)
        return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR).astype(np.float32)/255.0

    @staticmethod
    def _mixup(a_low, a_high, b_low, b_high, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        return lam*a_low+(1-lam)*b_low, lam*a_high+(1-lam)*b_high

    @staticmethod
    def _add_noise(img, sigma_max=0.02):
        sigma = np.random.uniform(0, sigma_max)
        return np.clip(img + np.random.randn(*img.shape).astype(np.float32)*sigma, 0, 1)

    def __getitem__(self, idx):
        fname = self.files[idx]
        low  = self._load(os.path.join(self.low,  fname))
        high = self._load(os.path.join(self.high, fname))
        R_tv, L_tv, L_hint = self._decomp(fname, low)

        if self.augment:
            # 1. spatial augmentation (same for low and high)
            low, high = self._spatial_aug(low, high)
            R_tv, _   = self._spatial_aug(R_tv, R_tv)
            L_tv = np.rot90(L_tv, np.random.randint(4)).copy()
            L_hint = np.rot90(L_hint, np.random.randint(4)).copy()

            # 2. random colour jitter on low only
            if np.random.rand() > 0.5:
                low = self._color_jitter(low)

            # 3. random noise on low only (simulates additional noise)
            if np.random.rand() > 0.6:
                low = self._add_noise(low, sigma_max=0.02)

            # 4. MixUp (20% probability)
            if np.random.rand() < 0.2:
                idx2 = np.random.randint(len(self.files))
                f2   = self.files[idx2]
                l2   = self._load(os.path.join(self.low,  f2))
                h2   = self._load(os.path.join(self.high, f2))
                low, high = self._mixup(low, high, l2, h2)

        # Build 9-channel tensor
        inp9 = np.concatenate([
            low,                      # 3ch
            R_tv,                     # 3ch
            L_tv[:,:,None],           # 1ch
            L_hint[:,:,None],         # 1ch
            np.zeros_like(L_tv[:,:,None]),  # 1ch placeholder
        ], axis=2)                    # HxWx9

        def t(a):
            a = np.ascontiguousarray(a)
            if a.ndim == 2: return torch.from_numpy(a[None]).float()
            return torch.from_numpy(a.transpose(2,0,1)).float()

        return t(inp9), t(high), t(R_tv), t(L_tv)
