"""
Patch-based LOL dataset for Stage 1.

Key difference from full-image training:
  - 200 random 128×128 patches per image per epoch
  - Effective dataset = 485 × 200 = 97,000 samples/epoch (200× more diversity)
  - This is how ALL 23+ dB methods on LOL-v1 work
"""

import os, cv2, numpy as np, torch
from torch.utils.data import Dataset
from retinex_utils import decompose_retinex


class PatchLOLDataset(Dataset):
    """
    Returns (inp, gt) pairs where inp = [I_low(3) | R_low(3) | I_tv(1)] = 7ch.
    Patches: 128×128, 200 random crops per image → 97,000 samples/epoch for LOL-485.
    """
    def __init__(self, root, split="our485", patch=128, n_patches=200, augment=True):
        self.low_dir  = os.path.join(root, split, "low")
        self.high_dir = os.path.join(root, split, "high")
        self.patch    = patch
        self.n        = n_patches
        self.augment  = augment

        self.files = sorted(
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        )
        # Pre-load all images into RAM (485 images × 2 = ~500MB at full res, fine on H100 node)
        print(f"Loading {split} ({len(self.files)} images) into RAM...", flush=True)
        self.lows, self.highs = [], []
        for f in self.files:
            low  = cv2.imread(os.path.join(self.low_dir,  f)).astype(np.float32)/255.0
            high = cv2.imread(os.path.join(self.high_dir, f)).astype(np.float32)/255.0
            self.lows.append(low)
            self.highs.append(high)
        print("Done.", flush=True)

    def __len__(self):
        return len(self.files) * self.n

    def __getitem__(self, idx):
        img_idx = idx // self.n
        low  = self.lows[img_idx]
        high = self.highs[img_idx]

        H, W = low.shape[:2]
        p = self.patch

        # Random crop (same location for low and high)
        r = np.random.randint(0, max(H - p, 1))
        c = np.random.randint(0, max(W - p, 1))
        low  = low[r:r+p, c:c+p]
        high = high[r:r+p, c:c+p]

        # Augmentation: 8 geometric transforms
        if self.augment:
            k = np.random.randint(4)
            low  = np.rot90(low,  k).copy()
            high = np.rot90(high, k).copy()
            if np.random.rand() > 0.5:
                low  = low[:, ::-1].copy()
                high = high[:, ::-1].copy()
            if np.random.rand() > 0.5:
                low  = low[::-1].copy()
                high = high[::-1].copy()

        # Retinex decompose (on patch — fast)
        R_low, I_tv = decompose_retinex(low)   # R:3ch, I:1ch
        I_tv_3d = I_tv[:, :, np.newaxis] if I_tv.ndim == 2 else I_tv

        # 7-channel input: [I_low(3) | R_low(3) | I_tv(1)]
        inp = np.concatenate([low, R_low, I_tv_3d], axis=2)  # HxWx7

        def t(a):
            a = np.ascontiguousarray(a)
            if a.ndim == 2: return torch.from_numpy(a[None]).float()
            return torch.from_numpy(a.transpose(2,0,1)).float()

        return t(inp), t(high), t(R_low), t(I_tv_3d)


class FullLOLDataset(Dataset):
    """Full-image dataset for evaluation only (no patches)."""
    def __init__(self, root, split="eval15", size=256):
        self.low_dir  = os.path.join(root, split, "low")
        self.high_dir = os.path.join(root, split, "high")
        self.size = size
        self.files = sorted(
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png','.jpg','.jpeg'))
        )

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        low  = cv2.imread(os.path.join(self.low_dir,  f))
        high = cv2.imread(os.path.join(self.high_dir, f))
        # Resize to multiple of 32 (safe for U-Net)
        H, W = low.shape[:2]
        H32 = (H // 32) * 32
        W32 = (W // 32) * 32
        low  = cv2.resize(low,  (W32, H32)).astype(np.float32)/255.0
        high = cv2.resize(high, (W32, H32)).astype(np.float32)/255.0

        from retinex_utils import decompose_retinex
        R_low, I_tv = decompose_retinex(low)
        I_tv_3d = I_tv[:,:,np.newaxis] if I_tv.ndim==2 else I_tv
        inp = np.concatenate([low, R_low, I_tv_3d], axis=2)

        def t(a):
            a = np.ascontiguousarray(a)
            if a.ndim==2: return torch.from_numpy(a[None]).float()
            return torch.from_numpy(a.transpose(2,0,1)).float()
        return t(inp), t(high), t(R_low), t(I_tv_3d)
