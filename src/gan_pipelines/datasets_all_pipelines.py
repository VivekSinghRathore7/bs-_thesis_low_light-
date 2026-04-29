"""
All dataset loaders (LOL-v1).
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from retinex_utils import decompose_retinex, get_all_enhanced_illuminations, enhance_clahe


class BaseLOLDataset(Dataset):
    """Base dataset that loads low/high pairs and does Retinex decomposition."""

    def __init__(self, root_dir, split="our485", img_size=256):
        self.low_dir = os.path.join(root_dir, split, "low")
        self.high_dir = os.path.join(root_dir, split, "high")
        self.img_size = img_size
        filenames = sorted([
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # --- Curriculum pre-calculation ---
        # Calculate mean brightness to sort from brightest (easiest) to darkest (hardest)
        brightness_list = []
        for f in filenames:
            path = os.path.join(self.low_dir, f)
            img = cv2.imread(path)
            brightness = img.mean() if img is not None else 0
            brightness_list.append((f, brightness))
            
        brightness_list.sort(key=lambda x: x[1], reverse=True)
        self.all_filenames = [x[0] for x in brightness_list]
        self.filenames = list(self.all_filenames)

    def set_curriculum_ratio(self, ratio):
        """Updates the active subset of files for curriculum learning (0.0 to 1.0)"""
        ratio = max(0.01, min(1.0, ratio))
        n = max(1, int(len(self.all_filenames) * ratio))
        self.filenames = self.all_filenames[:n]

    def __len__(self):
        return len(self.filenames)

    def _load_and_resize(self, path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read {path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        return img.astype(np.float32) / 255.0

    def _to_tensor(self, img):
        """HWC float32 [0,1] → CHW tensor."""
        if img.ndim == 2:
            img = img[np.newaxis, :, :]  # 1HW
        else:
            img = img.transpose(2, 0, 1)  # CHW
        return torch.from_numpy(img).float()


class Pipeline1Dataset(BaseLOLDataset):
    """Returns: (concat_4ch, gt_rgb) where concat = [low_rgb, I_enhanced]."""

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))

        R_low, I_low = decompose_retinex(low)

        # Enhance illumination with CLAHE
        I_u8 = (I_low * 255).astype(np.uint8)
        I_enh = enhance_clahe(I_u8).astype(np.float32) / 255.0

        # Concat: low_rgb(3ch) + I_enhanced(1ch) = 4ch
        I_enh_3d = I_enh[:, :, np.newaxis]
        inp = np.concatenate([low, I_enh_3d], axis=2)  # HxWx4

        return self._to_tensor(inp), self._to_tensor(high)


class Pipeline2Dataset(BaseLOLDataset):
    """Returns: (R_low, I_low, R_high, I_high, gt_rgb)."""

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))

        R_low, I_low = decompose_retinex(low)
        R_high, I_high = decompose_retinex(high)

        return (
            self._to_tensor(R_low),      # 3ch
            self._to_tensor(I_low),      # 1ch
            self._to_tensor(R_high),     # 3ch
            self._to_tensor(I_high),     # 1ch
            self._to_tensor(high),       # 3ch GT
        )


class Pipeline3Dataset(BaseLOLDataset):
    """Returns: (concat_15ch, gt_rgb) where concat = [low_rgb, 4 candidates]."""

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        low = self._load_and_resize(os.path.join(self.low_dir, fname))
        high = self._load_and_resize(os.path.join(self.high_dir, fname))

        R_low, I_low = decompose_retinex(low)
        enhanced_maps = get_all_enhanced_illuminations(I_low)

        # Generate candidate images: R_low * I_enhanced_k
        candidates = []
        for key in ['clahe', 'agc', 'glagc', 'log']:
            I_enh = enhanced_maps[key][:, :, np.newaxis]
            candidate = np.clip(R_low * I_enh, 0, 1)
            candidates.append(candidate)

        # Concat: low_rgb(3) + 4 candidates(4x3=12) = 15ch
        inp = np.concatenate([low] + candidates, axis=2)

        return self._to_tensor(inp), self._to_tensor(high)
