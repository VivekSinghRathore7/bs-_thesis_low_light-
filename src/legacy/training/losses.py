"""
losses.py
=========
Loss functions for RAIN training.

Total loss:
    L = λ1·L1(enhanced, gt)
      + λ2·(1 - SSIM(enhanced, gt))
      + λ3·Perceptual(enhanced, gt)   [VGG-16 features]
      + λ4·L1(i_pred, i_gt)           [auxiliary illumination]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ── SSIM loss ────────────────────────────────────────────────────────────────

def gaussian_kernel(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g      = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g      = g / g.sum()
    kernel = g.outer(g)
    return kernel.view(1, 1, size, size)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
        super().__init__()
        self.ws  = window_size
        self.sigma = sigma
        self.C1  = C1
        self.C2  = C2

    def forward(self, pred, target):
        """pred, target: (B, C, H, W) in [−1,1] or [0,1]."""
        B, C, H, W = pred.shape
        kernel = gaussian_kernel(self.ws, self.sigma, pred.device).expand(C, 1, -1, -1)
        pad    = self.ws // 2

        def conv(x):
            return F.conv2d(x, kernel, padding=pad, groups=C)

        mu1 = conv(pred);   mu2 = conv(target)
        mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu12 = mu1 * mu2
        s1  = conv(pred   * pred)    - mu1_sq
        s2  = conv(target * target)  - mu2_sq
        s12 = conv(pred   * target)  - mu12

        num = (2 * mu12  + self.C1) * (2 * s12  + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (s1 + s2 + self.C2)
        ssim_map = num / (den + 1e-8)
        return 1.0 - ssim_map.mean()


# ── Perceptual loss (VGG-16 relu2_2, relu3_3) ────────────────────────────────

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg    = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        blocks = list(vgg.features.children())
        # relu2_2  = up to index 9
        # relu3_3  = up to index 16
        self.slice1 = nn.Sequential(*blocks[:9]).eval()
        self.slice2 = nn.Sequential(*blocks[9:16]).eval()
        for p in self.parameters():
            p.requires_grad = False

        # ImageNet normalisation (applied to [0,1] inputs)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalise(self, x):
        """x in [−1,1] → ImageNet-normalised."""
        x = (x + 1.0) * 0.5   # [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p = self.normalise(pred);   t = self.normalise(target)
        f1_p = self.slice1(p);      f1_t = self.slice1(t)
        f2_p = self.slice2(f1_p);   f2_t = self.slice2(f1_t)
        return F.l1_loss(f1_p, f1_t) + F.l1_loss(f2_p, f2_t)


# ── Combined loss ─────────────────────────────────────────────────────────────

class RAINLoss(nn.Module):
    """
    Parameters
    ----------
    lambda_l1     : weight for pixel L1 loss
    lambda_ssim   : weight for SSIM loss
    lambda_perc   : weight for perceptual loss
    lambda_illum  : weight for auxiliary illumination loss
    """

    def __init__(self,
                 lambda_l1=1.0,
                 lambda_ssim=1.0,
                 lambda_perc=0.1,
                 lambda_illum=0.5):
        super().__init__()
        self.l1    = nn.L1Loss()
        self.ssim  = SSIMLoss()
        self.perc  = PerceptualLoss()
        self.lam   = dict(l1=lambda_l1, ssim=lambda_ssim,
                          perc=lambda_perc, illum=lambda_illum)

    def forward(self, enhanced, i_pred, img_high, i_high):
        """
        enhanced  : (B,3,H,W) tanh output  [−1,1]
        i_pred    : (B,1,H,W) sigmoid output [0,1]
        img_high  : (B,3,H,W) ground-truth  [0,1]
        i_high    : (B,1,H,W) GT illumination [0,1]
        """
        # Bring GT to [−1,1] for L1 / SSIM
        gt_11 = img_high * 2.0 - 1.0

        l_l1   = self.l1(enhanced, gt_11)
        l_ssim = self.ssim(enhanced, gt_11)
        l_perc = self.perc(enhanced, gt_11)
        l_ilm  = self.l1(i_pred, i_high)

        total = (self.lam["l1"]   * l_l1
               + self.lam["ssim"] * l_ssim
               + self.lam["perc"] * l_perc
               + self.lam["illum"]* l_ilm)

        return total, {
            "l1":    l_l1.item(),
            "ssim":  l_ssim.item(),
            "perc":  l_perc.item(),
            "illum": l_ilm.item(),
            "total": total.item(),
        }
