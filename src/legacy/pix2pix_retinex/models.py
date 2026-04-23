"""
RetinexPix2Pix — models.py

Generator  : Enhanced pix2pix U-Net (8-ch Retinex input, InstanceNorm, CBAM, Dropout)
Discriminator : Multi-scale PatchGAN × 3 (spectral norm, conditional 6-ch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ─────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────

class CBAM(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch//r, 8)), nn.ReLU(),
            nn.Linear(max(ch//r, 8), ch), nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x * self.ca(x).view(B, C, 1, 1)
        avg = x.mean(1, keepdim=True)
        mx  = x.max(1, keepdim=True)[0]
        x = x * self.sa(torch.cat([avg, mx], 1))
        return x


def _norm(ch, norm_type="instance"):
    if norm_type == "instance":
        return nn.InstanceNorm2d(ch, affine=True)
    return nn.BatchNorm2d(ch)


# ─────────────────────────────────────────────────
# U-Net encoder block  (conv-norm-activation + dropout)
# ─────────────────────────────────────────────────

class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, norm=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
        if norm:
            layers.append(_norm(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────
# U-Net decoder block  (transposed-conv + skip + CBAM)
# ─────────────────────────────────────────────────

class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, use_cbam=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False)
        self.norm_up = _norm(out_ch)

        fused = out_ch + skip_ch
        self.conv = nn.Sequential(
            nn.Conv2d(fused, out_ch, 3, 1, 1, bias=False),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, skip):
        x = F.relu(self.norm_up(self.up(x)), inplace=True)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], 1)
        x = self.conv(x)
        x = self.cbam(x)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────────
# RetinexUNet — 5-level, 8-ch input, 3-ch output
# ─────────────────────────────────────────────────
#
# Input channels (8):
#   [0:3]  I_low  — original low-light RGB
#   [3:6]  R_tv   — TV-decomposed reflectance
#   [6]    L_tv   — TV illumination (1ch)
#   [7]    L_hint — normalised illumination boost (1ch)

class RetinexUNet(nn.Module):
    def __init__(self, in_ch=8, out_ch=3, base=64):
        super().__init__()
        b = base
        # Encoder: no norm on first block (standard pix2pix convention)
        self.e1 = EncBlock(in_ch,  b,    norm=False)          # 128
        self.e2 = EncBlock(b,      b*2,  dropout=0.0)         # 64
        self.e3 = EncBlock(b*2,    b*4,  dropout=0.0)         # 32
        self.e4 = EncBlock(b*4,    b*8,  dropout=0.1)         # 16
        self.e5 = EncBlock(b*8,    b*8,  dropout=0.1)         # 8

        # Bottleneck
        self.bottle = nn.Sequential(
            nn.Conv2d(b*8, b*8, 4, 2, 1, bias=False),         # 4
            nn.ReLU(inplace=True),
        )

        # Decoder (dropout on first 3 for regularization like pix2pix)
        self.d5 = DecBlock(b*8, b*8, b*8, dropout=0.5)
        self.d4 = DecBlock(b*8, b*8, b*8, dropout=0.5)
        self.d3 = DecBlock(b*8, b*4, b*4, dropout=0.5)
        self.d2 = DecBlock(b*4, b*2, b*2, dropout=0.0)
        self.d1 = DecBlock(b*2, b,   b,   dropout=0.0, use_cbam=False)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(b, out_ch, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        b  = self.bottle(e5)

        d = self.d5(b,  e5)
        d = self.d4(d,  e4)
        d = self.d3(d,  e3)
        d = self.d2(d,  e2)
        d = self.d1(d,  e1)
        return (self.out(d) + 1.0) / 2.0   # [0,1]


# ─────────────────────────────────────────────────
# Single PatchGAN discriminator (spectral-normed)
# Conditional: input = [low(3ch) | enhanced(3ch)] = 6ch
# ─────────────────────────────────────────────────

class PatchDisc(nn.Module):
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        def _block(ic, oc, stride, norm=True):
            layers = [spectral_norm(nn.Conv2d(ic, oc, 4, stride, 1, bias=False))]
            if norm:
                layers.append(nn.InstanceNorm2d(oc, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *_block(in_ch,   ndf,   2, norm=False),
            *_block(ndf,     ndf*2, 2),
            *_block(ndf*2,   ndf*4, 2),
            *_block(ndf*4,   ndf*8, 1),
            spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 1)),
        )

    def forward(self, x):
        return self.model(x)

    def get_features(self, x):
        """Return intermediate feature maps for feature-matching loss."""
        feats = []
        for layer in self.model:
            x = layer(x)
            if isinstance(layer, nn.LeakyReLU):
                feats.append(x)
        return feats


# ─────────────────────────────────────────────────
# Multi-Scale Discriminator (3 scales)
# ─────────────────────────────────────────────────

class MultiScaleDisc(nn.Module):
    """
    3 PatchDisc discriminators at scales ×1, ×0.5, ×0.25.
    Downsampling is done on the input before each discriminator.
    """
    def __init__(self, in_ch=6, ndf=64, n_scales=3):
        super().__init__()
        self.discs = nn.ModuleList([PatchDisc(in_ch, ndf) for _ in range(n_scales)])
        self.n_scales = n_scales

    def forward(self, x):
        outs = []
        for i, D in enumerate(self.discs):
            if i > 0:
                x = F.avg_pool2d(x, 3, stride=2, padding=1)
            outs.append(D(x))
        return outs

    def get_all_features(self, x):
        all_feats = []
        for i, D in enumerate(self.discs):
            if i > 0:
                x = F.avg_pool2d(x, 3, stride=2, padding=1)
            all_feats.append(D.get_features(x))
        return all_feats
