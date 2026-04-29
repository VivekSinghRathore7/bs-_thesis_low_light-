"""
U-Net architectures for Pipeline 1 and 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, use_bn=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# U-Net Generator (used by Pipeline 1 and 3)
# ============================================================

class UNetGenerator(nn.Module):
    """
    U-Net with skip connections.
    in_ch: input channels (4 for P1, 15 for P3)
    out_ch: output channels (3 for RGB)
    """

    def __init__(self, in_ch=4, out_ch=3, base_filters=64):
        super().__init__()
        bf = base_filters

        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(in_ch, bf), ConvBlock(bf, bf))
        self.enc2 = nn.Sequential(ConvBlock(bf, bf*2, stride=2), ConvBlock(bf*2, bf*2))
        self.enc3 = nn.Sequential(ConvBlock(bf*2, bf*4, stride=2), ConvBlock(bf*4, bf*4))
        self.enc4 = nn.Sequential(ConvBlock(bf*4, bf*8, stride=2), ConvBlock(bf*8, bf*8))

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(bf*8, bf*8, stride=2),
            ResBlock(bf*8),
            ResBlock(bf*8),
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(bf*8, bf*8, 4, 2, 1)
        self.dec4 = nn.Sequential(ConvBlock(bf*16, bf*8), ConvBlock(bf*8, bf*4))

        self.up3 = nn.ConvTranspose2d(bf*4, bf*4, 4, 2, 1)
        self.dec3 = nn.Sequential(ConvBlock(bf*8, bf*4), ConvBlock(bf*4, bf*2))

        self.up2 = nn.ConvTranspose2d(bf*2, bf*2, 4, 2, 1)
        self.dec2 = nn.Sequential(ConvBlock(bf*4, bf*2), ConvBlock(bf*2, bf))

        self.up1 = nn.ConvTranspose2d(bf, bf, 4, 2, 1)
        self.dec1 = nn.Sequential(ConvBlock(bf*2, bf), ConvBlock(bf, bf))

        self.out_conv = nn.Sequential(
            nn.Conv2d(bf, out_ch, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b)
        d4 = self._match_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return (out + 1) / 2  # Map [-1,1] -> [0,1]

    def _match_and_cat(self, up, skip):
        """Match spatial dims and concatenate."""
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up, skip], dim=1)


# ============================================================
# CBAM (Convolutional Block Attention Module)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False),
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        att = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * att


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ============================================================
# Pipeline 2: U-Net generators with skip connections + CBAM
# ============================================================

class IllumNet(nn.Module):
    """Illumination enhancement U-Net: 1ch → 1ch with skip connections."""

    def __init__(self, base_filters=32):
        super().__init__()
        bf = base_filters

        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(1, bf), ConvBlock(bf, bf))
        self.enc2 = nn.Sequential(ConvBlock(bf, bf*2, stride=2), ConvBlock(bf*2, bf*2))
        self.enc3 = nn.Sequential(ConvBlock(bf*2, bf*4, stride=2), ConvBlock(bf*4, bf*4))

        # Bottleneck with CBAM
        self.bottleneck = nn.Sequential(
            ConvBlock(bf*4, bf*4, stride=2),
            ResBlock(bf*4),
            ResBlock(bf*4),
            CBAM(bf*4, reduction=8),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(bf*4, bf*4, 4, 2, 1)
        self.dec3 = nn.Sequential(ConvBlock(bf*8, bf*4), ConvBlock(bf*4, bf*2))

        self.up2 = nn.ConvTranspose2d(bf*2, bf*2, 4, 2, 1)
        self.dec2 = nn.Sequential(ConvBlock(bf*4, bf*2), ConvBlock(bf*2, bf))

        self.up1 = nn.ConvTranspose2d(bf, bf, 4, 2, 1)
        self.dec1 = nn.Sequential(ConvBlock(bf*2, bf), ConvBlock(bf, bf))

        self.out_conv = nn.Sequential(nn.Conv2d(bf, 1, 1), nn.Sigmoid())

    def _match_and_cat(self, up, skip):
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class ReflecNet(nn.Module):
    """Reflectance denoising U-Net: 3ch → 3ch with skip connections + CBAM."""

    def __init__(self, base_filters=48):
        super().__init__()
        bf = base_filters

        # Encoder
        self.enc1 = nn.Sequential(ConvBlock(3, bf), ConvBlock(bf, bf))
        self.enc2 = nn.Sequential(ConvBlock(bf, bf*2, stride=2), ConvBlock(bf*2, bf*2))
        self.enc3 = nn.Sequential(ConvBlock(bf*2, bf*4, stride=2), ConvBlock(bf*4, bf*4))

        # Bottleneck with CBAM
        self.bottleneck = nn.Sequential(
            ConvBlock(bf*4, bf*4, stride=2),
            ResBlock(bf*4),
            ResBlock(bf*4),
            CBAM(bf*4, reduction=8),
        )

        # Decoder with skip connections
        self.up3 = nn.ConvTranspose2d(bf*4, bf*4, 4, 2, 1)
        self.dec3 = nn.Sequential(ConvBlock(bf*8, bf*4), ConvBlock(bf*4, bf*2))

        self.up2 = nn.ConvTranspose2d(bf*2, bf*2, 4, 2, 1)
        self.dec2 = nn.Sequential(ConvBlock(bf*4, bf*2), ConvBlock(bf*2, bf))

        self.up1 = nn.ConvTranspose2d(bf, bf, 4, 2, 1)
        self.dec1 = nn.Sequential(ConvBlock(bf*2, bf), ConvBlock(bf, bf))

        self.out_conv = nn.Sequential(nn.Conv2d(bf, 3, 1), nn.Sigmoid())

    def _match_and_cat(self, up, skip):
        if up.shape[2:] != skip.shape[2:]:
            up = F.interpolate(up, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = self._match_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._match_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._match_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class RefineBlock(nn.Module):
    """Residual refinement block to fix recombination artifacts."""

    def __init__(self, base_filters=48):
        super().__init__()
        bf = base_filters
        self.net = nn.Sequential(
            ConvBlock(3, bf),
            ResBlock(bf),
            ResBlock(bf),
            ConvBlock(bf, bf),
            nn.Conv2d(bf, 3, 3, 1, 1),
        )

    def forward(self, x):
        # Residual learning: learn the correction, not the full image
        return torch.clamp(x + self.net(x), 0.0, 1.0)


# ============================================================
# PatchGAN Discriminator (shared by all pipelines)
# ============================================================

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_ch=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)
