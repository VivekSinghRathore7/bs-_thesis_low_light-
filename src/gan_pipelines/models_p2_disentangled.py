"""
Advanced architectures for Disentangled Pipeline 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building blocks
# ============================================================

class ConvBlock(nn.Module):
    """Conv-InstanceNorm-LeakyReLU block."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1,
                 use_norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                            bias=(not use_norm))]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with InstanceNorm."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


# ============================================================
# Self-Attention Module
# ============================================================

class SelfAttention(nn.Module):
    """Self-Attention for capturing long-range dependencies."""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(torch.bmm(q, k) / (C // 8) ** 0.5, dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return x + self.gamma * out


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
# Improved IllumNet (Deeper U-Net for Illumination)
# ============================================================

class IllumNetV2(nn.Module):
    """Deeper illumination U-Net: 1ch -> 1ch with 4-level encoder + CBAM."""

    def __init__(self, base_filters=32):
        super().__init__()
        bf = base_filters

        # Encoder (4 levels)
        self.enc1 = nn.Sequential(ConvBlock(1, bf), ConvBlock(bf, bf))
        self.enc2 = nn.Sequential(ConvBlock(bf, bf*2, stride=2), ConvBlock(bf*2, bf*2))
        self.enc3 = nn.Sequential(ConvBlock(bf*2, bf*4, stride=2), ConvBlock(bf*4, bf*4))
        self.enc4 = nn.Sequential(ConvBlock(bf*4, bf*8, stride=2), ConvBlock(bf*8, bf*8))

        # Bottleneck with CBAM
        self.bottleneck = nn.Sequential(
            ConvBlock(bf*8, bf*8, stride=2),
            ResBlock(bf*8),
            ResBlock(bf*8),
            CBAM(bf*8, reduction=8),
        )

        # Decoder with skip connections
        self.up4 = nn.ConvTranspose2d(bf*8, bf*8, 4, 2, 1)
        self.dec4 = nn.Sequential(ConvBlock(bf*16, bf*8), ConvBlock(bf*8, bf*4))

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
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))
        return self.out_conv(d1)


# ============================================================
# Improved ReflecNet (Deeper + Self-Attention)
# ============================================================

class ReflecNetV2(nn.Module):
    """Deeper reflectance U-Net: 3ch -> 3ch with 4-level + Self-Attention + CBAM."""

    def __init__(self, base_filters=64):
        super().__init__()
        bf = base_filters

        self.enc1 = nn.Sequential(ConvBlock(3, bf), ConvBlock(bf, bf))
        self.enc2 = nn.Sequential(ConvBlock(bf, bf*2, stride=2), ConvBlock(bf*2, bf*2))
        self.enc3 = nn.Sequential(ConvBlock(bf*2, bf*4, stride=2), ConvBlock(bf*4, bf*4))
        self.enc4 = nn.Sequential(ConvBlock(bf*4, bf*8, stride=2), ConvBlock(bf*8, bf*8))

        self.bottleneck = nn.Sequential(
            ConvBlock(bf*8, bf*8, stride=2),
            ResBlock(bf*8),
            ResBlock(bf*8),
            ResBlock(bf*8),
            SelfAttention(bf*8),
            CBAM(bf*8, reduction=8),
        )

        self.up4 = nn.ConvTranspose2d(bf*8, bf*8, 4, 2, 1)
        self.dec4 = nn.Sequential(ConvBlock(bf*16, bf*8), ConvBlock(bf*8, bf*4), CBAM(bf*4))

        self.up3 = nn.ConvTranspose2d(bf*4, bf*4, 4, 2, 1)
        self.dec3 = nn.Sequential(ConvBlock(bf*8, bf*4), ConvBlock(bf*4, bf*2), CBAM(bf*2))

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
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))
        return self.out_conv(d1)


# ============================================================
# Multi-Scale Refinement Block
# ============================================================

class MultiScaleRefineBlock(nn.Module):
    """Multi-scale residual refinement at 1x, 0.5x, 0.25x scales."""

    def __init__(self, base_filters=64):
        super().__init__()
        bf = base_filters

        self.scale1 = nn.Sequential(ConvBlock(3, bf), ResBlock(bf), ResBlock(bf))
        self.scale2 = nn.Sequential(ConvBlock(3, bf), ResBlock(bf), ResBlock(bf))
        self.scale3 = nn.Sequential(ConvBlock(3, bf // 2), ResBlock(bf // 2))

        self.fusion = nn.Sequential(
            ConvBlock(bf + bf + bf // 2, bf * 2),
            CBAM(bf * 2),
            ConvBlock(bf * 2, bf),
            nn.Conv2d(bf, 3, 3, 1, 1),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        f1 = self.scale1(x)

        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        f2 = self.scale2(x_half)
        f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        f3 = self.scale3(x_quarter)
        f3 = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False)

        fused = torch.cat([f1, f2, f3], dim=1)
        residual = self.fusion(fused)
        return torch.clamp(x + residual, 0.0, 1.0)


# ============================================================
# Conditional PatchGAN Discriminator
# ============================================================

class ConditionalPatchDiscriminator(nn.Module):
    """PatchGAN: input = concat(low_image, output_image) = 6ch."""

    def __init__(self, in_ch=6, base_filters=64):
        super().__init__()
        bf = base_filters
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, bf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(bf, bf*2, 4, 2, 1)),
            nn.InstanceNorm2d(bf*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(bf*2, bf*4, 4, 2, 1)),
            nn.InstanceNorm2d(bf*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(bf*4, bf*8, 4, 1, 1)),
            nn.InstanceNorm2d(bf*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(bf*8, 1, 4, 1, 1),
        )

    def forward(self, input_img, output_img):
        x = torch.cat([input_img, output_img], dim=1)
        return self.model(x)


class DualScaleDiscriminator(nn.Module):
    """Full + half resolution discriminator for local AND structural quality."""

    def __init__(self, in_ch=6, base_filters=64):
        super().__init__()
        self.disc_full = ConditionalPatchDiscriminator(in_ch, base_filters)
        self.disc_half = ConditionalPatchDiscriminator(in_ch, base_filters)

    def forward(self, input_img, output_img):
        d_full = self.disc_full(input_img, output_img)
        input_half = F.interpolate(input_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        output_half = F.interpolate(output_img, scale_factor=0.5, mode='bilinear', align_corners=False)
        d_half = self.disc_half(input_half, output_half)
        return d_full, d_half


# ============================================================
# EMA (Exponential Moving Average) Helper
# ============================================================

class EMA:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + \
                                    (1 - self.decay) * param.data

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}
