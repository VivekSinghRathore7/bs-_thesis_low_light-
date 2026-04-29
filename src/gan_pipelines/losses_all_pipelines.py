"""
Loss function library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# Charbonnier Loss (smoother gradients than L1)
# ============================================================

class CharbonnierLoss(nn.Module):
    """L_charb = sqrt((pred - target)^2 + eps^2)"""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


# ============================================================
# SSIM Loss
# ============================================================

class SSIMLoss(nn.Module):
    """Differentiable SSIM loss (1 - SSIM)."""
    def __init__(self, window_size=11, channels=3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer('window', self._create_window(window_size, channels))

    def _create_window(self, window_size, channels):
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) * g.unsqueeze(0)
        return window_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

    def forward(self, pred, target):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        ch = pred.shape[1]
        window = self.window.to(pred.device)
        if ch != self.channels:
            window = self._create_window(self.window_size, ch).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=pad, groups=ch)
        mu2 = F.conv2d(target, window, padding=pad, groups=ch)
        mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=ch) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=ch) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=ch) - mu12

        ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim.mean()


# ============================================================
# Multi-Scale SSIM Loss
# ============================================================

class MultiScaleSSIMLoss(nn.Module):
    """MS-SSIM: compute SSIM at multiple scales for better structural supervision."""
    def __init__(self, window_size=11, channels=3, weights=None):
        super().__init__()
        self.ssim = SSIMLoss(window_size=window_size, channels=channels)
        self.weights = weights or [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def forward(self, pred, target):
        msssim_loss = 0.0
        for i, w in enumerate(self.weights):
            ssim_val = self.ssim(pred, target)
            msssim_loss += w * ssim_val
            if i < len(self.weights) - 1:
                pred = F.avg_pool2d(pred, 2)
                target = F.avg_pool2d(target, 2)
                if pred.shape[2] < 11 or pred.shape[3] < 11:
                    break
        return msssim_loss


# ============================================================
# VGG Perceptual Loss
# ============================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss using conv3_3 features."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.features = nn.Sequential(*list(vgg.features[:16])).eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        f_pred = self.features(pred)
        f_target = self.features(target)
        return F.l1_loss(f_pred, f_target)


# ============================================================
# LPIPS Training Loss (directly optimize perceptual similarity)
# ============================================================

class LPIPSLoss(nn.Module):
    """
    Differentiable LPIPS-like loss using VGG16 multi-layer features.
    Directly optimizing this improves the LPIPS evaluation metric.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        # Extract features at multiple layers
        self.slice1 = nn.Sequential(*list(vgg[:4])).eval()    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9])).eval()   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:16])).eval()  # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[16:23])).eval() # relu4_3
        self.slice5 = nn.Sequential(*list(vgg[23:30])).eval() # relu5_3

        for p in self.parameters():
            p.requires_grad = False

        # Learned linear weights for each layer (trainable)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0 / 5)),
            nn.Parameter(torch.tensor(1.0 / 5)),
            nn.Parameter(torch.tensor(1.0 / 5)),
            nn.Parameter(torch.tensor(1.0 / 5)),
            nn.Parameter(torch.tensor(1.0 / 5)),
        ])

        self.register_buffer('mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred_n = self._normalize(pred)
        target_n = self._normalize(target)

        loss = 0.0
        x_p, x_t = pred_n, target_n
        slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]

        for i, s in enumerate(slices):
            x_p = s(x_p)
            x_t = s(x_t)
            # Normalize features
            p_norm = x_p / (x_p.norm(dim=1, keepdim=True) + 1e-10)
            t_norm = x_t / (x_t.norm(dim=1, keepdim=True) + 1e-10)
            loss += self.weights[i].abs() * (p_norm - t_norm).pow(2).mean()

        return loss


# ============================================================
# Color Consistency Loss (LAB space)
# ============================================================

class ColorConsistencyLoss(nn.Module):
    """
    Encourages color consistency by comparing mean colors in LAB-like space.
    Uses a differentiable RGB → opponent color space conversion.
    """
    def __init__(self):
        super().__init__()

    def _rgb_to_opponent(self, x):
        """Convert RGB to opponent color space (differentiable)."""
        # L = 0.299*R + 0.587*G + 0.114*B (luminance)
        # a = R - G
        # b = 0.5*(R + G) - B
        R, G, B = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        L = 0.299 * R + 0.587 * G + 0.114 * B
        a = R - G
        b = 0.5 * (R + G) - B
        return torch.cat([L, a, b], dim=1)

    def forward(self, pred, target):
        pred_opp = self._rgb_to_opponent(pred)
        target_opp = self._rgb_to_opponent(target)

        # Global color difference
        mean_pred = pred_opp.mean(dim=[2, 3])
        mean_target = target_opp.mean(dim=[2, 3])
        global_loss = F.mse_loss(mean_pred, mean_target)

        # Local color difference (patch-wise)
        pred_patches = F.avg_pool2d(pred_opp, kernel_size=16, stride=16)
        target_patches = F.avg_pool2d(target_opp, kernel_size=16, stride=16)
        local_loss = F.mse_loss(pred_patches, target_patches)

        return global_loss + 0.5 * local_loss


# ============================================================
# Sobel Edge Loss
# ============================================================

class Sobel(nn.Module):
    """Sobel edge detection for gradient-guided texture loss."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('weight_y', kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, groups=3, padding=1)
        grad_y = F.conv2d(x, self.weight_y, groups=3, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


# ============================================================
# Reflectance Loss (improved)
# ============================================================

class ReflectanceLossV2(nn.Module):
    """
    Consolidated reflectance loss:
    1. Noise-aware Charbonnier (replaces L1)
    2. Gradient-guided texture loss (Sobel)
    3. Channel-decoupled supervision
    4. Frequency-domain loss (FFT)
    5. LPIPS perceptual loss on reflectance
    """
    def __init__(self, alpha=2.0, lambda_grad=0.5, w_b=1.2, w_g=1.0, w_r=1.5,
                 lambda_freq=0.1, lambda_percp=0.15):
        super().__init__()
        self.alpha = alpha
        self.lambda_grad = lambda_grad
        self.w_b, self.w_g, self.w_r = w_b, w_g, w_r
        self.lambda_freq = lambda_freq
        self.lambda_percp = lambda_percp

        self.charb = CharbonnierLoss()
        self.sobel = Sobel()
        self.lpips = LPIPSLoss()

    def forward(self, R_pred, R_high, R_low=None):
        loss_dict = {}
        total = 0.0

        # 1. Noise-aware Charbonnier
        if R_low is not None:
            noise_weight = 1 + self.alpha * (1 - R_low)
            diff = torch.sqrt((R_pred - R_high) ** 2 + 1e-6)
            l_noise = torch.mean(noise_weight * diff)
        else:
            l_noise = self.charb(R_pred, R_high)
        loss_dict['noise_charb'] = l_noise
        total += l_noise

        # 2. Gradient-guided texture
        l_grad = F.l1_loss(self.sobel(R_pred), self.sobel(R_high))
        loss_dict['grad'] = l_grad
        total += self.lambda_grad * l_grad

        # 3. Channel-decoupled
        l_ch = (self.w_b * F.l1_loss(R_pred[:, 0], R_high[:, 0]) +
                self.w_g * F.l1_loss(R_pred[:, 1], R_high[:, 1]) +
                self.w_r * F.l1_loss(R_pred[:, 2], R_high[:, 2]))
        loss_dict['ch_decoupled'] = l_ch
        total += l_ch

        # 4. Frequency-domain
        fft_pred = torch.fft.fft2(R_pred)
        fft_high = torch.fft.fft2(R_high)
        l_freq = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_high))
        loss_dict['freq'] = l_freq
        total += self.lambda_freq * l_freq

        # 5. LPIPS perceptual
        l_percp = self.lpips(R_pred, R_high)
        loss_dict['lpips'] = l_percp
        total += self.lambda_percp * l_percp

        loss_dict['total'] = total
        return total, loss_dict


# ============================================================
# Combined Loss V2 (for final output)
# ============================================================

class CombinedLossV2(nn.Module):
    """
    Combined loss for final enhanced output:
    Charbonnier + MS-SSIM + VGG Perceptual + LPIPS + Color + Adversarial
    """
    def __init__(self, lambda_charb=1.0, lambda_msssim=1.5, lambda_vgg=0.1,
                 lambda_lpips=0.5, lambda_color=0.3, lambda_adv=0.01):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_msssim = lambda_msssim
        self.lambda_vgg = lambda_vgg
        self.lambda_lpips = lambda_lpips
        self.lambda_color = lambda_color
        self.lambda_adv = lambda_adv

        self.charb = CharbonnierLoss()
        self.msssim = MultiScaleSSIMLoss()
        self.vgg = VGGPerceptualLoss()
        self.lpips = LPIPSLoss()
        self.color = ColorConsistencyLoss()

    def forward(self, pred, target, disc_fake=None):
        loss_dict = {}

        l_charb = self.charb(pred, target)
        loss_dict['charb'] = l_charb

        l_msssim = self.msssim(pred, target)
        loss_dict['ms_ssim'] = l_msssim

        l_vgg = self.vgg(pred, target)
        loss_dict['vgg'] = l_vgg

        l_lpips = self.lpips(pred, target)
        loss_dict['lpips'] = l_lpips

        l_color = self.color(pred, target)
        loss_dict['color'] = l_color

        total = (self.lambda_charb * l_charb +
                 self.lambda_msssim * l_msssim +
                 self.lambda_vgg * l_vgg +
                 self.lambda_lpips * l_lpips +
                 self.lambda_color * l_color)

        if disc_fake is not None:
            adv_loss = F.mse_loss(disc_fake, torch.ones_like(disc_fake))
            loss_dict['adv'] = adv_loss
            total += self.lambda_adv * adv_loss

        loss_dict['total'] = total
        return total, loss_dict


# ============================================================
# Reconstruction Consistency Loss
# ============================================================

class ReconstructionConsistencyLoss(nn.Module):
    """Enforces R_pred * I_pred ≈ GT for Retinex physical consistency."""
    def __init__(self, lambda_charb=1.0, lambda_ssim=0.5):
        super().__init__()
        self.lambda_charb = lambda_charb
        self.lambda_ssim = lambda_ssim
        self.charb = CharbonnierLoss()
        self.ssim = SSIMLoss()

    def forward(self, recombined, gt):
        loss = self.lambda_charb * self.charb(recombined, gt)
        loss += self.lambda_ssim * self.ssim(recombined, gt)
        return loss
