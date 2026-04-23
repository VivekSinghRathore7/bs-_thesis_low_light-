"""
RetinexPix2Pix — losses.py

Loss suite designed to maximise PSNR, SSIM, and LPIPS simultaneously.

Total generator loss:
  L_G = λ_l1 · L1  +  λ_ms · MS-SSIM  +  λ_vgg · VGG(multi-layer)
       + λ_adv · GAN  +  λ_fm · FeatureMatch
       + λ_fft · FFT-magnitude  +  λ_ret · Retinex-consistency
       + λ_mono · Illum-monotonicity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ─────────────────────────────────────────────────
# SSIM helper
# ─────────────────────────────────────────────────

def _ssim_map(p, g, win, pad, ch):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1  = F.conv2d(p*p, win, padding=pad, groups=ch) - mu1**2
    s2  = F.conv2d(g*g, win, padding=pad, groups=ch) - mu2**2
    s12 = F.conv2d(p*g, win, padding=pad, groups=ch) - mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))


def _make_gauss_win(ws, ch, device):
    sigma = 1.5
    coords = torch.arange(ws, dtype=torch.float32, device=device) - ws//2
    g = torch.exp(-coords**2/(2*sigma**2)); g /= g.sum()
    win = (g.unsqueeze(1)*g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return win.repeat(ch, 1, 1, 1)


class SSIMLoss(nn.Module):
    def __init__(self, ws=11, ch=3):
        super().__init__()
        self.ws, self.ch = ws, ch

    def forward(self, p, g):
        win = _make_gauss_win(self.ws, self.ch, p.device)
        return 1.0 - _ssim_map(p, g, win, self.ws//2, self.ch).mean()


class MSSSIMLoss(nn.Module):
    """Multi-scale SSIM (5 scales). Stronger structural supervision than single-scale."""
    def __init__(self, ws=11, ch=3):
        super().__init__()
        self.ws, self.ch = ws, ch
        self.weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def forward(self, p, g):
        loss = 0.0
        for i, w in enumerate(self.weights):
            win = _make_gauss_win(self.ws, self.ch, p.device)
            loss += w * (1.0 - _ssim_map(p, g, win, self.ws//2, self.ch).mean())
            if i < len(self.weights)-1:
                p = F.avg_pool2d(p, 2)
                g = F.avg_pool2d(g, 2)
                if p.shape[-1] < self.ws:
                    break
        return loss


# ─────────────────────────────────────────────────
# Multi-layer VGG perceptual loss
# relu1_2, relu2_2, relu3_3, relu4_3
# ─────────────────────────────────────────────────

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:18],  # relu3_4
            vgg[18:27], # relu4_4
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
        # scale weights: earlier layers → pixel quality, later → structure
        self.layer_w = [0.1, 0.2, 0.4, 0.3]

    def forward(self, p, g):
        p = (p - self.mean) / self.std
        g = (g - self.mean) / self.std
        loss = 0.0
        for s, w in zip(self.slices, self.layer_w):
            p = s(p); g = s(g)
            loss += w * F.l1_loss(p, g)
        return loss


# ─────────────────────────────────────────────────
# FFT magnitude loss  — forces frequency consistency
# ─────────────────────────────────────────────────

class FFTLoss(nn.Module):
    def forward(self, p, g):
        fp = torch.fft.fft2(p, norm='ortho')
        fg = torch.fft.fft2(g, norm='ortho')
        return F.l1_loss(fp.abs(), fg.abs()) + F.l1_loss(fp.angle(), fg.angle()) * 0.1


# ─────────────────────────────────────────────────
# Feature-matching loss  (pix2pix-HD style)
# Forces G to reproduce D's intermediate activations
# ─────────────────────────────────────────────────

class FeatureMatchingLoss(nn.Module):
    def __init__(self, n_scales=3):
        super().__init__()
        self.n = n_scales

    def forward(self, real_feats_list, fake_feats_list):
        loss = 0.0
        total = 0
        for real_feats, fake_feats in zip(real_feats_list, fake_feats_list):
            for rf, ff in zip(real_feats, fake_feats):
                loss += F.l1_loss(ff, rf.detach())
                total += 1
        return loss / max(total, 1)


# ─────────────────────────────────────────────────
# Retinex physical consistency  R⊗I ≈ GT
# Also enforces illumination monotonicity  I_pred ≥ I_low
# ─────────────────────────────────────────────────

class RetinexConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()

    def forward(self, R_pred, I_pred, gt):
        recon = (R_pred * I_pred).clamp(0, 1)
        return F.l1_loss(recon, gt) + 0.5 * self.ssim(recon, gt)


class IllumMonotonicityLoss(nn.Module):
    """Penalise I_pred < I_low  (enhanced illumination must be >= original)."""
    def forward(self, I_pred, I_low):
        return F.relu(I_low - I_pred).mean()


# ─────────────────────────────────────────────────
# GAN losses (LSGAN — more stable than vanilla)
# ─────────────────────────────────────────────────

def gan_loss_real(disc_out):
    return sum(F.mse_loss(d, torch.ones_like(d)) for d in disc_out) / len(disc_out)


def gan_loss_fake(disc_out):
    return sum(F.mse_loss(d, torch.zeros_like(d)) for d in disc_out) / len(disc_out)


def gan_loss_gen(disc_out):
    return sum(F.mse_loss(d, torch.ones_like(d)) for d in disc_out) / len(disc_out)


# ─────────────────────────────────────────────────
# Combined generator loss
# ─────────────────────────────────────────────────

class GeneratorLoss(nn.Module):
    def __init__(self,
                 lam_l1=10.0,
                 lam_ms=2.0,
                 lam_vgg=1.0,
                 lam_adv=1.0,
                 lam_fm=10.0,
                 lam_fft=0.5,
                 lam_ret=2.0,
                 lam_mono=0.5):
        super().__init__()
        self.lam = dict(l1=lam_l1, ms=lam_ms, vgg=lam_vgg,
                        adv=lam_adv, fm=lam_fm, fft=lam_fft,
                        ret=lam_ret, mono=lam_mono)
        self.ms_ssim  = MSSSIMLoss()
        self.vgg      = VGGLoss()
        self.fft      = FFTLoss()
        self.fm       = FeatureMatchingLoss()
        self.ret_cons = RetinexConsistencyLoss()
        self.mono     = IllumMonotonicityLoss()

    def forward(self, pred, gt,
                disc_fake_outs, disc_real_feats, disc_fake_feats,
                R_pred, I_pred, I_low,
                adv_weight=1.0):
        d = {}
        d['l1']   = F.l1_loss(pred, gt)
        d['ms']   = self.ms_ssim(pred, gt)
        d['vgg']  = self.vgg(pred, gt)
        d['adv']  = gan_loss_gen(disc_fake_outs) * adv_weight
        d['fm']   = self.fm(disc_real_feats, disc_fake_feats)
        d['fft']  = self.fft(pred, gt)
        d['ret']  = self.ret_cons(R_pred, I_pred, gt)
        d['mono'] = self.mono(I_pred, I_low)

        total = sum(self.lam[k] * v for k, v in d.items())
        d['total'] = total
        return total, d
