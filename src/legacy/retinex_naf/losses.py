"""
RetinexNAF-MS — losses.py

Loss suite maximising PSNR (L1 + freq), SSIM (MS-SSIM), LPIPS (VGG19 multi-layer).

Total loss:
  L = λ_l1·L1  +  λ_ms·MS-SSIM  +  λvgg·VGG  +  λ_fft·FFT
    + λ_ret·Retinex-consistency  +  λ_mono·Illum-mono
    + λ_aux·sum(aux_losses)      +  λ_edge·Edge-aware
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# ── SSIM ──────────────────────────────────────────────────────────────────────

def _gauss_win(ws, ch, device):
    sigma = 1.5
    c = torch.arange(ws, dtype=torch.float32, device=device) - ws//2
    g = torch.exp(-c**2/(2*sigma**2)); g /= g.sum()
    w = (g.unsqueeze(1)*g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    return w.repeat(ch,1,1,1)


def ssim_map(p, g, ws=11, ch=3):
    C1,C2 = 0.01**2, 0.03**2
    win = _gauss_win(ws, ch, p.device)
    pad = ws//2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1  = F.conv2d(p*p, win, padding=pad, groups=ch) - mu1**2
    s2  = F.conv2d(g*g, win, padding=pad, groups=ch) - mu2**2
    s12 = F.conv2d(p*g, win, padding=pad, groups=ch) - mu1*mu2
    return ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))


class MSSSIMLoss(nn.Module):
    """MS-SSIM (5 scales) — strong structural supervision."""
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def forward(self, p, g):
        loss = 0.0
        for i, w in enumerate(self.weights):
            loss += w * (1.0 - ssim_map(p, g).mean())
            if i < len(self.weights)-1:
                p = F.avg_pool2d(p, 2); g = F.avg_pool2d(g, 2)
                if p.shape[-1] < 11: break
        return loss


# ── VGG19 perceptual (multi-layer) ────────────────────────────────────────────

class VGGLoss(nn.Module):
    """
    Multi-layer VGG19 perceptual loss.
    Captures: texture (relu1_2), structure (relu2_2), semantics (relu3_4, relu4_4).
    """
    def __init__(self):
        super().__init__()
        vgg = tvm.vgg19(weights=tvm.VGG19_Weights.DEFAULT).features
        self.s = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:18], vgg[18:27]])
        for p in self.parameters(): p.requires_grad=False
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.w = [0.1, 0.2, 0.4, 0.3]

    def forward(self, p, g):
        p=(p-self.mean)/self.std; g=(g-self.mean)/self.std
        loss=0.0
        for s,w in zip(self.s, self.w):
            p=s(p); g=s(g)
            loss += w*F.l1_loss(p,g)
        return loss


# ── FFT magnitude + phase loss ────────────────────────────────────────────────

class FFTLoss(nn.Module):
    def forward(self, p, g):
        fp = torch.fft.fft2(p, norm='ortho')
        fg = torch.fft.fft2(g, norm='ortho')
        return F.l1_loss(fp.abs(), fg.abs()) + 0.1*F.l1_loss(fp.angle(), fg.angle())


# ── Retinex physical consistency ──────────────────────────────────────────────

class RetinexConsistency(nn.Module):
    """
    R_cnn ⊗ L_enh ≈ GT.
    Enforces that our decomposition is physically consistent.
    """
    def forward(self, R_cnn, L_enh, gt):
        recon = (R_cnn * L_enh).clamp(0,1)
        return F.l1_loss(recon, gt) + 0.3*(1-ssim_map(recon,gt).mean())


# ── Illumination monotonicity ─────────────────────────────────────────────────

class IllumMono(nn.Module):
    """Penalise L_enh < L_tv (enhanced illumination must not be darker)."""
    def forward(self, L_enh, L_tv):
        return F.relu(L_tv - L_enh).mean()


# ── Edge-aware loss (preserves sharp edges in output) ────────────────────────

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx.repeat(3,1,1,1))
        self.register_buffer('ky', ky.repeat(3,1,1,1))

    def _grad(self, x):
        gx = F.conv2d(x, self.kx, groups=3, padding=1)
        gy = F.conv2d(x, self.ky, groups=3, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, p, g):
        return F.l1_loss(self._grad(p), self._grad(g))


# ── Combined loss ─────────────────────────────────────────────────────────────

class TotalLoss(nn.Module):
    def __init__(self, lam_l1=1.0, lam_ms=1.0, lam_vgg=0.5,
                 lam_fft=0.3, lam_ret=1.5, lam_mono=0.3,
                 lam_aux=0.5, lam_edge=0.1):
        super().__init__()
        self.lam = dict(l1=lam_l1, ms=lam_ms, vgg=lam_vgg, fft=lam_fft,
                        ret=lam_ret, mono=lam_mono, aux=lam_aux, edge=lam_edge)
        self.ms    = MSSSIMLoss()
        self.vgg   = VGGLoss()
        self.fft   = FFTLoss()
        self.ret_c = RetinexConsistency()
        self.mono  = IllumMono()
        self.edge  = EdgeLoss()

    def forward(self, pred, gt, aux_outs, R_cnn, L_enh, L_tv):
        d = {}
        d['l1']   = F.l1_loss(pred, gt)
        d['ms']   = self.ms(pred, gt)
        d['vgg']  = self.vgg(pred, gt)
        d['fft']  = self.fft(pred, gt)
        d['ret']  = self.ret_c(R_cnn, L_enh, gt)
        d['mono'] = self.mono(L_enh, L_tv)
        d['edge'] = self.edge(pred, gt)

        # Multi-scale auxiliary supervision (downsampled GT)
        aux_loss = torch.tensor(0.0, device=pred.device)
        for k, aux in enumerate(aux_outs):
            scale = 2 ** (len(aux_outs) - k)
            gt_s  = F.avg_pool2d(gt, scale)
            if aux.shape[2:] != gt_s.shape[2:]:
                aux = F.interpolate(aux, gt_s.shape[2:], mode='bilinear', align_corners=False)
            aux_loss = aux_loss + F.l1_loss(aux, gt_s) + 0.3*(1-ssim_map(aux, gt_s).mean())
        d['aux'] = aux_loss / max(len(aux_outs), 1)

        total = sum(self.lam[k]*v for k,v in d.items())
        d['total'] = total
        return total, d
