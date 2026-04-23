"""
Stage 2: Conditional DDPM — retinex_diffusion/model.py

Denoising U-Net conditioned on:
  [stage1_output(3) | I_low(3) | R_cnn(3) | L_tv(1)] = 10 channels concat to noisy x_t(3)
  → total 13 input channels

Why diffusion (not GAN) for Stage 2:
  - GANs with only 485 real samples → discriminator memorises training set
  - DDPM trains with MSE on predicted noise → stable, no adversarial collapse
  - Score function generalises better with small data
  - DDIM sampling: 20 steps → fast inference

Architecture: Lightweight U-Net (width=64, 4 levels)
  Each block: ResBlock (GroupNorm + SiLU) + time-embedding injection
  Bottleneck: self-attention (single head, efficient)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding ────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) *
                         torch.arange(half, dtype=torch.float32, device=t.device) / half)
        emb = t.float()[:, None] * freq[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


class TimeEmbed(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPE(ch),
            nn.Linear(ch, ch*4), nn.SiLU(),
            nn.Linear(ch*4, ch*4),
        )
    def forward(self, t): return self.net(t)   # (B, ch*4)


# ── ResBlock with time injection ─────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_ch, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.t_proj = nn.Linear(t_ch, out_ch*2)   # scale + shift
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Time injection (scale + shift via AdaGN)
        t = self.act(t_emb)
        scale, shift = self.t_proj(t).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:,:,None,None]) + shift[:,:,None,None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)


# ── Efficient self-attention (bottleneck only) ───────────────────────────────

class SelfAttn(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv  = nn.Conv2d(ch, ch*3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.heads, C//self.heads, H*W)
        q, k, v = qkv.unbind(1)
        scale = (C // self.heads) ** -0.5
        attn  = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn  = attn.softmax(-1)
        out   = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out   = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ── Denoising U-Net ──────────────────────────────────────────────────────────

class DenoisingUNet(nn.Module):
    """
    Conditional denoising U-Net for DDPM.

    in_ch = 3 (x_t) + 10 (condition) = 13
    out_ch = 3 (predicted noise)
    width  = 64
    depths = (2, 2, 2, 2)  — kept small to avoid overfitting on 485 images
    """
    def __init__(self, in_ch=13, out_ch=3, width=64, depths=(2,2,2,2)):
        super().__init__()
        t_ch = width * 4
        self.t_embed = TimeEmbed(width)

        ch = width
        self.intro = nn.Conv2d(in_ch, ch, 3, 1, 1)

        # Encoder
        self.enc = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_chs = []
        for d in depths:
            blks = nn.ModuleList([ResBlock(ch, ch, t_ch) for _ in range(d)])
            self.enc.append(blks)
            enc_chs.append(ch)
            self.downs.append(nn.Conv2d(ch, ch*2, 4, 2, 1))
            ch *= 2

        # Bottleneck
        self.mid1 = ResBlock(ch, ch, t_ch)
        self.attn  = SelfAttn(ch)
        self.mid2  = ResBlock(ch, ch, t_ch)

        # Decoder
        self.dec   = nn.ModuleList()
        self.ups   = nn.ModuleList()
        for i, d in enumerate(reversed(depths)):
            self.ups.append(nn.ConvTranspose2d(ch, ch//2, 4, 2, 1))
            ch //= 2
            blks = nn.ModuleList([ResBlock(ch*2, ch, t_ch) for _ in range(d)])
            self.dec.append(blks)

        self.out = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, 1, 1),
        )

    def forward(self, x, t, cond):
        """
        x    : (B, 3,  H, W)  noisy image at timestep t
        t    : (B,)            integer timestep
        cond : (B, 10, H, W)  [stage1(3)|I_low(3)|R_cnn(3)|L_tv(1)]
        """
        t_emb = self.t_embed(t)         # (B, width*4)
        h = self.intro(torch.cat([x, cond], 1))

        skips = []
        for blks, down in zip(self.enc, self.downs):
            for blk in blks: h = blk(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.attn(h)
        h = self.mid2(h, t_emb)

        for blks, up, skip in zip(self.dec, self.ups, reversed(skips)):
            h = up(h)
            if h.shape[2:] != skip.shape[2:]:
                h = F.interpolate(h, skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], 1)
            for blk in blks: h = blk(h, t_emb)

        return self.out(h)


# ── DDPM noise schedule ──────────────────────────────────────────────────────

class DDPMSchedule:
    """
    Cosine beta schedule (better than linear for image restoration).
    T=1000 training steps, 20-step DDIM at inference.
    """
    def __init__(self, T=1000, device='cpu'):
        self.T = T
        s = 0.008
        t = torch.linspace(0, T, T+1, device=device)
        alpha_bar = torch.cos((t/T + s) / (1+s) * math.pi/2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = beta.clamp(0, 0.999)

        self.beta      = beta
        self.alpha     = 1 - beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.device    = device

    def to(self, device):
        self.beta      = self.beta.to(device)
        self.alpha     = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.device    = device
        return self

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t."""
        if noise is None: noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1,1,1,1)
        return ab.sqrt() * x0 + (1-ab).sqrt() * noise, noise

    @torch.no_grad()
    def ddim_sample(self, model, cond, steps=20, eta=0.0):
        """
        Reverse DDIM sampling.
        eta=0 → deterministic (better PSNR); eta>0 → stochastic (better perceptual).
        """
        device = cond.device
        B, _, H, W = cond.shape
        x = torch.randn(B, 3, H, W, device=device)

        ts = torch.linspace(self.T-1, 0, steps+1, dtype=torch.long, device=device)
        for i in range(steps):
            t_cur  = ts[i].expand(B)
            t_next = ts[i+1].expand(B)

            ab_cur  = self.alpha_bar[t_cur].view(-1,1,1,1)
            ab_next = self.alpha_bar[t_next.clamp(0)].view(-1,1,1,1) if ts[i+1] >= 0 \
                      else torch.ones_like(ab_cur)

            pred_noise = model(x, t_cur, cond)
            x0_pred    = (x - (1-ab_cur).sqrt() * pred_noise) / ab_cur.sqrt()
            x0_pred    = x0_pred.clamp(-1, 1)

            # DDIM update
            sigma = eta * ((1-ab_next)/(1-ab_cur)).sqrt() * (1 - ab_cur/ab_next).sqrt()
            dir_x  = (1 - ab_next - sigma**2).sqrt() * pred_noise
            x      = ab_next.sqrt() * x0_pred + dir_x
            if eta > 0:
                x += sigma * torch.randn_like(x)

        return x.clamp(-1, 1)   # [-1,1] → caller converts to [0,1]
