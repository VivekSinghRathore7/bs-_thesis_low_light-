"""
RetinexRestormer — retinex_restormer/model.py

Restormer-style U-Net for low-light image enhancement.
Input : 7ch [I_low(3) | R_low(3) | I_tv(1)]
Output: 3ch enhanced image in [0,1]

Key design choices:
  - MDTA (transposed attention) is O(C²) not O(HW²) → works on 400×600 full images
  - GDFN (gated DConv FFN) better than plain MLP for local structure
  - width=32, depths=(2,2,2,4) → ~5M params — safe for 485-image LOL-v1
  - Sigmoid output (no clamp needed downstream)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class MDTA(nn.Module):
    """Multi-DConv Head Transposed Attention — O(C²) complexity."""
    def __init__(self, ch, heads, bias=False):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.temp  = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv    = nn.Conv2d(ch, ch * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(ch * 3, ch * 3, 3, 1, 1, groups=ch * 3, bias=bias)
        self.proj   = nn.Conv2d(ch, ch, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)           # each (B, C, H, W)
        q = q.reshape(B, self.heads, C // self.heads, H * W)
        k = k.reshape(B, self.heads, C // self.heads, H * W)
        v = v.reshape(B, self.heads, C // self.heads, H * W)
        q = F.normalize(q, dim=-1)              # normalize along HW
        k = F.normalize(k, dim=-1)
        # Transposed: (C/h, HW) @ (HW, C/h) → (C/h, C/h)
        attn = (q @ k.transpose(-2, -1)) * self.temp   # (B, h, C/h, C/h)
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).reshape(B, C, H, W)          # (B, h, C/h, HW) → (B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated DConv Feed-Forward Network."""
    def __init__(self, ch, expansion=2.66, bias=False):
        super().__init__()
        hidden = int(ch * expansion)
        self.proj_in  = nn.Conv2d(ch, hidden * 2, 1, bias=bias)
        self.dw       = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, ch, 1, bias=bias)

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        return self.proj_out(F.gelu(x1) * x2)


class RestormerBlock(nn.Module):
    def __init__(self, ch, heads, ffn_exp=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm2d(ch)
        self.attn  = MDTA(ch, heads, bias)
        self.norm2 = LayerNorm2d(ch)
        self.ffn   = GDFN(ch, ffn_exp, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def _make_level(ch, heads, depth, ffn_exp=2.66):
    return nn.Sequential(*[RestormerBlock(ch, heads, ffn_exp) for _ in range(depth)])


class RetinexRestormer(nn.Module):
    """
    4-level Restormer U-Net.
    Default config (~5M params at width=32, depths=(2,2,2,4)):
        width=32  → channels [32, 64, 128, 256]
        heads     → [1, 2, 4, 8]
    """
    def __init__(self, in_ch=7, out_ch=3, width=32,
                 depths=(2, 2, 2, 4), heads=(1, 2, 4, 8), ffn_exp=2.66):
        super().__init__()
        W  = width
        ch = [W, W * 2, W * 4, W * 8]

        self.stem = nn.Conv2d(in_ch, ch[0], 3, 1, 1, bias=False)

        # Encoder
        self.enc1  = _make_level(ch[0], heads[0], depths[0], ffn_exp)
        self.down1 = nn.Conv2d(ch[0], ch[1], 2, 2)

        self.enc2  = _make_level(ch[1], heads[1], depths[1], ffn_exp)
        self.down2 = nn.Conv2d(ch[1], ch[2], 2, 2)

        self.enc3  = _make_level(ch[2], heads[2], depths[2], ffn_exp)
        self.down3 = nn.Conv2d(ch[2], ch[3], 2, 2)

        # Bottleneck
        self.bot   = _make_level(ch[3], heads[3], depths[3], ffn_exp)

        # Decoder (skip connections fused with 1×1 conv)
        self.up3   = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.fuse3 = nn.Conv2d(ch[2] * 2, ch[2], 1, bias=False)
        self.dec3  = _make_level(ch[2], heads[2], depths[2], ffn_exp)

        self.up2   = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.fuse2 = nn.Conv2d(ch[1] * 2, ch[1], 1, bias=False)
        self.dec2  = _make_level(ch[1], heads[1], depths[1], ffn_exp)

        self.up1   = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.fuse1 = nn.Conv2d(ch[0] * 2, ch[0], 1, bias=False)
        self.dec1  = _make_level(ch[0], heads[0], depths[0], ffn_exp)

        self.out_conv = nn.Sequential(
            nn.Conv2d(ch[0], out_ch, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.stem(x)

        e1 = self.enc1(s)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b  = self.bot(self.down3(e3))

        d3 = self.dec3(self.fuse3(torch.cat([self.up3(b),  e3], 1)))
        d2 = self.dec2(self.fuse2(torch.cat([self.up2(d3), e2], 1)))
        d1 = self.dec1(self.fuse1(torch.cat([self.up1(d2), e1], 1)))

        return self.out_conv(d1)
