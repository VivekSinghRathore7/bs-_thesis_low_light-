"""
RetinexNAF-MS — models.py
=========================
Retinex-conditioned NAFNet with Multi-Scale supervision.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  Retinex Encoder (LightRefineNet frozen)                │
  │    I_low(3) → R_cnn(3), L_tv(1), L_enhanced(1)         │
  └─────────────────────────────────────────────────────────┘
                     │ 9-channel concat
  ┌─────────────────────────────────────────────────────────┐
  │  RetinexNAF U-Net  (NAFNet blocks, 4 levels)            │
  │  Each encoder level: NAFBlocks + Retinex cross-inject   │
  │  Each decoder level: NAFBlocks + Aux output head        │
  └─────────────────────────────────────────────────────────┘

NAFBlock (from NAFNet, ECCV 2022):
  - LayerNorm (instead of BatchNorm)
  - Depthwise conv
  - SimpleGate: x1*x2  (no activation function at all)
  - Simplified Channel Attention
  - Skip connection

Why NAFNet beats U-Net for restoration:
  - SimpleGate removes saturation → cleaner gradient flow
  - Layer norm handles distribution shift from low-light
  - Proven: NAFNet-Large achieves 33.69 dB on GoPro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
#  NAFNet core blocks
# ─────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):
        # x: (B, C, H, W) → norm over C
        return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class SimpleGate(nn.Module):
    """Split channels in half, multiply: f(x) = x1 * x2. No activation needed."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    NAFNet block.  Input: (B, C, H, W).
    expand controls internal width before SimpleGate halves channels.
    dw_expand controls depthwise conv width.
    """
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop=0.0):
        super().__init__()
        dw_ch  = c * dw_expand
        ffn_ch = c * ffn_expand

        # ── spatial mixing ────────────────────────────────
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c,    dw_ch, 1)               # expand
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, 3, 1, 1,        # depthwise
                                groups=dw_ch)
        self.gate1 = SimpleGate()                              # → dw_ch//2
        self.sca   = nn.Sequential(                            # simplified channel attn
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch//2, dw_ch//2, 1),
        )
        self.conv3 = nn.Conv2d(dw_ch//2, c, 1)               # contract

        # ── channel mixing (FFN) ──────────────────────────
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn_ch, 1)
        self.gate2 = SimpleGate()                              # → ffn_ch//2
        self.conv5 = nn.Conv2d(ffn_ch//2, c, 1)

        self.drop  = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        # learnable per-channel scale (beta, gamma)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1))

    def forward(self, x):
        inp = x
        # spatial mixing
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.drop(x)
        y = inp + x * self.beta

        # channel mixing
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.gate2(x)
        x = self.conv5(x)
        x = self.drop(x)
        return y + x * self.gamma


# ─────────────────────────────────────────────────
#  Retinex Injection Module
#  Injects Retinex-extracted features into NAF features
#  at each encoder/decoder level
# ─────────────────────────────────────────────────

class RetinexInject(nn.Module):
    """
    Learns a spatial attention map from Retinex features,
    then scales the image features.
    Retinex features have different channel count → project first.
    """
    def __init__(self, img_ch, ret_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(ret_ch, img_ch, 1),
            nn.GELU(),
            nn.Conv2d(img_ch, img_ch, 3, 1, 1, groups=img_ch),
            nn.Sigmoid(),
        )

    def forward(self, img_feat, ret_feat):
        if ret_feat.shape[2:] != img_feat.shape[2:]:
            ret_feat = F.interpolate(ret_feat, img_feat.shape[2:],
                                     mode='bilinear', align_corners=False)
        scale = self.proj(ret_feat)
        return img_feat * scale + img_feat   # residual modulation


# ─────────────────────────────────────────────────
#  LightRefineNet  (from retinex_cnn.py, ~330K params)
#  Used to produce R_cnn from R_tv
# ─────────────────────────────────────────────────

class _ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(x + self.block(x))


class LightRefineNet(nn.Module):
    """Frozen reflectance refiner — loaded from existing checkpoint."""
    def __init__(self, nf=48, nb=10):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1, bias=False))
        self.body = nn.Sequential(*[_ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        f = self.body(F.relu(self.head(x), inplace=True))
        return torch.clamp(x + self.tail(f), 0, 1)


# ─────────────────────────────────────────────────
#  Retinex Feature Extractor
#  Processes {I_low → R_tv, L_tv} then R_tv → R_cnn
#  Also produces L_enhanced via simple gamma correction
# ─────────────────────────────────────────────────

class RetinexExtractor(nn.Module):
    """
    Holds frozen LightRefineNet. At forward:
      I_low → (TV decomp already done in dataset) → R_tv, L_tv, L_hint (input channels)
      R_tv  → LightRefineNet → R_cnn
      L_tv  → simple learnable gamma → L_enhanced
    Input to this module: the 9-ch tensor from dataset
      [0:3] I_low, [3:6] R_tv, [6] L_tv, [7] L_hint, [8] dummy (padding)
    Returns: Retinex feature maps for injection
    """
    def __init__(self, refine_ckpt=None):
        super().__init__()
        self.refiner = LightRefineNet(nf=48, nb=10)
        if refine_ckpt is not None:
            state = torch.load(refine_ckpt, map_location='cpu', weights_only=True)
            self.refiner.load_state_dict(state)
        # Freeze
        for p in self.refiner.parameters():
            p.requires_grad = False

        # Learnable illumination enhancer (small MLP-in-conv)
        self.illum_enh = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid(),
        )

        # Multi-scale Retinex feature encoder
        # Output: 5 feature maps at scales 1, 1/2, 1/4, 1/8, 1/16
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(8,   32,  3, 1, 1), nn.GELU()),   # s1
            nn.Sequential(nn.Conv2d(32,  64,  3, 2, 1), nn.GELU()),   # s2
            nn.Sequential(nn.Conv2d(64,  128, 3, 2, 1), nn.GELU()),   # s4
            nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.GELU()),   # s8
            nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1), nn.GELU()),   # s16
        ])

    @torch.no_grad()
    def _get_rcnn(self, R_tv):
        return self.refiner(R_tv)

    def forward(self, inp9):
        """
        inp9: (B, 9, H, W)
          [0:3] I_low
          [3:6] R_tv
          [6:7] L_tv
          [7:8] L_hint
          [8:9] (unused placeholder — zero)
        Returns:
          feats: list of 4 multi-scale Retinex feature maps
          R_cnn: (B,3,H,W)  CNN-refined reflectance
          L_enh: (B,1,H,W)  enhanced illumination
        """
        R_tv = inp9[:, 3:6]
        L_tv = inp9[:, 6:7]

        R_cnn = self._get_rcnn(R_tv)
        L_enh = self.illum_enh(L_tv)

        # Build Retinex feature input: [I_low | R_cnn | L_tv | L_enh]
        ret_in = torch.cat([inp9[:, :3], R_cnn, L_tv, L_enh], 1)  # (B,8,H,W)

        feats = []
        x = ret_in
        for layer in self.enc:
            x = layer(x)
            feats.append(x)
        return feats, R_cnn, L_enh   # feats: list of 5 maps (or 4 if enc has 4 layers)


# ─────────────────────────────────────────────────
#  RetinexNAF-MS  —  main model
# ─────────────────────────────────────────────────

class RetinexNAF(nn.Module):
    """
    NAFNet backbone conditioned by Retinex features.

    depths = [2, 2, 4, 8] (NAFNet-Large style per level)
    width  = 64 (base channels)

    Multi-scale auxiliary outputs at decoder levels 1, 2, 3
    for progressive supervision during training.
    """

    def __init__(self,
                 in_ch=9,
                 out_ch=3,
                 width=64,
                 depths=(2, 2, 4, 8),
                 refine_ckpt=None):
        super().__init__()
        self.ret_enc = RetinexExtractor(refine_ckpt)

        # ── image feature projection ──────────────────────────────
        self.intro = nn.Conv2d(in_ch, width, 3, 1, 1)

        # ── encoder ───────────────────────────────────────────────
        enc_blks, downs = [], []
        ret_chs  = [32, 64, 128, 256, 512]   # Retinex feature channels per level (5 max)
        self.enc_injects = nn.ModuleList()

        ch = width
        enc_ch = []
        for i, d in enumerate(depths):
            enc_blks.append(nn.Sequential(*[NAFBlock(ch) for _ in range(d)]))
            self.enc_injects.append(RetinexInject(ch, ret_chs[i]))
            enc_ch.append(ch)
            downs.append(nn.Conv2d(ch, ch*2, 2, 2))    # stride-2 downsample
            ch *= 2

        self.enc_blks = nn.ModuleList(enc_blks)
        self.downs    = nn.ModuleList(downs)

        # ── bottleneck ────────────────────────────────────────────
        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(4)])

        # ── decoder ───────────────────────────────────────────────
        dec_blks, ups = [], []
        self.dec_injects = nn.ModuleList()
        self.aux_heads   = nn.ModuleList()   # auxiliary output heads

        for i in reversed(range(len(depths))):
            ups.append(nn.ConvTranspose2d(ch, ch//2, 2, 2))
            ch //= 2
            dec_blks.append(nn.Sequential(*[NAFBlock(ch) for _ in range(depths[i])]))
            self.dec_injects.append(RetinexInject(ch, ret_chs[i]))
            # Auxiliary head at each decoder level except final
            self.aux_heads.append(
                nn.Conv2d(ch, out_ch, 1) if i > 0 else nn.Identity()
            )

        self.ups      = nn.ModuleList(ups)
        self.dec_blks = nn.ModuleList(dec_blks)

        self.out_conv = nn.Conv2d(width, out_ch, 3, 1, 1)

    def forward(self, x9):
        """
        x9: (B,9,H,W)  [I_low | R_tv | L_tv | L_hint | zero]
        Returns:
            main: (B,3,H,W) final output
            aux : list of (B,3,H,W) auxiliary outputs (len = n_levels-1)
            R_cnn, L_enh for Retinex consistency loss
        """
        ret_feats, R_cnn, L_enh = self.ret_enc(x9)

        # ── encode ────────────────────────────────────────────────
        x = self.intro(x9)
        enc_outs = []
        for i, (blk, down, inj) in enumerate(
                zip(self.enc_blks, self.downs, self.enc_injects)):
            x = blk(x)
            x = inj(x, ret_feats[i])
            enc_outs.append(x)
            x = down(x)

        # ── bottleneck ────────────────────────────────────────────
        x = self.middle(x)

        # ── decode ────────────────────────────────────────────────
        aux_outs = []
        n = len(self.ups)
        for i, (up, blk, inj, aux_h) in enumerate(
                zip(self.ups, self.dec_blks, self.dec_injects, self.aux_heads)):
            x = up(x)
            skip = enc_outs[n-1-i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, skip.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip             # add skip (not concat, saves memory, NAFNet style)
            x = blk(x)
            x = inj(x, ret_feats[n-1-i])
            if i < n-1:
                aux_out = torch.sigmoid(aux_h(x))
                aux_outs.append(aux_out)

        out = torch.sigmoid(self.out_conv(x))
        return out, aux_outs, R_cnn, L_enh
