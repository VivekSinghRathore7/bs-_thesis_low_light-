"""
retinex_cnn.py
==============
Deep-learning reflectance refinement: supervised CNN trained on LOL-485 pairs.

Pipeline
--------
Phase 1  Decompose all 485 train + 15 eval images with a fast TV-illumination
         method to obtain aligned (R_low, R_high) patch pairs.

Phase 2  Train LightRefineNet — a compact residual CNN (~330 K params) that
         learns the R_low → R_high mapping directly.
         Loss = L1 + λ_SSIM · (1 − SSIM)
         Trained with patch augmentation; H100 GPU; ~10 min total.

Phase 3  Evaluate on eval15:
         (a)  TV-decompose → CNN          (fast baseline)
         (b)  A5 (LogTV-Retinex) → CNN   (best classical + CNN refinement)
         Compare both to A5 alone and report full metrics.

Architecture: LightRefineNet
  Input 3-ch → Head (3→48) → 8 ResBlocks (48-ch, BN+ReLU) → Tail (48→3)
  Residual skip: output = clamp(input + tail, 0, 1)   (learn correction)
"""

import os, time, math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TR_LOW  = os.path.join(BASE, "datasets/LOL_dataset/our485/low")
TR_HIGH = os.path.join(BASE, "datasets/LOL_dataset/our485/high")
EV_LOW  = os.path.join(BASE, "datasets/LOL_dataset/eval15/low")
EV_HIGH = os.path.join(BASE, "datasets/LOL_dataset/eval15/high")
OUT_DIR = os.path.join(BASE, "results/reflectance_cnn")
CKP_DIR = os.path.join(BASE, "results/reflectance_cnn/checkpoints")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKP_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = 64
BATCH_SIZE = 128
N_EPOCHS   = 200
LR         = 2e-4
LAMBDA_SSIM= 0.2

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")
    print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# ══════════════════════════════════════════════════════════════════════════════
# DECOMPOSITION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tv_illum(img_f, weight=0.08):
    I = np.max(img_f, axis=2).astype(np.float32)
    I = denoise_tv_chambolle(I, weight=weight, channel_axis=None)
    return np.expand_dims(np.clip(I, 1e-4, 1.0), 2)


def fast_decompose_R(img_f):
    """TV illumination → naive divide. Fast enough for 485 images."""
    I = _tv_illum(img_f)
    R = np.clip(img_f / (I + 1e-6), 0, 1).astype(np.float32)
    return R


def a5_decompose_R(img_f, lambda_I=0.15, lambda_R=0.05, n_iter=50, lr=0.05):
    """Log-TV Retinex (A5) + BayesShrink wavelet — best classical method."""
    S    = np.clip(img_f, 1e-4, 1.0).astype(np.float32)
    logS = np.log(S)
    I0   = np.clip(denoise_tv_chambolle(np.max(img_f, axis=2).astype(np.float32),
                                        weight=0.1, channel_axis=None), 1e-4, 1.0)
    logI = np.log(I0); logI_init = logI.copy(); eps_tv = 1e-4
    for _ in range(n_iter):
        grad  = 2.0*(logI - logI_init)
        dx    = np.pad(np.diff(logI,axis=1),((0,0),(0,1)),'constant')
        dy    = np.pad(np.diff(logI,axis=0),((0,1),(0,0)),'constant')
        dxb   = np.pad(np.diff(logI,axis=1),((0,0),(1,0)),'constant')
        dyb   = np.pad(np.diff(logI,axis=0),((1,0),(0,0)),'constant')
        nf    = np.sqrt(dx**2+dy**2+eps_tv); nb=np.sqrt(dxb**2+dyb**2+eps_tv)
        grad += lambda_I*(dx/nf-dxb/nb+dy/nf-dyb/nb)
        logR  = logS - logI[...,np.newaxis]
        dRx   = np.pad(np.diff(logR,axis=1),((0,0),(0,1),(0,0)),'constant')
        dRy   = np.pad(np.diff(logR,axis=0),((0,1),(0,0),(0,0)),'constant')
        dRxb  = np.pad(np.diff(logR,axis=1),((0,0),(1,0),(0,0)),'constant')
        dRyb  = np.pad(np.diff(logR,axis=0),((1,0),(0,0),(0,0)),'constant')
        nRf   = np.sqrt((dRx**2+dRy**2).sum(2,keepdims=True)+eps_tv)
        nRb   = np.sqrt((dRxb**2+dRyb**2).sum(2,keepdims=True)+eps_tv)
        grad += lambda_R*(-(dRx/nRf-dRxb/nRb+dRy/nRf-dRyb/nRb)).sum(2)
        logI -= lr*grad; logI = np.clip(logI, np.log(1e-4), 0.0)
    I_fin = np.exp(logI)[...,np.newaxis]
    R_fin = np.clip(img_f/(I_fin+1e-6), 0, 1)
    return denoise_wavelet(R_fin.astype(np.float64), channel_axis=2,
                           method="BayesShrink", mode="soft",
                           rescale_sigma=True).astype(np.float32)


def guided_decompose_Rhigh(high_f):
    """R_high: guided-filter illumination + BayesShrink wavelet (ground truth)."""
    from skimage.restoration import denoise_wavelet as dw
    I  = _tv_illum(high_f, weight=0.08)
    R  = np.clip(high_f / (I + 1e-6), 0, 1)
    return dw(R.astype(np.float64), channel_axis=2,
              method="BayesShrink", mode="soft",
              rescale_sigma=True).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PREPARE REFLECTANCE PAIRS
# ══════════════════════════════════════════════════════════════════════════════

def prepare_pairs(low_dir, high_dir, cache_dir, decomp_fn_low=fast_decompose_R):
    """Load or compute (R_low, R_high) pairs; cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(low_dir) if f.endswith((".png",".jpg")))
    pairs = []
    for i, name in enumerate(names):
        base = os.path.splitext(name)[0]
        p_low  = os.path.join(cache_dir, f"{base}_Rlow.npy")
        p_high = os.path.join(cache_dir, f"{base}_Rhigh.npy")
        if os.path.exists(p_low) and os.path.exists(p_high):
            Rl = np.load(p_low); Rh = np.load(p_high)
        else:
            low_f  = cv2.imread(os.path.join(low_dir,  name)).astype(np.float32)/255.0
            high_f = cv2.imread(os.path.join(high_dir, name)).astype(np.float32)/255.0
            Rl = decomp_fn_low(low_f)
            Rh = guided_decompose_Rhigh(high_f)
            np.save(p_low,  Rl); np.save(p_high, Rh)
            if (i+1) % 50 == 0:
                print(f"  decomposed {i+1}/{len(names)}")
        pairs.append((Rl, Rh))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# PATCH DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ReflectancePatchDataset(Dataset):
    def __init__(self, pairs, patch_size=64, patches_per_image=200, augment=True):
        self.pairs   = pairs
        self.ps      = patch_size
        self.ppi     = patches_per_image
        self.augment = augment
        self.n       = len(pairs) * patches_per_image

    def __len__(self): return self.n

    def __getitem__(self, idx):
        img_idx = idx // self.ppi
        Rl, Rh  = self.pairs[img_idx]
        H, W    = Rl.shape[:2]; ps = self.ps

        # Random crop
        r = np.random.randint(0, max(1, H - ps))
        c = np.random.randint(0, max(1, W - ps))
        pl = Rl[r:r+ps, c:c+ps, :]
        ph = Rh[r:r+ps, c:c+ps, :]

        # Augmentation: random flips + 90° rotations
        if self.augment:
            k = np.random.randint(0, 4)
            pl = np.rot90(pl, k).copy(); ph = np.rot90(ph, k).copy()
            if np.random.rand() > 0.5:
                pl = pl[:, ::-1, :].copy(); ph = ph[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                pl = pl[::-1, :, :].copy(); ph = ph[::-1, :, :].copy()

        # HWC → CHW, float32
        pl = torch.from_numpy(pl.transpose(2,0,1).astype(np.float32))
        ph = torch.from_numpy(ph.transpose(2,0,1).astype(np.float32))
        return pl, ph


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — LightRefineNet
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nf),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class LightRefineNet(nn.Module):
    """
    Compact residual network for reflectance refinement.
    Learns the correction  ΔR = R_high − R_low  in a 48-channel feature space.
    ~330 K parameters.
    """
    def __init__(self, nf=48, nb=10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        res  = self.tail(feat)
        return torch.clamp(x + res, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(win=11, sigma=1.5, channels=3):
    g1d = torch.Tensor([math.exp(-(x-win//2)**2/(2*sigma**2)) for x in range(win)])
    g1d /= g1d.sum()
    k2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    kernel = k2d.expand(channels, 1, win, win).contiguous()
    return kernel

_KERNEL = None
def ssim_loss(pred, target, win=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    global _KERNEL
    if _KERNEL is None or _KERNEL.device != pred.device:
        _KERNEL = _gaussian_kernel(win, sigma, pred.shape[1]).to(pred.device)
    ch = pred.shape[1]; pad = win//2
    mu_p  = F.conv2d(pred,   _KERNEL, padding=pad, groups=ch)
    mu_t  = F.conv2d(target, _KERNEL, padding=pad, groups=ch)
    mu_pp = F.conv2d(pred*pred,     _KERNEL, padding=pad, groups=ch)
    mu_tt = F.conv2d(target*target, _KERNEL, padding=pad, groups=ch)
    mu_pt = F.conv2d(pred*target,   _KERNEL, padding=pad, groups=ch)
    sig_p  = mu_pp - mu_p*mu_p
    sig_t  = mu_tt - mu_t*mu_t
    sig_pt = mu_pt - mu_p*mu_t
    num = (2*mu_p*mu_t + C1) * (2*sig_pt + C2)
    den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
    return 1.0 - (num/den).mean()


def combined_loss(pred, target, lam=LAMBDA_SSIM):
    return F.l1_loss(pred, target) + lam * ssim_loss(pred, target)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, epochs, lr, ckp_path):
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_val = float("inf"); best_ep = 0
    hist = {"train": [], "val": []}

    for ep in range(1, epochs+1):
        # Train
        model.train(); tr_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
            out  = model(inp)
            loss = combined_loss(out, tgt)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Validate
        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
                va_loss += combined_loss(model(inp), tgt).item()
        va_loss /= max(len(val_loader), 1)

        sched.step()
        hist["train"].append(tr_loss); hist["val"].append(va_loss)

        if va_loss < best_val:
            best_val = va_loss; best_ep = ep
            torch.save(model.state_dict(), ckp_path)

        if ep % 20 == 0 or ep == 1:
            print(f"  ep {ep:3d}/{epochs}  tr={tr_loss:.5f}  va={va_loss:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  best@{best_ep}")

    return hist


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def infer(model, R_np):
    """Run model on a full H×W×3 reflectance image (numpy float32)."""
    model.eval()
    with torch.no_grad():
        t   = torch.from_numpy(R_np.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        out = model(t).squeeze(0).permute(1,2,0).cpu().numpy()
    return np.clip(out, 0, 1).astype(np.float32)


def metrics_np(est, ref):
    u8 = lambda x: (np.clip(x,0,1)*255).astype(np.uint8)
    p  = calc_psnr(u8(ref), u8(est), data_range=255)
    s  = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    m  = float(np.mean(np.abs(est-ref)))
    return p, s, m


def to_rgb(arr):
    return cv2.cvtColor((np.clip(arr,0,1)*255).astype(np.uint8), cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Phase 1: decompose training data ────────────────────────────────────
    print("\n[Phase 1] Decomposing 485 training pairs ...")
    t0 = time.time()
    cache_tr = os.path.join(OUT_DIR, "cache_train")
    train_pairs = prepare_pairs(TR_LOW, TR_HIGH, cache_tr)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(train_pairs)} pairs)")

    # ── Phase 1b: decompose eval15 (TV for input, R_high for target) ─────────
    print("\n[Phase 1b] Decomposing eval15 ...")
    cache_ev = os.path.join(OUT_DIR, "cache_eval")
    eval_pairs_tv = prepare_pairs(EV_LOW, EV_HIGH, cache_ev)
    eval_names    = sorted(f for f in os.listdir(EV_LOW) if f.endswith((".png",".jpg")))
    print(f"  Done. {len(eval_pairs_tv)} eval pairs")

    # ── Phase 1c: pre-compute A5 decomposition for eval15 ────────────────────
    cache_a5 = os.path.join(OUT_DIR, "cache_eval_a5")
    os.makedirs(cache_a5, exist_ok=True)
    print("\n[Phase 1c] A5 decomposition for eval15 ...")
    eval_R_a5 = []
    for name in eval_names:
        base  = os.path.splitext(name)[0]
        p_a5  = os.path.join(cache_a5, f"{base}_Ra5.npy")
        if os.path.exists(p_a5):
            eval_R_a5.append(np.load(p_a5))
        else:
            low_f = cv2.imread(os.path.join(EV_LOW, name)).astype(np.float32)/255.0
            Ra5   = a5_decompose_R(low_f)
            np.save(p_a5, Ra5)
            eval_R_a5.append(Ra5)
            print(f"  A5: {name}")
    print(f"  Done.")

    # ── Phase 2: Train ───────────────────────────────────────────────────────
    print("\n[Phase 2] Training LightRefineNet ...")

    # Split: last 50 train images as validation
    val_pairs   = train_pairs[-50:]
    tr_pairs    = train_pairs[:-50]

    tr_ds  = ReflectancePatchDataset(tr_pairs,  PATCH_SIZE, patches_per_image=250)
    va_ds  = ReflectancePatchDataset(val_pairs, PATCH_SIZE, patches_per_image=100, augment=False)
    tr_dl  = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    va_dl  = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = LightRefineNet(nf=48, nb=10).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    ckp_path = os.path.join(CKP_DIR, "best_model.pth")
    t1 = time.time()
    hist = train(model, tr_dl, va_dl, N_EPOCHS, LR, ckp_path)
    print(f"\n  Training done in {(time.time()-t1)/60:.1f} min")

    # Load best checkpoint
    model.load_state_dict(torch.load(ckp_path, map_location=DEVICE))
    model.eval()

    # ── Loss curve ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d0d1a"); ax.set_facecolor("#0a0a14")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa")
    ax.plot(hist["train"], color="#5599cc", label="train")
    ax.plot(hist["val"],   color="#00ff88", label="val")
    ax.set_xlabel("Epoch", color="#ccc"); ax.set_ylabel("Loss", color="#ccc")
    ax.set_title("LightRefineNet Training Loss", color="white", fontweight="bold")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curve.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # ── Phase 3: Evaluate ────────────────────────────────────────────────────
    print("\n[Phase 3] Evaluation on eval15 ...")

    acc = {k: [0,0,0] for k in ["A5", "TV+CNN", "A5+CNN"]}
    rows = []

    for i, name in enumerate(eval_names):
        base      = os.path.splitext(name)[0]
        Rl_tv, Rh = eval_pairs_tv[i]       # TV-decomposed low, GT high
        Ra5       = eval_R_a5[i]            # A5-decomposed low

        # Run CNN
        Rcnn_tv  = infer(model, Rl_tv)      # TV → CNN
        Rcnn_a5  = infer(model, Ra5)        # A5 → CNN

        pa5,  sa5,  ma5  = metrics_np(Ra5,      Rh)
        ptv,  stv,  mtv  = metrics_np(Rcnn_tv,  Rh)
        pa5c, sa5c, ma5c = metrics_np(Rcnn_a5,  Rh)

        acc["A5"][0]    += pa5;  acc["A5"][1]    += sa5;  acc["A5"][2]    += ma5
        acc["TV+CNN"][0]+= ptv;  acc["TV+CNN"][1]+= stv;  acc["TV+CNN"][2]+= mtv
        acc["A5+CNN"][0]+= pa5c; acc["A5+CNN"][1]+= sa5c; acc["A5+CNN"][2]+= ma5c

        rows.append((base, Rh, Ra5, Rcnn_tv, Rcnn_a5,
                     pa5, sa5, ptv, stv, pa5c, sa5c))

        # Save best output
        best_R = Rcnn_a5
        cv2.imwrite(os.path.join(OUT_DIR, f"{base}_R_cnn.png"),
                    (best_R*255).astype(np.uint8))

        print(f"  {base:<6}  A5={pa5:.2f}/{sa5:.3f}  "
              f"TV+CNN={ptv:.2f}/{stv:.3f}  A5+CNN={pa5c:.2f}/{sa5c:.3f}")

    n = len(eval_names)
    print(f"\n{'='*70}")
    print(f"  GLOBAL AVERAGE METRICS  (n={n})")
    print(f"{'='*70}")
    print(f"  {'Method':<18}  {'PSNR':>8}  {'SSIM':>8}  {'MAE':>8}  {'vs A5':>8}")
    print(f"  {'-'*56}")
    a5_p = acc['A5'][0]/n
    for key in ["A5", "TV+CNN", "A5+CNN"]:
        p,s,m = acc[key][0]/n, acc[key][1]/n, acc[key][2]/n
        dp    = p - a5_p
        print(f"  {key:<18}  {p:>8.3f}  {s:>8.4f}  {m*100:>7.2f}%  {dp:>+8.3f}")
    print(f"  {'Perfect match':<18}  {'inf':>8}  {'1.0000':>8}  {'0.00%':>8}")
    print(f"\n  SSIM gap:  A5={1-acc['A5'][1]/n:.4f}  "
          f"TV+CNN={1-acc['TV+CNN'][1]/n:.4f}  "
          f"A5+CNN={1-acc['A5+CNN'][1]/n:.4f}")
    print(f"{'='*70}")

    # ── Per-image comparison figure ──────────────────────────────────────────
    print("\n[Saving comparison figures ...]")
    for (base, Rh, Ra5, Rcnn_tv, Rcnn_a5,
         pa5, sa5, ptv, stv, pa5c, sa5c) in rows:

        H, W = Rh.shape[:2]
        cx, cy, cw, ch = W//6, H//6, W//3, H//3

        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor("#0c0c18")
        gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.05,
                                left=0.01, right=0.99, top=0.91, bottom=0.03)
        TKW = dict(color="white", fontsize=8.5, fontweight="bold", pad=4)
        MKW = dict(color="#ffe066", fontsize=7.5, va="top", ha="left")

        panels_r0 = [
            (Rh,       "R_high [GT]",           None),
            (Ra5,      f"A5 only\nPSNR {pa5:.2f} SSIM {sa5:.3f}", None),
            (Rcnn_tv,  f"TV+CNN\nPSNR {ptv:.2f} SSIM {stv:.3f}",  ptv),
            (Rcnn_a5,  f"A5+CNN\nPSNR {pa5c:.2f} SSIM {sa5c:.3f}", pa5c),
            (np.abs(Rcnn_a5 - Rh)*5, "Diff (A5+CNN, GT)×5", None),
        ]
        best_col = 3 if pa5c >= ptv else 2

        for j, (img, title, psnr_val) in enumerate(panels_r0):
            ax = fig.add_subplot(gs[0, j])
            ax.set_facecolor("#0c0c18"); ax.axis("off")
            for sp in ax.spines.values():
                sp.set_edgecolor("#00ff88" if j==best_col else "#333")
                sp.set_linewidth(2.5 if j==best_col else 0.8)
            ax.imshow(to_rgb(img) if j<4 else np.clip(img,0,1)*255//1//255)
            ax.set_title(title, **TKW)
            if j == 4:
                ax.imshow(np.clip(img.mean(2)*1, 0, 1), cmap="inferno")

        for j, (img, title, _) in enumerate(panels_r0):
            ax = fig.add_subplot(gs[1, j])
            ax.set_facecolor("#0c0c18"); ax.axis("off")
            crop = img[cy:cy+ch, cx:cx+cw]
            if j==4:
                ax.imshow(np.clip(crop.mean(2),0,1), cmap="inferno")
            else:
                ax.imshow(to_rgb(crop))
            ax.set_title(f"crop: {title.split(chr(10))[0]}", **TKW)
            if j==best_col:
                for sp in ax.spines.values():
                    sp.set_edgecolor("#00ff88"); sp.set_linewidth(2.5)

        gain = pa5c - pa5
        fig.suptitle(
            f"{base}.png  —  A5={pa5:.2f}dB → A5+CNN={pa5c:.2f}dB "
            f"({'▲' if gain>=0 else '▼'}{abs(gain):.2f}dB)  "
            f"SSIM: {sa5:.3f} → {sa5c:.3f}",
            color="white", fontsize=10, fontweight="bold", y=0.96)
        plt.savefig(os.path.join(OUT_DIR, f"{base}_cnn_compare.png"),
                    dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()

    # ── Global summary bar chart ─────────────────────────────────────────────
    bases  = [r[0] for r in rows]
    x      = np.arange(len(bases))
    w      = 0.28
    p_a5   = [r[5]  for r in rows]
    p_tv   = [r[7]  for r in rows]
    p_a5c  = [r[9]  for r in rows]
    s_a5   = [r[6]  for r in rows]
    s_tv   = [r[8]  for r in rows]
    s_a5c  = [r[10] for r in rows]

    fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(20, 10))
    fig.patch.set_facecolor("#0c0c18")
    for ax,vals_list,ylabel,title in [
        (ax1, [(p_a5,"#ff8855","A5 only"),(p_tv,"#5599cc","TV+CNN"),(p_a5c,"#00ff88","A5+CNN")],
         "PSNR (dB)", "Per-image PSNR vs R_high  (orange=A5, blue=TV+CNN, green=A5+CNN)"),
        (ax2, [(s_a5,"#ff8855","A5 only"),(s_tv,"#5599cc","TV+CNN"),(s_a5c,"#00ff88","A5+CNN")],
         "SSIM", "Per-image SSIM vs R_high"),
    ]:
        ax.set_facecolor("#08080f")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
        for k, (vals, color, label) in enumerate(vals_list):
            ax.bar(x + (k-1)*w, vals, w, color=color, alpha=0.88, label=label)
        ax.set_xticks(x); ax.set_xticklabels(bases, rotation=30, ha="right", fontsize=9, color="#ccc")
        ax.set_ylabel(ylabel, color="#ccc")
        ax.set_title(title, color="white", fontsize=9, fontweight="bold")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)
        ax.axhline(np.mean([v[0] for v in vals_list]), color="#fff", linewidth=0.5, linestyle=":")

    fig.suptitle(
        f"LightRefineNet Results — n={n}  |  "
        f"Avg PSNR: A5={acc['A5'][0]/n:.3f}  TV+CNN={acc['TV+CNN'][0]/n:.3f}  "
        f"A5+CNN={acc['A5+CNN'][0]/n:.3f}dB  |  "
        f"Avg SSIM: A5={acc['A5'][1]/n:.4f}  A5+CNN={acc['A5+CNN'][1]/n:.4f}",
        color="white", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "global_summary_cnn.png"),
                dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}/")
