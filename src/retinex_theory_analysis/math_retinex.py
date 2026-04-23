"""
math_retinex.py  (v2 — fully fixed)
=====================================
Mathematically rigorous techniques to close the reflectance gap.

Variants
--------
A5   Log-domain TV Retinex (gradient descent) — previous best [baseline]
M0   VST-Retinex: Anscombe on raw S → wavelet-denoise S → TV-illumination decompose
       (physically correct: VST applied to photon-count domain before any division)
M1   Pure MSR: multi-scale log-Retinex, 3 Gaussian scales, robust normalisation
       log R_c = Σ_s w_s [log S_c − log(G_σs * S_c)], then percentile-normalise
M2   L₀-Illumination + BayesShrink wavelet
       min_I ||I−I₀||² + λ·||∇I||₀  (FFT half-quadratic splitting, Xu 2011)
M3   Tikhonov-Regularised Retinex
       min_R ||I·R − S||² + λ·||∇R||²  (closed-form via FFT, per channel)
M4   WNNM-lite (fixed): weighted nuclear norm minimisation on reflectance patches
M5   Log-TV Retinex via ADMM  (proximal TV sub-problems; better than GD)
M6   M2 + M4:  L₀ illumination  →  WNNM-lite denoising  (best-of-stack)
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle
import scipy.ndimage as ndimage

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
low_dir   = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
high_dir  = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/high")
out_dir   = os.path.join(BASE_DIR, "results/reflectance_math")
os.makedirs(out_dir, exist_ok=True)

VARIANT_NAMES = [
    "A5  LogTV-GD [prev best]",
    "M0  VST-Retinex",
    "M1  PureMSR",
    "M2  L0+Wavelet",
    "M3  Tikhonov-FFT",
    "M4  WNNM-lite",
    "M5  LogTV-ADMM",
    "M6  L0+WNNM",
]
N_VARIANTS = len(VARIANT_NAMES)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _box(img, r):
    return cv2.blur(img.astype(np.float32), (2*r+1, 2*r+1))


def _guided_illum(img, r=16, eps=1e-3):
    """Self-guided filter illumination on max-channel."""
    I     = np.max(img, axis=2).astype(np.float32)
    guide = cv2.cvtColor((img*255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    mI=_box(I,r); mg=_box(guide,r)
    mIg=_box(I*guide,r); mgg=_box(guide*guide,r)
    a = (mIg-mI*mg)/(mgg-mg*mg+eps)
    b = mI-a*mg
    return np.expand_dims(np.clip(_box(a,r)*guide+_box(b,r), 1e-4, 1.0), 2)


def _tv_illum(img, weight=0.08):
    I = np.max(img, axis=2).astype(np.float32)
    I_tv = denoise_tv_chambolle(I, weight=weight, channel_axis=None)
    return np.expand_dims(np.clip(I_tv, 1e-4, 1.0).astype(np.float32), 2)


def metrics(est, ref):
    u8 = lambda x: (np.clip(x,0,1)*255).astype(np.uint8)
    p  = calc_psnr(u8(ref), u8(est), data_range=255)
    s  = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    return p, s


def to_rgb(arr):
    return cv2.cvtColor((np.clip(arr,0,1)*255).astype(np.uint8), cv2.COLOR_BGR2RGB)


# ══════════════════════════════════════════════════════════════════════════════
# A5 — Log-domain TV Retinex (gradient descent)  [prev best]
# ══════════════════════════════════════════════════════════════════════════════

def log_tv_retinex_gd(img, lambda_I=0.15, lambda_R=0.05, n_iter=50, lr=0.05):
    S    = np.clip(img, 1e-4, 1.0).astype(np.float32)
    logS = np.log(S)
    I0   = np.clip(denoise_tv_chambolle(np.max(img,axis=2).astype(np.float32),
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
    R_fin = np.clip(img/(I_fin+1e-6), 0, 1)
    return denoise_wavelet(R_fin.astype(np.float64), channel_axis=2,
                           method="BayesShrink", mode="soft",
                           rescale_sigma=True).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# M0 — VST-Retinex  (Anscombe on raw S, then decompose)
# ══════════════════════════════════════════════════════════════════════════════

def anscombe_fwd(x, N=255.0):
    return 2.0 * np.sqrt(np.maximum(x*N + 3.0/8.0, 0.0))

def anscombe_inv(y, N=255.0):
    y2 = np.maximum(y, 1e-8)
    return np.clip((y2/2.0)**2/N - 3.0/(8.0*N) - 1.0/(8.0*N*(y2/2.0)**2+1e-12), 0, 1)

def vst_retinex(img):
    """
    1. Anscombe VST on each channel of S  →  noise becomes ~N(0,1)
    2. BayesShrink wavelet denoising of the VST image
    3. Inverse VST  →  denoised S_clean
    4. TV illumination decomposition on S_clean
    Returns R in [0,1].
    """
    S = np.clip(img, 0, 1).astype(np.float32)
    T = anscombe_fwd(S)                                # H×W×3, Gaussian noise
    T_dn = denoise_wavelet(T.astype(np.float64),
                           channel_axis=2,
                           method="BayesShrink",
                           mode="soft",
                           rescale_sigma=True).astype(np.float32)
    S_clean = np.clip(anscombe_inv(T_dn), 1e-6, 1.0)  # denoised image

    # Decompose denoised image
    I = _tv_illum(S_clean, weight=0.08)
    R = np.clip(S_clean / (I + 1e-6), 0, 1)
    return R.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# M1 — Pure Multi-Scale Retinex
# ══════════════════════════════════════════════════════════════════════════════

def pure_msr(img, sigmas=(15, 80, 250)):
    """
    Multi-Scale Retinex reflectance estimate:
      log R_c(x) = Σ_s w_s · [log S_c(x) − log(G_σs * S_c)(x)]

    Result is the log of the reflectance averaged across scales.
    Per-channel percentile normalisation maps it to [0,1].
    """
    S = np.clip(img, 1e-6, 1.0).astype(np.float64)
    w = 1.0 / len(sigmas)
    log_R = np.zeros_like(S)

    for sigma in sigmas:
        ksize = int(6*sigma + 1) | 1
        for c in range(3):
            blur = cv2.GaussianBlur(S[..., c].astype(np.float32),
                                    (ksize, ksize), sigma).astype(np.float64)
            log_R[..., c] += w * (np.log(S[..., c]) - np.log(np.maximum(blur, 1e-6)))

    R = np.exp(log_R).astype(np.float32)

    # Robust per-channel normalisation (1st–99th percentile)
    for c in range(3):
        lo  = np.percentile(R[..., c], 1)
        hi  = np.percentile(R[..., c], 99)
        if hi > lo + 1e-6:
            R[..., c] = (R[..., c] - lo) / (hi - lo)
        else:
            R[..., c] = 0.5

    return np.clip(R, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# M2 — L₀ Gradient Illumination  (Xu et al. 2011)
# ══════════════════════════════════════════════════════════════════════════════

def illum_l0(img, lam=2e-2, beta0=2e-1, beta_max=1e5, kappa=2.0):
    """min_I ||I−I₀||² + λ·||∇I||₀  via half-quadratic splitting + FFT."""
    I0 = np.max(img, axis=2).astype(np.float32)
    H, W = I0.shape
    kx = np.zeros((H,W),np.float32); kx[0,0]=-1; kx[0,W-1 if W>1 else 0]=1
    ky = np.zeros((H,W),np.float32); ky[0,0]=-1; ky[H-1 if H>1 else 0,0]=1
    FKx = np.fft.fft2(kx); FKy = np.fft.fft2(ky)
    Den = np.abs(FKx)**2 + np.abs(FKy)**2
    FI0 = np.fft.fft2(I0)
    I   = I0.copy(); beta = beta0
    while beta < beta_max:
        gx = np.roll(I,-1,axis=1)-I; gy = np.roll(I,-1,axis=0)-I
        mag2 = gx**2 + gy**2; thr = lam/beta
        h = np.where(mag2>thr, gx, 0.0); v = np.where(mag2>thr, gy, 0.0)
        dxTh = h - np.roll(h,1,axis=1); dyTv = v - np.roll(v,1,axis=0)
        rhs  = FI0 + beta*(np.fft.fft2(dxTh)+np.fft.fft2(dyTv))
        I    = np.real(np.fft.ifft2(rhs/(2.0+beta*Den))).astype(np.float32)
        I    = np.clip(I, 1e-4, 1.0); beta *= kappa
    return np.expand_dims(I, 2)


def pipeline_m2(img):
    I = illum_l0(img)
    R = np.clip(img/(I+1e-6), 0, 1)
    return denoise_wavelet(R.astype(np.float64), channel_axis=2,
                           method="BayesShrink", mode="soft",
                           rescale_sigma=True).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# M3 — Tikhonov-Regularised Retinex  (FFT closed-form per channel)
# ══════════════════════════════════════════════════════════════════════════════

def tikhonov_retinex(img, lam=0.05):
    """
    Solve  min_R ||I·R_c − S_c||² + λ·||∇R_c||²  per channel.

    With spatially-uniform approximation  I ≈ I_mean:
      R_c = IFFT[ I_mean · FFT[S_c] / (I_mean² + λ·|k|²) ]

    For spatially-varying I we run a few gradient-descent steps.
    """
    I      = illum_l0(img)          # L0 illumination H×W×1
    I_sq   = I**2                   # H×W×1
    H, W   = img.shape[:2]

    # Build Laplacian spectrum (2 - 2cosω  for each axis)
    freq_y = np.fft.fftfreq(H, d=1.0/(2*np.pi))
    freq_x = np.fft.fftfreq(W, d=1.0/(2*np.pi))
    FX, FY = np.meshgrid(freq_x, freq_y)
    L_spec = (2 - 2*np.cos(FX)) + (2 - 2*np.cos(FY))  # |k|² approximation

    R = np.empty_like(img)
    for c in range(3):
        Sc  = img[..., c].astype(np.float32)
        Ic  = I[..., 0]
        # Wiener-Tikhonov in spatially-averaged sense
        I_avg = float(Ic.mean())
        num  = I_avg * np.fft.fft2(Sc)
        den  = I_avg**2 + lam * L_spec
        Rc   = np.real(np.fft.ifft2(num / den)).astype(np.float32)

        # Refinement: 10 gradient-descent steps with true spatially-varying I
        Rc = np.clip(Rc, 0, 1)
        lr = 0.1
        for _ in range(10):
            resid    = Ic * Rc - Sc
            data_g   = Ic * resid
            smooth_g = -lam * ndimage.laplace(Rc).astype(np.float32)
            Rc -= lr * (data_g + smooth_g)
            Rc = np.clip(Rc, 0, 1)

        R[..., c] = Rc

    return R.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# M4 — WNNM-lite  (fixed weights bug)
# ══════════════════════════════════════════════════════════════════════════════

def wnnm_lite(R, patch_size=6, ref_step=6, search_half=12, k=8):
    """
    Weighted Nuclear Norm Minimisation (simplified):
    1. Group k non-locally similar patches per reference position.
    2. SVD of the patch matrix (p²  ×  k).
    3. Weighted soft-threshold:  σ̂_i = max(σ_i − C·σ_n²/(σ_i+ε), 0)
    4. Reconstruct and overlap-average.

    FIX vs v1: weights accumulated ONCE per candidate (not per channel).
    """
    p   = patch_size
    ps  = p * p
    H, W = R.shape[:2]

    # Estimate noise from high-frequency residuals
    lap     = cv2.Laplacian((R*255).mean(axis=2).astype(np.float32), cv2.CV_32F)/255.0
    sigma_n = float(np.clip(np.std(lap)/0.6745, 0.01, 0.15))
    C       = 2.0 * np.sqrt(k) * sigma_n**2

    R_out   = np.zeros_like(R)
    weights = np.zeros((H, W), dtype=np.float32)

    ref_rows = range(0, H - p + 1, ref_step)
    ref_cols = range(0, W - p + 1, ref_step)

    for rr in ref_rows:
        for rc in ref_cols:
            ref_flat = R[rr:rr+p, rc:rc+p, :].reshape(ps, 3)   # ps × 3

            # Candidate positions
            r_lo = max(0, rr - search_half); r_hi = min(H-p+1, rr+search_half+1)
            c_lo = max(0, rc - search_half); c_hi = min(W-p+1, rc+search_half+1)
            cand_pos = [(cr, cc)
                        for cr in range(r_lo, r_hi, 2)
                        for cc in range(c_lo, c_hi, 2)]
            if len(cand_pos) < k:
                continue

            patches = np.array([R[cr:cr+p, cc:cc+p, :].reshape(ps, 3)
                                 for cr, cc in cand_pos])   # N × ps × 3
            dists   = ((patches - ref_flat[np.newaxis])**2).sum(axis=(1,2))
            idx     = np.argsort(dists)[:k]
            sel_pos = [cand_pos[i] for i in idx]

            # ── Per-channel SVD ──────────────────────────────────────────────
            for c in range(3):
                M        = patches[idx, :, c].T      # ps × k
                U, s, Vt = np.linalg.svd(M, full_matrices=False)
                w_sv     = C / (s + 1e-8)
                s_hat    = np.maximum(s - w_sv, 0.0)
                M_hat    = (U * s_hat) @ Vt          # ps × k
                for j, (rj, cj) in enumerate(sel_pos):
                    R_out[rj:rj+p, cj:cj+p, c] += M_hat[:, j].reshape(p, p)

            # ── Weights: once per candidate (NOT per channel) ────────────────
            for rj, cj in sel_pos:
                weights[rj:rj+p, cj:cj+p] += 1.0

    wgt   = np.maximum(weights, 1.0)[..., np.newaxis]
    covered = (weights > 0)[..., np.newaxis]
    R_agg = np.where(covered, R_out / wgt, R)
    return np.clip(R_agg, 0, 1).astype(np.float32)


def pipeline_m4(img):
    I = _guided_illum(img)
    R = np.clip(img/(I+1e-6), 0, 1)
    return wnnm_lite(R)


# ══════════════════════════════════════════════════════════════════════════════
# M5 — Log-TV Retinex via ADMM  (proximal TV sub-problems)
# ══════════════════════════════════════════════════════════════════════════════

def log_tv_admm(img, lambda_I=0.15, lambda_R=0.05, n_iter=25):
    """
    Minimise  ‖logI + logR − logS‖² + λ_I·TV(logI) + λ_R·TV(logR)

    ADMM splitting (Gauss-Seidel alternating minimisation):
      logI ← prox_{λ_I·TV} ( logS − logR )  →  TV-denoise of (logS − logR)
      logR ← prox_{λ_R·TV} ( logS − logI )  →  TV-denoise per channel

    Each prox step is one call to denoise_tv_chambolle.
    Converges faster and more stably than gradient descent.
    """
    S    = np.clip(img, 1e-4, 1.0).astype(np.float32)
    logS = np.log(S)

    # Initialise logI from TV-smoothed max-channel
    I0   = np.clip(denoise_tv_chambolle(np.max(img,axis=2).astype(np.float32),
                                        weight=0.1, channel_axis=None), 1e-4, 1.0)
    logI = np.log(I0)
    logR = logS - logI[..., np.newaxis]
    logR = np.clip(logR, np.log(1e-4), 0.0)

    for _ in range(n_iter):
        # logI update: TV-proximal on mean of (logS − logR)
        rhs_I = (logS - logR).mean(axis=2).astype(np.float32)
        logI  = denoise_tv_chambolle(rhs_I, weight=lambda_I, channel_axis=None)
        logI  = np.clip(logI.astype(np.float32), np.log(1e-4), 0.0)

        # logR update: TV-proximal per channel on (logS − logI)
        rhs_R = (logS - logI[..., np.newaxis]).astype(np.float32)
        logR  = denoise_tv_chambolle(rhs_R, weight=lambda_R, channel_axis=2)
        logR  = np.clip(logR.astype(np.float32), np.log(1e-4), 0.0)

    R = np.exp(logR).astype(np.float32)
    # Final BayesShrink wavelet pass
    R = denoise_wavelet(R.astype(np.float64), channel_axis=2,
                        method="BayesShrink", mode="soft",
                        rescale_sigma=True).astype(np.float32)
    return np.clip(R, 0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# M6 — L₀ Illumination + WNNM-lite
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_m6(img):
    I = illum_l0(img)
    R = np.clip(img/(I+1e-6), 0, 1)
    return wnnm_lite(R)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def build_variants(img):
    results = []
    for tag, fn in [
        ("A5(LogTV-GD)",     lambda: log_tv_retinex_gd(img)),
        ("M0(VST-Retinex)",  lambda: vst_retinex(img)),
        ("M1(PureMSR)",      lambda: pure_msr(img)),
        ("M2(L0+Wav)",       lambda: pipeline_m2(img)),
        ("M3(Tikhonov)",     lambda: tikhonov_retinex(img)),
        ("M4(WNNM)",         lambda: pipeline_m4(img)),
        ("M5(LogTV-ADMM)",   lambda: log_tv_admm(img)),
        ("M6(L0+WNNM)",      lambda: pipeline_m6(img)),
    ]:
        print(f"  {tag}...", end=" ", flush=True)
        results.append(fn())
    print("done")
    return list(zip(VARIANT_NAMES, results))


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx):
    n = len(variants); ncols = n+2
    fig = plt.figure(figsize=(34, 14))
    fig.patch.set_facecolor("#0c0c18")
    gs  = gridspec.GridSpec(3, ncols, figure=fig,
                            hspace=0.42, wspace=0.04,
                            left=0.01, right=0.99, top=0.91, bottom=0.03)
    TKW = dict(color="white", fontsize=6.5, fontweight="bold", pad=3)
    MKW = dict(color="#ffe066", fontsize=6, va="top", ha="left")
    H, W = low.shape[:2]; cx,cy,cw,ch = W//6,H//6,W//3,H//3

    def mk(r,c):
        ax=fig.add_subplot(gs[r,c]); ax.set_facecolor("#0c0c18")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.axis("off"); return ax

    mk(0,0).imshow(to_rgb(low));    fig.axes[-1].set_title("Low input",**TKW)
    mk(0,1).imshow(to_rgb(R_high)); fig.axes[-1].set_title("R_high [GT]",**TKW)
    for j,(vname,R) in enumerate(variants):
        ax=mk(0,j+2); ax.imshow(to_rgb(R))
        bc="#00ff88" if j==best_idx else ("#ff8855" if j==0 else "#333")
        lw=2.5 if j in (0,best_idx) else 0.8
        for sp in ax.spines.values(): sp.set_edgecolor(bc); sp.set_linewidth(lw)
        short=vname.split(" ",1)[1]; star=" ★" if j==best_idx else ""
        ax.set_title(f"{short}{star}",**TKW)
        ax.text(0.01,0.99,f"PSNR {all_psnr[j]:.2f}\nSSIM {all_ssim[j]:.4f}",
                transform=ax.transAxes,**MKW)
    mk(1,0).imshow(to_rgb(low[cy:cy+ch,cx:cx+cw]));    fig.axes[-1].set_title("Low crop",**TKW)
    mk(1,1).imshow(to_rgb(R_high[cy:cy+ch,cx:cx+cw])); fig.axes[-1].set_title("GT crop",**TKW)
    for j,(vname,R) in enumerate(variants):
        ax=mk(1,j+2); ax.imshow(to_rgb(R[cy:cy+ch,cx:cx+cw]))
        ax.set_title(f"{vname.split(' ',1)[1]} crop",**TKW)
        if j==best_idx:
            for sp in ax.spines.values(): sp.set_edgecolor("#00ff88"); sp.set_linewidth(2.5)
    ax_p=fig.add_subplot(gs[2,:n//2+1]); ax_s=fig.add_subplot(gs[2,n//2+1:n+1])
    for ax in [ax_p,ax_s]:
        ax.set_facecolor("#08080f")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")
        ax.tick_params(colors="#aaa")
    sn=[v.split(" ",1)[1] for v,_ in variants]
    cs=["#ff8855" if j==0 else ("#00ff88" if j==best_idx else "#5599cc") for j in range(n)]
    ax_p.bar(range(n),all_psnr,color=cs,alpha=0.85)
    ax_p.set_xticks(range(n)); ax_p.set_xticklabels(sn,rotation=35,ha="right",fontsize=6.5,color="#ccc")
    ax_p.set_ylabel("PSNR",color="#ccc"); ax_p.set_title("PSNR vs R_high",**TKW)
    for i,v in enumerate(all_psnr): ax_p.text(i,v+.05,f"{v:.2f}",ha="center",fontsize=5.5,color="#fff")
    ax_s.bar(range(n),all_ssim,color=cs,alpha=0.85)
    ax_s.set_xticks(range(n)); ax_s.set_xticklabels(sn,rotation=35,ha="right",fontsize=6.5,color="#ccc")
    ax_s.set_ylabel("SSIM",color="#ccc"); ax_s.set_title("SSIM vs R_high",**TKW); ax_s.set_ylim(0,1.05)
    for i,v in enumerate(all_ssim): ax_s.text(i,v+.005,f"{v:.4f}",ha="center",fontsize=5.5,color="#fff")
    _,Rb=variants[best_idx]; diff=np.abs(Rb-R_high).mean(axis=2)
    ax_d=fig.add_subplot(gs[2,n+1]); ax_d.axis("off")
    ax_d.imshow(np.clip(diff*5,0,1),cmap="inferno")
    ax_d.set_title("Diff(best,GT)×5",color="white",fontsize=6.5,pad=3)
    bn=VARIANT_NAMES[best_idx].split(" ",1)[1]
    fig.suptitle(f"Math-Retinex — {img_name}  Best: {bn}  "
                 f"[PSNR {all_psnr[best_idx]:.2f}dB | SSIM {all_ssim[best_idx]:.4f}]  "
                 f"(orange=prev A5)",
                 color="white",fontsize=9.5,fontweight="bold",y=0.96)
    base=os.path.splitext(img_name)[0]
    plt.savefig(os.path.join(out_dir,f"{base}_math.png"),
                dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

img_names   = sorted(f for f in os.listdir(low_dir) if f.endswith((".png",".jpg")))
global_psnr = np.zeros(N_VARIANTS)
global_ssim = np.zeros(N_VARIANTS)
count = 0

print(f"\nProcessing {len(img_names)} images  —  {N_VARIANTS} variants each\n")

for img_name in img_names:
    base = os.path.splitext(img_name)[0]
    print(f"[{base}]")
    low  = cv2.imread(os.path.join(low_dir,  img_name)).astype(np.float32)/255.0
    high = cv2.imread(os.path.join(high_dir, img_name)).astype(np.float32)/255.0

    # Consistent R_high definition (same as advanced_reflectance.py)
    I_high = _guided_illum(high)
    R_high = denoise_wavelet(np.clip(high/(I_high+1e-6),0,1).astype(np.float64),
                             channel_axis=2, method="BayesShrink",
                             mode="soft", rescale_sigma=True).astype(np.float32)

    variants  = build_variants(low)
    all_psnr  = []; all_ssim = []
    for _, R in variants:
        p, s = metrics(R, R_high); all_psnr.append(p); all_ssim.append(s)

    best_idx = int(np.argmax(all_psnr))
    global_psnr += np.array(all_psnr); global_ssim += np.array(all_ssim); count += 1

    _, R_best = variants[best_idx]
    cv2.imwrite(os.path.join(out_dir, f"{base}_R_math_best.png"),
                (R_best*255).astype(np.uint8))
    save_comparison(img_name, low, R_high, variants, all_psnr, all_ssim, best_idx)
    print(f"  best={VARIANT_NAMES[best_idx].split(' ',1)[1]}  "
          f"PSNR={all_psnr[best_idx]:.2f}  SSIM={all_ssim[best_idx]:.4f}\n")

# ── Global summary ─────────────────────────────────────────────────────────────
avg_psnr = global_psnr/count; avg_ssim = global_ssim/count
ob = int(np.argmax(avg_psnr)); a5p = avg_psnr[0]

print(f"\n{'='*72}")
print(f"  GLOBAL AVERAGE METRICS  (n={count})")
print(f"{'='*72}")
print(f"  {'Variant':<38}  {'PSNR':>8}  {'SSIM':>8}  {'vs A5':>7}")
print(f"  {'-'*64}")
for j, name in enumerate(VARIANT_NAMES):
    star=" ★" if j==ob else "  "; tag=" [prev]" if j==0 else ""
    print(f"  {star}{name:<36}  {avg_psnr[j]:>8.3f}  {avg_ssim[j]:>8.4f}  {avg_psnr[j]-a5p:>+7.3f}{tag}")
print(f"\n  Best: {VARIANT_NAMES[ob]}")
print(f"  PSNR={avg_psnr[ob]:.3f} dB  SSIM={avg_ssim[ob]:.4f}")
print(f"  Gap to perfect: SSIM gap={1-avg_ssim[ob]:.4f}")
print(f"{'='*72}")

# ── Summary figure ─────────────────────────────────────────────────────────────
sn  = [v.split(" ",1)[1] for v in VARIANT_NAMES]
cs  = ["#ff8855" if i==0 else ("#00ff88" if i==ob else "#5599cc") for i in range(N_VARIANTS)]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,5))
fig.patch.set_facecolor("#0c0c18")
for ax,vals,ylabel,title in [
    (ax1,avg_psnr,"PSNR (dB)","Avg PSNR vs GT Reflectance (orange=prev A5, green=new best)"),
    (ax2,avg_ssim,"SSIM",     "Avg SSIM vs GT Reflectance"),
]:
    ax.set_facecolor("#08080f")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa")
    ax.bar(range(N_VARIANTS),vals,color=cs,alpha=0.88)
    ax.set_xticks(range(N_VARIANTS))
    ax.set_xticklabels(sn,rotation=40,ha="right",fontsize=8,color="#ccc")
    ax.set_ylabel(ylabel,color="#ccc"); ax.set_title(title,color="white",fontsize=9,fontweight="bold")
    for i,v in enumerate(vals):
        ax.text(i,v+max(vals)*.004,f"{v:.3f}",ha="center",fontsize=7,color="#fff")
fig.suptitle(f"Math-Retinex Global Summary (n={count})  best={VARIANT_NAMES[ob].split(' ',1)[1]}",
             color="white",fontsize=10,fontweight="bold",y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir,"global_summary_math.png"),
            dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close()
print(f"\nAll outputs saved to: {out_dir}/")
