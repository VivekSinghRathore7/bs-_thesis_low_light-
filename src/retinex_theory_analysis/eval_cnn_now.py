"""
Quick evaluation of the current best_model.pth checkpoint on eval15.
Runs independently from the ongoing training process.
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = "/home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light"
EV_LOW  = os.path.join(ROOT, "datasets/LOL_dataset/eval15/low")
EV_HIGH = os.path.join(ROOT, "datasets/LOL_dataset/eval15/high")
CKP     = os.path.join(ROOT, "results/reflectance_cnn/checkpoints/best_model.pth")
CACHE_A5 = os.path.join(ROOT, "results/reflectance_cnn/cache_eval_a5")
CACHE_TV = os.path.join(ROOT, "results/reflectance_cnn/cache_eval")

DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Model ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(x + self.block(x))


class LightRefineNet(nn.Module):
    def __init__(self, nf=48, nb=10):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3, nf, 3, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf, 3, 3, 1, 1)
    def forward(self, x):
        feat = self.head(x); feat = self.body(feat); res = self.tail(feat)
        return torch.clamp(x + res, 0.0, 1.0)


# ── Load model ─────────────────────────────────────────────────────────────────
model = LightRefineNet(nf=48, nb=10).to(DEVICE)
model.load_state_dict(torch.load(CKP, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {CKP}")
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def infer(R_np):
    with torch.no_grad():
        t   = torch.from_numpy(R_np.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        out = model(t).squeeze(0).permute(1,2,0).cpu().numpy()
    return np.clip(out, 0, 1).astype(np.float32)

def metrics(est, ref):
    u8 = lambda x: (np.clip(x,0,1)*255).astype(np.uint8)
    p = calc_psnr(u8(ref), u8(est), data_range=255)
    s = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    m = float(np.mean(np.abs(est - ref)))
    return p, s, m

def tv_decompose(img, iters=50, lr=0.05, lam=0.15):
    """Fast Log-TV illumination → naive R = S/I (same as training TV decomp)."""
    S    = np.clip(img, 1e-4, 1.0)
    Imax = np.max(S, axis=2, keepdims=True)
    logI = np.log(Imax + 1e-4)
    logS = np.log(S)
    for _ in range(iters):
        R    = logS - logI
        gx   = np.diff(logI, axis=1, append=logI[:, -1:, :])
        gy   = np.diff(logI, axis=0, append=logI[-1:, :, :])
        norm = np.sqrt(gx**2 + gy**2 + 1e-8)
        dx   = gx / norm; dy = gy / norm
        divx = dx - np.concatenate([dx[:, :1, :], dx[:, :-1, :]], axis=1)
        divy = dy - np.concatenate([dy[:1, :, :], dy[:-1, :, :]], axis=0)
        tv_grad = -(divx + divy)
        fid_grad = logI - logS.mean(axis=2, keepdims=True)
        logI -= lr * (fid_grad + lam * tv_grad)
    I = np.exp(logI).repeat(3, axis=2)
    R = np.clip(S / (I + 1e-6), 0, 1)
    return R.astype(np.float32)


# ── Evaluate ───────────────────────────────────────────────────────────────────
eval_names = sorted(f for f in os.listdir(EV_LOW) if f.endswith((".png", ".jpg")))

rows = []
acc_a5    = [0.0, 0.0, 0.0]
acc_tv_cnn= [0.0, 0.0, 0.0]
acc_a5_cnn= [0.0, 0.0, 0.0]

print(f"\n{'Image':<25} {'A5 PSNR':>9} {'A5 SSIM':>8} {'TV+CNN PSNR':>11} {'TV+CNN SSIM':>11} {'A5+CNN PSNR':>11} {'A5+CNN SSIM':>11}")
print("-"*100)

for name in eval_names:
    base = os.path.splitext(name)[0]

    # Ground truth high
    hf  = cv2.imread(os.path.join(EV_HIGH, name))
    if hf is None:
        # try finding by number
        stem = base
        candidates = [f for f in os.listdir(EV_HIGH) if os.path.splitext(f)[0] == stem]
        if candidates:
            hf = cv2.imread(os.path.join(EV_HIGH, candidates[0]))
    if hf is None:
        print(f"  SKIP {name}: high not found")
        continue
    hf = hf.astype(np.float32) / 255.0

    # GT reflectance (TV-decomposed high, or cache)
    p_rh = os.path.join(CACHE_TV, f"{base}_Rhigh.npy")
    if os.path.exists(p_rh):
        Rh = np.load(p_rh)
    else:
        Rh = tv_decompose(hf, iters=30, lam=0.05)

    # A5 low reflectance from cache
    p_a5 = os.path.join(CACHE_A5, f"{base}_Ra5.npy")
    if os.path.exists(p_a5):
        Ra5 = np.load(p_a5)
    else:
        print(f"  SKIP {name}: A5 cache missing")
        continue

    # TV low reflectance from cache
    p_tv = os.path.join(CACHE_TV, f"{base}_Rlow.npy")
    if os.path.exists(p_tv):
        Rl_tv = np.load(p_tv)
    else:
        lf = cv2.imread(os.path.join(EV_LOW, name)).astype(np.float32) / 255.0
        Rl_tv = tv_decompose(lf)

    # CNN inference
    Rcnn_tv  = infer(Rl_tv)
    Rcnn_a5  = infer(Ra5)

    pa5,   sa5,   ma5   = metrics(Ra5,     Rh)
    ptvcnn,stvcnn,mtvcnn= metrics(Rcnn_tv, Rh)
    pa5cnn,sa5cnn,ma5cnn= metrics(Rcnn_a5, Rh)

    for i, v in enumerate([pa5, sa5, ma5]):         acc_a5[i]     += v
    for i, v in enumerate([ptvcnn, stvcnn, mtvcnn]): acc_tv_cnn[i] += v
    for i, v in enumerate([pa5cnn, sa5cnn, ma5cnn]): acc_a5_cnn[i] += v

    print(f"  {name:<23} {pa5:9.3f} {sa5:8.4f} {ptvcnn:11.3f} {stvcnn:11.4f} {pa5cnn:11.3f} {sa5cnn:11.4f}")
    rows.append((name, pa5, sa5, ptvcnn, stvcnn, pa5cnn, sa5cnn))

n = len(rows)
print("-"*100)
print(f"  {'AVG':<23} {acc_a5[0]/n:9.3f} {acc_a5[1]/n:8.4f} "
      f"{acc_tv_cnn[0]/n:11.3f} {acc_tv_cnn[1]/n:11.4f} "
      f"{acc_a5_cnn[0]/n:11.3f} {acc_a5_cnn[1]/n:11.4f}")

print(f"\n{'='*60}")
print(f"SUMMARY  ({n} images)")
print(f"  A5 (classical best):    PSNR={acc_a5[0]/n:.3f}  SSIM={acc_a5[1]/n:.4f}  MAE={acc_a5[2]/n*100:.2f}%")
print(f"  TV+CNN (fast):          PSNR={acc_tv_cnn[0]/n:.3f}  SSIM={acc_tv_cnn[1]/n:.4f}  MAE={acc_tv_cnn[2]/n*100:.2f}%")
print(f"  A5+CNN (best pipeline): PSNR={acc_a5_cnn[0]/n:.3f}  SSIM={acc_a5_cnn[1]/n:.4f}  MAE={acc_a5_cnn[2]/n*100:.2f}%")
delta_psnr = acc_a5_cnn[0]/n - acc_a5[0]/n
delta_ssim = acc_a5_cnn[1]/n - acc_a5[1]/n
print(f"\n  CNN gain over A5:  ΔPSNR={delta_psnr:+.3f} dB   ΔSSIM={delta_ssim:+.4f}")
print(f"{'='*60}")
