"""
Generate visual comparison images for reflectance enhancement results.
Shows: Low Input | A5 Classical | TV+CNN (best) | Ground Truth
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim

ROOT     = "/home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light"
EV_LOW   = os.path.join(ROOT, "datasets/LOL_dataset/eval15/low")
EV_HIGH  = os.path.join(ROOT, "datasets/LOL_dataset/eval15/high")
CKP      = os.path.join(ROOT, "results/reflectance_cnn/checkpoints/best_model.pth")
CACHE_A5 = os.path.join(ROOT, "results/reflectance_cnn/cache_eval_a5")
CACHE_TV = os.path.join(ROOT, "results/reflectance_cnn/cache_eval")
OUT_DIR  = os.path.join(ROOT, "results/reflectance_cnn/final_images")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        return torch.clamp(x + self.tail(self.body(self.head(x))), 0.0, 1.0)

model = LightRefineNet().to(DEVICE)
model.load_state_dict(torch.load(CKP, map_location=DEVICE))
model.eval()
print(f"Checkpoint loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Helpers ────────────────────────────────────────────────────────────────────
def infer(R_np):
    with torch.no_grad():
        t = torch.from_numpy(R_np.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        return model(t).squeeze(0).permute(1,2,0).cpu().numpy().clip(0,1).astype(np.float32)

def bgr2rgb(img_bgr):
    return cv2.cvtColor((img_bgr*255).astype(np.uint8), cv2.COLOR_BGR2RGB)

def metrics(est, ref):
    u8 = lambda x: (np.clip(x,0,1)*255).astype(np.uint8)
    p = calc_psnr(u8(ref), u8(est), data_range=255)
    s = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    return p, s

# ── Per-image comparison ───────────────────────────────────────────────────────
eval_names = sorted(f for f in os.listdir(EV_LOW) if f.endswith((".png",".jpg")))

all_results = []

print(f"\nGenerating per-image comparisons...")
for name in eval_names:
    base = os.path.splitext(name)[0]

    low_bgr  = cv2.imread(os.path.join(EV_LOW,  name)).astype(np.float32)/255.0
    high_bgr = cv2.imread(os.path.join(EV_HIGH, name)).astype(np.float32)/255.0

    Ra5   = np.load(os.path.join(CACHE_A5, f"{base}_Ra5.npy"))
    Rl_tv = np.load(os.path.join(CACHE_TV, f"{base}_Rlow.npy"))
    Rh    = np.load(os.path.join(CACHE_TV, f"{base}_Rhigh.npy"))

    Rcnn = infer(Rl_tv)

    pa5,  sa5  = metrics(Ra5,  Rh)
    pcnn, scnn = metrics(Rcnn, Rh)
    all_results.append((name, pa5, sa5, pcnn, scnn))

    # ── Figure: 5-panel row ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor("#0d0d1a")

    panels = [
        (bgr2rgb(low_bgr),  "Low-Light Input",          "#aaaaaa", None,  None),
        (bgr2rgb(Ra5),      f"A5 Reflectance\n(classical best)",   "#ff7777", pa5,  sa5),
        (bgr2rgb(Rcnn),     f"TV+CNN Reflectance\n(LightRefineNet)","#00ff88", pcnn, scnn),
        (bgr2rgb(Rh),       "Ground Truth\nReflectance",           "#5599ff", None, None),
        (bgr2rgb(high_bgr), "High-Light Image\n(reference)",       "#ffcc44", None, None),
    ]

    for ax, (img, title, col, psnr, ssim) in zip(axes, panels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_facecolor("#0d0d1a")
        lbl = title
        if psnr is not None:
            lbl += f"\nPSNR={psnr:.2f} dB  SSIM={ssim:.4f}"
        ax.set_title(lbl, color=col, fontsize=9, fontweight="bold", pad=4)

    fig.suptitle(f"Reflectance Enhancement — {name}", color="white",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.5)
    out_path = os.path.join(OUT_DIR, f"{base}_compare.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  {name:12s}  A5: {pa5:.2f}/{sa5:.4f}   CNN: {pcnn:.2f}/{scnn:.4f}")

# ── Global Summary Bar Chart ───────────────────────────────────────────────────
print("\nGenerating global summary chart...")
names   = [r[0].replace(".png","") for r in all_results]
a5_p    = [r[1] for r in all_results]
cnn_p   = [r[3] for r in all_results]
a5_s    = [r[2] for r in all_results]
cnn_s   = [r[4] for r in all_results]

x = np.arange(len(names)); w = 0.38

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor("#0d0d1a")

for ax in (ax1, ax2):
    ax.set_facecolor("#0a0a14")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.yaxis.label.set_color("#ccc"); ax.xaxis.label.set_color("#ccc")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")

# PSNR
bars1 = ax1.bar(x - w/2, a5_p,  w, label="A5 classical", color="#ff5555", alpha=0.9)
bars2 = ax1.bar(x + w/2, cnn_p, w, label="TV+CNN",       color="#00cc66", alpha=0.9)
avg_a5p  = np.mean(a5_p);  avg_cp = np.mean(cnn_p)
ax1.axhline(avg_a5p, color="#ff5555", ls="--", lw=1.2, alpha=0.7)
ax1.axhline(avg_cp,  color="#00cc66", ls="--", lw=1.2, alpha=0.7)
ax1.set_ylabel("PSNR (dB)", color="#ccc")
ax1.set_title(f"PSNR per Image\nAvg A5={avg_a5p:.2f} dB  →  CNN={avg_cp:.2f} dB  (Δ={avg_cp-avg_a5p:+.2f})",
              color="white", fontweight="bold")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)
ax1.set_ylim(0, max(cnn_p)*1.12)
for bar in bars2:
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f"{bar.get_height():.1f}", ha="center", va="bottom", color="white", fontsize=6.5)

# SSIM
bars3 = ax2.bar(x - w/2, a5_s,  w, label="A5 classical", color="#ff5555", alpha=0.9)
bars4 = ax2.bar(x + w/2, cnn_s, w, label="TV+CNN",       color="#00cc66", alpha=0.9)
avg_a5s = np.mean(a5_s); avg_cs = np.mean(cnn_s)
ax2.axhline(avg_a5s, color="#ff5555", ls="--", lw=1.2, alpha=0.7)
ax2.axhline(avg_cs,  color="#00cc66", ls="--", lw=1.2, alpha=0.7)
ax2.set_ylabel("SSIM", color="#ccc")
ax2.set_title(f"SSIM per Image\nAvg A5={avg_a5s:.4f}  →  CNN={avg_cs:.4f}  (Δ={avg_cs-avg_a5s:+.4f})",
              color="white", fontweight="bold")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)
ax2.set_ylim(0, min(1.0, max(cnn_s)*1.12))

fig.suptitle("Reflectance Enhancement: A5 Classical  vs  LightRefineNet (TV+CNN)\nLOL Dataset — eval15",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_summary.png"), dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# ── Highlight strip: best/worst ───────────────────────────────────────────────
print("Generating highlight strip (best/worst images)...")
sorted_by_gain = sorted(all_results, key=lambda r: r[3]-r[1], reverse=True)
top3  = sorted_by_gain[:3]
bot3  = sorted_by_gain[-3:]
picks = top3 + bot3

fig, axes = plt.subplots(len(picks), 4, figsize=(22, len(picks)*4.5))
fig.patch.set_facecolor("#0d0d1a")

for row_i, (name, pa5, sa5, pcnn, scnn) in enumerate(picks):
    base    = os.path.splitext(name)[0]
    low_bgr = cv2.imread(os.path.join(EV_LOW,  name)).astype(np.float32)/255.0
    Ra5     = np.load(os.path.join(CACHE_A5, f"{base}_Ra5.npy"))
    Rl_tv   = np.load(os.path.join(CACHE_TV, f"{base}_Rlow.npy"))
    Rh      = np.load(os.path.join(CACHE_TV, f"{base}_Rhigh.npy"))
    Rcnn    = infer(Rl_tv)

    gain = pcnn - pa5
    tag  = "BEST GAIN" if row_i < 3 else "LEAST GAIN"
    col  = "#00ff88" if row_i < 3 else "#ff7777"

    for col_i, (img, lbl) in enumerate([
        (bgr2rgb(low_bgr),    "Low-Light Input"),
        (bgr2rgb(Ra5),        f"A5  PSNR={pa5:.2f} SSIM={sa5:.4f}"),
        (bgr2rgb(Rcnn),       f"TV+CNN  PSNR={pcnn:.2f} SSIM={scnn:.4f}"),
        (bgr2rgb(Rh),         "Ground Truth Reflectance"),
    ]):
        ax = axes[row_i][col_i]
        ax.imshow(img); ax.axis("off"); ax.set_facecolor("#0d0d1a")
        if col_i == 0:
            ax.set_title(f"[{tag}]  {name}\nΔPSNR={gain:+.2f} dB", color=col,
                         fontsize=9, fontweight="bold", pad=4)
        else:
            ax.set_title(lbl, color="#dddddd", fontsize=8, pad=3)

plt.suptitle("Top-3 & Bottom-3 Images by CNN Gain  |  LightRefineNet on LOL eval15",
             color="white", fontsize=12, fontweight="bold")
plt.tight_layout(pad=0.6)
plt.savefig(os.path.join(OUT_DIR, "highlight_best_worst.png"), dpi=130,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

print(f"\n{'='*60}")
print(f"All images saved to: {OUT_DIR}/")
print(f"  {len(eval_names)} per-image comparison PNGs")
print(f"  global_summary.png")
print(f"  highlight_best_worst.png")
print(f"\nFINAL RESULTS:")
print(f"  A5 (classical):  PSNR={np.mean(a5_p):.3f}  SSIM={np.mean(a5_s):.4f}")
print(f"  TV+CNN:          PSNR={np.mean(cnn_p):.3f}  SSIM={np.mean(cnn_s):.4f}")
print(f"  CNN Gain:        ΔPSNR={np.mean(cnn_p)-np.mean(a5_p):+.3f} dB  ΔSSIM={np.mean(cnn_s)-np.mean(a5_s):+.4f}")
print(f"{'='*60}")
