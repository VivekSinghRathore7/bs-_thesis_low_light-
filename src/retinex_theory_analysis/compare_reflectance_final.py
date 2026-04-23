"""
Proper comparison: TV+CNN improved reflectance vs Ground Truth reflectance
Shows per image: Low | Improved R | GT R | Difference (×5) | Inverse Difference
Plus a full-grid overview and a channel-wise diff analysis.
"""
import os, sys, numpy as np, cv2, torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity   as calc_ssim

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = "/home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light"
EV_LOW   = os.path.join(ROOT, "datasets/LOL_dataset/eval15/low")
EV_HIGH  = os.path.join(ROOT, "datasets/LOL_dataset/eval15/high")
CKP      = os.path.join(ROOT, "results/reflectance_cnn/checkpoints/best_model.pth")
CACHE_TV = os.path.join(ROOT, "results/reflectance_cnn/cache_eval")
OUT_DIR  = os.path.join(ROOT, "results/reflectance_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BG  = "#0d0d1a"
BG2 = "#0a0a14"

# ── Model ──────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(nf,nf,3,1,1,bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True),
            nn.Conv2d(nf,nf,3,1,1,bias=False), nn.BatchNorm2d(nf))
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x): return self.relu(x+self.block(x))

class LightRefineNet(nn.Module):
    def __init__(self,nf=48,nb=10):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(3,nf,3,1,1,bias=False), nn.ReLU(inplace=True))
        self.body = nn.Sequential(*[ResBlock(nf) for _ in range(nb)])
        self.tail = nn.Conv2d(nf,3,3,1,1)
    def forward(self,x): return torch.clamp(x+self.tail(self.body(self.head(x))),0,1)

model = LightRefineNet().to(DEVICE)
model.load_state_dict(torch.load(CKP, map_location=DEVICE))
model.eval()
print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

def infer(R_np):
    with torch.no_grad():
        t = torch.from_numpy(R_np.transpose(2,0,1)).unsqueeze(0).float().to(DEVICE)
        return model(t).squeeze(0).permute(1,2,0).cpu().numpy().clip(0,1).astype(np.float32)

def u8(x):  return (np.clip(x,0,1)*255).astype(np.uint8)
def rgb(bgr): return cv2.cvtColor(u8(bgr), cv2.COLOR_BGR2RGB)

def metrics(est, ref):
    p = calc_psnr(u8(ref), u8(est), data_range=255)
    s = calc_ssim(u8(ref), u8(est), channel_axis=2, data_range=255)
    mae = float(np.mean(np.abs(est-ref)))
    rmse = float(np.sqrt(np.mean((est-ref)**2)))
    return p, s, mae, rmse

def style_ax(ax, title, color="#dddddd", fs=8):
    ax.axis("off"); ax.set_facecolor(BG2)
    ax.set_title(title, color=color, fontsize=fs, fontweight="bold", pad=3)

# ── Collect data ───────────────────────────────────────────────────────────────
eval_names = sorted(f for f in os.listdir(EV_LOW) if f.endswith((".png",".jpg")))
records = []

print("\nRunning CNN inference on eval15 ...")
for name in eval_names:
    base   = os.path.splitext(name)[0]
    low    = cv2.imread(os.path.join(EV_LOW,  name)).astype(np.float32)/255.0
    Rl_tv  = np.load(os.path.join(CACHE_TV, f"{base}_Rlow.npy"))
    Rh     = np.load(os.path.join(CACHE_TV, f"{base}_Rhigh.npy"))
    Rcnn   = infer(Rl_tv)
    p, s, mae, rmse = metrics(Rcnn, Rh)

    diff        = np.abs(Rcnn - Rh)                           # absolute error
    diff_amp    = np.clip(diff * 5, 0, 1)                     # ×5 amplified
    inv_diff    = 1.0 - diff_amp                              # inverse (bright=good)
    diff_signed = (Rcnn - Rh + 1) / 2                        # signed [0..1] around 0.5
    diff_gray   = diff.mean(axis=2)                           # scalar error map

    records.append(dict(
        name=name, base=base, low=low, Rl=Rl_tv, Rh=Rh, Rcnn=Rcnn,
        diff=diff, diff_amp=diff_amp, inv_diff=inv_diff,
        diff_signed=diff_signed, diff_gray=diff_gray,
        psnr=p, ssim=s, mae=mae, rmse=rmse))
    print(f"  {name:<12}  PSNR={p:.2f}  SSIM={s:.4f}  MAE={mae*100:.2f}%  RMSE={rmse*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-image 6-panel figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Generating per-image 6-panel figures ...")
for r in records:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f"Reflectance Comparison — {r['name']}\n"
        f"PSNR={r['psnr']:.2f} dB  |  SSIM={r['ssim']:.4f}  |  MAE={r['mae']*100:.2f}%  |  RMSE={r['rmse']*100:.2f}%",
        color="white", fontsize=12, fontweight="bold", y=1.01)

    panels = [
        # row 0
        (rgb(r['low']),       "Low-Light Input",                          "#aaaaaa"),
        (rgb(r['Rcnn']),      "Improved Reflectance  (TV+CNN)",           "#00ff88"),
        (rgb(r['Rh']),        "Ground Truth Reflectance  (R_high)",       "#5599ff"),
        # row 1
        (r['diff_amp'],       "Absolute Difference  ×5  (brighter=worse)","#ff6666"),
        (r['inv_diff'],       "Inverse Difference  (brighter=better match)","#ffcc44"),
        (r['diff_signed'],    "Signed Difference  (0.5=no error, blue=under, red=over)","#cc88ff"),
    ]

    for ax, (img, title, col) in zip(axes.flat, panels):
        ax.set_facecolor(BG2)
        ax.axis("off")
        if img.ndim == 2:
            im = ax.imshow(img, cmap="hot", vmin=0, vmax=1)
        elif img.shape[2] == 3 and img.dtype == np.float32 and img.max() <= 1.0:
            # check if it's a colour diff or a normal RGB
            if title.startswith("Signed"):
                im = ax.imshow(img, cmap="RdBu_r", vmin=0, vmax=1)
            else:
                im = ax.imshow(img)
        else:
            im = ax.imshow(img)
        ax.set_title(title, color=col, fontsize=8.5, fontweight="bold", pad=3)

    plt.tight_layout(pad=0.8)
    plt.savefig(os.path.join(OUT_DIR, f"{r['base']}_fullcompare.png"),
                dpi=140, bbox_inches="tight", facecolor=BG)
    plt.close()

print(f"  Saved {len(records)} per-image figures.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Grand comparison grid  (all 15 images × 4 columns: Improved | GT | Diff×5 | Inverse)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Generating grand comparison grid ...")
N   = len(records)
COL_LABELS = ["Improved R  (TV+CNN)", "GT Reflectance  (R_high)",
              "Difference ×5\n(brighter = worse)", "Inverse Diff\n(brighter = better)"]
COL_COLORS = ["#00ff88", "#5599ff", "#ff6666", "#ffcc44"]

fig = plt.figure(figsize=(22, N * 2.9))
fig.patch.set_facecolor(BG)
gs  = gridspec.GridSpec(N + 1, 4, figure=fig,
                        hspace=0.06, wspace=0.03,
                        top=0.97, bottom=0.01, left=0.08, right=0.99)

# Column headers
for c, (lbl, col) in enumerate(zip(COL_LABELS, COL_COLORS)):
    ax = fig.add_subplot(gs[0, c])
    ax.set_facecolor(BG2); ax.axis("off")
    ax.text(0.5, 0.5, lbl, color=col, fontsize=10, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)

for row, r in enumerate(records):
    imgs = [rgb(r['Rcnn']), rgb(r['Rh']), r['diff_amp'], r['inv_diff']]
    for c, img in enumerate(imgs):
        ax = fig.add_subplot(gs[row + 1, c])
        ax.set_facecolor(BG2); ax.axis("off")
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3 and img.dtype != np.uint8):
            ax.imshow(img, cmap="hot" if img.ndim == 2 else None, vmin=0, vmax=1)
        else:
            ax.imshow(img)
        if c == 0:
            ax.set_ylabel(f"{r['base']}\nPSNR={r['psnr']:.1f}  SSIM={r['ssim']:.3f}",
                          color="#cccccc", fontsize=7, labelpad=3,
                          rotation=0, ha="right", va="center")

fig.suptitle(
    "LightRefineNet  |  Improved Reflectance vs Ground Truth  |  LOL eval15\n"
    "Column 3: |R_CNN − R_GT| × 5  (amplified absolute error)   "
    "Column 4: 1 − |R_CNN − R_GT| × 5  (agreement map)",
    color="white", fontsize=11, fontweight="bold")

plt.savefig(os.path.join(OUT_DIR, "grand_comparison_grid.png"),
            dpi=120, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved grand_comparison_grid.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Error distribution + channel analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Generating error distribution & channel analysis ...")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.patch.set_facecolor(BG)
axes = axes.flat

# Collect all pixel errors across dataset
all_diff_flat  = np.concatenate([r['diff'].ravel()            for r in records])
all_diff_r     = np.concatenate([r['diff'][:,:,2].ravel()     for r in records])
all_diff_g     = np.concatenate([r['diff'][:,:,1].ravel()     for r in records])
all_diff_b     = np.concatenate([r['diff'][:,:,0].ravel()     for r in records])

# 3a: histogram of per-pixel absolute error (all channels combined)
ax = next(axes); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.tick_params(colors="#aaa")
ax.hist(all_diff_flat*100, bins=120, color="#00cc66", alpha=0.8, density=True, range=(0,40))
ax.axvline(np.mean(all_diff_flat)*100, color="#ffcc00", lw=2,
           label=f"Mean={np.mean(all_diff_flat)*100:.2f}%")
ax.axvline(np.median(all_diff_flat)*100, color="#ff6666", lw=2, ls="--",
           label=f"Median={np.median(all_diff_flat)*100:.2f}%")
ax.set_xlabel("Absolute pixel error (%)", color="#ccc"); ax.set_ylabel("Density", color="#ccc")
ax.set_title("Pixel Error Distribution (all channels)", color="white", fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)

# 3b: per-channel error histograms (R/G/B)
ax = next(axes); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.tick_params(colors="#aaa")
for ch_data, col, lbl in [(all_diff_r,"#ff5555","Red"), (all_diff_g,"#55dd55","Green"), (all_diff_b,"#5599ff","Blue")]:
    ax.hist(ch_data*100, bins=80, color=col, alpha=0.55, density=True, range=(0,40), label=lbl)
ax.set_xlabel("Absolute pixel error (%)", color="#ccc"); ax.set_ylabel("Density", color="#ccc")
ax.set_title("Per-Channel Error Distribution", color="white", fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)

# 3c: per-image PSNR bar
ax = next(axes); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.tick_params(colors="#aaa", labelsize=7)
names_short = [r['base'] for r in records]
psnrs = [r['psnr'] for r in records]
cols  = ["#00ff88" if p >= 20 else "#ffcc44" if p >= 15 else "#ff5555" for p in psnrs]
ax.bar(names_short, psnrs, color=cols, alpha=0.9)
ax.axhline(np.mean(psnrs), color="white", ls="--", lw=1.5, label=f"Avg={np.mean(psnrs):.2f} dB")
ax.set_xticklabels(names_short, rotation=45, ha="right")
ax.set_ylabel("PSNR (dB)", color="#ccc")
ax.set_title("Per-Image PSNR  (green≥20, yellow≥15, red<15)", color="white", fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)

# 3d: per-image SSIM bar
ax = next(axes); ax.set_facecolor(BG2)
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.tick_params(colors="#aaa", labelsize=7)
ssims = [r['ssim'] for r in records]
cols2 = ["#00ff88" if s >= 0.8 else "#ffcc44" if s >= 0.65 else "#ff5555" for s in ssims]
ax.bar(names_short, ssims, color=cols2, alpha=0.9)
ax.axhline(np.mean(ssims), color="white", ls="--", lw=1.5, label=f"Avg={np.mean(ssims):.4f}")
ax.set_xticklabels(names_short, rotation=45, ha="right")
ax.set_ylabel("SSIM", color="#ccc"); ax.set_ylim(0, 1)
ax.set_title("Per-Image SSIM  (green≥0.80, yellow≥0.65, red<0.65)", color="white", fontweight="bold")
ax.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#555", fontsize=9)

# 3e: mean error map averaged across all images
ax = next(axes); ax.set_facecolor(BG2); ax.axis("off")
mean_diff_map = np.mean(np.stack([r['diff_amp'] for r in records], axis=0), axis=0)
im = ax.imshow(mean_diff_map, cmap="hot", vmin=0, vmax=1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im, cax=cax).ax.tick_params(colors="#aaa", labelsize=7)
ax.set_title("Mean Error Map (×5 amplified)\naveraged over all 15 images",
             color="#ff6666", fontweight="bold")

# 3f: mean inverse diff map (agreement)
ax = next(axes); ax.set_facecolor(BG2); ax.axis("off")
mean_inv_map = np.mean(np.stack([r['inv_diff'] for r in records], axis=0), axis=0)
im2 = ax.imshow(mean_inv_map, cmap="hot", vmin=0, vmax=1)
divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("right", size="3%", pad=0.05)
plt.colorbar(im2, cax=cax2).ax.tick_params(colors="#aaa", labelsize=7)
ax.set_title("Mean Agreement Map (inverse diff)\nareas close to 1.0 = near-perfect match",
             color="#ffcc44", fontweight="bold")

fig.suptitle("Error Analysis: TV+CNN Reflectance vs Ground Truth  |  LOL eval15",
             color="white", fontsize=13, fontweight="bold")
plt.tight_layout(pad=1.0)
plt.savefig(os.path.join(OUT_DIR, "error_analysis.png"),
            dpi=140, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved error_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Summary metrics table figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Generating metrics summary table ...")

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")

col_labels = ["Image", "PSNR (dB)", "SSIM", "MAE (%)", "RMSE (%)", "Quality"]
rows_data  = []
for r in records:
    q = "Excellent" if r['psnr']>=22 else "Good" if r['psnr']>=18 else "Fair" if r['psnr']>=14 else "Poor"
    rows_data.append([r['base'], f"{r['psnr']:.2f}", f"{r['ssim']:.4f}",
                      f"{r['mae']*100:.2f}", f"{r['rmse']*100:.2f}", q])

avg_psnr = np.mean([r['psnr'] for r in records])
avg_ssim = np.mean([r['ssim'] for r in records])
avg_mae  = np.mean([r['mae']  for r in records])
avg_rmse = np.mean([r['rmse'] for r in records])
rows_data.append(["AVERAGE", f"{avg_psnr:.2f}", f"{avg_ssim:.4f}",
                  f"{avg_mae*100:.2f}", f"{avg_rmse*100:.2f}", "—"])

table = ax.table(cellText=rows_data, colLabels=col_labels,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(9)
table.scale(1, 1.55)

# Style header
for c in range(len(col_labels)):
    table[0, c].set_facecolor("#1a3a5c"); table[0, c].set_text_props(color="white", fontweight="bold")

# Style data rows
q_colors = {"Excellent": "#003300", "Good": "#1a3300", "Fair": "#332200", "Poor": "#330000", "—": "#111122"}
for i, r in enumerate(records + [None]):
    for c in range(len(col_labels)):
        cell = table[i+1, c]
        if i == len(records):  # average row
            cell.set_facecolor("#1a1a3a"); cell.set_text_props(color="#ffcc44", fontweight="bold")
        else:
            q = rows_data[i][5]
            cell.set_facecolor(q_colors.get(q, "#111122"))
            cell.set_text_props(color="#dddddd")
            if c == 5:
                qc = {"Excellent":"#00ff88","Good":"#88ff44","Fair":"#ffcc00","Poor":"#ff5555"}
                cell.set_text_props(color=qc.get(q,"white"), fontweight="bold")

ax.set_title("LightRefineNet — Reflectance Quality vs Ground Truth  (LOL eval15)",
             color="white", fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "metrics_table.png"),
            dpi=140, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved metrics_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Side-by-side strip: best 4 + worst 2 (Improved | GT | Diff | Inv)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Generating best/worst highlight strip ...")

sorted_recs = sorted(records, key=lambda r: r['psnr'], reverse=True)
picks = sorted_recs[:4] + sorted_recs[-2:]  # top4 + bottom2

fig, axes = plt.subplots(len(picks), 4, figsize=(22, len(picks)*4.0))
fig.patch.set_facecolor(BG)

col_hdrs = ["Improved Reflectance (TV+CNN)", "Ground Truth (R_high)",
            "Difference ×5", "Inverse Difference"]
hdr_cols = ["#00ff88", "#5599ff", "#ff6666", "#ffcc44"]

for ci, (hdr, col) in enumerate(zip(col_hdrs, hdr_cols)):
    axes[0][ci].set_title(hdr, color=col, fontsize=9, fontweight="bold", pad=4)

for ri, r in enumerate(picks):
    tag = "BEST" if ri < 4 else "WORST"
    tc  = "#00ff88" if ri < 4 else "#ff5555"
    imgs = [rgb(r['Rcnn']), rgb(r['Rh']), r['diff_amp'], r['inv_diff']]
    for ci, img in enumerate(imgs):
        ax = axes[ri][ci]
        ax.set_facecolor(BG2); ax.axis("off")
        kw = dict(cmap="hot", vmin=0, vmax=1) if img.ndim==3 and img.max()<=1.0 and img.dtype!=np.uint8 else {}
        ax.imshow(img, **kw)
        if ci == 0:
            ax.set_ylabel(
                f"[{tag}] {r['base']}\nPSNR={r['psnr']:.2f} dB\nSSIM={r['ssim']:.4f}",
                color=tc, fontsize=8, fontweight="bold", rotation=0,
                ha="right", va="center", labelpad=6)

fig.suptitle("Top-4 & Bottom-2 Images by PSNR  |  Improved Reflectance vs Ground Truth",
             color="white", fontsize=12, fontweight="bold")
plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(OUT_DIR, "best_worst_strip.png"),
            dpi=130, bbox_inches="tight", facecolor=BG)
plt.close()
print("  Saved best_worst_strip.png")

# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"All outputs saved → {OUT_DIR}/")
print(f"  • {len(records)} × *_fullcompare.png  (per-image 6-panel)")
print(f"  • grand_comparison_grid.png")
print(f"  • error_analysis.png")
print(f"  • metrics_table.png")
print(f"  • best_worst_strip.png")
print(f"\nFINAL METRICS  (n={len(records)})")
print(f"  PSNR : {avg_psnr:.3f} dB")
print(f"  SSIM : {avg_ssim:.4f}")
print(f"  MAE  : {avg_mae*100:.2f}%")
print(f"  RMSE : {avg_rmse*100:.2f}%")
print(f"  SSIM gap remaining : {1-avg_ssim:.4f}")
print(f"{'='*60}")
