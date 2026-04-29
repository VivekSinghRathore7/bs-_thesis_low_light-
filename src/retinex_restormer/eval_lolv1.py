"""
RetinexRestormer evaluation — retinex_restormer/evaluate.py

Usage (from project root):
    # Without TTA
    conda run -n viv python src/retinex_restormer/evaluate.py

    # With TTA×8
    conda run -n viv python src/retinex_restormer/evaluate.py --tta
"""

import os, sys, argparse, json
import numpy as np, cv2
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "gan_pipelines"))  # for retinex_utils
sys.path.insert(0, ROOT)

from src.legacy.retinex_patch.dataset import FullLOLDataset
from src.retinex_restormer.architecture import RetinexRestormer


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()


def ssim_fn(p, g, ws=11, ch=3):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    d   = torch.arange(ws, dtype=torch.float32, device=p.device) - ws // 2
    g1  = torch.exp(-d ** 2 / 4.5)
    g1  = g1 / g1.sum()
    win = (g1.unsqueeze(1) * g1.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = ws // 2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1  = F.conv2d(p * p, win, padding=pad, groups=ch) - mu1 ** 2
    s2  = F.conv2d(g * g, win, padding=pad, groups=ch) - mu2 ** 2
    s12 = F.conv2d(p * g, win, padding=pad, groups=ch) - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) /
            ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean().item()


def get_lpips(device):
    try:
        import lpips
        fn = lpips.LPIPS(net='alex').to(device).eval()
        def _lpips(p, g):
            with torch.no_grad():
                return fn(p * 2 - 1, g * 2 - 1).mean().item()
        return _lpips
    except ImportError:
        print("WARNING: lpips not installed. Run: pip install lpips")
        return lambda p, g: float('nan')


# ── TTA×8 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_forward(fn, inp, device):
    preds = []
    for k in range(4):
        for flip in (False, True):
            x = inp.clone()
            x = torch.rot90(x, k, dims=[2, 3])
            if flip:
                x = torch.flip(x, dims=[3])
            pred = fn(x.to(device)).clamp(0, 1)
            if flip:
                pred = torch.flip(pred, dims=[3])
            pred = torch.rot90(pred, -k, dims=[2, 3])
            preds.append(pred.cpu())
    return torch.stack(preds).mean(0).to(device)


# ── SOTA table (LOL-v1) ───────────────────────────────────────────────────────
# Format: (name, params, psnr, ssim, lpips)  — None for unavailable values

SOTA_TABLE = [
    ("Zero-DCE",        "0.07M",   14.86, 0.589, 0.335),
    ("RetinexNet",      "0.84M",   16.77, 0.559, 0.473),
    ("EnlightenGAN",    "8.63M",   17.48, 0.677, 0.322),
    ("KinD++",          "—",       21.30, 0.823, 0.160),
    ("SNR-Net",         "—",       None,  None,  None),
    ("URetinex-Net",    "1.48M",   21.33, 0.835, 0.201),
    ("SNR-Aware",       "—",       24.61, 0.842, 0.151),
    ("Restormer",       "26.13M",  22.43, 0.823, 0.125),
    ("MIRNet",          "—",       24.14, 0.830, None),
    ("RetinexFormer",   "—",       None,  None,  None),
    ("GLARE",           "—",       None,  None,  None),
    ("LLFormer",        "24.5M",   23.65, 0.816, 0.108),
    ("PairLIE",         "—",       19.51, 0.736, 0.248),
]

OURS_TABLE = [
    ("Baseline cGAN",      "~2.5M",  16.52, 0.654, 0.301),
    ("Disentangled GAN",   "~2.8M",  22.29, 0.838, 0.165),
    ("Ensemble GAN",       "~6.0M",  22.84, 0.861, 0.142),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.chdir(ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load EMA weights
    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found: {args.ckpt}")
        return

    model = RetinexRestormer(
        in_ch=7, out_ch=3,
        width=args.width,
        depths=tuple(args.depths),
        heads=tuple(args.heads),
    ).to(device).eval()

    c = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = c.get('ema', c.get('model', c))
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded: {args.ckpt}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {n_params:.2f}M\n")

    lpips_fn = get_lpips(device)

    ds     = FullLOLDataset(args.data_root, split="eval15")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    os.makedirs(args.out_dir, exist_ok=True)

    psnrs, ssims, lpipss = [], [], []

    for i, (inp7, gt, _, _) in enumerate(tqdm(loader, desc="Eval")):
        inp7, gt = inp7.to(device), gt.to(device)

        def run(x):
            return model(x).clamp(0, 1)

        if args.tta:
            pred = tta_forward(run, inp7, device)
        else:
            with torch.no_grad():
                pred = run(inp7)

        psnrs.append(psnr_fn(pred, gt))
        ssims.append(ssim_fn(pred, gt))
        lpipss.append(lpips_fn(pred, gt))

        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(args.out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    pm, sm, lm = np.mean(psnrs), np.mean(ssims), np.nanmean(lpipss)

    def fmt(v, w, dec):
        return f"{v:>{w}.{dec}f}" if v is not None else f"{'—':>{w}}"

    sep = "─" * 75
    print(f"\n{sep}")
    print(f"{'Method':<24} {'Params':>8} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8}")
    print(sep)
    for name, params, p, s, l in SOTA_TABLE:
        print(f"{name:<24} {params:>8} {fmt(p,8,2)} {fmt(s,8,3)} {fmt(l,8,3)}")
    print(sep)
    for name, params, p, s, l in OURS_TABLE:
        print(f"{name:<24} {params:>8} {fmt(p,8,2)} {fmt(s,8,3)} {fmt(l,8,3)}")
    suffix = " +TTA×8" if args.tta else ""
    label  = f"RetinexRestormer{suffix}"
    print(f"\033[1m{label:<24} {'~5.06M':>8} {pm:>8.2f} {sm:>8.3f} {lm:>8.3f}\033[0m")
    print(sep)

    # Rank among all methods with available PSNR
    all_psnrs = [p for _, _, p, _, _ in SOTA_TABLE + OURS_TABLE if p is not None]
    rank = sum(1 for x in all_psnrs if x > pm) + 1
    print(f"\nRanking: #{rank} / {len(all_psnrs)+1} on LOL-v1 (by PSNR)")

    results = {"psnr": psnrs, "ssim": ssims, "lpips": lpipss,
               "mean": {"psnr": pm, "ssim": sm, "lpips": lm},
               "tta": args.tta}
    with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/lolv1/retinex_restormer/best_psnr.pth")
    p.add_argument("--data_root",  default="datasets/LOL_dataset")
    p.add_argument("--out_dir",    default="results/lolv1/retinex_restormer_images")
    p.add_argument("--tta",        action="store_true")
    p.add_argument("--width",      type=int, default=32)
    p.add_argument("--depths",     type=int, nargs=4, default=[2, 2, 2, 4])
    p.add_argument("--heads",      type=int, nargs=4, default=[1, 2, 4, 8])
    evaluate(p.parse_args())
