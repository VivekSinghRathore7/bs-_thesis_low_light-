"""
RetinexRestormer evaluation on LOL-v2 Real test set.

Usage (from project root):
    conda run -n viv python src/retinex_restormer/eval_lolv2.py
    conda run -n viv python src/retinex_restormer/eval_lolv2.py --tta
"""

import os, sys, argparse, json
import numpy as np, cv2
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)

from retinex_utils import decompose_retinex
from src.retinex_restormer.architecture import RetinexRestormer


class FullLOLv2Dataset(Dataset):
    """Full-image dataset for LOL-v2 evaluation."""
    def __init__(self, root, split="Test"):
        self.low_dir  = os.path.join(root, split, "Input")
        self.high_dir = os.path.join(root, split, "GT")
        self.files = sorted(
            f for f in os.listdir(self.low_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        low  = cv2.imread(os.path.join(self.low_dir,  f))
        high = cv2.imread(os.path.join(self.high_dir, f))
        H, W = low.shape[:2]
        H32 = (H // 32) * 32
        W32 = (W // 32) * 32
        low  = cv2.resize(low,  (W32, H32)).astype(np.float32) / 255.0
        high = cv2.resize(high, (W32, H32)).astype(np.float32) / 255.0

        R_low, I_tv = decompose_retinex(low)
        I_tv_3d = I_tv[:, :, np.newaxis] if I_tv.ndim == 2 else I_tv
        inp = np.concatenate([low, R_low, I_tv_3d], axis=2)

        def t(a):
            a = np.ascontiguousarray(a)
            if a.ndim == 2: return torch.from_numpy(a[None]).float()
            return torch.from_numpy(a.transpose(2, 0, 1)).float()
        return t(inp), t(high), f


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1, 2, 3])
    return (10 * torch.log10(1.0 / (mse + 1e-10))).mean().item()


def ssim_fn(p, g, ws=11, ch=3):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    d = torch.arange(ws, dtype=torch.float32, device=p.device) - ws // 2
    g1 = torch.exp(-d ** 2 / 4.5)
    g1 = g1 / g1.sum()
    win = (g1.unsqueeze(1) * g1.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(ch, 1, 1, 1)
    pad = ws // 2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1 = F.conv2d(p * p, win, padding=pad, groups=ch) - mu1 ** 2
    s2 = F.conv2d(g * g, win, padding=pad, groups=ch) - mu2 ** 2
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


# ── SOTA comparison table ─────────────────────────────────────────────────────

SOTA_LOLv1 = [
    ("RetinexNet (BMVC'18)",      16.77, 0.560, None),
    ("KinD++ (IJCV'21)",          21.30, 0.823, None),
    ("URetinex-Net (CVPR'22)",    21.32, 0.835, 1.220),
    ("SNR-Aware (CVPR'22)",       21.48, 0.849, None),
    ("GSAD (NeurIPS'23)",         23.23, 0.852, None),
    ("MBLLIE-Net (SciRep'24)",    23.33, 0.829, 0.116),
    ("LLFormer (AAAI'23)",        23.65, 0.857, None),
    ("MIRNet-v2 (TPAMI'22)",      24.74, 0.851, None),
    ("Diff-Retinex++ (TPAMI'25)", 24.67, 0.867, 0.101),
    ("TFFormer (CVPR'25)",        26.13, 0.888, 0.061),
    ("HFL (DCN'25)",              27.26, 0.930, 0.100),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    os.chdir(ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: LOL-v2 Real (Test: 100 images)\n")

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
    print(f"Params: {n_params:.2f}M")
    print(f"Best train PSNR: {c.get('best_psnr', 'N/A')}\n")

    lpips_fn = get_lpips(device)

    ds = FullLOLv2Dataset(args.data_root, split="Test")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    os.makedirs(args.out_dir, exist_ok=True)

    psnrs, ssims, lpipss, fnames = [], [], [], []

    for i, (inp7, gt, fname) in enumerate(tqdm(loader, desc="Eval LOL-v2")):
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
        fnames.append(fname[0] if isinstance(fname, (list, tuple)) else fname)

        img = pred[0].cpu().numpy().transpose(1, 2, 0) * 255
        cv2.imwrite(os.path.join(args.out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    pm, sm, lm = np.mean(psnrs), np.mean(ssims), np.nanmean(lpipss)

    sep = "─" * 65
    print(f"\n{sep}")
    print(f"  LOL-v2 Real — RetinexRestormer Results")
    print(sep)
    suffix = " +TTA×8" if args.tta else ""
    print(f"  PSNR  : {pm:.4f} dB")
    print(f"  SSIM  : {sm:.4f}")
    print(f"  LPIPS : {lm:.4f}")
    print(f"  Mode  : {'TTA×8' if args.tta else 'Standard'}")
    print(f"  Images: {len(psnrs)}")
    print(sep)

    # Compare with LOL-v1 results
    print(f"\n  For reference, LOL-v1 results: PSNR=23.75, SSIM=0.886, LPIPS=0.104")

    results = {
        "dataset": "LOL-v2-Real",
        "psnr": psnrs, "ssim": ssims, "lpips": lpipss,
        "mean": {"psnr": pm, "ssim": sm, "lpips": lm},
        "tta": args.tta,
        "n_images": len(psnrs),
    }
    with open(os.path.join(args.out_dir, "metrics_lolv2.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="checkpoints/lolv2/retinex_restormer/best_psnr.pth")
    p.add_argument("--data_root",  default="datasets/LOL_v2_real")
    p.add_argument("--out_dir",    default="results/lolv2/retinex_restormer_images")
    p.add_argument("--tta",        action="store_true")
    p.add_argument("--width",      type=int, default=32)
    p.add_argument("--depths",     type=int, nargs=4, default=[2, 2, 2, 4])
    p.add_argument("--heads",      type=int, nargs=4, default=[1, 2, 4, 8])
    evaluate(p.parse_args())
