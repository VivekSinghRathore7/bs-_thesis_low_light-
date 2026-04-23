"""
RetinexNAF-MS — evaluate.py

Evaluates with TTA ×8 using EMA model weights.
Prints full comparison table against SOTA.

Usage (from project root):
    python src/retinex_naf/evaluate.py                     # TTA on, best_ema.pth
    python src/retinex_naf/evaluate.py --ckpt checkpoints/retinex_naf/epoch_200.pth --tta 0
"""

import os, sys, argparse, json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.retinex_naf.dataset import RetinexNAFDataset
from src.retinex_naf.models  import RetinexNAF


# ── Metrics ───────────────────────────────────────────────────────────────────

def calc_psnr(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1.0/(mse+1e-10))).mean().item()


def calc_ssim(p, g, ws=11):
    C1,C2,ch = 0.01**2, 0.03**2, 3
    d = torch.arange(ws, dtype=torch.float32, device=p.device) - ws//2
    g1d = torch.exp(-d**2/4.5); g1d /= g1d.sum()
    win = (g1d.unsqueeze(1)*g1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(ch,1,1,1)
    pad = ws//2
    mu1 = F.conv2d(p, win, padding=pad, groups=ch)
    mu2 = F.conv2d(g, win, padding=pad, groups=ch)
    s1  = F.conv2d(p*p, win, padding=pad, groups=ch)-mu1**2
    s2  = F.conv2d(g*g, win, padding=pad, groups=ch)-mu2**2
    s12 = F.conv2d(p*g, win, padding=pad, groups=ch)-mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()


class LPIPSMetric:
    """Standard LPIPS (Zhang et al. 2018) — values comparable to published SOTA."""
    def __init__(self, device):
        import lpips
        self.fn = lpips.LPIPS(net='alex').to(device)
        self.fn.eval()
        self.device = device

    def __call__(self, p, g):
        # lpips expects inputs in [-1, 1]
        p2 = p * 2 - 1
        g2 = g * 2 - 1
        with torch.no_grad():
            return self.fn(p2, g2).mean().item()


class _LPIPSCustom(nn.Module):
    """Custom VGG LPIPS kept for internal consistency checks only."""
    def __init__(self, device):
        super().__init__()
        import torchvision.models as M
        vgg = M.vgg16(weights=M.VGG16_Weights.DEFAULT).features.eval().to(device)
        self.sl = nn.ModuleList([vgg[:4],vgg[4:9],vgg[9:16],vgg[16:23]]).eval()
        for p in self.parameters(): p.requires_grad=False
        self.register_buffer('mean',torch.tensor([.485,.456,.406],device=device).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([.229,.224,.225],device=device).view(1,3,1,1))
    def forward(self, p, g):
        p=(p-self.mean)/self.std; g=(g-self.mean)/self.std
        diffs=[]
        for s in self.sl:
            p=s(p); g=s(g)
            pn=p/(p.norm(dim=1,keepdim=True)+1e-10)
            gn=g/(g.norm(dim=1,keepdim=True)+1e-10)
            diffs.append((pn-gn).pow(2).mean(dim=[1,2,3]))
        return torch.stack(diffs).mean(0).item()


# ── TTA ×8 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_predict(model, inp, device):
    preds = []
    for k in range(4):
        for flip in (False, True):
            x = inp.clone()
            x = torch.rot90(x, k, dims=[2,3])
            if flip: x = torch.flip(x, dims=[3])
            out, _, _, _ = model(x.to(device))
            out = out.clamp(0,1)
            if flip: out = torch.flip(out, dims=[3])
            out = torch.rot90(out, -k, dims=[2,3])
            preds.append(out.cpu())
    return torch.stack(preds).mean(0)


# ── Main ──────────────────────────────────────────────────────────────────────

SOTA = [
    ("RetinexNet (BMVC'18)",     16.77, 0.560,  None),
    ("KinD++ (IJCV'21)",         21.30, 0.823,  None),
    ("SNR-Aware (CVPR'22)",      21.48, 0.849,  None),
    ("URetinex-Net (CVPR'22)",   21.32, 0.835,  None),
    ("MIRNet-v2 (TPAMI'22)",     24.74, 0.851,  None),
    ("LLFormer (AAAI'23)",       23.65, 0.857,  None),
    ("GSAD (NeurIPS'23)",        23.23, 0.852,  None),
    ("MBLLIE-Net (SciRep'24)",   23.33, 0.829,  0.116),
    ("Diff-Retinex++ (TPAMI'25)",24.67, 0.867,  0.101),
    ("TFFormer (CVPR'25)",       26.13, 0.888,  0.061),
    ("HFL (DCN'25)",             27.26, 0.930,  0.100),
]


def evaluate(args):
    os.chdir(ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    refine_ckpt = "results/reflectance_cnn/checkpoints/best_model.pth"
    refine_ckpt = refine_ckpt if os.path.exists(refine_ckpt) else None

    model = RetinexNAF(in_ch=9, out_ch=3, width=64, depths=(2,2,4,8),
                       refine_ckpt=refine_ckpt).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    # Handle both full checkpoint (with 'ema'/'model' key) and bare state_dict
    state = ckpt.get('ema', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded : {args.ckpt}")

    ds = RetinexNAFDataset(args.data_root, split="eval15",
                            size=args.img_size, augment=False, cache=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    lpips_fn = LPIPSMetric(device)
    psnrs, ssims, lpipss = [], [], []
    os.makedirs(args.out_dir, exist_ok=True)

    for i, (inp9, gt, _, _) in enumerate(tqdm(loader, desc="Eval")):
        gt = gt.to(device)
        if args.tta:
            pred = tta_predict(model, inp9, device).to(device)
        else:
            with torch.no_grad():
                pred, _, _, _ = model(inp9.to(device))
                pred = pred.clamp(0,1)

        psnrs.append(calc_psnr(pred, gt))
        ssims.append(calc_ssim(pred, gt))
        lpipss.append(lpips_fn(pred, gt))

        img = pred[0].cpu().numpy().transpose(1,2,0)*255
        cv2.imwrite(os.path.join(args.out_dir, f"{i:04d}.png"),
                    img.astype(np.uint8))

    pm, sm, lm = np.mean(psnrs), np.mean(ssims), np.mean(lpipss)

    # ── Print comparison table ─────────────────────────────────────────────────
    sep = "─"*62
    print(f"\n{sep}")
    print(f"{'Method':<32} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>8}")
    print(sep)
    for name, p, s, l in SOTA:
        ls = f"{l:.3f}" if l else "  —  "
        print(f"{name:<32} {p:>7.2f} {s:>7.3f} {ls:>8}")
    print(sep)
    tta_str = " (TTA×8)" if args.tta else ""
    our = f"RetinexNAF-MS{tta_str}"
    print(f"\033[1m{our:<32} {pm:>7.2f} {sm:>7.4f} {lm:>8.5f}\033[0m")
    print(sep)

    # rank against SOTA
    sota_psnrs = [x[1] for x in SOTA]
    rank = sum(1 for x in sota_psnrs if x > pm) + 1
    print(f"\nRanking: #{rank} out of {len(SOTA)+1} methods on LOL-v1")

    results = {"psnr": psnrs, "ssim": ssims, "lpips": lpipss,
               "mean": {"psnr": pm, "ssim": sm, "lpips": lm},
               "tta": bool(args.tta)}
    with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved  → {args.out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      default="checkpoints/retinex_naf/best_ema.pth")
    parser.add_argument("--data_root", default="datasets/LOL_dataset")
    parser.add_argument("--out_dir",   default="results/retinex_naf_eval")
    parser.add_argument("--img_size",  type=int, default=256)
    parser.add_argument("--tta",       type=int, default=1)
    args = parser.parse_args()
    evaluate(args)
