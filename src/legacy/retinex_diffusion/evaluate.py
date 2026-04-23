"""
Final evaluation — Stage 1 alone OR Stage 1 + Stage 2 diffusion + TTA×8.

Usage (from project root):
    # Stage 1 only
    conda run -n viv python src/retinex_diffusion/evaluate.py --stage 1

    # Stage 1 + Diffusion (default)
    conda run -n viv python src/retinex_diffusion/evaluate.py

    # Stage 1 + Diffusion + TTA×8
    conda run -n viv python src/retinex_diffusion/evaluate.py --tta
"""

import os, sys, argparse, json
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)

from models import IllumNet, ReflecNet, RefineBlock
from src.retinex_patch.dataset import FullLOLDataset
from src.retinex_diffusion.model import DenoisingUNet, DDPMSchedule


# ── Metrics ───────────────────────────────────────────────────────────────────

def psnr_fn(p, g):
    mse = F.mse_loss(p, g, reduction='none').mean(dim=[1,2,3])
    return (10*torch.log10(1./(mse+1e-10))).mean().item()

def ssim_fn(p, g, ws=11, ch=3):
    C1,C2 = 0.01**2,0.03**2
    d = torch.arange(ws, dtype=torch.float32, device=p.device)-ws//2
    g1 = torch.exp(-d**2/4.5); g1/=g1.sum()
    win=(g1.unsqueeze(1)*g1.unsqueeze(0)).unsqueeze(0).unsqueeze(0).repeat(ch,1,1,1)
    pad=ws//2
    mu1=F.conv2d(p,win,padding=pad,groups=ch); mu2=F.conv2d(g,win,padding=pad,groups=ch)
    s1=F.conv2d(p*p,win,padding=pad,groups=ch)-mu1**2
    s2=F.conv2d(g*g,win,padding=pad,groups=ch)-mu2**2
    s12=F.conv2d(p*g,win,padding=pad,groups=ch)-mu1*mu2
    return (((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean().item()

def get_lpips(device):
    try:
        import lpips
        fn = lpips.LPIPS(net='alex').to(device).eval()
        def _lpips(p, g):
            with torch.no_grad():
                return fn(p*2-1, g*2-1).mean().item()
        return _lpips
    except ImportError:
        print("WARNING: lpips not installed. Install with: pip install lpips")
        return lambda p,g: float('nan')


# ── Model loading ─────────────────────────────────────────────────────────────

def load_stage1(ckpt, device):
    illum  = IllumNet(base_filters=32).to(device).eval()
    reflec = ReflecNet(base_filters=48).to(device).eval()
    refine = RefineBlock(base_filters=48).to(device).eval()
    c = torch.load(ckpt, map_location=device, weights_only=True)
    illum.load_state_dict(c['ema_illum'])
    reflec.load_state_dict(c['ema_reflec'])
    refine.load_state_dict(c['ema_refine'])
    for m in [illum,reflec,refine]:
        for p in m.parameters(): p.requires_grad_(False)
    return illum, reflec, refine

@torch.no_grad()
def s1_forward(illum, reflec, refine, inp7):
    L=inp7[:,6:7]; R=inp7[:,3:6]
    return refine((reflec(R)*illum(L)).clamp(0,1)).clamp(0,1)


# ── TTA ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def tta_forward(fn, inp, device):
    """8-fold TTA: average predictions over all flip+rotation combos."""
    preds = []
    for k in range(4):
        for flip in (False, True):
            x = inp.clone()
            x = torch.rot90(x, k, dims=[2,3])
            if flip: x = torch.flip(x, dims=[3])
            pred = fn(x.to(device)).clamp(0,1)
            if flip: pred = torch.flip(pred, dims=[3])
            pred = torch.rot90(pred, -k, dims=[2,3])
            preds.append(pred.cpu())
    return torch.stack(preds).mean(0).to(device)


# ── Main ──────────────────────────────────────────────────────────────────────

SOTA_TABLE = [
    ("RetinexNet (BMVC'18)",      16.77, 0.560, None),
    ("KinD++ (IJCV'21)",          21.30, 0.823, None),
    ("SNR-Aware (CVPR'22)",       21.48, 0.849, None),
    ("URetinex-Net (CVPR'22)",    21.32, 0.835, 1.220),
    ("MIRNet-v2 (TPAMI'22)",      24.74, 0.851, None),
    ("LLFormer (AAAI'23)",        23.65, 0.857, None),
    ("GSAD (NeurIPS'23)",         23.23, 0.852, None),
    ("MBLLIE-Net (SciRep'24)",    23.33, 0.829, 0.116),
    ("Diff-Retinex++ (TPAMI'25)", 24.67, 0.867, 0.101),
    ("TFFormer (CVPR'25)",        26.13, 0.888, 0.061),
    ("HFL (DCN'25)",              27.26, 0.930, 0.100),
]

def evaluate(args):
    os.chdir(ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    illum, reflec, refine = load_stage1(args.s1_ckpt, device)
    lpips_fn = get_lpips(device)

    # Stage 2
    denoiser = None
    sched    = None
    if args.stage == 2:
        if not os.path.exists(args.s2_ckpt):
            print(f"Stage 2 ckpt not found: {args.s2_ckpt}. Falling back to Stage 1.")
            args.stage = 1
        else:
            denoiser = DenoisingUNet(in_ch=13, out_ch=3, width=64).to(device).eval()
            c = torch.load(args.s2_ckpt, map_location=device, weights_only=True)
            denoiser.load_state_dict(c.get('ema', c.get('model', c)))
            sched = DDPMSchedule(T=1000, device=device)
            for p in denoiser.parameters(): p.requires_grad_(False)
            print(f"Stage 2 loaded: {args.s2_ckpt}")

    ds     = FullLOLDataset(args.data_root, split="eval15")
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    os.makedirs(args.out_dir, exist_ok=True)

    psnrs, ssims, lpipss = [], [], []

    for i, (inp7, gt, _, _) in enumerate(tqdm(loader, desc="Eval")):
        inp7, gt = inp7.to(device), gt.to(device)

        def run(x):
            s1 = s1_forward(illum, reflec, refine, x)
            if denoiser is None:
                return s1
            I_low = x[:,:3]; R_cnn = x[:,3:6]; L_tv = x[:,6:7]
            cond  = torch.cat([s1, I_low, R_cnn, L_tv], 1)
            pred_norm = sched.ddim_sample(denoiser, cond, steps=args.ddim_steps, eta=0.0)
            return ((pred_norm + 1) / 2).clamp(0,1)

        if args.tta:
            pred = tta_forward(run, inp7, device)
        else:
            with torch.no_grad():
                pred = run(inp7)

        psnrs.append(psnr_fn(pred, gt))
        ssims.append(ssim_fn(pred, gt))
        lpipss.append(lpips_fn(pred, gt))

        img = pred[0].cpu().numpy().transpose(1,2,0)*255
        cv2.imwrite(os.path.join(args.out_dir, f"{i:04d}.png"), img.astype(np.uint8))

    pm, sm, lm = np.mean(psnrs), np.mean(ssims), np.nanmean(lpipss)

    sep = "─"*65
    print(f"\n{sep}")
    print(f"{'Method':<34} {'PSNR':>6} {'SSIM':>7} {'LPIPS':>8}")
    print(sep)
    for name, p, s, l in SOTA_TABLE:
        ls = f"{l:.3f}" if l else "  —"
        print(f"{name:<34} {p:>6.2f} {s:>7.4f} {ls:>8}")
    print(sep)
    stage_str = f"Stage{args.stage}" + (" +TTA×8" if args.tta else "")
    our = f"Ours — RetinexPatch ({stage_str})"
    print(f"\033[1m{our:<34} {pm:>6.2f} {sm:>7.4f} {lm:>8.4f}\033[0m")
    print(sep)

    rank = sum(1 for x in SOTA_TABLE if x[1] > pm) + 1
    print(f"\nRanking: #{rank} / {len(SOTA_TABLE)+1} on LOL-v1 (by PSNR)")

    results = {"psnr":psnrs,"ssim":ssims,"lpips":lpipss,
               "mean":{"psnr":pm,"ssim":sm,"lpips":lm},
               "stage":args.stage,"tta":args.tta}
    with open(os.path.join(args.out_dir, "metrics.json"),'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {args.out_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage",      type=int, default=2, choices=[1,2])
    p.add_argument("--s1_ckpt",    default="checkpoints/retinex_patch/best_psnr.pth")
    p.add_argument("--s2_ckpt",    default="checkpoints/retinex_diffusion/best.pth")
    p.add_argument("--data_root",  default="datasets/LOL_dataset")
    p.add_argument("--out_dir",    default="results/retinex_final_eval")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--tta",        action="store_true")
    evaluate(p.parse_args())
