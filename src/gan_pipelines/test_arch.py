"""Quick test for architecture and losses."""
import torch
import sys

try:
    from models_pix2pix import IllumNetV2, ReflecNetV2, MultiScaleRefineBlock, DualScaleDiscriminator
    print("Models imported OK")

    x1 = torch.randn(1, 1, 384, 384).cuda()
    x3 = torch.randn(1, 3, 384, 384).cuda()

    illum = IllumNetV2(32).cuda()
    reflec = ReflecNetV2(64).cuda()
    refine = MultiScaleRefineBlock(64).cuda()
    disc = DualScaleDiscriminator(6, 64).cuda()

    I = illum(x1)
    R = reflec(x3)
    enh = refine(R * I)
    df, dh = disc(x3, enh)

    print(f"IllumNet: {x1.shape} -> {I.shape}")
    print(f"ReflecNet: {x3.shape} -> {R.shape}")
    print(f"Refined: {enh.shape}")
    print(f"Disc: full={df.shape}, half={dh.shape}")
    print(f"Params: Illum={sum(p.numel() for p in illum.parameters()):,}")
    print(f"Params: Reflec={sum(p.numel() for p in reflec.parameters()):,}")
    print(f"Params: Refine={sum(p.numel() for p in refine.parameters()):,}")
    print(f"Params: Disc={sum(p.numel() for p in disc.parameters()):,}")

    from losses import CombinedLossV2, ReflectanceLossV2
    print("\nLosses imported OK")

    gt = torch.randn(1, 3, 384, 384).cuda().clamp(0, 1)
    rl = ReflectanceLossV2().cuda()
    cl = CombinedLossV2().cuda()
    lr, _ = rl(R, gt)
    lc, _ = cl(enh, gt)
    print(f"ReflectanceLossV2: {lr.item():.4f}")
    print(f"CombinedLossV2: {lc.item():.4f}")

    print("\nALL TESTS PASSED ✅")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
