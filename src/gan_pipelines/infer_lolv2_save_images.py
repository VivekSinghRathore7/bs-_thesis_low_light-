"""
Batch image enhancement script for LOL-v2.
"""

import os, sys, time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src", "newpipline", "llie_project"))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from train_eval_lolv2 import P1LOLv2Dataset, P2LOLv2Dataset, P3LOLv2Dataset
from models_p1_p3_unet import UNetGenerator, IllumNet, ReflecNet, RefineBlock

device = torch.device('cuda')
data_root = "datasets/LOL_v2_real"
out_base = "results/lolv2/metrics"

@torch.no_grad()
def save_gans():
    print("Saving P1 images...")
    p1_dir = os.path.join(out_base, "p1_images")
    os.makedirs(p1_dir, exist_ok=True)
    gen1 = UNetGenerator(in_ch=4, out_ch=3).to(device)
    gen1.load_state_dict(torch.load("checkpoints/lolv2/gan_p1/gen_final.pth", map_location=device, weights_only=True))
    gen1.eval()
    ds1 = P1LOLv2Dataset(data_root, split="Test", img_size=256) # Usually we evaluating exactly what we trained on
    for i in tqdm(range(len(ds1))):
        inp, _ = ds1[i]
        inp = inp.unsqueeze(0).to(device)
        pred = gen1(inp).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        img = (pred * 255).astype(np.uint8)
        fname = ds1.filenames[i]
        cv2.imwrite(os.path.join(p1_dir, fname), img)
        
    print("Saving P2 images...")
    p2_dir = os.path.join(out_base, "p2_images")
    os.makedirs(p2_dir, exist_ok=True)
    illum_net = IllumNet(base_filters=32).to(device)
    reflec_net = ReflecNet(base_filters=48).to(device)
    refine = RefineBlock(base_filters=48).to(device)
    ckpt2 = torch.load("checkpoints/lolv2/gan_p2/gen_final.pth", map_location=device, weights_only=True)
    illum_net.load_state_dict(ckpt2['illum_net'])
    reflec_net.load_state_dict(ckpt2['reflec_net'])
    refine.load_state_dict(ckpt2['refine'])
    illum_net.eval(); reflec_net.eval(); refine.eval()
    ds2 = P2LOLv2Dataset(data_root, split="Test", img_size=256)
    for i in tqdm(range(len(ds2))):
        R_low, I_low, _, _, _ = ds2[i]
        R_low = R_low.unsqueeze(0).to(device)
        I_low = I_low.unsqueeze(0).to(device)
        pred = refine(reflec_net(R_low) * illum_net(I_low)).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        img = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(p2_dir, ds2.filenames[i]), img)

    print("Saving P3 images...")
    p3_dir = os.path.join(out_base, "p3_images")
    os.makedirs(p3_dir, exist_ok=True)
    gen3 = UNetGenerator(in_ch=15, out_ch=3).to(device)
    gen3.load_state_dict(torch.load("checkpoints/lolv2/gan_p3/gen_final.pth", map_location=device, weights_only=True))
    gen3.eval()
    ds3 = P3LOLv2Dataset(data_root, split="Test", img_size=256)
    for i in tqdm(range(len(ds3))):
        inp, _ = ds3[i]
        inp = inp.unsqueeze(0).to(device)
        pred = gen3(inp).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
        img = (pred * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(p3_dir, ds3.filenames[i]), img)
        
    print("Done generating all LOL-v2 images!")

if __name__ == "__main__":
    save_gans()
