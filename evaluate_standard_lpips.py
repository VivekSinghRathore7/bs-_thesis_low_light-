import torch
import torchvision.transforms as transforms
import lpips
import os
import cv2
import numpy as np

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

gt_dir = "datasets/LOL_v2_real/Test/GT"
p1_dir = "results/lolv2/gan_p1_images"
p2_dir = "results/lolv2/gan_p2_images"
p3_dir = "results/lolv2/gan_p3_images"

files = sorted(f for f in os.listdir(gt_dir) if f.endswith('.png'))

def get_lpips(pred_dir):
    dists = []
    for f in files:
        pred = cv2.imread(os.path.join(pred_dir, f))
        if pred is None: continue
        pred = pred[:,:,::-1]/255.0
        
        gt = cv2.imread(os.path.join(gt_dir, f))[:,:,::-1]
        # Resize GT to match exactly the prediction shape
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))/255.0
        
        # Convert to tensor and scale to [-1, 1] as required by official lpips
        gt_t = torch.from_numpy(gt).permute(2,0,1).unsqueeze(0).float() * 2 - 1
        pred_t = torch.from_numpy(pred).permute(2,0,1).unsqueeze(0).float() * 2 - 1
        
        with torch.no_grad():
            dist = loss_fn_vgg(pred_t.cuda(), gt_t.cuda())
        dists.append(dist.item())
    return np.mean(dists)

print(f"P1 LPIPS: {get_lpips(p1_dir):.4f}")
print(f"P2 LPIPS: {get_lpips(p2_dir):.4f}")
print(f"P3 LPIPS: {get_lpips(p3_dir):.4f}")
