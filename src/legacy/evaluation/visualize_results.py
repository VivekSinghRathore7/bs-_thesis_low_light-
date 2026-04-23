import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

input_dir = "./data/test/low/"
result_dir = "./test_results_trained/"
gt_dir = "./data/test/high/"
output_path = "./eval_visual.png"

rows = []
for fname in sorted(os.listdir(gt_dir)):
    name, ext = os.path.splitext(fname)
    input_path = os.path.join(input_dir, fname)
    res_path = os.path.join(result_dir, name + "_S" + ext)
    gt_path = os.path.join(gt_dir, fname)
    if not os.path.exists(res_path) or not os.path.exists(input_path):
        continue
    input_img = Image.open(input_path).convert('RGB')
    result_img = Image.open(res_path).convert('RGB')
    gt_img = Image.open(gt_path).convert('RGB')
    w, h = gt_img.size
    input_img = input_img.resize((w, h))
    result_img = result_img.resize((w, h))
    result_np = np.array(result_img)
    gt_np = np.array(gt_img)
    p = psnr(gt_np, result_np)
    s = ssim(gt_np, result_np, channel_axis=2)
    rows.append((fname, input_img, result_img, gt_img, p, s))

thumb_w, thumb_h = 400, 300
label_h = 30
pad = 5
cols = 3
total_w = cols * thumb_w + (cols + 1) * pad
total_h = len(rows) * (thumb_h + label_h + pad * 2) + pad

canvas = Image.new('RGB', (total_w, total_h), color=(240, 240, 240))
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

for i, (fname, inp, res, gt, p, s) in enumerate(rows):
    y = pad + i * (thumb_h + label_h + pad * 2)
    x0 = pad
    x1 = pad * 2 + thumb_w
    x2 = pad * 3 + thumb_w * 2
    canvas.paste(inp.resize((thumb_w, thumb_h)), (x0, y + label_h))
    canvas.paste(res.resize((thumb_w, thumb_h)), (x1, y + label_h))
    canvas.paste(gt.resize((thumb_w, thumb_h)), (x2, y + label_h))
    draw.text((x0 + 5, y + 5), f"Input: {fname}", fill=(50, 50, 50), font=font)
    draw.text((x1 + 5, y + 5), f"RetinexNet  PSNR={p:.2f}  SSIM={s:.4f}", fill=(0, 100, 0), font=font)
    draw.text((x2 + 5, y + 5), "Ground Truth", fill=(50, 50, 50), font=font)

canvas.save(output_path, quality=95)
print(f"Saved to {output_path}")
