import cv2
import numpy as np
import os

# --- Paths ---
high_dir = "../../datasets/LOL_dataset/eval15/high"
enhanced_dir = "../../results/enhanced"
output_dir = "../../results/gt_vs_enhanced"
os.makedirs(output_dir, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_COLOR = (255, 255, 255)
HEADER_BG = (40, 40, 40)
HEADER_H = 40
CELL_H, CELL_W = 300, 400


def text_bar(text, h, w, font_scale=0.65):
    bar = np.full((h, w, 3), HEADER_BG, dtype=np.uint8)
    sz = cv2.getTextSize(text, FONT, font_scale, 2)[0]
    cv2.putText(bar, text, ((w - sz[0]) // 2, (h + sz[1]) // 2),
                FONT, font_scale, FONT_COLOR, 2, cv2.LINE_AA)
    return bar


for img_name in sorted(os.listdir(high_dir)):
    base = os.path.splitext(img_name)[0]

    gt  = cv2.imread(os.path.join(high_dir, img_name))
    enh = cv2.imread(os.path.join(enhanced_dir, f"{base}_enhanced.png"))

    if gt is None or enh is None:
        print(f"  Skipping {img_name}")
        continue

    # Resize to cell size
    gt_r  = cv2.resize(gt,  (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
    enh_r = cv2.resize(enh, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)

    # Headers
    total_w = CELL_W * 2
    title = text_bar(f"Ground Truth vs Enhanced  |  {img_name}", 45, total_w, font_scale=0.6)
    header = np.hstack([
        text_bar("Ground Truth", HEADER_H, CELL_W),
        text_bar("Enhanced", HEADER_H, CELL_W),
    ])

    # Side by side
    pair = np.hstack([gt_r, enh_r])
    final = np.vstack([title, header, pair])

    out_path = f"{output_dir}/{base}_gt_vs_enh.png"
    cv2.imwrite(out_path, final)
    print(f"  Saved: {out_path}")

print(f"\nAll comparisons saved to: {output_dir}")
