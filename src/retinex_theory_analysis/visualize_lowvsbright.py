import cv2
import numpy as np
import os

# --- Paths ---
low_dir = "../../datasets/LOL_dataset/eval15/low"
high_dir = "../../datasets/LOL_dataset/eval15/high"
result_dir = "../../results/decomposition"
output_dir = "../../results/lowvsbright_comparison"
os.makedirs(output_dir, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_COLOR = (255, 255, 255)
HEADER_BG = (40, 40, 40)
LABEL_W = 130
HEADER_H = 35
CELL_H, CELL_W = 200, 280


def text_bar(text, h, w, bg=HEADER_BG):
    bar = np.full((h, w, 3), bg, dtype=np.uint8)
    sz = cv2.getTextSize(text, FONT, FONT_SCALE, 1)[0]
    cv2.putText(bar, text, ((w - sz[0]) // 2, (h + sz[1]) // 2),
                FONT, FONT_SCALE, FONT_COLOR, 1, cv2.LINE_AA)
    return bar


def to_3ch(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


def load(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def cell(img):
    if img is None:
        c = np.full((CELL_H, CELL_W, 3), 30, dtype=np.uint8)
        cv2.putText(c, "-", (CELL_W // 2 - 8, CELL_H // 2 + 8),
                    FONT, 1.0, (80, 80, 80), 2)
        return c
    return cv2.resize(to_3ch(img), (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)


# --- Main ---
for img_name in sorted(os.listdir(low_dir)):
    b = os.path.splitext(img_name)[0]

    # Load original images
    low  = load(os.path.join(low_dir, img_name))
    high = load(os.path.join(high_dir, img_name))

    # Load decomposition results
    I_low      = load(f"{result_dir}/{b}_I_low.png")
    R_low      = load(f"{result_dir}/{b}_R_low.png")
    I_high     = load(f"{result_dir}/{b}_I_high.png")
    R_high     = load(f"{result_dir}/{b}_R_high.png")

    I_low_inv  = load(f"{result_dir}/{b}_I_low_inv.png")
    R_low_inv  = load(f"{result_dir}/{b}_R_low_inv.png")
    I_high_inv = load(f"{result_dir}/{b}_I_high_inv.png")
    R_high_inv = load(f"{result_dir}/{b}_R_high_inv.png")

    I_diff     = load(f"{result_dir}/{b}_I_diff.png")
    R_diff     = load(f"{result_dir}/{b}_R_diff.png")

    # Layout:
    # Columns: Original | Illumination | Reflectance | Illum (Inv) | Reflect (Inv)
    # Row 1: Low-Light
    # Row 2: Bright
    # Row 3: Difference (|Bright - Low|)

    col_titles = ["Original", "Illumination", "Reflectance", "Illum (Inv)", "Reflect (Inv)"]
    rows = [
        ("Low-Light",    [low,  I_low,  R_low,  I_low_inv,  R_low_inv]),
        ("Ground Truth", [high, I_high, R_high, I_high_inv, R_high_inv]),
        ("Difference",   [None, I_diff, R_diff, None,       None]),
    ]

    n_cols = len(col_titles)
    total_w = LABEL_W + n_cols * CELL_W

    title_bar = text_bar(f"Retinex Comparison: Low-Light vs Ground Truth  |  {img_name}", 45, total_w)
    corner = np.full((HEADER_H, LABEL_W, 3), HEADER_BG, dtype=np.uint8)
    header = np.hstack([corner] + [text_bar(t, HEADER_H, CELL_W) for t in col_titles])

    img_rows = []
    for label, imgs in rows:
        row_label = text_bar(label, CELL_H, LABEL_W)
        row_cells = [cell(img) for img in imgs]
        img_rows.append(np.hstack([row_label] + row_cells))

    final = np.vstack([title_bar, header] + img_rows)

    out_path = f"{output_dir}/{b}_lowvsbright.png"
    cv2.imwrite(out_path, final)
    print(f"  Saved: {out_path}")

print(f"\nAll Low vs Bright comparisons saved to: {output_dir}")
