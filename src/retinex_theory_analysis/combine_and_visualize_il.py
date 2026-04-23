import cv2
import numpy as np
import os
import sys

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
low_dir = os.path.join(BASE_DIR, "datasets/LOL_dataset/eval15/low")
illum_dir = os.path.join(BASE_DIR, "results/illumination_enhancement")
output_dir = os.path.join(BASE_DIR, "results/combined_illumination_comparison")
os.makedirs(output_dir, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX

def text_bar(text, h, w):
    bar = np.full((h, w, 3), (40, 40, 40), dtype=np.uint8)
    sz = cv2.getTextSize(text, FONT, 0.6, 2)[0]
    cv2.putText(bar, text, ((w - sz[0]) // 2, (h + sz[1]) // 2),
                FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return bar

def resize_and_pad(img, cell_w=300, cell_h=200):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    return resized

def main():
    print("Starting Combination and Visualization...")
    if not os.path.exists(illum_dir):
        print(f"Error: Directory not found: {illum_dir}")
        sys.exit(1)

    # Get a list of unique base names
    files = sorted(os.listdir(illum_dir))
    base_names = sorted(list(set([f.split("_Il")[0] for f in files if "_Il" in f])))

    for base_name in base_names:
        low_path = os.path.join(low_dir, f"{base_name}.png")
        if not os.path.exists(low_path):
            continue

        low_img = cv2.imread(low_path)
        
        # Load Il0 through Il9
        Il0 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il0.png"), 0)
        Il1 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il1_CLAHE.png"), 0)
        Il2 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il2_BBHE.png"), 0)
        Il3 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il3_RMSHE.png"), 0)
        Il4 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il4_AGC.png"), 0)
        Il5 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il5_AGCWHD.png"), 0)
        Il6 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il6_GLAGC.png"), 0)
        Il7 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il7_MinMax.png"), 0)
        Il8 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il8_PWLC.png"), 0)
        Il9 = cv2.imread(os.path.join(illum_dir, f"{base_name}_Il9_Log.png"), 0)

        images = [Il1, Il2, Il3, Il4, Il5, Il6, Il7, Il8, Il9]
        if any(i is None for i in images) or Il0 is None:
            print(f"Missing images for {base_name}. Skipping.")
            continue

        # Combine Il1 to Il9 into Il' (Average Ensemble)
        stacked = np.stack([img.astype(np.float32) for img in images], axis=0)
        Il_prime = np.mean(stacked, axis=0)
        Il_prime = np.clip(Il_prime, 0, 255).astype(np.uint8)

        # Save Il' back to illumination_enhancement directory
        prime_path = os.path.join(illum_dir, f"{base_name}_Il_prime.png")
        cv2.imwrite(prime_path, Il_prime)

        # ==========================================
        # Create Montage (Grid: 3 rows x 4 columns)
        # ==========================================
        cell_w, cell_h = 300, 200
        header_h = 40

        cells = [
            (low_img, "Original Low-Light"),
            (Il0, "Il0 (Raw Illum)"),
            (Il1, "Il1 (CLAHE)"),
            (Il2, "Il2 (BBHE)"),
            
            (Il3, "Il3 (RMSHE)"),
            (Il4, "Il4 (AGC)"),
            (Il5, "Il5 (AGCWHD)"),
            (Il6, "Il6 (GLAGC)"),
            
            (Il7, "Il7 (MinMax Stretch)"),
            (Il8, "Il8 (PWLC)"),
            (Il9, "Il9 (Log Transform)"),
            (Il_prime, "Il' (Combined Avg)"),
        ]

        rows = []
        for i in range(0, 12, 4):
            row_images = []
            for j in range(4):
                img, label = cells[i+j]
                r_img = resize_and_pad(img, cell_w, cell_h)
                header = text_bar(label, header_h, cell_w)
                cell = np.vstack([header, r_img])
                row_images.append(cell)
            rows.append(np.hstack(row_images))

        montage = np.vstack(rows)

        # Add a main title
        title_bar = text_bar(f"Illumination Enhancements Comparison | {base_name}", 50, cell_w * 4)
        final_img = np.vstack([title_bar, montage])

        out_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        cv2.imwrite(out_path, final_img)
        print(f"Saved: {out_path}")

    print(f"\nAll combination mappings (Il') saved to: {illum_dir}")
    print(f"All comparison grids saved to: {output_dir}")

if __name__ == "__main__":
    main()
