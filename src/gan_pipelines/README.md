# LLIE Pipelines — Low-Light Image Enhancement with Retinex + GAN

Three novel pipelines that combine classical Retinex decomposition with GAN-based enhancement.

## Project Structure

```
llie_project/
├── retinex_utils.py    # Retinex decomposition + classical enhancement methods
├── dataset.py          # PyTorch datasets for all 3 pipelines
├── models.py           # Generator & Discriminator architectures
├── losses.py           # L1, SSIM, VGG Perceptual, Adversarial losses
├── train.py            # Training script (all 3 pipelines)
├── evaluate.py         # Evaluation + comparison graphs
└── requirements.txt
```

## Dataset Layout

```
datasets/LOL_dataset/
├── our485/          # Training set (485 pairs)
│   ├── low/
│   └── high/
└── eval15/          # Test set (15 pairs)
    ├── low/
    └── high/
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 3 pipelines
python train.py --pipeline all --epochs 100 --data_root datasets/LOL_dataset

# Train a single pipeline
python train.py --pipeline 1 --epochs 100
python train.py --pipeline 2 --epochs 100
python train.py --pipeline 3 --epochs 100

# Evaluate all + generate comparison graphs
python evaluate.py --data_root datasets/LOL_dataset --checkpoint_dir checkpoints
```

## Pipelines

| # | Name | Input to GAN | Key Idea |
|---|------|-------------|----------|
| 1 | Dual-Input Conditional GAN | RGB + enhanced illumination (4ch) | Illumination map as explicit prior |
| 2 | Disentangled GAN | Separate R and I branches | Component-level supervision |
| 3 | Ensemble GAN | RGB + 4 candidate images (15ch) | Classical ensemble as GAN prior |

## Outputs

After evaluation, `results/` will contain:
- `metrics.json` — per-image PSNR, SSIM, LPIPS for each pipeline
- `graphs/compare_psnr.png` — bar chart comparing average PSNR
- `graphs/compare_ssim.png` — bar chart comparing average SSIM
- `graphs/compare_lpips.png` — bar chart comparing average LPIPS
- `graphs/per_image_*.png` — per-image metric curves
- `graphs/summary_comparison.png` — combined comparison chart
- `graphs/training_loss.png` — training loss curves
- `pipeline*_outputs/` — enhanced images from each pipeline

## Training Options

```
--pipeline    1 | 2 | 3 | all     (default: all)
--epochs      int                  (default: 100)
--batch_size  int                  (default: 4)
--lr          float                (default: 2e-4)
--img_size    int                  (default: 256)
--save_every  int                  (default: 20)
```
