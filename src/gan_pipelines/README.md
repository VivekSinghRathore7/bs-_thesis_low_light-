# LLIE GAN Pipelines — Low-Light Image Enhancement

This directory contains three GAN-based pipelines for low-light image enhancement, combining Retinex theory with adversarial learning.

## 📂 Project Structure (As per Implementation Plan)

### 1. Models & Logic
- **[models_p1_p3_unet.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/models_p1_p3_unet.py)**: U-Net architectures for Pipeline 1 and 3, plus base blocks.
- **[models_p2_disentangled.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/models_p2_disentangled.py)**: Advanced architectures for Pipeline 2 (Disentangled).
- **[datasets_all_pipelines.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/datasets_all_pipelines.py)**: Data loaders for all pipelines (LOL-v1).
- **[losses_all_pipelines.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/losses_all_pipelines.py)**: GAN, SSIM, and VGG perceptual losses.
- **[retinex_utils.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/retinex_utils.py)**: Retinex decomposition and classical pre-processing.

### 2. LOL-v1 (Dataset 1)
- **[train_lolv1_p3_ensemble.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/train_lolv1_p3_ensemble.py)**: Primary training script for all 3 pipelines on LOL-v1.
- **[eval_lolv1_p1_p3.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/eval_lolv1_p1_p3.py)**: Evaluation and benchmarking on LOL-v1 test set.
- **[train_lolv1_p2_disentangled.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/train_lolv1_p2_disentangled.py)**: Specific advanced training for P2.
- **[eval_lolv1_p2_disentangled.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/eval_lolv1_p2_disentangled.py)**: Specific advanced eval for P2.
- **[metrics_lolv1_evaluation.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/metrics_lolv1_evaluation.py)**: Standalone metric tools.
- **[viz_lolv1_thesis_comparisons.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/viz_lolv1_thesis_comparisons.py)**: Visual comparisons for thesis.

### 3. LOL-v2 (Dataset 2)
- **[run_lolv2_all_pipelines.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/run_lolv2_all_pipelines.py)**: Unified script to train/eval all pipelines on LOL-v2 Real.
- **[train_lolv2_p2_disentangled.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/train_lolv2_p2_disentangled.py)**: Advanced P2 training for LOL-v2.
- **[infer_lolv2_save_images.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/infer_lolv2_save_images.py)**: Batch inference for LOL-v2.
- **[viz_lolv2_thesis_comparisons.py](file:///home/samiran_iiserb/asip_lab/UG/VIVEK/bs_thesis_low_light/src/gan_pipelines/viz_lolv2_thesis_comparisons.py)**: Visualization specifically for LOL-v2.

---

## 🚀 Execution

**Training (V1):**
```bash
python train_lolv1_p3_ensemble.py --pipeline all
```

**Training (V2):**
```bash
python run_lolv2_all_pipelines.py
```
