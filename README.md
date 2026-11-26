# Skin Cancer Detection from Dermoscopic Images Using Deep Learning

An end-to-end **PyTorch** transfer learning pipeline for **9-class skin lesion classification** on the **ISIC** dermoscopic image dataset, featuring two-stage fine-tuning, class-imbalance weighting, multi-class ROC-AUC evaluation, and **Grad-CAM** visual explainability.

---

## Project Overview

- **Authors:** Debayudh Kundu (2351062) & Aditiya Majumdar (2351058) — Group B55, Dept. of CSE (2025–2026)
- **Backbone:** Pretrained ImageNet **MobileNetV2** (also supports `efficientnet_b0` and `resnet50` via CLI)
- **Dataset:** ISIC 9-Class Skin Cancer Dataset (`2,239` training images split 85/15 stratified into `1,903` train / `336` validation + `118` held-out test images)

### 9 Skin Lesion Classes
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

---

## Pipeline Architecture

1. **Preprocessing & Augmentation (`src/dataset.py`)**
   - Training: `RandomResizedCrop(224)`, random horizontal & vertical flips, random rotation (`±20°`), color jitter, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
   - Validation / Test: Resize to `224×224` + ImageNet normalization.
   - Class imbalance handling: Smoothed inverse-frequency class weights (`power=0.5`) passed into `CrossEntropyLoss(label_smoothing=0.05)`.

2. **Two-Stage Transfer Learning (`src/train.py`, `src/model.py`)**
   - **Stage 1 (Warmup):** Freeze backbone, train `Dropout(0.3) -> Linear(1280, 9)` head with `AdamW (lr=1e-3)`.
   - **Stage 2 (Fine-Tuning):** Unfreeze top 5 inverted residual blocks (`features[14:]`) + classifier with differential learning rates (`1e-4` backbone, `5e-4` head) and `CosineAnnealingLR`.

3. **Evaluation & Explainability (`src/evaluate.py`, `src/gradcam.py`, `src/predict.py`)**
   - Saves training curves, validation/test confusion matrices, one-vs-rest ROC curves, per-class metrics (`outputs/metrics.json`), and Grad-CAM heatmaps.

---

## Results (MobileNetV2 — 4 Warmup + 6 Fine-Tuning Epochs on CPU)

| Metric | Previous Baseline (Frozen Only) | New 2-Stage Pipeline |
| :--- | :---: | :---: |
| **Train Accuracy** | 44.1% | **74.5%** |
| **Validation Accuracy (n=336)** | 35.6% | **66.07%** |
| **Validation Macro ROC-AUC** | — | **0.9275** |
| **Held-Out Test Accuracy (n=118)** | — | **59.32%** |
| **Held-Out Test Macro ROC-AUC** | — | **0.9052** |

### Visual Outputs
- **Training History:** `outputs/training_history.png`
- **Validation Confusion Matrix:** `outputs/confusion_matrix_val.png`
- **Test Confusion Matrix:** `outputs/confusion_matrix_test.png`
- **ROC Curves (One-vs-Rest):** `outputs/roc_curves_val.png`
- **Grad-CAM Explanations:** `outputs/gradcam_samples.png`

---

## Quick Start

### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Dataset
Extract `Skin_ISIC.zip` into `data/` so that the structure is:
```
data/Skin cancer ISIC The International Skin Imaging Collaboration/
├── Train/
└── Test/
```

### 3. Train & Evaluate
```powershell
# Default MobileNetV2 (2-stage transfer learning)
python -m src.train --model mobilenet_v2 --stage1-epochs 4 --stage2-epochs 6

# Or train longer / try other backbones (GPU recommended for ResNet50 / EfficientNet-B0)
python -m src.train --model efficientnet_b0 --stage1-epochs 5 --stage2-epochs 15
```

### 4. Predict on a Single Dermoscopic Image (+ Grad-CAM)
```powershell
python -m src.predict path\to\lesion.jpg --gradcam-out outputs\pred_gradcam.png
```

---

## Repository Structure

```
.
├── src/
│   ├── config.py        # Paths, hyperparameters, device selection
│   ├── dataset.py       # Stratified splits, augmentations, class weights
│   ├── model.py         # MobileNetV2 / EfficientNet-B0 / ResNet50 builders
│   ├── train.py         # 2-stage training loop + checkpointing
│   ├── evaluate.py      # Metrics, confusion matrices, ROC curves, Grad-CAM grid
│   ├── gradcam.py       # Grad-CAM implementation
│   └── predict.py       # Single-image CLI inference
├── outputs/             # Saved plots and metrics.json
├── requirements.txt
└── README.md
```
