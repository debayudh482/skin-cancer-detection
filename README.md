# Skin Cancer Detection from Dermoscopic Images Using Deep Learning

An end-to-end **PyTorch** transfer learning and deep ensemble pipeline for **skin lesion classification** on the **ISIC** dermoscopic image dataset, featuring two-stage full fine-tuning, 4x Test-Time Augmentation (TTA), weighted soft-voting ensembling, binary malignant-vs-benign screening, and **Grad-CAM** visual explainability.

---

## Project Overview

- **Authors:** Debayudh Kundu (2351062) & Aditiya Majumdar (2351058) — Group B55, Dept. of CSE (2025–2026)
- **Ensemble Backbones:** **EfficientNet-B0 + MobileNetV2** (also supports `densenet121` and `resnet50` via CLI)
- **Dataset:** ISIC 9-Class Skin Cancer Dataset (`2,239` training images split 85/15 stratified into `1,903` train / `336` validation + `118` held-out test images)

### 9 Skin Lesion Classes
- **Malignant / Pre-Malignant:** Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma, Actinic Keratosis
- **Benign:** Nevus, Pigmented Benign Keratosis, Seborrheic Keratosis, Dermatofibroma, Vascular Lesion

---

## Progression & Results

| Model / Strategy | 9-Class Val Acc | 9-Class Macro ROC-AUC | Binary Screening Val Acc (Mal vs Ben) | Binary Sensitivity / Specificity |
| :--- | :---: | :---: | :---: | :---: |
| Slides Baseline (Frozen MobileNetV2, 10 ep) | 35.60% | — | — | — |
| MobileNetV2 (2-Stage Top-Block Fine-Tune) | 66.07% | 0.9275 | — | — |
| EfficientNet-B0 (Full Fine-Tune + 4x TTA) | 69.64% | — | — | — |
| **Ensemble: EfficientNet-B0 + MobileNetV2 (4x TTA)** | **71.73%** | **0.9390** | **80.36%** | **84.3% / 76.5%** |

- **Held-Out Test (`n=118`):** 9-Class Macro ROC-AUC = **0.9272**, Binary Screening Accuracy = **70.34%** (Specificity **83.3%**, AUC **0.8192**).

### Visual Outputs (`outputs/`)
- `outputs/training_history.png` — Training & validation loss/accuracy curves
- `outputs/confusion_matrix_val.png` & `outputs/confusion_matrix_test.png` — 9-class confusion matrices
- `outputs/roc_curves_val.png` — Multi-class one-vs-rest ROC curves
- `outputs/gradcam_samples.png` — Grad-CAM visual explanations on held-out test lesions
- `outputs/metrics.json` — Full per-class and binary screening metrics

---

## Quick Start

### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Dataset (Pre-Resize for Fast Training)
Extract `Skin_ISIC.zip` into `data/`, then pre-resize images once to `256×256`:
```powershell
python -m src.prepare_data
```

### 3. Train Individual Models & Run Ensemble with 4x TTA
```powershell
# Train EfficientNet-B0 (full fine-tuning)
python -m src.train --model efficientnet_b0 --unfreeze all --stage1-epochs 3 --stage2-epochs 12 --skip-eval

# Train MobileNetV2 (full fine-tuning)
python -m src.train --model mobilenet_v2 --unfreeze all --stage1-epochs 3 --stage2-epochs 12 --skip-eval

# Optional 3rd backbone: DenseNet121
python -m src.train --model densenet121 --unfreeze all --stage1-epochs 3 --stage2-epochs 12 --skip-eval

# Evaluate weighted soft-voting ensemble with 4x TTA
python -m src.ensemble --checkpoints checkpoints/efficientnet_b0_best.pth checkpoints/mobilenet_v2_best.pth
```

### 4. Single-Image Prediction + Grad-CAM
```powershell
python -m src.predict path\to\lesion.jpg --checkpoint checkpoints/efficientnet_b0_best.pth --gradcam-out outputs\pred_gradcam.png
```
