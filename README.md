# Hierarchical Skin Cancer Detection System (Deep Learning + Grad-CAM)

A **Two-Stage Hierarchical Clinical Decision Support System** for dermoscopic skin lesion analysis on the **ISIC** dataset, combining:
1. **Stage 1 — Hierarchical Binary Triage (Malignant/Pre-malignant vs. Benign):** Fuses a dedicated binary **EfficientNet-B0** classifier with the 9-class ensemble (`EfficientNet-B0 + MobileNetV2`) using **4x Test-Time Augmentation (TTA)** to output a clinical risk tier (**Low / Moderate / High Risk**).
2. **Stage 2 — 9-Class Fine-Grained Subtype Ensemble:** Classifies the exact lesion subtype across all 9 ISIC categories.
3. **Stage 3 — Grad-CAM Visual Explainability:** Highlights the lesion regions responsible for the model's prediction.

- **Authors:** Debayudh Kundu (2351062) & Aditiya Majumdar (2351058) — Group B55, Dept. of CSE (2025–2026)

---

## Experimental Results (ISIC 9-Class Dataset: 1,903 Train / 336 Val / 118 Test)

### Stage 1: Binary Clinical Screening (Malignant/Pre-Malignant vs. Benign)

| Model / Strategy | Val Accuracy | Sensitivity (Cancer Recall) | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Standalone Binary EfficientNet-B0 (4x TTA) | 77.98% | 86.1% | 70.0% | 0.8445 |
| 9-Class Ensemble Aggregated to Binary (4x TTA) | 80.36% | 84.3% | 76.5% | 0.8553 |
| **Hierarchical Fusion (Binary + 9-Class Ensemble, 4x TTA)** | **81.55%** | **89.2%** | **74.1%** | **0.8576** |

### Stage 2: 9-Class Fine-Grained Subtype Classification

| Model / Strategy | 9-Class Val Acc | 9-Class Macro F1 | 9-Class Macro ROC-AUC |
| :--- | :---: | :---: | :---: |
| Original Slides Baseline (Frozen MobileNetV2, 10 ep) | 35.60% | — | — |
| MobileNetV2 (2-Stage Top-Block Fine-Tune) | 66.07% | 0.5754 | 0.9275 |
| EfficientNet-B0 (Full Fine-Tune + 4x TTA) | 69.64% | — | — |
| **Ensemble: EfficientNet-B0 + MobileNetV2 (4x TTA)** | **71.73%** | **0.6125** | **0.9390** |

> **Held-Out Test (`n=118`):** 9-Class Macro ROC-AUC = **0.9272** | Hierarchical Binary Specificity = **81.5%** (AUC **0.8073**).

---

## Visual Outputs (`outputs/`)
- `outputs/binary_triage_evaluation.png` — Stage-1 Hierarchical Triage Confusion Matrix & ROC Curve
- `outputs/training_history.png` — Training & validation loss/accuracy curves
- `outputs/confusion_matrix_val.png` & `outputs/confusion_matrix_test.png` — 9-class confusion matrices
- `outputs/roc_curves_val.png` — Multi-class one-vs-rest ROC curves
- `outputs/gradcam_samples.png` — Grad-CAM visual explanations on held-out test lesions
- `outputs/hierarchical_triage_metrics.json` & `outputs/metrics.json` — Full quantitative metrics

---

## Quick Start

### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Prepare Dataset (Pre-Resize for Fast Training)
Extract `Skin_ISIC.zip` into `data/`, then run:
```powershell
python -m src.prepare_data
```

### 3. Train & Evaluate the Hierarchical System
```powershell
# Stage 2 backbones (9-class)
python -m src.train --model efficientnet_b0 --unfreeze all --stage1-epochs 3 --stage2-epochs 12 --skip-eval
python -m src.train --model mobilenet_v2 --unfreeze all --stage1-epochs 3 --stage2-epochs 12 --skip-eval
python -m src.ensemble --checkpoints checkpoints/efficientnet_b0_best.pth checkpoints/best_model.pth

# Stage 1 dedicated binary triage + hierarchical fusion
python -m src.train_binary --model efficientnet_b0 --stage1-epochs 2 --stage2-epochs 5
```

### 4. Run Unified Hierarchical Clinical Prediction (+ Grad-CAM)
```powershell
python -m src.predict_hierarchical path\to\lesion.jpg --gradcam-out outputs\pred_gradcam.png
```

### 5. Scale Up on Free Google Colab T4 GPU
Open `notebooks/train_colab_ham10000.ipynb` in Google Colab (`Runtime -> Change runtime type -> T4 GPU`) to train the full **3-model ensemble (`EfficientNet-B0 + DenseNet121 + MobileNetV2`)** in ~5–8 minutes.
