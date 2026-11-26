"""Soft-voting deep ensemble with 4x Test-Time Augmentation (TTA) and binary screening metrics."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, MALIGNANT_CLASSES, OUTPUT_DIR, get_device
from src.dataset import create_dataloaders, get_transforms
from src.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
)
from src.gradcam import save_gradcam_grid
from src.model import build_model


@torch.no_grad()
def predict_loader_tta(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run 4x flip TTA inference on a DataLoader and return (y_true, y_probs)."""
    model.eval()
    all_true, all_probs = [], []
    for images, labels in loader:
        images = images.to(device)
        p0 = F.softmax(model(images), dim=1)
        p1 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
        p2 = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
        p3 = F.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)
        probs = ((p0 + p1 + p2 + p3) / 4.0).cpu().numpy()
        all_true.extend(labels.numpy().tolist())
        all_probs.append(probs)
    return np.array(all_true), np.vstack(all_probs)


def compute_binary_screening_metrics(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str]) -> Dict:
    """Aggregate 9-class probabilities into Malignant/Pre-malignant vs Benign screening metrics."""
    mal_indices = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    y_true_bin = np.isin(y_true, mal_indices).astype(int)
    p_mal = y_probs[:, mal_indices].sum(axis=1)
    y_pred_bin = (p_mal >= 0.5).astype(int)

    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "malignant_classes": sorted(list(MALIGNANT_CLASSES)),
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1_score": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_bin, p_mal)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_ensemble(checkpoint_paths: List[Path], output_dir: Path = OUTPUT_DIR) -> Dict:
    device = get_device()
    data = create_dataloaders(batch_size=32)
    class_names = data["class_names"]

    individual_results = {}
    val_prob_list, test_prob_list, weights = [], [], []
    best_single_model = None
    best_single_arch = None
    best_single_acc = -1.0
    best_history = None

    for ckpt_path in checkpoint_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch = ckpt["arch"]
        model = build_model(arch=arch, num_classes=len(class_names), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        val_true, val_probs = predict_loader_tta(model, data["val_loader"], device)
        test_true, test_probs = predict_loader_tta(model, data["test_loader"], device)

        val_acc = float(accuracy_score(val_true, val_probs.argmax(axis=1)))
        test_acc = float(accuracy_score(test_true, test_probs.argmax(axis=1)))

        individual_results[arch] = {
            "checkpoint": ckpt_path.name,
            "val_accuracy_tta": val_acc,
            "test_accuracy_tta": test_acc,
        }
        print(f"  [{arch}] Val Acc (4x TTA): {val_acc*100:.2f}% | Test Acc (4x TTA): {test_acc*100:.2f}%", flush=True)

        val_prob_list.append(val_probs)
        test_prob_list.append(test_probs)
        weights.append(val_acc)

        if val_acc > best_single_acc:
            best_single_acc = val_acc
            best_single_model = model
            best_single_arch = arch
            best_history = ckpt.get("history")

    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    ens_val_probs = sum(wi * p for wi, p in zip(w, val_prob_list))
    ens_test_probs = sum(wi * p for wi, p in zip(w, test_prob_list))
    ens_val_pred = ens_val_probs.argmax(axis=1)
    ens_test_pred = ens_test_probs.argmax(axis=1)

    val_metrics = compute_metrics(val_true, ens_val_pred, ens_val_probs, class_names)
    test_metrics = compute_metrics(test_true, ens_test_pred, ens_test_probs, class_names)
    val_binary = compute_binary_screening_metrics(val_true, ens_val_probs, class_names)
    test_binary = compute_binary_screening_metrics(test_true, ens_test_probs, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    if best_history is not None:
        plot_training_history(best_history, output_dir / "training_history.png")

    ens_label = " + ".join(individual_results.keys())
    plot_confusion_matrix(
        val_true, ens_val_pred, class_names, output_dir / "confusion_matrix_val.png",
        title=f"Validation Confusion Matrix — Ensemble ({ens_label}, n={len(val_true)})",
    )
    plot_confusion_matrix(
        test_true, ens_test_pred, class_names, output_dir / "confusion_matrix_test.png",
        title=f"Held-Out Test Confusion Matrix — Ensemble ({ens_label}, n={len(test_true)})",
    )
    plot_roc_curves(val_true, ens_val_probs, class_names, output_dir / "roc_curves_val.png")

    _, eval_tf = get_transforms()
    save_gradcam_grid(
        model=best_single_model,
        arch=best_single_arch,
        samples=data["test_samples"],
        class_names=class_names,
        eval_transform=eval_tf,
        device=device,
        save_path=output_dir / "gradcam_samples.png",
        max_samples=6,
    )

    summary = {
        "ensemble_models": individual_results,
        "ensemble_weights": {k: float(wi) for k, wi in zip(individual_results.keys(), w)},
        "tta": "4x_flips",
        "dataset": {
            "train_samples": data["train_size"],
            "val_samples": data["val_size"],
            "test_samples": data["test_size"],
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        "validation_metrics_9class": val_metrics,
        "test_metrics_9class": test_metrics,
        "validation_metrics_binary_screening": val_binary,
        "test_metrics_binary_screening": test_binary,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ ENSEMBLE SUMMARY ================")
    print(f"Models               : {ens_label}")
    print(f"9-Class Val  Acc     : {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']:.4f} | Macro AUC: {val_metrics['macro_roc_auc']:.4f}")
    print(f"9-Class Test Acc     : {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f} | Macro AUC: {test_metrics['macro_roc_auc']:.4f}")
    print(f"Binary  Val  Acc     : {val_binary['accuracy']*100:.2f}% | Sens: {val_binary['sensitivity']*100:.1f}% | Spec: {val_binary['specificity']*100:.1f}% | AUC: {val_binary['roc_auc']:.4f}")
    print(f"Binary  Test Acc     : {test_binary['accuracy']*100:.2f}% | Sens: {test_binary['sensitivity']*100:.1f}% | Spec: {test_binary['specificity']*100:.1f}% | AUC: {test_binary['roc_auc']:.4f}")
    print("==================================================")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate soft-voting ensemble with 4x TTA.")
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    evaluate_ensemble(args.checkpoints, args.output_dir)


if __name__ == "__main__":
    main()
