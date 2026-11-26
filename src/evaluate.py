"""Evaluation utilities: classification report, confusion matrix, ROC-AUC, and training curves."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, OUTPUT_DIR, get_device
from src.dataset import create_dataloaders, get_transforms
from src.gradcam import save_gradcam_grid
from src.model import build_model


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (y_true, y_pred, y_probs)."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(preds.tolist())
        all_probs.append(probs)

    return np.array(all_true), np.array(all_pred), np.vstack(all_probs)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, class_names: List[str]) -> Dict:
    """Compute overall and per-class classification metrics."""
    num_classes = len(class_names)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        macro_auc = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auc = float("nan")

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_roc_auc": macro_auc,
        "classification_report": report,
    }


def plot_training_history(history: Dict, save_path: Path) -> None:
    """Plot training and validation loss/accuracy curves across epochs."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    stage1_end = history.get("stage1_epochs", 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=4)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", markersize=4)
    if stage1_end and stage1_end < len(epochs):
        ax1.axvline(stage1_end + 0.5, color="gray", linestyle="--", label="Fine-tuning Start")
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", marker="o", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", marker="s", markersize=4)
    if stage1_end and stage1_end < len(epochs):
        ax2.axvline(stage1_end + 0.5, color="gray", linestyle="--", label="Fine-tuning Start")
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], save_path: Path, title: str) -> None:
    """Save a labeled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Plot one-vs-rest ROC curves for all 9 skin lesion classes."""
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc_val = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC = {auc_val:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_evaluation(checkpoint_path: Path, output_dir: Path = OUTPUT_DIR) -> Dict:
    """Load a saved checkpoint and run full validation + test evaluation with plots."""
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    class_names = ckpt["class_names"]

    data = create_dataloaders(batch_size=32)
    model = build_model(arch=arch, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_true, val_pred, val_probs = predict_loader(model, data["val_loader"], device)
    test_true, test_pred, test_probs = predict_loader(model, data["test_loader"], device)

    val_metrics = compute_metrics(val_true, val_pred, val_probs, class_names)
    test_metrics = compute_metrics(test_true, test_pred, test_probs, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    if "history" in ckpt:
        plot_training_history(ckpt["history"], output_dir / "training_history.png")

    plot_confusion_matrix(
        val_true, val_pred, class_names, output_dir / "confusion_matrix_val.png",
        title=f"Validation Confusion Matrix ({arch}, n={len(val_true)})",
    )
    plot_confusion_matrix(
        test_true, test_pred, class_names, output_dir / "confusion_matrix_test.png",
        title=f"Held-Out Test Confusion Matrix ({arch}, n={len(test_true)})",
    )
    plot_roc_curves(val_true, val_probs, class_names, output_dir / "roc_curves_val.png")

    _, eval_tf = get_transforms()
    save_gradcam_grid(
        model=model,
        arch=arch,
        samples=data["test_samples"],
        class_names=class_names,
        eval_transform=eval_tf,
        device=device,
        save_path=output_dir / "gradcam_samples.png",
        max_samples=6,
    )

    summary = {
        "architecture": arch,
        "device": str(device),
        "dataset": {
            "train_samples": data["train_size"],
            "val_samples": data["val_size"],
            "test_samples": data["test_size"],
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation Summary ===")
    print(f"Architecture     : {arch}")
    print(f"Val  Accuracy    : {val_metrics['accuracy']*100:.2f}% | Macro F1: {val_metrics['macro_f1']:.4f} | Macro AUC: {val_metrics['macro_roc_auc']:.4f}")
    print(f"Test Accuracy    : {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f} | Macro AUC: {test_metrics['macro_roc_auc']:.4f}")
    print(f"Saved outputs to : {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained skin cancer classifier.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
