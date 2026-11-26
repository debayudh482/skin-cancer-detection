"""Stage-1 Dedicated Binary Clinical Triage Classifier (Malignant/Pre-malignant vs Benign)
with Hierarchical Fusion (combining dedicated Binary model + 9-Class Ensemble + 4x TTA).
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINT_DIR,
    MALIGNANT_CLASSES,
    OUTPUT_DIR,
    RANDOM_SEED,
    get_device,
)
from src.dataset import SubsetWithTransform, create_dataloaders, get_transforms
from src.ensemble import predict_loader_tta
from src.model import build_model, freeze_backbone, unfreeze_layers


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_binary_loaders(batch_size: int = 32, seed: int = RANDOM_SEED) -> Dict:
    """Reuse the exact same stratified split as 9-class, mapping labels to 0=Benign, 1=Malignant."""
    base = create_dataloaders(batch_size=batch_size, seed=seed)
    class_names = base["class_names"]
    mal_ids = {i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES}

    def remap(samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        return [(p, 1 if lbl in mal_ids else 0) for p, lbl in samples]

    train_tf, eval_tf = get_transforms()
    train_samples = remap(base["train_loader"].dataset.samples)
    val_samples = remap(base["val_loader"].dataset.samples)
    test_samples = remap(base["test_loader"].dataset.samples)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(SubsetWithTransform(train_samples, train_tf), batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(SubsetWithTransform(val_samples, eval_tf), batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(SubsetWithTransform(test_samples, eval_tf), batch_size=batch_size, shuffle=False, pin_memory=pin)

    pos = sum(lbl for _, lbl in train_samples)
    neg = len(train_samples) - pos
    class_weights = torch.tensor([len(train_samples) / (2.0 * neg), len(train_samples) / (2.0 * pos)], dtype=torch.float32)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "class_names_9": class_names,
        "base_9class_loaders": base,
    }


def run_binary_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            total_loss += loss.item() * images.size(0)
            y_true.extend(labels.cpu().numpy().tolist())
            y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_prob))
    return total_loss / max(len(y_true), 1), acc, auc


def binary_metrics_dict(y_true: np.ndarray, p_mal: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (p_mal >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, p_mal)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def save_binary_plots(y_true: np.ndarray, p_mal: np.ndarray, threshold: float, output_dir: Path) -> None:
    y_pred = (p_mal >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant/Pre-Mal"],
        yticklabels=["Benign", "Malignant/Pre-Mal"],
        ax=ax1,
    )
    ax1.set_title(f"Stage-1 Binary Triage Confusion Matrix (n={len(y_true)})")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")

    fpr, tpr, _ = roc_curve(y_true, p_mal)
    auc_val = roc_auc_score(y_true, p_mal)
    ax2.plot(fpr, tpr, lw=2.2, color="#d62728", label=f"Hierarchical Triage (AUC = {auc_val:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax2.set_xlabel("False Positive Rate (1 - Specificity)")
    ax2.set_ylabel("True Positive Rate (Sensitivity)")
    ax2.set_title("Stage-1 Binary Triage ROC Curve")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "binary_triage_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_fuse(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Training Stage-1 Binary Triage ({args.model}) on {device}", flush=True)

    bundle = make_binary_loaders(batch_size=args.batch_size, seed=args.seed)
    model = build_model(arch=args.model, num_classes=2, dropout=0.3, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=bundle["class_weights"].to(device), label_smoothing=0.03)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "binary_triage_best.pth"
    best_score = -1.0

    # Stage 1: Warmup
    freeze_backbone(model, args.model)
    opt1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-2)
    print(f"\n--- Stage 1: Frozen Warmup ({args.stage1_epochs} epochs) ---", flush=True)
    for ep in range(1, args.stage1_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_binary_epoch(model, bundle["train_loader"], criterion, device, opt1)
        va_loss, va_acc, va_auc = run_binary_epoch(model, bundle["val_loader"], criterion, device)
        saved = ""
        if va_acc > best_score:
            best_score = va_acc
            torch.save({"arch": args.model, "val_acc": va_acc, "val_auc": va_auc, "model_state_dict": model.state_dict()}, ckpt_path)
            saved = " *best*"
        print(f"[Bin S1 {ep:02d}/{args.stage1_epochs:02d}] tr_acc={tr_acc*100:.1f}% | val_acc={va_acc*100:.1f}% val_auc={va_auc:.4f} ({time.time()-t0:.1f}s){saved}", flush=True)

    # Stage 2: Full Fine-tuning
    backbone_params, head_params = unfreeze_layers(model, args.model, mode="all")
    opt2 = torch.optim.AdamW(
        [{"params": backbone_params, "lr": 1.5e-4}, {"params": head_params, "lr": 5e-4}],
        weight_decay=1e-2,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.stage2_epochs, eta_min=1e-6)
    print(f"\n--- Stage 2: Full Fine-Tuning ({args.stage2_epochs} epochs) ---", flush=True)
    for ep in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_binary_epoch(model, bundle["train_loader"], criterion, device, opt2)
        va_loss, va_acc, va_auc = run_binary_epoch(model, bundle["val_loader"], criterion, device)
        sched.step()
        saved = ""
        if va_acc > best_score:
            best_score = va_acc
            torch.save({"arch": args.model, "val_acc": va_acc, "val_auc": va_auc, "model_state_dict": model.state_dict()}, ckpt_path)
            saved = " *best*"
        print(f"[Bin S2 {ep:02d}/{args.stage2_epochs:02d}] tr_acc={tr_acc*100:.1f}% | val_acc={va_acc*100:.1f}% val_auc={va_auc:.4f} ({time.time()-t0:.1f}s){saved}", flush=True)

    # Evaluate best binary checkpoint with 4x TTA
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_true, val_probs_bin = predict_loader_tta(model, bundle["val_loader"], device)
    test_true, test_probs_bin = predict_loader_tta(model, bundle["test_loader"], device)
    p_val_bin = val_probs_bin[:, 1]
    p_test_bin = test_probs_bin[:, 1]

    standalone_val = binary_metrics_dict(val_true, p_val_bin, 0.5)
    standalone_test = binary_metrics_dict(test_true, p_test_bin, 0.5)

    # Hierarchical Fusion with 9-class ensemble (if checkpoints exist)
    p_val_fused = p_val_bin
    p_test_fused = p_test_bin
    fusion_models = [f"binary_{args.model}"]

    mc_ckpts = [CHECKPOINT_DIR / "efficientnet_b0_best.pth", CHECKPOINT_DIR / "best_model.pth"]
    existing_mc = [p for p in mc_ckpts if p.exists()]
    if existing_mc:
        class_names = bundle["class_names_9"]
        mal_ids = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
        mc_val_p, mc_test_p = [], []
        for mp in existing_mc:
            ck = torch.load(mp, map_location=device, weights_only=False)
            m = build_model(arch=ck["arch"], num_classes=len(class_names), pretrained=False).to(device)
            m.load_state_dict(ck["model_state_dict"])
            _, vp = predict_loader_tta(m, bundle["base_9class_loaders"]["val_loader"], device)
            _, tp = predict_loader_tta(m, bundle["base_9class_loaders"]["test_loader"], device)
            mc_val_p.append(vp[:, mal_ids].sum(axis=1))
            mc_test_p.append(tp[:, mal_ids].sum(axis=1))
            fusion_models.append(f"9class_{ck['arch']}")

        p_val_mc = np.mean(mc_val_p, axis=0)
        p_test_mc = np.mean(mc_test_p, axis=0)
        # 50/50 hierarchical fusion between dedicated binary model and 9-class ensemble
        p_val_fused = 0.5 * p_val_bin + 0.5 * p_val_mc
        p_test_fused = 0.5 * p_test_bin + 0.5 * p_test_mc

    fused_val = binary_metrics_dict(val_true, p_val_fused, 0.5)
    fused_test = binary_metrics_dict(test_true, p_test_fused, 0.5)

    save_binary_plots(val_true, p_val_fused, 0.5, OUTPUT_DIR)

    report = {
        "stage1_binary_model": args.model,
        "hierarchical_fusion_models": fusion_models,
        "standalone_binary_val_tta": standalone_val,
        "standalone_binary_test_tta": standalone_test,
        "hierarchical_fused_val_tta": fused_val,
        "hierarchical_fused_test_tta": fused_test,
    }
    with open(OUTPUT_DIR / "hierarchical_triage_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n================ STAGE-1 BINARY TRIAGE SUMMARY ================")
    print(f"Standalone Binary Val  (4x TTA): Acc={standalone_val['accuracy']*100:.2f}% | Sens={standalone_val['sensitivity']*100:.1f}% | Spec={standalone_val['specificity']*100:.1f}% | AUC={standalone_val['roc_auc']:.4f}")
    print(f"Standalone Binary Test (4x TTA): Acc={standalone_test['accuracy']*100:.2f}% | Sens={standalone_test['sensitivity']*100:.1f}% | Spec={standalone_test['specificity']*100:.1f}% | AUC={standalone_test['roc_auc']:.4f}")
    print(f"Hierarchical Fused Val (4x TTA): Acc={fused_val['accuracy']*100:.2f}% | Sens={fused_val['sensitivity']*100:.1f}% | Spec={fused_val['specificity']*100:.1f}% | AUC={fused_val['roc_auc']:.4f}")
    print(f"Hierarchical Fused Test(4x TTA): Acc={fused_test['accuracy']*100:.2f}% | Sens={fused_test['sensitivity']*100:.1f}% | Spec={fused_test['specificity']*100:.1f}% | AUC={fused_test['roc_auc']:.4f}")
    print("===============================================================")


def main():
    parser = argparse.ArgumentParser(description="Train Stage-1 Binary Clinical Triage classifier.")
    parser.add_argument("--model", type=str, default="efficientnet_b0", choices=["mobilenet_v2", "efficientnet_b0", "densenet121"])
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    train_and_fuse(parser.parse_args())


if __name__ == "__main__":
    main()
