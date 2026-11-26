"""Two-stage transfer learning training script for 9-class ISIC skin cancer classification."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import (
    CHECKPOINT_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_MODEL,
    DEFAULT_STAGE1_EPOCHS,
    DEFAULT_STAGE1_LR,
    DEFAULT_STAGE2_BACKBONE_LR,
    DEFAULT_STAGE2_EPOCHS,
    DEFAULT_STAGE2_HEAD_LR,
    DEFAULT_WEIGHT_DECAY,
    OUTPUT_DIR,
    RANDOM_SEED,
    get_device,
)
from src.dataset import create_dataloaders
from src.evaluate import run_evaluation
from src.model import build_model, freeze_backbone, unfreeze_layers


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, tta: bool = False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                probs = F.softmax(logits, dim=1)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                if tta:
                    p0 = F.softmax(logits, dim=1)
                    p1 = F.softmax(model(torch.flip(images, dims=[3])), dim=1)
                    p2 = F.softmax(model(torch.flip(images, dims=[2])), dim=1)
                    p3 = F.softmax(model(torch.flip(images, dims=[2, 3])), dim=1)
                    probs = (p0 + p1 + p2 + p3) / 4.0
                else:
                    probs = F.softmax(logits, dim=1)

            running_loss += loss.item() * images.size(0)
            preds = probs.argmax(dim=1)
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(preds.cpu().numpy().tolist())

    total = len(all_true)
    avg_loss = running_loss / max(total, 1)
    acc = float(np.mean(np.array(all_true) == np.array(all_pred)))
    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


def train(args) -> Path:
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device} | model={args.model} | unfreeze={args.unfreeze}", flush=True)

    data = create_dataloaders(batch_size=args.batch_size, seed=args.seed)
    class_names = data["class_names"]
    num_classes = len(class_names)
    print(
        f"Dataset loaded: {data['train_size']} train | {data['val_size']} val | "
        f"{data['test_size']} test | {num_classes} classes",
        flush=True,
    )

    model = build_model(arch=args.model, num_classes=num_classes, dropout=args.dropout, pretrained=True).to(device)
    class_weights = data["class_weights"].to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "stage1_epochs": args.stage1_epochs,
    }

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_name = args.save_name or f"{args.model}_best.pth"
    best_ckpt_path = CHECKPOINT_DIR / save_name
    best_val_acc = -1.0

    if args.resume and Path(args.resume).exists():
        prev = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state_dict"])
        best_val_acc = float(prev.get("val_acc", -1.0))
        if "history" in prev:
            history = prev["history"]
        print(f"Resumed weights from {args.resume} (best_val_acc={best_val_acc*100:.2f}%)", flush=True)

    def maybe_save(epoch_num: int, val_acc: float, val_f1: float):
        nonlocal best_val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "arch": args.model,
                    "class_names": class_names,
                    "epoch": epoch_num,
                    "val_acc": val_acc,
                    "val_macro_f1": val_f1,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                },
                best_ckpt_path,
            )
            return True
        return False

    # -------- Stage 1: Frozen backbone warmup --------
    print(f"\n--- Stage 1: Warmup (Frozen {args.model} backbone, {args.stage1_epochs} epochs) ---", flush=True)
    freeze_backbone(model, args.model)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )

    total_epoch = 0
    for ep in range(1, args.stage1_epochs + 1):
        total_epoch += 1
        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, data["train_loader"], criterion, device, optimizer=opt1)
        va_loss, va_acc, va_f1 = run_epoch(model, data["val_loader"], criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["val_macro_f1"].append(va_f1)
        saved = maybe_save(total_epoch, va_acc, va_f1)
        mark = " *best*" if saved else ""
        print(
            f"[S1 {ep:02d}/{args.stage1_epochs:02d}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}% | "
            f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}% val_f1={va_f1:.3f} "
            f"({time.time()-t0:.1f}s){mark}",
            flush=True,
        )

    # -------- Stage 2: Full or Top-Block Fine-Tuning --------
    if args.stage2_epochs > 0:
        print(f"\n--- Stage 2: Fine-Tuning ({args.unfreeze}, {args.stage2_epochs} epochs) ---", flush=True)
        backbone_params, head_params = unfreeze_layers(model, args.model, mode=args.unfreeze)
        opt2 = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.stage2_backbone_lr},
                {"params": head_params, "lr": args.stage2_head_lr},
            ],
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.stage2_epochs, eta_min=1e-6)

        for ep in range(1, args.stage2_epochs + 1):
            total_epoch += 1
            t0 = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, data["train_loader"], criterion, device, optimizer=opt2)
            va_loss, va_acc, va_f1 = run_epoch(model, data["val_loader"], criterion, device, tta=False)
            scheduler.step()
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["val_macro_f1"].append(va_f1)
            saved = maybe_save(total_epoch, va_acc, va_f1)
            mark = " *best*" if saved else ""
            print(
                f"[S2 {ep:02d}/{args.stage2_epochs:02d}] "
                f"train_loss={tr_loss:.4f} train_acc={tr_acc*100:.1f}% | "
                f"val_loss={va_loss:.4f} val_acc={va_acc*100:.1f}% val_f1={va_f1:.3f} "
                f"({time.time()-t0:.1f}s){mark}",
                flush=True,
            )

    ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    ckpt["history"] = history
    torch.save(ckpt, best_ckpt_path)

    print(f"\nBest checkpoint saved to {best_ckpt_path} (val_acc={best_val_acc*100:.2f}%)", flush=True)
    if not args.skip_eval:
        run_evaluation(best_ckpt_path, OUTPUT_DIR)
    return best_ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train skin cancer classification model on ISIC dataset.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=["mobilenet_v2", "efficientnet_b0", "densenet121", "resnet50"],
    )
    parser.add_argument("--unfreeze", type=str, default="all", choices=["all", "top"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stage1-epochs", type=int, default=DEFAULT_STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=DEFAULT_STAGE2_EPOCHS)
    parser.add_argument("--stage1-lr", type=float, default=DEFAULT_STAGE1_LR)
    parser.add_argument("--stage2-backbone-lr", type=float, default=DEFAULT_STAGE2_BACKBONE_LR)
    parser.add_argument("--stage2-head-lr", type=float, default=DEFAULT_STAGE2_HEAD_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
