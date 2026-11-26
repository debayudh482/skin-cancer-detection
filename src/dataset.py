"""Dataset loading, stratified splitting, augmentations, and class weights."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
    TEST_DIR,
    TRAIN_DIR,
    VAL_SPLIT,
)


def get_transforms(image_size: int = IMAGE_SIZE) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return training (augmented) and evaluation transforms."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


class SubsetWithTransform(Dataset):
    """Wrap (path, label) pairs with a dedicated transform."""

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def compute_class_weights(labels: List[int], num_classes: int, power: float = 0.5) -> torch.Tensor:
    """Compute smoothed inverse-frequency class weights normalized to mean 1.0."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = (total / (num_classes * np.maximum(counts, 1.0))) ** power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
    batch_size: int = 32,
    val_split: float = VAL_SPLIT,
    image_size: int = IMAGE_SIZE,
    num_workers: int = 0,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Create stratified train/val loaders and held-out test loader."""
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Dataset directories not found at:\n  Train: {train_dir}\n  Test:  {test_dir}\n"
            "Extract Skin_ISIC.zip into the data/ folder first."
        )

    base_train = datasets.ImageFolder(root=str(train_dir))
    base_test = datasets.ImageFolder(root=str(test_dir))

    class_names = base_train.classes
    if base_test.classes != class_names:
        raise ValueError(f"Train/Test class mismatch: {class_names} vs {base_test.classes}")

    all_samples = base_train.samples
    all_labels = [label for _, label in all_samples]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(all_labels)), all_labels))

    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    test_samples = base_test.samples

    train_tf, eval_tf = get_transforms(image_size=image_size)

    train_ds = SubsetWithTransform(train_samples, transform=train_tf)
    val_ds = SubsetWithTransform(val_samples, transform=eval_tf)
    test_ds = SubsetWithTransform(test_samples, transform=eval_tf)

    train_labels = [label for _, label in train_samples]
    class_weights = compute_class_weights(train_labels, num_classes=len(class_names))

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "class_weights": class_weights,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "test_samples": test_samples,
    }
