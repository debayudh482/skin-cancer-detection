"""Configuration constants and default hyperparameters."""

from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "Skin cancer ISIC The International Skin Imaging Collaboration"
TRAIN_DIR = DATA_ROOT / "Train"
TEST_DIR = DATA_ROOT / "Test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Image preprocessing (ImageNet stats for pretrained CNN backbones)
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reproducibility & split
RANDOM_SEED = 42
VAL_SPLIT = 0.15

# Default training hyperparameters
DEFAULT_MODEL = "mobilenet_v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_STAGE1_EPOCHS = 5
DEFAULT_STAGE2_EPOCHS = 10
DEFAULT_STAGE1_LR = 1e-3
DEFAULT_STAGE2_BACKBONE_LR = 1e-4
DEFAULT_STAGE2_HEAD_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DROPOUT = 0.3
DEFAULT_LABEL_SMOOTHING = 0.05


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
