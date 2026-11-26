"""Configuration constants and default hyperparameters."""

from pathlib import Path
import torch

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_ROOT = PROJECT_ROOT / "data" / "Skin cancer ISIC The International Skin Imaging Collaboration"
RESIZED_DATA_ROOT = PROJECT_ROOT / "data" / "resized_256"

# Automatically prefer fast pre-resized dataset when available
DATA_ROOT = RESIZED_DATA_ROOT if (RESIZED_DATA_ROOT / "Train").exists() else RAW_DATA_ROOT
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

# Clinical grouping for binary screening evaluation (Malignant/Pre-malignant vs Benign)
MALIGNANT_CLASSES = {
    "melanoma",
    "basal cell carcinoma",
    "squamous cell carcinoma",
    "actinic keratosis",
}

# Default training hyperparameters
DEFAULT_MODEL = "efficientnet_b0"
DEFAULT_BATCH_SIZE = 32
DEFAULT_STAGE1_EPOCHS = 3
DEFAULT_STAGE2_EPOCHS = 15
DEFAULT_STAGE1_LR = 1e-3
DEFAULT_STAGE2_BACKBONE_LR = 2e-4
DEFAULT_STAGE2_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_DROPOUT = 0.3
DEFAULT_LABEL_SMOOTHING = 0.05


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
