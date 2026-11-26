"""Model factory for transfer learning backbones (MobileNetV2, EfficientNet-B0, ResNet50)."""

from typing import Tuple

import torch.nn as nn
from torchvision import models

from src.config import DEFAULT_DROPOUT


def build_model(
    arch: str,
    num_classes: int,
    dropout: float = DEFAULT_DROPOUT,
    pretrained: bool = True,
) -> nn.Module:
    """Create a pretrained CNN backbone with a custom classification head."""
    arch = arch.lower()

    if arch == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    if arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported architecture '{arch}'. Choose from: mobilenet_v2, efficientnet_b0, resnet50.")


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all backbone parameters, keeping only the classifier head trainable."""
    for p in model.parameters():
        p.requires_grad = False

    head = model.fc if arch.lower() == "resnet50" else model.classifier
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_top_layers(model: nn.Module, arch: str) -> Tuple[list, list]:
    """Unfreeze top convolutional blocks for Stage-2 fine-tuning and return (backbone_params, head_params)."""
    arch = arch.lower()
    for p in model.parameters():
        p.requires_grad = False

    if arch == "mobilenet_v2":
        # Unfreeze inverted residual blocks 14..18 (last 5 feature blocks)
        for p in model.features[14:].parameters():
            p.requires_grad = True
        head = model.classifier
    elif arch == "efficientnet_b0":
        for p in model.features[5:].parameters():
            p.requires_grad = True
        head = model.classifier
    elif arch == "resnet50":
        for p in model.layer4.parameters():
            p.requires_grad = True
        head = model.fc
    else:
        raise ValueError(f"Unsupported architecture '{arch}'.")

    for p in head.parameters():
        p.requires_grad = True

    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head_params = [p for p in head.parameters() if p.requires_grad]
    return backbone_params, head_params


def get_gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the final convolutional layer used for Grad-CAM visualization."""
    arch = arch.lower()
    if arch in ("mobilenet_v2", "efficientnet_b0"):
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unsupported architecture '{arch}'.")
