"""Grad-CAM visual explainability for dermoscopic skin lesion predictions."""

from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from src.model import get_gradcam_target_layer


class GradCAM:
    """Compute Grad-CAM saliency heatmaps for a given target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None) -> Tuple[np.ndarray, int, float]:
        """Return (heatmap [H,W] in [0,1], predicted_class, confidence)."""
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = int(probs.argmax(dim=1).item())
        target_class = pred_class if class_idx is None else class_idx
        confidence = float(probs[0, target_class].item())

        score = logits[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, confidence

    def close(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap onto an RGB uint8 image using the jet colormap."""
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * image_rgb.astype(np.float32) + alpha * colored.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_gradcam_grid(
    model: torch.nn.Module,
    arch: str,
    samples: List[Tuple[str, int]],
    class_names: List[str],
    eval_transform,
    device: torch.device,
    save_path: Path,
    max_samples: int = 6,
) -> None:
    """Generate and save a side-by-side Grad-CAM grid for diverse test samples."""
    # Pick up to max_samples spread across different classes
    seen_classes = set()
    chosen = []
    for path, label in samples:
        if label not in seen_classes:
            chosen.append((path, label))
            seen_classes.add(label)
        if len(chosen) >= max_samples:
            break

    target_layer = get_gradcam_target_layer(model, arch)
    gradcam = GradCAM(model, target_layer)

    cols = 3
    rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 5.2, rows * 3.0))
    axes = np.atleast_2d(axes)

    for idx, (img_path, true_label) in enumerate(chosen):
        r, c = divmod(idx, cols)
        with Image.open(img_path) as pil_img:
            pil_rgb = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            img_np = np.array(pil_rgb)

        tensor = eval_transform(pil_rgb).unsqueeze(0).to(device)
        heatmap, pred_idx, conf = gradcam.generate(tensor)
        overlay = overlay_heatmap(img_np, heatmap)

        ax_orig = axes[r, c * 2]
        ax_cam = axes[r, c * 2 + 1]

        ax_orig.imshow(img_np)
        ax_orig.set_title(f"True: {class_names[true_label]}", fontsize=8.5)
        ax_orig.axis("off")

        color = "green" if pred_idx == true_label else "red"
        ax_cam.imshow(overlay)
        ax_cam.set_title(f"Pred: {class_names[pred_idx]} ({conf:.0%})", fontsize=8.5, color=color)
        ax_cam.axis("off")

    # Hide unused axes
    for idx in range(len(chosen), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")

    gradcam.close()
    plt.suptitle("Grad-CAM Visual Explanations on Held-Out Test Images", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
