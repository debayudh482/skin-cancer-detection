"""Single-image CLI inference with optional Grad-CAM overlay export."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, IMAGE_SIZE, get_device
from src.dataset import get_transforms
from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model, get_gradcam_target_layer


def predict_image(image_path: Path, checkpoint_path: Path, save_gradcam: Path = None):
    device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt["arch"]
    class_names = ckpt["class_names"]

    model = build_model(arch=arch, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, eval_tf = get_transforms()
    with Image.open(image_path) as img:
        pil_rgb = img.convert("RGB")
        resized_np = np.array(pil_rgb.resize((IMAGE_SIZE, IMAGE_SIZE)))
        tensor = eval_tf(pil_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    ranked = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
    print(f"\nImage: {image_path}")
    print(f"Top Prediction: {ranked[0][0]} ({ranked[0][1]*100:.2f}%)\n")
    print("All Class Probabilities:")
    for name, p in ranked:
        print(f"  {name:<30s}: {p*100:6.2f}%")

    if save_gradcam is not None:
        gradcam = GradCAM(model, get_gradcam_target_layer(model, arch))
        heatmap, _, _ = gradcam.generate(tensor)
        gradcam.close()
        overlay = overlay_heatmap(resized_np, heatmap)
        save_gradcam.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(save_gradcam)
        print(f"\nGrad-CAM overlay saved to: {save_gradcam}")

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Predict skin lesion class from a dermoscopic image.")
    parser.add_argument("image", type=Path, help="Path to dermoscopic image (.jpg/.png)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best_model.pth")
    parser.add_argument("--gradcam-out", type=Path, default=None, help="Optional path to save Grad-CAM overlay image")
    args = parser.parse_args()
    predict_image(args.image, args.checkpoint, args.gradcam_out)


if __name__ == "__main__":
    main()
