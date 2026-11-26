"""Unified Two-Stage Hierarchical Clinical Inference (Binary Triage + 9-Class Ensemble + Grad-CAM)."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, IMAGE_SIZE, MALIGNANT_CLASSES, get_device
from src.dataset import get_transforms
from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model, get_gradcam_target_layer


@torch.no_grad()
def _predict_tta_single(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    p0 = F.softmax(model(tensor), dim=1)
    p1 = F.softmax(model(torch.flip(tensor, dims=[3])), dim=1)
    p2 = F.softmax(model(torch.flip(tensor, dims=[2])), dim=1)
    p3 = F.softmax(model(torch.flip(tensor, dims=[2, 3])), dim=1)
    return ((p0 + p1 + p2 + p3) / 4.0).squeeze(0).cpu().numpy()


def run_hierarchical_prediction(image_path: Path, gradcam_out: Path = None):
    device = get_device()
    _, eval_tf = get_transforms()

    with Image.open(image_path) as img:
        pil_rgb = img.convert("RGB")
        resized_np = np.array(pil_rgb.resize((IMAGE_SIZE, IMAGE_SIZE)))
        tensor = eval_tf(pil_rgb).unsqueeze(0).to(device)

    # Load 9-class ensemble checkpoints
    mc_paths = [p for p in [CHECKPOINT_DIR / "efficientnet_b0_best.pth", CHECKPOINT_DIR / "best_model.pth"] if p.exists()]
    if not mc_paths:
        raise FileNotFoundError("No 9-class checkpoints found in checkpoints/. Run src.train first.")

    mc_probs_list = []
    class_names = None
    primary_mc_model = None
    primary_mc_arch = None

    for idx, mp in enumerate(mc_paths):
        ck = torch.load(mp, map_location=device, weights_only=False)
        class_names = ck["class_names"]
        m = build_model(arch=ck["arch"], num_classes=len(class_names), pretrained=False).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        mc_probs_list.append(_predict_tta_single(m, tensor))
        if idx == 0:
            primary_mc_model = m
            primary_mc_arch = ck["arch"]

    mc_probs = np.mean(mc_probs_list, axis=0)
    mal_ids = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    p_mal_mc = float(mc_probs[mal_ids].sum())

    # Stage 1: Dedicated Binary Triage (fused with 9-class malignant probability)
    bin_path = CHECKPOINT_DIR / "binary_triage_best.pth"
    if bin_path.exists():
        bck = torch.load(bin_path, map_location=device, weights_only=False)
        bm = build_model(arch=bck["arch"], num_classes=2, pretrained=False).to(device)
        bm.load_state_dict(bck["model_state_dict"])
        bm.eval()
        p_mal_bin = float(_predict_tta_single(bm, tensor)[1])
        p_mal_fused = 0.5 * p_mal_bin + 0.5 * p_mal_mc
    else:
        p_mal_fused = p_mal_mc

    if p_mal_fused >= 0.65:
        risk_tier = "HIGH RISK (Malignant / Pre-Malignant — Recommend Urgent Dermatology Review)"
    elif p_mal_fused >= 0.35:
        risk_tier = "MODERATE RISK (Indeterminate — Recommend Clinical Follow-up)"
    else:
        risk_tier = "LOW RISK (Likely Benign)"

    ranked = sorted(zip(class_names, mc_probs.tolist()), key=lambda x: x[1], reverse=True)

    print(f"\n================ HIERARCHICAL CLINICAL REPORT ================")
    print(f"Image              : {image_path}")
    print(f"Stage 1 — Triage   : {risk_tier}")
    print(f"Malignant Risk Prob: {p_mal_fused*100:.2f}%")
    print(f"\nStage 2 — Top-3 Fine-Grained Subtypes (9-Class Ensemble + 4x TTA):")
    for name, prob in ranked[:3]:
        tag = "[Malignant/Pre-Mal]" if name in MALIGNANT_CLASSES else "[Benign]"
        print(f"  - {name:<28s}: {prob*100:6.2f}%  {tag}")
    print("==============================================================\n")

    if gradcam_out is not None:
        gradcam = GradCAM(primary_mc_model, get_gradcam_target_layer(primary_mc_model, primary_mc_arch))
        heatmap, _, _ = gradcam.generate(tensor)
        gradcam.close()
        overlay = overlay_heatmap(resized_np, heatmap)
        gradcam_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(gradcam_out)
        print(f"Stage 3 — Grad-CAM explainability map saved to: {gradcam_out}")


def main():
    parser = argparse.ArgumentParser(description="Run Two-Stage Hierarchical Skin Cancer Inference.")
    parser.add_argument("image", type=Path, help="Path to dermoscopic image")
    parser.add_argument("--gradcam-out", type=Path, default=None)
    args = parser.parse_args()
    run_hierarchical_prediction(args.image, args.gradcam_out)


if __name__ == "__main__":
    main()
