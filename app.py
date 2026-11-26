"""Interactive Clinical Web UI for Two-Stage Hierarchical Skin Cancer Detection + Grad-CAM."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F

from src.config import CHECKPOINT_DIR, IMAGE_SIZE, MALIGNANT_CLASSES, OUTPUT_DIR, RAW_DATA_ROOT, get_device
from src.dataset import get_transforms
from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model, get_gradcam_target_layer

st.set_page_config(
    page_title="DermaScan AI — Skin Cancer Detection",
    page_icon="🔬",
    layout="wide",
)


@st.cache_resource
def load_models():
    """Load Stage-1 Binary Triage and Stage-2 9-Class Ensemble checkpoints once."""
    device = get_device()
    mc_paths = [
        p for p in [
            CHECKPOINT_DIR / "efficientnet_b0_best.pth",
            CHECKPOINT_DIR / "densenet121_best.pth",
            CHECKPOINT_DIR / "mobilenet_v2_best.pth",
            CHECKPOINT_DIR / "best_model.pth",
        ]
        if p.exists()
    ]
    if not mc_paths:
        raise FileNotFoundError("No trained checkpoints found in checkpoints/. Run `python -m src.train` first.")

    mc_models = []
    class_names = None
    for mp in mc_paths:
        ck = torch.load(mp, map_location=device, weights_only=False)
        class_names = ck["class_names"]
        m = build_model(arch=ck["arch"], num_classes=len(class_names), pretrained=False).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        mc_models.append((ck["arch"], m))

    bin_model = None
    bin_path = CHECKPOINT_DIR / "binary_triage_best.pth"
    if bin_path.exists():
        bck = torch.load(bin_path, map_location=device, weights_only=False)
        bm = build_model(arch=bck["arch"], num_classes=2, pretrained=False).to(device)
        bm.load_state_dict(bck["model_state_dict"])
        bm.eval()
        bin_model = (bck["arch"], bm)

    return device, class_names, mc_models, bin_model


@torch.no_grad()
def _tta_probs(model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
    p0 = F.softmax(model(tensor), dim=1)
    p1 = F.softmax(model(torch.flip(tensor, dims=[3])), dim=1)
    p2 = F.softmax(model(torch.flip(tensor, dims=[2])), dim=1)
    p3 = F.softmax(model(torch.flip(tensor, dims=[2, 3])), dim=1)
    return ((p0 + p1 + p2 + p3) / 4.0).squeeze(0).cpu().numpy()


def analyze_image(pil_img: Image.Image) -> Dict:
    device, class_names, mc_models, bin_model = load_models()
    _, eval_tf = get_transforms()

    pil_rgb = pil_img.convert("RGB")
    resized_np = np.array(pil_rgb.resize((IMAGE_SIZE, IMAGE_SIZE)))
    tensor = eval_tf(pil_rgb).unsqueeze(0).to(device)

    mc_probs_list = [_tta_probs(m, tensor) for _, m in mc_models]
    mc_probs = np.mean(mc_probs_list, axis=0)

    mal_ids = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    p_mal_mc = float(mc_probs[mal_ids].sum())

    if bin_model is not None:
        p_mal_bin = float(_tta_probs(bin_model[1], tensor)[1])
        p_mal_fused = 0.5 * p_mal_bin + 0.5 * p_mal_mc
    else:
        p_mal_fused = p_mal_mc

    # Grad-CAM on primary Stage-2 backbone (EfficientNet-B0)
    primary_arch, primary_model = mc_models[0]
    gradcam = GradCAM(primary_model, get_gradcam_target_layer(primary_model, primary_arch))
    heatmap, _, _ = gradcam.generate(tensor)
    gradcam.close()
    cam_overlay = overlay_heatmap(resized_np, heatmap, alpha=0.45)

    ranked = sorted(zip(class_names, mc_probs.tolist()), key=lambda x: x[1], reverse=True)
    return {
        "malignant_prob": p_mal_fused,
        "ranked": ranked,
        "resized_rgb": resized_np,
        "gradcam_overlay": cam_overlay,
        "models_used": [a for a, _ in mc_models],
        "has_binary_head": bin_model is not None,
    }


def find_sample_images() -> List[Tuple[str, Path]]:
    test_dir = RAW_DATA_ROOT / "Test"
    if not test_dir.exists():
        return []
    picks = []
    for cls_name in ["basal cell carcinoma", "melanoma", "nevus", "vascular lesion", "pigmented benign keratosis"]:
        cls_dir = test_dir / cls_name
        if cls_dir.exists():
            imgs = sorted(cls_dir.glob("*.jpg"))
            if imgs:
                picks.append((f"{cls_name.title()} ({imgs[0].name})", imgs[0]))
    return picks


# ---------------- UI Layout ----------------
st.title("🔬 Hierarchical Skin Cancer Detection System")
st.caption(
    "Two-Stage Clinical Triage (Binary EfficientNet-B0 + 9-Class Ensemble + 4x TTA) with Grad-CAM Visual Explainability | "
    "ISIC Dermoscopic Dataset — Group B55 (Debayudh Kundu & Aditiya Majumdar)"
)

tab_demo, tab_perf = st.tabs(["🩺 Live Lesion Analysis", "📊 Model Performance & Evaluation"])

with tab_demo:
    col_in, col_res = st.columns([1, 1.4], gap="large")

    with col_in:
        st.subheader("1. Upload Dermoscopic Image")
        uploaded = st.file_uploader("Drag and drop a skin lesion image (.jpg, .png)", type=["jpg", "jpeg", "png"])

        samples = find_sample_images()
        chosen_sample = None
        if samples:
            st.markdown("**Or pick a held-out ISIC test sample:**")
            sample_labels = ["— Select a sample —"] + [label for label, _ in samples]
            sel = st.selectbox("Sample Test Lesion", sample_labels, label_visibility="collapsed")
            if sel != sample_labels[0]:
                chosen_sample = dict(samples)[sel]

        pil_image = None
        source_label = ""
        if uploaded is not None:
            pil_image = Image.open(uploaded)
            source_label = uploaded.name
        elif chosen_sample is not None:
            pil_image = Image.open(chosen_sample)
            source_label = chosen_sample.name

        if pil_image is not None:
            st.image(pil_image, caption=f"Input: {source_label}", use_container_width=True)
        else:
            st.info("Upload a dermoscopic image or select a held-out test sample above to run the 3-stage analysis.")

    with col_res:
        st.subheader("2. Three-Stage AI Clinical Report")
        if pil_image is not None:
            with st.spinner("Running 4x TTA Ensemble + Grad-CAM..."):
                res = analyze_image(pil_image)

            p_mal = res["malignant_prob"]
            if p_mal >= 0.65:
                st.error(
                    f"**STAGE 1 — HIGH RISK ({p_mal*100:.1f}% Malignant / Pre-Malignant Probability)**\n\n"
                    "Urgent dermatologist evaluation / dermoscopic biopsy review recommended."
                )
            elif p_mal >= 0.35:
                st.warning(
                    f"**STAGE 1 — MODERATE RISK ({p_mal*100:.1f}% Malignant / Pre-Malignant Probability)**\n\n"
                    "Indeterminate lesion features — clinical follow-up recommended."
                )
            else:
                st.success(
                    f"**STAGE 1 — LOW RISK ({p_mal*100:.1f}% Malignant / Pre-Malignant Probability)**\n\n"
                    "Lesion appears likely benign. Monitor for any ABCD changes over time."
                )

            st.markdown("#### Stage 2 — Top-3 Fine-Grained Subtype Predictions (9-Class Ensemble)")
            for name, prob in res["ranked"][:3]:
                tag = "🔴 Malignant / Pre-Mal" if name in MALIGNANT_CLASSES else "🟢 Benign"
                st.markdown(f"**{name.title()}** — `{prob*100:.2f}%` ({tag})")
                st.progress(min(max(prob, 0.0), 1.0))

            with st.expander("View all 9 class probabilities"):
                for name, prob in res["ranked"]:
                    st.write(f"- **{name.title()}**: {prob*100:.2f}%")

            st.markdown("#### Stage 3 — Grad-CAM Visual Explainability")
            c1, c2 = st.columns(2)
            c1.image(res["resized_rgb"], caption="Preprocessed Input (224×224)", use_container_width=True)
            c2.image(res["gradcam_overlay"], caption="Grad-CAM Attention Map (Red = High Influence)", use_container_width=True)

            st.caption(
                f"Active models: {', '.join(res['models_used'])}"
                + (" + Binary EfficientNet-B0 Triage" if res["has_binary_head"] else "")
                + " | 4x Flip TTA"
            )
        else:
            st.markdown(
                """
                **What this system outputs for each lesion:**
                - **Stage 1 (Binary Triage):** Malignant vs. Benign risk tier with **81.55% accuracy & 89.2% cancer sensitivity**.
                - **Stage 2 (9-Class Subtype Ensemble):** Exact lesion probability across all 9 ISIC classes (**0.939 Macro ROC-AUC**).
                - **Stage 3 (Grad-CAM):** Visual heatmap showing *which pixels* drove the AI decision.
                """
            )

    st.divider()
    st.caption("⚠️ *Medical Disclaimer: This tool is an academic AI research prototype for clinical decision support only and is not a substitute for professional dermatological diagnosis or biopsy.*")

with tab_perf:
    st.subheader("Validation & Held-Out Test Benchmarks")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stage-1 Binary Accuracy", "81.55%", "+45.9% vs baseline")
    m2.metric("Cancer Sensitivity (Recall)", "89.2%", "High-safety triage")
    m3.metric("9-Class Ensemble Accuracy", "71.73%", "+36.1% vs 35.6% slides")
    m4.metric("9-Class Macro ROC-AUC", "0.9390", "0.9272 held-out test")

    p1, p2 = st.columns(2)
    bin_plot = OUTPUT_DIR / "binary_triage_evaluation.png"
    roc_plot = OUTPUT_DIR / "roc_curves_val.png"
    cm_plot = OUTPUT_DIR / "confusion_matrix_val.png"
    hist_plot = OUTPUT_DIR / "training_history.png"
    cam_plot = OUTPUT_DIR / "gradcam_samples.png"

    if bin_plot.exists():
        p1.image(str(bin_plot), caption="Stage-1 Hierarchical Binary Triage (Confusion Matrix & ROC)", use_container_width=True)
    if roc_plot.exists():
        p2.image(str(roc_plot), caption="Stage-2 Multi-Class One-vs-Rest ROC Curves", use_container_width=True)

    p3, p4 = st.columns(2)
    if cm_plot.exists():
        p3.image(str(cm_plot), caption="9-Class Validation Confusion Matrix", use_container_width=True)
    if hist_plot.exists():
        p4.image(str(hist_plot), caption="Training & Validation Learning Curves", use_container_width=True)

    if cam_plot.exists():
        st.image(str(cam_plot), caption="Grad-CAM Sample Explanations on Held-Out Test Lesions", use_container_width=True)
