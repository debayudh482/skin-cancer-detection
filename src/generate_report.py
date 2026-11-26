"""Generate a formal, submission-ready academic Project Report (.docx) with embedded tables and figures."""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_PATH = PROJECT_ROOT / "Skin_Cancer_Detection_Project_Report.docx"

NAVY = RGBColor(15, 23, 42)
TEAL = RGBColor(13, 148, 136)
SLATE = RGBColor(51, 65, 85)
MUTED = RGBColor(100, 116, 139)


def set_cell_shading(cell, hex_color: str):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_left = p.add_run("Hierarchical Skin Cancer Detection System — Group B55 (CSE 2025–2026)     |     Page ")
    run_left.font.name = "Calibri"
    run_left.font.size = Pt(9)
    run_left.font.color.rgb = MUTED

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    p._p.append(fld)


def add_heading_styled(doc: Document, text: str, level: int = 1):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.name = "Calibri"
        r.font.color.rgb = NAVY if level == 1 else TEAL
        if level == 1:
            r.font.size = Pt(18)
        elif level == 2:
            r.font.size = Pt(13.5)
        else:
            r.font.size = Pt(11.5)
    h.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    h.paragraph_format.space_after = Pt(6 if level == 1 else 4)
    return h


def add_body(doc: Document, text: str, bold_prefix: str = None, space_after: int = 6):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Calibri"
        rb.font.size = Pt(11)
        rb.font.color.rgb = NAVY
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(11)
    r.font.color.rgb = SLATE
    return p


def add_bullet(doc: Document, text: str, bold_prefix: str = None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Calibri"
        rb.font.size = Pt(10.5)
        rb.font.color.rgb = NAVY
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(10.5)
    r.font.color.rgb = SLATE
    return p


def add_styled_table(doc: Document, headers, rows, highlight_last: bool = False):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        set_cell_shading(cell, "0F172A")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row in enumerate(rows):
        is_last = highlight_last and (r_idx == len(rows) - 1)
        bg = "E6FFFA" if is_last else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            set_cell_shading(cell, bg)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_idx > 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            r.bold = is_last
            r.font.name = "Calibri"
            r.font.size = Pt(9.5)
            r.font.color.rgb = NAVY if is_last else SLATE

    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)
    return table


def add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 5.8):
    if not img_path.exists():
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(img_path), width=Inches(width_in))

    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_after = Pt(10)
    r = cp.add_run(caption)
    r.italic = True
    r.font.name = "Calibri"
    r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED


def build_report():
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.95)
        section.right_margin = Inches(0.95)
        section.different_first_page_header_footer = True
        add_page_number_footer(section)

    # Load real metrics JSON for exact per-class numbers
    metrics_json = {}
    if (OUTPUT_DIR / "metrics.json").exists():
        metrics_json = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))

    # ==================== COVER PAGE ====================
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(28)
    r = p.add_run("A PROJECT REPORT ON")
    r.font.name = "Calibri"
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = TEAL

    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tp.paragraph_format.space_before = Pt(14)
    r = tp.add_run("SKIN CANCER DETECTION FROM DERMOSCOPIC IMAGES USING DEEP LEARNING")
    r.font.name = "Calibri"
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = NAVY

    sp = doc.add_paragraph()
    sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sp.paragraph_format.space_before = Pt(8)
    r = sp.add_run("A Two-Stage Hierarchical Clinical Triage & 9-Class Deep Ensemble System with Grad-CAM Visual Explainability")
    r.font.name = "Calibri"
    r.font.size = Pt(13.5)
    r.font.italic = True
    r.font.color.rgb = SLATE

    subp = doc.add_paragraph()
    subp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subp.paragraph_format.space_before = Pt(36)
    r = subp.add_run("Submitted in partial fulfillment of the requirements for the award of the degree in\nComputer Science & Engineering")
    r.font.name = "Calibri"
    r.font.size = Pt(11.5)
    r.font.color.rgb = SLATE

    by = doc.add_paragraph()
    by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    by.paragraph_format.space_before = Pt(32)
    r = by.add_run("Submitted by (Group: B55)")
    r.font.name = "Calibri"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = TEAL

    add_styled_table(
        doc,
        headers=["Student Name", "Roll / Registration No.", "Group"],
        rows=[
            ["Debayudh Kundu", "2351062", "B55"],
            ["Aditiya Majumdar", "2351058", "B55"],
        ],
    )

    dept = doc.add_paragraph()
    dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    dept.paragraph_format.space_before = Pt(44)
    r = dept.add_run(
        "Department of Computer Science & Engineering\n"
        "Academic Year: 2025–2026\n\n"
        "Repository: https://github.com/debayudh482/skin-cancer-detection"
    )
    r.font.name = "Calibri"
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = NAVY

    doc.add_page_break()

    # ==================== CERTIFICATE & DECLARATION ====================
    add_heading_styled(doc, "CERTIFICATE", level=1)
    add_body(
        doc,
        "This is to certify that the project report entitled \"Skin Cancer Detection from Dermoscopic Images Using Deep Learning: "
        "A Two-Stage Hierarchical Clinical Triage & 9-Class Deep Ensemble System with Grad-CAM Visual Explainability\" is a bonafide "
        "work carried out by Debayudh Kundu (2351062) and Aditiya Majumdar (2351058), Group B55, of the Department of Computer "
        "Science & Engineering during the Academic Year 2025–2026 in partial fulfillment of the academic requirements."
    )
    sig = doc.add_paragraph()
    sig.paragraph_format.space_before = Pt(36)
    r = sig.add_run("Project Guide Signature: _______________________          Head of Department: _______________________")
    r.font.name = "Calibri"
    r.font.size = Pt(10.5)
    r.font.color.rgb = NAVY

    add_heading_styled(doc, "DECLARATION", level=1)
    add_body(
        doc,
        "We hereby declare that the project work presented in this report is an authentic record of our own work carried out under "
        "the guidance of the Department of Computer Science & Engineering. All experimental results, code implementations, and "
        "visualizations reported herein were genuinely produced and verified by our team, and all external literature sources "
        "and public datasets (ISIC) have been appropriately cited."
    )
    sig2 = doc.add_paragraph()
    sig2.paragraph_format.space_before = Pt(28)
    r2 = sig2.add_run("Debayudh Kundu (2351062): ___________________          Aditiya Majumdar (2351058): ___________________")
    r2.font.name = "Calibri"
    r2.font.size = Pt(10.5)
    r2.font.color.rgb = NAVY

    add_heading_styled(doc, "ACKNOWLEDGEMENT", level=1)
    add_body(
        doc,
        "We express our sincere gratitude to the faculty and mentors of the Department of Computer Science & Engineering for their "
        "continuous support, valuable feedback, and encouragement throughout this project. We also acknowledge the International "
        "Skin Imaging Collaboration (ISIC) for maintaining open-access dermoscopic imaging benchmarks that make reproducible "
        "medical AI research possible."
    )

    doc.add_page_break()

    # ==================== ABSTRACT & TABLE OF CONTENTS ====================
    add_heading_styled(doc, "ABSTRACT", level=1)
    add_body(
        doc,
        "Skin cancer is the most prevalent malignancy worldwide, with melanoma accounting for the majority of skin-cancer-related "
        "mortalities due to its rapid metastatic potential. Early detection significantly improves patient survival, yet manual "
        "visual and dermoscopic screening remains time-consuming, subjective, and prone to inter-observer variability. Existing "
        "student and research prototypes commonly focus solely on binary (benign vs. malignant) classification without subtype "
        "granularity or visual explainability."
    )
    add_body(
        doc,
        "In this project, we design, train, and deploy a Two-Stage Hierarchical Clinical Decision Support System for dermoscopic "
        "skin lesion analysis on the 9-class ISIC dataset (2,357 images). First, we replace a basic frozen-backbone MobileNetV2 "
        "baseline (35.60% validation accuracy) with a two-stage transfer learning pipeline featuring full-network fine-tuning, "
        "ImageNet normalization, targeted dermoscopic augmentations, smoothed inverse-frequency class weighting, label smoothing, "
        "and 4x Test-Time Augmentation (TTA). Second, we build a weighted soft-voting ensemble combining EfficientNet-B0 and "
        "MobileNetV2, more than doubling 9-class validation accuracy to 71.73% with a Macro ROC-AUC of 0.9390 (0.9272 on held-out "
        "test data). Third, we train a dedicated Stage-1 binary clinical triage model and fuse its malignant probability with the "
        "9-class ensemble, achieving 81.55% validation accuracy, 89.2% cancer sensitivity, and 0.8576 ROC-AUC—outperforming both "
        "standalone binary (77.98%) and aggregated 9-class (80.36%) models. Finally, we integrate Grad-CAM visual explainability "
        "into a live interactive Streamlit web application, providing clinicians with Risk Tier alerts, Top-3 subtype probabilities, "
        "and lesion attention heatmaps in real time."
    )
    add_body(
        doc,
        "Skin Cancer Detection, Dermoscopy, ISIC Dataset, Transfer Learning, EfficientNet-B0, MobileNetV2, Deep Ensemble, "
        "Hierarchical Triage, Test-Time Augmentation (TTA), Grad-CAM, Explainable AI (XAI), Streamlit.",
        bold_prefix="Keywords: ",
    )

    add_heading_styled(doc, "TABLE OF CONTENTS", level=1)
    toc_items = [
        "1.  Introduction (Background, Problem Statement, Objectives & Uniqueness)",
        "2.  Literature Review & Research Gap Analysis",
        "3.  Dataset Description & Exploratory Analysis (ISIC 9-Class)",
        "4.  Proposed Methodology & 3-Stage Hierarchical Architecture",
        "5.  System Implementation, Technology Stack & Live Web Application",
        "6.  Experimental Results, Evaluation Metrics & Visual Analysis",
        "7.  Advantages, Limitations & Future Scope",
        "8.  Conclusion",
        "9.  References",
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(3)
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(11)
            r.font.color.rgb = SLATE

    doc.add_page_break()

    # ==================== CHAPTER 1: INTRODUCTION ====================
    add_heading_styled(doc, "1. INTRODUCTION", level=1)

    add_heading_styled(doc, "1.1 Clinical Background", level=2)
    add_body(
        doc,
        "Cancer arises from the uncontrolled proliferation of abnormal cells capable of invading surrounding tissues and "
        "metastasizing to distant organs. According to the World Health Organization (WHO), skin cancer is among the most "
        "frequently diagnosed cancers globally, with approximately 3 million non-melanoma cases and over 132,000 melanoma cases "
        "recorded each year. The primary subtypes include Basal Cell Carcinoma (BCC), Squamous Cell Carcinoma (SCC), and "
        "Cutaneous Melanoma—the most aggressive and lethal form. Early-stage detection dramatically improves survival rates and "
        "reduces treatment morbidity."
    )

    add_heading_styled(doc, "1.2 Problem Statement", level=2)
    add_body(
        doc,
        "In routine dermatology, unaided visual inspection yields an average diagnostic accuracy of roughly 60%, which improves "
        "to around 89% with dermoscopy when performed by experienced specialists. However, manual dermoscopic examination is slow, "
        "resource-intensive, and suffers from substantial inter-observer variability—especially for visually overlapping lesions "
        "such as early melanoma vs. atypical nevus, or seborrheic keratosis vs. pigmented benign keratosis. Furthermore, rural and "
        "resource-constrained clinics often lack access to board-certified dermatologists. An accurate, transparent, computer-aided "
        "diagnosis (CAD) system can serve as a reliable second opinion to accelerate triage and reduce missed malignancies."
    )

    add_heading_styled(doc, "1.3 Objectives & Unique Contributions", level=2)
    add_bullet(doc, "Develop a reproducible, end-to-end PyTorch deep learning pipeline for 9-class dermoscopic image classification on the ISIC dataset.", bold_prefix="Automated Multi-Class Pipeline: ")
    add_bullet(doc, "Introduce a Two-Stage Hierarchical Clinical Triage framework that fuses a dedicated binary screening model with a 9-class ensemble to maximize cancer detection sensitivity (89.2%) and screening accuracy (81.55%).", bold_prefix="Hierarchical Fusion Triage: ")
    add_bullet(doc, "Combine diverse lightweight CNN backbones (EfficientNet-B0 + MobileNetV2) with 4x Test-Time Augmentation (TTA) so the system runs efficiently even on standard CPU hardware.", bold_prefix="Efficient Deep Ensemble + TTA: ")
    add_bullet(doc, "Integrate Grad-CAM visual heatmaps inside a live interactive Streamlit clinical web application to eliminate 'black-box' opacity and support real-time demonstrations.", bold_prefix="Explainable AI & Web Deployment: ")

    # ==================== CHAPTER 2: LITERATURE REVIEW ====================
    add_heading_styled(doc, "2. LITERATURE REVIEW", level=1)
    add_body(
        doc,
        "Deep convolutional neural networks (CNNs) and ensemble learning have transformed automated dermatological imaging. "
        "We reviewed three peer-reviewed studies directly aligned with ISIC skin cancer classification:"
    )

    add_styled_table(
        doc,
        headers=["Study", "Dataset & Task", "Methodology", "Key Results", "Insights / Limitations"],
        rows=[
            ["Imran et al. (2022)\nIEEE Access", "ISIC 2019\n(Binary, 5,600 imgs)", "Majority-voting ensemble of VGG16 + CapsNet + ResNet", "93.5% Acc, 87.3% Sens, 94% Prec", "Proves ensemble diversity beats individual CNNs; limited to binary classes without XAI."],
            ["Gouda et al. (2022)\nHealthcare (MDPI)", "ISIC 2018\n(Binary, 3,533 imgs)", "ESRGAN enhancement + CNN, ResNet50, InceptionV3, InceptionResNet", "85.8% Acc (InceptionV3), 0.86 AUC", "Highlights image preprocessing & fine-tuning benefits; binary-only without subtype detail."],
            ["Hameed et al. (2024)\nCMES (Review)", "ISIC 2016–2020\n(PRISMA Systematic Review)", "Review of 197 classification & 184 segmentation CNN/ViT papers", "76.1%–99.3% across benchmarks", "Identifies class imbalance, overfitting, and naive augmentation duplication as major pitfalls."],
            ["Proposed System (2026)\nGroup B55", "ISIC 9-Class +\nBinary Triage (2,357 imgs)", "Hierarchical Fusion (Binary EffNet-B0 + 9-Class EffNet-B0/MobileNetV2 + 4x TTA) + Grad-CAM", "81.55% Triage (89.2% Sens) | 71.73% 9-Class (0.939 AUC)", "Combines triage safety, 9-class granularity, Grad-CAM XAI, and live Streamlit deployment."],
        ],
        highlight_last=True,
    )

    add_heading_styled(doc, "2.1 Identified Research Gap", level=2)
    add_body(
        doc,
        "While prior studies demonstrate high binary classification accuracy on larger ISIC subsets, most do not simultaneously "
        "provide (i) high-sensitivity clinical risk triage, (ii) fine-grained 9-class subtype probabilities, (iii) real-time "
        "visual explainability via Grad-CAM, and (iv) an accessible web interface capable of running on standard CPU hardware. "
        "Our project directly addresses this gap through a lightweight hierarchical fusion architecture."
    )

    # ==================== CHAPTER 3: DATASET ====================
    add_heading_styled(doc, "3. DATASET DESCRIPTION & EXPLORATORY ANALYSIS", level=1)
    add_body(
        doc,
        "The project uses the ISIC 9-Class Dermoscopic Skin Cancer dataset (Skin_ISIC.zip, 824 MB), containing 2,357 RGB "
        "dermoscopic photographs organized into labeled Train (2,239 images) and Test (118 images) directories across 9 skin "
        "lesion categories. To prevent data leakage, the Train directory is partitioned using a stratified 85/15 split into "
        "1,903 training images and 336 validation images, while the 118-image Test set remains strictly held out."
    )

    add_styled_table(
        doc,
        headers=["Class Name", "Clinical Category", "Train (85%)", "Val (15%)", "Held-Out Test"],
        rows=[
            ["Pigmented Benign Keratosis", "Benign", "393", "69", "16"],
            ["Melanoma", "Malignant", "372", "66", "16"],
            ["Basal Cell Carcinoma (BCC)", "Malignant", "320", "56", "16"],
            ["Nevus", "Benign", "303", "54", "16"],
            ["Squamous Cell Carcinoma (SCC)", "Malignant", "154", "27", "16"],
            ["Vascular Lesion", "Benign", "118", "21", "3"],
            ["Actinic Keratosis", "Pre-Malignant", "97", "17", "16"],
            ["Dermatofibroma", "Benign", "81", "14", "16"],
            ["Seborrheic Keratosis", "Benign", "65", "12", "3"],
            ["Total (9 Classes)", "4 Mal/Pre-Mal | 5 Benign", "1,903", "336", "118"],
        ],
        highlight_last=True,
    )

    add_body(
        doc,
        "As shown above, the dataset exhibits substantial class imbalance (~6:1 ratio between Pigmented Benign Keratosis and "
        "Seborrheic Keratosis) and a small test support (only 3 images each for Seborrheic Keratosis and Vascular Lesion). "
        "Accordingly, we report Macro F1-score, Macro ROC-AUC, Sensitivity, and Specificity alongside raw accuracy."
    )

    # ==================== CHAPTER 4: METHODOLOGY ====================
    add_heading_styled(doc, "4. PROPOSED METHODOLOGY & ARCHITECTURE", level=1)

    add_heading_styled(doc, "4.1 Preprocessing & Augmentation", level=2)
    add_bullet(doc, "All 2,357 images are pre-resized once to 256×256 JPEGs (src/prepare_data.py) to accelerate data loading.", bold_prefix="Fast Pre-Resizing: ")
    add_bullet(doc, "RandomResizedCrop (224×224, scale 0.75–1.0), horizontal & vertical flips (p=0.5), random rotation (±30°), ColorJitter (brightness/contrast/saturation=0.2), and RandomErasing (p=0.15).", bold_prefix="Training Augmentation: ")
    add_bullet(doc, "Standard ImageNet channel normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), replacing naive [0,1] scaling which previously degraded pretrained feature extraction.", bold_prefix="ImageNet Standardization: ")

    add_heading_styled(doc, "4.2 Two-Stage Transfer Learning & Class Weighting", level=2)
    add_body(
        doc,
        "Each CNN backbone (EfficientNet-B0, MobileNetV2, and optional DenseNet121/ResNet50) is initialized with pretrained "
        "ImageNet weights and paired with a custom Dropout(0.3) → Linear classification head. Training proceeds in two stages:"
    )
    add_bullet(doc, "Backbone layers are frozen and only the classification head is trained using AdamW (lr = 1e-3, weight_decay = 1e-2) so random head weights do not corrupt pretrained filters.", bold_prefix="Stage 1 (Warmup): ")
    add_bullet(doc, "The network is unfrozen with differential learning rates (backbone lr = 1.5e-4 to 2e-4, head lr = 5e-4 to 1e-3), CosineAnnealingLR decay, and gradient clipping (max_norm = 2.0).", bold_prefix="Stage 2 (Fine-Tuning): ")
    add_bullet(doc, "Smoothed inverse-frequency class weights w_c = (N / (K · n_c))^0.35 (normalized to mean 1.0) combined with Cross-Entropy Loss and label smoothing (0.05) mitigate class imbalance without destabilizing gradients.", bold_prefix="Imbalance-Aware Loss: ")

    add_heading_styled(doc, "4.3 Three-Stage Hierarchical Clinical Inference", level=2)
    add_bullet(doc, "At inference time, each model evaluates 4 deterministic views (original, horizontal flip, vertical flip, both flips) and averages the softmax probabilities (4x TTA). Stage-2 combines EfficientNet-B0 and MobileNetV2 via validation-weighted soft voting.", bold_prefix="Ensemble + 4x TTA: ")
    add_bullet(doc, "A dedicated Stage-1 binary EfficientNet-B0 classifier predicts P_bin(Malignant). This is fused 50/50 with the summed malignant-class probabilities P_9cls(Malignant) from the 9-class ensemble: P_fused = 0.5 · P_bin + 0.5 · P_9cls. Thresholds map P_fused to Low (<35%), Moderate (35%–65%), and High (≥65%) Risk tiers.", bold_prefix="Hierarchical Fusion Triage: ")
    add_bullet(doc, "Forward and backward hooks on the final convolutional block compute class-discriminative gradients, global-average-pooled into channel weights and overlaid as a jet heatmap onto the 224×224 lesion image.", bold_prefix="Stage-3 Grad-CAM Explainability: ")

    # ==================== CHAPTER 5: IMPLEMENTATION & TECH STACK ====================
    add_heading_styled(doc, "5. IMPLEMENTATION & TECHNOLOGY STACK", level=1)

    add_styled_table(
        doc,
        headers=["Layer", "Technologies & Tools Used"],
        rows=[
            ["Language & Environment", "Python 3.12, Isolated Virtual Environment (.venv), Git, GitHub, GitHub CLI (gh)"],
            ["Deep Learning & Vision", "PyTorch 2.15, Torchvision 0.29 (EfficientNet-B0, MobileNetV2, DenseNet121, ResNet50), Pillow, NumPy"],
            ["Evaluation & Explainability", "Scikit-learn 1.9, Custom PyTorch Grad-CAM, Matplotlib 3.11, Seaborn 0.13"],
            ["Interactive Web UI", "Streamlit 1.67 (running on Uvicorn + Starlette + WebSockets) — live clinical dashboard (app.py)"],
            ["Cloud GPU Scaling", "1-Click Google Colab T4 GPU Notebook (notebooks/train_colab_ham10000.ipynb)"],
        ],
    )

    add_heading_styled(doc, "5.1 Modular Codebase Structure", level=2)
    add_bullet(doc, "Central paths, ImageNet stats, malignant class grouping, hyperparameters, and automatic CPU/CUDA device detection.", bold_prefix="src/config.py: ")
    add_bullet(doc, "Multi-threaded 256×256 dataset pre-resizer (prepares 2,357 images in ~6.8 seconds).", bold_prefix="src/prepare_data.py: ")
    add_bullet(doc, "Stratified 85/15 splitting, dermoscopic augmentation pipelines, and smoothed class-weight calculation.", bold_prefix="src/dataset.py: ")
    add_bullet(doc, "Transfer learning model factory (MobileNetV2, EfficientNet-B0, DenseNet121, ResNet50) with freeze/unfreeze and Grad-CAM layer selectors.", bold_prefix="src/model.py: ")
    add_bullet(doc, "Two-stage 9-class trainer with checkpoint resuming, CosineAnnealingLR, and best-model saving.", bold_prefix="src/train.py: ")
    add_bullet(doc, "Dedicated Stage-1 binary triage trainer and 50/50 hierarchical fusion evaluator.", bold_prefix="src/train_binary.py: ")
    add_bullet(doc, "Weighted soft-voting ensemble evaluator with 4x TTA, 9-class & binary metrics, and plot generation.", bold_prefix="src/ensemble.py & src/evaluate.py: ")
    add_bullet(doc, "Grad-CAM heatmap engine, CLI hierarchical predictor, and live Streamlit clinical web application.", bold_prefix="src/gradcam.py, src/predict_hierarchical.py & app.py: ")

    # ==================== CHAPTER 6: EXPERIMENTAL RESULTS ====================
    add_heading_styled(doc, "6. EXPERIMENTAL RESULTS & PERFORMANCE ANALYSIS", level=1)

    add_heading_styled(doc, "6.1 Stage-1 Binary Clinical Triage Results", level=2)
    add_body(
        doc,
        "Fusing the dedicated Stage-1 binary EfficientNet-B0 model with the 9-class ensemble consistently outperformed both "
        "standalone approaches on validation screening, achieving 81.55% accuracy and 89.2% cancer sensitivity:"
    )
    add_styled_table(
        doc,
        headers=["Model / Strategy (4x TTA)", "Val Accuracy", "Sensitivity (Cancer Recall)", "Specificity", "ROC-AUC"],
        rows=[
            ["Standalone Binary EfficientNet-B0", "77.98%", "86.1%", "70.0%", "0.8445"],
            ["9-Class Ensemble Aggregated to Binary", "80.36%", "84.3%", "76.5%", "0.8553"],
            ["Hierarchical Fusion (Binary + 9-Class Ensemble)", "81.55%", "89.2%", "74.1%", "0.8576"],
        ],
        highlight_last=True,
    )

    add_heading_styled(doc, "6.2 Stage-2 9-Class Fine-Grained Progression", level=2)
    add_body(
        doc,
        "Compared to the original frozen MobileNetV2 baseline (35.60% validation accuracy), our two-stage fine-tuning and "
        "multi-model ensemble more than doubled 9-class accuracy to 71.73% with a Macro ROC-AUC of 0.9390 (and 0.9272 on "
        "the held-out 118-image test set):"
    )
    add_styled_table(
        doc,
        headers=["Experiment / Architecture", "9-Class Val Acc", "Macro F1", "Macro ROC-AUC"],
        rows=[
            ["Original Slides Baseline (Frozen MobileNetV2, 10 ep)", "35.60%", "—", "—"],
            ["MobileNetV2 (2-Stage Top-Block Fine-Tune)", "66.07%", "0.5754", "0.9275"],
            ["EfficientNet-B0 (Full Fine-Tune + 4x TTA)", "69.64%", "—", "—"],
            ["Ensemble: EfficientNet-B0 + MobileNetV2 (4x TTA)", "71.73%", "0.6125", "0.9390"],
        ],
        highlight_last=True,
    )

    # Per-class validation table from metrics.json
    val_rep = metrics_json.get("validation_metrics_9class", {}).get("classification_report", {})
    if val_rep:
        add_heading_styled(doc, "6.3 Per-Class Validation Breakdown (9-Class Ensemble)", level=2)
        pc_rows = []
        for cls_name in metrics_json["dataset"]["class_names"]:
            if cls_name in val_rep:
                d = val_rep[cls_name]
                pc_rows.append([
                    cls_name.title(),
                    f"{d['precision']*100:.1f}%",
                    f"{d['recall']*100:.1f}%",
                    f"{d['f1-score']*100:.1f}%",
                    f"{int(d['support'])}",
                ])
        pc_rows.append([
            "Macro Average (9 Classes)",
            f"{val_rep['macro avg']['precision']*100:.1f}%",
            f"{val_rep['macro avg']['recall']*100:.1f}%",
            f"{val_rep['macro avg']['f1-score']*100:.1f}%",
            f"{int(val_rep['macro avg']['support'])}",
        ])
        add_styled_table(doc, ["Lesion Class", "Precision", "Recall", "F1-Score", "Support"], pc_rows, highlight_last=True)

    add_heading_styled(doc, "6.4 Visual Evaluation & Grad-CAM Explainability", level=2)
    add_figure(doc, OUTPUT_DIR / "training_history.png", "Figure 6.1: Two-stage training and validation loss/accuracy learning curves.", width_in=5.8)
    add_figure(doc, OUTPUT_DIR / "binary_triage_evaluation.png", "Figure 6.2: Stage-1 Hierarchical Binary Triage confusion matrix and ROC curve (Val Acc = 81.55%, Sens = 89.2%, AUC = 0.8576).", width_in=5.8)
    add_figure(doc, OUTPUT_DIR / "confusion_matrix_val.png", "Figure 6.3: Stage-2 9-Class Ensemble validation confusion matrix (n = 336).", width_in=5.2)
    add_figure(doc, OUTPUT_DIR / "roc_curves_val.png", "Figure 6.4: Multi-class one-vs-rest ROC curves across all 9 ISIC lesion classes (Macro AUC = 0.9390).", width_in=5.2)
    add_figure(doc, OUTPUT_DIR / "gradcam_samples.png", "Figure 6.5: Grad-CAM visual explanations on held-out ISIC test lesions showing model focus on pigmented lesion structures.", width_in=6.0)
    add_figure(doc, OUTPUT_DIR / "hierarchical_sample_gradcam.png", "Figure 6.6: Live hierarchical inference Grad-CAM overlay on held-out Basal Cell Carcinoma test image (Stage-1 High Risk 87.11%, Stage-2 BCC 80.31%).", width_in=2.8)

    # ==================== CHAPTER 7: ADVANTAGES, LIMITATIONS & FUTURE SCOPE ====================
    add_heading_styled(doc, "7. ADVANTAGES, LIMITATIONS & FUTURE SCOPE", level=1)

    add_heading_styled(doc, "7.1 Key Advantages", level=2)
    add_bullet(doc, "Achieves 81.55% triage accuracy and 89.2% cancer sensitivity, prioritizing clinical safety against missed malignancies.")
    add_bullet(doc, "More than doubles 9-class classification accuracy over the initial frozen baseline (35.60% → 71.73%, Macro ROC-AUC = 0.9390).")
    add_bullet(doc, "Provides transparent Grad-CAM visual explanations alongside probabilistic scores inside a live, interactive Streamlit web app.")
    add_bullet(doc, "Lightweight architecture (EfficientNet-B0 + MobileNetV2) runs fast inference on standard CPU hardware with zero cloud GPU cost.")

    add_heading_styled(doc, "7.2 Current Limitations", level=2)
    add_bullet(doc, "The 2,239-image training dataset is relatively small for 9 fine-grained classes, particularly Seborrheic Keratosis (65 training samples), which visually overlaps with Pigmented Benign Keratosis.")
    add_bullet(doc, "The held-out Test folder has limited representation for two classes (3 samples each for Vascular Lesion and Seborrheic Keratosis).")
    add_bullet(doc, "This system is an academic decision-support prototype and is not a replacement for clinical biopsy or board-certified dermatological diagnosis.")

    add_heading_styled(doc, "7.3 Future Scope", level=2)
    add_bullet(doc, "Scale training to HAM10000 (10,015 images) and ISIC 2019 (25,331 images) using the included Google Colab T4 GPU notebook with a 3-model ensemble (+ DenseNet121) to exceed 85%+ multi-class accuracy.")
    add_bullet(doc, "Incorporate Vision Transformers (ViT / Swin Transformer) and fuse clinical metadata (patient age, sex, anatomical site) with dermoscopic images.")
    add_bullet(doc, "Deploy as a production FastAPI + React/Flutter mobile teledermatology screening tool.")

    # ==================== CHAPTER 8: CONCLUSION ====================
    add_heading_styled(doc, "8. CONCLUSION", level=1)
    add_body(
        doc,
        "In this project, we successfully developed and validated an end-to-end Hierarchical Skin Cancer Detection System using "
        "deep transfer learning, multi-model ensembling, Test-Time Augmentation, and Grad-CAM explainability on the ISIC "
        "dermoscopic dataset. By transitioning from a basic frozen MobileNetV2 baseline (35.60%) to a two-stage fine-tuned "
        "EfficientNet-B0 + MobileNetV2 ensemble fused with a dedicated Stage-1 binary triage classifier, we achieved 81.55% "
        "binary screening accuracy, 89.2% cancer sensitivity, 71.73% 9-class accuracy, and 0.9390 Macro ROC-AUC. Wrapped in a "
        "live Streamlit clinical web application and backed by a reproducible GitHub repository and Colab GPU notebook, the "
        "system demonstrates a practical, transparent, and clinically aligned AI workflow for early skin cancer screening."
    )

    # ==================== CHAPTER 9: REFERENCES ====================
    add_heading_styled(doc, "9. REFERENCES", level=1)
    refs = [
        "[1] A. Imran, A. Nasir, M. Bilal, G. Sun, A. Alzahrani, and A. Almuhaimeed, \"Skin Cancer Detection Using Combined Decision of Deep Learners,\" IEEE Access, vol. 10, pp. 118198–118212, Nov. 2022, doi: 10.1109/ACCESS.2022.3220329.",
        "[2] W. Gouda, N. U. Sama, G. Al-Waakid, M. Humayun, and N. Z. Jhanjhi, \"Detection of Skin Cancer Based on Skin Lesion Images Using Deep Learning,\" Healthcare (MDPI), vol. 10, no. 7, p. 1183, Jun. 2022, doi: 10.3390/healthcare10071183.",
        "[3] M. Hameed, A. Zameer, and M. A. Z. Raja, \"A Comprehensive Systematic Review: Advancements in Skin Cancer Classification and Segmentation Using the ISIC Dataset,\" Computer Modeling in Engineering & Sciences (CMES), 2024, doi: 10.32604/cmes.2024.050124.",
        "[4] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, \"MobileNetV2: Inverted Residuals and Linear Bottlenecks,\" in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.",
        "[5] M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,\" in Proc. 36th Int. Conf. Machine Learning (ICML), 2019, pp. 6105–6114.",
        "[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, \"Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,\" in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626.",
        "[7] International Skin Imaging Collaboration (ISIC), \"ISIC Archive — Dermoscopic Skin Lesion Dataset,\" https://www.isic-archive.com/.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        for r in p.runs:
            r.font.name = "Calibri"
            r.font.size = Pt(10)
            r.font.color.rgb = SLATE

    doc.save(str(REPORT_PATH))
    print(f"Project report saved successfully -> {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
