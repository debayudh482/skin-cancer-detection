"""Generate a polished 12-slide Project Viva / Seminar PowerPoint presentation (.pptx)."""

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PPTX_PATH = PROJECT_ROOT / "Skin_Cancer_Detection_Final_Presentation.pptx"

# Clinical Modern Theme Palette
NAVY = RGBColor(15, 23, 42)        # #0F172A (Primary dark)
SLATE = RGBColor(51, 65, 85)       # #334155 (Body text)
MUTED = RGBColor(100, 116, 139)    # #64748B (Captions)
TEAL = RGBColor(13, 148, 136)      # #0D9488 (Primary accent)
TEAL_LIGHT = RGBColor(240, 253, 250)
BLUE = RGBColor(37, 99, 235)       # #2563EB (Secondary accent)
CORAL = RGBColor(225, 29, 72)      # #E11D48 (Alert / cancer highlight)
AMBER = RGBColor(217, 119, 6)      # #D97706
BG_LIGHT = RGBColor(248, 250, 252) # #F8FAFC (Slide canvas)
WHITE = RGBColor(255, 255, 255)
BORDER = RGBColor(226, 232, 240)   # #E2E8F0

FONT_MAIN = "Segoe UI"


def set_bg(slide, color: RGBColor):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title: str, subtitle: str, page_num: int, total_pages: int = 12):
    """Consistent top header bar + subtle footer on content slides."""
    # Top accent strip
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    # Title
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.25), Inches(11.5), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT_MAIN
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

    # Subtitle
    sb = slide.shapes.add_textbox(Inches(0.6), Inches(0.8), Inches(11.8), Inches(0.35))
    stf = sb.text_frame
    stf.word_wrap = True
    sp = stf.paragraphs[0]
    sp.text = subtitle
    sp.font.name = FONT_MAIN
    sp.font.size = Pt(13)
    sp.font.color.rgb = MUTED

    # Divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.2), Inches(12.133), Inches(0.02))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    # Footer
    ft = slide.shapes.add_textbox(Inches(0.6), Inches(7.1), Inches(9.5), Inches(0.3))
    fp = ft.text_frame.paragraphs[0]
    fp.text = "Hierarchical Skin Cancer Detection System  |  Group B55  |  Dept. of CSE (2025–2026)"
    fp.font.name = FONT_MAIN
    fp.font.size = Pt(9.5)
    fp.font.color.rgb = MUTED

    pg = slide.shapes.add_textbox(Inches(11.7), Inches(7.1), Inches(1.0), Inches(0.3))
    pp = pg.text_frame.paragraphs[0]
    pp.text = f"{page_num} / {total_pages}"
    pp.alignment = PP_ALIGN.RIGHT
    pp.font.name = FONT_MAIN
    pp.font.size = Pt(9.5)
    pp.font.bold = True
    pp.font.color.rgb = TEAL


def add_card(slide, left, top, width, height, title=None, bullets=None, accent=TEAL, fill_color=WHITE, title_size=15, body_size=12):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.04
    card.fill.solid()
    card.fill.fore_color.rgb = fill_color
    card.line.color.rgb = BORDER
    card.line.width = Pt(1)

    # Left accent bar inside card
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top + 0.12), Inches(0.07), Inches(height - 0.24))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.2), Inches(top + 0.12), Inches(width - 0.32), Inches(height - 0.22))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    if title:
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_MAIN
        p.font.size = Pt(title_size)
        p.font.bold = True
        p.font.color.rgb = NAVY
        p.space_after = Pt(6)
        first = False

    if bullets:
        for b in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = f"•  {b}"
            p.font.name = FONT_MAIN
            p.font.size = Pt(body_size)
            p.font.color.rgb = SLATE
            p.space_after = Pt(4)
    return card


def add_kpi(slide, left, top, width, height, value, label, sublabel, accent=TEAL):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.08
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = accent
    card.line.width = Pt(1.5)

    tb = slide.shapes.add_textbox(Inches(left + 0.12), Inches(top + 0.12), Inches(width - 0.24), Inches(height - 0.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = value
    p1.alignment = PP_ALIGN.CENTER
    p1.font.name = FONT_MAIN
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.text = label
    p2.alignment = PP_ALIGN.CENTER
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(12)
    p2.font.bold = True
    p2.font.color.rgb = NAVY
    p2.space_before = Pt(2)

    p3 = tf.add_paragraph()
    p3.text = sublabel
    p3.alignment = PP_ALIGN.CENTER
    p3.font.name = FONT_MAIN
    p3.font.size = Pt(10)
    p3.font.color.rgb = MUTED


def add_table(slide, left, top, width, height, headers, rows, col_widths=None, highlight_last=False):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    # Header row
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_MAIN
        p.font.size = Pt(11.5)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT

    # Data rows
    for r_idx, row in enumerate(rows):
        is_last = highlight_last and (r_idx == len(rows) - 1)
        bg = TEAL_LIGHT if is_last else (WHITE if r_idx % 2 == 0 else BG_LIGHT)
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = FONT_MAIN
            p.font.size = Pt(11)
            p.font.bold = is_last
            p.font.color.rgb = NAVY if is_last else SLATE
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
    return shape


def add_picture_fitted(slide, img_path: Path, left, top, max_w, max_h):
    """Insert image preserving aspect ratio inside (max_w, max_h) box."""
    if not img_path.exists():
        return
    from PIL import Image as PILImage
    with PILImage.open(img_path) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = iw * ratio, ih * ratio
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    slide.shapes.add_picture(str(img_path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ==================== SLIDE 1: TITLE SLIDE ====================
    s1 = prs.slides.add_slide(blank)
    set_bg(s1, NAVY)

    # Decorative teal left bar
    accent = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), Inches(7.5))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()

    # Pill tag
    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(1.1), Inches(4.5), Inches(0.45))
    pill.adjustments[0] = 0.5
    pill.fill.solid()
    pill.fill.fore_color.rgb = TEAL
    pill.line.fill.background()
    ptf = pill.text_frame.paragraphs[0]
    ptf.text = "PROJECT VIVA / SEMINAR PRESENTATION  •  2025–2026"
    ptf.alignment = PP_ALIGN.CENTER
    ptf.font.name = FONT_MAIN
    ptf.font.size = Pt(11)
    ptf.font.bold = True
    ptf.font.color.rgb = WHITE

    # Main Title
    tb = s1.shapes.add_textbox(Inches(0.9), Inches(1.8), Inches(11.4), Inches(2.0))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Skin Cancer Detection from Dermoscopic Images Using Deep Learning"
    p.font.name = FONT_MAIN
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "A Two-Stage Hierarchical Clinical Triage & 9-Class Ensemble System with Grad-CAM Explainability"
    p2.font.name = FONT_MAIN
    p2.font.size = Pt(19)
    p2.font.color.rgb = RGBColor(148, 226, 213)
    p2.space_before = Pt(12)

    # Team info box
    box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.9), Inches(4.55), Inches(11.4), Inches(1.85))
    box.adjustments[0] = 0.06
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(30, 41, 59)
    box.line.color.rgb = TEAL
    box.line.width = Pt(1.2)

    itb = s1.shapes.add_textbox(Inches(1.2), Inches(4.78), Inches(10.8), Inches(1.4))
    itf = itb.text_frame
    itf.word_wrap = True
    ip1 = itf.paragraphs[0]
    ip1.text = "Presented by:   Debayudh Kundu (2351062)   &   Aditiya Majumdar (2351058)"
    ip1.font.name = FONT_MAIN
    ip1.font.size = Pt(17)
    ip1.font.bold = True
    ip1.font.color.rgb = WHITE

    ip2 = itf.add_paragraph()
    ip2.text = "Department of Computer Science & Engineering   |   Group: B55   |   Academic Year: 2025–2026"
    ip2.font.name = FONT_MAIN
    ip2.font.size = Pt(13.5)
    ip2.font.color.rgb = RGBColor(203, 213, 225)
    ip2.space_before = Pt(8)

    ip3 = itf.add_paragraph()
    ip3.text = "GitHub Repository:  github.com/debayudh482/skin-cancer-detection"
    ip3.font.name = FONT_MAIN
    ip3.font.size = Pt(12.5)
    ip3.font.color.rgb = RGBColor(94, 234, 212)
    ip3.space_before = Pt(8)

    # ==================== SLIDE 2: INTRODUCTION & PROBLEM ====================
    s2 = prs.slides.add_slide(blank)
    set_bg(s2, BG_LIGHT)
    add_header(s2, "Introduction & Problem Statement", "Why automated, explainable AI screening is critical for skin cancer", 2)

    add_card(
        s2, 0.6, 1.45, 5.9, 2.55,
        title="Clinical Background",
        bullets=[
            "Skin cancer is the most prevalent cancer worldwide (>3M non-melanoma & ~132K melanoma cases annually — WHO).",
            "Melanoma is the deadliest form due to rapid metastasis if not caught early.",
            "Basal Cell Carcinoma (BCC) and Squamous Cell Carcinoma (SCC) are the most common malignant types.",
        ],
        accent=CORAL,
    )
    add_card(
        s2, 6.8, 1.45, 5.9, 2.55,
        title="The Diagnostic Gap",
        bullets=[
            "Unaided visual examination achieves only ~60% accuracy; dermoscopy improves it to ~89% (Hameed et al., 2024).",
            "Manual dermoscopic review is slow, subjective, and suffers from inter-observer variability.",
            "Visually overlapping lesions (e.g., Melanoma vs. Nevus; Pigmented vs. Seborrheic Keratosis) cause misdiagnosis.",
        ],
        accent=AMBER,
    )
    add_card(
        s2, 0.6, 4.25, 12.1, 2.55,
        title="Project Objectives — Closing the Gap with Hierarchical AI",
        bullets=[
            "Stage 1 (Clinical Triage): Build a high-sensitivity binary screener (Malignant/Pre-malignant vs. Benign) to minimize missed cancers.",
            "Stage 2 (Fine-Grained Diagnosis): Classify dermoscopic lesions across all 9 ISIC categories using a multi-model deep ensemble + 4x TTA.",
            "Stage 3 (Explainable AI & Deployment): Provide Grad-CAM visual heatmaps inside a live interactive web app for clinical decision support.",
        ],
        accent=TEAL,
        fill_color=TEAL_LIGHT,
    )

    # ==================== SLIDE 3: LITERATURE REVIEW (FIXED!) ====================
    s3 = prs.slides.add_slide(blank)
    set_bg(s3, BG_LIGHT)
    add_header(s3, "Literature Review", "Key studies from our reference papers and the research gap addressed", 3)

    add_table(
        s3, 0.6, 1.5, 12.1, 3.1,
        headers=["Reference Study", "Dataset & Task", "Methodology", "Reported Result", "Key Takeaway"],
        rows=[
            ["Imran et al. (2022)\nIEEE Access", "ISIC 2019\n(Binary: 2 classes)", "Majority-voting ensemble:\nVGG16 + CapsNet + ResNet", "93.5% Acc\n87.3% Sens", "Ensembling diverse CNNs outperforms every individual learner."],
            ["Gouda et al. (2022)\nHealthcare (MDPI)", "ISIC 2018\n(Binary: 2 classes)", "ESRGAN enhancement +\nCNN, ResNet50, InceptionV3", "85.8% Acc\n(InceptionV3)", "Preprocessing & fine-tuning boost accuracy on limited data."],
            ["Hameed et al. (2024)\nCMES (Review)", "ISIC 2016–2020\n(PRISMA Review)", "Systematic review of 197 SCC\n& 184 SCS CNN / ViT papers", "76.1%–99.3%\n(across tasks)", "Flags class imbalance, overfitting & naive augmentation bias."],
            ["Our Project (2026)\nGroup B55", "ISIC 9-Class +\nBinary Triage", "Hierarchical Fusion (Binary +\nEffNet-B0 + MobileNetV2 + 4x TTA)", "81.55% Triage\n89.2% Sens | 0.939 AUC", "Combines triage + 9-class + Grad-CAM in a live web app."],
        ],
        col_widths=[2.1, 1.9, 3.0, 2.0, 3.1],
        highlight_last=True,
    )

    add_card(
        s3, 0.6, 4.9, 12.1, 1.95,
        title="Research Gap Addressed by Our Work",
        bullets=[
            "Most prior studies focus strictly on binary (benign vs. malignant) classification without subtype granularity or visual explainability.",
            "Heavy models are hard to deploy locally; our EfficientNet-B0 + MobileNetV2 ensemble runs fast even on standard CPU hardware.",
            "We bridge clinical safety (high-sensitivity binary triage) with 9-class diagnosis and live Grad-CAM transparency.",
        ],
        accent=BLUE,
    )

    # ==================== SLIDE 4: DATASET ====================
    s4 = prs.slides.add_slide(blank)
    set_bg(s4, BG_LIGHT)
    add_header(s4, "Dataset: ISIC 9-Class Dermoscopic Collection", "2,357 dermoscopic images across 9 lesion classes with stratified splitting", 4)

    add_table(
        s4, 0.6, 1.45, 7.6, 5.3,
        headers=["Skin Lesion Class", "Clinical Group", "Train", "Test"],
        rows=[
            ["Pigmented Benign Keratosis", "Benign", "462", "16"],
            ["Melanoma", "Malignant", "438", "16"],
            ["Basal Cell Carcinoma (BCC)", "Malignant", "376", "16"],
            ["Nevus", "Benign", "357", "16"],
            ["Squamous Cell Carcinoma (SCC)", "Malignant", "181", "16"],
            ["Vascular Lesion", "Benign", "139", "3"],
            ["Actinic Keratosis", "Pre-Malignant", "114", "16"],
            ["Dermatofibroma", "Benign", "95", "16"],
            ["Seborrheic Keratosis", "Benign", "77", "3"],
            ["Total (9 Classes)", "4 Mal / 5 Ben", "2,239", "118"],
        ],
        col_widths=[3.2, 2.0, 1.2, 1.2],
        highlight_last=True,
    )

    add_card(
        s4, 8.5, 1.45, 4.2, 2.55,
        title="Data Splitting Strategy",
        bullets=[
            "Stratified 85/15 split on Train → 1,903 train + 336 validation images.",
            "Held-out Test folder (118 images) kept strictly untouched during training.",
            "Pre-resized all 2,357 images to 256×256 for fast I/O.",
        ],
        accent=TEAL,
    )
    add_card(
        s4, 8.5, 4.2, 4.2, 2.55,
        title="Handling Class Imbalance",
        bullets=[
            "Severe 6:1 imbalance (462 Pigmented Keratosis vs. 77 Seborrheic Keratosis).",
            "Applied smoothed inverse-frequency class weights + label smoothing (0.05).",
            "Evaluated using Macro F1 & Macro ROC-AUC (not accuracy alone).",
        ],
        accent=CORAL,
    )

    # ==================== SLIDE 5: PREPROCESSING & AUGMENTATION ====================
    s5 = prs.slides.add_slide(blank)
    set_bg(s5, BG_LIGHT)
    add_header(s5, "Preprocessing, Augmentation & Training Strategy", "Overcoming the 35.6% frozen-model baseline with modern deep learning practices", 5)

    add_card(
        s5, 0.6, 1.5, 5.9, 5.25,
        title="Image Preprocessing & Augmentation Pipeline",
        bullets=[
            "Pre-Resize (256×256) + RandomResizedCrop (224×224, scale 0.75–1.0) for scale invariance.",
            "Random Horizontal & Vertical Flips (p=0.5) — dermoscopic lesions are orientation-invariant.",
            "Random Rotation (±30°) + ColorJitter (brightness, contrast, saturation) to handle varied dermatoscopes.",
            "Random Erasing (p=0.15) to simulate hair/ruler occlusions and reduce overfitting.",
            "ImageNet Standardization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) — critical fix over naive [0,1] scaling.",
        ],
        accent=TEAL,
    )
    add_card(
        s5, 6.8, 1.5, 5.9, 5.25,
        title="Two-Stage Transfer Learning & Optimization",
        bullets=[
            "Stage 1 (Warmup): Freeze pretrained ImageNet backbone; train Dropout(0.3) + Linear head with AdamW (lr=1e-3).",
            "Stage 2 (Fine-Tuning): Unfreeze full network with differential learning rates (backbone 1.5e-4–2e-4, head 5e-4–1e-3).",
            "CosineAnnealingLR scheduler + gradient clipping (max_norm=2.0) for smooth convergence.",
            "Weighted Cross-Entropy Loss + Label Smoothing (0.05) to calibrate probabilities.",
            "4x Test-Time Augmentation (TTA): averages predictions across original, H-flip, V-flip, and HV-flip views.",
        ],
        accent=BLUE,
    )

    # ==================== SLIDE 6: UNIQUE HIERARCHICAL ARCHITECTURE ====================
    s6 = prs.slides.add_slide(blank)
    set_bg(s6, BG_LIGHT)
    add_header(s6, "Proposed Architecture: 3-Stage Hierarchical System", "Combining dedicated binary triage, multi-model 9-class ensembling, and Grad-CAM", 6)

    # Flow boxes
    stages = [
        ("INPUT", "Dermoscopic\nImage (224×224)", "ImageNet Norm\n+ 4x Flip TTA", NAVY),
        ("STAGE 1", "Binary Triage\n(Mal vs. Benign)", "Dedicated EffNet-B0\n+ 50/50 Fusion\n81.55% Acc | 89.2% Sens", CORAL),
        ("STAGE 2", "9-Class Subtype\nDeep Ensemble", "EfficientNet-B0 +\nMobileNetV2\n71.73% Acc | 0.939 AUC", TEAL),
        ("STAGE 3", "Explainable AI\n(Grad-CAM)", "Final Conv Layer\nAttention Heatmap\n+ Live Web UI", BLUE),
    ]
    x = 0.6
    for idx, (tag, heading, desc, col) in enumerate(stages):
        c = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.65), Inches(2.75), Inches(2.55))
        c.adjustments[0] = 0.06
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = col
        c.line.width = Pt(2)

        hdr = s6.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.65), Inches(2.75), Inches(0.42))
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = col
        hdr.line.fill.background()
        hp = hdr.text_frame.paragraphs[0]
        hp.text = tag
        hp.alignment = PP_ALIGN.CENTER
        hp.font.name = FONT_MAIN
        hp.font.size = Pt(11)
        hp.font.bold = True
        hp.font.color.rgb = WHITE

        tb = s6.shapes.add_textbox(Inches(x + 0.12), Inches(2.15), Inches(2.5), Inches(1.95))
        tf = tb.text_frame
        tf.word_wrap = True
        p1 = tf.paragraphs[0]
        p1.text = heading
        p1.alignment = PP_ALIGN.CENTER
        p1.font.name = FONT_MAIN
        p1.font.size = Pt(14.5)
        p1.font.bold = True
        p1.font.color.rgb = NAVY

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.alignment = PP_ALIGN.CENTER
        p2.font.name = FONT_MAIN
        p2.font.size = Pt(11.5)
        p2.font.color.rgb = SLATE
        p2.space_before = Pt(8)

        if idx < 3:
            arr = s6.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.83), Inches(2.72), Inches(0.28), Inches(0.32))
            arr.fill.solid()
            arr.fill.fore_color.rgb = MUTED
            arr.line.fill.background()
        x += 3.12

    add_card(
        s6, 0.6, 4.55, 12.1, 2.25,
        title="Why This Hierarchical Fusion Design Is Unique",
        bullets=[
            "Experimental proof: Fusing Stage-1 Binary EfficientNet-B0 (77.98%) with the 9-Class Ensemble (80.36%) achieves 81.55% accuracy & 89.2% sensitivity — outperforming both individual approaches.",
            "Clinically aligned workflow: Doctors first triage High/Moderate/Low risk (minimizing missed cancers), then inspect Top-3 subtype probabilities and Grad-CAM evidence.",
        ],
        accent=TEAL,
        fill_color=TEAL_LIGHT,
    )

    # ==================== SLIDE 7: QUANTITATIVE RESULTS ====================
    s7 = prs.slides.add_slide(blank)
    set_bg(s7, BG_LIGHT)
    add_header(s7, "Experimental Results & Progression", "More than 2× improvement over the original baseline + 81.55% cancer triage accuracy", 7)

    add_kpi(s7, 0.6, 1.45, 2.85, 1.35, "81.55%", "Stage-1 Triage Acc", "Hierarchical Fusion (4x TTA)", CORAL)
    add_kpi(s7, 3.68, 1.45, 2.85, 1.35, "89.2%", "Cancer Sensitivity", "Catches ~9 of 10 cancers", TEAL)
    add_kpi(s7, 6.76, 1.45, 2.85, 1.35, "71.73%", "9-Class Ensemble Acc", "Up from 35.6% baseline (+36.1%)", BLUE)
    add_kpi(s7, 9.84, 1.45, 2.85, 1.35, "0.9390", "9-Class Macro AUC", "0.9272 on held-out Test", NAVY)

    add_table(
        s7, 0.6, 3.05, 12.1, 3.7,
        headers=["Model / Experiment", "Task", "Val Accuracy", "Sensitivity / Macro F1", "ROC-AUC"],
        rows=[
            ["Original Slides Baseline (Frozen MobileNetV2, 10 ep)", "9-Class", "35.60%", "—", "—"],
            ["MobileNetV2 (2-Stage Top-Block Fine-Tune)", "9-Class", "66.07%", "Macro F1: 0.5754", "0.9275"],
            ["EfficientNet-B0 (Full Fine-Tune + 4x TTA)", "9-Class", "69.64%", "—", "—"],
            ["Ensemble: EfficientNet-B0 + MobileNetV2 (4x TTA)", "9-Class", "71.73%", "Macro F1: 0.6125", "0.9390"],
            ["Standalone Binary EfficientNet-B0 (4x TTA)", "Binary Triage", "77.98%", "Sens: 86.1% | Spec: 70.0%", "0.8445"],
            ["Hierarchical Fusion (Binary + 9-Class Ensemble, 4x TTA)", "Binary Triage", "81.55%", "Sens: 89.2% | Spec: 74.1%", "0.8576"],
        ],
        col_widths=[4.9, 1.6, 1.6, 2.5, 1.5],
        highlight_last=True,
    )

    # ==================== SLIDE 8: STAGE-1 TRIAGE & ROC PLOTS ====================
    s8 = prs.slides.add_slide(blank)
    set_bg(s8, BG_LIGHT)
    add_header(s8, "Stage-1 Binary Triage & Learning Curves", "Confusion matrix, ROC curve (AUC = 0.8576), and 2-stage training progression", 8)

    add_picture_fitted(s8, OUTPUT_DIR / "binary_triage_evaluation.png", 0.5, 1.4, 12.3, 2.8)
    add_picture_fitted(s8, OUTPUT_DIR / "training_history.png", 0.5, 4.3, 12.3, 2.65)

    # ==================== SLIDE 9: 9-CLASS CONFUSION MATRIX & ROC ====================
    s9 = prs.slides.add_slide(blank)
    set_bg(s9, BG_LIGHT)
    add_header(s9, "Stage-2 9-Class Evaluation: Confusion Matrix & ROC", "Strong per-class separation (Macro ROC-AUC = 0.9390 across all 9 classes)", 9)

    add_picture_fitted(s9, OUTPUT_DIR / "confusion_matrix_val.png", 0.5, 1.4, 6.2, 5.45)
    add_picture_fitted(s9, OUTPUT_DIR / "roc_curves_val.png", 6.8, 1.4, 6.0, 5.45)

    # ==================== SLIDE 10: EXPLAINABLE AI (GRAD-CAM) ====================
    s10 = prs.slides.add_slide(blank)
    set_bg(s10, BG_LIGHT)
    add_header(s10, "Stage-3 Explainable AI: Grad-CAM Heatmaps", "Visualizing exact lesion regions driving model predictions on held-out test images", 10)

    add_picture_fitted(s10, OUTPUT_DIR / "gradcam_samples.png", 0.5, 1.35, 12.3, 4.45)

    add_card(
        s10, 0.6, 5.9, 12.1, 1.05,
        bullets=[
            "Grad-CAM hooks into the final convolutional layer, highlighting pigmented lesion cores & irregular borders in red/yellow rather than background skin.",
            "Builds clinical trust and directly answers the 'black-box AI' concern in medical diagnostics.",
        ],
        accent=TEAL,
        body_size=11.5,
    )

    # ==================== SLIDE 11: LIVE WEB APP & TECH STACK ====================
    s11 = prs.slides.add_slide(blank)
    set_bg(s11, BG_LIGHT)
    add_header(s11, "Live Clinical Web Application & Tech Stack", "Interactive Streamlit dashboard deployed locally + full reproducible engineering stack", 11)

    add_card(
        s11, 0.6, 1.45, 5.9, 2.65,
        title="Deep Learning & XAI Stack",
        bullets=[
            "Python 3.12, PyTorch 2.15, Torchvision 0.29",
            "Backbones: EfficientNet-B0, MobileNetV2 (+ DenseNet121 / ResNet50)",
            "Custom Grad-CAM hooks + 4x Flip TTA + Hierarchical Fusion",
            "Scikit-learn, NumPy, Pillow, Matplotlib, Seaborn",
        ],
        accent=TEAL,
    )
    add_card(
        s11, 0.6, 4.3, 5.9, 2.55,
        title="Web App & MLOps Stack",
        bullets=[
            "Frontend & UI: Streamlit 1.67 (drag-and-drop + 1-click samples)",
            "ASGI Server: Uvicorn + Starlette + WebSockets",
            "Version Control: Git & GitHub (debayudh482/skin-cancer-detection)",
            "GPU Scaling: 1-click Google Colab T4 Notebook included",
        ],
        accent=BLUE,
    )

    add_card(
        s11, 6.8, 1.45, 5.9, 2.55,
        title="Live Demo Output (Held-Out BCC Test Lesion)",
        bullets=[
            "Stage 1 Triage: HIGH RISK — 87.11% Malignant Probability",
            "Stage 2 Top-3: Basal Cell Carcinoma (80.31%), Dermatofibroma (8.74%), SCC (4.35%)",
            "Stage 3: Live Grad-CAM overlay + full 9-class probabilities",
        ],
        accent=CORAL,
        fill_color=TEAL_LIGHT,
    )
    add_picture_fitted(s11, OUTPUT_DIR / "hierarchical_sample_gradcam.png", 8.1, 4.15, 3.3, 2.75)

    # ==================== SLIDE 12: ADVANTAGES, LIMITATIONS & FUTURE WORK ====================
    s12 = prs.slides.add_slide(blank)
    set_bg(s12, BG_LIGHT)
    add_header(s12, "Advantages, Limitations & Future Work", "Summary of contributions and clear roadmap for clinical scaling", 12)

    add_card(
        s12, 0.6, 1.5, 3.85, 5.2,
        title="Key Advantages",
        bullets=[
            "81.55% triage accuracy with high 89.2% cancer sensitivity.",
            "More than 2× jump on 9-class accuracy (35.6% → 71.73%, 0.939 AUC).",
            "Explainable via live Grad-CAM visual heatmaps.",
            "Lightweight enough to run real-time inference on standard CPUs.",
            "Complete working Streamlit web app + GitHub repo.",
        ],
        accent=TEAL,
    )
    add_card(
        s12, 4.75, 1.5, 3.85, 5.2,
        title="Current Limitations",
        bullets=[
            "Small training set (2,239 images) limits fine-grained 9-class ceiling.",
            "Severe minority-class scarcity (only 77 Seborrheic Keratosis samples).",
            "Visual overlap between Seborrheic & Pigmented Benign Keratosis.",
            "Decision-support prototype — not a substitute for clinical biopsy.",
        ],
        accent=AMBER,
    )
    add_card(
        s12, 8.9, 1.5, 3.85, 5.2,
        title="Future Work",
        bullets=[
            "Train 3-model ensemble (+ DenseNet121) on HAM10000 / ISIC 2019 (10K–25K images) via our Colab GPU notebook.",
            "Fuse patient metadata (age, sex, anatomical site) with image features.",
            "Add Vision Transformer (ViT / Swin) backbone to the ensemble.",
            "Deploy as FastAPI + React/Flutter mobile teledermatology app.",
        ],
        accent=BLUE,
    )

    prs.save(str(PPTX_PATH))
    print(f"Presentation saved successfully -> {PPTX_PATH} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_presentation()
