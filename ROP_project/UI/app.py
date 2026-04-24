import io
import os
import numpy as np
import cv2
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datetime import datetime
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as T
from reportlab.platypus import SimpleDocTemplate, Image as RLImage, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import cm
 
from scripts.stage_classification import GradCAM, ROPNet
from scripts.validation import retina_score
from scripts.train_segmentation import VesselUNet, preprocess_retina
from style import apply_style, render_header, render_footer, render_step_bar, render_referral_badge, render_confidence_bar, render_prob_breakdown
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Severe"]
K           = NUM_CLASSES - 1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE   = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(__file__)
 
    # ── Stage model ──
    stage_model = ROPNet().to(DEVICE)
    stage_path  = os.path.join(BASE_DIR, "..", "results", "final_stage_model.pt")
    checkpoint  = torch.load(stage_path, map_location=DEVICE, weights_only=False)
 
    state_dict  = checkpoint.get("model", checkpoint)
    temperature = checkpoint.get("temperature", 1.0)
 
    model_dict    = stage_model.state_dict()
    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    stage_model.load_state_dict(filtered_dict, strict=False)
    stage_model.eval()
 
    # ── U-Net ──
    unet      = VesselUNet().to(DEVICE)
    unet_path = os.path.join(BASE_DIR, "..", "results", "vessel_unet_512.pt")
    unet.load_state_dict(torch.load(unet_path, map_location=DEVICE, weights_only=False))
    unet.eval()
 
    cam_extractor = GradCAM(stage_model)
    return stage_model, unet, cam_extractor, temperature
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def ben_graham(x: torch.Tensor) -> torch.Tensor:
    """Fix: kernel_size must be a list, not a scalar."""
    blur = gaussian_blur(x, kernel_size=31)
    return torch.clamp(4 * x - 4 * blur + 0.5, 0, 1)
 
 
def preprocess_image(img_rgb: np.ndarray) -> torch.Tensor:
    img   = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_t = ben_graham(img_t)
    img_t = NORMALIZE(img_t)
    return img_t.to(DEVICE)
 
 
def get_vessel_mask(unet: nn.Module, img_rgb: np.ndarray) -> torch.Tensor:
    img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_proc = preprocess_retina(img_bgr, 512)
    x        = torch.tensor(img_proc).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
 
    with torch.no_grad():
        pred = torch.sigmoid(unet(x))[0, 0]
 
    # Soft mask — no hard threshold, preserves gradient signal for Grad-CAM
    mask = cv2.resize(pred.cpu().numpy(), (IMG_SIZE, IMG_SIZE))
    mask = (mask - mask.min()) / (mask.max() + 1e-6)
    return torch.tensor(mask).to(DEVICE)
 
 
def ordinal_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=1)[0]      # enforce monotonicity
 
    B, K   = cumulative.shape
    probs  = torch.zeros((B, K + 1), device=logits.device)
    probs[:, 0] = 1 - cumulative[:, 0]
    for k in range(1, K):
        probs[:, k] = cumulative[:, k - 1] - cumulative[:, k]
    probs[:, -1] = cumulative[:, -1]
    return torch.clamp(probs, 1e-6, 1.0)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_circular_mask(shape):
    h, w   = shape
    y, x   = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r      = min(cy, cx) * 0.95
    return ((x - cx) ** 2 + (y - cy) ** 2 <= r ** 2).astype(np.float32)
 
 
def get_vessel_soft_mask(mask_tensor: torch.Tensor, target_size):
    """Build a smooth soft mask from the vessel mask tensor for CAM refinement."""
    m = (mask_tensor.cpu().numpy() > 0).astype(np.uint8)
    m = cv2.dilate(m, np.ones((15, 15), np.uint8), iterations=1)
    m = cv2.GaussianBlur(m.astype(np.float32), (21, 21), 0)
    m = cv2.resize(m, target_size, interpolation=cv2.INTER_LINEAR)
    m = m / (m.max() + 1e-6)
    return np.nan_to_num(np.clip(m, 0.0, 1.0))
 
 
def overlay_cam(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.35) -> Image.Image:
    img_np  = np.array(img_pil)
    h, w    = img_np.shape[:2]
    cam_res = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat    = cv2.applyColorMap((cam_res * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat    = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blend   = cv2.addWeighted(img_np, 1.0, heat, alpha, 0)
    return Image.fromarray(blend)
 
 
def add_colorbar(img_pil: Image.Image, cam: np.ndarray, bar_w: int = 28) -> Image.Image:
    img_np    = np.array(img_pil)
    h         = img_np.shape[0]
    cbar      = np.linspace(0, 1, h).reshape(h, 1)
    cbar      = np.repeat(cbar, bar_w, axis=1)
    cbar_rgb  = cv2.applyColorMap((cbar * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cbar_rgb  = cv2.cvtColor(cv2.resize(cbar_rgb, (bar_w, h)), cv2.COLOR_BGR2RGB)
    combined  = np.concatenate([img_np, cbar_rgb], axis=1)
    return Image.fromarray(combined)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def predict(stage_model, unet, img_pil, ga, bw, cam_extractor, temperature, show_cam=True):
    img_rgb = np.array(img_pil)
 
    # Step 1 — vessel mask
    mask = get_vessel_mask(unet, img_rgb)
 
    # Step 2 — image preprocessing
    img_t = preprocess_image(img_rgb)
    x     = torch.cat([img_t, mask.unsqueeze(0)], dim=0).unsqueeze(0).to(DEVICE)
    if show_cam:
        x.requires_grad_(True)
 
    # Step 3 — normalise clinical inputs with physiological bounds
    ga_norm  = (ga - 22) / (40 - 22)
    bw_norm  = (bw - 400) / (2500 - 400)
    clinical = torch.tensor([[ga_norm, bw_norm]], device=DEVICE).float()
 
    # Step 4 — forward pass
    stage_model.eval()
    with torch.set_grad_enabled(True):
        logits = stage_model(x, clinical)
        logits = logits / temperature          # temperature scaling
        probs  = ordinal_probs_from_logits(logits)
        pred   = int(probs.argmax(dim=1).item())
        probs_np = probs.detach().cpu().numpy()[0]
 
    # Step 5 — Grad-CAM
    cam_map = None
    if show_cam and cam_extractor:
        stage_model.zero_grad()
        cam_map = cam_extractor.generate(x, clinical, pred)[0]
 
        cam_map = np.nan_to_num(cam_map)
        cam_map = np.clip(cam_map, 0, None)
        cam_map -= cam_map.min()
        cam_map /= (cam_map.max() + 1e-6)
 
        tw, th   = img_pil.size          # PIL: (width, height)
        h_mask   = cv2.resize(get_circular_mask(cam_map.shape), (tw, th))
        s_mask   = get_vessel_soft_mask(mask, (tw, th))
        cam_res  = cv2.resize(cam_map, (tw, th), interpolation=cv2.INTER_LINEAR)
 
        final_mask          = h_mask * (s_mask ** 1.5)
        final_mask[final_mask < 0.2] = 0
        cam_map             = cam_res * final_mask
 
        # Smooth
        cam_map = cv2.medianBlur((cam_map * 255).astype(np.uint8), 9) / 255.0
        cam_map = cv2.GaussianBlur(cam_map, (15, 15), 0)
 
        # Suppress Grad-CAM intensity for Normal — less visual noise
        if pred == 0:
            cam_map          *= 0.3
            cam_map[cam_map < 0.4] = 0
 
        cam_map = np.clip(cam_map, 0, 1)
 
    return pred, probs_np, mask.cpu().numpy(), cam_map
 
 
# ─────────────────────────────────────────────────────────────────────────────
# REFERRAL — FIX: receives label_name (str), not pred (int)
# ─────────────────────────────────────────────────────────────────────────────
def referral(label_name: str) -> str:
    if label_name == "Severe":
        return "URGENT REFERRAL"
    elif label_name == "Mild":
        return "MONITOR CLOSELY"
    return "ROUTINE FOLLOW-UP"
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "Severe": "#DC2626",
    "Mild":   "#F59E0B",
    "Normal": "#16A34A",
}
 
def confidence_text(confidence: float, label_name: str) -> str:
    if confidence < 0.70:
        return "Low confidence — specialist review required"
    if confidence < 0.85:
        return "Moderate confidence"
    return "High confidence"
 
 
def apply_safety_override(ref: str, confidence: float) -> str:
    """
    Safety override: escalate any low-confidence prediction.
    Applied regardless of predicted class to catch all uncertain cases.
    """
    if confidence < 0.70:
        if ref == "ROUTINE FOLLOW-UP":
            return "REQUIRES EXPERT REVIEW"
        return ref + " (Low Confidence — Verify)"
    return ref
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PDF EXPORT — FIX: actually builds the document
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(img_pil, cam_pil, label_name, ref_text, confidence, probs, ga, bw) -> bytes:
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
 
    title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                 fontSize=16, textColor=rl_colors.HexColor("#0D47A1"),
                                 spaceAfter=6)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                 fontSize=10, textColor=rl_colors.HexColor("#64748B"),
                                 spaceAfter=16)
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                 fontSize=11, leading=16, spaceAfter=8)
 
    story.append(Paragraph("ROP Screening Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}", sub_style))
 
    # Result table
    ref_final   = apply_safety_override(ref_text, confidence)
    conf_str    = f"{confidence*100:.1f}%"
    result_data = [
        ["Assessment",          label_name],
        ["Clinical Action",     ref_final],
        ["Confidence",          conf_str + f"  ({confidence_text(confidence, label_name)})"],
        ["Gestational Age",     f"{ga} weeks"],
        ["Birth Weight",        f"{bw} g"],
        ["Normal Probability",  f"{probs[0]*100:.1f}%"],
        ["Mild Probability",    f"{probs[1]*100:.1f}%"],
        ["Severe Probability",  f"{probs[2]*100:.1f}%"],
    ]
    tbl = Table(result_data, colWidths=[6*cm, 10*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), rl_colors.HexColor("#EFF6FF")),
        ("TEXTCOLOR",   (0, 0), (0, -1), rl_colors.HexColor("#1565C0")),
        ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
            [rl_colors.white, rl_colors.HexColor("#F8FAFD")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#E1E9F5")),
        ("TOPPADDING",  (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING",(0, 0),(-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))
 
    # Images
    def pil_to_rl(pil_img, max_w=8*cm):
        buf2 = io.BytesIO()
        pil_img.save(buf2, format="PNG")
        buf2.seek(0)
        ir = RLImage(buf2)
        ratio = max_w / ir.imageWidth
        ir.drawWidth  = max_w
        ir.drawHeight = ir.imageHeight * ratio
        return ir
 
    img_data = [[pil_to_rl(img_pil), pil_to_rl(cam_pil)]]
    img_tbl  = Table(img_data, colWidths=[8.5*cm, 8.5*cm])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",  (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 0.4*cm))
 
    caps_data = [["Original Image", "Grad-CAM Explanation"]]
    caps_tbl  = Table(caps_data, colWidths=[8.5*cm, 8.5*cm])
    caps_tbl.setStyle(TableStyle([
        ("ALIGN",    (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Oblique"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",(0, 0), (-1, -1), rl_colors.HexColor("#64748B")),
    ]))
    story.append(caps_tbl)
    story.append(Spacer(1, 0.5*cm))
 
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by a research AI system and is intended "
        "for decision support only. It does not constitute a clinical diagnosis. All findings "
        "must be reviewed and confirmed by a qualified ophthalmologist.",
        ParagraphStyle("disc", parent=styles["Normal"], fontSize=9,
                       textColor=rl_colors.HexColor("#64748B"),
                       borderColor=rl_colors.HexColor("#CBD5E1"),
                       borderWidth=0.5, borderPadding=8,
                       backColor=rl_colors.HexColor("#F8FAFD"))
    ))
 
    doc.build(story)
    buf.seek(0)
    return buf.read()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
def page_upload(stage_model, unet, cam_extractor, temperature):
    render_step_bar(1)
 
    col_img, col_clin = st.columns([1, 1], gap="large")
 
    # ── Image upload card ──
    with col_img:
        st.markdown('<div class="card"><div class="card-title">📷 Retinal Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop a fundus image or click to browse",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state.uploaded_file = uploaded
            preview = Image.open(uploaded)
            st.image(preview, width="stretch",
                     caption="Uploaded image preview")
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Clinical data card ──
    with col_clin:
        st.markdown('<div class="card"><div class="card-title">🏥 Clinical Parameters</div>',
                    unsafe_allow_html=True)
        ga = st.number_input(
            "Gestational Age (weeks)",
            min_value=22, max_value=40, value=None, step=1,
            placeholder="e.g. 28",
            help="Gestational age at birth in weeks (22–40)",
        )
        bw = st.number_input(
            "Birth Weight (g)",
            min_value=400, max_value=2500, value=None, step=10,
            placeholder="e.g. 1200",
            help="Birth weight in grams (400–2500)",
        )
 
        st.markdown("<br>", unsafe_allow_html=True)
        show_cam = st.checkbox("Generate Grad-CAM explanation", value=True,
                               help="Produces a heatmap showing which retinal regions influenced the prediction. Adds ~2–3 seconds to analysis.")
 
        st.markdown('<hr style="margin:18px 0">', unsafe_allow_html=True)
 
        run = st.button("▶  Run Analysis", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
 
    # ── Validation & run ──
    if run:
        if st.session_state.get("uploaded_file") is None:
            st.warning("⚠️  Please upload a retinal image before running.")
            return
        if ga is None or bw is None:
            st.warning("⚠️  Please enter both Gestational Age and Birth Weight.")
            return
 
        img = Image.open(st.session_state.uploaded_file).convert("RGB")
 
        # Quality validation
        is_valid, msg = retina_score(img, unet, DEVICE)
        if not is_valid:
            st.error(f"❌  Image quality check failed: {msg}")
            return
 
        with st.spinner("Step 1 / 2 — Generating vessel mask…"):
            # Pre-run mask generation separately to allow step feedback
            pass  # mask is generated inside predict()
 
        with st.spinner("Step 2 / 2 — Running classification and Grad-CAM…"):
            pred, probs, mask, cam_map = predict(
                stage_model, unet, img, ga, bw,
                cam_extractor, temperature, show_cam=show_cam
            )
 
        label_name = CLASS_NAMES[pred]
        ref_text   = referral(label_name)           # FIX: pass string not int
        confidence = float(probs[pred])
        ref_final  = apply_safety_override(ref_text, confidence)
 
        st.session_state.result = {
            "img":        img,
            "pred":       pred,
            "label_name": label_name,
            "probs":      probs.tolist(),
            "mask":       mask.tolist(),
            "cam":        cam_map.tolist() if cam_map is not None else None,
            "ref":        ref_final,
            "confidence": confidence,
            "ga":         ga,
            "bw":         bw,
        }
        st.session_state.page = "result"
        st.rerun()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE: RESULTS
# ─────────────────────────────────────────────────────────────────────────────
def page_result():
    data       = st.session_state.result
    label_name = data["label_name"]
    confidence = data["confidence"]
    color      = SEVERITY_COLORS.get(label_name, "#16A34A")
    probs      = data["probs"]
    cam_arr    = np.array(data["cam"], dtype=np.float32) if data.get("cam") else None
 
    render_step_bar(3)
 
    # ── Top row: back + export ──
    nav_l, nav_r = st.columns([5, 2])
    with nav_l:
        if st.button("← Back to Upload", use_container_width=False):
            st.session_state.page = "upload"
            st.rerun()
    with nav_r:
        if cam_arr is not None:
            cam_overlay = overlay_cam(data["img"], cam_arr)
            export_img  = add_colorbar(cam_overlay, cam_arr)
 
            # Build PNG
            png_buf = io.BytesIO()
            export_img.save(png_buf, format="PNG")
 
            # Build PDF
            pdf_bytes = build_pdf(
                data["img"], cam_overlay,
                label_name, data["ref"], confidence, probs,
                data["ga"], data["bw"]
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # ── Hamburger popover ──
            with st.popover("☰  Export", use_container_width=False):
                st.markdown(
                    '<div style="font-size:12px;font-weight:600;color:#94A3B8;'
                    'text-transform:uppercase;letter-spacing:0.6px;'
                    'margin-bottom:10px;">Export Results</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label=" Download PNG",
                    data=png_buf.getvalue(),
                    file_name=f"ROP_Analysis_{timestamp}.png",
                    mime="image/png",
                    use_container_width=True,
                    help="Grad-CAM overlay with colour bar",
                )
                st.download_button(
                    label="📄  Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"ROP_Report_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="Full clinical report with images and results table",
                )
 
    st.markdown("<br>", unsafe_allow_html=True)

 
    # ── Two-column results layout ──
    left_col, right_col = st.columns([2, 3], gap="large")
 
    with left_col:
        # Result card
        st.markdown(f"""
        <div class="result-card" style="border-top:4px solid {color};">
            <div style="font-size:12px;font-weight:600;color:#94A3B8;
                        text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">
                Assessment
            </div>
            <div class="result-label" style="color:{color};">{label_name}</div>
        """, unsafe_allow_html=True)
 
        render_referral_badge(data["ref"], label_name)
        render_confidence_bar(confidence, color)
 
        st.markdown(f"""
            <div style="font-size:13px;color:#64748B;margin-bottom:14px;">
                {confidence_text(confidence, label_name)}
            </div>
            <hr>
        """, unsafe_allow_html=True)
 
        render_prob_breakdown(probs, class_names=CLASS_NAMES)
 
        # Clinical info grid
        st.markdown(f"""
        <div class="info-grid">
            <div class="info-item">
                <div class="label">Gest. Age</div>
                <div class="value">{data['ga']} wk</div>
            </div>
            <div class="info-item">
                <div class="label">Birth Weight</div>
                <div class="value">{data['bw']} g</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        <div class="disclaimer">
            ⚠ This prediction is AI-assisted decision support only.
            Final diagnosis must be confirmed by a qualified ophthalmologist
            following the ICROP guidelines.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    with right_col:
        st.markdown('<div class="card"><div class="card-title">🔬 Visual Analysis</div>',
                    unsafe_allow_html=True)
 
        view_mode = st.radio(
            "View",
            ["Grad-CAM Overlay", "Original Image"],
            horizontal=True,
            label_visibility="collapsed",
        )
 
        alpha = 0.35
        if view_mode == "Grad-CAM Overlay" and cam_arr is not None:
            alpha = st.slider(
                "Heatmap opacity",
                0.1, 0.8, 0.35, 0.05,
                help="Lower = more original image visible. Higher = stronger heatmap.",
            )
 
        st.divider()
 
        img_col, orig_col = st.columns(2)
 
        show_cam_left = (view_mode == "Grad-CAM Overlay") and cam_arr is not None
        main_img      = overlay_cam(data["img"], cam_arr, alpha) if show_cam_left else data["img"]
        side_img      = data["img"] if show_cam_left else (
            overlay_cam(data["img"], cam_arr, 0.35) if cam_arr is not None else None
        )
 
        with img_col:
            st.markdown(
                f'<div class="img-label">{"Grad-CAM" if show_cam_left else "Original"}</div>',
                unsafe_allow_html=True,
            )
            st.image(main_img, width="stretch")
 
        with orig_col:
            if side_img:
                st.markdown(
                    f'<div class="img-label">{"Original" if show_cam_left else "Grad-CAM"}</div>',
                    unsafe_allow_html=True,
                )
                st.image(side_img, width="stretch")
 
        st.markdown("</div>", unsafe_allow_html=True)
 
    # ── New analysis ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("＋  New Analysis", width="stretch"):
        for key in ["uploaded_file", "result"]:
            st.session_state.pop(key, None)
        st.session_state.page = "upload"
        st.rerun()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="ROP Screening System",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_style()
    render_header()
 
    # Session state init
    if "page" not in st.session_state:
        st.session_state.page = "upload"
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "result" not in st.session_state:
        st.session_state.result = None
 
    stage_model, unet, cam_extractor, temperature = load_models()
 
    if st.session_state.page == "upload":
        page_upload(stage_model, unet, cam_extractor, temperature)
    elif st.session_state.page == "result":
        page_result()
 
    render_footer()
 
 
if __name__ == "__main__":
    main()
