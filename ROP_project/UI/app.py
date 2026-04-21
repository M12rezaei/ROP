# app.py
import io
import sys
import os
import json
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
#from torchcam.methods import GradCAM
from scripts.stage_classification import GradCAM
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import gaussian_blur
import torchvision.transforms as T

from scripts.stage_classification import ROPNet
from scripts.validation import retina_score
from scripts.train_segmentation import VesselUNet, preprocess_retina

from style import apply_style, footer, header, subheader

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Severe"]
K = NUM_CLASSES - 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NORMALIZE = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# =========================
# MODEL WRAPPER (Grad-CAM)
# =========================
#def get_last_conv(model):
#    conv_layers = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
#    return conv_layers[-1]

def get_last_conv(model):
    return model.conv_head1
def get_retina_mask_for_cam(mask, img_size):
    # mask: vessel mask (224x224)
    # img_size: original image size (width, height)

    retina_mask = (mask.cpu().numpy() > 0).astype(np.uint8)

    # Dilate slightly to cover entire retina area, not just vessels
    retina_mask = cv2.dilate(retina_mask, np.ones((15,15), np.uint8), iterations=1)

    # Smooth edges
    retina_mask = cv2.GaussianBlur(retina_mask.astype(np.float32), (21,21), 0)

    # Resize to original image
    retina_mask = cv2.resize(retina_mask, img_size, interpolation=cv2.INTER_LINEAR)

    # Normalize
    retina_mask = retina_mask / retina_mask.max()

    retina_mask = retina_mask / (retina_mask.max() + 1e-6)
    retina_mask = np.nan_to_num(retina_mask)
    retina_mask = np.clip(retina_mask, 0.0, 1.0)

    return retina_mask
def overlay_light_cam(img, cam_map, alpha=0.3, colormap=cv2.COLORMAP_JET):
    """
    img: PIL Image (RGB)
    cam_map: 2D numpy array, values in [0,1]
    alpha: how strong the heatmap is over the image
    colormap: OpenCV colormap
    """
    img_np = np.array(img)
    cam_uint8 = (cam_map*255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlayed = cv2.addWeighted(img_np, 1.0, heatmap, alpha, 0)
    return Image.fromarray(overlayed)

def add_colorbar_to_image(img: Image.Image, cam_map: np.ndarray, colormap=cv2.COLORMAP_JET, bar_width=30):
    """
    Vertical heatmap colorbar to the right of the image.
    img: PIL.Image
    cam_map: 2D numpy array with values in [0,1]
    colormap: OpenCV colormap
    bar_width: width in pixels
    """
    img_np = np.array(img)
    h, w = cam_map.shape

    # Create colorbar
    colorbar = np.linspace(0, 1, h).reshape(h,1)
    colorbar = np.repeat(colorbar, bar_width, axis=1)  # width
    colorbar_uint8 = (colorbar*255).astype(np.uint8)
    colorbar_rgb = cv2.applyColorMap(colorbar_uint8, colormap)
    colorbar_rgb = cv2.cvtColor(colorbar_rgb, cv2.COLOR_BGR2RGB)

    # Resize to match main image height
    colorbar_rgb = cv2.resize(colorbar_rgb, (bar_width, img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Concatenate image + colorbar
    combined = np.concatenate([img_np, colorbar_rgb], axis=1)

    return Image.fromarray(combined)
# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(__file__)

    stage_model = ROPNet().to(DEVICE)

    stage_path = os.path.join(BASE_DIR, "..", "results", "final_stage_model.pt")
    checkpoint = torch.load(stage_path, map_location=DEVICE)

    state_dict = checkpoint.get("model", checkpoint)
    temperature = checkpoint.get("temperature", 1.0)

    model_dict = stage_model.state_dict()

    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }

    stage_model.load_state_dict(filtered_dict, strict=False)
    stage_model.eval()

    unet = VesselUNet().to(DEVICE)
    unet_path = os.path.join(BASE_DIR, "..", "vessel_unet_512.pt")
    unet.load_state_dict(torch.load(unet_path, map_location=DEVICE))
    unet.eval()

    cam_extractor = GradCAM(stage_model)

    return stage_model, unet, cam_extractor, overlay_mask, temperature

stage_model, unet, cam_extractor, overlay_mask, TEMPERATURE = load_models()

# =========================
# PREPROCESS & MASK
# =========================
def ben_graham_inference(x):
    blur = gaussian_blur(x, kernel_size=31)
    return torch.clamp(4*x - 4*blur + 0.5, 0, 1)

def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    img_t = ben_graham_inference(img_t)
    img_t = NORMALIZE(img_t)
    return img_t.to(DEVICE)

def ordinal_probs_from_logits(logits):
    # logits: (B, K) where K = NUM_CLASSES - 1
    cumulative = torch.sigmoid(logits)

    # enforce monotonic decreasing
    cumulative = torch.cummin(cumulative, dim=1)[0]

    B, K = cumulative.shape

    probs = torch.zeros((B, K + 1), device=logits.device)

    probs[:, 0] = 1 - cumulative[:, 0]

    for k in range(1, K):
        probs[:, k] = cumulative[:, k-1] - cumulative[:, k]

    probs[:, -1] = cumulative[:, -1]

    return torch.clamp(probs, 1e-6, 1.0)
def get_mask(unet, img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_proc = preprocess_retina(img_bgr, 512)

    x = torch.tensor(img_proc).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred = torch.sigmoid(unet(x))[0,0]

    # keep soft mask (NO hard thresholding)
    mask = cv2.resize(pred.cpu().numpy(), (IMG_SIZE, IMG_SIZE))

    # normalize instead of binarize
    mask = (mask - mask.min()) / (mask.max() + 1e-6)

    return torch.tensor(mask).to(DEVICE)
def get_circular_retina_mask(shape):
    h, w = shape

    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)

    radius = min(center[0], center[1]) * 0.95

    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    mask = mask.astype(np.float32)

    return mask
def predict(stage_model, unet, img, ga, bw, cam_extractor=None, overlay_fn=None, show_cam=True):
    img_rgb = np.array(img)
    img_t = preprocess(img_rgb)
    
    # Get vessel mask
    mask = get_mask(unet, img_rgb)
    
    # Prepare input for model
    x = torch.cat([img_t, mask.unsqueeze(0)], dim=0).unsqueeze(0).to(DEVICE)
    if show_cam:
        x.requires_grad_(True) 

    # Normalize clinical data
    ga_norm = (ga - 22) / (40 - 22)
    bw_norm = (bw - 400) / (2500 - 400)
    clinical = torch.tensor([[ga_norm, bw_norm]], device=DEVICE).float()

    # 1. Model prediction
    stage_model.eval()
    with torch.set_grad_enabled(True):
        logits_ord = stage_model(x, clinical)
        logits_ord = logits_ord / TEMPERATURE
        probs = ordinal_probs_from_logits(logits_ord)
        pred = int(probs.argmax(dim=1).item())
        probs_np = probs.detach().cpu().numpy()[0]

    # 2. Grad-CAM
    cam_img = None
    if show_cam and cam_extractor:
        stage_model.zero_grad()
        cam_class_idx = pred if pred < logits_ord.shape[1] else logits_ord.shape[1] - 1
        #activation_map = cam_extractor(cam_class_idx, logits_ord)
        #cam_map = np.squeeze(activation_map[0].detach().cpu().numpy())
        cam_map = cam_extractor.generate(x, clinical, cam_class_idx)[0]

        # Resize CAM to original image size
        cam_map = cv2.resize(cam_map, (img.size[0], img.size[1]), interpolation=cv2.INTER_LINEAR)

        # HARD retina boundary (prevents leakage)
        hard_mask = get_circular_retina_mask(cam_map.shape)

        # soft vessel mask (kept as guidance, not boundary)
        soft_mask = get_retina_mask_for_cam(mask, (img.size[0], img.size[1]))

        # combine: HARD × SOFT
        final_mask = hard_mask * (soft_mask > 0.2).astype(np.float32)

        cam_map = cam_map * final_mask

        # Smooth CAM: median + Gaussian to remove moiré
        cam_map = cv2.medianBlur((cam_map*255).astype(np.uint8), 9) / 255.0
        cam_map = cv2.GaussianBlur(cam_map, (21,21), 0)

        # Normalize 0-1
        cam_map = np.nan_to_num(cam_map)

        cam_map = np.nan_to_num(cam_map)

        cam_map = np.clip(cam_map, 0, None)
        cam_map = cam_map - cam_map.min()

        cam_map = cam_map / (cam_map.max() + 1e-6)

        # force zero outside retina again (final safety)
        cam_map = cam_map * final_mask

        # Overlay as light heatmap
        #cam_img = np.array(overlay_light_cam(img, cam_map, alpha=0.3))
        cam_img = cam_map
    return pred, probs_np, mask.cpu().numpy(), cam_map

def referral(label_name):
    if label_name == "Severe":
        return "URGENT REFERRAL"
    elif label_name == "Mild":
        return "MONITOR CLOSELY"
    else:
        return "ROUTINE FOLLOW-UP"

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(layout="wide")
apply_style()
header()

# Initialize session state keys if they don't exist
if "page" not in st.session_state:
    st.session_state.page = "upload"

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

if "result" not in st.session_state:
    st.session_state.result = None
# ---- UPLOAD PAGE ----
if st.session_state.page == "upload":
    subheader("Upload a retinal image and enter clinical data to get an AI-assisted ROP stage prediction, referral recommendation, and visual explanation of the model's focus areas.")
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Upload Retinal Image")
        file = st.file_uploader(
            "Drag & Drop or Click",
            type=["jpg", "png"],
            key="file_uploader"
        )
        if file is not None:
            # Create a small preview of the uploaded file
            preview_img = Image.open(file)
            st.image(preview_img, caption="Image Preview", width=250)
            st.session_state.uploaded_file = file

        st.session_state.uploaded_file = file # store in session_state

    with col2:
        st.markdown("### Clinical Data")
        ga = st.number_input(
            "Gestational Age (22-40 weeks)",
            min_value=22,
            max_value=40,
            value=None,
            step=1,
            placeholder="Enter GA (e.g. 28)"
        )
        bw = st.number_input(
            "Birth Weight (400-2500 g)",
            min_value=400,
            max_value=2500,
            value=None,
            step=10,
            placeholder="Enter BW (e.g. 1200)"
        )


        show_cam = st.checkbox("Show Grad-CAM", value=True)

    if st.button("Run Analysis", key="run_analysis"):

        # 1. Check image
        if st.session_state.uploaded_file is None:
            st.warning("Please upload an image")
            st.stop()

        # 2. Check clinical inputs
        if ga is None or bw is None:
            st.warning("Please enter both Gestational Age and Birth Weight")
            st.stop()

        # 3. Now safe to proceed
        img = Image.open(st.session_state.uploaded_file).convert("RGB")

        is_valid, msg = retina_score(img, unet, DEVICE)
        if not is_valid:
            st.error(msg)
            st.stop()

        with st.spinner("Running analysis..."):
            pred, probs, mask, cam_img = predict(
                stage_model, unet, img, ga, bw,
                cam_extractor=cam_extractor,
                overlay_fn=overlay_mask,
                show_cam=show_cam
            )

        st.session_state.result = {
            "img": img,
            "pred": pred,
            "probs": probs.tolist(),
            "mask": mask.tolist(),
            "cam": cam_img.tolist() if cam_img is not None else None,
            "ref": referral(pred),
            "ga": ga,
            "bw": bw
        }

        st.session_state.page = "result"
        st.rerun()

# ---- RESULT PAGE ----
elif st.session_state.page == "result":
    data = st.session_state.result

    # ---- BACK BUTTON ----
    if st.button("← Back"):
        st.session_state.page = "upload"
        st.rerun()
    # =========================
    # 1. DIAGNOSTIC SUMMARY (TOP)
    # =========================
    st.markdown("### Case Summary")

    pred_stage = data["pred"]
    pred_prob = data["probs"][pred_stage]

    col1, col2, col3 = st.columns(3)

    # --- Patient Info ---
    with col1:
        st.markdown(f"**Gestational Age**  \n{data['ga']} weeks")
        st.markdown(f"**Birth Weight**  \n{data['bw']} g")

    # --- Diagnosis + Recommendation ---
    with col2:
        label_name = CLASS_NAMES[pred_stage]

        if label_name == "Severe":
            st.error(f"**Predicted Condition:** {label_name}")
        elif label_name == "Mild":
            st.warning(f"**Predicted Condition:** {label_name}")
        else:
            st.success(f"**Predicted Condition:** {label_name}")

        # =========================
        # BASE REFERRAL DECISION
        # =========================
        if label_name == "Severe":
            final_ref = "URGENT REFERRAL"
        elif label_name == "Mild":
            final_ref = "MONITOR CLOSELY"
        else:
            final_ref = "ROUTINE FOLLOW-UP"
        # =========================
        # SAFETY OVERRIDE (confidence-aware)
        # =========================
        if pred_prob < 0.7:
            if final_ref == "ROUTINE FOLLOW-UP":
                final_ref = "REQUIRES EXPERT REVIEW"
            else:
                final_ref += " (Low Confidence - Verify)"

        # =========================
        # DISPLAY RECOMMENDATION (FIX ADDED)
        # =========================
        if "URGENT" in final_ref:
            st.error(f"**Recommendation:** {final_ref}")
        elif "MONITOR" in final_ref:
            st.warning(f"**Recommendation:** {final_ref}")
        else:
            st.success(f"**Recommendation:** {final_ref}")

        # STORE IT FOR LATER USE (IMPORTANT FIX)
        data["ref"] = final_ref

    # --- Confidence ---
    with col3:
        st.markdown(f"**Confidence**  \n{pred_prob*100:.1f}%")

        if pred_prob > 0.85:
            st.success("High")
        elif pred_prob > 0.7:
            st.warning("Moderate")
        else:
            st.error("Low")

        st.markdown("---")
    # =========================
    # 2. VISUAL SECTION
    # =========================
    st.markdown("## Visual Explanation")

    # --- Toggle View ---
    view_mode = st.radio(
        "Select View",
        ["Grad-CAM", "Original Image"],
        horizontal=True
    )

    # --- Slider (only for CAM) ---
    alpha = 0.3
    if view_mode == "Grad-CAM" and data.get("cam") is not None:
       alpha = st.slider("Heatmap Intensity", 0.1, 0.8, 0.3)

    # Prepare images
    cam_map = np.array(data["cam"], dtype=np.float32) if data.get("cam") is not None else None
    
    if cam_map is not None:
        heatmap_img = overlay_light_cam(data["img"], cam_map, alpha=alpha)
        cam_img = overlay_light_cam(data["img"], cam_map, alpha=alpha)
    else:
        cam_img = None

    # --- Main + Thumbnail Layout ---
    main_col, side_col = st.columns([3,1])

    # MAIN IMAGE
    with main_col:
        if view_mode == "Grad-CAM" and cam_img is not None:
            st.image(cam_img, caption="Grad-CAM (Enlarged)", width="stretch")
        else:
            st.image(data["img"], caption="Original Image (Enlarged)", width="stretch")

    # THUMBNAIL (PiP)
    with side_col:
        st.markdown("### Preview")

        if view_mode == "Grad-CAM":
            st.image(data["img"], caption="Original", width="stretch")
        else:
            if cam_img is not None:
                st.image(cam_img, caption="Grad-CAM", width="stretch")

    # =========================
    # 3. DOWNLOAD SECTION
    # =========================
    st.markdown("---")
    st.markdown("## Export Results")

    if data.get("cam") is not None:
        cam_pil = add_colorbar_to_image(
            overlay_light_cam(data["img"], cam_map, alpha=alpha),
            cam_map
        )

        col1, col2 = st.columns(2)

        # PNG
        with col1:
            buf = io.BytesIO()
            cam_pil.save(buf, format="PNG")
            buf.seek(0)

            st.download_button(
                "Download PNG",
                data=buf,
                file_name="gradcam.png",
                mime="image/png"
            )

        # PDF
        with col2:
            pdf_buf = io.BytesIO()
            c = canvas.Canvas(pdf_buf, pagesize=A4)
            width, height = A4

            img_width, img_height = cam_pil.size
            ratio = min(width/img_width, height/img_height)

            c.drawImage(
                ImageReader(cam_pil),
                (width - img_width*ratio)/2,
                (height - img_height*ratio)/2,
                img_width*ratio,
                img_height*ratio
            )

            c.showPage()
            c.save()
            pdf_buf.seek(0)

            st.download_button(
                "Download PDF",
                data=pdf_buf,
                file_name="gradcam.pdf",
                mime="application/pdf"
            )

    # =========================
    # 4. ACTIONS
    # =========================
    st.markdown("---")

    if st.button("New Analysis"):
        for key in ["uploaded_file", "result", "ga", "bw"]:
            st.session_state.pop(key, None)

        st.session_state.page = "upload"
        st.rerun()
footer() 