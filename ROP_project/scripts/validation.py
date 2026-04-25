import numpy as np
import cv2
import torch

from scripts.train_segmentation import preprocess_retina

CIRCULARITY_MIN   = 0.45   # < 0.45 → rectangle/face/landscape
FILL_RATIO_MIN    = 0.55   # bright region must fill >55% of bounding box
FILL_RATIO_MAX    = 0.99   # > 0.99 → fully white/overexposed, not retinal
 
# Check 3 — colour profile
# Fundus images: red channel dominant, blue suppressed
# Moon: all channels roughly equal (achromatic)
RG_RATIO_MIN      = 1.05   # red / green must be > 1.05 (relaxed for high pigmentation)
BLUE_RATIO_MAX    = 0.90   # blue / green must be < 0.90 (moons ≈ 1.0)
COLOUR_STD_MIN    = 8.0    # channels must differ; greyscale scan → all equal
 
# Check 4 — texture complexity
# Vasculature produces fine, directional texture; moons have smooth gradients
LAPLACIAN_VAR_MIN = 30.0   # variance of Laplacian; moon ≈ 5–20, retina ≈ 40–500
LOCAL_STD_MIN     = 6.0    # mean local std in 16×16 patches; smooth image = fails
 
# Check 5 — U-Net vessel activation
# Using soft composite score to avoid rejecting low-contrast retinal images
VESSEL_SCORE_MIN  = 0.012  # composite score threshold (very relaxed)
                            # typical retina: 0.04–0.25, moon: < 0.008

# HELPER: extract the bright circular region mask
def _get_retinal_mask(gray: np.ndarray):
    """
    Returns (mask, circularity, fill_ratio) for the largest bright contour.
    mask is a uint8 binary image.
    """
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
 
    # Morphological close to handle fragmented edges from poor illumination
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
 
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, 0.0
 
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True) + 1e-6
    circularity = 4 * np.pi * area / (perimeter ** 2)
 
    x, y, w, h = cv2.boundingRect(cnt)
    bounding_area = max(w * h, 1)
    fill_ratio = area / bounding_area
 
    # Build mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
 
    return mask, circularity, fill_ratio
 

# HELPER: local texture complexity
def _local_texture_score(gray: np.ndarray, patch_size: int = 16) -> float:
    """
    Divides the image into patches and computes mean standard deviation.
    Smooth surfaces (moon, blank image) score low; textured retina scores high.
    """
    h, w = gray.shape
    stds = []
    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            stds.append(float(patch.std()))
    return float(np.mean(stds)) if stds else 0.0
 
 

# MAIN VALIDATOR
def retina_score(img, unet_model, device="cpu"):
    """
    Parameters
    ----------
    img        : PIL.Image (RGB)
    unet_model : trained VesselUNet, already in eval mode
    device     : 'cpu' or 'cuda'
 
    Returns
    -------
    (bool, str) — (is_valid, message)
    """
 
    img_np = np.array(img)
 
    # Check 1: Format
    if img_np.ndim != 3 or img_np.shape[2] != 3:
        return False, "Invalid image format. Please upload an RGB image."
 
    h, w = img_np.shape[:2]
    if h < 64 or w < 64:
        return False, "Image is too small to be a retinal scan."
 
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
 
    # Reject near-blank images
    if gray.mean() < 5 or gray.std() < 3:
        return False, "Image appears blank or corrupted."
 
    # Check 2: Circular structure
    mask, circularity, fill_ratio = _get_retinal_mask(gray)
 
    if mask is None:
        return False, "No detectable structure in image."
 
    if circularity < CIRCULARITY_MIN:
        return False, (
            "Image does not appear to be a retinal fundus scan "
            "(no circular disc structure detected)."
        )
 
    if not (FILL_RATIO_MIN <= fill_ratio <= FILL_RATIO_MAX):
        return False, (
            "Image structure does not match a retinal fundus scan "
            "(unexpected fill ratio)."
        )
 
    # Check 3: Colour profile 
    # Analyse only within the detected circular region to avoid black borders
    roi = img_np.copy()
    roi[mask == 0] = 0
 
    r_mean = float(roi[:, :, 0][mask > 0].mean()) if (mask > 0).any() else 0
    g_mean = float(roi[:, :, 1][mask > 0].mean()) if (mask > 0).any() else 0
    b_mean = float(roi[:, :, 2][mask > 0].mean()) if (mask > 0).any() else 0
 
    g_safe   = max(g_mean, 1.0)
    rg_ratio = r_mean / g_safe
    bg_ratio = b_mean / g_safe
 
    # Channel difference: greyscale images have all channels equal
    channel_std = float(np.std([r_mean, g_mean, b_mean]))
 
    # Moon rejection: moons are achromatic (rg_ratio ≈ 1.0, bg_ratio ≈ 1.0)
    # Fundus images: rg_ratio > 1.05, bg_ratio < 0.90
    is_achromatic = (
        abs(rg_ratio - 1.0) < 0.08 and
        abs(bg_ratio - 1.0) < 0.08 and
        channel_std < COLOUR_STD_MIN
    )
    if is_achromatic:
        return False, (
            "Image colour profile does not match a retinal fundus scan. "
            "Retinal images have a characteristic red/green dominance."
        )
 
    if rg_ratio < RG_RATIO_MIN and channel_std < COLOUR_STD_MIN:
        return False, (
            "Image colour profile suggests this is not a retinal fundus scan."
        )
 
    if bg_ratio > BLUE_RATIO_MAX and channel_std < COLOUR_STD_MIN:
        return False, (
            "Image has an unusual colour profile for a retinal scan."
        )
 
    # Check 4: Texture complexity 
    # Apply to the ROI only — crop to bounding box of the mask
    ys, xs = np.where(mask > 0)
    if len(ys) > 0:
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        gray_roi = gray[y1:y2, x1:x2]
    else:
        gray_roi = gray
 
    laplacian_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
    local_std     = _local_texture_score(gray_roi)
 
    # Both metrics must fail to trigger rejection
    # (one alone could fail on unusually smooth or sharp fundus images)
    if laplacian_var < LAPLACIAN_VAR_MIN and local_std < LOCAL_STD_MIN:
        return False, (
            "Image lacks the vascular texture expected in a retinal scan. "
            "Please check the image is a fundus photograph."
        )
 
    # Check 5: U-Net vessel activation 
    # Soft composite score — avoids single hard thresholds that reject
    # valid low-contrast or high-pigmentation retinal images.
    img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_proc = preprocess_retina(img_bgr, 512)
    x_tensor = (
        torch.tensor(img_proc)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )
 
    with torch.no_grad():
        pred_mask = torch.sigmoid(unet_model(x_tensor))[0, 0].cpu().numpy()
 
    # Only evaluate within the detected retinal region
    mask_512 = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    roi_pixels = pred_mask[mask_512 > 0]
 
    if len(roi_pixels) == 0:
        roi_pixels = pred_mask.ravel()
 
    mean_act       = float(roi_pixels.mean())
    high_conf_frac = float((roi_pixels > 0.25).mean())
    activation_std = float(roi_pixels.std())
 
    # Composite score: weighted sum rewarding both mean activation
    # and spatial spread of confident predictions
    vessel_score = (0.5 * mean_act) + (0.3 * high_conf_frac) + (0.2 * activation_std)
 
    if vessel_score < VESSEL_SCORE_MIN:
        return False, (
            "No retinal vascular structure detected by the vessel model. "
            "Please ensure this is a fundus photograph of sufficient quality."
        )
 
    # All checks passed 
    return True, "Valid retinal image."
 