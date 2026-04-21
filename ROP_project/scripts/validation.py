import numpy as np
import cv2
import torch

from scripts.train_segmentation import preprocess_retina


def retina_score(img, unet_model, device="cpu"):
    """
    User-friendly validation:
    Ensures the uploaded image looks like a full retinal scan.
    Returns:
        (bool, message)
    """

    img_np = np.array(img)

    # -------------------------
    # Basic format check
    # -------------------------
    if len(img_np.shape) != 3:
        return False, "Please upload a valid retinal image (RGB format)."

    # -------------------------
    # 1. SHAPE CHECK
    # -------------------------
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, "We couldn’t detect an eye image. Please upload a clear retinal scan."

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    rect_area = w * h
    solidity = float(area) / (rect_area + 1e-6)

    # Too rectangular → likely screenshot or crop
    if solidity > 0.92:
        return False, "The image doesn’t appear to be a full eye scan. Please upload the complete retinal image."

    # Too fragmented / noisy
    if solidity < 0.4:
        return False, "The image is unclear. Please upload a sharper retinal image."

    # -------------------------
    # 2. FIELD COVERAGE CHECK (NEW)
    # -------------------------
    retina_pixels = np.sum(thresh > 0)
    total_pixels = thresh.size
    coverage = retina_pixels / total_pixels

    if coverage < 0.5:
        return False, "The image seems zoomed or cropped. Please upload the full retinal scan."

    # -------------------------
    # 3. VESSEL PRESENCE CHECK
    # -------------------------
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_proc = preprocess_retina(img_bgr, 512)

    x_tensor = torch.tensor(img_proc).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred_mask = torch.sigmoid(unet_model(x_tensor))[0, 0].cpu().numpy()

    vessel_density = (pred_mask > 0.2).mean()

    if vessel_density < 0.005:
        return False, "We couldn’t detect a clear retinal structure. Please upload a proper eye scan."

    # -------------------------
    # 4. VESSEL DISTRIBUTION (ADVANCED)
    # -------------------------
    h, w = pred_mask.shape
    quadrants = [
        pred_mask[:h//2, :w//2],
        pred_mask[:h//2, w//2:],
        pred_mask[h//2:, :w//2],
        pred_mask[h//2:, w//2:]
    ]

    active_quadrants = sum([(q > 0.2).mean() > 0.002 for q in quadrants])

    if active_quadrants < 3:
        return False, "The image appears incomplete. Please upload the full retinal image."

    # -------------------------
    # PASS
    # -------------------------
    return True, "Valid retinal image"