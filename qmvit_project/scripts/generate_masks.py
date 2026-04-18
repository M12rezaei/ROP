import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.filters import frangi
import warnings

warnings.filterwarnings("ignore")

# =========================================
# Vessel Probability Map (Cleaned)
# =========================================
def extract_vessel_prob_map(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Green channel
    green = img[:, :, 1]

    # CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = clahe.apply(green)

    # Normalize for Frangi
    green_norm = green.astype(np.float32) / 255.0

    # Frangi vesselness
    vessel = frangi(
        green_norm,
        sigmas=np.linspace(1, 3, 4),
        alpha=0.5,
        beta=0.5,
        gamma=15,
        black_ridges=True
    )
    vessel = np.nan_to_num(vessel)

    # Clip extreme values
    vessel = np.clip(vessel, 0, np.percentile(vessel, 99))
    vessel = vessel / (vessel.max() + 1e-6)

    # Convert to 0-255
    vessel = (vessel * 255).astype(np.uint8)

    # --- Remove small noisy white spots ---
    _, vessel_bin = cv2.threshold(vessel, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_clean = cv2.morphologyEx(vessel_bin, cv2.MORPH_OPEN, kernel)

    return vessel_clean

# =========================================
# Paths
# =========================================
ROOT = Path(r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification")
MASK_ROOT = ROOT.parent / "vessel_masks_clean"

PLUS_MASK_DIR = MASK_ROOT / "plus"
STAGE_MASK_DIR = MASK_ROOT / "stages"

PLUS_MASK_DIR.mkdir(parents=True, exist_ok=True)
STAGE_MASK_DIR.mkdir(parents=True, exist_ok=True)

# =========================================
# Plus Disease
# =========================================
plus_paths = {
    "no_plus": ROOT / "plus" / "no_plus",
    "plus": ROOT / "plus" / "plus",
}

plus_rows = []

for label, folder in plus_paths.items():
    if not folder.exists():
        continue

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    for img_path in tqdm(images, desc=f"Processing Plus {label}"):
        mask = extract_vessel_prob_map(img_path)
        if mask is None:
            continue

        mask_name = f"{label}_{img_path.stem}_mask.png"
        mask_path = PLUS_MASK_DIR / mask_name
        cv2.imwrite(str(mask_path), mask)

        plus_rows.append({
            "full_image_path": str(img_path),
            "full_mask_path": str(mask_path),
            "Plus": 1 if label == "plus" else 0
        })

# Save CSV
plus_csv = ROOT.parent / "fused_plus_clean.csv"
pd.DataFrame(plus_rows).to_csv(plus_csv, index=False)
print(f"✅ Plus masks saved | CSV: {plus_csv}")

# =========================================
# ROP Stages
# =========================================
stage_paths = {f"stage{i}": ROOT / "stages" / f"stage{i}" for i in range(6)}

stage_rows = []

for label, folder in stage_paths.items():
    if not folder.exists():
        continue

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    for img_path in tqdm(images, desc=f"Processing {label}"):
        mask = extract_vessel_prob_map(img_path)
        if mask is None:
            continue

        mask_name = f"{label}_{img_path.stem}_mask.png"
        mask_path = STAGE_MASK_DIR / mask_name
        cv2.imwrite(str(mask_path), mask)

        stage_rows.append({
            "full_image_path": str(img_path),
            "full_mask_path": str(mask_path),
            "Stage": int(label.replace("stage", ""))
        })

# Save CSV
stage_csv = ROOT.parent / "fused_stage_clean.csv"
pd.DataFrame(stage_rows).to_csv(stage_csv, index=False)
print(f" Stage masks saved | CSV: {stage_csv}")

print("\n Vessel mask generation complete (Plus + Stage)")
