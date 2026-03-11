import pandas as pd
from pathlib import Path
import re

# ==================================================
# BASE PATHS
# ==================================================
DATA_DIR = Path(r"C:\Users\m12re\ROP\qmvit_project\data")

ZIP_XLSX    = DATA_DIR / "zip information.xlsx"
INFANT_XLSX = DATA_DIR / "infant_retinal_database_info.xlsx"

STAGE_CSV = DATA_DIR / "stage_data.csv"

# Source Folders
STAGE_MASK_ROOT = {
    0: DATA_DIR / "mask/stage0",
    1: DATA_DIR / "mask/stage1",
    2: DATA_DIR / "mask/stage2",
    3: DATA_DIR / "mask/stage3",
    4: DATA_DIR / "mask/stage4",
    5: DATA_DIR / "mask/stage5",
}

STAGE_DIRS = {
    0: DATA_DIR / "stage/stage0",
    1: DATA_DIR / "stage/stage1",
    2: DATA_DIR / "stage/stage2",
    3: DATA_DIR / "stage/stage3",
    4: DATA_DIR / "stage/stage4",
    5: DATA_DIR / "stage/stage5",
}

# ==================================================
# UTILITIES
# ==================================================
PID_REGEX = re.compile(r'(?<!\d)(\d{1,5})(?!\d)')

def extract_patient_id(name: str):
    m = PID_REGEX.search(name)
    return int(m.group(1)) if m else None

def find_mask_by_filename(mask_root: Path, target_stem: str):
    for m_path in mask_root.rglob("*.png"):
        if m_path.stem == target_stem:
            return str(m_path)
    return None

# ==================================================
# CLINICAL DATA LOADER (WITH MEAN IMPUTATION)
# ==================================================
def load_and_standardize_clinical():
    def standardize(df):
        df.columns = (
            df.columns.str.strip().str.lower()
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
        col_map = {}
        for c in df.columns:
            if c in ["id", "patient id", "patient_id"]:
                col_map[c] = "patient_id"
            elif "gestational age" in c:
                col_map[c] = "ga_week"
            elif "birth weight" in c:
                col_map[c] = "bw_g"
        df = df.rename(columns=col_map)
        keep = [c for c in ["patient_id", "ga_week", "bw_g"] if c in df.columns]
        return df[keep]

    try:
        zip_df = standardize(pd.read_excel(ZIP_XLSX))
        infant_df = standardize(pd.read_excel(INFANT_XLSX))
        clinical = pd.concat([zip_df, infant_df], ignore_index=True)
        
        clinical = clinical.dropna(subset=["patient_id"])
        clinical["patient_id"] = clinical["patient_id"].astype(int)

        # MEAN IMPUTATION LOGIC
        ga_mean = clinical["ga_week"].mean()
        bw_mean = clinical["bw_g"].mean()

        clinical["ga_week"] = clinical["ga_week"].fillna(ga_mean)
        clinical["bw_g"] = clinical["bw_g"].fillna(bw_mean)

        print(f" Imputation Complete: Filled missing GA with {ga_mean:.2f} and BW with {bw_mean:.2f}")
        return clinical, ga_mean, bw_mean
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return pd.DataFrame(columns=["patient_id", "ga_week", "bw_g"]), 0, 0

# ==================================================
# STAGE CSV BUILDER
# ==================================================
def build_stage_csv(clinical_df, global_ga, global_bw):
    rows = []
    skipped_no_mask = 0
    
    for stage, img_dir in STAGE_DIRS.items():
        if not img_dir.exists(): 
            print(f" Warning: Directory not found for Stage {stage}")
            continue
            
        for img in img_dir.glob("*"):
            if img.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue

            pid = extract_patient_id(img.name)
            
            # Match clinical data
            clin = clinical_df[clinical_df.patient_id == pid]
            
            if not clin.empty:
                ga = clin.iloc[0]["ga_week"]
                bw = clin.iloc[0]["bw_g"]
            else:
                # Fallback to global mean if patient not in Excel at all
                ga = global_ga
                bw = global_bw

            # Match vessel mask
            mask = find_mask_by_filename(STAGE_MASK_ROOT[stage], img.stem)
            if mask is None:
                skipped_no_mask += 1
                continue

            rows.append({
                "patient_id": pid,
                "ga_week": ga,
                "bw_g": bw,
                "image_path": str(img),
                "mask_path": mask,
                "stage_label": stage
            })

    df = pd.DataFrame(rows)
    df.to_csv(STAGE_CSV, index=False)
    print(f" STAGE CSV saved: {len(df)} samples.")
    if skipped_no_mask > 0:
        print(f" Skipped {skipped_no_mask} images due to missing vessel masks.")

# ==================================================
# MAIN
# ==================================================
if __name__ == "__main__":
    print("Starting Stage CSV data synchronization...")
    clinical_data, g_ga, g_bw = load_and_standardize_clinical()
    build_stage_csv(clinical_data, g_ga, g_bw)
    print("\nDONE — Stage dataset updated with clinical imputation.")