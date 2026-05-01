import pandas as pd
from pathlib import Path
import re
import os

# ==================================================
# BASE PATHS
# ==================================================
DATA_DIR = Path(r"C:\Users\m12re\ROP\ROP_project\data")

# Excel files containing clinical metadata
ZIP_XLSX    = DATA_DIR / "zip information.xlsx"
INFANT_XLSX = DATA_DIR / "infant_retinal_database_info.xlsx"

# Output CSV file for final structured dataset
STAGE_CSV = DATA_DIR / "stage_data.csv"

STAGE_DIRS = {
    0: DATA_DIR / "classification/Normal",
    1: DATA_DIR / "classification/Mild",
    2: DATA_DIR / "classification/Severe",
}

STAGE_MASK_ROOT = {
    0: DATA_DIR / "unet_masks/Normal",
    1: DATA_DIR / "unet_masks/Mild",
    2: DATA_DIR / "unet_masks/Severe",
}

# ==================================================
# DATA LOADING & LOOKUP TABLE
# ==================================================
def create_lookup_table():
    """
    Loads clinical metadata from Excel files and creates lookup tables.

    Returns:
        archive_lookup: DataFrame mapping filenames to clinical data (Archive dataset)
        infant_db: DataFrame containing Ostrava dataset clinical data
    """
    if not ZIP_XLSX.exists() or not INFANT_XLSX.exists():
        print(f"ERROR: Excel files not found in {DATA_DIR}")
        return pd.DataFrame(), pd.DataFrame()

    try:
        # Load Zip Information
        s1 = pd.read_excel(ZIP_XLSX, sheet_name='Sheet1')
        s2 = pd.read_excel(ZIP_XLSX, sheet_name='Sheet2')
        
        # Merge both sheets on patient ID to combine filename + clinical data
        archive_lookup = pd.merge(s1, s2, on='ID', how='left')
        
        archive_lookup = archive_lookup.rename(columns={
            'img_name': 'filename',
            'Gestational age at birth(week)': 'ga',
            'Birth weight(g)': 'bw',
            'ID': 'patient_id'
        })
        
        # Load Infant Database (Ostrava data)
        infant_db = pd.read_excel(INFANT_XLSX, sheet_name='database')
        infant_db = infant_db.rename(columns={
            'GESTATIONAL AGE (GA)': 'ga',
            'BIRTH WEIGHT (BW)': 'bw',
            'ID': 'patient_id'
        })
        
        return archive_lookup, infant_db
    except Exception as e:
        print(f"Error reading Excel files: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_mask_map(mask_root):
    """
    Creates a mapping between image filenames and their corresponding mask paths.

    Args:
        mask_root: Root directory of masks

    Returns:
        mask_map: Dictionary {image_stem: mask_path}
    """
    mask_map = {}
    if not mask_root.exists(): return mask_map
    for m_path in mask_root.rglob("*.png"):
        mask_map[m_path.stem] = str(m_path)
    return mask_map

# ==================================================
# MAIN BUILDER
# ==================================================
def build_dataset():
    archive_lookup, infant_db = create_lookup_table()
    
    # Stop execution if lookup table failed
    if archive_lookup.empty:
        print("Stopping: Lookup table could not be created.")
        return

    rows = []
    # Compute global fallback values for missing clinical data
    global_ga = archive_lookup['ga'].mean()
    global_bw = archive_lookup['bw'].mean()

    # Iterate through each stage/class directory
    for stage, img_dir in STAGE_DIRS.items():
        if not img_dir.exists(): continue
        
        print(f"Processing Stage {stage}...")
        # Load mask mappings for this stage
        mask_map = get_mask_map(STAGE_MASK_ROOT[stage])
        # Iterate through all images
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
            
            name = img_path.name
            ga, bw, pid = None, None, None
            
            # PATHWAY A: OSTRAVA (ID and Data in Filename)
            if "_GA" in name and "_BW" in name:
                try:
                    pid = name.split('_')[0] # Usually the first part is the ID
                    ga = int(re.search(r'GA(\d+)', name).group(1))
                    bw = int(re.search(r'BW(\d+)', name).group(1))
                except: pass
            
            # PATHWAY B: ARCHIVE2 (Lookup via Excel)
            if ga is None:
                match = archive_lookup[archive_lookup['filename'] == name]
                if not match.empty:
                    pid = match['patient_id'].values[0]
                    ga = match['ga'].values[0]
                    bw = match['bw'].values[0]
            
            # Fallback values
            ga = ga if pd.notnull(ga) else global_ga
            bw = bw if pd.notnull(bw) else global_bw
            pid = pid if pid is not None else "Unknown"
            
            mask_path = mask_map.get(img_path.stem)
            # Only include samples that have masks
            if mask_path:
                rows.append({
                    "patient_id": pid,
                    "image_path": str(img_path),
                    "mask_path": mask_path,
                    "ga": ga,
                    "bw": bw,
                    "label": stage
                })

    df = pd.DataFrame(rows)
    # Ensure patient_id is string to prevent leading zero loss
    df['patient_id'] = df['patient_id'].astype(str)
    
    df.to_csv(STAGE_CSV, index=False)
    print(f"Success! Saved {len(df)} samples with Patient IDs to {STAGE_CSV}")

if __name__ == "__main__":
    build_dataset()
