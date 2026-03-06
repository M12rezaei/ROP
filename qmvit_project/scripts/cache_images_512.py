import os
import torch
import pandas as pd
import cv2
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur

IMG_SIZE = 224

def ben_graham(x):
    """Ben Graham preprocessing."""
    blur = gaussian_blur(x, kernel_size=31)
    return torch.clamp(4*x - 4*blur + 0.5, 0, 1)

# Image transform: to tensor, resize, normalize
tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


def main(csv_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)

    print(f"Caching {len(df)} images with masks at {IMG_SIZE}×{IMG_SIZE}...")

    for _, r in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(out_dir, os.path.basename(r.image_path) + ".pt")
        if os.path.exists(out_path):
            continue

        try:
            # --- Load and preprocess image ---
            img = cv2.imread(r.image_path)[:,:,::-1].copy()  # BGR → RGB + force contiguous
            img = tf(img)  # (3, IMG_SIZE, IMG_SIZE)
            img = ben_graham(img)

            # --- Load and preprocess mask ---
            mask = cv2.imread(r.mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = torch.zeros(IMG_SIZE, IMG_SIZE)
            else:
                mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)).copy()  # force contiguous
                mask = torch.tensor(mask, dtype=torch.float32)/255.0

            # --- Combine image + mask as 4 channels ---
            img = torch.cat([img, mask.unsqueeze(0)], dim=0)  # (4, IMG_SIZE, IMG_SIZE)

        except Exception as e:
            print(f"Error processing {r.image_path}: {e}")
            img = torch.zeros(4, IMG_SIZE, IMG_SIZE)

        # --- Save to .pt file ---
        torch.save({
            "img": img,
            "patient_id": r.patient_id,
            "stage": int(r.stage_label),
            "ga": r.ga_week,
            "bw": r.bw_g
        }, out_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="data/stage_data.csv")
    parser.add_argument("--out_dir", default="cache_512")
    args = parser.parse_args()
    main(args.csv_file, args.out_dir)
