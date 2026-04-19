import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Must keep architecture identical to training
class VesselUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )
        self.enc1, self.enc2, self.enc3 = block(3, 32), block(32, 64), block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(128, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), block(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

def preprocess_retina(img_bgr, target_size=512):
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    green_ch = img_resized[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_ch)
    img_final = cv2.merge([enhanced, enhanced, enhanced])
    return img_final.astype(np.float32) / 255.0

def run_generation():
    WEIGHTS_PATH = "vessel_unet_512.pt"
    OUT_DIR = Path(r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\unet_masks")
    
    sources = {
        "Normal": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\Normal",
        "Mild": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\Mild",
        "Severe": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\Severe"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselUNet().to(device)
    
    # Load the trained weights
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: {WEIGHTS_PATH} not found. Run training script first.")
        return
        
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        for tag, root in sources.items():
            path_root = Path(root)
            if not path_root.exists(): continue
            
            target_out = OUT_DIR / tag
            target_out.mkdir(parents=True, exist_ok=True)

            for img_p in tqdm(path_root.glob("*"), desc=f"Processing {tag}"):
                if img_p.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
                
                img_raw = cv2.imread(str(img_p))
                h, w = img_raw.shape[:2]
                img_proc = preprocess_retina(img_raw, 512)
                x = torch.from_numpy(img_proc).permute(2,0,1).float().unsqueeze(0).to(device)
                
                pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()
                mask = (pred > 0.4).astype(np.uint8) * 255
                
                # Resize to original image size
                final = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(target_out / f"{img_p.stem}.png"), final)

if __name__ == "__main__":
    run_generation()
    print("Mask generation complete.")
