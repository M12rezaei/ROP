import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# 1. PRE-PROCESSING (FOR 512x512 QUALITY)
# =========================================================
def preprocess_retina(img_bgr, target_size=512):
    img_resized = cv2.resize(img_bgr, (target_size, target_size))
    green_ch = img_resized[:, :, 1] # Green channel has best vessel contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_ch)
    # Stack to 3-channel for the UNet input
    img_final = cv2.merge([enhanced, enhanced, enhanced])
    return img_final.astype(np.float32) / 255.0

# =========================================================
# 2. UNet ARCHITECTURE
# =========================================================
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

# =========================================================
# 3. DATASET & LOSS
# =========================================================
class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.size = size
        valid = [".png", ".jpg", ".jpeg"]
        self.imgs = sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in valid])
        self.masks = sorted([p for p in Path(mask_dir).glob("*") if p.suffix.lower() in valid])

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = preprocess_retina(cv2.imread(str(self.imgs[idx])), self.size)
        mask = cv2.resize(cv2.imread(str(self.masks[idx]), 0), (self.size, self.size))
        return torch.from_numpy(img).permute(2,0,1).float(), (torch.from_numpy(mask).unsqueeze(0).float() > 127).float()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6
    return 1 - (num / den).mean()

# =========================================================
# 4. TRAINING WITH LIVE PLOTTING LOGIC
# =========================================================
def train_vessel_unet(img_dir, mask_dir, epochs=50, size=512, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # Metrics
    met_acc = BinaryAccuracy().to(device)
    met_auc = BinaryAUROC().to(device)
    
    ds = VesselDataset(img_dir, mask_dir, size)
    tr_len = int(len(ds) * 0.85)
    tr_ds, va_ds = random_split(ds, [tr_len, len(ds) - tr_len])
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size)

    history = {'loss': [], 'acc': [], 'auc': []}

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for x, y in tqdm(tr_loader, desc=f"Epoch {ep}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(out, y) + dice_loss(out, y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                preds = torch.sigmoid(model(x))
                met_acc.update(preds, y); met_auc.update(preds, y)
        
        # Save and Print Epoch Metrics
        history['loss'].append(train_loss / len(tr_loader))
        history['acc'].append(met_acc.compute().item())
        history['auc'].append(met_auc.compute().item())
        
        print(f"Epoch [{ep:02d}] Loss: {history['loss'][-1]:.4f} | Acc: {history['acc'][-1]:.3f} | AUC: {history['auc'][-1]:.3f}")
        met_acc.reset(); met_auc.reset()

    # Save Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], color='red', label='Loss')
    plt.title('Training Loss (512x512)'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['acc'], color='green', label='Accuracy')
    plt.plot(history['auc'], color='blue', label='AUC')
    plt.title('Validation Acc & AUC'); plt.legend()
    
    plt.savefig("vessel_metrics_512.png")
    print(" Training plots saved as 'vessel_metrics_512.png'")
    
    return model, device

# =========================================================
# 5. MASK GENERATION (KEEPING YOUR PATHS)
# =========================================================
def generate_masks(model, device, base_out, size=512):
    base_out = Path(base_out)
    sources = {
        "plus/no_plus": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\plus\no_plus",
        "plus/plus": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\plus\plus",
        "stage/stage0": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage0",
        "stage/stage1": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage1",
        "stage/stage2": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage2",
        "stage/stage3": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage3",
        "stage/stage4": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage4",
        "stage/stage5": r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\classification\stages\stage5",
    }

    model.eval()
    with torch.no_grad():
        for tag, root in sources.items():
            path_root = Path(root)
            if not path_root.exists(): continue
            out_dir = base_out / tag
            out_dir.mkdir(parents=True, exist_ok=True)

            for img_p in tqdm(path_root.glob("*"), desc=f"Generating {tag}"):
                if img_p.suffix.lower() not in [".jpg", ".png", ".jpeg"]: continue
                img_raw = cv2.imread(str(img_p))
                h, w = img_raw.shape[:2]
                img_proc = preprocess_retina(img_raw, size)
                x = torch.from_numpy(img_proc).permute(2,0,1).float().unsqueeze(0).to(device)
                
                pred = torch.sigmoid(model(x)).squeeze().cpu().numpy()
                mask = (pred > 0.4).astype(np.uint8) * 255
                
                # Resize back to original
                final = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out_dir / f"{img_p.stem}.png"), final)

if __name__ == "__main__":
    IMG_DIR = r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\segmentation\images"
    MSK_DIR = r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\segmentation\masks"
    OUT_DIR = r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\unet_masks"

    # Train and Save Metrics
    model, device = train_vessel_unet(IMG_DIR, MSK_DIR, epochs=50, size=512)
    torch.save(model.state_dict(), "vessel_unet_512.pt")
    
    # Generate Masks
    generate_masks(model, device, OUT_DIR, size=512)
    print(f" Process Complete. High-res masks in: {OUT_DIR}")