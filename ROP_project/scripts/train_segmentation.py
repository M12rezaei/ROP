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
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
import warnings

warnings.filterwarnings("ignore")

def preprocess_retina(img_bgr, target_size=512):

    """
    Preprocess retinal fundus images for vessel segmentation.

    Steps:
    - Resize image to fixed resolution (512x512)
    - Extract green channel (best contrast for retinal vessels)
    - Apply CLAHE to enhance local contrast
    - Convert to 3-channel format for CNN input
    - Normalize pixel values to [0,1]
    """
    img_resized = cv2.resize(img_bgr, (target_size, target_size))

    # Green channel is most informative for vessel structure
    green_ch = img_resized[:, :, 1]

    # Green channel is most informative for vessel structure
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_ch)

    # Convert back to 3-channel format
    img_final = cv2.merge([enhanced, enhanced, enhanced])

    # Normalisze to [0, 1]
    return img_final.astype(np.float32) / 255.0

class VesselUNet(nn.Module):
    """
    U-Net architecture for retinal vessel segmentation.

    Encoder-decoder structure with skip connections to preserve
    fine-grained vessel details.
    """
    def __init__(self):
        super().__init__()
        # Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )
        
        # Encoder (feature extraction path)
        self.enc1, self.enc2, self.enc3 = block(3, 32), block(32, 64), block(64, 128)

        # Downsampling with max pooling
        self.pool = nn.MaxPool2d(2)

        # Bottleneck layer (deepest features)
        self.bottleneck = block(128, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), block(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), block(64, 32)

        # Final segmentation output (1 channel mask)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1)); e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder path with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.size = size
        valid = [".png", ".jpg", ".jpeg"]
        self.imgs = sorted([p for p in Path(img_dir).glob("*") if p.suffix.lower() in valid])
        self.masks = sorted([p for p in Path(mask_dir).glob("*") if p.suffix.lower() in valid])
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        # Load image and apply preprocessing
        img = preprocess_retina(cv2.imread(str(self.imgs[idx])), self.size)

        # Load grayscale mask and resize
        mask = cv2.resize(cv2.imread(str(self.masks[idx]), 0), (self.size, self.size))
        return torch.from_numpy(img).permute(2,0,1).float(), (torch.from_numpy(mask).unsqueeze(0).float() > 127).float()

def dice_loss(pred, target):
    """
    Dice losss measures overlap between prediction and ground truth.
    Useful for imbalanced segmentation tasks (vessels vs background).
    """
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + 1e-6
    return 1 - (num / den).mean()

if __name__ == "__main__":
    IMG_DIR = r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\segmentation\images"
    MSK_DIR = r"C:\Users\m12re\Downloads\Retinopathy_of_Prematurity\data\segmentation\masks"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VesselUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    ds = VesselDataset(IMG_DIR, MSK_DIR, size=512)

    # Train-validation split (85% / 15%)
    tr_len = int(len(ds) * 0.85)
    tr_ds, va_ds = random_split(ds, [tr_len, len(ds) - tr_len])
    tr_loader = DataLoader(tr_ds, batch_size=4, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=4)

    # Evaluation metrics
    met_acc = BinaryAccuracy().to(device)
    met_auc = BinaryAUROC().to(device)
    
    for ep in range(1, 51):
        model.train()
        t_loss = 0
        for x, y in tqdm(tr_loader, desc=f"Epoch {ep}/50"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            # Combined loss: BCE + Dice Loss
            loss = nn.functional.binary_cross_entropy_with_logits(out, y) + dice_loss(out, y)
            loss.backward(); optimizer.step()
            t_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for x, y in va_loader:
                x, y = x.to(device), y.to(device)
                p = torch.sigmoid(model(x))
                met_acc.update(p, y); met_auc.update(p, y)
        
        print(f"Epoch {ep} | Loss: {t_loss/len(tr_loader):.4f} | Acc: {met_acc.compute():.3f}")
        met_acc.reset(); met_auc.reset()

    torch.save(model.state_dict(), "vessel_unet_512.pt")
    print("Training Complete. Model saved as vessel_unet_512.pt")