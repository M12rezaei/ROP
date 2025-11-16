"""
Segmentation training stub for TinySwinUNet.

Usage:
    python segmentation_train.py --data_dir /path/to/data --output_dir ./outputs --epochs 10

This stub expects a folder data/segmentation with subfolders 'images' and 'masks' where images and masks
are paired by filename. It trains a very small model for demonstration and saves a checkpoint.
"""
import argparse, os, sys, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models.swin_unet import TinySwinUNet

class SegDataset(Dataset):
    def __init__(self, root, size=(128,128)):
        self.root = Path(root)
        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'masks'
        self.samples = [p for p in self.img_dir.glob('*') if p.suffix.lower() in ['.jpg','.png','.jpeg']]
        self.size = size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p = self.samples[idx]
        img = Image.open(p).convert('RGB').resize(self.size)
        m = Image.open(self.mask_dir / p.name).convert('L').resize(self.size)
        img = np.array(img).transpose(2,0,1)/255.0
        m = (np.array(m)>127).astype('float32')[None,:,:]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SegDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    model = TinySwinUNet(in_ch=3, out_ch=1, img_size=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, masks in dl:
            imgs = imgs.to(device); masks = masks.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, masks)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss: {running/len(dl):.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'swin_unet_checkpoint.pt'))
    print("Saved checkpoint to", args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train(args)
