"""
QMViT training stub.

Usage:
    python qmvit_train.py --data_dir /path/to/data --output_dir ./outputs --task stages --epochs 5

Expects dataset organized as in README with folders for stages/ and plus/
This script trains the QMViT model (or fallback if PennyLane not installed) for classification.
"""
import argparse, os, time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from models.qmvit import QMViT

class SimpleClsDataset(Dataset):
    def __init__(self, root, task='stages', size=(128,128), csv_path=None):
        self.root = Path(root)
        self.task = task
        self.size = size
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.samples = []
        # If a CSV is provided, read filepaths and labels from it (expects columns: filepath, stage, plus)
        if self.csv_path is not None and self.csv_path.exists():
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            # Choose column depending on task
            label_col = 'stage' if task=='stages' else 'plus'
            for _,row in df.iterrows():
                fp = Path(root)/'images'/row['filepath'] if (Path(root)/'images'/str(row['filepath'])).exists() else Path(root)/str(row['filepath'])
                if fp.exists():
                    self.samples.append((str(fp), int(row[label_col])))
        else:
            # fallback to folder structure
            base = self.root / task
            if base.exists():
                self.classes = sorted([p.name for p in base.iterdir() if p.is_dir()])
                for i,c in enumerate(self.classes):
                    for img in (base/c).glob('*'):
                        if img.suffix.lower() in ['.jpg','.png','.jpeg']:
                            self.samples.append((str(img), i))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p,label = self.samples[idx]
        img = Image.open(p).convert('RGB').resize(self.size)
        arr = np.array(img).transpose(2,0,1)/255.0
        return torch.tensor(arr, dtype=torch.float32), int(label)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SimpleClsDataset(args.data_dir, task=args.task, csv_path=args.csv)
    if len(ds)==0:
        print("No data found for task", args.task); return
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    num_classes = len(ds.classes)
    model = QMViT(num_classes=num_classes, input_size=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for imgs, labels in dl:
            imgs = imgs.to(device); labels = labels.to(device)
            out = model(imgs)
            loss = loss_fn(out, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss: {running/len(dl):.4f}")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'qmvit_checkpoint.pt'))
    print("Saved QMViT checkpoint to", args.output_dir)
    print("Classes:", ds.classes)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--csv', required=False, help='Path to CSV file with columns: filepath, stage, plus')
    parser.add_argument('--output_dir', default='./outputs')
    parser.add_argument('--task', choices=['stages','plus'], default='stages')
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    train(args)
