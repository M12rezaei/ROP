# CPU/GPU-FRIENDLY ROP STAGE CLASSIFICATION
# Using cached .pt files (images + masks + clinical data)
# Ensemble: convNeXt + EfficientNet + Ben Graham + Mask
# Includes MIL, Grad-CAM, ECE, Temp Scaling, metrics, patient aggregation
# ============================================================
import os
import argparse
import warnings
import json
import random
import hashlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split

import torchvision.transforms.functional as TF
import timm

import cv2
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")          # non-interactive backend, safe on all servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# REPRODUCIBILITY
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# UTILITIES
def tensor_hash(t: torch.Tensor) -> str:
    return hashlib.md5(t.cpu().numpy().tobytes()).hexdigest()


def remap_stage(stage) -> int:
    """Collapse Stages 2-5 → 2  (Normal=0, Mild=1, Severe=2)."""
    return min(int(stage), 2)


def to_ordinal(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer labels to cumulative binary threshold vectors.
    0 → [0,0]   1 → [1,0]   2 → [1,1]
    """
    out = torch.zeros((labels.size(0), num_classes - 1), device=labels.device)
    for i in range(labels.size(0)):
        out[i, :labels[i]] = 1
    return out


def ordinal_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Decode K-1 ordinal logits → K class probabilities with monotonicity."""
    cumulative = torch.sigmoid(logits)
    cumulative = torch.cummin(cumulative, dim=1)[0]   # enforce P(>=k) non-increasing

    B, K1 = cumulative.shape
    probs  = torch.zeros(B, K1 + 1, device=logits.device)
    probs[:, 0] = 1 - cumulative[:, 0]
    for k in range(1, K1):
        probs[:, k] = cumulative[:, k - 1] - cumulative[:, k]
    probs[:, -1] = cumulative[:, -1]
    return torch.clamp(probs, 1e-6, 1.0)


# MODULE-LEVEL LOSS
def compute_loss(
    logits:   torch.Tensor,
    targets:  torch.Tensor,
    weights:  torch.Tensor | None = None,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Ordinal BCE with optional per-sample class weighting and label smoothing.
    smoothing=0.1 → true label 1→0.95, 0→0.05
    """
    targets = targets.float() * (1 - smoothing) + 0.5 * smoothing
    loss    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    if weights is not None:
        # Recover integer class from smoothed ordinal vector
        true_ints    = targets.sum(dim=1).round().long()          # [F6] round after smoothing
        true_ints    = torch.clamp(true_ints, 0, NUM_CLASSES - 1)
        batch_weights = weights[true_ints].unsqueeze(1)
        loss          = loss * batch_weights

    return loss.mean()


# AUGMENTATION
def random_rotate(img: torch.Tensor) -> torch.Tensor:
    return TF.rotate(img, float(np.random.uniform(-15, 15)))

def color_jitter(img: torch.Tensor) -> torch.Tensor:
    return torch.clamp(img * np.random.uniform(0.8, 1.2), 0.0, 1.0)

def gaussian_noise(img: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * std, 0.0, 1.0)

def horizontal_flip(img: torch.Tensor) -> torch.Tensor:
    return img.flip(-1) if img.dim() == 3 else img

def random_gamma(img: torch.Tensor) -> torch.Tensor:
    return torch.clamp(img ** np.random.uniform(0.8, 1.2), 0.0, 1.0)

def apply_clahe(img: torch.Tensor) -> torch.Tensor:
    """CLAHE in LAB space; operates on a clone — does not mutate input."""
    img_np  = (img[:3].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    lab     = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    planes  = list(cv2.split(lab))
    planes[0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(planes[0])
    lab     = cv2.merge(planes)
    img_np  = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    out     = img.clone()
    out[:3] = torch.tensor(img_np.transpose(2, 0, 1), device=img.device)
    return out

# DATASET
class ROPDatasetPT(Dataset):
    def __init__(self, pt_paths: list, augment: bool = False):
        self.pt_paths = pt_paths
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.pt_paths)

    def __getitem__(self, idx: int):
        data = torch.load(self.pt_paths[idx], map_location="cpu")

        img = data["img"].float()

        # Resize if cached at wrong resolution
        if img.shape[1:] != (224, 224):
            img = F.interpolate(
                img.unsqueeze(0), (224, 224),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # Ensure 4 channels (RGB + vessel mask)
        if img.shape[0] == 3:
            img = torch.cat([img, torch.zeros(1, 224, 224)], dim=0)

        # Augmentation — training only
        if self.augment:
            if np.random.rand() < 0.2:
                img = apply_clahe(img)
            if np.random.rand() < 0.5:
                img = horizontal_flip(img)
            if np.random.rand() < 0.5:
                img = random_rotate(img)
            if np.random.rand() < 0.4:
                img = color_jitter(img)
            if np.random.rand() < 0.3:
                img = gaussian_noise(img)
            if np.random.rand() < 0.3:
                img = random_gamma(img)

        # Clinical features — normalised with fixed physiological bounds
        ga = float(np.clip(data.get("ga", 30), 22, 40))
        bw = float(np.clip(data.get("bw", 1200), 400, 2500))
        clinical = torch.tensor(
            [(ga - 22) / 18.0, (bw - 400) / 2100.0],
            dtype=torch.float32
        )

        # Label, ordinal encoding
        raw_label = int(data["label"])
        target    = to_ordinal(torch.tensor([raw_label]), NUM_CLASSES).squeeze(0)

        return img, clinical, target, str(data["patient_id"])

# MODEL: ROPNet
class ROPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone1 = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True, in_chans=4)
        self.backbone2 = timm.create_model(
            "efficientnet_b3", pretrained=True, features_only=True, in_chans=4)

        with torch.no_grad():
            dummy = torch.randn(1, 4, 224, 224)
            f1_ch = self.backbone1(dummy)[-1].shape[1]
            f2_ch = self.backbone2(dummy)[-1].shape[1]

        self.conv_head1 = nn.Conv2d(f1_ch, 256, kernel_size=1)
        self.conv_head2 = nn.Conv2d(f2_ch, 256, kernel_size=1)

        self.attn1 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1))
        self.attn2 = nn.Sequential(
            nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 1, 1))

        self.clinical_fc = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Dropout(0.2))

        self.norm_img1 = nn.LayerNorm(256)
        self.norm_img2 = nn.LayerNorm(256)
        self.norm_clin = nn.LayerNorm(64)

        self.dropout      = nn.Dropout(0.3)
        self.ordinal_head = nn.Linear(256 + 256 + 64, NUM_CLASSES - 1)

    def attention_pool(self, f: torch.Tensor, attn_layer: nn.Module) -> torch.Tensor:
        attn = attn_layer(f)
        B, _, H, W = attn.shape
        attn = attn.view(B, -1)
        attn = attn - attn.max(dim=1, keepdim=True)[0]   # numerical stability
        attn = torch.softmax(attn, dim=1).view(B, 1, H, W)
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        return (f * attn).sum(dim=(2, 3))

    def forward(self, x: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        f1 = self.norm_img1(self.attention_pool(self.conv_head1(self.backbone1(x)[-1]), self.attn1))
        f2 = self.norm_img2(self.attention_pool(self.conv_head2(self.backbone2(x)[-1]), self.attn2))
        c  = self.norm_clin(self.clinical_fc(clinical))
        return self.ordinal_head(self.dropout(torch.cat([f1, f2, c], dim=1)))


# TEMPERATURE SCALING
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def set_temperature(self, logits_ord: torch.Tensor, labels: torch.Tensor) -> None:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        labels    = labels.long()

        def closure():
            optimizer.zero_grad()
            probs = torch.clamp(
                ordinal_probs_from_logits(logits_ord / self.temperature), 1e-6, 1.0)
            loss  = F.nll_loss(torch.log(probs), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"Optimal temperature: {self.temperature.item():.3f}")


# ECE
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs, labels   = np.asarray(probs), np.asarray(labels)
    confidences     = probs.max(axis=1)
    predictions     = probs.argmax(axis=1)
    ece             = 0.0
    bins            = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi   = bins[i], bins[i + 1]
        in_bin   = (confidences >= lo) & (confidences <= hi) if i == 0 else \
                   (confidences > lo)  & (confidences <= hi)
        prop     = in_bin.mean()
        if prop > 0:
            ece += prop * abs(
                (predictions[in_bin] == labels[in_bin]).mean() -
                confidences[in_bin].mean()
            )
    return float(ece)


# GRAD-CAM
class GradCAM:
    def __init__(self, model: ROPNet):
        self.model = model
        self.grad  = None
        self.act   = None
        self.fwd_handle = model.conv_head1.register_forward_hook(self._fwd)
        self.bwd_handle = model.conv_head1.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, output):           self.act  = output
    def _bwd(self, _, __, grad_out):         self.grad = grad_out[0]

    def generate(self, x: torch.Tensor, clinical: torch.Tensor, cls: int) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        probs = ordinal_probs_from_logits(self.model(x, clinical))
        probs[:, cls].sum().backward()

        if self.grad is None or self.act is None:
            raise RuntimeError("GradCAM hooks did not fire.")

        weights = self.grad.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.act).sum(1))
        cam     = cam - cam.min()
        cam     = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()

    def remove_hooks(self) -> None:
        self.fwd_handle.remove()
        self.bwd_handle.remove()

# DUPLICATE DETECTION — now returns list for caller to act on
def check_duplicates(train_paths: list, val_paths: list) -> list:
    """
    Hash-based duplicate detection across train/val split.
    Returns list of (val_path, train_path) pairs that are duplicates.
    NOTE: these are within-patient duplicates (same patient, different sessions).
    They do not constitute patient-level leakage but may inflate validation
    estimates slightly.
    """
    print("\n[CHECK] Duplicate image detection...")
    train_hashes = {}
    for p in train_paths:
        try:
            h = tensor_hash(torch.load(p, map_location="cpu")["img"])
            train_hashes[h] = p
        except Exception as e:
            print(f"[WARNING] Could not hash {p}: {e}")

    duplicates = []
    for p in val_paths:
        try:
            h = tensor_hash(torch.load(p, map_location="cpu")["img"])
            if h in train_hashes:
                duplicates.append((p, train_hashes[h]))
        except Exception as e:
            print(f"[WARNING] Could not hash {p}: {e}")

    print(f"Duplicates found: {len(duplicates)}")
    if duplicates:
        print(f"Example: {duplicates[0]}")
        print("[INFO] These are within-fold duplicates (same patient, different sessions).")
        print("[INFO] No patient-level leakage — but may marginally inflate val estimates.")

    return duplicates

# PATIENT-LEVEL AGGREGATION
def aggregate_patient_metrics(df_val: pd.DataFrame, stage_probs: np.ndarray) -> float:
    df_val = df_val.copy()
    df_val["stage_pred_prob"] = list(stage_probs)
    preds, gts = [], []
    for _, g in df_val.groupby("patient_id"):
        preds.append(np.argmax(np.stack(g["stage_pred_prob"].values).mean(axis=0)))
        gts.append(g["label"].iloc[0])
    return accuracy_score(gts, preds)


def patient_level_auc(df_val: pd.DataFrame, stage_probs: np.ndarray) -> float:
    df_val = df_val.copy()
    df_val["probs"] = list(np.asarray(stage_probs))
    patient_probs, patient_labels = [], []
    for _, g in df_val.groupby("patient_id"):
        patient_probs.append(np.stack(g["probs"].values).mean(axis=0))
        patient_labels.append(g["label"].iloc[0])

    patient_probs  = np.stack(patient_probs)
    patient_labels = np.array(patient_labels)

    if len(np.unique(patient_labels)) < 2:
        print("[WARNING] AUC skipped: only one class present in validation.")
        return np.nan
    try:
        return roc_auc_score(patient_labels, patient_probs,
                             multi_class="ovr", average="macro")
    except Exception as e:
        print(f"AUC error: {e}")
        return np.nan


# SENSITIVITY THRESHOLDS (95% recall target on Severe class)
def sensitivity_thresholds(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    target_sens: float = 0.95,
    critical_stages: list = [2],
) -> dict:
    thresholds = {}
    for stage in critical_stages:
        y_bin = (y_true >= stage).astype(int)
        _, tpr, ths = roc_curve(y_bin, y_probs[:, stage])
        thresholds[stage] = ths[np.argmin(np.abs(tpr - target_sens))] if len(ths) else 0.5
    return thresholds


def patient_sensitivity_flags(
    df_val: pd.DataFrame,
    stage_probs: np.ndarray,
    thresholds: dict,
    critical_stages: list = [2],
) -> pd.DataFrame:
    df_val      = df_val.copy()
    stage_probs = np.asarray(stage_probs)
    df_val["probs"] = list(stage_probs)
    rows = []
    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["probs"].values)
        flag  = int(any(
            (probs[:, s] >= thresholds.get(s, 0.5)).any()
            for s in critical_stages
        ))
        rows.append({"patient_id": pid, "gt_stage": g["label"].iloc[0], "flag_critical": flag})
    return pd.DataFrame(rows)


# PLOTTING HELPERS
def plot_confusion(y_true, y_pred, classes, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_pr(stage_true, stage_probs, out_dir: str) -> None:
    for s in range(NUM_CLASSES):
        y_bin = (stage_true == s).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue

        fpr, tpr, _          = roc_curve(y_bin, stage_probs[:, s])
        precision, recall, _ = precision_recall_curve(y_bin, stage_probs[:, s])

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — Stage {s}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"roc_stage{s}.png"))
        plt.close()

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — Stage {s}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pr_stage{s}.png"))
        plt.close()


def plot_random_samples(pt_paths: list, out_dir: str, num_per_class: int = 5) -> None:
    """[F3] Saves sample images to disk instead of calling plt.show()."""
    class_samples: dict = {0: [], 1: [], 2: []}
    for p in pt_paths:
        data  = torch.load(p, map_location="cpu")
        label = int(data["label"])
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append(data["img"][:3])
        if all(len(v) >= num_per_class for v in class_samples.values()):
            break

    for cls, imgs in class_samples.items():
        if not imgs:
            continue
        plt.figure(figsize=(10, 5))
        for i, img in enumerate(imgs):
            plt.subplot(1, num_per_class, i + 1)
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            plt.imshow(img_np); plt.axis("off")
        plt.suptitle(f"Class {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"debug_samples_class{cls}.png"))
        plt.close()


# HELPER TO FILTER PT PATHS BASED ON DF SUBSET (e.g., train/val split)
def filter_paths(df_subset: pd.DataFrame, path_to_pid: dict) -> list:
    allowed = set(df_subset["patient_id"].astype(str))
    return [p for p in path_to_pid if path_to_pid[p] in allowed]


# GRAD-CAM VISUALISATION (post fold)
def visualise_gradcam(model, val_loader, preds, stage_true, out_dir: str) -> None:
    gradcam     = GradCAM(model)
    gradcam_dir = os.path.join(out_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    mis_idx = np.where(preds != stage_true)[0][:5]

    for idx in mis_idx:
        img, clin, label, pid = val_loader.dataset[idx]
        img  = img.unsqueeze(0).to(DEVICE)
        clin = clin.unsqueeze(0).to(DEVICE)
        cam  = gradcam.generate(img, clin, int(preds[idx]))[0]

        img_np = img[0, :3].cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np - img_np.min()) / (img_np.max() + 1e-6)
        cam    = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        # Circular hard mask
        h, w   = cam.shape
        y, xg  = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        hard   = ((xg - cx) ** 2 + (y - cy) ** 2 <= (min(cy, cx) * 0.95) ** 2).astype(np.float32)

        # Soft image-based mask
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, soft = cv2.threshold(gray, 0.05, 1, cv2.THRESH_BINARY)
        soft = cv2.GaussianBlur(soft.astype(np.float32), (31, 31), 0)
        soft = soft / (soft.max() + 1e-6)

        cam = cv2.GaussianBlur(
            cv2.medianBlur(
                (cam * hard * soft * 255).astype(np.uint8), 9
            ) / 255.0,
            (21, 21), 0
        )
        cam = (cam - cam.min()) / (cam.max() + 1e-6) * hard * soft

        heat    = cv2.cvtColor(
            cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET),
            cv2.COLOR_BGR2RGB
        )
        overlay = np.clip(0.6 * img_np + 0.4 * heat / 255.0, 0, 1)

        plt.figure()
        plt.imshow(overlay)
        plt.title(f"GT:{label.argmax().item()}  Pred:{int(preds[idx])}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(gradcam_dir, f"{pid}_{idx}.png"))
        plt.close()

    gradcam.remove_hooks()

# TRAIN FOLD
def train_fold(
    df: pd.DataFrame,
    pt_dir: str,
    fold: int,
    args,
    path_to_pid: dict,
    class_weights: torch.Tensor | None = None,
    patience: int = 5,
) -> None:

    out = os.path.join(args.out_dir, f"fold_{fold}")
    os.makedirs(out, exist_ok=True)

    train_paths = filter_paths(df[df["fold"] != fold], path_to_pid)
    val_paths   = filter_paths(df[df["fold"] == fold],  path_to_pid)

    # Duplicate check, log but do not remove (within-fold, not leakage)
    duplicates = check_duplicates(train_paths, val_paths)

    if duplicates:
        dup_val = set([v for v, _ in duplicates])
        before = len(val_paths)
        val_paths = [p for p in val_paths if p not in dup_val]
        print(f"[CLEAN] Removed {before - len(val_paths)} duplicate images from validation")

    # Sampler
    train_labels = np.array([
        df[df["patient_id"] == path_to_pid[p]]["label"].values[0]
        for p in train_paths
    ])
    class_count = np.bincount(train_labels, minlength=NUM_CLASSES)

    # Use 1/sqrt to match the loss weighting strategy
    sw = 1.0 / np.sqrt(class_count + 1e-6)
    sw = (sw / sw.sum()) * NUM_CLASSES
    #sampler = WeightedRandomSampler(sw[train_labels], len(train_labels), replacement=True)

    # DataLoaders
    train_loader = DataLoader(
        ROPDatasetPT(train_paths, augment=True),
        batch_size=args.batch_size, #sampler=sampler,
        shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        ROPDatasetPT(val_paths, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Model / optimiser
    model     = ROPNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # Debug sample visualisation (fold 0, epoch 1 only)
    if fold == 0:
        plot_random_samples(train_paths, out)

    # Training loop
    best_qwk, best_epoch, no_improve = -1.0, 0, 0
    best_metrics: dict = {}
    history = {k: [] for k in
               ["train_loss", "val_loss", "img_acc", "f1", "qwk", "ece", "pat_acc", "pat_auc"]}

    for epoch in range(1, args.epochs + 1):

        # Train
        model.train()
        train_loss = 0.0

        for imgs, clin, targets, _ in tqdm(train_loader, desc=f"[Fold {fold}] Ep {epoch}"):
            imgs, clin, targets = imgs.to(DEVICE), clin.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(imgs, clin)
                # Pass class_weights to training loss
                loss   = compute_loss(logits, targets, class_weights)

            amp_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_preds, all_probs, all_true, all_pids = [], [], [], []

        with torch.no_grad():
            for imgs, clin, targets, pids in val_loader:
                imgs, clin, targets = imgs.to(DEVICE), clin.to(DEVICE), targets.to(DEVICE)
                logits = model(imgs, clin)
                val_loss += compute_loss(logits, targets, class_weights).item()

                probs = ordinal_probs_from_logits(logits)
                all_preds.append(probs.argmax(dim=1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_true.append(targets.sum(dim=1).long().cpu().numpy())
                all_pids.extend(pids)

        val_loss /= len(val_loader)
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs)

        img_acc = accuracy_score(y_true, y_pred)
        f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
        qwk     = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ece_val = compute_ece(y_prob, y_true)

        df_val_img = pd.DataFrame({"patient_id": all_pids, "label": y_true})
        pat_acc    = aggregate_patient_metrics(df_val_img, y_prob)
        pat_auc    = patient_level_auc(df_val_img, y_prob)

        for k, v in zip(history, [train_loss, val_loss, img_acc, f1, qwk, ece_val, pat_acc, pat_auc]):
            history[k].append(v)

        print(f"[Fold {fold}][Ep {epoch}] "
              f"Train={train_loss:.3f} Val={val_loss:.3f} "
              f"Acc={img_acc:.3f} F1={f1:.3f} QWK={qwk:.3f} "
              f"PatAcc={pat_acc:.3f} AUC={pat_auc:.3f} ECE={ece_val:.3f}")

        # Early stopping on QWK
        if qwk > best_qwk:
            best_qwk, best_epoch, no_improve = qwk, epoch, 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
            best_metrics = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                img_acc=img_acc, f1=f1, qwk=qwk, ece=ece_val,
                                pat_acc=pat_acc, pat_auc=pat_auc)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save artefacts 
    for fname, obj in [("history.json", history),
                        ("metrics_best.json", best_metrics),
                        ("metrics_last.json", {k: history[k][-1] for k in history})]:
        with open(os.path.join(out, fname), "w") as fh:
            json.dump(obj, fh, indent=4)

    pd.DataFrame({**history, "epoch": np.arange(1, len(history["qwk"]) + 1)}) \
      .to_csv(os.path.join(out, "metrics.csv"), index=False)

    print(f"\n Best Epoch: {best_epoch} | Best QWK: {best_qwk:.4f}")

    # Final evaluation pass with best weights 
    model.load_state_dict(torch.load(os.path.join(out, "best.pt")))
    model.eval()

    stage_probs, stage_true, preds_all, stage_pids = [], [], [], []
    with torch.no_grad():
        for imgs, clin, targets, pids in val_loader:
            imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)
            probs = ordinal_probs_from_logits(model(imgs, clin))
            stage_probs.append(probs.cpu().numpy())
            stage_true.append(targets.sum(dim=1).long().cpu().numpy())
            preds_all.append(probs.argmax(dim=1).cpu().numpy())
            stage_pids.extend(pids)

    stage_probs = np.concatenate(stage_probs)
    stage_true  = np.concatenate(stage_true)
    preds       = np.concatenate(preds_all)

    thresholds = sensitivity_thresholds(stage_true, stage_probs, target_sens=0.95)
    print("Sensitivity thresholds:", thresholds)

    df_flags = patient_sensitivity_flags(
        pd.DataFrame({"patient_id": stage_pids, "label": stage_true}),
        stage_probs, thresholds,
    )
    df_flags.to_csv(os.path.join(out, "patient_flags.csv"), index=False)

    plot_confusion(stage_true, preds, list(range(NUM_CLASSES)),
                   os.path.join(out, "confusion_best.png"))
    plot_roc_pr(stage_true, stage_probs, out)
    visualise_gradcam(model, val_loader, preds, stage_true, out)

# FINAL TRAINING  (full dataset)
def final_training(
    df: pd.DataFrame,
    args,
    path_to_pid: dict,
    class_weights: torch.Tensor | None = None,
) -> None:
    print("\n=== FINAL TRAINING ON FULL DATASET ===")

    all_paths = filter_paths(df, path_to_pid)

    # Hold out 15% for calibration BEFORE training begins
    # Calibration set is never seen during training.
    n_total  = len(all_paths)
    n_calib  = max(1, int(0.15 * n_total))
    n_train  = n_total - n_calib

    rng = torch.Generator().manual_seed(42)
    # Patient-level split (NO LEAKAGE)
    all_patients = df["patient_id"].astype(str).unique()
    rng_np = np.random.RandomState(42)
    rng_np.shuffle(all_patients)

    n_calib_patients = max(1, int(0.15 * len(all_patients)))
    calib_patient_ids = set(all_patients[:n_calib_patients])

    train_paths_final = [
        p for p in all_paths if path_to_pid[p] not in calib_patient_ids
    ]
    calib_paths_final = [
        p for p in all_paths if path_to_pid[p] in calib_patient_ids
    ]

    print(f"Final training: {len(train_paths_final)} images | "
          f"Calibration hold-out: {len(calib_paths_final)} images | "
          f"Patients (calib): {len(calib_patient_ids)}")

    print(f"Final training: {len(train_paths_final)} images | "
          f"Calibration hold-out: {len(calib_paths_final)} images")

    train_loader = DataLoader(
        ROPDatasetPT(train_paths_final, augment=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    calib_loader = DataLoader(
        ROPDatasetPT(calib_paths_final, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model     = ROPNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for imgs, clin, stage, _ in tqdm(train_loader, desc=f"Final Train Ep {epoch}"):
            imgs, clin, stage = imgs.to(DEVICE), clin.to(DEVICE), stage.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                loss = compute_loss(model(imgs, clin), stage, class_weights)
            amp_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"[Final][Ep {epoch}] Loss={epoch_loss / len(train_loader):.4f}")

    # Temperature scaling on the held-out calibration set
    # Uses calib_loader (unseen data), NOT the training set
    print("\nApplying temperature scaling on held-out calibration set...")
    model.eval()
    logits_all, labels_all = [], []

    with torch.no_grad():
        for imgs, clin, stage, _ in calib_loader:
            imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)
            logits_all.append(model(imgs, clin).cpu())
            labels_all.append(stage.sum(dim=1).long().cpu())

    logits_all  = torch.cat(logits_all).to(DEVICE)
    labels_all  = torch.cat(labels_all).to(DEVICE)

    temp_scaler = TempScaler().to(DEVICE)
    temp_scaler.set_temperature(logits_all, labels_all)

    # Verify calibration on the held-out set
    with torch.no_grad():
        calib_probs = ordinal_probs_from_logits(
            logits_all / temp_scaler.temperature
        ).cpu().numpy()
    calib_ece = compute_ece(calib_probs, labels_all.cpu().numpy())
    print(f" Calibration ECE (held-out): {calib_ece:.4f}")
    print(f" Final temperature: {temp_scaler.temperature.item():.3f}")

    torch.save(
        {"model": model.state_dict(),
         "temperature": temp_scaler.temperature.item()},
        os.path.join(args.out_dir, "final_stage_model.pt"),
    )
    print(" Final model saved.")

# =========================
# MAIN
# =========================
def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle_labels", action="store_true")
    parser.add_argument("--pt_dir", required=True, help="Directory with cached .pt files")
    parser.add_argument("--csv_file", required=True, help="CSV with patient_id, label for folds")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out_dir", default="./results_final")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("\n=========================")
    print("ROP TRAINING STARTED")
    print("=========================\n")

    # Load and clean CSV
    df = pd.read_csv(args.csv_file).dropna()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["label"]      = df["label"].apply(remap_stage)
    print(f"Loaded {len(df)} samples")

    if args.shuffle_labels:
        print("\n[WARNING] Shuffling labels for sanity check...")
        df["label"] = np.random.permutation(df["label"].values)

    # Load cached .pt paths
    cache_file = os.path.join(args.out_dir, "pt_cache_paths.pt")
    if os.path.exists(cache_file):
        print("\n[CACHE] Loading paths...")
        all_paths = torch.load(cache_file)
    else:
        all_paths = [
            os.path.join(args.pt_dir, f)
            for f in os.listdir(args.pt_dir)
            if f.endswith(".pt") and "index" not in f.lower()
        ]
        torch.save(all_paths, cache_file)

    print(f"[INFO] Total images: {len(all_paths)}")

    # Build path, patient map and filter out unreadable files
    path_to_pid: dict = {}
    valid_paths: list = []
    print("\n[INFO] Mapping patient IDs...")

    for p in tqdm(all_paths):
        try:
            data = torch.load(p, map_location="cpu")
            path_to_pid[p] = str(data["patient_id"]).strip()
            valid_paths.append(p)
        except Exception as e:
            print(f"[WARNING] Skipped unreadable file {p}: {e}")

    all_paths = valid_paths
    print(f"[INFO] Valid images: {len(all_paths)}")

    # Patient-level stratified fold assignment
    patient_df = df[["patient_id", "label"]].drop_duplicates().copy()
    skf        = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    patient_df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        skf.split(patient_df["patient_id"], patient_df["label"])
    ):
        pids = patient_df.iloc[val_idx]["patient_id"].values
        patient_df.loc[patient_df["patient_id"].isin(pids), "fold"] = fold_idx

    df = df.merge(patient_df[["patient_id", "fold"]], on="patient_id", how="left")

    # Leakage check (raises on any overlap)
    for f in range(args.folds):
        overlap = set(df[df.fold != f]["patient_id"]) & set(df[df.fold == f]["patient_id"])
        if overlap:
            raise ValueError(f"[LEAKAGE] Fold {f}: {len(overlap)} patients overlap!")
    print("No patient-level leakage detected")

    # Cross-validation
    for f in range(args.folds):
        print(f"\n====================\nFOLD {f} START\n====================")

        train_df    = df[df.fold != f]
        train_paths = [p for p in all_paths if path_to_pid[p] in set(train_df["patient_id"])]
        val_paths   = [p for p in all_paths if path_to_pid[p] in set(df[df.fold == f]["patient_id"])]

        print(f"Train images: {len(train_paths)} | Val images: {len(val_paths)}")

        class_counts = (
            train_df["label"].value_counts().sort_index()
            .reindex(range(NUM_CLASSES), fill_value=0).values
        )
        w = 1.0 / np.sqrt(class_counts + 1e-6)
        w = (w / w.sum()) * NUM_CLASSES
        class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)

        print(f"Class counts: {class_counts}")
        print(f"Class weights: {w}")

        try:
            train_fold(df, args.pt_dir, f, args, path_to_pid, class_weights)
        except Exception as e:
            print(f"[ERROR] Fold {f} failed: {e}")

    # Final training
    print("\n=========================\nFINAL TRAINING (FULL DATA)\n=========================\n")

    counts = (
        df["label"].value_counts().sort_index()
        .reindex(range(NUM_CLASSES), fill_value=0).values
    )
    w = 1.0 / np.sqrt(counts + 1e-6)
    w = (w / w.sum()) * NUM_CLASSES
    class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    print(f"Final class weights: {w}")

    final_training(df, args, path_to_pid, class_weights)
    print("\nTRAINING COMPLETE")
if __name__ == "__main__":
    main()