# CPU/GPU-FRIENDLY ROP STAGE CLASSIFICATION
# Using cached .pt files (images + masks + clinical data)
# Ensemble: MaxViT + EfficientNet + Ben Graham + Mask
# Includes MIL, Grad-CAM, ECE, Temp Scaling, metrics, patient aggregation
# ============================================================

import os, argparse, warnings
from collections import defaultdict

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
from torch.utils.data import WeightedRandomSampler
import json
from tqdm import tqdm
import cv2


warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

# --------------------------
# Rotations
# --------------------------
def random_rotate(img):
    angle = np.random.uniform(-15, 15)  # degrees
    return TF.rotate(img, angle)

def color_jitter(img):
    factor = np.random.uniform(0.8, 1.2)
    img = img * factor
    return torch.clamp(img, 0.0, 1.0)


def gaussian_noise(img, mean=0.0, std=0.01):
    noise = torch.randn_like(img) * std
    img = img + noise
    return torch.clamp(img, 0.0, 1.0)


def horizontal_flip(img):
    return img.flip(-1) if img.dim() == 3 else img

# CLAHE (Contrast Limited Adaptive Histogram Equalization)

def apply_clahe(img):
    img_np = img[:3].cpu().numpy().transpose(1,2,0)
    img_np = (img_np * 255).astype(np.uint8)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab_planes = list(cv2.split(lab))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_np = img_np.astype(np.float32) / 255.0

    img_new = img.clone()
    img_new[:3] = torch.tensor(img_np.transpose(2,0,1), device=img.device)

    return img_new

def random_gamma(img):
    gamma = np.random.uniform(0.8, 1.2)
    return torch.clamp(img ** gamma, 0, 1)

def remap_stage(stage):
    # Maps any stage >= 2 to 2 (Normal=0, Mild=1, Severe+=2)
    return min(int(stage), 2)

def to_ordinal(labels, num_classes):
    # Converts label 1 to [1, 0], label 2 to [1, 1], label 0 to [0, 0]
    out = torch.zeros((labels.size(0), num_classes - 1), device=labels.device)
    for i in range(labels.size(0)):
        out[i, :labels[i]] = 1
    return out
# ========================
# Seed for reproducibility
# ========================
def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Dataset using cached .pt files
# =========================
class ROPDatasetPT(Dataset):
    def __init__(self, pt_paths, augment=False):
        self.pt_paths = pt_paths
        self.augment = augment

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, idx):
        # -------------------------
        # Load cached .pt
        # -------------------------
        data = torch.load(self.pt_paths[idx], map_location="cpu")

        # -------------------------
        # Image
        # -------------------------
        img = data["img"].float()

        # Resize if needed
        if img.shape[1:] != (224, 224):
            img = F.interpolate(
                img.unsqueeze(0),
                (224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Ensure 4 channels (RGB + mask)
        if img.shape[0] == 3:
            img = torch.cat([img, torch.zeros(1, 224, 224)], dim=0)

        # -------------------------
        # AUGMENTATION (TRAIN ONLY)
        # -------------------------
        if self.augment:

            # ---- CLAHE (low probability, improves vessels) ----
            if np.random.rand() < 0.2:
                img = apply_clahe(img)

            # ---- Horizontal flip ----
            if np.random.rand() < 0.5:
                img = horizontal_flip(img)

            # ---- Rotation (small angles only) ----
            if np.random.rand() < 0.5:
                img = random_rotate(img)

            # ---- Brightness jitter ----
            if np.random.rand() < 0.4:
                img = color_jitter(img)

            # ---- Gaussian noise ----
            if np.random.rand() < 0.3:
                img = gaussian_noise(img, std=0.02)

            if np.random.rand() < 0.3:
              img = random_gamma(img)

        # =========================
        # NORMALIZATION (ADD HERE)
        # =========================
        mean = torch.tensor([0.485, 0.456, 0.406, 0.0], device=img.device).view(4,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225, 1.0], device=img.device).view(4,1,1)

        img = (img - mean) / std

        # -------------------------
        # Clinical Data
        # -------------------------
        ga = np.clip(data.get("ga", 30), 22, 40)
        bw = np.clip(data.get("bw", 1200), 400, 2500)

        clinical = torch.tensor([
            (ga - 22) / 18,
            (bw - 400) / 2100
        ], dtype=torch.float32)

        # -------------------------
        # Label → Ordinal
        # -------------------------
        raw_label = int(data["label"])
        target = to_ordinal(
            torch.tensor([raw_label]),
            NUM_CLASSES
        ).squeeze(0)

        return img, clinical, target, str(data["patient_id"])

# =========================
# Ensemble Model
# =========================
class ROPNet(nn.Module):
    def __init__(self):
        super().__init__()

        # =========================
        # Backbones (feature extractors)
        # =========================
        self.backbone1 = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True,
            in_chans=4
        )

        self.backbone2 = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            features_only=True,
            in_chans=4
        )

        # ---- infer feature dims safely ----
        with torch.no_grad():
            dummy = torch.randn(1, 4, 224, 224)
            f1_ch = self.backbone1(dummy)[-1].shape[1]
            f2_ch = self.backbone2(dummy)[-1].shape[1]

        # =========================
        # Projection heads
        # =========================
        self.conv_head1 = nn.Conv2d(f1_ch, 256, kernel_size=1)
        self.conv_head2 = nn.Conv2d(f2_ch, 256, kernel_size=1)

        # =========================
        # Attention modules
        # =========================
        self.attn1 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        self.attn2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

        # =========================
        # Clinical branch
        # =========================
        self.clinical_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # =========================
        # Normalization (IMPORTANT FIX)
        # =========================
        self.norm_img1 = nn.LayerNorm(256)
        self.norm_img2 = nn.LayerNorm(256)
        self.norm_clin = nn.LayerNorm(64)

        # =========================
        # Fusion head
        # =========================
        self.dropout = nn.Dropout(0.3)

        self.ordinal_head = nn.Linear(256 + 256 + 64, NUM_CLASSES - 1)

    # =========================
    # Stable attention pooling
    # =========================
    def attention_pool(self, f, attn_layer):
        """
        f: (B, C, H, W)
        """
        attn = attn_layer(f)  # (B,1,H,W)

        B, _, H, W = attn.shape

        attn = attn.view(B, -1)

        # stable softmax
        attn = attn - attn.max(dim=1, keepdim=True)[0]
        attn = torch.softmax(attn, dim=1)

        attn = attn.view(B, 1, H, W)

        # normalize attention (important fix)
        attn = attn / (attn.sum(dim=(2,3), keepdim=True) + 1e-6)

        f = (f * attn).sum(dim=(2,3))  # weighted pooling

        return f

    # =========================
    # Forward
    # =========================
    def forward(self, x, clinical):

        # -------- Backbone 1 --------
        f1 = self.backbone1(x)[-1]
        f1 = self.conv_head1(f1)
        f1 = self.attention_pool(f1, self.attn1)
        f1 = self.norm_img1(f1)

        # -------- Backbone 2 --------
        f2 = self.backbone2(x)[-1]
        f2 = self.conv_head2(f2)
        f2 = self.attention_pool(f2, self.attn2)
        f2 = self.norm_img2(f2)

        # -------- Clinical --------
        c = self.clinical_fc(clinical)
        c = self.norm_clin(c)

        # -------- Fusion --------
        fused = torch.cat([f1, f2, c], dim=1)
        fused = self.dropout(fused)

        logits_ord = self.ordinal_head(fused)

        return logits_ord
# =========================
# ECE
# =========================
def compute_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            acc_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()

            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin

    return float(ece)

# =========================
# Temperature Scaling
# =========================
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


    def set_temperature(self, logits_ord, labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        labels = labels.long()

        def eval():
            optimizer.zero_grad()

            scaled_logits = logits_ord / self.temperature

            # convert ordinal → class logits (NOT probs)
            probs = ordinal_probs_from_logits(scaled_logits)

            probs = torch.clamp(probs, 1e-6, 1.0)  # stability

            loss = F.nll_loss(torch.log(probs), labels)

            loss.backward()
            return loss

        optimizer.step(eval)

        print(f"Optimal temperature: {self.temperature.item():.3f}")

def apply_temperature_scaling(logits_ord, labels):
    scaler = TempScaler().to(DEVICE)

    logits_ord = logits_ord.detach().to(DEVICE)
    labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)

    scaler.set_temperature(logits_ord, labels)

    scaled_logits = scaler(logits_ord)
    scaled_probs = ordinal_probs_from_logits(scaled_logits)

    return scaled_probs.cpu().numpy(), scaler.temperature.item()

# =========================
# Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        # ---- safer hook target ----
        self.target_layer = self._find_last_conv_layer(model)

        self.fwd_handle = self.target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self.backward_hook)

    def _find_last_conv_layer(self, model):
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise ValueError("No Conv2d layer found for GradCAM")

    def forward_hook(self, module, input, output):
        self.act = output

    def backward_hook(self, module, grad_in, grad_out):
        self.grad = grad_out[0]

    def generate(self, x, clinical, cls):
        self.model.zero_grad(set_to_none=True)

        logits_ord = self.model(x, clinical)
        probs = ordinal_probs_from_logits(logits_ord)

        # stable class score
        score = probs[:, cls]

        score.sum().backward()

        weights = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.act).sum(1)

        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-6)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()
# =========================
# Aggregate patient-level metrics
# =========================
def aggregate_patient_metrics(df_val, stage_probs):
    """
    df_val: validation dataframe (one row per image)
    stage_probs: np.array or list of shape (N_images, n_classes)
    """

    df_val = df_val.copy()

    # Attach image-level predictions safely
    df_val["stage_pred_prob"] = list(stage_probs)
    df_val["stage_pred"] = df_val["stage_pred_prob"].apply(np.argmax)

    # --- Aggregate per patient ---
    patient_preds = []
    patient_gts = []

    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["stage_pred_prob"].values)  # (n_imgs, n_classes)

        # Mean probability aggregation
        mean_prob = probs.mean(axis=0)

        patient_preds.append(np.argmax(mean_prob))
        patient_gts.append(g["label"].iloc[0])

    patient_acc = accuracy_score(patient_gts, patient_preds)
    return patient_acc

def patient_level_auc(df_val, stage_probs, num_classes=NUM_CLASSES):

    stage_probs = np.asarray(stage_probs)

    df_val = df_val.copy()
    df_val["probs"] = list(stage_probs)

    patient_probs, patient_labels = [], []

    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["probs"].values)
        mean_prob = probs.mean(axis=0)

        patient_probs.append(mean_prob)
        patient_labels.append(g["label"].iloc[0])

    patient_probs = np.stack(patient_probs)
    patient_labels = np.array(patient_labels)

    unique_classes = np.unique(patient_labels)

    if len(unique_classes) < 2:
        print("[WARNING] AUC skipped: only one class present")
        return np.nan

    try:
        return roc_auc_score(
            patient_labels,
            patient_probs,
            multi_class="ovr", 
            average="macro"
        )
    except Exception as e:
        print("AUC error:", e)
        return np.nan
# =========================
# Plot metrics
# =========================
def plot_best_metrics(history, best_epoch, out_dir, fold):
    plt.figure(figsize=(10,6))
    metrics = ["train_loss","val_loss","img_acc","f1","qwk","ece"]
    colors = ["blue","red","green","orange","purple","brown"]
    for m,c in zip(metrics,colors):
        plt.plot([best_epoch],
         [history[m][best_epoch - 1]],
         "o", label=m, color=c)
    plt.title(f"Fold {fold} - Best Epoch Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.ylim(0, max(max(history[m]) for m in metrics)*1.1)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir,f"fold{fold}_metrics_best.png"))
    plt.close()

# =========================
# Patient-level thresholding for critical stages
# =========================
def patient_sensitivity_flags(df_val, stage_probs, thresholds, critical_stages=[2]):

    df_val = df_val.copy()
    stage_probs = np.asarray(stage_probs)

    if len(df_val) != len(stage_probs):
        raise ValueError("Mismatch between df_val and stage_probs")

    df_val["probs"] = list(stage_probs)

    patient_flags = []

    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["probs"].values)

        flag = 0

        for s in critical_stages:
            thr = thresholds.get(s, 0.5)

            if (probs[:, s] >= thr).any():
                flag = 1
                break

        patient_flags.append({
            "patient_id": pid,
            "gt_stage": g["label"].iloc[0],
            "flag_critical": flag
        })

    return pd.DataFrame(patient_flags)
# =========================
# Threshold Tuning for Sensitivity
# =========================
def sensitivity_thresholds(y_true, y_probs, target_sens=0.95, critical_stages=[2]):

    thresholds = {}

    for stage in critical_stages:

        y_bin = (y_true >= stage).astype(int)

        fpr, tpr, ths = roc_curve(y_bin, y_probs[:, stage])

        # avoid empty / edge case
        if len(ths) == 0:
            thresholds[stage] = 0.5
            continue

        # closest to target sensitivity
        idx = np.argmin(np.abs(tpr - target_sens))

        thresholds[stage] = ths[idx]

    return thresholds

# =========================
# Confusion Matrix Plot
# =========================
def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()
# =========================
# Loss (SIMPLIFIED)
# =========================
def ordinal_probs_from_logits(logits):
    cumulative = torch.sigmoid(logits)

    # enforce monotonic decreasing
    cumulative = torch.cummin(cumulative, dim=1)[0]

    B, K1 = cumulative.shape
    probs = torch.zeros(B, K1 + 1, device=logits.device)

    probs[:, 0] = 1 - cumulative[:, 0]

    for k in range(1, K1):
        probs[:, k] = cumulative[:, k-1] - cumulative[:, k]

    probs[:, -1] = cumulative[:, -1]

    return torch.clamp(probs, 1e-6, 1.0)
# =================
# Train
# =================
def filter_paths(pt_dir, df):
    # Convert all CSV patient_ids to strings and strip whitespace
    allowed_ids = set(df["patient_id"].astype(str).str.strip().values)

    all_paths = [
        os.path.join(pt_dir, f)
        for f in os.listdir(pt_dir)
        if f.endswith(".pt") and "index" not in f.lower()
    ]

    filtered = []
    print(f"Scanning {len(all_paths)} .pt files in {pt_dir}...")

    for p in all_paths:
        try:
            # We use weights_only=False because these are custom dicts
            data = torch.load(p, map_location="cpu", weights_only=False)

            # Ensure we read the ID as a clean string
            pid = str(data.get("patient_id", "")).strip()

            if pid in allowed_ids:
                filtered.append(p)
        except Exception as e:
            print(f"Error loading {p}: {e}")
            continue

    print(f"Matched {len(filtered)} images for the current split.")
    return filtered
def train_fold(df, pt_dir, fold, args, class_weights=None, patience=5):

    out = os.path.join(args.out_dir, f"fold_{fold}")
    os.makedirs(out, exist_ok=True)

    df_train = df[df.fold != fold].copy()
    df_val   = df[df.fold == fold].copy()

    # -------------------------
    # PATHS
    # -------------------------
    train_paths = filter_paths(pt_dir, df_train)
    val_paths   = filter_paths(pt_dir, df_val)

    # -------------------------
    # SAMPLER
    # -------------------------
    labels = df_train["label"].values
    class_count = np.bincount(labels, minlength=NUM_CLASSES)
    weights = 1.0 / (class_count + 1e-6)
    sample_weights = weights[labels]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # -------------------------
    # LOADERS
    # -------------------------
    train_loader = DataLoader(
        ROPDatasetPT(train_paths, augment=True),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        ROPDatasetPT(val_paths, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = ROPNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    # -------------------------
    # LOSS + DECODER
    # -------------------------
    def compute_loss(logits, targets, weights=None, smoothing=0.1):
        # Smooth targets: 1 -> 0.95, 0 -> 0.05
        targets = targets.float() * (1 - smoothing) + 0.5 * smoothing
        
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        if weights is not None:
            true_ints = targets.sum(dim=1).round().long() # round because of smoothing
            true_ints = torch.clamp(true_ints, 0, NUM_CLASSES - 1)
            batch_weights = weights[true_ints].unsqueeze(1)
            loss = loss * batch_weights
            
        return loss.mean()

    def decode(logits):
        probs = ordinal_probs_from_logits(logits)
        preds = probs.argmax(dim=1)
        return preds, probs

    # -------------------------
    # TRACKING
    # -------------------------
    best_qwk = -1
    best_epoch = 0
    no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "img_acc": [],
        "f1": [],
        "qwk": [],
        "ece": [],
        "pat_acc": [],
        "pat_auc": []
    }

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(1, args.epochs + 1):

        # ========= TRAIN =========
        model.train()
        train_loss = 0

        for imgs, clin, targets, _ in tqdm(train_loader, desc=f"[Fold {fold}] Ep {epoch}"):

            imgs = imgs.to(DEVICE)
            clin = clin.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(imgs, clin)
                loss = compute_loss(logits, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ========= VALIDATION =========
        model.eval()

        val_loss = 0
        all_preds, all_probs, all_true, all_pids = [], [], [], []

        with torch.no_grad():
            for imgs, clin, targets, pids in val_loader:

                imgs = imgs.to(DEVICE)
                clin = clin.to(DEVICE)
                targets = targets.to(DEVICE)

                logits = model(imgs, clin)
                
                # Compute loss using the ordinal targets
                loss = compute_loss(logits, targets, class_weights)
                val_loss += loss.item()

                # Get class predictions and probability distributions
                # Ensure your decode function uses ordinal_probs_from_logits
                probs = ordinal_probs_from_logits(logits)
                preds = probs.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                # FIX: Convert ordinal vector [1, 1, 0] back to integer class 2
                # We sum the 1s across the last dimension
                true_ints = targets.sum(dim=1).long()
                all_true.append(targets.sum(dim=1).long().cpu().numpy())
                
                all_pids.extend(pids)

        val_loss /= len(val_loader)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_probs)

        # ================= METRICS =================
        img_acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ece_val = compute_ece(y_prob, y_true)

        df_val_images = pd.DataFrame({
            "patient_id": all_pids,
            "label": y_true
        })

        pat_acc = aggregate_patient_metrics(df_val_images, y_prob)
        pat_auc = patient_level_auc(df_val_images, y_prob)

        # store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["img_acc"].append(img_acc)
        history["f1"].append(f1)
        history["qwk"].append(qwk)
        history["ece"].append(ece_val)
        history["pat_acc"].append(pat_acc)
        history["pat_auc"].append(pat_auc)

        print(f"[Fold {fold}][Ep {epoch}] "
              f"Train={train_loss:.3f} Val={val_loss:.3f} "
              f"Acc={img_acc:.3f} F1={f1:.3f} QWK={qwk:.3f} "
              f"PatAcc={pat_acc:.3f} AUC={pat_auc:.3f} ECE={ece_val:.3f}")

        # ================= EARLY STOP =================
        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = epoch
            no_improve = 0

            torch.save(model.state_dict(), os.path.join(out, "best.pt"))

            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "img_acc": img_acc,
                "f1": f1,
                "qwk": qwk,
                "ece": ece_val,
                "pat_acc": pat_acc,
                "pat_auc": pat_auc
            }

        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # ================= SAVE HISTORY =================
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # ================= SAVE BEST METRICS =================
    with open(os.path.join(out, "metrics_best.json"), "w") as f:
        json.dump(best_metrics, f, indent=4)

    # ================= SAVE LAST METRICS =================
    last_metrics = {k: history[k][-1] for k in history}
    with open(os.path.join(out, "metrics_last.json"), "w") as f:
        json.dump(last_metrics, f, indent=4)

    # ================= SAVE CSV (FOR REPORT TABLES) =================
    df_hist = pd.DataFrame(history)
    df_hist["epoch"] = np.arange(1, len(df_hist)+1)
    df_hist.to_csv(os.path.join(out, "metrics.csv"), index=False)

    print(f"\n Best Epoch: {best_epoch} | Best QWK: {best_qwk:.4f}")

    # ================= FINAL PASS =================
    model.load_state_dict(torch.load(os.path.join(out, "best.pt")))
    model.eval()

    stage_probs, stage_true, preds_all, stage_pids = [], [], [], []

    with torch.no_grad():
        for imgs, clin, targets, pids in val_loader:
            imgs, clin = imgs.to(DEVICE), clin.to(DEVICE)

            logits = model(imgs, clin)
            preds, probs = decode(logits)

            stage_probs.append(probs.cpu().numpy())
            stage_true.append(targets.sum(dim=1).long().cpu().numpy())
            preds_all.append(preds.cpu().numpy())
            stage_pids.extend(pids)

    stage_probs = np.concatenate(stage_probs)
    stage_true = np.concatenate(stage_true)
    preds = np.concatenate(preds_all)
    # ================= PLOTS =================
    plot_confusion(stage_true, preds, list(range(NUM_CLASSES)),
                   os.path.join(out, "confusion_best.png"))

    for s in range(NUM_CLASSES):
        y_bin = (stage_true == s).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, stage_probs[:, s])
        if len(np.unique(y_bin)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_bin, stage_probs[:, s])

        precision, recall, _ = precision_recall_curve(y_bin, stage_probs[:, s])
        precision, recall, _ = precision_recall_curve(y_bin, stage_probs[:, s])

        plt.figure()
        plt.plot(fpr, tpr)
        plt.savefig(os.path.join(out, f"roc_stage{s}.png"))
        plt.close()

        plt.figure()
        plt.plot(recall, precision)
        plt.savefig(os.path.join(out, f"pr_stage{s}.png"))
        plt.close()

    # ================= GRAD-CAM =================
    gradcam = GradCAM(model)
    gradcam_dir = os.path.join(out, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    mis_idx = np.where(preds != stage_true)[0][:5]

    for idx in mis_idx:
        img, clin, label, pid = val_loader.dataset[idx]

        img = img.unsqueeze(0).to(DEVICE)
        clin = clin.unsqueeze(0).to(DEVICE)

        cam = gradcam.generate(img, clin, preds[idx])[0]

        img_np = img[0][:3].cpu().numpy().transpose(1,2,0)
        img_np = (img_np - img_np.min()) / (img_np.max() + 1e-6)

        cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        cam = cv2.GaussianBlur(cam, (11,11), 0)

        cam = (cam - cam.min()) / (cam.max() + 1e-6)

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # STRONGER OVERLAY
        overlay = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

        plt.imshow(overlay)
        plt.title(f"GT:{label.argmax().item()} Pred:{preds[idx]}")
        plt.axis("off")
        plt.savefig(os.path.join(gradcam_dir, f"{pid}_{idx}.png"))
        plt.close()

    gradcam.remove_hooks()

def final_training(df, args, class_weights=None):
    print("\n=== FINAL TRAINING ON FULL DATASET ===")

    model = ROPNet().to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    all_paths = filter_paths(args.pt_dir, df)

    train_loader = DataLoader(
        ROPDatasetPT(all_paths, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    calib_loader = DataLoader(
        ROPDatasetPT(all_paths, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # LOSS (DEFINED HERE)
    # =========================
    def compute_loss(logits, targets, weights=None):
        loss = F.binary_cross_entropy_with_logits(logits, targets)

        if weights is not None:
            cls = targets.sum(dim=1).long().clamp(0, NUM_CLASSES - 1)
            loss = loss * weights[cls].unsqueeze(1)

        return loss.mean()

    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0

        for imgs, clin, stage, _ in tqdm(train_loader, desc=f"Final Train Ep {epoch}"):

            imgs = imgs.to(DEVICE)
            clin = clin.to(DEVICE)
            stage = stage.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                logits = model(imgs, clin)
                loss = compute_loss(logits, stage, class_weights)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        print(f"[Final][Ep {epoch}] Loss={epoch_loss / len(train_loader):.4f}")

    # =========================
    # TEMPERATURE SCALING (FIXED)
    # =========================
    print("\nApplying temperature scaling...")

    model.eval()

    logits_all, labels_all = [], []

    with torch.no_grad():
        for imgs, clin, stage, _ in calib_loader:

            imgs = imgs.to(DEVICE)
            clin = clin.to(DEVICE)

            logits = model(imgs, clin)

            logits_all.append(logits.cpu())

            # FIX: ordinal → class index
            labels_all.append(stage.sum(dim=1).long().cpu())

    logits_all = torch.cat(logits_all).to(DEVICE)
    labels_all = torch.cat(labels_all).to(DEVICE)

    scaler_temp = TempScaler().to(DEVICE)
    scaler_temp.set_temperature(logits_all, labels_all)

    torch.save({
        "model": model.state_dict(),
        "temperature": scaler_temp.temperature.item()
    }, os.path.join(args.out_dir, "final_stage_model.pt"))

    print(f" Final temperature: {scaler_temp.temperature.item():.3f}")
    print(" Final model saved.")
# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", required=True, help="Directory with cached .pt files")
    parser.add_argument("--csv_file", required=True, help="CSV with patient_id, label for folds")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out_dir", default="./stage_results_final")
    args = parser.parse_args()

    # =========================
    # SETUP
    # =========================
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv_file).dropna()

    # Ensure labels are integers
    df["label"] = df["label"].astype(int)

    print(f"Loaded {len(df)} samples")

    # =========================
    # PATIENT-LEVEL SPLIT
    # =========================
    df["fold"] = -1

    pat = df.groupby("patient_id")["label"].max().reset_index()

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)

    for f, (_, val_idx) in enumerate(
        sgkf.split(pat, pat["label"], groups=pat["patient_id"])
    ):
        val_pids = pat.loc[val_idx, "patient_id"]
        df.loc[df["patient_id"].isin(val_pids), "fold"] = f

    # =========================
    # LEAKAGE CHECK (CRITICAL)
    # =========================
    for f in range(args.folds):
        train_pids = set(df[df.fold != f]["patient_id"])
        val_pids = set(df[df.fold == f]["patient_id"])

        overlap = train_pids.intersection(val_pids)
        if len(overlap) > 0:
            raise ValueError(f"[LEAKAGE] Fold {f} has overlap: {overlap}")

    print(" No patient-level leakage detected")

    # =========================
    # CROSS-VALIDATION
    # =========================
    for f in range(args.folds):
        print(f"\n{'='*40}")
        print(f"STARTING FOLD {f}")
        print(f"{'='*40}")

        train_df = df[df.fold != f]

        # Compute class weights ONLY on training data
        class_counts = (
            train_df["label"]
            .value_counts()
            .sort_index()
            .reindex(range(NUM_CLASSES), fill_value=0)
            .values
        )

        # Use sqrt weighting (stable)
        weights = 1.0 / np.sqrt(class_counts + 1e-6)
        weights = (weights / weights.sum()) * NUM_CLASSES

        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        print(f"[Fold {f}] Class counts: {class_counts}")
        print(f"[Fold {f}] Class weights: {weights}")

        train_fold(df, args.pt_dir, f, args, class_weights)

    # =========================
    # FINAL TRAINING ON FULL DATASET
    # =========================
    print("\n=== FINAL TRAINING ON FULL DATASET ===")

    class_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .values
    )

    weights = 1.0 / np.sqrt(class_counts + 1e-6)
    weights = (weights / weights.sum()) * NUM_CLASSES

    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"[FINAL] Class counts: {class_counts}")
    print(f"[FINAL] Class weights: {weights}")

    final_training(df, args, class_weights)

    print("\n TRAINING COMPLETE")


if __name__ == "__main__":
    main()
