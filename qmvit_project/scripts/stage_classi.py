# CPU/GPU-FRIENDLY ROP STAGE CLASSIFICATION
# Using cached .pt files (images + masks + clinical data)
# Ensemble: MaxViT + EfficientNet + Ben Graham + Mask
# Includes MIL, Grad-CAM, ECE, Temp Scaling, metrics, patient aggregation
# ============================================================

import os, argparse, warnings
from collections import defaultdict

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns
import json
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6

# =========================
# BN → GN (CPU SAFE)
# =========================
def convert_bn_to_gn(module, groups=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            g = min(groups, child.num_features)
            while child.num_features % g != 0:
                g -= 1
            setattr(module, name, nn.GroupNorm(g, child.num_features))
        else:
            convert_bn_to_gn(child, groups)
# =========================
# Dataset using cached .pt files
# =========================
class ROPDatasetPT(Dataset):
    def __init__(self, pt_dir, fold=None, df=None):
        """
        If df is provided, fold filtering is applied; otherwise loads all .pt files.
        """
        self.pt_paths = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith(".pt")]
        self.pt_paths.sort()
        if df is not None and fold is not None:
            allowed_ids = set(df[df.fold == fold].patient_id) if fold is not None else set(df.patient_id)
            filtered_paths = []
            for p in self.pt_paths:
                data = torch.load(p)
                if data["patient_id"] in allowed_ids:
                    filtered_paths.append(p)
            self.pt_paths = filtered_paths

    def __len__(self):
        return len(self.pt_paths)

    def __getitem__(self, idx):
        data = torch.load(self.pt_paths[idx])
        img = data["img"]  # Tensor (4, H, W)
        ga = np.clip(data["ga"], 22, 40)
        bw = np.clip(data["bw"], 400, 2500)
        ga = (ga - 22) / (40 - 22)
        bw = (bw - 400) / (2500 - 400)
        clinical = torch.tensor([ga, bw], dtype=torch.float32)
        label = torch.tensor(data["stage"], dtype=torch.long)
        patient_id = data["patient_id"]
        return img, clinical, label, patient_id

# =========================
# Ensemble Model
# =========================
class ROPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone1 = timm.create_model("convnext_nano", pretrained=True, features_only=True, in_chans=4)
        self.conv_head1 = nn.Conv2d(640, 256, 1)  # Change 1024 to 640
        self.attn1 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

        # Backbone 2: EfficientNet-B0
        self.backbone2 = timm.create_model("efficientnet_b0", pretrained=True, features_only=True, in_chans=4)
        # Corrected: last feature map of EfficientNet-B0 has 320 channels
        self.conv_head2 = nn.Conv2d(320, 256, 1)
        self.attn2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1)
        )

        # Clinical features
        self.clinical_fc = nn.Linear(2, 64)

        # Final fusion
        self.fc = nn.Linear(256 + 256 + 64, NUM_CLASSES)

    def forward(self, x, clinical):
        # Backbone 1
        f1 = self.backbone1(x)[-1]  # (B, 1024, H, W)
        f1 = self.conv_head1(f1)    # (B, 256, H, W)
        a1 = torch.softmax(self.attn1(f1).view(f1.size(0), 1, -1), dim=-1)
        f1 = (f1.view(f1.size(0), 256, -1) * a1).sum(-1)  # (B, 256)

        # Backbone 2
        f2 = self.backbone2(x)[-1]  # (B, 320, H, W)
        f2 = self.conv_head2(f2)    # (B, 256, H, W)
        a2 = torch.softmax(self.attn2(f2).view(f2.size(0), 1, -1), dim=-1)
        f2 = (f2.view(f2.size(0), 256, -1) * a2).sum(-1)  # (B, 256)

        # Clinical features
        c = F.relu(self.clinical_fc(clinical))  # (B, 64)

        # Fusion
        fused = torch.cat([f1, f2, c], dim=1)  # (B, 576)
        return self.fc(fused)  # (B, NUM_CLASSES)

# =========================
# EMD Loss
# =========================
def emd_loss(logits, targets):
    probs = torch.softmax(logits, dim=1)
    cum_p = torch.cumsum(probs, dim=1)
    cum_t = torch.cumsum(F.one_hot(targets, NUM_CLASSES).float(), dim=1)
    return torch.mean((cum_p - cum_t)**2)

# =========================
# ECE
# =========================
def compute_ece(probs, labels, n_bins=15):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf>bins[i]) & (conf<=bins[i+1])
        if mask.sum()==0: continue
        ece += np.abs(acc[mask].mean() - conf[mask].mean())*mask.mean()
    return ece

# =========================
# Temperature Scaling
# =========================
class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        nll = nn.CrossEntropyLoss()

        def eval():
            optimizer.zero_grad()
            loss = nll(logits/self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        print(f"Optimal temperature: {self.temperature.item():.3f}")

# =========================
# Grad-CAM
# =========================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        def forward_hook(module, input, output):
            self.act = output
        def backward_hook(module, grad_in, grad_out):
            self.grad = grad_out[0]

        model.conv_head1.register_forward_hook(forward_hook)
        model.conv_head1.register_full_backward_hook(backward_hook)

    def generate(self, x, clinical, cls):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x, clinical)
        out[:,cls].sum().backward()
        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (w * self.act).sum(1)
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        self.model.conv_head1._forward_hooks.clear()
        self.model.conv_head1._backward_hooks.clear()

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
        patient_gts.append(g["stage_label"].iloc[0])

    patient_acc = accuracy_score(patient_gts, patient_preds)
    return patient_acc

def patient_level_auc(df_val, stage_probs, num_classes=NUM_CLASSES):
    """
    df_val: DataFrame with columns [patient_id, stage_label]
    stage_probs: (N_images, num_classes)
    """
    df_val = df_val.copy()
    df_val["probs"] = list(stage_probs)

    patient_probs = []
    patient_labels = []

    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["probs"].values)      # (n_imgs, C)
        mean_prob = probs.mean(axis=0)           # (C,)

        patient_probs.append(mean_prob)
        patient_labels.append(g["stage_label"].iloc[0])

    patient_probs = np.stack(patient_probs)
    patient_labels = np.array(patient_labels)

    try:
        auc = roc_auc_score(
            patient_labels,
            patient_probs,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc = np.nan

    return auc

# =========================
# Plot metrics
# =========================
def plot_best_metrics(history, best_epoch, out_dir, fold):
    plt.figure(figsize=(10,6))
    metrics = ["train_loss","val_loss","img_acc","f1","qwk","ece"]
    colors = ["blue","red","green","orange","purple","brown"]
    for m,c in zip(metrics,colors):
        plt.plot([best_epoch],[history[m][best_epoch]],"o",label=m,color=c)
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
def patient_sensitivity_flags(df_val, stage_probs, thresholds, critical_stages=[3,4,5]):
    """
    Flags patients if any image exceeds the Stage ≥3 probability threshold.
    df_val: DataFrame with columns ['patient_id', 'stage_label']
    stage_probs: (N_images, num_classes) predicted probabilities
    thresholds: dict from sensitivity_thresholds()
    Returns: DataFrame with patient-level flags
    """
    df_val = df_val.copy()
    df_val["probs"] = list(stage_probs)
    
    patient_flags = []
    for pid, g in df_val.groupby("patient_id"):
        probs = np.stack(g["probs"].values)  # (n_imgs, C)
        flag = 0
        for s in critical_stages:
            if (probs[:, s] >= thresholds[s]).any():
                flag = 1
                break
        patient_flags.append({"patient_id": pid, 
                              "gt_stage": g["stage_label"].iloc[0],
                              "flag_critical": flag})
    return pd.DataFrame(patient_flags)
# =========================
# Threshold Tuning for Sensitivity
# =========================
def sensitivity_thresholds(y_true, y_probs, target_sens=0.95, critical_stages=[3,4,5]):
    """
    For each critical stage, compute probability threshold achieving target sensitivity.
    y_true: (N,) true stage labels
    y_probs: (N, num_classes) predicted probabilities
    Returns: dict {stage: threshold}
    """
    thresholds = {}
    for stage in critical_stages:
        # Binary: stage vs not-stage
        y_bin = (y_true == stage).astype(int)
        fpr, tpr, ths = roc_curve(y_bin, y_probs[:, stage])
        # Find threshold where sensitivity >= target_sens
        idx = np.where(tpr >= target_sens)[0]
        if len(idx) == 0:
            thresholds[stage] = 0.5  # fallback
        else:
            thresholds[stage] = ths[idx[0]]
    return thresholds

# =========================
# Confusion Matrix Plot
# =========================
def plot_confusion(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(out_path)
    plt.close()

# =========================
# Fold Training with Threshold & ROC/PR
# =========================
def train_fold(df, pt_dir, fold, args, patience=3):
    out = os.path.join(args.out_dir, f"fold_{fold}")
    os.makedirs(out, exist_ok=True)
    gradcam_dir = os.path.join(out, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)

    train_loader = DataLoader(
        ROPDatasetPT(pt_dir, fold=None, df=df[df.fold != fold]),
        batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        ROPDatasetPT(pt_dir, fold=None, df=df[df.fold == fold]),
        batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = ROPNet().to(DEVICE)
    convert_bn_to_gn(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Grad-CAM hooks
    class GradCAMHook:
        def __init__(self, module):
            self.grad = None
            self.act = None
            module.register_forward_hook(self.forward_hook)
            module.register_full_backward_hook(self.backward_hook)
        def forward_hook(self, module, input, output):
            self.act = output
        def backward_hook(self, module, grad_input, grad_output):
            self.grad = grad_output[0]
    gradcam = GradCAMHook(model.conv_head1)

    best_acc = -1
    wait = 0
    history = defaultdict(list)

    for epoch in range(1, args.epochs + 1):
        # ========== TRAIN ==========
        model.train()
        train_loss = 0
        for imgs, clin, stage, _ in tqdm(train_loader, desc=f"Fold {fold} Ep {epoch}"):
            imgs, clin, stage = imgs.to(DEVICE), clin.to(DEVICE), stage.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs, clin)
            loss = emd_loss(logits, stage)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        stage_probs, stage_true, stage_pids = [], [], []

        with torch.no_grad():
            for imgs, clin, stage, pids in val_loader:
                imgs, clin, stage = imgs.to(DEVICE), clin.to(DEVICE), stage.to(DEVICE)
                logits = model(imgs, clin)
                loss = emd_loss(logits, stage)
                val_loss += loss.item()
                stage_probs.append(F.softmax(logits, dim=1).cpu().numpy())
                stage_true.append(stage.cpu().numpy())
                stage_pids.extend(pids)

        val_loss /= len(val_loader)
        stage_probs = np.concatenate(stage_probs, axis=0)
        stage_true = np.concatenate(stage_true, axis=0)
        preds = stage_probs.argmax(axis=1)

        # --- Image-level metrics ---
        img_acc = (preds == stage_true).mean()
        f1 = f1_score(stage_true, preds, average="macro")
        qwk = cohen_kappa_score(stage_true, preds, weights="quadratic")
        ece_val = compute_ece(stage_probs, stage_true)

        df_val_images = pd.DataFrame({
            "patient_id": stage_pids,
            "stage_label": stage_true
        })
        pat_acc = aggregate_patient_metrics(df_val_images, stage_probs)
        pat_auc = patient_level_auc(df_val_images, stage_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["img_acc"].append(img_acc)
        history["f1"].append(f1)
        history["qwk"].append(qwk)
        history["ece"].append(ece_val)
        history["pat_acc"].append(pat_acc)
        history["pat_auc"].append(pat_auc)

        print(f"[Fold {fold}][Ep {epoch}] Train={train_loss:.3f} Val={val_loss:.3f} "
              f"ImgAcc={img_acc:.3f} PatAcc={pat_acc:.3f} PatAuc={pat_auc:.3f} "
              f"F1={f1:.3f} QWK={qwk:.3f} ECE={ece_val:.3f}")

        # --- Save best model ---
        if img_acc > best_acc:
            best_acc = img_acc
            wait = 0
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        #else:
         #   wait += 1
          #  if wait >= patience:
           #     print("Early stopping")
            #    break

    # ======================= POST-TRAINING: Best Epoch Outputs =======================
    best_epoch = int(np.argmax(history["img_acc"]))
    print(f"Best epoch: {best_epoch + 1} | Best image-level accuracy: {best_acc:.3f}")

    # Reload best model
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=DEVICE))
    model.eval()

    # Recompute predictions
    stage_probs, stage_true, stage_pids = [], [], []
    with torch.no_grad():
        for imgs, clin, stage, pids in val_loader:
            imgs, clin, stage = imgs.to(DEVICE), clin.to(DEVICE), stage.to(DEVICE)
            logits = model(imgs, clin)
            stage_probs.append(F.softmax(logits, dim=1).cpu().numpy())
            stage_true.append(stage.cpu().numpy())
            stage_pids.extend(pids)

    stage_probs = np.concatenate(stage_probs, axis=0)
    stage_true = np.concatenate(stage_true, axis=0)
    preds = stage_probs.argmax(axis=1)
    df_val_images = pd.DataFrame({"patient_id": stage_pids, "stage_label": stage_true})

    # --- Threshold tuning for critical stages (≥3) ---
    thresholds = sensitivity_thresholds(stage_true, stage_probs, target_sens=0.95)
    print(f"[Fold {fold}] Stage ≥3 thresholds (95% sensitivity): {thresholds}")

    patient_flags_df = patient_sensitivity_flags(df_val_images, stage_probs, thresholds)
    true_critical = patient_flags_df[patient_flags_df["gt_stage"] >= 3]
    flagged_critical = true_critical["flag_critical"].sum()

    print(f"Sensitivity: {flagged_critical}/{len(true_critical)} critical patients flagged")

    # --- ROC & PR curves ---
    for s in range(NUM_CLASSES):
        y_bin = (stage_true == s).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, stage_probs[:, s])
        precision, recall, _ = precision_recall_curve(y_bin, stage_probs[:, s])

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"Stage {s} ROC (AUC={roc_auc_score(y_bin, stage_probs[:, s]):.3f})")
        plt.plot([0,1],[0,1],"k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Fold {fold} ROC - Stage {s}")
        plt.legend()
        plt.savefig(os.path.join(out, f"roc_stage{s}_best.png"))
        plt.close()

        plt.figure(figsize=(6,5))
        plt.plot(recall, precision, label=f"Stage {s} PR")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Fold {fold} PR - Stage {s}")
        plt.legend()
        plt.savefig(os.path.join(out, f"pr_stage{s}_best.png"))
        plt.close()
    
    # =========================
    # Save Validation Progress Plots (Separate Per Metric)
    # =========================

    epochs = range(1, len(history["val_loss"]) + 1)

    val_metrics = {
        "val_loss": history["val_loss"],
        "val_img_acc": history["img_acc"],
        "val_pat_acc": history["pat_acc"],
        "val_pat_auc": history["pat_auc"],
        "val_f1": history["f1"],
        "val_qwk": history["qwk"],
        "val_ece": history["ece"],
    }

    for metric_name, values in val_metrics.items():
        plt.figure(figsize=(8,6))
        plt.plot(epochs, values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.replace("_", " ").upper())
        plt.title(f"Fold {fold} - {metric_name.replace('_',' ').upper()} Progress")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"{metric_name}_progress.png"))
        plt.close()

    # --- Confusion matrix ---
    plot_confusion(stage_true, preds, classes=list(range(NUM_CLASSES)), 
                   out_path=os.path.join(out, "confusion_best.png"))

    # --- Grad-CAM for top misclassified images ---
    misclassified_idx = np.where(preds != stage_true)[0][:5]

    for idx in misclassified_idx:

        img, clin, label, pid = val_loader.dataset[idx]

        img = img.unsqueeze(0).to(DEVICE)
        clin = clin.unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)

        logits = model(img, clin)

        pred_class = preds[idx]

        logits[:, pred_class].backward()

        # Grad-CAM computation
        gradients = gradcam.grad
        activations = gradcam.act

        weights = gradients.mean(dim=(2,3), keepdim=True)

        cam = (weights * activations).sum(dim=1)

        cam = F.relu(cam)

        cam = cam / (cam.max() + 1e-6)

        cam = cam[0].detach().cpu().numpy()

        # Convert image to numpy
        img_np = img[0][:3].detach().cpu().numpy().transpose(1,2,0)

        # Resize CAM to image resolution
        cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay heatmap on fundus image
        overlay = 0.6 * img_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)

        plt.figure(figsize=(6,6))
        plt.imshow(overlay)
        plt.title(f"PID {pid} | GT:{label} Pred:{pred_class}")
        plt.axis("off")

        plt.savefig(os.path.join(gradcam_dir, f"gradcam_{pid}_{idx}.png"))
        plt.close()

    # --- Save history & thresholds ---
    with open(os.path.join(out, "history.json"), "w") as f:
        json.dump({k:list(v) for k,v in history.items()}, f)
    with open(os.path.join(out, "history.json"), "w") as f:
        clean_history = {k: [float(x) for x in v] for k, v in history.items()}
        json.dump(clean_history, f)

# =========================
# FINAL TRAINING (RUNS ONCE)
# =========================
def final_training(df, args):
    print("\n=== FINAL TRAINING ON FULL DATASET ===")

    model = ROPNet().to(DEVICE)
    convert_bn_to_gn(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    full_loader = DataLoader(
        ROPDatasetPT(args.pt_dir, fold=None, df=None),  # load ALL pt files
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0

        for imgs, clin, stage, _ in tqdm(full_loader, desc=f"Final Train Ep {epoch}"):
            imgs, clin, stage = imgs.to(DEVICE), clin.to(DEVICE), stage.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs, clin)
            loss = emd_loss(logits, stage)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(full_loader)
        print(f"[Final][Ep {epoch}] Loss={epoch_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_stage_model.pt"))
    print("Final model saved.")
# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", required=True, help="Directory with cached .pt files")
    parser.add_argument("--csv_file", required=True, help="CSV with patient_id, stage_label for folds")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out_dir", default="./stage_results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file).dropna()
    df["fold"]=-1
    pat = df.groupby("patient_id")["stage_label"].max().reset_index()
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for f,(_,idx) in enumerate(skf.split(pat.patient_id, pat.stage_label)):
        df.loc[df.patient_id.isin(pat.patient_id.iloc[idx]), "fold"]=f
    for f in range(5):
        train_fold(df, args.pt_dir, f, args)

    final_training(df, args)

if __name__=="__main__":
    main()