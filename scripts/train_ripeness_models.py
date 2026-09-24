"""
scripts/train_ripeness_models.py
================================
Training pipeline for multi-stage apple ripeness computer vision models.

Models trained and compared:
  1. Baseline 1: Color Moments & HSV Histograms -> Random Forest Classifier
  2. Model 2: MobileNetV3-Small (Pretrained ImageNet -> 3 classes)
  3. Model 3: ResNet18 (Pretrained ImageNet -> 3 classes)
  4. Model 4: EfficientNet-B0 (Pretrained ImageNet -> 3 classes)

Classes (3):
  0: immature (green)
  1: semi_mature (color transition: green/red)
  2: mature (red)

Saves best model checkpoints to models/quality/
"""

import os
import sys

# Windows PyTorch DLL conflict fix: MUST execute before importing anything else
os.environ["PATH"] = ";".join([
    p for p in os.environ.get("PATH", "").split(";")
    if "mingw" not in p.lower() and "intel" not in p.lower()
])

# Import torch FIRST before any other C-extension or package
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ripeness_multistage"
MODELS_DIR = PROJECT_ROOT / "models" / "quality"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["immature", "semi_mature", "mature"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Dataset & Feature Extraction for Baseline 1 (Random Forest)
# ---------------------------------------------------------------------------

def extract_color_features(img_path: Path) -> np.ndarray:
    """Extract 54 color moments, HSV, and RGB distribution features."""
    with Image.open(img_path) as im:
        im_small = im.resize((128, 128)).convert("RGB")
        rgb = np.array(im_small, dtype=float) / 255.0

    # RGB Moments (Mean, Std)
    r_mean, g_mean, b_mean = rgb[:, :, 0].mean(), rgb[:, :, 1].mean(), rgb[:, :, 2].mean()
    r_std, g_std, b_std = rgb[:, :, 0].std(), rgb[:, :, 1].std(), rgb[:, :, 2].std()

    # HSV conversion
    hsv = im_small.convert("HSV")
    hsv_arr = np.array(hsv, dtype=float) / 255.0
    h_mean, s_mean, v_mean = hsv_arr[:, :, 0].mean(), hsv_arr[:, :, 1].mean(), hsv_arr[:, :, 2].mean()
    h_std, s_std, v_std = hsv_arr[:, :, 0].std(), hsv_arr[:, :, 1].std(), hsv_arr[:, :, 2].std()

    # Color ratios & vegetation indices (standard for fruit maturity)
    # NDVI-like green-red ratio: (G - R) / (G + R + eps)
    rg_ratio = (g_mean - r_mean) / (g_mean + r_mean + 1e-6)
    rb_ratio = (r_mean - b_mean) / (r_mean + b_mean + 1e-6)

    # Histograms
    h_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=16, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv_arr[:, :, 1], bins=8, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv_arr[:, :, 2], bins=8, range=(0, 1), density=True)
    r_hist, _ = np.histogram(rgb[:, :, 0], bins=8, range=(0, 1), density=True)
    g_hist, _ = np.histogram(rgb[:, :, 1], bins=8, range=(0, 1), density=True)

    moments = [r_mean, g_mean, b_mean, r_std, g_std, b_std,
               h_mean, s_mean, v_mean, h_std, s_std, v_std,
               rg_ratio, rb_ratio]

    return np.concatenate([moments, h_hist, s_hist, v_hist, r_hist, g_hist])


def load_tabular_split(split_name: str):
    """Load features and labels for Random Forest."""
    split_dir = DATA_DIR / split_name
    X_list, y_list = [], []
    for cls_idx, c_name in enumerate(CLASS_NAMES):
        c_dir = split_dir / c_name
        for p in c_dir.glob("*.jpg"):
            try:
                feats = extract_color_features(p)
                X_list.append(feats)
                y_list.append(cls_idx)
            except Exception:
                continue
    return np.array(X_list), np.array(y_list)


def train_random_forest():
    """Train and evaluate Random Forest baseline."""
    print("\n" + "=" * 60)
    print("TRAINING BASELINE 1: Random Forest (Color Moments & HSV)")
    print("=" * 60)

    X_train, y_train = load_tabular_split("train")
    X_val, y_val = load_tabular_split("val")
    X_test, y_test = load_tabular_split("test")

    print(f"Loaded feature matrix: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_bal = balanced_accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")

    test_preds = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_bal = balanced_accuracy_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds, average="macro")

    print(f"  Validation: Acc = {val_acc*100:.2f}%, Bal Acc = {val_bal*100:.2f}%, Macro-F1 = {val_f1*100:.2f}%")
    print(f"  Test:       Acc = {test_acc*100:.2f}%, Bal Acc = {test_bal*100:.2f}%, Macro-F1 = {test_f1*100:.2f}%")

    model_path = MODELS_DIR / "ripeness_rf_baseline.joblib"
    joblib.dump({"model": rf, "val_acc": val_acc, "test_acc": test_acc}, model_path)
    print(f"  Saved baseline model to {model_path}")

    return {"model_name": "Random Forest (HSV)", "val_acc": val_acc, "test_acc": test_acc, "test_f1": test_f1}


# ---------------------------------------------------------------------------
# PyTorch Dataset & Transforms for Deep Learning Models
# ---------------------------------------------------------------------------

class AppleCropDataset(Dataset):
    def __init__(self, split_dir: Path, transform=None):
        self.samples = []
        self.transform = transform
        for cls_idx, c_name in enumerate(CLASS_NAMES):
            c_dir = split_dir / c_name
            for p in c_dir.glob("*.jpg"):
                self.samples.append((str(p), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_transforms():
    """Data augmentations for training and normalization for validation/test."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def build_deep_model(arch: str) -> nn.Module:
    """Build pretrained model with 3-class classification head."""
    if arch == "mobilenet_v3":
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_feats = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_feats, 3)
    elif arch == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, 3)
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_feats = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feats, 3)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return net


def train_pytorch_model(arch: str, epochs_phase1: int = 4, epochs_phase2: int = 8, batch_size: int = 32):
    """2-Phase transfer learning with cosine decay."""
    print("\n" + "=" * 60)
    print(f"TRAINING {arch.upper()} (2-Phase Transfer Learning)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_tf, val_tf = get_data_transforms()
    train_ds = AppleCropDataset(DATA_DIR / "train", transform=train_tf)
    val_ds   = AppleCropDataset(DATA_DIR / "val", transform=val_tf)
    test_ds  = AppleCropDataset(DATA_DIR / "test", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    model = build_deep_model(arch).to(device)
    criterion = nn.CrossEntropyLoss()

    # PHASE 1: Freeze backbone, train classification head
    print(f"\n--- Phase 1: Feature Extractor Frozen ({epochs_phase1} epochs) ---")
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head
    if arch == "mobilenet_v3":
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif arch == "resnet18":
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == "efficientnet_b0":
        for param in model.classifier.parameters():
            param.requires_grad = True

    opt_p1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, epochs_phase1 + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_p1.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt_p1.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Val evaluation
        val_acc, val_loss = eval_pytorch(model, val_loader, criterion, device)
        print(f"  [P1 Ep {ep}/{epochs_phase1}] Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%  ||  Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

    # PHASE 2: Unfreeze top layers and fine-tune
    print(f"\n--- Phase 2: Fine-Tuning ({epochs_phase2} epochs) ---")
    for param in model.parameters():
        param.requires_grad = True

    opt_p2 = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_p2, T_max=epochs_phase2)

    best_val_acc = 0.0
    best_weights_path = MODELS_DIR / f"ripeness_{arch}.pth"

    for ep in range(1, epochs_phase2 + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt_p2.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt_p2.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        sched.step()
        train_acc = correct / total
        train_loss = running_loss / total

        val_acc, val_loss = eval_pytorch(model, val_loader, criterion, device)
        print(f"  [P2 Ep {ep}/{epochs_phase2}] Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%  ||  Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "arch": arch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "classes": CLASS_NAMES,
            }, best_weights_path)

    # Load best checkpoint and evaluate on Test set
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc, test_f1 = eval_test_metrics(model, test_loader, device)

    print(f"\n  ✓ Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"  ✓ Independent Test Accuracy: {test_acc*100:.2f}%, Test Macro-F1: {test_f1*100:.2f}%")
    print(f"  Saved best weights to {best_weights_path}")

    return {"model_name": arch, "val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}


def eval_pytorch(model, dataloader, criterion, device):
    """Evaluate loss and accuracy on dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total, running_loss / total


def eval_test_metrics(model, test_loader, device):
    """Compute detailed accuracy and macro-F1 on test loader."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train multi-stage apple ripeness models.")
    parser.add_argument("--epochs1", type=int, default=3, help="Phase 1 frozen epochs")
    parser.add_argument("--epochs2", type=int, default=6, help="Phase 2 fine-tune epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    results = []

    # 1. Train Baseline 1: Random Forest
    rf_res = train_random_forest()
    results.append(rf_res)

    # 2. Train Model 2: MobileNetV3-Small (CFRD Edge Standard)
    mb_res = train_pytorch_model("mobilenet_v3", epochs_phase1=args.epochs1, epochs_phase2=args.epochs2, batch_size=args.batch_size)
    results.append(mb_res)

    # 3. Train Model 3: ResNet18
    res_res = train_pytorch_model("resnet18", epochs_phase1=args.epochs1, epochs_phase2=args.epochs2, batch_size=args.batch_size)
    results.append(res_res)

    # 4. Train Model 4: EfficientNet-B0
    eff_res = train_pytorch_model("efficientnet_b0", epochs_phase1=args.epochs1, epochs_phase2=args.epochs2, batch_size=args.batch_size)
    results.append(eff_res)

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINING & VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Model Architecture':<25} | {'Validation Acc':<16} | {'Test Acc':<12} | {'Test Macro-F1':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['model_name']:<25} | {r['val_acc']*100:6.2f}%          | {r['test_acc']*100:6.2f}%     | {r['test_f1']*100:6.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
