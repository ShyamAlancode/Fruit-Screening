"""
08_train_camera_quality.py

Phase 8: Camera Quality Module
Trains a lightweight visual quality classifier using MobileNetV3-Small on:
- Good apples (Good_/: 1,000 images)
- Defective apples (Bad/: 1,000 images)

Outputs: "External visual quality: Good / Possible defect"
CRITICAL: Never classifies as "Edible" or "Not edible".

Split: 70% Train, 15% Val, 15% Test (stratified).
Saves model to: models/camera/camera_quality_mobilenet.pth

Usage:
    python scripts/08_train_camera_quality.py
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    GOOD_BAD_ROOT,
    CAMERA_OUTPUT_LABEL,
    DISCLAIMER_BANNER,
    RANDOM_SEED,
)

MODELS_DIR = r"d:\CFRD\project\models\camera"
FIGURES_DIR = r"d:\CFRD\project\reports\figures"
TABLES_DIR = r"d:\CFRD\project\reports\tables"


class AppleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def collect_images():
    good_dir = os.path.join(GOOD_BAD_ROOT, "Good_")
    bad_dir = os.path.join(GOOD_BAD_ROOT, "Bad")

    paths, labels = [], []
    for f in sorted(os.listdir(good_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(good_dir, f))
            labels.append(0)  # 0: Good

    for f in sorted(os.listdir(bad_dir)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            paths.append(os.path.join(bad_dir, f))
            labels.append(1)  # 1: Defect

    return paths, labels


def main():
    print(f"\n{DISCLAIMER_BANNER}\n")
    print(f"Task: {CAMERA_OUTPUT_LABEL} (Good vs Possible defect)\n")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths, labels = collect_images()
    print(f"Total images collected: {len(paths)} (Good: {labels.count(0)}, Defect: {labels.count(1)})")

    # 70% Train, 15% Val, 15% Test
    p_train, p_temp, l_train, l_temp = train_test_split(
        paths, labels, test_size=0.30, random_state=RANDOM_SEED, stratify=labels
    )
    p_val, p_test, l_val, l_test = train_test_split(
        p_temp, l_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=l_temp
    )

    print(f"Splits -> Train: {len(p_train)}, Val: {len(p_val)}, Test: {len(p_test)}")

    # Preprocessing transforms (ImageNet normalization)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = AppleImageDataset(p_train, l_train, transform=train_transform)
    val_ds   = AppleImageDataset(p_val, l_val, transform=val_test_transform)
    test_ds  = AppleImageDataset(p_test, l_test, transform=val_test_transform)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize MobileNetV3-Small
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze feature backbone for rapid transfer learning
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head for 2 classes: Good (0) vs Possible Defect (1)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    # Train for 5 epochs
    epochs = 5
    print("\nStarting MobileNetV3-Small fine-tuning...")
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(lbls)
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += len(lbls)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += len(lbls)
        val_acc = val_correct / val_total
        print(f"  Epoch {ep+1}/{epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.1f}%, Val Acc: {val_acc*100:.1f}%")

    # Evaluate on Test Set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, zero_division=0)
    test_rec = recall_score(all_labels, all_preds, zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print(f"Test Set Evaluation ({len(all_labels)} images):")
    print(f"  Accuracy:  {test_acc*100:.2f}%")
    print(f"  Precision: {test_prec*100:.2f}%")
    print(f"  Recall:    {test_rec*100:.2f}%")
    print(f"  Macro-F1:  {test_f1*100:.2f}%")
    print("=" * 60)

    # Save model weights
    save_path = os.path.join(MODELS_DIR, "camera_quality_mobilenet.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": ["Good", "Possible defect"],
        "metrics": {"acc": test_acc, "f1": test_f1, "prec": test_prec, "rec": test_rec},
        "disclaimer": "External visual quality only — not food safety certification.",
    }, save_path)
    print(f"Saved model artifact: {save_path}")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Good", "Possible defect"]
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f"External Visual Quality (Test Set)\nAccuracy: {test_acc*100:.1f}%",
           ylabel='True Label', xlabel='Predicted Label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"confusion_camera_quality_{ts}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Summary table
    report_md = f"""# Camera Visual Quality Module Report

> **Output Label:** {CAMERA_OUTPUT_LABEL} (Good / Possible defect)  
> **Notice:** This model evaluates surface visual appearance only. Never used for residue or safety decisions.

## Test Performance (15% Held-Out Split)
| Metric | Value |
|---|---|
| Test Images | {len(all_labels)} |
| Accuracy | {test_acc*100:.2f}% |
| Precision | {test_prec*100:.2f}% |
| Recall | {test_rec*100:.2f}% |
| Macro-F1 | {test_f1*100:.2f}% |

## Saved Model
`{save_path}`
"""
    table_path = os.path.join(TABLES_DIR, f"camera_quality_{ts}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report saved: {table_path}")


if __name__ == "__main__":
    main()
