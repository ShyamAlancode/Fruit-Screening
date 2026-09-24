"""
scripts/evaluate_ripeness_benchmark.py
======================================
Comprehensive evaluation and benchmarking of trained apple ripeness models:
  1. Detailed evaluation on the independent test set (408 crops: immature, semi-mature, mature)
     - Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1
     - 3x3 Confusion Matrices
  2. External zero-shot validation on the RawRipe dataset (213 apples: raw -> immature, ripe -> mature)
  3. Exports reports/ripeness_benchmark_report.md and JSON metrics.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure safe Windows PyTorch environment
os.environ["PATH"] = ";".join([
    p for p in os.environ.get("PATH", "").split(";")
    if "mingw" not in p.lower() and "intel" not in p.lower()
])

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ripeness_multistage"
MODELS_DIR = PROJECT_ROOT / "models" / "quality"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["immature", "semi_mature", "mature"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class CropDataset(Dataset):
    def __init__(self, folder_path: Path, transform=None):
        self.samples = []
        self.transform = transform
        for cls_idx, c_name in enumerate(CLASS_NAMES):
            c_dir = folder_path / c_name
            if c_dir.exists():
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
        return img, label, path


def load_model(arch: str, weights_path: Path, device: torch.device):
    if arch == "mobilenet_v3":
        net = models.mobilenet_v3_small(weights=None)
        in_feats = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_feats, 3)
    elif arch == "resnet18":
        net = models.resnet18(weights=None)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, 3)
    elif arch == "efficientnet_b0":
        net = models.efficientnet_b0(weights=None)
        in_feats = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feats, 3)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    return net


def evaluate_deep_model(net, dataloader, device):
    all_preds, all_labels = [], []
    all_probs = []
    with torch.no_grad():
        for imgs, labels, _ in dataloader:
            imgs = imgs.to(device)
            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def evaluate_rawripe_external(net, device, transform):
    """External zero-shot validation on RawRipe: 105 raw (immature) + 108 ripe (mature)."""
    raw_dir = DATA_DIR / "external_rawripe" / "immature"
    ripe_dir = DATA_DIR / "external_rawripe" / "mature"

    records = []
    for p in raw_dir.glob("*.jpg"):
        records.append((p, 0))  # ground truth: immature
    for p in ripe_dir.glob("*.jpg"):
        records.append((p, 2))  # ground truth: mature

    preds, trues = [], []
    with torch.no_grad():
        for p, true_cls in records:
            with Image.open(p) as im:
                t_img = transform(im.convert("RGB")).unsqueeze(0).to(device)
            output = net(t_img)
            pred_cls = output.argmax(dim=1).item()
            preds.append(pred_cls)
            trues.append(true_cls)

    preds = np.array(preds)
    trues = np.array(trues)

    # In RawRipe, semi-mature doesn't exist as ground truth.
    # Count exact matches (immature->immature, mature->mature)
    exact_match = (preds == trues).sum() / len(trues)
    # Binary distinction: immature vs non-immature (or mature vs non-mature)
    raw_correct = (preds[trues == 0] == 0).sum() / (trues == 0).sum()
    ripe_correct = (preds[trues == 2] == 2).sum() / (trues == 2).sum()

    return {
        "total_external_samples": len(trues),
        "exact_match_accuracy": float(exact_match),
        "raw_immature_recall": float(raw_correct),
        "ripe_mature_recall": float(ripe_correct),
        "predicted_distribution": {
            "immature": int((preds == 0).sum()),
            "semi_mature": int((preds == 1).sum()),
            "mature": int((preds == 2).sum()),
        }
    }


def main():
    print("=" * 70)
    print("APPLE RIPENESS COMPREHENSIVE BENCHMARK & EXTERNAL VALIDATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = CropDataset(DATA_DIR / "test", transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"Independent Test Set: {len(test_ds)} apple crops")

    models_to_test = [
        ("MobileNetV3-Small", "mobilenet_v3", MODELS_DIR / "ripeness_mobilenet_v3.pth"),
        ("EfficientNet-B0", "efficientnet_b0", MODELS_DIR / "ripeness_efficientnet_b0.pth"),
        ("ResNet18", "resnet18", MODELS_DIR / "ripeness_resnet18.pth"),
    ]

    benchmark_results = {}

    for display_name, arch, w_path in models_to_test:
        if not w_path.exists():
            print(f"Weights not found for {display_name} at {w_path}")
            continue

        print(f"\n--- Evaluating {display_name} ---")
        net = load_model(arch, w_path, device)
        preds, labels, probs = evaluate_deep_model(net, test_loader, device)

        acc = accuracy_score(labels, preds)
        bal_acc = balanced_accuracy_score(labels, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        cm = confusion_matrix(labels, preds)

        print(f"  Accuracy:         {acc*100:.2f}%")
        print(f"  Balanced Acc:     {bal_acc*100:.2f}%")
        print(f"  Macro Precision:  {prec*100:.2f}%")
        print(f"  Macro Recall:     {rec*100:.2f}%")
        print(f"  Macro-F1 Score:   {f1*100:.2f}%")
        print("  Confusion Matrix:")
        print(cm)

        # External RawRipe validation
        rawripe_res = evaluate_rawripe_external(net, device, eval_tf)
        print(f"  External RawRipe Accuracy: {rawripe_res['exact_match_accuracy']*100:.2f}% (Raw Recall: {rawripe_res['raw_immature_recall']*100:.2f}%, Ripe Recall: {rawripe_res['ripe_mature_recall']*100:.2f}%)")

        benchmark_results[display_name] = {
            "test_accuracy": float(acc),
            "test_balanced_accuracy": float(bal_acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_macro_f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "external_rawripe_validation": rawripe_res,
        }

    # Save to JSON
    json_path = REPORTS_DIR / "ripeness_benchmark_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[OK] Saved benchmark metrics JSON to: {json_path}")

    # Generate Markdown Report
    md_path = REPORTS_DIR / "ripeness_benchmark_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🍎 CFRD: Multi-Stage Apple Ripeness Vision Benchmark Report\n\n")
        f.write("Generated from automated evaluation across independent test crops (408 apples) and external RawRipe dataset (213 apples).\n\n")
        f.write("## 1. Executive Summary: Test Set Benchmark Accuracies\n\n")
        f.write("| Model Architecture | Test Accuracy | Balanced Acc | Macro-Precision | Macro-Recall | Macro-F1 | External RawRipe Acc | Status |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for name, m in benchmark_results.items():
            ext_acc = m["external_rawripe_validation"]["exact_match_accuracy"] * 100
            status = "🟢 Exceeds 90% Target" if m["test_accuracy"] >= 0.90 else "🟡 Strong Baseline"
            f.write(f"| **{name}** | **{m['test_accuracy']*100:.2f}%** | {m['test_balanced_accuracy']*100:.2f}% | {m['test_precision']*100:.2f}% | {m['test_recall']*100:.2f}% | **{m['test_macro_f1']*100:.2f}%** | **{ext_acc:.2f}%** | {status} |\n")

        f.write("\n## 2. Per-Class Confusion Matrices (Independent Test Set: 408 Crops)\n\n")
        f.write("Classes: `0: Immature (Green)`, `1: Semi-Mature (Transition)`, `2: Mature (Red)`\n\n")
        for name, m in benchmark_results.items():
            cm = m["confusion_matrix"]
            f.write(f"### {name}\n\n")
            f.write("| True \\ Pred | Immature | Semi-Mature | Mature |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| **Immature (141)** | **{cm[0][0]}** | {cm[0][1]} | {cm[0][2]} |\n")
            f.write(f"| **Semi-Mature (136)** | {cm[1][0]} | **{cm[1][1]}** | {cm[1][2]} |\n")
            f.write(f"| **Mature (131)** | {cm[2][0]} | {cm[2][1]} | **{cm[2][2]}** |\n\n")

        f.write("## 3. External Dataset Generalization: RawRipe Zero-Shot Test\n\n")
        f.write("Evaluates whether models trained on the 2026 orchard dataset generalize to completely different apple photographs.\n\n")
        for name, m in benchmark_results.items():
            ext = m["external_rawripe_validation"]
            f.write(f"### {name}\n")
            f.write(f"- **Total External Samples**: {ext['total_external_samples']} apples (105 raw, 108 ripe)\n")
            f.write(f"- **Exact Match Accuracy**: **{ext['exact_match_accuracy']*100:.2f}%**\n")
            f.write(f"- **Raw (Immature) Recall**: {ext['raw_immature_recall']*100:.2f}%\n")
            f.write(f"- **Ripe (Mature) Recall**: {ext['ripe_mature_recall']*100:.2f}%\n")
            f.write(f"- **Predicted Distribution**: Immature: {ext['predicted_distribution']['immature']}, Semi-Mature: {ext['predicted_distribution']['semi_mature']}, Mature: {ext['predicted_distribution']['mature']}\n\n")

    print(f"[OK] Saved markdown report to: {md_path}")
    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND BENCHMARKS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
