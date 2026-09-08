"""
09_train_ripeness_variety.py

Phase 9: Ripeness and Variety Classification
Parses the dataset: Ripeness Grading/day X / variety / APPLE Y / apple*.jpg
Groups by: {variety}_{apple_number} (GroupKFold to prevent apple leakage across folds)

Tasks:
1. Ripeness Grading (3 classes):
   - Under-ripe: Days 1–6 (class 0)
   - Ripe:       Days 7–12 (class 1)
   - Over-ripe:  Days 13–18 (class 2)
2. Variety Grading (5 classes):
   - Kinnaur (0)
   - Red Delicious New Zealand (1)
   - Red Delicious USA (2)
   - Royal Gala (3)
   - Washington (4)

Uses color + spatial feature extraction (HSV color moments + histogram) + GroupKFold classifier
for fast, reliable CPU cross-validation and evaluation.

Usage:
    python scripts/09_train_ripeness_variety.py
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import re
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    RIPENESS_ROOT,
    DISCLAIMER_BANNER,
    RANDOM_SEED,
)

MODELS_DIR = r"d:\CFRD\project\models\quality"
FIGURES_DIR = r"d:\CFRD\project\reports\figures"
TABLES_DIR = r"d:\CFRD\project\reports\tables"
CACHE_PATH = r"d:\CFRD\project\data\quality\ripeness_variety_features.csv"

VARIETIES = [
    "kinnaur",
    "red delicious new zealand",
    "red delicious usa",
    "royal gala",
    "washington",
]

RIPENESS_NAMES = {0: "Under-ripe", 1: "Ripe", 2: "Over-ripe"}


def extract_visual_features(img_path: str) -> np.ndarray:
    """Extract HSV moments and RGB color statistics."""
    with Image.open(img_path) as im:
        # Resize for consistent rapid feature extraction
        im_small = im.resize((128, 128)).convert("RGB")
        rgb = np.array(im_small, dtype=float) / 255.0

    # RGB means and stds
    rgb_means = rgb.mean(axis=(0, 1))
    rgb_stds  = rgb.std(axis=(0, 1))

    # HSV conversion
    hsv = im_small.convert("HSV")
    hsv_arr = np.array(hsv, dtype=float) / 255.0
    hsv_means = hsv_arr.mean(axis=(0, 1))
    hsv_stds  = hsv_arr.std(axis=(0, 1))

    # 10-bin histograms per channel
    h_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=8, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv_arr[:, :, 1], bins=8, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv_arr[:, :, 2], bins=8, range=(0, 1), density=True)

    feats = np.concatenate([rgb_means, rgb_stds, hsv_means, hsv_stds, h_hist, s_hist, v_hist])
    return feats


def collect_dataset(max_images_per_apple: int = 3) -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)

    print(f"Scanning Ripeness Grading folder: {RIPENESS_ROOT}")
    rows = []

    # Iterate over day 1 to day 18
    for d in range(1, 19):
        day_dir = os.path.join(RIPENESS_ROOT, f"day {d}")
        if not os.path.isdir(day_dir):
            continue

        # Ripeness class: 1-6 -> Under-ripe (0), 7-12 -> Ripe (1), 13-18 -> Over-ripe (2)
        if d <= 6:
            rip_class = 0
        elif d <= 12:
            rip_class = 1
        else:
            rip_class = 2

        for var_name in os.listdir(day_dir):
            var_dir = os.path.join(day_dir, var_name)
            if not os.path.isdir(var_dir):
                continue

            var_clean = var_name.lower().strip()
            if var_clean not in VARIETIES:
                continue
            var_idx = VARIETIES.index(var_clean)

            for apple_dir_name in os.listdir(var_dir):
                apple_dir = os.path.join(var_dir, apple_dir_name)
                if not os.path.isdir(apple_dir):
                    continue

                # Unique group: same physical apple tracked across days
                group_id = f"{var_clean}_{apple_dir_name.lower().replace(' ', '')}"

                img_files = [
                    os.path.join(apple_dir, f)
                    for f in sorted(os.listdir(apple_dir))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ][:max_images_per_apple]

                for img_p in img_files:
                    try:
                        feats = extract_visual_features(img_p)
                        row = {
                            "filename": os.path.basename(img_p),
                            "day": d,
                            "ripeness_class": rip_class,
                            "variety_name": var_clean,
                            "variety_class": var_idx,
                            "group_id": group_id,
                        }
                        for fi, fval in enumerate(feats):
                            row[f"feat_{fi+1}"] = round(float(fval), 5)
                        rows.append(row)
                    except Exception as e:
                        pass

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Collected and cached {len(df)} samples across {df['group_id'].nunique()} unique apples")
    return df


def main():
    print(f"\n{DISCLAIMER_BANNER}\n")
    print("Ripeness & Variety GroupKFold Classification\n")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = collect_dataset(max_images_per_apple=3)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat_cols].values
    groups = df["group_id"].values
    n_fruits = len(np.unique(groups))

    print(f"Total rows: {len(df)}, Features: {len(feat_cols)}, Unique Apples: {n_fruits}")

    # -------------------------------------------------------------
    # Task 1: Ripeness Classification (3 classes)
    # -------------------------------------------------------------
    y_rip = df["ripeness_class"].values
    print("\n" + "=" * 60)
    print("TASK 1: Ripeness Classification (Under-ripe / Ripe / Over-ripe)")
    print(f"Classes: {RIPENESS_NAMES}")

    gkf = GroupKFold(n_splits=5)
    rip_oof = np.zeros(len(y_rip), dtype=int)

    for tr, te in gkf.split(X, y_rip, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=60, max_depth=8, random_state=RANDOM_SEED)),
        ])
        clf.fit(X[tr], y_rip[tr])
        rip_oof[te] = clf.predict(X[te])

    rip_acc = accuracy_score(y_rip, rip_oof)
    rip_bal = balanced_accuracy_score(y_rip, rip_oof)
    rip_f1  = f1_score(y_rip, rip_oof, average="macro")
    cm_rip  = confusion_matrix(y_rip, rip_oof)

    print(f"Ripeness GroupKFold Results:")
    print(f"  Accuracy:          {rip_acc*100:.2f}%")
    print(f"  Balanced Accuracy: {rip_bal*100:.2f}%")
    print(f"  Macro-F1:          {rip_f1*100:.2f}%")

    # Save final Ripeness model
    final_rip = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=8, random_state=RANDOM_SEED)),
    ])
    final_rip.fit(X, y_rip)
    rip_model_path = os.path.join(MODELS_DIR, f"ripeness_classifier_{ts}.joblib")
    joblib.dump({
        "pipeline": final_rip,
        "classes": list(RIPENESS_NAMES.values()),
        "metrics": {"acc": rip_acc, "bal_acc": rip_bal, "f1": rip_f1},
    }, rip_model_path)
    print(f"Saved: {rip_model_path}")

    # Plot Ripeness Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_rip, annot=True, fmt="d", cmap="Greens",
                xticklabels=list(RIPENESS_NAMES.values()),
                yticklabels=list(RIPENESS_NAMES.values()), ax=ax)
    ax.set_title(f"Ripeness Classification (GroupKFold)\nAccuracy: {rip_acc*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    rip_fig_path = os.path.join(FIGURES_DIR, f"confusion_ripeness_{ts}.png")
    fig.savefig(rip_fig_path, dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------
    # Task 2: Variety Classification (5 classes)
    # -------------------------------------------------------------
    y_var = df["variety_class"].values
    print("\n" + "=" * 60)
    print("TASK 2: Variety Classification (5 Apple Varieties)")
    print(f"Varieties: {VARIETIES}")

    var_oof = np.zeros(len(y_var), dtype=int)
    for tr, te in gkf.split(X, y_var, groups):
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=60, max_depth=8, random_state=RANDOM_SEED)),
        ])
        clf.fit(X[tr], y_var[tr])
        var_oof[te] = clf.predict(X[te])

    var_acc = accuracy_score(y_var, var_oof)
    var_bal = balanced_accuracy_score(y_var, var_oof)
    var_f1  = f1_score(y_var, var_oof, average="macro")
    cm_var  = confusion_matrix(y_var, var_oof)

    print(f"Variety GroupKFold Results:")
    print(f"  Accuracy:          {var_acc*100:.2f}%")
    print(f"  Balanced Accuracy: {var_bal*100:.2f}%")
    print(f"  Macro-F1:          {var_f1*100:.2f}%")

    # Save final Variety model
    final_var = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=8, random_state=RANDOM_SEED)),
    ])
    final_var.fit(X, y_var)
    var_model_path = os.path.join(MODELS_DIR, f"variety_classifier_{ts}.joblib")
    joblib.dump({
        "pipeline": final_var,
        "classes": VARIETIES,
        "metrics": {"acc": var_acc, "bal_acc": var_bal, "f1": var_f1},
    }, var_model_path)
    print(f"Saved: {var_model_path}")

    # Plot Variety Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    var_short = [v.replace("red delicious ", "RD ") for v in VARIETIES]
    sns.heatmap(cm_var, annot=True, fmt="d", cmap="Oranges",
                xticklabels=var_short, yticklabels=var_short, ax=ax)
    ax.set_title(f"Variety Classification (GroupKFold)\nAccuracy: {var_acc*100:.1f}%")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    var_fig_path = os.path.join(FIGURES_DIR, f"confusion_variety_{ts}.png")
    fig.savefig(var_fig_path, dpi=150)
    plt.close(fig)

    # Save Markdown Summary
    report_md = f"""# Ripeness and Variety Classification Results

Leakage-Free Evaluation: GroupKFold by physical apple identifier (`variety_apple`).

## Task 1: Ripeness (3-Class)
- **Classes:** Under-ripe (Days 1–6), Ripe (Days 7–12), Over-ripe (Days 13–18)
- **Accuracy:** {rip_acc*100:.2f}%
- **Balanced Accuracy:** {rip_bal*100:.2f}%
- **Macro-F1:** {rip_f1*100:.2f}%
- **Model:** `{rip_model_path}`

## Task 2: Variety (5-Class)
- **Classes:** {', '.join(VARIETIES)}
- **Accuracy:** {var_acc*100:.2f}%
- **Balanced Accuracy:** {var_bal*100:.2f}%
- **Macro-F1:** {var_f1*100:.2f}%
- **Model:** `{var_model_path}`
"""
    table_path = os.path.join(TABLES_DIR, f"ripeness_variety_{ts}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nReport saved: {table_path}")


if __name__ == "__main__":
    main()
