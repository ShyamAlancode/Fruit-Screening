"""
10_train_brix_regression.py

Trains a Brix sweetness regression model from the 16-band multispectral apple dataset.
Each 1080x17280 image strip is sliced into 16 narrow spectral band tiles (1080x1080 each).
The central ROI mean intensity per band forms a 16-dimensional feature vector.

Target: Soluble solids content (°Brix) from ground-truth folder labels.
Model: Cross-validated Ridge regression and PLSR.
Mandatory note: Software demonstration only — not validated against AS7265x hardware.

Usage:
    python scripts/10_train_brix_regression.py
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import glob
import re
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    SWEETNESS_ROOT,
    BRIX_LIMITATION,
    DISCLAIMER_BANNER,
    RANDOM_SEED,
)

MODELS_DIR = r"d:\CFRD\project\models\quality"
FIGURES_DIR = r"d:\CFRD\project\reports\figures"
TABLES_DIR = r"d:\CFRD\project\reports\tables"
CACHE_PATH = r"d:\CFRD\project\data\quality\sweetness_16band_features.csv"


def extract_16band_vector(img_path: str) -> np.ndarray:
    """Extract 16 mean tile intensities from a 1080x17280 image strip."""
    with Image.open(img_path) as im:
        # Convert to grayscale for band intensity
        gray = np.array(im.convert("L"), dtype=float)
    
    H, W = gray.shape
    tile_w = W // 16
    band_means = np.zeros(16, dtype=float)
    
    for b in range(16):
        tile = gray[:, b * tile_w : (b + 1) * tile_w]
        # Central ROI (middle 40% of tile)
        ch, cw = tile.shape
        r_h, r_w = int(0.2 * ch), int(0.2 * cw)
        roi = tile[r_h : ch - r_h, r_w : cw - r_w]
        band_means[b] = roi.mean()
        
    return band_means


def load_or_extract_dataset(max_images_per_class: int = 25) -> pd.DataFrame:
    """Load cached features or extract from images."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached features: {CACHE_PATH}")
        return pd.read_csv(CACHE_PATH)
        
    print(f"Extracting 16-band features from: {SWEETNESS_ROOT}")
    rows = []
    
    # Folders are Brix values: 10, 12, 12.1, 12.5, 12.8, 13, 13.8, 15
    for entry in sorted(os.listdir(SWEETNESS_ROOT)):
        entry_path = os.path.join(SWEETNESS_ROOT, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            brix = float(entry)
        except ValueError:
            continue
            
        all_files = [
            os.path.join(entry_path, f)
            for f in sorted(os.listdir(entry_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        img_files = all_files[:max_images_per_class]
        print(f"  Brix {brix}°Bx: processing {len(img_files)} images...")
        
        for p in img_files:
            try:
                bands = extract_16band_vector(p)
                row = {
                    "filename": os.path.basename(p),
                    "brix": brix,
                }
                for i, v in enumerate(bands):
                    row[f"band_{i+1}"] = round(float(v), 4)
                rows.append(row)
            except Exception as e:
                print(f"    [WARN] Failed {p}: {e}")
                
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Cached {len(df)} samples to {CACHE_PATH}")
    return df


def main():
    print(f"\n{DISCLAIMER_BANNER}\n")
    print(f"{BRIX_LIMITATION}\n")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_or_extract_dataset()
    band_cols = [c for c in df.columns if c.startswith("band_")]
    X = df[band_cols].values
    y = df["brix"].values

    print(f"\nDataset: {len(df)} samples, 16 spectral bands")
    print(f"Brix range: [{y.min():.1f}, {y.max():.1f}] °Bx, Mean = {y.mean():.2f}")

    # 5-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y))

    for train_idx, test_idx in kf.split(X, y):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        oof_preds[test_idx] = model.predict(X[test_idx])

    r2 = r2_score(y, oof_preds)
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    mae = mean_absolute_error(y, oof_preds)

    print(f"\nCross-Validated Ridge Regression Results:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f} °Bx")
    print(f"  MAE:       {mae:.4f} °Bx")

    # Fit final model on all data and save
    final_model = Ridge(alpha=1.0)
    final_model.fit(X, y)
    save_path = os.path.join(MODELS_DIR, f"brix_ridge_model_{ts}.joblib")
    joblib.dump({
        "model": final_model,
        "band_cols": band_cols,
        "metrics": {"r2": r2, "rmse": rmse, "mae": mae},
        "disclaimer": BRIX_LIMITATION,
    }, save_path)
    print(f"\nModel saved: {save_path}")

    # Plot True vs Predicted Brix
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, oof_preds, color="darkorange", alpha=0.7, edgecolors="k", s=50)
    ideal = [min(y.min(), oof_preds.min()) - 0.5, max(y.max(), oof_preds.max()) + 0.5]
    ax.plot(ideal, ideal, "b--", label="1:1 Perfect Prediction")
    ax.set_xlabel("Reference Brix (°Bx)")
    ax.set_ylabel("Predicted Brix (°Bx)")
    ax.set_title(f"Brix Sweetness Prediction (16-Band Multispectral)\nR² = {r2:.3f}, RMSE = {rmse:.2f} °Bx")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    fig_path = os.path.join(FIGURES_DIR, f"brix_prediction_{ts}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Save summary report
    report_md = f"""# Brix Sweetness Regression Report

> **Limitation:** {BRIX_LIMITATION}

## Performance (5-Fold Cross-Validation)
- **Model:** Ridge Regression (L2 regularization)
- **R² Score:** {r2:.4f}
- **RMSE:** {rmse:.4f} °Bx
- **MAE:** {mae:.4f} °Bx
- **Number of samples:** {len(df)}

## Model Path
`{save_path}`
"""
    table_path = os.path.join(TABLES_DIR, f"brix_regression_{ts}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Report saved: {table_path}")


if __name__ == "__main__":
    main()
