"""
15_train_master_brix_cultivar.py
================================
Trains and evaluates improved Sweetness (°Brix) and Cultivar models on the
109-apple master dataset (d:\\CFRD\\Brixsweetness\\master_essex_hyperspectral_dataset.csv).

Key Methods:
  1. Standard Normal Variate (SNV) normalization per spectrum: cancels cuticle scattering offsets
  2. Cultivar one-hot stratification: accounts for variety-specific baseline sugar bands
  3. Strict GroupKFold cross-validation on physical apple_id (zero group leakage)
  4. Saves calibrated models to models/brix/ and models/quality/
"""

import sys
import os
from pathlib import Path
import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    balanced_accuracy_score,
)
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MASTER_CSV = Path(r"d:\CFRD\Brixsweetness\master_essex_hyperspectral_dataset.csv")


def main():
    if not MASTER_CSV.exists():
        logger.error(f"Master CSV not found at {MASTER_CSV}")
        return

    logger.info("=" * 65)
    logger.info("CFRD Master Sweetness (°Brix) & Cultivar Training Pipeline")
    logger.info("=" * 65)

    df = pd.read_csv(MASTER_CSV)
    logger.info(f"Loaded master dataset: {len(df)} scans across {df['apple_id'].nunique()} unique apples")
    logger.info(f"Cultivars represented: {df['cultivar'].value_counts().to_dict()}")

    ch_cols = [c for c in df.columns if c.startswith("as7265x_")]
    X_raw = df[ch_cols].values
    y_brix = df["brix"].values
    y_cult = df["cultivar"].values
    groups = df["apple_id"].values

    # -----------------------------------------------------------------------
    # Part 1: Sweetness (°Brix) Regression with SNV + Cultivar
    # -----------------------------------------------------------------------
    logger.info("-" * 65)
    logger.info("Part 1: Training Sweetness (°Brix) Regressor with SNV + Cultivar")

    # Apply Standard Normal Variate (SNV) to remove surface peel scattering
    X_snv = (X_raw - X_raw.mean(axis=1, keepdims=True)) / (X_raw.std(axis=1, keepdims=True) + 1e-6)

    # One-hot cultivars
    cult_dummies = pd.get_dummies(df["cultivar"], drop_first=True)
    cult_cols = list(cult_dummies.columns)
    X_all = np.hstack([X_snv, cult_dummies.astype(float).values])

    gkf = GroupKFold(n_splits=5)
    brix_preds = np.zeros(len(y_brix))

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_all, y_brix, groups=groups), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_idx])
        X_te = scaler.transform(X_all[test_idx])
        model = Ridge(alpha=10.0, random_state=42)
        model.fit(X_tr, y_brix[train_idx])
        brix_preds[test_idx] = model.predict(X_te)

    r2 = r2_score(y_brix, brix_preds)
    rmse = np.sqrt(mean_squared_error(y_brix, brix_preds))
    mae = mean_absolute_error(y_brix, brix_preds)

    logger.info(f"Cross-Validated GroupKFold Results (N={len(y_brix)}):")
    logger.info(f"  * R² Score:  {r2:.4f}  (Prior verified baseline: -0.03)")
    logger.info(f"  * RMSE:      {rmse:.4f} °Bx")
    logger.info(f"  * MAE:       {mae:.4f} °Bx")

    # Fit final production model on full dataset
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_brix_model = Ridge(alpha=10.0, random_state=42)
    final_brix_model.fit(X_all_scaled, y_brix)

    brix_bundle = {
        "model": final_brix_model,
        "scaler": final_scaler,
        "cultivar_columns": cult_cols,
        "metrics": {"r2": r2, "rmse": rmse, "mae": mae, "n_apples": int(df['apple_id'].nunique())},
        "snv_enabled": True,
    }

    out_brix_1 = PROJECT_ROOT / "models" / "quality" / "brix_as7265x_18channel_best.joblib"
    out_brix_2 = PROJECT_ROOT / "models" / "brix" / "brix_as7265x_18channel_best.joblib"
    joblib.dump(brix_bundle, out_brix_1)
    joblib.dump(brix_bundle, out_brix_2)
    logger.info(f"Saved improved Brix model to {out_brix_1.name}")

    # -----------------------------------------------------------------------
    # Part 2: Cultivar Classification on 18-channel reflectance
    # -----------------------------------------------------------------------
    logger.info("-" * 65)
    logger.info("Part 2: Training Cultivar Classifier on 18-Channel Spectral Data")

    cult_preds = []
    cult_trues = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y_cult, groups=groups), 1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[train_idx])
        X_te = scaler.transform(X_raw[test_idx])
        clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        clf.fit(X_tr, y_cult[train_idx])
        cult_preds.extend(clf.predict(X_te))
        cult_trues.extend(y_cult[test_idx])

    cult_acc = accuracy_score(cult_trues, cult_preds)
    cult_bal = balanced_accuracy_score(cult_trues, cult_preds)

    logger.info(f"Cross-Validated Cultivar Results (5 classes, N={len(y_cult)}):")
    logger.info(f"  * Accuracy:          {cult_acc:.1%}  (Prior baseline: 47.5%)")
    logger.info(f"  * Balanced Accuracy: {cult_bal:.1%}")

    # Fit final production cultivar model
    final_cult_scaler = StandardScaler()
    X_cult_scaled = final_cult_scaler.fit_transform(X_raw)
    final_cult_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    final_cult_clf.fit(X_cult_scaled, y_cult)

    from sklearn.pipeline import Pipeline
    cult_pipe = Pipeline([("scaler", final_cult_scaler), ("classifier", final_cult_clf)])

    cult_bundle = {
        "pipeline": cult_pipe,
        "classes": list(final_cult_clf.classes_),
        "metrics": {"acc": cult_acc, "bal_acc": cult_bal, "model_type": "RandomForest on 18-Channel Spectrum"},
    }

    out_cult_1 = PROJECT_ROOT / "models" / "quality" / "variety_classifier_master.joblib"
    joblib.dump(cult_bundle, out_cult_1)
    logger.info(f"Saved improved Cultivar model to {out_cult_1.name}")

    logger.info("=" * 65)
    logger.info("Master Training Complete!")


if __name__ == "__main__":
    main()
