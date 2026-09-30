"""
train_pesticide_differential.py
===============================
Phase 5 of CFRD_REAL_IMPLEMENTATION_PLAN.md:
Intra-Fruit Differential Agrochemical Screening with Strict LOAO Validation.

Physical Principle:
  Inter-apple cuticle wax shifts reflectance by up to 68%, while surface agrochemical
  residues contribute <2% signal. By referencing equatorial surface scans against
  a control spot on the SAME apple (calyx / bottom), the 68:1 confounding background
  is cancelled, exposing subtle chemical absorption curvature shifts.

Evaluation Integrity (Mandatory Gates):
  - Leave-One-Apple-Out (LOAO) cross-validation grouped strictly by fruit_id.
  - No fruit ever appears in both train and test partitions.
  - Reports Sensitivity, Specificity, AUROC, Wilson 95% CI, and majority baseline.
  - Zero fabricated numbers.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

# Ensure src is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from constants import (
    CHANNEL_COLS,
    SYNTHETIC_CSV_PATH,
    DISCLAIMER_BANNER,
    TABLES_DIR,
    MODELS_DIR,
)
from features import compute_intra_fruit_differential

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% normal quantile
    p = successes / total
    denom = 1.0 + (z**2 / total)
    centre = (p + (z**2 / (2.0 * total))) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / total) + (z**2 / (4.0 * (total**2))))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def prepare_differential_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct differential feature vectors for each fruit.
    Extracts control spot (first scan or designated control) and calculates
    intra-fruit differential features (17 adjacent differences) against equatorial spots.
    """
    fruits = df["fruit_id"].unique()
    X_diff_list = []
    y_list = []
    group_list = []

    for fruit in fruits:
        f_df = df[df["fruit_id"] == fruit]
        specs = f_df[CHANNEL_COLS].values

        if len(specs) < 2:
            # Need at least 1 control + 1 target
            continue

        # Use first scan as reference spot (calyx proxy) and remaining as targets
        control_spec = specs[0]
        target_specs = specs[1:]

        diff_res = compute_intra_fruit_differential(control_spec, [s for s in target_specs])
        # 17 curvature features
        feat = diff_res["delta_diff_17"]

        # Binary label: 0 for clean/control, 1 for treated
        is_treated = 1 if f_df["class_3"].iloc[0] > 0 else 0

        X_diff_list.append(feat)
        y_list.append(is_treated)
        group_list.append(fruit)

    return np.array(X_diff_list), np.array(y_list), np.array(group_list)


def run_loao_evaluation(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, Any]:
    """Execute strict Leave-One-Apple-Out evaluation."""
    logo = LeaveOneGroupOut()
    y_true = []
    y_pred = []
    y_prob = []

    n_samples = len(y)
    if n_samples < 4:
        raise ValueError("Insufficient fruit samples for LOAO cross-validation.")

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Ensure both classes exist in train fold
        if len(np.unique(y_train)) < 2:
            continue

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l2", C=1.0, random_state=42, max_iter=500)),
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)[0]
        prob = pipeline.predict_proba(X_test)[0, 1]

        y_true.append(y_test[0])
        y_pred.append(pred)
        y_prob.append(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5

    ci_acc = wilson_score_interval(int(np.sum(y_true == y_pred)), len(y_true))
    ci_spec = wilson_score_interval(tn, tn + fp)
    ci_sens = wilson_score_interval(tp, tp + fn)

    majority_baseline = max(np.mean(y_true == 1), np.mean(y_true == 0))

    return {
        "num_apples": len(y_true),
        "accuracy": round(float(acc), 4),
        "majority_baseline": round(float(majority_baseline), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auroc": round(float(auroc), 4),
        "confusion_matrix": cm.tolist(),
        "ci_accuracy_95": [round(ci_acc[0], 3), round(ci_acc[1], 3)],
        "ci_specificity_95": [round(ci_spec[0], 3), round(ci_spec[1], 3)],
        "ci_sensitivity_95": [round(ci_sens[0], 3), round(ci_sens[1], 3)],
    }


def main():
    logger.info("=" * 65)
    logger.info("CFRD Intra-Fruit Differential Screening (LOAO Protocol)")
    logger.info("=" * 65)

    # 1. Check data sources
    real_csv = PROJECT_ROOT / "data" / "pesticide_screening" / "roomi_real_18ch" / "roomi_real_18ch.csv"
    synth_csv = Path(SYNTHETIC_CSV_PATH)

    if real_csv.exists():
        data_path = real_csv
        data_type = "Real Roomi Extractions"
    elif synth_csv.exists():
        data_path = synth_csv
        data_type = "Synthetic Simulation Pilot"
    else:
        logger.error("No pesticide dataset found. Download datasets per README.md.")
        return

    logger.info(f"Loading dataset: {data_path.name} ({data_type})")
    df = pd.read_csv(data_path)

    # 2. Extract differential features
    X_diff, y_diff, groups = prepare_differential_dataset(df)
    logger.info(f"Differential instances extracted: N={len(y_diff)} apples across {len(np.unique(groups))} groups")

    # 3. LOAO Evaluation
    results = run_loao_evaluation(X_diff, y_diff, groups)

    logger.info("-" * 65)
    logger.info(f"LOAO Results Summary (N={results['num_apples']} apples):")
    logger.info(f"  * Accuracy:          {results['accuracy']:.1%} (Majority Baseline: {results['majority_baseline']:.1%})")
    logger.info(f"  * 95% Wilson CI:     [{results['ci_accuracy_95'][0]:.1%}, {results['ci_accuracy_95'][1]:.1%}]")
    logger.info(f"  * Sensitivity:       {results['sensitivity']:.1%} [{results['ci_sensitivity_95'][0]:.1%}, {results['ci_sensitivity_95'][1]:.1%}]")
    logger.info(f"  * Specificity:       {results['specificity']:.1%} [{results['ci_specificity_95'][0]:.1%}, {results['ci_specificity_95'][1]:.1%}]")
    logger.info(f"  * AUROC:             {results['auroc']:.3f}")
    logger.info(f"  * Confusion Matrix:  {results['confusion_matrix']}")
    logger.info("-" * 65)
    logger.info("Notice: Pre-hardware research prototype. Requires physical sensor + LC-MS/MS ground truth.")


if __name__ == "__main__":
    main()
