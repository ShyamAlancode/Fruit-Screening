"""
common.py
=========
Shared task registry, data loading, and evaluation utilities for the
plain-CLI apple ML models (pesticide treatment, ripeness, variety, brix).

No UI. No dashboard. Just: load data -> cross-validate honestly ->
train final model -> save bundle -> predict on new rows.

Every task carries an explicit `reliability` string that gets printed
on every train/predict run. This is not decoration -- the pesticide
and brix tasks are, on current data, at or below chance / negative R^2.
A model bundle does not get to hide that from whoever runs predict.py
six months from now having forgotten this conversation.
"""

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    r2_score, mean_absolute_error,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class TaskSpec:
    name: str
    kind: str                      # "classification" | "regression"
    csv_path: Path
    feature_cols: list[str]        # empty -> use feature_prefix
    feature_prefix: str
    target_col: str
    group_col: Optional[str]       # column to group by; None -> derive
    group_derive: Optional[Callable]  # fn(df) -> group series, if group_col is None
    chance_level: Optional[float]  # classification baseline
    reliability: str
    model_builder: Callable[[], Pipeline]


def _pesticide_groups(df: pd.DataFrame) -> pd.Series:
    return df["fruit_id"]


def _brix_groups(df: pd.DataFrame) -> pd.Series:
    # filenames look like apple1_1_k.jpg -> group by physical fruit "apple1"
    if "filename" in df.columns:
        return df["filename"].astype(str).str.extract(r"^(apple\d+)", expand=False).fillna("unknown")
    return pd.Series(np.arange(len(df)))


def _lda_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=5, random_state=42)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def _rf_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=60, max_depth=8, random_state=42)),
    ])


def _ridge_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


TASKS: dict[str, TaskSpec] = {

    "pesticide3": TaskSpec(
        name="pesticide3",
        kind="classification",
        csv_path=DATA_DIR / "pesticide_screening/roomi_real_hsi_18ch/roomi_real_hsi_18ch.csv",
        feature_cols=[], feature_prefix="ch_",
        target_col="class_3",
        group_col="fruit_id", group_derive=None,
        chance_level=1 / 3,
        reliability=(
            "UNRELIABLE. 13 physical fruits / 26 scans, 3-class GroupKFold. "
            "Last verified real-data run: 15.00% accuracy vs 33.3% chance -- "
            "BELOW random guessing. Do not use for real decisions. "
            "See project README 'Current Validated Scientific Status'."
        ),
        model_builder=_lda_pipeline,
    ),

    "pesticide_binary": TaskSpec(
        name="pesticide_binary",
        kind="classification",
        csv_path=DATA_DIR / "pesticide_screening/roomi_real_hsi_18ch/roomi_real_hsi_18ch.csv",
        feature_cols=[], feature_prefix="ch_",
        target_col="__binary_treated__",   # derived: control vs any treatment
        group_col="fruit_id", group_derive=None,
        chance_level=0.5,
        reliability=(
            "UNRELIABLE. Every binary treatment split (control vs treated, "
            "per-chemical, per-dose) evaluated at or below the 50% coin-flip "
            "line in prior verified runs, with std devs up to +/-45.8%. "
            "13 fruits is not enough data for this claim, full stop."
        ),
        model_builder=_lda_pipeline,
    ),

    "ripeness": TaskSpec(
        name="ripeness",
        kind="classification",
        csv_path=DATA_DIR / "quality/ripeness_variety_features.csv",
        feature_cols=[], feature_prefix="feat_",
        target_col="ripeness_class",
        group_col="group_id", group_derive=None,
        chance_level=1 / 3,
        reliability=(
            "WORKING, MODEST SIGNAL. 20 physical fruits, GroupKFold. "
            "Last verified: 64.07% accuracy vs 33.3% chance. Real but not "
            "product-grade -- treat as a coarse 3-bucket indicator, not a spec."
        ),
        model_builder=_rf_pipeline,
    ),

    "variety": TaskSpec(
        name="variety",
        kind="classification",
        csv_path=DATA_DIR / "quality/ripeness_variety_features.csv",
        feature_cols=[], feature_prefix="feat_",
        target_col="variety_class",
        group_col="group_id", group_derive=None,
        chance_level=1 / 5,
        reliability=(
            "WORKING, MODEST SIGNAL. 20 physical fruits, 5-class GroupKFold. "
            "Last verified: 47.54% accuracy vs 20.0% chance. Real signal, "
            "well above chance, but not precise enough to be a sole arbiter."
        ),
        model_builder=_rf_pipeline,
    ),

    "brix": TaskSpec(
        name="brix",
        kind="regression",
        csv_path=DATA_DIR / "quality/sweetness_16band_features.csv",
        feature_cols=[], feature_prefix="band_",
        target_col="brix",
        group_col=None, group_derive=_brix_groups,
        chance_level=None,
        reliability=(
            "DOES NOT WORK. Only 4 physical fruits behind this file. Every "
            "documented variant (real 204-band hyperspectral, AS7265x digital "
            "twin, with/without cultivar features) produced R^2 between -0.95 "
            "and +0.07 -- at best noise, at worst worse than predicting the "
            "mean every time. This model will run and produce numbers. "
            "Do not trust those numbers for anything."
        ),
        model_builder=_ridge_pipeline,
    ),
}


def load_task(task: TaskSpec) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (df, X, y, groups) for a task, with feature_cols resolved."""
    df = pd.read_csv(task.csv_path)

    if not task.feature_cols and task.feature_prefix:
        task.feature_cols = [c for c in df.columns if c.startswith(task.feature_prefix)]

    if task.target_col == "__binary_treated__":
        df = df.copy()
        df["__binary_treated__"] = (df["level"].astype(str).str.lower() != "control").astype(int)

    X = df[task.feature_cols].to_numpy(dtype=float)
    y = df[task.target_col].to_numpy()

    if task.group_col is not None:
        groups = df[task.group_col].to_numpy()
    else:
        groups = task.group_derive(df).to_numpy()

    return df, X, y, groups


def cross_validate(task: TaskSpec, X, y, groups) -> dict:
    """Honest GroupKFold (or LOGO if too few groups) evaluation. No shortcuts."""
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)

    if n_splits < 2:
        raise ValueError(f"Only {n_groups} group(s) in data -- cannot cross-validate.")

    splitter = LeaveOneGroupOut() if n_groups <= 5 else GroupKFold(n_splits=n_splits)

    fold_scores = []
    fold_secondary = []

    for train_idx, test_idx in splitter.split(X, y, groups):
        model = task.model_builder()
        with warnings.catch_warnings():
            # Suppress expected small-sample label skew warnings in held-out folds
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])

            if task.kind == "classification":
                fold_scores.append(accuracy_score(y[test_idx], preds))
                fold_secondary.append(balanced_accuracy_score(y[test_idx], preds))
            else:
                fold_scores.append(r2_score(y[test_idx], preds))
                fold_secondary.append(mean_absolute_error(y[test_idx], preds))

    result = {
        "n_folds": len(fold_scores),
        "n_groups": n_groups,
        "n_samples": len(y),
    }
    if task.kind == "classification":
        result["mean_accuracy"] = float(np.mean(fold_scores))
        result["std_accuracy"] = float(np.std(fold_scores))
        result["mean_balanced_accuracy"] = float(np.mean(fold_secondary))
        result["chance_level"] = task.chance_level
    else:
        result["mean_r2"] = float(np.mean(fold_scores))
        result["std_r2"] = float(np.std(fold_scores))
        result["mean_mae"] = float(np.mean(fold_secondary))

    return result


def save_bundle(task: TaskSpec, model, cv_result: dict, classes=None):
    bundle = {
        "task": task.name,
        "kind": task.kind,
        "feature_cols": task.feature_cols,
        "reliability": task.reliability,
        "cv_result": cv_result,
        "classes": list(classes) if classes is not None else None,
    }
    import joblib
    path = MODELS_DIR / f"{task.name}.joblib"
    joblib.dump({"model": model, "meta": bundle}, path)
    return path


def print_report(task: TaskSpec, cv_result: dict):
    print(f"\n{'='*70}")
    print(f"TASK: {task.name}  ({task.kind})")
    print(f"{'='*70}")
    print(f"Samples: {cv_result['n_samples']}  |  Groups (physical fruits): {cv_result['n_groups']}  |  Folds: {cv_result['n_folds']}")
    if task.kind == "classification":
        acc = cv_result["mean_accuracy"]
        std = cv_result["std_accuracy"]
        chance = cv_result["chance_level"]
        flag = "BELOW CHANCE" if acc < chance else ("AT CHANCE" if abs(acc - chance) < 0.03 else "ABOVE CHANCE")
        print(f"Accuracy: {acc:.2%} +/- {std:.2%}   (chance = {chance:.2%})   [{flag}]")
        print(f"Balanced accuracy: {cv_result['mean_balanced_accuracy']:.2%}")
    else:
        print(f"R^2: {cv_result['mean_r2']:.4f} +/- {cv_result['std_r2']:.4f}   (R^2 <= 0 means worse than predicting the mean)")
        print(f"MAE: {cv_result['mean_mae']:.4f}")
    print(f"\nRELIABILITY: {task.reliability}")
    print(f"{'='*70}\n")
