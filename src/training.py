"""
training.py

Core ML training utilities for the treatment-condition classification pipeline.

Pipeline: StandardScaler → PCA(n_components) → LinearDiscriminantAnalysis
Validation: GroupKFold by fruit_id (NO row-level train_test_split)
"""

import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix,
)

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    CHANNEL_COLS, RANDOM_SEED, PCA_N_COMPONENTS,
    GROUPKFOLD_N_SPLITS, SYNTHETIC_LABEL, CLASS_LABELS,
)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def build_lda_pipeline(n_components: int = PCA_N_COMPONENTS,
                       random_state: int = RANDOM_SEED) -> Pipeline:
    """StandardScaler → PCA → LDA pipeline (primary baseline)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=random_state)),
        ("lda",    LinearDiscriminantAnalysis()),
    ])


def build_svm_pipeline(n_components: int = PCA_N_COMPONENTS,
                       C: float = 1.0,
                       random_state: int = RANDOM_SEED) -> Pipeline:
    """StandardScaler → PCA → SVM(RBF) pipeline (comparison baseline)."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=n_components, random_state=random_state)),
        ("svm",    SVC(kernel="rbf", C=C, probability=True, random_state=random_state)),
    ])


# ---------------------------------------------------------------------------
# GroupKFold cross-validation
# ---------------------------------------------------------------------------

def run_group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    n_splits: int = GROUPKFOLD_N_SPLITS,
    pipeline_name: str = "PCA+LDA",
    verbose: bool = True,
) -> dict:
    """
    Run GroupKFold cross-validation. Each fruit (group) appears in exactly
    one test fold. Scaler and PCA are fitted only on training data per fold.

    Parameters
    ----------
    X : np.ndarray (N, 18) — feature matrix (CHANNEL_COLS only, no fruit_id)
    y : np.ndarray (N,)    — class labels
    groups : np.ndarray (N,) — fruit_id per row
    pipeline : sklearn Pipeline
    n_splits : int — number of CV folds
    pipeline_name : str — label for reporting
    verbose : bool

    Returns
    -------
    dict with keys:
        fold_metrics: list of per-fold metric dicts
        oof_preds: np.ndarray (N,) — out-of-fold predictions
        oof_proba: np.ndarray (N, n_classes) — out-of-fold probabilities
        confusion: np.ndarray — pooled confusion matrix
        mean_acc, std_acc, mean_bal_acc, mean_f1: float
    """
    # Critical safety check: fruit_id must NOT be in X
    # (X should be a pure numpy array at this point, but check if caller
    #  accidentally passed a DataFrame with fruit_id column)
    if isinstance(X, pd.DataFrame):
        if "fruit_id" in X.columns:
            raise ValueError(
                "fruit_id found in feature DataFrame X. "
                "Remove it before calling run_group_kfold. "
                "Use X = df[CHANNEL_COLS].values"
            )
        X = X.values

    if X.ndim != 2 or X.shape[1] != len(CHANNEL_COLS):
        raise ValueError(
            f"Feature matrix X must have exactly {len(CHANNEL_COLS)} columns (got shape {X.shape}). "
            f"Verify that no metadata/fruit_id leaked into X."
        )

    n_classes = int(np.max(y)) + 1
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    oof_preds = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), n_classes))

    if verbose:
        print(f"\n{'='*60}")
        print(f"GroupKFold CV: {n_splits} folds  |  {pipeline_name}")
        print(f"Samples: {len(y)}  |  Unique fruits: {len(np.unique(groups))}")
        print(f"Classes: {[CLASS_LABELS.get(i, i) for i in sorted(np.unique(y))]}")
        print(f"{'='*60}")

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_fruits = np.unique(groups[train_idx])
        test_fruits  = np.unique(groups[test_idx])

        # Critical safety check: must survive python -O (assert would be stripped)
        overlap = set(train_fruits) & set(test_fruits)
        if len(overlap) != 0:
            raise RuntimeError(
                f"Fold {fold_idx}: fruit_id leakage detected: {overlap}. "
                "Check GroupKFold groups argument."
            )

        if verbose:
            print(f"\nFold {fold_idx+1}/{n_splits}")
            print(f"  Train fruits ({len(train_fruits)}): {sorted(train_fruits)}")
            print(f"  Test fruits  ({len(test_fruits)}):  {sorted(test_fruits)}")
            print(f"  Train rows: {len(train_idx)}  |  Test rows: {len(test_idx)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit on train, transform test (NO data leakage)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Probabilities — index by pipeline.classes_ to handle folds where
        # a class is absent from training data (would return fewer columns)
        try:
            proba = pipeline.predict_proba(X_test)
            for col_idx, cls in enumerate(pipeline.classes_):
                oof_proba[test_idx, cls] = proba[:, col_idx]
        except AttributeError:
            pass  # estimator has no predict_proba; oof_proba stays zeros

        oof_preds[test_idx] = y_pred

        acc     = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_metrics.append({
            "fold": fold_idx + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_fruits": len(train_fruits),
            "n_test_fruits": len(test_fruits),
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": f1,
        })

        if verbose:
            print(f"  Accuracy: {acc:.4f}  |  Balanced: {bal_acc:.4f}  |  F1: {f1:.4f}")

    # Pooled metrics
    accs     = [m["accuracy"] for m in fold_metrics]
    bal_accs = [m["balanced_accuracy"] for m in fold_metrics]
    f1s      = [m["macro_f1"] for m in fold_metrics]

    pooled_cm = confusion_matrix(y, oof_preds)

    result = {
        "pipeline_name": pipeline_name,
        "fold_metrics": fold_metrics,
        "oof_preds": oof_preds,
        "oof_proba": oof_proba,
        "confusion": pooled_cm,
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_bal_acc": float(np.mean(bal_accs)),
        "mean_f1":  float(np.mean(f1s)),
        "n_folds": n_splits,
        "label": SYNTHETIC_LABEL,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS — {pipeline_name}  [{SYNTHETIC_LABEL}]")
        print(f"  Mean accuracy:          {result['mean_acc']:.4f} ± {result['std_acc']:.4f}")
        print(f"  Mean balanced accuracy: {result['mean_bal_acc']:.4f}")
        print(f"  Mean macro-F1:          {result['mean_f1']:.4f}")
        print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, result: dict,
               model_dir: str, model_name: str) -> str:
    """
    Save fitted pipeline and CV metadata to model_dir.
    Returns the saved filepath.
    """
    os.makedirs(model_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{ts}.joblib"
    filepath = os.path.join(model_dir, filename)

    payload = {
        "pipeline": pipeline,
        "cv_result": result,          # full result including oof_preds, oof_proba, confusion
        "saved_at": ts,
        "channel_cols": CHANNEL_COLS,
        "label": SYNTHETIC_LABEL,
    }

    joblib.dump(payload, filepath)
    print(f"Model saved: {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Metrics table formatter
# ---------------------------------------------------------------------------

def format_fold_table(fold_metrics: list[dict]) -> str:
    """Return a formatted markdown table of per-fold metrics."""
    header = "| Fold | Train rows | Test rows | Accuracy | Bal. Accuracy | Macro-F1 |"
    sep    = "|---|---|---|---|---|---|"
    rows = [
        f"| {m['fold']} | {m['n_train']} | {m['n_test']} "
        f"| {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['macro_f1']:.4f} |"
        for m in fold_metrics
    ]
    accs = [m["accuracy"] for m in fold_metrics]
    bals = [m["balanced_accuracy"] for m in fold_metrics]
    f1s  = [m["macro_f1"] for m in fold_metrics]
    mean_row = (
        f"| **Mean** | — | — "
        f"| **{np.mean(accs):.4f} ± {np.std(accs):.4f}** "
        f"| **{np.mean(bals):.4f}** "
        f"| **{np.mean(f1s):.4f}** |"
    )
    return "\n".join([header, sep] + rows + [mean_row])
