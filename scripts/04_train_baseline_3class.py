"""
04_train_baseline_3class.py

Train the core 3-class treatment-condition classifier on the Roomi synthetic
18-channel AS7265x CSV.

Pipeline:  StandardScaler → PCA(5) → LDA
Validation: 5-fold GroupKFold by fruit_id

Expected benchmark (synthetic):
  Mean accuracy:          65.57% ± 5.34%
  Mean balanced accuracy: 65.01%
  Mean macro-F1:          62.37%

Usage:
    cd d:\\CFRD\\project
    python scripts\\04_train_baseline_3class.py
"""

import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime

import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import (
    CHANNEL_COLS, REQUIRED_META_COLS, SYNTHETIC_CSV_PATH,
    ROOMI_REAL_18CH_PATH, ROOMI_SYNTHETIC_18CH_PATH,
    RANDOM_SEED, PCA_N_COMPONENTS, GROUPKFOLD_N_SPLITS,
    CLASS_LABELS, SYNTHETIC_LABEL, DISCLAIMER_BANNER,
    MODELS_DIR, REPORTS_DIR, PROJECT_ROOT,
)
from training import (
    build_lda_pipeline, build_svm_pipeline,
    run_group_kfold, save_model, format_fold_table,
)
from data_validation import validate_synthetic_csv
from ood import MahalanobisOOD
from model_registry import register_model, compute_sha256, get_git_commit

REPORT_DIR    = REPORTS_DIR
FIGURES_DIR   = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR    = os.path.join(REPORTS_DIR, "tables")
MODELS_DIR_TC = os.path.join(MODELS_DIR, "treatment_condition")


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3-class treatment-condition classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset to train on: 'synthetic' (240 rows / 20 fruits), 'real' (26 rows / 13 fruits), or explicit CSV path."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n{DISCLAIMER_BANNER}\n")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR_TC, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve dataset path
    if args.dataset == "synthetic":
        csv_path = ROOMI_SYNTHETIC_18CH_PATH
        is_synthetic = True
        status_label = "Synthetic Simulation Pipeline Test (Synthetic construct bias; proves ML pipeline learns label shifts)"
        model_key = "treatment_3class_synthetic"
    elif args.dataset == "real":
        csv_path = ROOMI_REAL_18CH_PATH
        is_synthetic = False
        status_label = "Real Roomi Extraction Pilot (Below random chance 33.3%; sample size 13 fruits inadequate for product claim)"
        model_key = "treatment_3class_real"
    else:
        csv_path = args.dataset
        is_synthetic = "synthetic" in csv_path.lower()
        status_label = "Custom Dataset Run"
        model_key = f"treatment_3class_custom_{ts}"

    # ------------------------------------------------------------------
    # 1. Load and validate CSV
    # ------------------------------------------------------------------
    print(f"Loading dataset: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: Dataset not found at {csv_path}")
        sys.exit(1)

    val_result = validate_synthetic_csv(csv_path, REPORT_DIR, verbose=False)
    if not val_result["passed"]:
        print("DATA VALIDATION FAILED. Aborting.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded: {df.shape[0]} rows, {df['fruit_id'].nunique()} unique fruit IDs")

    X      = df[CHANNEL_COLS].values
    y      = df["class_3"].values
    groups = df["fruit_id"].values

    # Safety assertion: fruit_id must NOT be in X
    assert "fruit_id" not in CHANNEL_COLS, "fruit_id found in CHANNEL_COLS!"

    # ------------------------------------------------------------------
    # 2. Primary baseline: PCA + LDA
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Training: StandardScaler → PCA(5) → LDA")
    lda_pipeline = build_lda_pipeline(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    lda_result   = run_group_kfold(X, y, groups, lda_pipeline, n_splits=GROUPKFOLD_N_SPLITS,
                                    pipeline_name="PCA+LDA", verbose=True)

    # ------------------------------------------------------------------
    # 3. Comparison: PCA + SVM
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Training: StandardScaler → PCA(5) → SVM(RBF) [comparison]")
    svm_pipeline = build_svm_pipeline(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    svm_result   = run_group_kfold(X, y, groups, svm_pipeline, n_splits=GROUPKFOLD_N_SPLITS,
                                    pipeline_name="PCA+SVM(RBF)", verbose=True)

    # ------------------------------------------------------------------
    # 4. Save OOF predictions
    # ------------------------------------------------------------------
    oof_preds_path = os.path.join(os.path.dirname(csv_path), "oof_predictions_3class.csv")
    oof_df = df[["fruit_id", "treatment_family", "level", "class_3"]].copy()
    oof_df["oof_pred_lda"] = lda_result["oof_preds"]
    for i, col in enumerate(CLASS_LABELS.values()):
        oof_df[f"proba_{col.replace(' ', '_')}"] = lda_result["oof_proba"][:, i] if lda_result["oof_proba"].shape[1] > i else 0.0
    os.makedirs(os.path.dirname(oof_preds_path), exist_ok=True)
    oof_df.to_csv(oof_preds_path, index=False)
    print(f"\nOOF predictions saved: {oof_preds_path}")

    # ------------------------------------------------------------------
    # 5. Confusion matrix plot
    # ------------------------------------------------------------------
    cm = lda_result["confusion"]
    class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS)]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"3-Class Treatment-Condition — PCA+LDA\n"
        f"Mean Acc: {lda_result['mean_acc']:.3f} ± {lda_result['std_acc']:.3f}  |  {SYNTHETIC_LABEL}"
    )
    cm_path = os.path.join(FIGURES_DIR, f"confusion_3class_lda_{ts}.png")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved: {cm_path}")

    # ------------------------------------------------------------------
    # 6. PCA scatter plot
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler_viz = StandardScaler()
    pca_viz    = PCA(n_components=2, random_state=RANDOM_SEED)
    X_scaled   = scaler_viz.fit_transform(X)
    X_pca      = pca_viz.fit_transform(X_scaled)

    colours = {0: "#2196F3", 1: "#FF9800", 2: "#F44336"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for cls in sorted(np.unique(y)):
        mask = y == cls
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colours[cls], label=CLASS_LABELS[cls],
            alpha=0.65, s=40, edgecolors="w", linewidths=0.4,
        )
    ax.set_xlabel(f"PC1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA Scatter — 18-Channel ({args.dataset})\n{SYNTHETIC_LABEL}")
    ax.legend()
    pca_path = os.path.join(FIGURES_DIR, f"pca_scatter_3class_{ts}.png")
    fig.tight_layout()
    fig.savefig(pca_path, dpi=150)
    plt.close(fig)
    print(f"PCA scatter saved: {pca_path}")

    # ------------------------------------------------------------------
    # 7. Fold metrics table
    # ------------------------------------------------------------------
    table_md = f"""# 3-Class Treatment-Condition Classifier Results ({args.dataset})

{status_label}

## PCA+LDA (Primary Baseline)

{format_fold_table(lda_result['fold_metrics'])}

## PCA+SVM RBF (Comparison)

{format_fold_table(svm_result['fold_metrics'])}

## Summary

| Pipeline | Mean Acc | Std Acc | Bal Acc | Macro-F1 |
|---|---|---|---|---|
| PCA+LDA  | {lda_result['mean_acc']:.4f} | {lda_result['std_acc']:.4f} | {lda_result['mean_bal_acc']:.4f} | {lda_result['mean_f1']:.4f} |
| PCA+SVM  | {svm_result['mean_acc']:.4f} | {svm_result['std_acc']:.4f} | {svm_result['mean_bal_acc']:.4f} | {svm_result['mean_f1']:.4f} |
"""
    table_path = os.path.join(TABLES_DIR, f"results_3class_{args.dataset}_{ts}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)
    print(f"Table saved: {table_path}")

    # ------------------------------------------------------------------
    # 8. Save model (refit on full dataset) + OOD scorer & Register Manifest
    # ------------------------------------------------------------------
    final_pipeline = build_lda_pipeline(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)
    final_pipeline.fit(X, y)

    ood_scorer = MahalanobisOOD(n_pca_components=PCA_N_COMPONENTS).fit(X)

    import joblib
    from datetime import datetime as _dt
    ts2 = _dt.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(MODELS_DIR_TC, exist_ok=True)
    payload_path = os.path.join(MODELS_DIR_TC, f"treatment_3class_pca_lda_{args.dataset}_{ts2}.joblib")
    joblib.dump({
        "pipeline":    final_pipeline,
        "ood_scorer":  ood_scorer,
        "cv_result":   lda_result,
        "saved_at":    ts2,
        "channel_cols": CHANNEL_COLS,
        "label":        status_label,
        "dataset_path": csv_path,
        "dataset_name": args.dataset,
    }, payload_path)
    print(f"Model + OOD scorer saved: {payload_path}")

    metrics = {
        "mean_accuracy": float(lda_result["mean_acc"]),
        "std_accuracy": float(lda_result["std_acc"]),
        "balanced_accuracy": float(lda_result["mean_bal_acc"]),
        "macro_f1": float(lda_result["mean_f1"]),
        "svm_accuracy": float(svm_result["mean_acc"]),
    }
    register_model(
        model_key=model_key,
        filepath=payload_path,
        dataset_path=csv_path,
        metrics=metrics,
        model_type="StandardScaler -> PCA(5) -> LDA",
        validation_method=f"5-fold GroupKFold by fruit_id ({df['fruit_id'].nunique()} fruits / {len(df)} scans)",
        scientific_status=status_label,
        notes=f"Trained via 04_train_baseline_3class.py on {args.dataset}"
    )
    print(f"Registered model in manifest: {model_key}")

    print(f"\n{'='*60}")
    print(f"DONE  |  {status_label}")
    print(f"PCA+LDA Mean Accuracy: {lda_result['mean_acc']*100:.2f}% ± {lda_result['std_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
