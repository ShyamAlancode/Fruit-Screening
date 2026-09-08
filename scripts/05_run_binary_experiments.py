"""
05_run_binary_experiments.py

Run 6 binary treatment-condition classification experiments.
All experiments use GroupKFold by fruit_id.
All outputs carry the mandatory synthetic label.

Binary tasks:
  1. Control vs Treated (all)
  2. Control vs Nativo fungicide formulation
  3. Control vs Monostar insecticide formulation
  4. Control vs Nativo-high formulation
  5. Control vs Monostar-high formulation
  6. Low vs High

Usage:
    cd d:\\CFRD\\project
    python scripts\\05_run_binary_experiments.py
"""

import sys, os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import (
    CHANNEL_COLS, SYNTHETIC_CSV_PATH, RANDOM_SEED,
    PCA_N_COMPONENTS, GROUPKFOLD_N_SPLITS,
    SYNTHETIC_LABEL, DISCLAIMER_BANNER,
)
from training import (
    build_lda_pipeline, build_svm_pipeline,
    run_group_kfold, format_fold_table,
)

TABLES_DIR = r"d:\CFRD\project\reports\tables"

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

def define_tasks(df: pd.DataFrame) -> list[dict]:
    """Return list of binary task dicts. Each task filters df to 2 classes."""
    tasks = []

    # Task 1: Control vs ALL treated
    t = df.copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 0 else 1)
    tasks.append({
        "name": "Control vs Treated (all)",
        "df": t,
        "class_names": ["Control-like", "Treated-like"],
    })

    # Task 2: Control vs Nativo-treated
    t = df[df["treatment_family"] == "Nativo"].copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 0 else 1)
    tasks.append({
        "name": "Control vs Nativo fungicide formulation",
        "df": t,
        "class_names": ["Control-like", "Nativo-treated-like"],
    })

    # Task 3: Control vs Monostar-treated
    t = df[df["treatment_family"] == "Monostar"].copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 0 else 1)
    tasks.append({
        "name": "Control vs Monostar insecticide formulation",
        "df": t,
        "class_names": ["Control-like", "Monostar-treated-like"],
    })

    # Task 4: Control vs Nativo-high
    t = df[(df["treatment_family"] == "Nativo") & (df["class_3"].isin([0, 2]))].copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 0 else 1)
    tasks.append({
        "name": "Control vs Nativo-high formulation",
        "df": t,
        "class_names": ["Control-like", "Nativo-high-like"],
    })

    # Task 5: Control vs Monostar-high
    t = df[(df["treatment_family"] == "Monostar") & (df["class_3"].isin([0, 2]))].copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 0 else 1)
    tasks.append({
        "name": "Control vs Monostar-high formulation",
        "df": t,
        "class_names": ["Control-like", "Monostar-high-like"],
    })

    # Task 6: Low vs High (exclude control)
    t = df[df["class_3"].isin([1, 2])].copy()
    t["binary"] = t["class_3"].apply(lambda x: 0 if x == 1 else 1)
    tasks.append({
        "name": "Low vs High treatment-formulation level",
        "df": t,
        "class_names": ["Low-treatment-like", "High-treatment-like"],
    })

    return tasks


def main():
    print(f"\n{DISCLAIMER_BANNER}\n")
    os.makedirs(TABLES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(SYNTHETIC_CSV_PATH)
    tasks = define_tasks(df)

    summary_rows = []

    report_lines = [
        "# Binary Treatment-Condition Classification Results",
        "",
        f"*{SYNTHETIC_LABEL}*",
        "",
    ]

    for task_idx, task in enumerate(tasks):
        t  = task["df"]
        X  = t[CHANNEL_COLS].values
        y  = t["binary"].values
        groups = t["fruit_id"].values

        print(f"\n{'='*65}")
        print(f"TASK {task_idx+1}: {task['name']}")
        print(f"  Classes: {task['class_names']}")
        print(f"  Rows: {len(t)} | Unique fruits: {t['fruit_id'].nunique()}")
        print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        lda = build_lda_pipeline(n_components=min(PCA_N_COMPONENTS, X.shape[1]-1),
                                  random_state=RANDOM_SEED)
        svm = build_svm_pipeline(n_components=min(PCA_N_COMPONENTS, X.shape[1]-1),
                                  random_state=RANDOM_SEED)

        r_lda = run_group_kfold(X, y, groups, lda, n_splits=min(GROUPKFOLD_N_SPLITS, t["fruit_id"].nunique()),
                                 pipeline_name="PCA+LDA", verbose=False)
        r_svm = run_group_kfold(X, y, groups, svm, n_splits=min(GROUPKFOLD_N_SPLITS, t["fruit_id"].nunique()),
                                 pipeline_name="PCA+SVM", verbose=False)

        print(f"  PCA+LDA: {r_lda['mean_acc']*100:.2f}% ± {r_lda['std_acc']*100:.2f}%")
        print(f"  PCA+SVM: {r_svm['mean_acc']*100:.2f}% ± {r_svm['std_acc']*100:.2f}%")

        summary_rows.append({
            "Task": task["name"],
            "n_rows": len(t),
            "n_fruits": t["fruit_id"].nunique(),
            "LDA_acc": f"{r_lda['mean_acc']*100:.1f}% ± {r_lda['std_acc']*100:.1f}%",
            "LDA_bal": f"{r_lda['mean_bal_acc']*100:.1f}%",
            "SVM_acc": f"{r_svm['mean_acc']*100:.1f}% ± {r_svm['std_acc']*100:.1f}%",
        })

        report_lines += [
            f"## Task {task_idx+1}: {task['name']}",
            f"Classes: {task['class_names']}  |  n={len(t)}  |  fruits={t['fruit_id'].nunique()}",
            "",
            "### PCA+LDA",
            format_fold_table(r_lda["fold_metrics"]),
            "",
            "### PCA+SVM (RBF) [comparison]",
            format_fold_table(r_svm["fold_metrics"]),
            "",
        ]

    # Summary table
    summary_df = pd.DataFrame(summary_rows)
    report_lines += [
        "## Summary Comparison",
        summary_df.to_markdown(index=False),
        "",
        f"*{SYNTHETIC_LABEL}*",
    ]

    table_path = os.path.join(TABLES_DIR, f"results_binary_{ts}.md")
    with open(table_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nBinary results saved: {table_path}")


if __name__ == "__main__":
    main()
