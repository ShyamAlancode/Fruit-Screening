"""
06_run_coverage_experiment.py

Runs the confidence rejection threshold sweep on out-of-fold predictions.
At each threshold, evaluates coverage and accepted scan accuracy.
Saves the coverage vs. balanced accuracy curve to reports/figures/coverage_accuracy.png.

Usage:
    python scripts/06_run_coverage_experiment.py
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    DISCLAIMER_BANNER,
    SYNTHETIC_LABEL,
    CONFIDENCE_THRESHOLDS,
    COVERAGE_FORMAT,
)
from ood import run_rejection_sweep

FIGURES_DIR = r"d:\CFRD\project\reports\figures"
TABLES_DIR = r"d:\CFRD\project\reports\tables"
OOF_PATH = r"d:\CFRD\project\data\pesticide_screening\roomi_synthetic_18ch\oof_predictions_3class.csv"


def main():
    print(f"\n{DISCLAIMER_BANNER}\n")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists(OOF_PATH):
        print(f"ERROR: OOF predictions not found: {OOF_PATH}. Run scripts/04_train_baseline_3class.py first.")
        sys.exit(1)

    df_oof = pd.read_csv(OOF_PATH)
    y_true = df_oof["class_3"].values
    y_pred = df_oof["oof_pred_lda"].values

    proba_cols = [c for c in df_oof.columns if c.startswith("proba_")]
    y_proba = df_oof[proba_cols].values

    # Run sweep
    results = run_rejection_sweep(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        thresholds=CONFIDENCE_THRESHOLDS,
        verbose=True,
    )

    res_df = pd.DataFrame(results)

    # Save Markdown Table
    table_path = os.path.join(TABLES_DIR, f"coverage_rejection_sweep_{ts}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# Confidence Rejection Sweep Results\n\n")
        f.write(f"*{SYNTHETIC_LABEL}*\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n")
    print(f"\nTable saved: {table_path}")

    # Plot Coverage vs Accuracy
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Coverage (%)", color=color)
    line1 = ax1.plot(
        res_df["threshold"], res_df["coverage_pct"], marker="o", color=color, label="Coverage %"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0, 105)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Accepted Accuracy (%)", color=color)
    line2 = ax2.plot(
        res_df["threshold"],
        res_df["accepted_accuracy_pct"],
        marker="s",
        color=color,
        label="Accepted Accuracy %",
    )
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(40, 105)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left")

    plt.title(f"Confidence Rejection: Coverage vs Accepted Accuracy\n{SYNTHETIC_LABEL}")
    plt.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # Save to canonical figure path AND timestamped path
    fig_path = os.path.join(FIGURES_DIR, "coverage_accuracy.png")
    fig_path_ts = os.path.join(FIGURES_DIR, f"coverage_accuracy_{ts}.png")
    fig.savefig(fig_path, dpi=150)
    fig.savefig(fig_path_ts, dpi=150)
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
