"""
data_validation.py

Validates the synthetic 18-channel AS7265x CSV before any model training.
Prints a mandatory warning about scan rows vs independent fruit count.
Generates a validation report.

Usage:
    python scripts/02_validate_csv.py
    or import and call validate_synthetic_csv() directly.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from constants import (
    CHANNEL_COLS, REQUIRED_META_COLS, REQUIRED_CSV_COLS,
    CLASS_LABELS, N_CHANNELS, SYNTHETIC_CSV_PATH
)


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

def validate_synthetic_csv(
    csv_path: str = None,
    report_dir: str = r"d:\CFRD\project\reports",
    verbose: bool = True,
) -> dict:
    """
    Validate the synthetic AS7265x CSV file.

    Parameters
    ----------
    csv_path : str, optional
        Path to the synthetic CSV. Defaults to SYNTHETIC_CSV_PATH from constants.
    report_dir : str
        Directory to write the validation report markdown.
    verbose : bool
        Print results to stdout.

    Returns
    -------
    dict with keys: passed (bool), errors (list), warnings (list), stats (dict)
    """
    if csv_path is None:
        csv_path = SYNTHETIC_CSV_PATH

    result = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    report_lines = [
        "# Synthetic AS7265x CSV Validation Report",
        f"File: `{csv_path}`",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    def _log(msg, level="INFO"):
        if verbose:
            prefix = {"INFO": "  ", "WARN": "  [WARN]", "FAIL": "  [FAIL]"}[level]
            print(f"{prefix} {msg}")

    # -----------------------------------------------------------------------
    # 1. File existence
    # -----------------------------------------------------------------------
    _log(f"Loading: {csv_path}")
    if not os.path.exists(csv_path):
        result["errors"].append(f"File not found: {csv_path}")
        result["passed"] = False
        report_lines.append(f"**FAIL** — File not found: `{csv_path}`")
        _write_report(report_lines, report_dir)
        return result

    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    _log(f"Loaded: {n_rows} rows × {n_cols} columns")
    result["stats"]["n_rows"] = n_rows
    result["stats"]["n_cols"] = n_cols

    # -----------------------------------------------------------------------
    # 2. Required columns
    # -----------------------------------------------------------------------
    _log("\n[CHECK] Required columns...")
    missing_cols = [c for c in REQUIRED_CSV_COLS if c not in df.columns]
    extra_cols   = [c for c in df.columns if c not in REQUIRED_CSV_COLS]

    if missing_cols:
        msg = f"Missing required columns: {missing_cols}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log(f"All {len(REQUIRED_CSV_COLS)} required columns present", "INFO")

    if extra_cols:
        msg = f"Unexpected extra columns (will be ignored in training): {extra_cols}"
        result["warnings"].append(msg)
        _log(msg, "WARN")

    report_lines += [
        "## Column Check",
        f"- Missing: {missing_cols if missing_cols else 'None'}",
        f"- Extra: {extra_cols if extra_cols else 'None'}",
        "",
    ]

    if missing_cols:
        _write_report(report_lines, report_dir)
        return result

    # -----------------------------------------------------------------------
    # 3. Channel column order
    # -----------------------------------------------------------------------
    _log("[CHECK] Channel column order...")
    actual_channels = [c for c in df.columns if c.startswith("ch_")]
    if actual_channels != CHANNEL_COLS:
        msg = (
            f"Channel column ORDER mismatch.\n"
            f"    Expected: {CHANNEL_COLS}\n"
            f"    Got:      {actual_channels}"
        )
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log(f"Channel order correct: ch_410 → ch_940 ({N_CHANNELS} channels)")

    # -----------------------------------------------------------------------
    # 4. NaN check
    # -----------------------------------------------------------------------
    _log("[CHECK] NaN values...")
    nan_counts = df[CHANNEL_COLS].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        msg = f"NaN found in spectral columns:\n{cols_with_nan.to_string()}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log("No NaN in spectral columns")

    nan_meta = df[REQUIRED_META_COLS].isnull().sum()
    meta_with_nan = nan_meta[nan_meta > 0]
    if not meta_with_nan.empty:
        msg = f"NaN found in metadata columns: {meta_with_nan.to_dict()}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log("No NaN in metadata columns")

    # -----------------------------------------------------------------------
    # 5. class_3 values
    # -----------------------------------------------------------------------
    _log("[CHECK] class_3 values...")
    unique_classes = sorted(df["class_3"].unique())
    valid_classes  = [0, 1, 2]
    unexpected_classes = [c for c in unique_classes if c not in valid_classes]
    if unexpected_classes:
        msg = f"Unexpected class_3 values: {unexpected_classes}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log(f"class_3 values valid: {unique_classes}")

    # -----------------------------------------------------------------------
    # 6. treatment_family values
    # -----------------------------------------------------------------------
    _log("[CHECK] treatment_family values...")
    valid_families = {"Monostar", "Nativo"}
    found_families = set(df["treatment_family"].unique())
    unexpected_families = found_families - valid_families
    if unexpected_families:
        msg = f"Unexpected treatment_family values: {unexpected_families}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    else:
        _log(f"treatment_family values valid: {sorted(found_families)}")

    # -----------------------------------------------------------------------
    # 7. Spectral value range
    # -----------------------------------------------------------------------
    _log("[CHECK] Spectral value range...")
    X = df[CHANNEL_COLS].values
    vmin, vmax = float(X.min()), float(X.max())
    if vmin < 0.0:
        msg = f"Negative spectral values found. Min = {vmin:.4f}"
        result["errors"].append(msg)
        result["passed"] = False
        _log(msg, "FAIL")
    elif vmax > 5.0:
        msg = f"Spectral values > 5.0 detected. Max = {vmax:.4f}. Check calibration."
        result["warnings"].append(msg)
        _log(msg, "WARN")
    else:
        _log(f"Spectral range plausible: [{vmin:.3f}, {vmax:.3f}]")

    result["stats"]["spectral_min"] = vmin
    result["stats"]["spectral_max"] = vmax
    result["stats"]["spectral_mean"] = float(X.mean())

    # -----------------------------------------------------------------------
    # 8. Fruit / scan statistics — MANDATORY WARNING
    # -----------------------------------------------------------------------
    _log("\n[CHECK] Fruit/scan statistics...")
    n_unique_fruits = df["fruit_id"].nunique()
    scans_per_fruit = df.groupby("fruit_id").size()
    min_scans = int(scans_per_fruit.min())
    max_scans = int(scans_per_fruit.max())
    mean_scans = float(scans_per_fruit.mean())

    result["stats"]["n_rows"] = n_rows
    result["stats"]["n_unique_fruits"] = n_unique_fruits
    result["stats"]["min_scans_per_fruit"] = min_scans
    result["stats"]["max_scans_per_fruit"] = max_scans
    result["stats"]["mean_scans_per_fruit"] = round(mean_scans, 1)

    # *** MANDATORY WARNING — must always be printed ***
    print()
    print("  " + "!" * 66)
    print(f"  ! MANDATORY WARNING                                              !")
    print(f"  ! {n_rows} scan rows in this CSV do NOT equal {n_rows} independent apples.  !")
    print(f"  ! Only {n_unique_fruits} independent fruit groups exist.                    !")
    print(f"  ! GroupKFold MUST use fruit_id as groups, not row indices.       !")
    print("  " + "!" * 66)
    print()

    _log(f"Total rows:           {n_rows}")
    _log(f"Unique fruit IDs:     {n_unique_fruits}")
    _log(f"Scans per fruit:      min={min_scans}, max={max_scans}, mean={mean_scans:.1f}")

    # -----------------------------------------------------------------------
    # 9. Class distribution
    # -----------------------------------------------------------------------
    _log("[CHECK] Class distribution...")
    class_counts = df.groupby(["class_3", "treatment_family"]).size().reset_index(name="count")
    _log("Class × Family distribution:")
    for _, row in class_counts.iterrows():
        label = CLASS_LABELS.get(row["class_3"], f"class_{row['class_3']}")
        _log(f"    {label:25s}  {row['treatment_family']:10s}  {row['count']} rows")

    # -----------------------------------------------------------------------
    # 10. Per-fruit scan count table
    # -----------------------------------------------------------------------
    per_fruit_table = df.groupby("fruit_id").agg(
        n_scans=("class_3", "count"),
        class_3=("class_3", "first"),
        treatment_family=("treatment_family", "first"),
        level=("level", "first"),
    ).reset_index()

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    report_lines += [
        "## Mandatory Warning",
        f"> **{n_rows} scan rows ≠ {n_unique_fruits} independent physical apples.**",
        f"> GroupKFold must use `fruit_id` as groups, not row indices.",
        "",
        "## Statistics",
        f"| Metric | Value |",
        "|---|---|",
        f"| Total rows | {n_rows} |",
        f"| Independent fruits | {n_unique_fruits} |",
        f"| Min scans per fruit | {min_scans} |",
        f"| Max scans per fruit | {max_scans} |",
        f"| Mean scans per fruit | {mean_scans:.1f} |",
        f"| Spectral min | {vmin:.4f} |",
        f"| Spectral max | {vmax:.4f} |",
        "",
        "## Class × Treatment Family Distribution",
    ]
    report_lines.append(class_counts.to_markdown(index=False))
    report_lines += [
        "",
        "## Per-Fruit Summary",
    ]
    report_lines.append(per_fruit_table.to_markdown(index=False))
    report_lines += [""]

    if result["passed"]:
        report_lines.append("## Result\n**ALL CHECKS PASSED** — Ready for model training.")
        _log("ALL CHECKS PASSED", "INFO")
    else:
        report_lines.append(f"## Result\n**FAILED** — {len(result['errors'])} error(s) found:")
        for e in result["errors"]:
            report_lines.append(f"- {e}")
        _log(f"FAILED — {len(result['errors'])} error(s)", "FAIL")

    _write_report(report_lines, report_dir)
    return result


def _write_report(lines: list, report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(report_dir, f"data_validation_{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {path}")


if __name__ == "__main__":
    result = validate_synthetic_csv()
    sys.exit(0 if result["passed"] else 1)
