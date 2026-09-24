"""
predict.py
==========
Headless Command-Line Interface for CFRD Apple Screening & Quality Analysis.
Predicts everything in the repository (pesticide level, ripeness, cultivar variety,
Brix sweetness, and visual defects) with ZERO UI / Streamlit dependencies.

Supports TWO operating modes:
  Mode 1 (Unified Apple Inspection):
      Score an apple across ALL targets at once (from photo, spectrum, or mock scan).
      Example: python predict.py --image data/test_user_apple.jpg
               python predict.py --mock
               python predict.py --spectrum "0.09,0.10,..."
               python predict.py --image data/test_user_apple.jpg --json

  Mode 2 (Task-Specific Batch CSV Scoring):
      Score a CSV of scans against a specific trained task model bundle.
      Example: python predict.py --task ripeness --input data/quality/ripeness_variety_features.csv
               python predict.py --task pesticide3 --input data/pesticide_screening/roomi_real_hsi_18ch/roomi_real_hsi_18ch.csv --output results.csv
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Fix Windows console encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure safe Windows PyTorch environment before importing pandas/numpy
os.environ["PATH"] = ";".join([
    p for p in os.environ.get("PATH", "").split(";")
    if "mingw" not in p.lower() and "intel" not in p.lower()
])
try:
    import torch
except Exception:
    pass

# Ensure root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd
from predictor import ApplePredictor
from common import MODELS_DIR, TASKS


def parse_spectrum_arg(spec_str: str) -> list[float]:
    """Parse comma-separated or space-separated floats into 18-channel list."""
    cleaned = spec_str.replace("[", "").replace("]", "").replace(";", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) != 18:
        parts = [p.strip() for p in cleaned.split() if p.strip()]
    if len(parts) != 18:
        raise ValueError(f"Expected exactly 18 channel values, got {len(parts)} from input: {spec_str!r}")
    return [float(x) for x in parts]


def run_task_prediction(task_name: str, input_csv: str, output_csv: str | None = None):
    """Task-specific batch scoring mode with mandatory reliability warning."""
    bundle_path = MODELS_DIR / f"{task_name}.joblib"
    if not bundle_path.exists():
        sys.exit(f"No trained model at {bundle_path}. Run: python train.py --task {task_name}")

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    meta = bundle["meta"]

    print(f"\n{'!'*70}")
    print(f"RELIABILITY WARNING for task '{task_name}':")
    print(meta["reliability"])
    print(f"{'!'*70}\n")

    df = pd.read_csv(input_csv)
    missing = [c for c in meta["feature_cols"] if c not in df.columns]
    if missing:
        sys.exit(f"Input CSV is missing required feature columns: {missing}")

    X = df[meta["feature_cols"]].to_numpy(dtype=float)
    preds = model.predict(X)

    out = df.copy()
    out[f"predicted_{task_name}"] = preds

    if meta["kind"] == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        out[f"predicted_{task_name}_confidence"] = proba.max(axis=1)

    display_cols = [c for c in out.columns if c not in meta["feature_cols"]]
    print(out[display_cols].to_string(index=False))

    if output_csv:
        out.to_csv(output_csv, index=False)
        print(f"\nWrote: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Headless ML Apple Screener: Pesticide Level, Ripeness, Variety, Brix & Defect Risk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Unified All-in-One Prediction
  python predict.py --image data/test_user_apple.jpg
  python predict.py --mock
  python predict.py --spectrum "0.09,0.10,0.11,0.14,0.22,0.38,0.44,0.42,0.40,0.35,0.22,0.48,0.72,0.76,0.79,0.81,0.82,0.83"
  python predict.py --image data/test_user_apple.jpg --json

  # Mode 2: Task-Specific Batch CSV Scoring
  python predict.py --task ripeness --input data/quality/ripeness_variety_features.csv
  python predict.py --task pesticide3 --input data/pesticide_screening/roomi_real_hsi_18ch/roomi_real_hsi_18ch.csv
        """,
    )

    # Mode 2 Task-specific arguments
    parser.add_argument("--task", choices=list(TASKS.keys()), default=None, help="Target specific task bundle (e.g. ripeness, pesticide3, brix)")
    parser.add_argument("--input", type=str, default="", help="Input CSV path for task-specific scoring")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for task-specific scoring")

    # Mode 1 Unified prediction arguments
    parser.add_argument("--image", type=str, default="", help="Path to apple photo (JPG/PNG)")
    parser.add_argument("--spectrum", type=str, default="", help="18 channel reflectance values (comma-separated)")
    parser.add_argument("--csv", type=str, default="", help="Path to 18-channel CSV dataset file")
    parser.add_argument("--row", type=int, default=0, help="Row index in CSV file (default: 0)")
    parser.add_argument("--cultivar", type=str, default="", help="Optional known cultivar (e.g. Gala, Braeburn, Cox)")
    parser.add_argument(
        "--pesticide-model",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Pesticide screening model to use for unified mode (default: synthetic)",
    )
    parser.add_argument("--mock", action="store_true", help="Use realistic AS7265x apple mock reflectance spectrum")
    parser.add_argument("--json", action="store_true", help="Output raw structured JSON instead of human-readable text")

    args = parser.parse_args()

    # --- Mode 2: Task-Specific Batch CSV Scoring ---
    if args.task:
        input_csv = args.input or args.csv
        if not input_csv:
            parser.error("When using --task, you must provide --input <csv_file>")
        run_task_prediction(args.task, input_csv, args.output)
        return

    # --- Mode 1: Unified All-in-One Prediction ---
    predictor = ApplePredictor(default_pesticide_model=args.pesticide_model)

    prediction = None

    if args.csv:
        csv_p = Path(args.csv)
        if not csv_p.exists():
            print(f"Error: CSV file not found: {csv_p}", file=sys.stderr)
            sys.exit(1)
        prediction = predictor.predict_csv(csv_p, row_index=args.row)

    elif args.spectrum:
        try:
            spec = parse_spectrum_arg(args.spectrum)
        except Exception as e:
            print(f"Error parsing spectrum: {e}", file=sys.stderr)
            sys.exit(1)
        prediction = predictor.predict(
            spectrum=spec,
            image=args.image if args.image else None,
            cultivar=args.cultivar if args.cultivar else None,
            pesticide_model_flavor=args.pesticide_model,
        )

    elif args.image:
        img_p = Path(args.image)
        if not img_p.exists():
            print(f"Error: Image file not found: {img_p}", file=sys.stderr)
            sys.exit(1)
        prediction = predictor.predict(
            image=img_p,
            cultivar=args.cultivar if args.cultivar else None,
            pesticide_model_flavor=args.pesticide_model,
        )

    elif args.mock:
        from as7265x_driver import MOCK_APPLE_REFLECTANCE
        prediction = predictor.predict(
            spectrum=MOCK_APPLE_REFLECTANCE,
            cultivar=args.cultivar if args.cultivar else None,
            pesticide_model_flavor=args.pesticide_model,
        )

    else:
        # Default behavior: run image test if present, else mock
        default_img = PROJECT_ROOT / "data" / "test_user_apple.jpg"
        if default_img.exists():
            print(f"No arguments provided. Running default prediction on: {default_img.name}\n")
            prediction = predictor.predict(image=default_img, cultivar=args.cultivar if args.cultivar else None)
        else:
            print("No arguments provided. Running mock sensor scan...\n")
            prediction = predictor.predict(cultivar=args.cultivar if args.cultivar else None)

    if args.json:
        print(prediction.to_json(indent=2))
    else:
        print(prediction.summary())


if __name__ == "__main__":
    main()
