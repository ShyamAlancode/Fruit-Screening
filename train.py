#!/usr/bin/env python3
"""
train.py -- plain CLI, no UI.

Trains one of: pesticide3, pesticide_binary, ripeness, variety, brix.
Cross-validates with GroupKFold/LeaveOneGroupOut (grouped by physical
fruit, never by scan), prints an honest report, then fits on all data
and saves a model bundle to models/<task>.joblib.

Usage:
    python train.py --task ripeness
    python train.py --task pesticide3
    python train.py --all
"""

import sys
from pathlib import Path
import argparse

# Ensure root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from common import TASKS, load_task, cross_validate, save_bundle, print_report


def train_one(task_name: str):
    task = TASKS[task_name]
    df, X, y, groups = load_task(task)

    cv_result = cross_validate(task, X, y, groups)
    print_report(task, cv_result)

    # Fit final model on ALL available data for the saved bundle.
    # (The CV result above, not this fit, is what tells you if the
    # model is any good -- fitting on everything just gives you the
    # artifact to run predict.py against.)
    final_model = task.model_builder()
    final_model.fit(X, y)

    classes = sorted(set(y)) if task.kind == "classification" else None
    path = save_bundle(task, final_model, cv_result, classes=classes)
    print(f"Saved: {path}\n")


def main():
    parser = argparse.ArgumentParser(description="Train apple sensing models (no UI).")
    parser.add_argument("--task", choices=list(TASKS.keys()), help="Task to train.")
    parser.add_argument("--all", action="store_true", help="Train every task.")
    args = parser.parse_args()

    if args.all:
        for name in TASKS:
            train_one(name)
    elif args.task:
        train_one(args.task)
    else:
        parser.error("Pass --task <name> or --all. Choices: " + ", ".join(TASKS.keys()))


if __name__ == "__main__":
    main()
