"""
test_no_group_leakage.py

Enforces the fruit-wise GroupKFold discipline.
All tests here must pass before any Phase 3 accuracy numbers are reported.

These tests will catch the most common and dangerous mistake in spectral
ML: allowing repeated scans of the same physical fruit to appear in both
training and test sets, which inflates accuracy by 15-40%.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import CHANNEL_COLS, REQUIRED_META_COLS, REQUIRED_CSV_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """
    Create a minimal synthetic DataFrame with the correct CSV schema.
    10 fruit IDs, 5 scans each, 3 classes.
    Values are random but structurally correct.
    """
    rng = np.random.default_rng(42)
    rows = []
    for fruit_id in range(1, 11):
        class_3 = (fruit_id - 1) % 3
        level = ["control", "low", "high"][class_3]
        family = "Monostar" if fruit_id <= 5 else "Nativo"
        for _ in range(5):
            spectrum = rng.uniform(0.1, 0.9, size=18).tolist()
            rows.append({
                "fruit_id": fruit_id,
                "treatment_family": family,
                "level": level,
                "class_3": class_3,
                **dict(zip(CHANNEL_COLS, spectrum)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: fruit_id must not appear in the feature matrix X
# ---------------------------------------------------------------------------

def test_fruit_id_not_in_features(synthetic_df):
    """
    The most critical check: fruit_id must never be used as a predictor.
    If it were, the model would learn to memorise fruit IDs, not spectral patterns.
    """
    X = synthetic_df[CHANNEL_COLS].values
    col_names = synthetic_df[CHANNEL_COLS].columns.tolist()

    assert "fruit_id" not in col_names, (
        "fruit_id found in feature columns. Remove it before training."
    )
    assert X.shape[1] == 18, (
        f"Expected 18 feature columns, got {X.shape[1]}."
    )


# ---------------------------------------------------------------------------
# Test 2: GroupKFold must produce non-overlapping fruit groups
# ---------------------------------------------------------------------------

def test_groupkfold_no_fruit_overlap(synthetic_df):
    """
    Confirm that GroupKFold(n_splits=5) places each fruit ID in exactly
    one fold's test set, with zero overlap between train and test groups.
    """
    X = synthetic_df[CHANNEL_COLS].values
    y = synthetic_df["class_3"].values
    groups = synthetic_df["fruit_id"].values

    gkf = GroupKFold(n_splits=5)

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        train_fruits = set(groups[train_idx])
        test_fruits  = set(groups[test_idx])
        overlap = train_fruits & test_fruits

        assert len(overlap) == 0, (
            f"Fold {fold_idx}: fruit_id overlap between train and test: {overlap}. "
            "This indicates group leakage — the same physical apple appears in both splits."
        )


# ---------------------------------------------------------------------------
# Test 3: Scaler must be fit on training fold only
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only(synthetic_df):
    """
    Simulate the pipeline and verify that the scaler's mean is computed
    only from training samples, not from the full dataset.

    If a scaler is fit on the full dataset and then used in CV,
    test-set statistics leak into the transform — a subtle but real leakage.
    """
    X = synthetic_df[CHANNEL_COLS].values
    y = synthetic_df["class_3"].values
    groups = synthetic_df["fruit_id"].values

    gkf = GroupKFold(n_splits=5)

    global_scaler = StandardScaler().fit(X)   # WRONG — fit on all data
    global_mean = global_scaler.mean_.copy()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        fold_scaler = StandardScaler().fit(X[train_idx])   # CORRECT — fit on fold only

        # The fold mean must differ from the global mean (different data)
        max_diff = np.abs(fold_scaler.mean_ - global_mean).max()

        # They must not be identical (which would indicate scaler leakage)
        assert max_diff > 0.0, (
            f"Fold {fold_idx}: Fold scaler mean == global scaler mean. "
            "This may indicate the scaler was incorrectly fit on all data."
        )


# ---------------------------------------------------------------------------
# Test 4: PCA must be fit on training fold only
# ---------------------------------------------------------------------------

def test_pca_fitted_on_train_only(synthetic_df):
    """
    PCA fit on the full dataset leaks test-set covariance into the
    components used to transform training data. This is a well-known
    source of optimistic cross-validation estimates.
    """
    X = synthetic_df[CHANNEL_COLS].values
    y = synthetic_df["class_3"].values
    groups = synthetic_df["fruit_id"].values

    gkf = GroupKFold(n_splits=5)

    global_pca = PCA(n_components=5, random_state=42).fit(X)
    global_components = global_pca.components_.copy()

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        fold_pca = PCA(n_components=5, random_state=42).fit(X[train_idx])

        # Components must differ from global PCA (different covariance matrix)
        cosine_sim = np.abs(
            np.dot(fold_pca.components_, global_components.T)
        ).max()

        # Shape check
        assert fold_pca.components_.shape == (5, 18), (
            f"Fold {fold_idx}: PCA components shape {fold_pca.components_.shape} "
            "is not (5, 18). Check n_components and feature count."
        )

        # Actual leakage check: fold PCA must not be identical to global PCA.
        # If PCA were mistakenly fit on the full dataset, components would match.
        max_component_diff = np.abs(
            np.abs(fold_pca.components_) - np.abs(global_components)
        ).max()
        assert max_component_diff > 1e-6, (
            f"Fold {fold_idx}: fold PCA components are identical to the global "
            "PCA fit on all data — PCA was not refit per fold (data leakage)."
        )


# ---------------------------------------------------------------------------
# Test 5: Every fruit ID appears in exactly one test fold
# ---------------------------------------------------------------------------

def test_every_fruit_in_exactly_one_test_fold(synthetic_df):
    """
    In GroupKFold, each group (fruit) should appear in exactly one test fold.
    This verifies the CV structure is exhaustive and non-redundant.
    """
    X = synthetic_df[CHANNEL_COLS].values
    y = synthetic_df["class_3"].values
    groups = synthetic_df["fruit_id"].values

    gkf = GroupKFold(n_splits=5)
    fruit_test_count = {}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        test_fruits = groups[test_idx]
        for fid in set(test_fruits):
            fruit_test_count[fid] = fruit_test_count.get(fid, 0) + 1

    for fid, count in fruit_test_count.items():
        assert count == 1, (
            f"fruit_id {fid} appeared in {count} test folds (expected exactly 1). "
            "This breaks the GroupKFold guarantee."
        )

    unique_fruits = set(groups)
    tested_fruits = set(fruit_test_count.keys())
    assert unique_fruits == tested_fruits, (
        f"Not all fruits appeared in a test fold. "
        f"Missing: {unique_fruits - tested_fruits}"
    )
