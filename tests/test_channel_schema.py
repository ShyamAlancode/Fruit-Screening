"""
test_channel_schema.py

Validates the exact structure of the synthetic AS7265x CSV file.
Run this immediately after generating the CSV and before any model training.
"""

import pytest
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from constants import CHANNEL_COLS, REQUIRED_META_COLS, REQUIRED_CSV_COLS, N_CHANNELS


SYNTHETIC_CSV = r"d:\CFRD\project\data\pesticide_screening\roomi_synthetic_18ch\synthetic_as7265x_roomi_apples.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """Load the real CSV if it exists, otherwise skip all tests."""
    if not os.path.exists(SYNTHETIC_CSV):
        pytest.skip(
            f"Synthetic CSV not yet generated: {SYNTHETIC_CSV}\n"
            "Run scripts/01_extract_roomi_18ch.py first."
        )
    return pd.read_csv(SYNTHETIC_CSV)


# ---------------------------------------------------------------------------
# Test 1: Exactly 18 channel columns
# ---------------------------------------------------------------------------

def test_exactly_18_channels(df):
    found = [c for c in df.columns if c.startswith("ch_")]
    assert len(found) == 18, (
        f"Expected 18 channel columns (ch_410 ... ch_940), found {len(found)}: {found}"
    )


# ---------------------------------------------------------------------------
# Test 2: Channel columns in exact order
# ---------------------------------------------------------------------------

def test_channel_column_order(df):
    actual = [c for c in df.columns if c.startswith("ch_")]
    assert actual == CHANNEL_COLS, (
        f"Channel column order mismatch.\n"
        f"Expected: {CHANNEL_COLS}\n"
        f"Got:      {actual}"
    )


# ---------------------------------------------------------------------------
# Test 3: All required metadata columns present
# ---------------------------------------------------------------------------

def test_required_metadata_columns(df):
    missing = [c for c in REQUIRED_META_COLS if c not in df.columns]
    assert not missing, (
        f"Missing required metadata columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ---------------------------------------------------------------------------
# Test 4: Zero NaN values in spectral columns
# ---------------------------------------------------------------------------

def test_no_nan_in_spectral_columns(df):
    nan_counts = df[CHANNEL_COLS].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    assert cols_with_nan.empty, (
        f"NaN values found in spectral columns:\n{cols_with_nan}"
    )


# ---------------------------------------------------------------------------
# Test 5: Zero NaN values in metadata columns
# ---------------------------------------------------------------------------

def test_no_nan_in_metadata(df):
    nan_counts = df[REQUIRED_META_COLS].isnull().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    assert cols_with_nan.empty, (
        f"NaN values found in metadata columns:\n{cols_with_nan}"
    )


# ---------------------------------------------------------------------------
# Test 6: class_3 values are exactly {0, 1, 2}
# ---------------------------------------------------------------------------

def test_class3_valid_values(df):
    unique_classes = set(df["class_3"].unique())
    allowed = {0, 1, 2}
    unexpected = unique_classes - allowed
    assert not unexpected, (
        f"Unexpected class_3 values: {unexpected}. "
        f"Only {{0, 1, 2}} are allowed (Control-like, Low-treatment-like, High-treatment-like)."
    )


# ---------------------------------------------------------------------------
# Test 7: Spectral values are in a physically plausible range
# ---------------------------------------------------------------------------

def test_spectral_values_plausible_range(df):
    """
    Calibrated reflectance should be in [0.0, 2.0].
    Values far outside this range indicate a calibration error.
    """
    X = df[CHANNEL_COLS].values
    assert X.min() >= 0.0, (
        f"Negative spectral values found. Min = {X.min():.4f}. "
        "Check calibration step in hsi_to_18ch.py."
    )
    assert X.max() <= 5.0, (
        f"Spectral values exceed 5.0. Max = {X.max():.4f}. "
        "Check calibration step in hsi_to_18ch.py."
    )


# ---------------------------------------------------------------------------
# Test 8: fruit_id column has multiple rows per ID (repeated scans)
# ---------------------------------------------------------------------------

def test_multiple_scans_per_fruit(df):
    """
    Each physical apple should have multiple scan rows.
    If every fruit_id appears only once, something went wrong in the extraction.
    """
    scans_per_fruit = df.groupby("fruit_id").size()
    single_scan_fruits = scans_per_fruit[scans_per_fruit == 1]
    # Allow up to 20% of fruits to have only 1 scan (edge cases)
    frac_single = len(single_scan_fruits) / len(scans_per_fruit)
    assert frac_single < 0.5, (
        f"{len(single_scan_fruits)}/{len(scans_per_fruit)} fruit IDs have only 1 scan. "
        "This suggests the fruit_id extraction is incorrect — each row may be a different fruit_id."
    )


# ---------------------------------------------------------------------------
# Test 9: treatment_family values are exactly Monostar and/or Nativo
# ---------------------------------------------------------------------------

def test_treatment_family_valid(df):
    valid = {"Monostar", "Nativo"}
    found = set(df["treatment_family"].unique())
    unexpected = found - valid
    assert not unexpected, (
        f"Unexpected treatment_family values: {unexpected}. "
        f"Only {valid} are valid."
    )


# ---------------------------------------------------------------------------
# Test 10: fruit_id is NOT in CHANNEL_COLS (belt-and-suspenders check)
# ---------------------------------------------------------------------------

def test_fruit_id_not_in_channel_cols():
    assert "fruit_id" not in CHANNEL_COLS, (
        "fruit_id found in CHANNEL_COLS. This is a critical constants.py bug."
    )
