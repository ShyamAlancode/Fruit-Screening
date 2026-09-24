"""
test_calibration.py

Tests for the sensor simulator's calibration pipeline.
Ensures dark/white/raw/calibrated chain is physically plausible.

FIX (reviewer H1): all tests now call the production simulate_scan()
directly instead of a local re-implementation that could silently drift.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from constants import N_CHANNELS, SIM_EPSILON
from sensor_simulator import simulate_scan


# ---------------------------------------------------------------------------
# Test 1: Dark reference must be lower than white reference (per channel)
# ---------------------------------------------------------------------------

def test_dark_less_than_white():
    """
    Physically: dark current (no light) must be lower than white reference
    (illuminated Teflon tile) for every channel.
    """
    X = np.full(N_CHANNELS, 0.5)
    result = simulate_scan(X, dark_offset=0.02, led_scale=1.0,
                           read_noise_std=0.0, rng=np.random.default_rng(0))
    D, W = result["dark"], result["white"]
    for i in range(N_CHANNELS):
        assert D[i] < W[i], (
            f"Channel {i}: dark ({D[i]:.4f}) >= white ({W[i]:.4f}). "
            "Calibration model is physically incorrect."
        )


# ---------------------------------------------------------------------------
# Test 2: Calibration denominator must never be zero
# ---------------------------------------------------------------------------

def test_calibration_denominator_valid():
    """
    (W - D + epsilon) must always be > 0 to prevent division by zero.
    epsilon is defined in constants.SIM_EPSILON.
    """
    X = np.full(N_CHANNELS, 0.5)
    result = simulate_scan(X, dark_offset=0.05, led_scale=1.0,
                           read_noise_std=0.01, rng=np.random.default_rng(0))
    D, W = result["dark"], result["white"]
    denom = W - D + SIM_EPSILON
    assert np.all(denom > 0), (
        f"Calibration denominator <= 0 in some channels: {denom}"
    )


# ---------------------------------------------------------------------------
# Test 3: Calibrated output is close to original when noise = 0
# ---------------------------------------------------------------------------

def test_calibrated_close_to_original_noiseless():
    """
    With noise=0, gain=1, placement=1, drift=0, led_scale=1,
    the calibration should recover the original spectral vector within tolerance.

    R_k = (S_k - D_k) / (W_k - D_k)
        = (dark + 1 * 1 * X_k - dark) / (dark + 1 * white_k - dark)
        = X_k / white_ref_k
        = X_k (when white_ref = 1 everywhere)
    """
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                  0.5, 0.4, 0.3, 0.35, 0.25, 0.2,
                  0.15, 0.12, 0.1, 0.08, 0.06, 0.05])
    assert len(X) == N_CHANNELS

    result = simulate_scan(X, gain=1.0, led_scale=1.0, placement_scale=1.0,
                           read_noise_std=0.0, dark_offset=0.02, led_drift=0.0,
                           rng=np.random.default_rng(42))
    R = result["calibrated"]
    max_error = np.abs(R - X).max()
    assert max_error < 0.02, (
        f"Calibrated output deviates from original by {max_error:.4f} "
        "(noiseless case). Expected deviation < 0.02. Check calibration formula."
    )


# ---------------------------------------------------------------------------
# Test 4: Calibrated output stays in physically plausible range
# ---------------------------------------------------------------------------

def test_calibrated_in_valid_range():
    """
    Calibrated reflectance must be in [0.0, 2.0] for valid sensor parameters.
    Values outside this range indicate a broken calibration model.
    """
    X = np.clip(np.random.default_rng(7).uniform(0.05, 0.95, N_CHANNELS), 0, 1)
    result = simulate_scan(X, gain=1.0, led_scale=1.0, placement_scale=1.0,
                           read_noise_std=0.005, dark_offset=0.02, led_drift=0.0,
                           rng=np.random.default_rng(7))
    R = result["calibrated"]
    assert np.all(R >= 0.0), f"Negative calibrated values found: {R[R < 0]}"
    assert np.all(R <= 2.0), f"Calibrated values > 2.0 found: {R[R > 2]}"


# ---------------------------------------------------------------------------
# Test 5: Saturation flag triggers correctly
# ---------------------------------------------------------------------------

def test_saturation_clamps_values():
    """
    When saturation_threshold is set, raw sensor readings S must be
    clamped at or below that value.
    """
    SATURATION_THRESHOLD = 0.8
    X = np.ones(N_CHANNELS) * 0.9
    result = simulate_scan(
        X, gain=1.2, led_scale=1.0, placement_scale=1.0,
        read_noise_std=0.0, dark_offset=0.0, led_drift=0.0,
        saturation_threshold=SATURATION_THRESHOLD,
        rng=np.random.default_rng(42),
    )
    S = result["raw"]
    assert np.all(S <= SATURATION_THRESHOLD + 1e-8), (
        f"Raw sensor S exceeded saturation threshold {SATURATION_THRESHOLD}. "
        f"Max S: {S.max():.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: Gain scaling propagates correctly
# ---------------------------------------------------------------------------

def test_gain_increases_raw_signal():
    """
    Higher gain should produce a higher raw signal S.
    This verifies the gain parameter is wired correctly.
    """
    X = np.full(N_CHANNELS, 0.5)

    res_low  = simulate_scan(X, gain=0.5, read_noise_std=0.0, dark_offset=0.0,
                              rng=np.random.default_rng(0))
    res_high = simulate_scan(X, gain=1.5, read_noise_std=0.0, dark_offset=0.0,
                              rng=np.random.default_rng(0))
    S_low  = res_low["raw"]
    S_high = res_high["raw"]
    assert np.all(S_high > S_low), (
        "Higher gain did not produce higher raw signal S. "
        "Check gain wiring in sensor_simulator.py."
    )


# ---------------------------------------------------------------------------
# Tests 7-10: Dedicated src/calibration.py module tests
# ---------------------------------------------------------------------------

from calibration import calibrate_reflectance, validate_calibration_pair, compute_reference_mean


def test_calibrate_reflectance_formula():
    """Verify R = (S - D) / (W - D + eps) with known values."""
    raw = np.full(18, 300.0, dtype=np.float32)
    dark = np.full(18, 100.0, dtype=np.float32)
    white = np.full(18, 500.0, dtype=np.float32)
    # (300 - 100) / (500 - 100) = 200 / 400 = 0.5
    cal = calibrate_reflectance(raw, dark, white)
    np.testing.assert_allclose(cal, 0.5, atol=1e-5)


def test_calibrate_reflectance_clips_negative():
    """Noise causing raw < dark should be clamped to 0.0."""
    raw = np.full(18, 50.0, dtype=np.float32)
    dark = np.full(18, 100.0, dtype=np.float32)
    white = np.full(18, 500.0, dtype=np.float32)
    cal = calibrate_reflectance(raw, dark, white)
    assert np.all(cal >= 0.0)


def test_validate_calibration_pair():
    """Validation passes for physical dark < white, fails when white <= dark."""
    dark = np.full(18, 100.0, dtype=np.float32)
    white = np.full(18, 500.0, dtype=np.float32)
    valid, msg = validate_calibration_pair(dark, white)
    assert valid, f"Expected valid, got: {msg}"

    # Invalid case: white <= dark
    invalid_white = np.full(18, 80.0, dtype=np.float32)
    valid, msg = validate_calibration_pair(dark, invalid_white)
    assert not valid


def test_compute_reference_mean():
    """Mean of repeated scans should equal channel average."""
    scans = np.array([
        np.full(18, 10.0),
        np.full(18, 20.0),
        np.full(18, 30.0),
    ], dtype=np.float32)
    mean = compute_reference_mean(scans)
    np.testing.assert_allclose(mean, 20.0, atol=1e-5)


def test_validate_calibration_pair_contrast_threshold():
    """
    Verify contrast ratio gate:
    W / D must be >= min_contrast_ratio (default 1.2, +20% headroom).
    """
    dark = np.full(18, 100.0, dtype=np.float32)

    # Borderline failing: white = 110 (contrast ratio = 1.1x < default 1.2x)
    low_contrast_white = np.full(18, 110.0, dtype=np.float32)
    valid, msg = validate_calibration_pair(dark, low_contrast_white)
    assert not valid
    assert "Insufficient illumination contrast" in msg

    # Passing: white = 125 (contrast ratio = 1.25x >= default 1.2x)
    valid_contrast_white = np.full(18, 125.0, dtype=np.float32)
    valid, msg = validate_calibration_pair(dark, valid_contrast_white)
    assert valid, f"Expected valid, got: {msg}"

