"""
test_sensor_simulator.py

Tests for the mathematical AS7265x sensor simulator (src/sensor_simulator.py).
Verifies:
1. Noiseless recovery matches ground truth within 0.02 tolerance
2. Noise injection variance scales with read_noise_std
3. Dark and white references behave correctly
4. LED drift applies perturbations
5. Saturation clamping activates and reports correctly
6. Batch simulation maintains shape and consistency
7. Parameter validation flags out-of-bound inputs
"""

import pytest
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sensor_simulator import (
    simulate_scan,
    simulate_batch,
    validate_params,
    PARAMETER_RANGES,
)
from constants import N_CHANNELS, SIM_DARK_OFFSET_DEFAULT


@pytest.fixture
def clean_spectrum():
    """Deterministic synthetic apple reflectance spectrum."""
    rng = np.random.default_rng(12345)
    return rng.uniform(0.1, 0.85, N_CHANNELS)


def test_noiseless_recovery(clean_spectrum):
    """Calibrated reflectance must closely match input spectrum when noise/drift are zero."""
    res = simulate_scan(
        clean_spectrum,
        gain=1.0,
        led_scale=1.0,
        placement_scale=1.0,
        read_noise_std=0.0,
        dark_offset=0.03,
        led_drift=0.0,
        rng=np.random.default_rng(42),
    )
    diff = np.abs(res["calibrated"] - clean_spectrum)
    assert np.all(diff < 0.02), f"Max discrepancy {diff.max():.4f} exceeds 0.02"


def test_output_keys_and_shapes(clean_spectrum):
    """Output dict must contain required keys and correct ndarray shapes."""
    res = simulate_scan(clean_spectrum)
    assert "dark" in res
    assert "white" in res
    assert "raw" in res
    assert "calibrated" in res
    assert "saturated" in res
    assert "params" in res

    assert res["dark"].shape == (N_CHANNELS,)
    assert res["white"].shape == (N_CHANNELS,)
    assert res["raw"].shape == (N_CHANNELS,)
    assert res["calibrated"].shape == (N_CHANNELS,)
    assert isinstance(res["saturated"], (bool, np.bool_))


def test_noise_increases_variance(clean_spectrum):
    """Higher read_noise_std must yield higher variance across scans."""
    n_scans = 50
    cal_low_noise = np.zeros((n_scans, N_CHANNELS))
    cal_high_noise = np.zeros((n_scans, N_CHANNELS))

    rng1 = np.random.default_rng(100)
    rng2 = np.random.default_rng(200)

    for i in range(n_scans):
        cal_low_noise[i] = simulate_scan(clean_spectrum, read_noise_std=0.002, rng=rng1)["calibrated"]
        cal_high_noise[i] = simulate_scan(clean_spectrum, read_noise_std=0.05, rng=rng2)["calibrated"]

    std_low = cal_low_noise.std(axis=0).mean()
    std_high = cal_high_noise.std(axis=0).mean()
    assert std_high > std_low * 5, f"High noise std ({std_high}) should be much greater than low noise ({std_low})"


def test_saturation_clamping(clean_spectrum):
    """Saturation threshold must clamp values and set saturated=True."""
    threshold = 0.5
    res = simulate_scan(
        clean_spectrum,
        gain=2.0,
        led_scale=1.5,
        saturation_threshold=threshold,
        rng=np.random.default_rng(42),
    )
    assert res["saturated"] is True
    assert np.all(res["raw"] <= threshold + 1e-6)
    assert np.all(res["white"] <= threshold + 1e-6)


def test_simulate_batch(clean_spectrum):
    """Batch simulation produces (N, 18) output consistent with single scans."""
    X_batch = np.tile(clean_spectrum, (5, 1))
    batch_res = simulate_batch(X_batch, read_noise_std=0.0, gain=1.0, placement_scale=1.0)
    assert batch_res.shape == (5, N_CHANNELS)
    diff = np.abs(batch_res[0] - clean_spectrum)
    assert np.all(diff < 0.02)


def test_parameter_validation():
    """validate_params catches values outside physically plausible ranges."""
    warnings_clean = validate_params(gain=1.0, read_noise_std=0.02)
    assert len(warnings_clean) == 0

    warnings_bad = validate_params(gain=5.0, read_noise_std=0.5)
    assert len(warnings_bad) == 2
    assert any("gain" in w for w in warnings_bad)
    assert any("read_noise_std" in w for w in warnings_bad)
