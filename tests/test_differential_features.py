"""
test_differential_features.py
==============================
Unit tests for biochemical vegetation indices and intra-fruit differential screening
as specified in CFRD_REAL_IMPLEMENTATION_PLAN.md (§5 & §6.1).
"""

import pytest
import numpy as np
from src.features import compute_biochemical_indices, compute_intra_fruit_differential
from src.predictor import ApplePredictor, DifferentialScreeningResult
from src.as7265x_driver import AS7265xDriver, MOCK_APPLE_REFLECTANCE


def test_biochemical_indices_keys_and_values():
    """Verify all biochemical indices (ARI, CI, CRI, NDVI, etc.) are computed."""
    spec = MOCK_APPLE_REFLECTANCE.copy()
    indices = compute_biochemical_indices(spec)

    expected_keys = {"ARI", "CI", "CRI", "red_edge", "nir_slope", "water_index", "ndvi_analog"}
    assert set(indices.keys()) == expected_keys

    # Check realistic ranges
    assert -5.0 < indices["ARI"] < 15.0
    assert -1.0 <= indices["red_edge"] <= 1.0
    assert 0.0 < indices["water_index"] < 2.0


def test_biochemical_indices_invalid_shape():
    """Verify ValueError is raised if spectrum is not 18 channels."""
    with pytest.raises(ValueError):
        compute_biochemical_indices([0.1, 0.2, 0.3])


def test_intra_fruit_differential_cancellation():
    """
    Verify that subtracting same-fruit control eliminates biological cuticle wax baseline.
    """
    # Simulate an apple with 60% higher cuticle reflectance (Apple A vs Apple B baseline dilemma)
    base_cuticle = MOCK_APPLE_REFLECTANCE * 1.5
    control_calyx = base_cuticle.copy()

    # Perimeter spots with identical baseline cuticle wax + subtle 1% surface variation
    perimeter_spots = [
        base_cuticle + np.random.normal(0, 0.005, size=18).astype(np.float32)
        for _ in range(5)
    ]

    res = compute_intra_fruit_differential(control_calyx, perimeter_spots)

    assert "delta_r" in res
    assert "delta_diff_17" in res
    assert len(res["delta_diff_17"]) == 17
    # Baseline cancellation should be high (>85%) because control cancels the 1.5x scaling
    assert res["wax_baseline_suppression_pct"] > 80.0


def test_predictor_predict_differential_clean_vs_treated():
    """
    Test 3-state differential screening in ApplePredictor:
    'LIKELY CLEAN', 'INCONCLUSIVE', 'TREATMENT SUSPECTED'.
    """
    predictor = ApplePredictor()
    base_spectrum = MOCK_APPLE_REFLECTANCE.copy()

    # Case 1: Pure clean apple (target spots match control calyx within low noise)
    clean_targets = [
        base_spectrum + np.random.normal(0, 0.002, size=18).astype(np.float32)
        for _ in range(4)
    ]
    res_clean = predictor.predict_differential(base_spectrum, clean_targets)

    assert isinstance(res_clean, DifferentialScreeningResult)
    assert res_clean.status == "LIKELY CLEAN"
    assert res_clean.anomaly_score < 0.40
    assert res_clean.wax_baseline_suppression_pct > 80.0

    # Case 2: Treated apple with distinct non-uniform residue curvature distortion
    treated_shift = np.zeros(18, dtype=np.float32)
    treated_shift[6:12] += 0.08  # strong selective absorption/reflection shift in 560-705nm
    treated_targets = [
        base_spectrum + treated_shift + np.random.normal(0, 0.004, size=18).astype(np.float32)
        for _ in range(4)
    ]
    res_treated = predictor.predict_differential(base_spectrum, treated_targets)

    assert res_treated.status == "TREATMENT SUSPECTED"
    assert res_treated.anomaly_score >= 0.70


def test_driver_read_with_illumination_sequence_mock():
    """Test multi-illumination time-multiplexing in mock mode."""
    driver = AS7265xDriver(mode="mock")
    modes = ["white", "uv", "ir", "dark"]
    seq_data = driver.read_with_illumination_sequence(modes)

    assert set(seq_data.keys()) == set(modes)
    for m in modes:
        assert isinstance(seq_data[m], np.ndarray)
        assert seq_data[m].shape == (18,)
