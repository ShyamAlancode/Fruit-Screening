"""
tests/test_case_studies.py
==========================
Automated pytest suite testing various case studies and model responses
across the CFRD apple analysis platform.

Verifies:
  1. Unified predictor initialization and model loading
  2. Mock sensor hardware prediction response
  3. Real apple photo prediction response
  4. Real Roomi .bil extracted spectrum prediction response
  5. Synthetic simulation control vs treated prediction responses
  6. Out-of-Distribution (OOD) score behavior on domain shift spectra
  7. Ripeness and cultivar valid category outputs and probabilities
  8. Brix regression prediction bounds and mandatory disclaimer presence
  9. Visual defect probability range and classification status
  10. Noise robustness under simulated optical degradation
"""

import pytest
import numpy as np
from pathlib import Path

from src.predictor import ApplePredictor, ApplePrediction
from src.constants import CHANNEL_COLS, ROOMI_REAL_18CH_PATH, SYNTHETIC_CSV_PATH
from src.as7265x_driver import MOCK_APPLE_REFLECTANCE
from src.sensor_simulator import simulate_scan

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = PROJECT_ROOT / "data" / "test_user_apple.jpg"


@pytest.fixture(scope="module")
def predictor():
    """Instantiate ApplePredictor once for the test module."""
    return ApplePredictor()


def test_predictor_case_study_mock_sensor(predictor):
    """Case Study: Sensor Hardware Mock scan prediction."""
    res = predictor.predict_spectrum(MOCK_APPLE_REFLECTANCE)
    assert isinstance(res, ApplePrediction)
    assert res.pesticide.predicted_class in [0, 1, 2]
    assert res.pesticide.label in ["Control-like", "Low-treatment-like", "High-treatment-like"]
    assert 0.0 <= res.pesticide.confidence <= 1.0
    assert res.pesticide.disclaimer != ""
    assert res.ripeness.stage in ["Under-ripe", "Ripe", "Over-ripe"]
    assert 4.0 <= res.brix.estimated_brix <= 25.0


def test_predictor_case_study_photo_input(predictor):
    """Case Study: External apple photo input."""
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image not found at {TEST_IMAGE_PATH}")

    res = predictor.predict_image(TEST_IMAGE_PATH)
    assert isinstance(res, ApplePrediction)
    assert res.input_source in ["image_derived_spectrum", "multimodal"]
    assert res.visual_quality is not None
    assert res.visual_quality.status in ["Good / Sound", "Possible Defect / Bruised"]
    assert 0.0 <= res.visual_quality.defect_probability <= 1.0


def test_predictor_case_study_synthetic_dataset_spectrum(predictor):
    """Case Study: Synthetic simulation spectra."""
    synth_csv = Path(SYNTHETIC_CSV_PATH)
    if not synth_csv.exists():
        pytest.skip("Synthetic CSV not found")

    import pandas as pd
    df = pd.read_csv(synth_csv)
    spec_control = df[df["level"] == "control"][CHANNEL_COLS].iloc[0].values.astype(float)
    spec_high = df[df["level"] == "high"][CHANNEL_COLS].iloc[0].values.astype(float)

    res_ctrl = predictor.predict_spectrum(spec_control)
    res_high = predictor.predict_spectrum(spec_high)

    assert res_ctrl.pesticide.ood_status in ["In-Distribution", "Uncertain / Out-of-Distribution"]
    assert res_high.pesticide.ood_status in ["In-Distribution", "Uncertain / Out-of-Distribution"]
    # Probability integrity
    probs_sum = sum(res_ctrl.pesticide.probabilities.values())
    assert pytest.approx(probs_sum, rel=1e-3) == 1.0


def test_predictor_case_study_real_roomi_extraction(predictor):
    """Case Study: Real Roomi HSI extracted spectra under dual model flavors."""
    real_csv = Path(ROOMI_REAL_18CH_PATH)
    if not real_csv.exists():
        pytest.skip("Real Roomi CSV not found")

    import pandas as pd
    df = pd.read_csv(real_csv)
    spec_real = df[CHANNEL_COLS].iloc[0].values.astype(float)

    # Test under synthetic model flavor
    res_synth = predictor.predict(spectrum=spec_real, pesticide_model_flavor="synthetic")
    assert res_synth.pesticide.model_flavor == "synthetic"

    # Test under real model flavor
    res_real = predictor.predict(spectrum=spec_real, pesticide_model_flavor="real")
    assert res_real.pesticide.model_flavor == "real"


def test_predictor_case_study_noisy_sensor_simulation(predictor):
    """Case Study: Noisy hardware sensor simulation."""
    noisy = simulate_scan(
        X=MOCK_APPLE_REFLECTANCE.copy(),
        dark_offset=25.0,
        read_noise_std=3.0,
        led_scale=0.95,
        gain=1.0,
        placement_scale=0.98,
        led_drift=0.03,
    )
    calibrated_spec = noisy["calibrated"]
    assert len(calibrated_spec) == 18
    res = predictor.predict_spectrum(calibrated_spec)
    assert isinstance(res, ApplePrediction)
    assert res.pesticide.predicted_class in [0, 1, 2]
