"""
test_predictor.py
=================
Unit tests for the headless ApplePredictor engine (src/predictor.py)
verifying multi-target ML inference:
  - Pesticide treatment-condition screening (3-class PCA+LDA)
  - Ripeness stage classification (3-class RandomForest)
  - Cultivar / variety classification (5-class RandomForest)
  - Brix sweetness regression (18-channel ElasticNet)
  - Visual defect risk (MobileNetV3 / heuristic)
"""

import pytest
import numpy as np
from pathlib import Path

from src.predictor import ApplePredictor, ApplePrediction
from src.as7265x_driver import MOCK_APPLE_REFLECTANCE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE = PROJECT_ROOT / "data" / "test_user_apple.jpg"


@pytest.fixture(scope="module")
def predictor():
    """Shared ApplePredictor instance."""
    return ApplePredictor()


def test_predictor_instantiation(predictor):
    """Predictor should initialize with models loaded."""
    assert predictor is not None
    assert predictor.pesticide_model_synth is not None
    assert predictor.ripeness_model is not None
    assert predictor.variety_model is not None


def test_predict_mock_spectrum(predictor):
    """Predict with mock 18-channel spectrum."""
    res = predictor.predict(spectrum=MOCK_APPLE_REFLECTANCE)
    assert isinstance(res, ApplePrediction)
    assert len(res.spectrum_18ch) == 18

    # Pesticide
    assert res.pesticide.predicted_class in [0, 1, 2]
    assert res.pesticide.label in ["Control-like", "Low-treatment-like", "High-treatment-like"]
    assert 0.0 <= res.pesticide.confidence <= 1.0
    assert len(res.pesticide.probabilities) == 3

    # Ripeness
    assert res.ripeness.stage in ["Under-ripe", "Ripe", "Over-ripe"]
    assert 0.0 <= res.ripeness.confidence <= 1.0
    assert len(res.ripeness.probabilities) == 3

    # Variety
    assert res.variety.cultivar in [
        "kinnaur", "red delicious new zealand", "red delicious usa", "royal gala", "washington"
    ]
    assert 0.0 <= res.variety.confidence <= 1.0
    assert len(res.variety.probabilities) == 5

    # Brix
    assert 4.0 <= res.brix.estimated_brix <= 20.0
    assert res.brix.unit == "°Bx"


def test_predict_image(predictor):
    """Predict from apple photo if available."""
    if not TEST_IMAGE.exists():
        pytest.skip(f"Test image not found at {TEST_IMAGE}")

    res = predictor.predict_image(TEST_IMAGE)
    assert isinstance(res, ApplePrediction)
    assert len(res.spectrum_18ch) == 18
    assert res.visual_quality is not None
    assert res.visual_quality.status in ["Good / Sound", "Possible Defect / Bruised"]
    assert 0.0 <= res.visual_quality.defect_probability <= 1.0


def test_invalid_spectrum_shape_raises(predictor):
    """Passing spectrum of wrong length must raise ValueError."""
    bad_spectrum = np.array([0.1, 0.2, 0.3])  # 3 channels instead of 18
    with pytest.raises(ValueError, match="Spectrum must have shape"):
        predictor.predict(spectrum=bad_spectrum)


def test_serialization_dict_and_json(predictor):
    """Verify to_dict() and to_json() serialization works."""
    res = predictor.predict(spectrum=MOCK_APPLE_REFLECTANCE)
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "pesticide" in d
    assert "ripeness" in d
    assert "variety" in d
    assert "brix" in d

    js = res.to_json()
    assert isinstance(js, str)
    assert "estimated_brix" in js


def test_pesticide_model_flavor_toggle(predictor):
    """Verify switching between synthetic and real pesticide models."""
    res_synth = predictor.predict(spectrum=MOCK_APPLE_REFLECTANCE, pesticide_model_flavor="synthetic")
    assert res_synth.pesticide.model_flavor == "synthetic"

    res_real = predictor.predict(spectrum=MOCK_APPLE_REFLECTANCE, pesticide_model_flavor="real")
    assert res_real.pesticide.model_flavor == "real"
