"""
tests/test_as7265x_driver.py
=============================
Unit tests for src/as7265x_driver.py using mock mode (no hardware required).

Covers:
  - Driver instantiation in mock mode
  - Gain and integration time validation
  - Dark / white reference capture
  - Manual calibration formula
  - Device temperature (mock returns NaN, that's OK)
  - Context manager protocol (__enter__ / __exit__)
"""

import sys
import os
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# Ensure project root and src/ are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.as7265x_driver import AS7265xDriver, AS7265X_CENTRES_NM, GAIN_OPTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_driver() -> AS7265xDriver:
    """Create a mock-mode driver (no hardware needed)."""
    return AS7265xDriver(mode="mock")


# ---------------------------------------------------------------------------
# Basic instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_mock_mode_creates_driver(self, mock_driver):
        assert mock_driver.mode == "mock"

    def test_channel_list_length(self):
        assert len(AS7265X_CENTRES_NM) == 18

    def test_channel_range(self):
        assert AS7265X_CENTRES_NM[0]  == 410
        assert AS7265X_CENTRES_NM[-1] == 940

    def test_gain_options_keys(self):
        assert set(GAIN_OPTIONS.keys()) == {1, 3, 16, 64}

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            AS7265xDriver(mode="bluetooth")


# ---------------------------------------------------------------------------
# Gain and integration time validation
# ---------------------------------------------------------------------------

class TestParameterValidation:
    def test_valid_gains(self, mock_driver):
        for g in [1, 3, 16, 64]:
            mock_driver.set_gain(g)  # should not raise
            assert mock_driver._gain == g

    def test_invalid_gain_raises(self, mock_driver):
        with pytest.raises(ValueError, match="gain must be one of"):
            mock_driver.set_gain(8)

    def test_valid_integration_time(self, mock_driver):
        mock_driver.set_integration_time(50)
        assert mock_driver._int_time == 50

    def test_min_integration_time(self, mock_driver):
        mock_driver.set_integration_time(3)
        assert mock_driver._int_time == 3

    def test_max_integration_time(self, mock_driver):
        mock_driver.set_integration_time(714)
        assert mock_driver._int_time == 714

    def test_integration_time_too_low_raises(self, mock_driver):
        with pytest.raises(ValueError):
            mock_driver.set_integration_time(2)

    def test_integration_time_too_high_raises(self, mock_driver):
        with pytest.raises(ValueError):
            mock_driver.set_integration_time(800)


# ---------------------------------------------------------------------------
# LED control (mock: no actual I/O, just no-ops)
# ---------------------------------------------------------------------------

class TestLEDControl:
    def test_valid_led_current(self, mock_driver):
        for ma in [12, 12.5, 25, 50, 100]:
            mock_driver.set_led(white=True, led_current_ma=ma)  # should not raise

    def test_valid_led_current_official_spec(self, mock_driver):
        """AMS AS7265x datasheet specifies 12.5 mA nominal minimum current."""
        mock_driver.set_led(white=True, uv=False, ir=False, led_current_ma=12.5)

    def test_ir_led_control(self, mock_driver):
        """Verify IR LED parameter is accepted and not dropped."""
        mock_driver.set_led(white=False, uv=False, ir=True, led_current_ma=50)

    def test_invalid_led_current_raises(self, mock_driver):
        with pytest.raises(ValueError, match="led_current_ma must be"):
            mock_driver.set_led(led_current_ma=30)


# ---------------------------------------------------------------------------
# Raw and calibrated reads (mock mode returns random arrays)
# ---------------------------------------------------------------------------

class TestReadFunctions:
    def test_read_raw_returns_array(self, mock_driver):
        raw = mock_driver.read_raw()
        assert isinstance(raw, np.ndarray)
        assert raw.shape == (18,)

    def test_read_raw_positive(self, mock_driver):
        raw = mock_driver.read_raw()
        assert (raw >= 0).all()

    def test_read_calibrated_returns_array(self, mock_driver):
        cal = mock_driver.read_calibrated()
        assert isinstance(cal, np.ndarray)
        assert cal.shape == (18,)


# ---------------------------------------------------------------------------
# Dark / white calibration and manual calibrate
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_dark_ref_stored(self, mock_driver):
        dark = mock_driver.measure_dark(n_readings=3)
        assert mock_driver._dark_ref is not None
        assert dark.shape == (18,)

    def test_white_ref_stored(self, mock_driver):
        mock_driver.measure_dark(n_readings=3)
        white = mock_driver.measure_white(n_readings=3)
        assert mock_driver._white_ref is not None
        assert white.shape == (18,)

    def test_manual_calibrate_no_refs_raises(self, mock_driver):
        with pytest.raises(RuntimeError, match="Dark and white references not set"):
            mock_driver.manual_calibrate(np.ones(18))

    def test_manual_calibrate_formula(self, mock_driver):
        """Verify R = (S - D) / (W - D) for known values."""
        mock_driver._dark_ref  = np.full(18, 100.0)
        mock_driver._white_ref = np.full(18, 600.0)
        raw_scan = np.full(18, 350.0)   # should give R = (350-100)/(600-100) = 0.5
        result = mock_driver.manual_calibrate(raw_scan)
        np.testing.assert_allclose(result, 0.5, atol=1e-4)

    def test_manual_calibrate_clips_at_zero(self, mock_driver):
        """Raw values below dark should be clipped to 0."""
        mock_driver._dark_ref  = np.full(18, 200.0)
        mock_driver._white_ref = np.full(18, 500.0)
        raw_scan = np.full(18, 50.0)   # below dark → clipped to 0
        result = mock_driver.manual_calibrate(raw_scan)
        assert (result == 0.0).all()

    def test_manual_calibrate_near_zero_denominator(self, mock_driver):
        """Dark ≈ White should not divide by zero (epsilon guard)."""
        mock_driver._dark_ref  = np.full(18, 500.0)
        mock_driver._white_ref = np.full(18, 500.0)
        raw_scan = np.full(18, 500.0)
        result = mock_driver.manual_calibrate(raw_scan)
        # Should return finite values (epsilon prevents division by zero)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# Temperature read
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_temperature_returns_numeric(self, mock_driver):
        temp = mock_driver.get_device_temperature()
        assert isinstance(temp, float)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_context_manager_enters(self):
        with AS7265xDriver(mode="mock") as drv:
            assert drv is not None

    def test_context_manager_close_called(self):
        drv = AS7265xDriver(mode="mock")
        close_called = []
        orig_close = drv.close
        drv.close = lambda: close_called.append(True) or orig_close()
        with drv:
            pass
        assert len(close_called) == 1


# ---------------------------------------------------------------------------
# Interface contract: simulate_scan parity
# ---------------------------------------------------------------------------

class TestSimulatorParity:
    """
    Verify that the driver mock output is compatible with the sensor_simulator
    interface (same channel count, same dtype expectations).
    """
    def test_mock_raw_shape_matches_simulator_output(self, mock_driver):
        from sensor_simulator import simulate_scan
        X_true = np.random.default_rng(0).uniform(0.2, 0.8, 18)
        sim_out = simulate_scan(X_true)
        raw     = mock_driver.read_raw()
        assert raw.shape == sim_out["calibrated"].shape

    def test_calibrate_output_range_compatible(self, mock_driver):
        """Manual calibration should produce values in [0, ∞), same as simulator calibrated."""
        mock_driver._dark_ref  = np.zeros(18)
        mock_driver._white_ref = np.ones(18)
        raw_scan = np.random.default_rng(42).uniform(0.0, 1.0, 18)
        result   = mock_driver.manual_calibrate(raw_scan)
        assert (result >= 0).all()
