"""
CFRD Multi-spectral Screening & Apple Quality Analysis core package.
"""

from .constants import CHANNEL_COLS, AS7265X_CENTRES_NM, N_CHANNELS
from .calibration import (
    calibrate_reflectance,
    validate_calibration_pair,
    compute_reference_mean,
    DEFAULT_MIN_CONTRAST_RATIO,
)
from .as7265x_driver import AS7265xDriver
from .sensor_simulator import simulate_scan
from .predictor import ApplePredictor, ApplePrediction

