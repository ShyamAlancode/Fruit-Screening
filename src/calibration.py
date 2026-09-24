"""
calibration.py
==============
Dark and white reference reflectance calibration mathematics for AS7265x
multispectral measurements.

Mathematical formulation:
    R_k = (S_k - D_k) / (W_k - D_k + ε)

Where:
    S_k : Raw sample sensor readout count for channel k
    D_k : Dark reference readout count (shutter/LED off, zero illumination)
    W_k : White reference readout count (calibrated Teflon/Spectralon tile)
    ε   : Small positive epsilon preventing division by zero
    R_k : Calibrated spectral reflectance factor for channel k (clamped >= 0.0)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

# Default numerical stabilizer
EPSILON_DEFAULT = 1e-8

# Conservative optical contrast threshold for calibration pair validation.
# NOTE: This is an empirical software engineering sanity heuristic, NOT an
# official AMS OSRAM datasheet hardware specification.
#
# Physical Derivation & Engineering Rationale:
#   Reflectance calibration normalises against available optical dynamic range:
#       R_k = (S_k - D_k) / (W_k - D_k + ε)
#
#   The denominator (W_k - D_k) represents illumination headroom above the dark floor.
#   A contrast ratio W_k / D_k = 1.2 corresponds to (W_k - D_k) = 0.2 * D_k (+20% headroom).
#   If W_k / D_k < 1.2 on any channel:
#     1. Illuminance dynamic range is critically collapsed, magnifying sensor read
#        noise by > 5x in the normalized reflectance output.
#     2. Indicates a catastrophic setup fault (e.g. illumination LED unpowered/failed,
#        optical shutter/aperture obstructed, or white standard missing).
#
#   In normal physical operation with a calibrated PTFE / Spectralon tile at 12.5 mA:
#     - Visible and NIR channels (560-940 nm) typically achieve W / D >> 3.0 to 15.0x.
#     - Short UV/blue channels (410, 435 nm) have lower LED output and silicon responsivity,
#       typically yielding lower ratios (~1.5 to 2.5x).
#   A conservative threshold of 1.2 ensures hardware faults are caught reliably without
#   triggering false alarms on lower-responsivity UV channels.
DEFAULT_MIN_CONTRAST_RATIO = 1.2


def calibrate_reflectance(
    raw_scan: np.ndarray,
    dark_ref: np.ndarray,
    white_ref: np.ndarray,
    epsilon: float = EPSILON_DEFAULT,
    clip_min: Optional[float] = 0.0,
) -> np.ndarray:
    """
    Compute calibrated reflectance vector from raw sample, dark, and white references.

    Parameters
    ----------
    raw_scan : np.ndarray
        Raw ADC channel counts, shape (N,) or (M, N).
    dark_ref : np.ndarray
        Dark reference channel counts (sensor noise/offset), shape (N,).
    white_ref : np.ndarray
        White reference channel counts (illuminated reference standard), shape (N,).
    epsilon : float
        Small positive stabilizer to prevent division by zero.
    clip_min : float, optional
        Minimum value to clamp calibrated reflectance to (default 0.0).

    Returns
    -------
    np.ndarray
        Calibrated reflectance factors matching input shape.
    """
    raw_scan = np.asarray(raw_scan, dtype=np.float32)
    dark_ref = np.asarray(dark_ref, dtype=np.float32)
    white_ref = np.asarray(white_ref, dtype=np.float32)

    denom = white_ref - dark_ref + epsilon
    num = raw_scan - dark_ref

    calibrated = num / denom
    if clip_min is not None:
        calibrated = np.clip(calibrated, clip_min, None)

    return calibrated


def validate_calibration_pair(
    dark_ref: np.ndarray,
    white_ref: np.ndarray,
    min_contrast_ratio: float = DEFAULT_MIN_CONTRAST_RATIO,
) -> Tuple[bool, str]:
    """
    Validate that a dark and white reference pair is physically plausible.

    Checks:
    1. Shape matching between dark and white reference vectors.
    2. Non-negativity of dark reference counts.
    3. Strict positivity of optical signal: W_k > D_k for all channels.
    4. Sufficient optical dynamic range: W_k / (D_k + ε) >= min_contrast_ratio.

    Parameters
    ----------
    dark_ref : np.ndarray
        Dark reference counts (LEDs off, light-tight window).
    white_ref : np.ndarray
        White reference counts (calibrated PTFE / Spectralon standard, LEDs on).
    min_contrast_ratio : float, default=1.2
        Minimum acceptable white-to-dark count ratio (W / (D + ε)).
        PROVENANCE NOTE: This threshold is an empirical software engineering
        safety heuristic (+20% illumination headroom), NOT an official AMS OSRAM
        datasheet hardware specification. In physical bring-up, VIS/NIR channels
        typically achieve 3.0-15.0x, but 1.2 is conservatively chosen to avoid
        false alarms on short-wavelength UV channels (410 nm) while still catching
        unplugged LEDs, obstructed optics, or absent reference tiles.

    Returns
    -------
    Tuple[bool, str]
        (is_valid, status_message)
    """
    dark_ref = np.asarray(dark_ref, dtype=np.float32)
    white_ref = np.asarray(white_ref, dtype=np.float32)

    if dark_ref.shape != white_ref.shape:
        return False, f"Shape mismatch: dark {dark_ref.shape} vs white {white_ref.shape}"

    if np.any(dark_ref < 0):
        return False, "Negative values detected in dark reference"

    diff = white_ref - dark_ref
    if np.any(diff <= 0):
        failing = np.where(diff <= 0)[0]
        return False, f"White reference <= dark reference on channels {failing.tolist()}"

    ratios = white_ref / (dark_ref + EPSILON_DEFAULT)
    if np.any(ratios < min_contrast_ratio):
        low_channels = np.where(ratios < min_contrast_ratio)[0]
        failing_ratios = ratios[low_channels].tolist()
        return (
            False,
            f"Insufficient illumination contrast on channels {low_channels.tolist()} "
            f"(ratios {[round(r, 2) for r in failing_ratios]} < threshold {min_contrast_ratio}). "
            "Check LED power, integration time, or reference tile placement.",
        )

    return True, "Calibration reference pair valid"



def compute_reference_mean(scans: np.ndarray) -> np.ndarray:
    """Compute channel-wise mean of repeated calibration reference scans."""
    arr = np.asarray(scans, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=0)
