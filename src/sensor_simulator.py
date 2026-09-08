"""
sensor_simulator.py

Mathematical model of the AS7265x 18-channel spectral sensor.
Simulates the full dark/white/raw/calibrated measurement pipeline.

The AS7265x is a 3-chip (AS72651 master + AS72652 + AS72653 slave)
18-channel optical sensor. Each channel has a Gaussian response centred
at a specific wavelength.

This model does NOT represent measured hardware performance.
It is a mathematical approximation for software pre-hardware development.

ALL 7 parameters are exposed as arguments with documented defaults.
No parameter is hardcoded inside any function.
"""

import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    N_CHANNELS, SIM_EPSILON, SIMULATOR_NOTE,
    SIM_DARK_OFFSET_DEFAULT, SIM_READ_NOISE_DEFAULT,
    SIM_LED_SCALE_DEFAULT, SIM_GAIN_DEFAULT,
    SIM_PLACEMENT_DEFAULT, SIM_DRIFT_DEFAULT,
)


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def simulate_scan(
    X: np.ndarray,
    gain: float = SIM_GAIN_DEFAULT,
    led_scale: float = SIM_LED_SCALE_DEFAULT,
    placement_scale: float = SIM_PLACEMENT_DEFAULT,
    read_noise_std: float = SIM_READ_NOISE_DEFAULT,
    dark_offset: float = SIM_DARK_OFFSET_DEFAULT,
    led_drift: float = SIM_DRIFT_DEFAULT,
    saturation_threshold: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    white_ref: Optional[np.ndarray] = None,
) -> dict:
    """
    Simulate one AS7265x scan of an apple with the given spectral vector X.

    Parameters
    ----------
    X : np.ndarray, shape (18,)
        True reflectance spectrum at the 18 AS7265x channel wavelengths.
        Values should be in [0.0, 1.0] for physically valid inputs.
    gain : float
        Sensor electronic gain (0.5 to 2.0). Scales raw signal amplitude.
    led_scale : float
        LED brightness scale (0.7 to 1.3). 1.0 = nominal brightness.
    placement_scale : float
        Fruit placement / coupling scatter factor (0.7 to 1.3).
        1.0 = ideal placement centred under sensor.
    read_noise_std : float
        Gaussian read noise standard deviation (0.0 to 0.10).
    dark_offset : float
        Dark current offset added to all readings (0.0 to 0.10).
    led_drift : float
        Per-channel LED spectral drift factor (0.0 to 0.20).
        Each channel shifts by (1 + led_drift * delta_k) where delta_k ~ U(-1, 1).
    saturation_threshold : float or None
        If provided, raw readings S and white W are clamped at this value.
        Models sensor saturation / analog-to-digital ceiling.
    rng : np.random.Generator or None
        Random number generator. If None, a new default_rng is created.
    white_ref : np.ndarray or None
        White reference reflectance spectrum (18 values). Defaults to ones
        (perfect Teflon / Spectralon panel with unit reflectance per channel).

    Returns
    -------
    dict with keys:
        dark   : np.ndarray (18,)  — dark current measurement
        white  : np.ndarray (18,)  — white reference measurement
        raw    : np.ndarray (18,)  — raw fruit scan
        calibrated : np.ndarray (18,) — dark/white calibrated reflectance
        saturated  : bool — True if any channel hit saturation threshold
        params : dict — all simulation parameters used
    """
    assert X.shape == (N_CHANNELS,), (
        f"Input X must have shape ({N_CHANNELS},), got {X.shape}"
    )

    if rng is None:
        rng = np.random.default_rng()

    if white_ref is None:
        white_ref = np.ones(N_CHANNELS)
    else:
        assert white_ref.shape == (N_CHANNELS,), (
            f"white_ref must have shape ({N_CHANNELS},), got {white_ref.shape}"
        )

    # Per-channel LED drift direction (sampled once per scan, fixed per channel)
    delta = rng.uniform(-1.0, 1.0, N_CHANNELS)

    # -----------------------------------------------------------------------
    # Dark reference: sensor readout with light source off
    # D_k = dark_offset + noise
    # -----------------------------------------------------------------------
    D = dark_offset + rng.normal(0.0, read_noise_std, N_CHANNELS)

    # -----------------------------------------------------------------------
    # White reference: scan of Teflon / Spectralon panel
    # W_k = dark_offset + led_scale * white_ref_k + noise
    # -----------------------------------------------------------------------
    W = dark_offset + led_scale * white_ref + rng.normal(0.0, read_noise_std, N_CHANNELS)

    # -----------------------------------------------------------------------
    # Raw fruit scan: apple reflectance with all degradation applied
    # S_k = dark_offset + led_scale * gain * placement * (1 + drift * delta_k) * X_k + noise
    # -----------------------------------------------------------------------
    S = (
        dark_offset
        + led_scale * gain * placement_scale * (1.0 + led_drift * delta) * X
        + rng.normal(0.0, read_noise_std, N_CHANNELS)
    )

    # -----------------------------------------------------------------------
    # Saturation clamping
    # -----------------------------------------------------------------------
    saturated = False
    if saturation_threshold is not None:
        S_clamped = np.minimum(S, saturation_threshold)
        W_clamped = np.minimum(W, saturation_threshold)
        if np.any(S > saturation_threshold) or np.any(W > saturation_threshold):
            saturated = True
        S = S_clamped
        W = W_clamped

    # -----------------------------------------------------------------------
    # Calibration: remove dark contribution and normalise against white ref
    # R_k = (S_k - D_k) / (W_k - D_k + epsilon)
    # -----------------------------------------------------------------------
    calibrated = (S - D) / (W - D + SIM_EPSILON)

    return {
        "dark": D,
        "white": W,
        "raw": S,
        "calibrated": calibrated,
        "saturated": saturated,
        "params": {
            "gain": gain,
            "led_scale": led_scale,
            "placement_scale": placement_scale,
            "read_noise_std": read_noise_std,
            "dark_offset": dark_offset,
            "led_drift": led_drift,
            "saturation_threshold": saturation_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Batch simulation
# ---------------------------------------------------------------------------

def simulate_batch(
    X_batch: np.ndarray,
    seed: int = 42,
    **kwargs,
) -> np.ndarray:
    """
    Simulate scans for a batch of spectra.

    Parameters
    ----------
    X_batch : np.ndarray, shape (N, 18)
        N spectral vectors to simulate.
    seed : int
        Random seed for reproducibility.
    **kwargs : dict
        Passed directly to simulate_scan(). Same parameters apply to all scans.

    Returns
    -------
    np.ndarray, shape (N, 18) — calibrated reflectance for each scan.
    """
    assert X_batch.ndim == 2 and X_batch.shape[1] == N_CHANNELS, (
        f"X_batch must have shape (N, {N_CHANNELS}), got {X_batch.shape}"
    )

    rng = np.random.default_rng(seed)
    results = np.zeros_like(X_batch, dtype=float)

    for i, X in enumerate(X_batch):
        scan = simulate_scan(X, rng=rng, **kwargs)
        results[i] = scan["calibrated"]

    return results


# ---------------------------------------------------------------------------
# Parameter range validation helper
# ---------------------------------------------------------------------------

PARAMETER_RANGES = {
    "gain":               (0.5,  2.0),
    "led_scale":          (0.7,  1.3),
    "placement_scale":    (0.7,  1.3),
    "read_noise_std":     (0.0,  0.10),
    "dark_offset":        (0.0,  0.10),
    "led_drift":          (0.0,  0.20),
}


def validate_params(**kwargs) -> list[str]:
    """
    Check that simulation parameters are within physically plausible ranges.
    Returns a list of warning strings (empty if all OK).
    """
    warnings = []
    for param, value in kwargs.items():
        if param in PARAMETER_RANGES:
            lo, hi = PARAMETER_RANGES[param]
            if value < lo or value > hi:
                warnings.append(
                    f"Parameter '{param}' value {value} is outside "
                    f"recommended range [{lo}, {hi}]."
                )
    return warnings


if __name__ == "__main__":
    print(SIMULATOR_NOTE)
    print()

    # Quick sanity check
    rng = np.random.default_rng(42)
    X_test = rng.uniform(0.1, 0.9, N_CHANNELS)

    result = simulate_scan(
        X_test,
        gain=1.0,
        led_scale=1.0,
        placement_scale=1.0,
        read_noise_std=0.0,
        dark_offset=0.02,
        led_drift=0.0,
        rng=np.random.default_rng(0),
    )

    max_err = float(np.abs(result["calibrated"] - X_test).max())
    print(f"Noiseless recovery test: max channel error = {max_err:.5f}")
    assert max_err < 0.02, f"Recovery error too large: {max_err}"
    print("PASS")
