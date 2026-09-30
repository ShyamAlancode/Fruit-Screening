"""
features.py
===========
Biochemical vegetation indices, spectral ratio metrics, and intra-fruit
differential feature extraction for apple spectroscopy.

Implements the physics-based indices specified in CFRD_REAL_IMPLEMENTATION_PLAN.md:
  1. Anthocyanin Reflectance Index (ARI) - red pigment / blush
  2. Chlorophyll Index (CI) - ground de-greening / ripeness
  3. Carotenoid Reflectance Index (CRI) - yellow pigment / background color
  4. Red-edge NDVI - chlorophyll absorption boundary
  5. NIR Plateau Slope - tissue density and cell scattering
  6. Water Index (WI) - 900/940 nm water absorption band
  7. Intra-Fruit Differential Features - canceling 60%+ biological cuticle wax baseline
"""

from __future__ import annotations
from typing import Dict, List, Union, Any, Tuple
import numpy as np

# AS7265x nominal center wavelengths (nm)
# Channels 0..17 map to:
# 410, 435, 460, 485, 510, 535, 560, 585, 610, 645, 680, 705, 730, 760, 810, 860, 900, 940
WAVELENGTH_MAP = {
    410: 0, 435: 1, 460: 2, 485: 3, 510: 4, 535: 5,
    560: 6, 585: 7, 610: 8, 645: 9, 680: 10, 705: 11,
    730: 12, 760: 13, 810: 14, 860: 15, 900: 16, 940: 17,
}


def compute_biochemical_indices(spectrum: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Compute key plant physiological & biochemical indices from calibrated 18-channel reflectance.

    Parameters
    ----------
    spectrum : array-like of shape (18,)
        Calibrated reflectance values in order [410nm ... 940nm].

    Returns
    -------
    Dict[str, float]
        Dictionary containing ARI, CI, CRI, red_edge, NIR_slope, water_index, and ndvi_analog.
    """
    spec = np.asarray(spectrum, dtype=float)
    if spec.shape != (18,):
        raise ValueError(f"Expected 18-channel spectrum, got shape {spec.shape}")

    eps = 1e-6
    # Map wavelengths
    r510 = max(float(spec[WAVELENGTH_MAP[510]]), eps)
    r535 = max(float(spec[WAVELENGTH_MAP[535]]), eps)
    r560 = max(float(spec[WAVELENGTH_MAP[560]]), eps)
    r680 = max(float(spec[WAVELENGTH_MAP[680]]), eps)
    r705 = max(float(spec[WAVELENGTH_MAP[705]]), eps)
    r730 = max(float(spec[WAVELENGTH_MAP[730]]), eps)
    r760 = max(float(spec[WAVELENGTH_MAP[760]]), eps)
    r810 = max(float(spec[WAVELENGTH_MAP[810]]), eps)
    r900 = max(float(spec[WAVELENGTH_MAP[900]]), eps)
    r940 = max(float(spec[WAVELENGTH_MAP[940]]), eps)

    # 1. Anthocyanin Reflectance Index (Gitelson et al., 2001)
    ari = (1.0 / r535) - (1.0 / r705)

    # 2. Chlorophyll Index (Gitelson & Merzlyak, 1994)
    ci = (r760 / r705) - 1.0

    # 3. Carotenoid Reflectance Index (Gitelson et al., 2002)
    cri = (1.0 / r510) - (1.0 / r560)

    # 4. Red-Edge NDVI (Normalized Difference Vegetation Index analog)
    red_edge = (r730 - r680) / (r730 + r680)

    # 5. NIR Plateau Slope (730 to 940 nm tissue scattering)
    nir_slope = (r940 - r730) / (940.0 - 730.0)

    # 6. Water Band Ratio (first vibrational overtone proxy)
    water_index = r900 / r940

    # 7. Standard NIR-Red ratio
    ndvi_analog = (r810 - r680) / (r810 + r680)

    return {
        "ARI": round(float(ari), 4),
        "CI": round(float(ci), 4),
        "CRI": round(float(cri), 4),
        "red_edge": round(float(red_edge), 4),
        "nir_slope": round(float(nir_slope), 6),
        "water_index": round(float(water_index), 4),
        "ndvi_analog": round(float(ndvi_analog), 4),
    }


def compute_intra_fruit_differential(
    control_spectrum: Union[List[float], np.ndarray],
    target_spectra: List[Union[List[float], np.ndarray]],
) -> Dict[str, Any]:
    """
    Implements the Intra-Fruit Differential Screening protocol (Plan §5 & §6.1).

    Cancels out apple cuticle wax and inter-fruit reflectance variations (>60%)
    by subtracting the reference control spot (calyx / clean area) on the SAME fruit.

    Parameters
    ----------
    control_spectrum : (18,) array
        Control baseline spot (e.g. apple calyx / unexposed area).
    target_spectra : list of (18,) arrays
        Equatorial perimeter target spots (where residues or variations deposit).

    Returns
    -------
    Dict[str, Any]
        - 'target_avg': (18,) mean target spectrum
        - 'control': (18,) control spectrum
        - 'delta_r': (18,) raw difference (T_avg - C)
        - 'delta_diff_17': (17,) adjacent spectral slope difference: diff(T_avg) - diff(C)
        - 'wax_baseline_suppression_pct': estimated biological offset cancelled
        - 'perimeter_spatial_std': standard deviation across target spots
    """
    c = np.asarray(control_spectrum, dtype=np.float32)
    if c.shape != (18,):
        raise ValueError(f"control_spectrum must have shape (18,), got {c.shape}")

    if not target_spectra:
        raise ValueError("target_spectra must contain at least 1 target spot measurement.")

    targets = [np.asarray(t, dtype=np.float32) for t in target_spectra]
    for i, t in enumerate(targets):
        if t.shape != (18,):
            raise ValueError(f"target_spectra[{i}] must have shape (18,), got {t.shape}")

    targets_stack = np.stack(targets, axis=0)
    t_avg = np.mean(targets_stack, axis=0)
    t_std = np.std(targets_stack, axis=0) if len(targets) > 1 else np.zeros(18, dtype=np.float32)

    # 1. First-order difference: ΔR = T_avg - C
    delta_r = t_avg - c

    # 2. Second-order slope difference: Δ(ΔR) = diff(T_avg) - diff(C)
    # 17 features capturing chemical band curvature shifts independent of overall brightness
    delta_diff_17 = np.diff(t_avg) - np.diff(c)

    # 3. Biological baseline suppression estimate
    # Raw magnitude vs differential magnitude
    raw_mag = float(np.linalg.norm(t_avg))
    diff_mag = float(np.linalg.norm(delta_r))
    suppression_pct = max(0.0, min(99.0, (1.0 - (diff_mag / (raw_mag + 1e-6))) * 100.0))

    return {
        "target_avg": t_avg,
        "control": c,
        "delta_r": delta_r,
        "delta_diff_17": delta_diff_17,
        "wax_baseline_suppression_pct": round(suppression_pct, 1),
        "perimeter_spatial_std": t_std,
        "num_target_spots": len(targets),
    }
