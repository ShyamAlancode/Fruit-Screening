"""
hsi_to_18ch.py

Extracts 18-channel synthetic AS7265x spectral vectors from ENVI .bil
hyperspectral image cubes (real Roomi pesticide dataset).

IMPORTANT: Only runs after scripts/00_verify_roomi_bil.py confirms GO.

Pipeline per apple:
  1. Load .bil cube (H x W x bands, uint16) via spectral.io.envi
  2. Apply circular ROI mask centred on image (radius = 0.4 * min(H, W))
  3. Average pixels inside ROI → 1D spectrum (bands,)
  4. Apply white/dark calibration if raw counts (check paper section 2)
  5. Gaussian-weighted integration → 18 AS7265x channel values
  6. Assign treatment labels from folder path
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(__file__))
from constants import (
    AS7265X_CENTRES_NM, AS7265X_SIGMA, CHANNEL_COLS,
    LEVEL_MAP, REQUIRED_CSV_COLS,
)


# ---------------------------------------------------------------------------
# Folder → label mapping
# ---------------------------------------------------------------------------
# Roomi folder structure after download and extraction:
# roomi_pesticide_hsi/
#   Apple_samples/
#     Monostar/  Fresh/ Low/ High/
#     Nativo/    Fresh/ Low/ High/
#   Fungicide_Apple/       (Nativo high concentration samples)
#   Insecticide_Apple/     (Monostar samples)
#
# The TIF folder structure that already exists on disk:
#   Apple-Hyperspectral images/
#     Apple_Samples/Monostar/Fresh/  Low/  High/
#     Apple_Samples/Nativo/  Fresh/  Low/  High/
#     Fungicide_Apple/DIB_AppleNativo/ DIB_AppleNativohigh/ DIB_AppleNativolow/
#     Insecticide_Apple/DIB_Applemono/ DIB_Applemonohigh/ DIB_Applemonolow/
#
# When Roomi .bil files are downloaded, this module will walk their
# structure. Adjust FOLDER_LABEL_MAP to match actual downloaded structure.

FOLDER_LABEL_MAP = {
    # Monostar (insecticide formulation)
    ("monostar", "fresh"):  {"treatment_family": "Monostar", "level": "control",  "class_3": 0},
    ("monostar", "low"):    {"treatment_family": "Monostar", "level": "low",       "class_3": 1},
    ("monostar", "high"):   {"treatment_family": "Monostar", "level": "high",      "class_3": 2},
    # Nativo (fungicide formulation)
    ("nativo", "fresh"):    {"treatment_family": "Nativo",   "level": "control",   "class_3": 0},
    ("nativo", "low"):      {"treatment_family": "Nativo",   "level": "low",       "class_3": 1},
    ("nativo", "high"):     {"treatment_family": "Nativo",   "level": "high",      "class_3": 2},
    # Alternate folder name patterns (adjust after download and inspection)
    ("dib_applemono", None):    {"treatment_family": "Monostar", "level": "control",  "class_3": 0},
    ("dib_applemonolow", None): {"treatment_family": "Monostar", "level": "low",      "class_3": 1},
    ("dib_applemonohigh", None):{"treatment_family": "Monostar", "level": "high",     "class_3": 2},
    ("dib_applenativo", None):  {"treatment_family": "Nativo",   "level": "control",  "class_3": 0},
    ("dib_applenativolow", None):{"treatment_family": "Nativo",  "level": "low",      "class_3": 1},
    ("dib_applenativohigh", None):{"treatment_family": "Nativo", "level": "high",     "class_3": 2},
}


def _match_labels_from_path(filepath: Path) -> Optional[dict]:
    """
    Determine treatment label dict from the filepath.
    Returns None if the path does not match any known pattern.
    """
    parts = [p.lower() for p in filepath.parts]

    for (family_key, level_key), labels in FOLDER_LABEL_MAP.items():
        family_match = any(family_key in p for p in parts)
        if level_key is None:
            # Match by full folder name
            leaf_match = any(family_key in p for p in parts)
            if leaf_match:
                return labels.copy()
        elif family_match:
            level_match = any(level_key in p for p in parts)
            if level_match:
                return labels.copy()

    return None


def _extract_fruit_id(filepath: Path) -> int:
    """Extract integer fruit ID from filename. Returns -1 if not found."""
    name = filepath.stem
    # Patterns: 1_1PU, A1_F1, MA1_F1, APP1_PU1, apple1
    matches = re.findall(r'\d+', name)
    if matches:
        return int(matches[0])
    return -1


# ---------------------------------------------------------------------------
# Gaussian channel integration
# ---------------------------------------------------------------------------

def gaussian_channel_integration(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    centres: list = AS7265X_CENTRES_NM,
    sigma: float = AS7265X_SIGMA,
) -> np.ndarray:
    """
    Integrate a continuous spectrum into 18 AS7265x-like channel values
    using Gaussian filter responses.

    V_k = sum_i [ R(lambda_i) * G_k(lambda_i) ] / sum_i [ G_k(lambda_i) ]

    Only wavelengths within [min(centres)-3*sigma, max(centres)+3*sigma]
    are used to avoid edge artifacts.

    Parameters
    ----------
    spectrum    : (B,) mean ROI spectrum
    wavelengths : (B,) wavelength in nm for each band
    centres     : list of 18 AS7265x channel centre wavelengths (nm)
    sigma       : Gaussian sigma in nm

    Returns
    -------
    np.ndarray (18,) — simulated channel values
    """
    channels = np.zeros(len(centres), dtype=float)

    for k, centre in enumerate(centres):
        # Gaussian weights
        G = np.exp(-0.5 * ((wavelengths - centre) / sigma) ** 2)
        weight_sum = G.sum()
        if weight_sum < 1e-10:
            channels[k] = 0.0
        else:
            channels[k] = float(np.dot(spectrum, G) / weight_sum)

    return channels


# ---------------------------------------------------------------------------
# Circular ROI extraction
# ---------------------------------------------------------------------------

def extract_roi_spectrum(
    cube: np.ndarray,
    radius_fraction: float = 0.4,
) -> np.ndarray:
    """
    Average spectrum over a circular region of interest centred on the image.

    Parameters
    ----------
    cube            : (H, W, B) hyperspectral cube
    radius_fraction : radius = radius_fraction * min(H, W)

    Returns
    -------
    np.ndarray (B,) — mean spectrum over ROI pixels
    """
    H, W, B = cube.shape
    cx, cy  = W // 2, H // 2
    radius  = int(radius_fraction * min(H, W))

    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2

    # Extract pixels inside ROI: shape (N_pixels_in_roi, B)
    roi_pixels = cube[mask]  # shape (N, B)
    return roi_pixels.mean(axis=0).astype(float)


# ---------------------------------------------------------------------------
# ENVI .bil loader
# ---------------------------------------------------------------------------

def load_bil_cube(
    bil_path: str,
    hdr_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an ENVI .bil hyperspectral cube.

    Returns
    -------
    cube        : (H, W, B) np.ndarray
    wavelengths : (B,) np.ndarray of wavelengths in nm
    """
    try:
        import spectral.io.envi as envi
    except ImportError:
        raise ImportError(
            "spectral library not found. Install with: pip install spectral"
        )

    img = envi.open(hdr_path, bil_path)
    cube = img.load()  # loads as (H, W, B) numpy array

    if img.bands.centers:
        wavelengths = np.array(img.bands.centers)
    else:
        raise ValueError(
            f"No wavelength information in header: {hdr_path}\n"
            "The .hdr file must contain a 'wavelength' list."
        )

    return np.array(cube), wavelengths


# ---------------------------------------------------------------------------
# Per-apple extraction
# ---------------------------------------------------------------------------

def extract_apple_18ch(
    bil_path: str,
    hdr_path: str,
    labels: dict,
    fruit_id: int,
    dark_ref: Optional[np.ndarray] = None,
    white_ref: Optional[np.ndarray] = None,
    radius_fraction: float = 0.4,
) -> Optional[pd.Series]:
    """
    Extract a single apple's 18-channel vector from its .bil cube.

    Parameters
    ----------
    bil_path    : path to .bil file
    hdr_path    : path to companion .hdr file
    labels      : dict with treatment_family, level, class_3
    fruit_id    : integer apple identity number
    dark_ref    : (B,) dark reference spectrum (if raw counts, required for calibration)
    white_ref   : (B,) white reference spectrum (if raw counts, required for calibration)
    radius_fraction : ROI radius as fraction of min image dimension

    Returns
    -------
    pd.Series with all REQUIRED_CSV_COLS, or None on failure
    """
    try:
        cube, wavelengths = load_bil_cube(bil_path, hdr_path)
    except Exception as e:
        print(f"  [WARN] Failed to load {bil_path}: {e}")
        return None

    # Apply calibration if raw counts and references are provided
    spectrum = extract_roi_spectrum(cube, radius_fraction)

    if dark_ref is not None and white_ref is not None:
        # Calibrated reflectance: R = (I - dark) / (white - dark)
        denom = white_ref - dark_ref
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        spectrum = (spectrum - dark_ref) / denom
        spectrum = np.clip(spectrum, 0.0, 2.0)

    # Integrate to 18 AS7265x channels
    channels = gaussian_channel_integration(spectrum, wavelengths)

    row = {
        "fruit_id": fruit_id,
        "treatment_family": labels["treatment_family"],
        "level": labels["level"],
        "class_3": labels["class_3"],
    }
    row.update(dict(zip(CHANNEL_COLS, channels)))

    return pd.Series(row)


# ---------------------------------------------------------------------------
# Batch extraction — walk entire Roomi root
# ---------------------------------------------------------------------------

def extract_all_roomi(
    roomi_root: str,
    dark_ref: Optional[np.ndarray] = None,
    white_ref: Optional[np.ndarray] = None,
    radius_fraction: float = 0.4,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Walk roomi_root, find all .bil/.hdr pairs, extract 18-channel vectors.
    Labels are inferred from folder path structure.

    Parameters
    ----------
    roomi_root  : path to extracted Roomi Zenodo 7818714 archive
    dark_ref    : (B,) dark reference (needed if stored as raw counts)
    white_ref   : (B,) white reference (needed if stored as raw counts)
    radius_fraction : ROI radius fraction

    Returns
    -------
    pd.DataFrame with columns: fruit_id, treatment_family, level, class_3, ch_410...ch_940
    """
    root = Path(roomi_root)
    bil_files = sorted(root.rglob("*.bil"))

    if not bil_files:
        raise FileNotFoundError(
            f"No .bil files found under {roomi_root}.\n"
            "Have you downloaded the real Roomi dataset from Zenodo 7818714?"
        )

    if verbose:
        print(f"Found {len(bil_files)} .bil files under {roomi_root}")

    rows = []
    skipped = 0

    for bil_path in bil_files:
        hdr_path = bil_path.with_suffix(".hdr")
        if not hdr_path.exists():
            if verbose:
                print(f"  [SKIP] No .hdr for {bil_path.name}")
            skipped += 1
            continue

        labels = _match_labels_from_path(bil_path)
        if labels is None:
            if verbose:
                print(f"  [SKIP] Could not determine labels for: {bil_path}")
            skipped += 1
            continue

        fruit_id = _extract_fruit_id(bil_path)
        if fruit_id < 0:
            if verbose:
                print(f"  [SKIP] Could not extract fruit ID from: {bil_path.name}")
            skipped += 1
            continue

        row = extract_apple_18ch(
            str(bil_path), str(hdr_path),
            labels, fruit_id,
            dark_ref=dark_ref,
            white_ref=white_ref,
            radius_fraction=radius_fraction,
        )
        if row is not None:
            rows.append(row)
            if verbose and len(rows) % 20 == 0:
                print(f"  Extracted {len(rows)} apples...")

    if not rows:
        raise ValueError(
            "No apple spectra were successfully extracted. "
            "Check folder structure and label mapping in hsi_to_18ch.FOLDER_LABEL_MAP."
        )

    df = pd.DataFrame(rows)[REQUIRED_CSV_COLS]

    if verbose:
        print(f"\nExtraction complete: {len(df)} rows from {len(df['fruit_id'].unique())} unique fruit IDs")
        print(f"Skipped: {skipped}")
        print(f"Class distribution:\n{df['class_3'].value_counts().sort_index().to_string()}")
        print(f"Treatment family:\n{df['treatment_family'].value_counts().to_string()}")

    return df


# ---------------------------------------------------------------------------
# SpectroFood 18-channel extraction (no labels, no treatment)
# ---------------------------------------------------------------------------

def extract_spectrofood_18ch(mat_path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Extract 18-channel vectors from SpectroFood Apple.mat (SpectroFood dataset).
    No treatment labels are assigned. For OOD stress test use only.

    Parameters
    ----------
    mat_path : path to spectrofood_apple_dry_matter/Apple.mat

    Returns
    -------
    pd.DataFrame with columns: sample_id, ch_410...ch_940
    """
    import scipy.io as sio

    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if k.startswith("A") and not k.startswith("__")]

    # Get wavelengths
    wl = mat["Wavelengths"].flatten().astype(float)

    if verbose:
        print(f"SpectroFood Apple.mat: {len(keys)} samples, {len(wl)} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm")

    rows = []
    for key in sorted(keys):
        cube = mat[key].astype(float)   # (H, W, 141)
        spectrum = cube.reshape(-1, cube.shape[-1]).mean(axis=0)  # (141,)
        channels = gaussian_channel_integration(spectrum, wl)
        row = {"sample_id": key.strip()}
        row.update(dict(zip(CHANNEL_COLS, channels)))
        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Extracted {len(df)} samples")
    return df
