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
# Real Roomi .bil folder structure on disk (Zenodo 7818714):
# roomi_pesticide_hsi/
#   Apple_samples/Apple_samples/
#     Monostar/  Fresh/ (1_1PU.bil, 1_2PU.bil)
#                High/  (1_1HG.bil, 1_2HG.bil)
#                Low/   (1_1LW.bil, 1_2LW.bil)
#                Unknown/fresh/ (APP1_PU1.bil, APP1_PU2.bil)
#                Unknown/pesticide_unknownconc/ (APP1_PST1.bil, APP1_PST2.bil)
#     Nativo/    Fresh/ (A1_F1.bil, A1_F2.bil)
#                High/  (A1_H1.bil, A1_H2.bil)
#                Low/   (A1_L1.bil, A1_L2.bil)
#   Fungicide_Apple/Fungicide_Apple/
#     DIB_Apple/          (A1_F1.bil, A1_F2.bil)
#     DIB_AppleNativohigh/(A1_H1.bil, A1_H2.bil)
#     DIB_AppleNativolow/ (A1_L1.bil, A1_L2.bil)
#   Pesticide_Apple/Pesticide_Apple/
#     DIB_Applemono/      (MA1_F1.bil, MA1_F2.bil)
#     DIB_Applemonohigh/  (MA1_H1.bil, MA1_H2.bil)
#     DIB_Applemonolow/   (MA1_L1.bil, MA1_L2.bil)
#
# NOTE: Do NOT assume TIF naming patterns map to .bil files. The .bil files
# follow the exact naming convention above. Label parsing matches these directly.

GLOBAL_FRUIT_ID_REGISTRY: dict[str, int] = {}


def reset_fruit_id_registry():
    """Clear the fruit ID registry for deterministic runs."""
    GLOBAL_FRUIT_ID_REGISTRY.clear()


def _match_labels_from_path(filepath: Path) -> Optional[dict]:
    """
    Determine treatment label dict from the filepath.
    Supports real Roomi Zenodo 7818714 folder layouts.
    Returns None if the path is unknown or unlabelled.
    """
    parts = [p.lower() for p in filepath.parts]
    name = filepath.stem.lower()

    # Determine formulation family
    if any("nativo" in p or "fungicide" in p for p in parts):
        family = "Nativo"
    elif any("mono" in p or "pesticide" in p or "insecticide" in p for p in parts):
        family = "Monostar"
    else:
        return None

    # Determine treatment level & class index
    # Explicitly check high, low, and control (fresh, pu, _f, dib)
    if any("high" in p or "hg" in name or "_h" in name for p in parts):
        level = "high"
        class_3 = 2
    elif any("low" in p or "lw" in name or "_l" in name for p in parts):
        level = "low"
        class_3 = 1
    elif any("fresh" in p or "pu" in name or "_f" in name or "dib" in p for p in parts):
        level = "control"
        class_3 = 0
    elif any("unknown" in p for p in parts):
        # Known unlabelled / uncalibrated exploratory scans (e.g. pesticide_unknownconc)
        return None
    else:
        raise ValueError(f"Unlabelled file in known treatment tree: {filepath}")

    return {
        "treatment_family": family,
        "level": level,
        "class_3": class_3,
    }


def _extract_fruit_id(filepath: Path, labels: Optional[dict] = None) -> int:
    """
    Extract globally unique integer fruit ID based on physical apple identity.
    Combines treatment family, level, parent folder, and local apple number.
    Ensures zero cross-treatment collisions!
    """
    name = filepath.stem
    digits = re.findall(r'\d+', name)
    local_fruit = digits[0] if digits else "1"

    fam = labels["treatment_family"] if labels else "Unknown"
    lev = labels["level"] if labels else "Unknown"
    unique_key = f"{fam}_{lev}_{filepath.parent.name}_{local_fruit}"

    if unique_key not in GLOBAL_FRUIT_ID_REGISTRY:
        GLOBAL_FRUIT_ID_REGISTRY[unique_key] = len(GLOBAL_FRUIT_ID_REGISTRY) + 1

    return GLOBAL_FRUIT_ID_REGISTRY[unique_key]


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
        # Truncate at 3*sigma to match docstring and prevent extreme edge leakage
        mask = np.abs(wavelengths - centre) <= 3.0 * sigma
        if not np.any(mask):
            channels[k] = 0.0
            continue
        G = np.exp(-0.5 * ((wavelengths[mask] - centre) / sigma) ** 2)
        weight_sum = G.sum()
        if weight_sum < 1e-10:
            channels[k] = 0.0
        else:
            channels[k] = float(np.dot(spectrum[mask], G) / weight_sum)

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


def generate_roi_sanity_check_image(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    output_path: str,
    radius_fraction: float = 0.4,
) -> str:
    """
    Generate an RGB preview from hyperspectral bands with the circular ROI overlaid.
    Provides visual verification that the central ROI falls strictly on fruit skin/flesh
    and does not sample background, edges, or shadows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_idx = int(np.argmin(np.abs(wavelengths - 645.0)))
    g_idx = int(np.argmin(np.abs(wavelengths - 535.0)))
    b_idx = int(np.argmin(np.abs(wavelengths - 460.0)))

    r_img = cube[:, :, r_idx].astype(float)
    g_img = cube[:, :, g_idx].astype(float)
    b_img = cube[:, :, b_idx].astype(float)

    rgb = np.zeros((*r_img.shape, 3), dtype=float)
    for i, ch in enumerate([r_img, g_img, b_img]):
        p2, p98 = np.percentile(ch, (2, 98))
        denom = (p98 - p2) if (p98 - p2) > 1e-6 else 1.0
        rgb[:, :, i] = np.clip((ch - p2) / denom, 0.0, 1.0)

    H, W = r_img.shape
    cx, cy = W // 2, H // 2
    radius = int(radius_fraction * min(H, W))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(rgb)
    circle = plt.Circle((cx, cy), radius, color="lime", fill=False, linewidth=2, linestyle="--", label=f"ROI (r={radius}px)")
    ax.add_patch(circle)
    ax.plot(cx, cy, "r+", markersize=10, label="Center")
    ax.set_title(f"Visual ROI Sanity Check (radius_frac={radius_fraction})")
    ax.axis("off")
    ax.legend(loc="upper right", framealpha=0.8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


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

    # Strong assertions on band count and spectral coverage
    if cube.ndim != 3:
        raise ValueError(
            f"Expected 3D hyperspectral cube (lines, samples, bands), got shape {cube.shape} from {bil_path}"
        )
    if cube.shape[2] != len(wavelengths):
        raise ValueError(
            f"Strong band-count check failed: cube has {cube.shape[2]} bands, "
            f"but header defines {len(wavelengths)} wavelengths in {hdr_path}"
        )
    if cube.shape[2] < 18:
        raise ValueError(
            f"Insufficient spectral bands: {cube.shape[2]} < 18 required for AS7265x channel integration in {bil_path}"
        )
    if wavelengths.min() > 420.0 or wavelengths.max() < 930.0:
        raise ValueError(
            f"Wavelength range [{wavelengths.min():.1f}, {wavelengths.max():.1f}] nm does not span "
            f"the required AS7265x spectral range (410–940 nm) in {hdr_path}"
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
    reset_fruit_id_registry()
    root = Path(roomi_root)
    bil_files = sorted(root.rglob("*.bil"), key=lambda p: str(p.relative_to(root)).lower().replace("\\", "/"))

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
            hdr_path = bil_path.parent / (bil_path.name + ".hdr")
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

        fruit_id = _extract_fruit_id(bil_path, labels)
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
    if "Wavelengths" not in mat:
        raise ValueError(f"Wavelengths key not found in {mat_path}. Keys present: {list(mat.keys())}")
    wl = mat["Wavelengths"].flatten().astype(float)

    if verbose:
        print(f"SpectroFood Apple.mat: {len(keys)} samples, {len(wl)} bands, {wl[0]:.1f}-{wl[-1]:.1f} nm")

    rows = []
    for key in sorted(keys):
        cube = mat[key].astype(float)   # (H, W, 141)
        # Unified ROI extraction: use central circular ROI matching Roomi extraction (M6 fix)
        if cube.ndim == 3 and cube.shape[0] > 10 and cube.shape[1] > 10:
            spectrum = extract_roi_spectrum(cube, radius_fraction=0.4)
        else:
            spectrum = cube.reshape(-1, cube.shape[-1]).mean(axis=0)
        channels = gaussian_channel_integration(spectrum, wl)
        row = {"sample_id": key.strip()}
        row.update(dict(zip(CHANNEL_COLS, channels)))
        rows.append(row)

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Extracted {len(df)} samples")
    return df
