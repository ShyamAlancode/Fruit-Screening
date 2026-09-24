"""
01_extract_roomi_18ch.py

Extracts 18-channel synthetic AS7265x CSV from real Roomi .bil ENVI cubes.
Run ONLY after scripts/00_verify_roomi_bil.py prints *** GO ***.

Usage:
    cd d:\\CFRD\\project
    python scripts\\01_extract_roomi_18ch.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hsi_to_18ch import extract_all_roomi
from constants import ROOMI_HSI_ROOT, ROOMI_REAL_18CH_PATH

# Writes strictly to dedicated real extraction directory
OUTPUT_PATH = ROOMI_REAL_18CH_PATH
ROOMI_ROOT  = ROOMI_HSI_ROOT

# ---------------------------------------------------------------------------
# Optional: provide dark/white reference paths if raw counts are stored.
# Read Roomi paper DOI 10.1016/j.dib.2023.109321 Section 2 to confirm
# whether .bil values are raw counts or calibrated reflectance.
# If calibrated already → set DARK_REF_PATH and WHITE_REF_PATH to None.
# ---------------------------------------------------------------------------
DARK_REF_PATH  = None   # e.g. r"d:\CFRD\DATA\Dataset\roomi_pesticide_hsi\dark_ref.bil"
WHITE_REF_PATH = None   # e.g. r"d:\CFRD\DATA\Dataset\roomi_pesticide_hsi\white_ref.bil"

import numpy as np

def load_ref(path):
    if path is None:
        return None
    from hsi_to_18ch import load_bil_cube, extract_roi_spectrum
    cube, wl = load_bil_cube(path, path.replace('.bil', '.hdr'))
    return extract_roi_spectrum(cube)

if __name__ == "__main__":
    print("="*60)
    print("Roomi 18-Channel Synthetic CSV Extraction")
    print("="*60)

    dark_ref  = load_ref(DARK_REF_PATH)
    white_ref = load_ref(WHITE_REF_PATH)

    df = extract_all_roomi(
        roomi_root=ROOMI_ROOT,
        dark_ref=dark_ref,
        white_ref=white_ref,
        radius_fraction=0.4,
        verbose=True,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
