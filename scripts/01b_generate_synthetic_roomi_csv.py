"""
01b_generate_synthetic_roomi_csv.py

Generates the synthetic 18-channel AS7265x CSV based on realistic apple reflectance
curves (derived from SpectroFood hyperspectral cubes) and published formulation
perturbations for Monostar and Nativo.

This serves as the pre-hardware development and validation dataset until the full
6.6 GB Roomi .bil dataset from Zenodo 7818714 is extracted by scripts/01_extract_roomi_18ch.py.

Usage:
    python scripts/01b_generate_synthetic_roomi_csv.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    CHANNEL_COLS,
    REQUIRED_CSV_COLS,
    ROOMI_SYNTHETIC_18CH_PATH,
    SYNTHETIC_CSV_PATH,
    SPECTROFOOD_18CH_PATH,
    RANDOM_SEED,
)

# NOTE ON SYNTHETIC CONSTRUCT BIAS:
# This script injects synthetic, class-dependent spectral perturbations into baseline spectra.
# Performance on this dataset validates the software and ML pipeline engineering, but
# does NOT serve as proof that physical AS7265x hardware can detect real pesticide residues.


def generate_synthetic_roomi_csv(
    output_path: str = ROOMI_SYNTHETIC_18CH_PATH,
    spectrofood_path: str = SPECTROFOOD_18CH_PATH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    np.random.seed(random_seed)

    # Load SpectroFood 18ch as baseline biological apple reflectance spectra
    if not os.path.exists(spectrofood_path):
        raise FileNotFoundError(
            f"SpectroFood 18ch CSV not found at {spectrofood_path}. "
            "Run scripts/03_extract_spectrofood_18ch.py first."
        )

    sf_df = pd.read_csv(spectrofood_path)
    X_sf = sf_df[CHANNEL_COLS].values
    # Calibrated reflectance scale [0.05, 0.95]
    X_sf_norm = X_sf / np.percentile(X_sf, 98)

    # Define 20 independent apples:
    # 6 control (3 Nativo, 3 Monostar)
    # 7 low-treated (4 Nativo, 3 Monostar)
    # 7 high-treated (3 Nativo, 4 Monostar)
    fruits = []
    for fid in range(1, 21):
        if fid in [1, 2, 3]:
            family, level, c3 = "Nativo", "control", 0
        elif fid in [4, 5, 6]:
            family, level, c3 = "Monostar", "control", 0
        elif fid in [7, 8, 9, 10]:
            family, level, c3 = "Nativo", "low", 1
        elif fid in [11, 12, 13]:
            family, level, c3 = "Monostar", "low", 1
        elif fid in [14, 15, 16]:
            family, level, c3 = "Nativo", "high", 2
        else:
            family, level, c3 = "Monostar", "high", 2
        fruits.append((fid, family, level, c3))

    scale = 0.35
    rows = []

    for fid, family, level, c3 in fruits:
        # Base apple spectrum selected from real apple spectra
        base_apple = X_sf_norm[(fid * 11) % len(X_sf_norm)].copy()

        # Treatment perturbation
        shift = np.zeros(18)
        if level != "control":
            mult = 1.0 if level == "low" else 2.0
            if family == "Nativo":
                # Effect on cuticle / chlorophyll bands: 610, 645, 680, 705, 730 nm
                shift[[8, 9, 10, 11, 12]] += (
                    np.array([-0.02, -0.045, -0.06, 0.045, 0.03]) * mult * scale
                )
            else:
                # Monostar: effect on 510, 535, 560 nm and 810, 860 nm
                shift[[4, 5, 6, 14, 15]] += (
                    np.array([0.02, 0.045, -0.035, -0.045, 0.03]) * mult * scale
                )

        apple_spec = np.clip(base_apple + shift, 0.05, 1.5)

        # 12 repeated scans per apple
        for s in range(12):
            scan_noise = np.random.normal(0, 0.010, size=18)
            illum = np.random.normal(1.0, 0.011)
            scan = np.clip(apple_spec * illum + scan_noise, 0.01, 1.8)

            row = {
                "fruit_id": fid,
                "treatment_family": family,
                "level": level,
                "class_3": c3,
            }
            for i, col in enumerate(CHANNEL_COLS):
                row[col] = round(float(scan[i]), 6)
            rows.append(row)

    df_synth = pd.DataFrame(rows)[REQUIRED_CSV_COLS]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_synth.to_csv(output_path, index=False)
    print(f"Generated synthetic CSV: {output_path}")
    print(f"Shape: {df_synth.shape} ({df_synth['fruit_id'].nunique()} independent fruits)")
    return df_synth


if __name__ == "__main__":
    generate_synthetic_roomi_csv()
