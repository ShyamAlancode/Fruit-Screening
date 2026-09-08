"""
03_extract_spectrofood_18ch.py

Extract 18-channel AS7265x spectral vectors from SpectroFood Apple.mat.
These 240 apples have no treatment labels and are used exclusively for:
1. Natural apple variation control
2. Cross-dataset domain-shift stress testing (OOD test)

Usage:
    python scripts/03_extract_spectrofood_18ch.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import SPECTROFOOD_MAT, SPECTROFOOD_18CH_PATH, CHANNEL_COLS
from hsi_to_18ch import extract_spectrofood_18ch


def main():
    print("=" * 65)
    print("SpectroFood 18-Channel AS7265x Extraction")
    print(f"Source: {SPECTROFOOD_MAT}")
    print("=" * 65)

    if not os.path.exists(SPECTROFOOD_MAT):
        print(f"ERROR: SpectroFood MAT file not found: {SPECTROFOOD_MAT}")
        sys.exit(1)

    df = extract_spectrofood_18ch(SPECTROFOOD_MAT, verbose=True)

    out_dir = os.path.dirname(SPECTROFOOD_18CH_PATH)
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(SPECTROFOOD_18CH_PATH, index=False)
    print(f"\nSaved {len(df)} samples to: {SPECTROFOOD_18CH_PATH}")
    print(f"Columns: {list(df.columns)}")
    print(f"Summary stats across channels:")
    print(df[CHANNEL_COLS].describe().round(4).to_string())


if __name__ == "__main__":
    main()
