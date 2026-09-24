"""
constants.py — Single source of truth for all project-wide constants.

All configuration here is intentionally explicit and immutable.
Nothing in this project should use magic numbers; import from here instead.

AUDIT STATUS (September 24, 2026):
  Third-party AI audit identified and corrected 7 critical issues.
  See project/AUDIT_CORRECTIONS.md for full details.
  Key corrections applied here:
    - Specim IQ has 204 bands (not 300). Roomi cubes are a separate 300-band instrument.
    - DATASET_IDENTITY_WARNING added (Issue 5 — fruit identity confounding).
    - AUDIT_HEADLINE replaces all prior performance claims as the authoritative summary.
"""

import os

# ---------------------------------------------------------------------------
# AS7265x Sensor Channel Specification
# ---------------------------------------------------------------------------


AS7265X_CENTRES_NM: list[int] = [
    410, 435, 460, 485, 510, 535,   # AS72653 — UV/Violet/Blue (channels A-F)
    560, 585, 610, 645, 680, 705,   # AS72652 — Green/Yellow/Red (channels G-H-R-S-T-U)
    730, 760, 810, 860, 900, 940,   # AS72651 — NIR Master (channels V-W-I-J-K-L)
]

AS7265X_FWHM_NM: float = 20.0      # Full-Width Half-Maximum per channel (nm)
AS7265X_SIGMA: float = AS7265X_FWHM_NM / 2.35482   # Gaussian sigma (nm)
N_CHANNELS: int = len(AS7265X_CENTRES_NM)           # Must always be 18

# Canonical column names for the 18-channel synthetic CSV
CHANNEL_COLS: list[str] = [f"ch_{wl}" for wl in AS7265X_CENTRES_NM]

# Required metadata columns alongside spectral data
REQUIRED_META_COLS: list[str] = ["fruit_id", "treatment_family", "level", "class_3"]

# Full required column set for synthetic CSV
REQUIRED_CSV_COLS: list[str] = REQUIRED_META_COLS + CHANNEL_COLS

# ---------------------------------------------------------------------------
# Classification Labels
# ---------------------------------------------------------------------------

CLASS_LABELS: dict[int, str] = {
    0: "Control-like",
    1: "Low-treatment-like",
    2: "High-treatment-like",
}

LEVEL_MAP: dict[str, int] = {
    "control": 0,
    "low": 1,
    "high": 2,
}

TREATMENT_FAMILIES: list[str] = ["Monostar", "Nativo"]

# Correct formulation names (use these everywhere — never just "pesticide")
FORMULATION_NAMES: dict[str, str] = {
    "Monostar": "Monostar insecticide formulation",
    "Nativo": "Nativo fungicide formulation (Tebuconazole + Trifloxystrobin)",
}

# ---------------------------------------------------------------------------
# Dataset Paths (relative to project root)
# ---------------------------------------------------------------------------

# DATASET_ROOT lives outside the repo — use env var override for portability
DATASET_ROOT = os.environ.get(
    "CFRD_DATASET_ROOT",
    r"d:\CFRD\DATA\Dataset"   # default for the original development machine
)

ROOMI_HSI_ROOT = os.path.join(DATASET_ROOT, "roomi_pesticide_hsi")
ROOMI_TIF_ROOT = os.path.join(DATASET_ROOT, "Apple-Hyperspectral images")
SPECTROFOOD_MAT = os.path.join(DATASET_ROOT, "spectrofood_apple_dry_matter", "Apple.mat")
GOOD_BAD_ROOT   = os.path.join(DATASET_ROOT, "hyperspectral-fruit-val",
                                "Good and bad classification of apple")
RIPENESS_ROOT   = os.path.join(DATASET_ROOT, "hyperspectral-fruit-val",
                                "Multispectral images of apples for ripeness, sweetness and variety grading [Data set]",
                                "Ripeness Grading")
SWEETNESS_ROOT  = os.path.join(DATASET_ROOT, "hyperspectral-fruit-val",
                                "Multispectral images of apples for ripeness, sweetness and variety grading [Data set]",
                                "Sweetness Grading")
VARIETY_ROOT    = os.path.join(DATASET_ROOT, "hyperspectral-fruit-val",
                                "Multispectral images of apples for ripeness, sweetness and variety grading [Data set]",
                                "Variety Grading")

# Project-internal paths — computed relative to this file so they work on any machine
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # src/ -> project/
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR    = os.path.join(PROJECT_ROOT, "reports")
TABLES_DIR     = os.path.join(REPORTS_DIR, "tables")
FIGURES_DIR    = os.path.join(REPORTS_DIR, "figures")

ROOMI_REAL_18CH_PATH      = os.path.join(DATA_DIR, "pesticide_screening",
                                         "roomi_real_18ch",
                                         "roomi_real_18ch.csv")
ROOMI_SYNTHETIC_18CH_PATH = os.path.join(DATA_DIR, "pesticide_screening",
                                         "roomi_synthetic_simulation",
                                         "roomi_synthetic_18ch.csv")
# Legacy alias pointing to verified synthetic simulation dataset
SYNTHETIC_CSV_PATH        = ROOMI_SYNTHETIC_18CH_PATH

SPECTROFOOD_18CH_PATH     = os.path.join(DATA_DIR, "natural_variation",
                                         "spectrofood_18ch_sim",
                                         "spectrofood_apple_as7265x_18ch.csv")

# AMS AS7265x hardware specification LED currents (datasheet: 12.5, 25, 50, 100 mA)
AS7265X_LED_CURRENTS_MA: list[float] = [12.5, 25.0, 50.0, 100.0]

# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42
PCA_N_COMPONENTS: int = 5
GROUPKFOLD_N_SPLITS: int = 5

# Sensor simulator defaults
SIM_DARK_OFFSET_DEFAULT: float = 0.02
SIM_READ_NOISE_DEFAULT: float = 0.01
SIM_LED_SCALE_DEFAULT: float = 1.0
SIM_GAIN_DEFAULT: float = 1.0
SIM_PLACEMENT_DEFAULT: float = 1.0
SIM_DRIFT_DEFAULT: float = 0.0
SIM_EPSILON: float = 1e-8           # Prevents division by zero in calibration

# OOD confidence thresholds to sweep
CONFIDENCE_THRESHOLDS: list[float] = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# ---------------------------------------------------------------------------
# Mandatory Display Strings (Never shorten or remove these)
# ---------------------------------------------------------------------------

DISCLAIMER_BANNER: str = (
    "Synthetic AS7265x Simulation  |  Research Demonstration Only — "
    "Not a Physical Hardware Measurement  |  Not a Pesticide Residue or MRL Decision"
)

SIMULATOR_NOTE: str = (
    "This is a mathematical sensor simulation. "
    "It does not represent measured AS7265x hardware performance."
)

SYNTHETIC_LABEL: str = "Synthetic treatment-condition classification — not a physical hardware measurement."

OOD_EXPERIMENT_LABEL: str = "Cross-dataset synthetic domain-shift stress test"

COVERAGE_FORMAT: str = "Synthetic accepted-scan accuracy: {acc:.1f}% at {cov:.1f}% coverage."

BRIX_LIMITATION: str = (
    "Software Brix demonstration from public images only. "
    "Physical AS7265x + refractometer labels required for validated hardware claims."
)

CAMERA_OUTPUT_LABEL: str = "External visual quality"

# ---------------------------------------------------------------------------
# Instrument Specification Corrections (Audit Issue — corrected Sep 24 2026)
# ---------------------------------------------------------------------------

# Specim IQ: 204 calibrated bands, 388-1026 nm (Essex apple quality dataset).
# Roomi et al. cubes: 300 bands, 388.85-1026.04 nm (DIFFERENT instrument).
# These two datasets must NOT be conflated. Earlier reports incorrectly
# stated "Specim IQ 300 bands" — this is wrong. See AUDIT_CORRECTIONS.md §7.
SPECIM_IQ_N_BANDS: int = 204         # Correct Specim IQ band count
ROOMI_CUBE_N_BANDS: int = 300        # Roomi et al. cube band count (different instrument)
SPECIM_IQ_RANGE_NM: tuple = (388, 1026)
ROOMI_CUBE_RANGE_NM: tuple = (388, 1026)

# ---------------------------------------------------------------------------
# Dataset Identity Warning (Audit Issue 5 — fruit confounding)
# ---------------------------------------------------------------------------

DATASET_IDENTITY_WARNING: str = (
    "WARNING (Audit Issue 5): Fruit identity and treatment condition are potentially "
    "confounded in the Roomi dataset. 'A1' and 'F_A1' share filenames (A1_F1...) and "
    "may be the same physical fruit rescanned. Unique MD5 hashes prove different byte "
    "streams, not different fruits. Shutter-time differences between clean and treated "
    "scans in cohorts 'apple 1' (18.8 vs 29.3 ms) and 'A1' (9.7 vs 11.5 ms) may "
    "contribute to apparent spectral separability independently of pesticide chemistry. "
    "No further pesticide ML modelling is recommended until the Roomi dataset authors "
    "clarify physical fruit identity and confirm calibrated-count usage."
)

# ---------------------------------------------------------------------------
# Authoritative Audit Headline (replaces all prior headline claims)
# ---------------------------------------------------------------------------

AUDIT_HEADLINE: str = (
    "CFRD is a well-instrumented, honestly audited pre-hardware research prototype. "
    "No reliable cross-apple pesticide signal has been established on the available "
    "28-cube dataset. Physical validation with the AMS AS7265x sensor and LC-MS/MS "
    "chemical ground truth is required before any screening performance claim can be made."
)

# ---------------------------------------------------------------------------
# Forbidden Terms (must not appear in any report or dashboard output)
# ---------------------------------------------------------------------------

FORBIDDEN_TERMS: list[str] = [
    "detects pesticide",
    "pesticide concentration",
    "pesticide percentage",
    "residue mg/kg",
    "MRL",
    "safe to eat",
    "unsafe",
    "edible",
    "food safety certified",
    "pesticide detected",
    "identifies pesticide",
    "pesticide level",
    "below MRL",
    "above MRL",
]


def assert_no_forbidden_terms(text: str, context: str = "") -> None:
    """
    Raise ValueError if any forbidden term appears in text.
    Call this before writing any report file or rendering any dashboard section.
    Skips explicit disclaimer lines and forbidden terms definitions.
    """
    for line in text.splitlines():
        line_clean = line.strip()
        line_lower = line_clean.lower()

        # Exempt exact disclaimer banner
        if line_clean == DISCLAIMER_BANNER.strip():
            continue

        # Skip explicit disclaimer/notice headers and forbidden-terms explanation lines
        if (
            line_lower.startswith(("disclaimer:", "> disclaimer:", "**disclaimer:**", "> **disclaimer:**",
                                   "notice:", "> notice:", "**notice:**", "> **notice:**",
                                   "# forbidden", "forbidden terms", "forbidden_terms", "- forbidden"))
            or "assert_no_forbidden_terms" in line_lower
            or "forbidden_terms in constants" in line_lower
        ):
            continue

        found = [t for t in FORBIDDEN_TERMS if t.lower() in line_lower]
        if found:
            raise ValueError(
                f"Forbidden term(s) found in output ({context}): {found}\n"
                f"Offending line: {line_clean}\n"
                "Review FORBIDDEN_TERMS in constants.py."
            )
