"""
constants.py — Single source of truth for all project-wide constants.

All configuration here is intentionally explicit and immutable.
Nothing in this project should use magic numbers; import from here instead.
"""

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
# Dataset Paths (relative to project root d:\CFRD)
# ---------------------------------------------------------------------------

DATASET_ROOT = r"d:\CFRD\DATA\Dataset"

ROOMI_HSI_ROOT = r"d:\CFRD\DATA\Dataset\roomi_pesticide_hsi"          # TO BE DOWNLOADED
ROOMI_TIF_ROOT = r"d:\CFRD\DATA\Dataset\Apple-Hyperspectral images"   # RGB previews only — label reference
SPECTROFOOD_MAT = r"d:\CFRD\DATA\Dataset\spectrofood_apple_dry_matter\Apple.mat"
GOOD_BAD_ROOT   = r"d:\CFRD\DATA\Dataset\hyperspectral-fruit-val\Good and bad classification of apple"
RIPENESS_ROOT   = r"d:\CFRD\DATA\Dataset\hyperspectral-fruit-val\Multispectral images of apples for ripeness, sweetness and variety grading [Data set]\Ripeness Grading"
SWEETNESS_ROOT  = r"d:\CFRD\DATA\Dataset\hyperspectral-fruit-val\Multispectral images of apples for ripeness, sweetness and variety grading [Data set]\Sweetness Grading"
VARIETY_ROOT    = r"d:\CFRD\DATA\Dataset\hyperspectral-fruit-val\Multispectral images of apples for ripeness, sweetness and variety grading [Data set]\Variety Grading"

PROJECT_ROOT = r"d:\CFRD\project"
SYNTHETIC_CSV_PATH  = r"d:\CFRD\project\data\pesticide_screening\roomi_synthetic_18ch\synthetic_as7265x_roomi_apples.csv"
SPECTROFOOD_18CH_PATH = r"d:\CFRD\project\data\natural_variation\spectrofood_18ch_sim\spectrofood_apple_as7265x_18ch.csv"

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
    "not edible",
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
    Skips disclaimer and forbidden terms explanation lines.
    """
    for line in text.splitlines():
        line_lower = line.lower()
        if "disclaimer" in line_lower or "notice" in line_lower or "forbidden" in line_lower or "prohibited" in line_lower:
            continue
        found = [t for t in FORBIDDEN_TERMS if t.lower() in line_lower]
        if found:
            raise ValueError(
                f"Forbidden term(s) found in output ({context}): {found}\n"
                f"Offending line: {line.strip()}\n"
                "Review FORBIDDEN_TERMS in constants.py."
            )
