"""
app.py — CFRD Portable Multispectral Pesticide Screening
Streamlit Dashboard — Pre-Hardware Software Demonstration

10-page interactive dashboard. Pages live in src/ui_pages/.
Theme / CSS lives in src/ui_theme.py.

Usage:
    cd project
    streamlit run app.py
"""

# ---------------------------------------------------------------------------
# Optional PyTorch import (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import streamlit as st
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from constants import (
    AS7265X_CENTRES_NM, AS7265X_SIGMA, CHANNEL_COLS,
    CLASS_LABELS, TREATMENT_FAMILIES, FORMULATION_NAMES,
    DISCLAIMER_BANNER, SIMULATOR_NOTE, SYNTHETIC_LABEL,
    OOD_EXPERIMENT_LABEL, COVERAGE_FORMAT,
    FORBIDDEN_TERMS, CONFIDENCE_THRESHOLDS,
    ROOMI_REAL_18CH_PATH, ROOMI_SYNTHETIC_18CH_PATH,
    SYNTHETIC_CSV_PATH, SPECTROFOOD_18CH_PATH,
    RANDOM_SEED, PCA_N_COMPONENTS,
    SIM_DARK_OFFSET_DEFAULT, SIM_READ_NOISE_DEFAULT,
    SIM_LED_SCALE_DEFAULT, SIM_GAIN_DEFAULT,
    SIM_PLACEMENT_DEFAULT, SIM_DRIFT_DEFAULT,
    GOOD_BAD_ROOT, MODELS_DIR, TABLES_DIR,
)
from sensor_simulator import simulate_scan, PARAMETER_RANGES
from ui_theme import inject_css
from ui_pages import (
    page_overview, page_hsi_demo, page_sim_scan,
    page_ml_evidence, page_binary, page_hardware,
    page_limits, page_roadmap, page_teacher, page_virtual,
    page_live_guided_scan,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CFRD — AS7265x Spectral Screening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------------------------------------------------------------------------
# Shared constants dict passed to every page
# ---------------------------------------------------------------------------
CONSTANTS = {
    "AS7265X_CENTRES_NM":  AS7265X_CENTRES_NM,
    "AS7265X_SIGMA":       AS7265X_SIGMA,
    "CHANNEL_COLS":        CHANNEL_COLS,
    "CLASS_LABELS":        CLASS_LABELS,
    "DISCLAIMER_BANNER":   DISCLAIMER_BANNER,
    "SIMULATOR_NOTE":      SIMULATOR_NOTE,
    "SYNTHETIC_LABEL":     SYNTHETIC_LABEL,
    "FORBIDDEN_TERMS":     FORBIDDEN_TERMS,
    "ROOMI_REAL_18CH_PATH": ROOMI_REAL_18CH_PATH,
    "ROOMI_SYNTHETIC_18CH_PATH": ROOMI_SYNTHETIC_18CH_PATH,
    "SYNTHETIC_CSV_PATH":  SYNTHETIC_CSV_PATH,
    "SPECTROFOOD_18CH_PATH": SPECTROFOOD_18CH_PATH,
    "RANDOM_SEED":         RANDOM_SEED,
    "PCA_N_COMPONENTS":    PCA_N_COMPONENTS,
    "SIM_DARK_OFFSET_DEFAULT": SIM_DARK_OFFSET_DEFAULT,
    "SIM_READ_NOISE_DEFAULT":  SIM_READ_NOISE_DEFAULT,
    "SIM_LED_SCALE_DEFAULT":   SIM_LED_SCALE_DEFAULT,
    "SIM_GAIN_DEFAULT":        SIM_GAIN_DEFAULT,
    "SIM_PLACEMENT_DEFAULT":   SIM_PLACEMENT_DEFAULT,
    "SIM_DRIFT_DEFAULT":       SIM_DRIFT_DEFAULT,
    "GOOD_BAD_ROOT":       GOOD_BAD_ROOT,
    "MODELS_DIR":          MODELS_DIR,
    "TABLES_DIR":          TABLES_DIR,
}

# ---------------------------------------------------------------------------
# Cached cross-validation (shared across pages)
# ---------------------------------------------------------------------------
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score


def _build_pipeline(X_tr, y_tr):
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_SEED)),
        ("lda", LinearDiscriminantAnalysis()),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


@st.cache_data
def run_page4_cv_cached(csv_path: str):
    """Run GroupKFold cross-validation once and cache results."""
    df     = pd.read_csv(csv_path)
    X      = df[CHANNEL_COLS].values
    y      = df["class_3"].values
    groups = df["fruit_id"].values
    gkf    = GroupKFold(n_splits=5)
    fold_results = []
    oof_y, oof_pred = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe = _build_pipeline(X[tr], y[tr])
        pred = pipe.predict(X[te])
        acc  = accuracy_score(y[te], pred)
        fold_results.append({
            "Fold": fold + 1,
            "Train Fruits": sorted([int(x) for x in np.unique(groups[tr])]),
            "Test Fruits":  sorted([int(x) for x in np.unique(groups[te])]),
            "Accuracy": f"{acc*100:.1f}%",
        })
        oof_y.extend([int(x) for x in y[te]])
        oof_pred.extend([int(x) for x in pred])
    return fold_results, oof_y, oof_pred


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "1 · Project Overview",
    "2 · HSI → 18-Channel Demo",
    "3 · Simulated Hardware Scan",
    "4 · Leakage-Free ML Evidence",
    "5 · Narrow Task & Rejection",
    "6 · Hardware Arrival Plan",
    "7 · Honest Limitations",
    "8 · Future Product Roadmap",
    "9 · Teacher Explanation Mode",
    "10 · 🔬 Virtual Apple Scan Demonstrator — Simulation Only",
    "11 · 🍏 5-Step Guided Scan Protocol (Live / Calibrated)",
]

with st.sidebar:
    st.markdown("## 🔬 CFRD Dashboard")
    st.markdown("*Portable AS7265x Spectral Screening*")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, index=0)
    st.markdown("---")
    st.markdown("### 📊 Validated Status")
    st.markdown(
        "<div style='background:#1e293b; padding:10px; border-radius:6px; font-size:0.80rem; border-left:3px solid #38bdf8; margin-bottom:12px; color:#e2e8f0;'>"
        "<b>Physical AS7265x:</b> 0 scans<br>"
        "<b>LC-MS/MS Chemistry:</b> 0 samples<br>"
        "<b>Real Roomi (13 fruits):</b> 15.0% Acc<br>"
        "<b>Synthetic Sim (20 fruits):</b> 69.2% Acc"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#546e7a;'>Pre-hardware software prototype<br>"
        "Zenodo 7818714 + 10301753<br>"
        "Research demonstration only</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Route to page modules
# ---------------------------------------------------------------------------
if page == PAGES[0]:
    page_overview(CONSTANTS)

elif page == PAGES[1]:
    page_hsi_demo(CONSTANTS)

elif page == PAGES[2]:
    page_sim_scan(CONSTANTS, simulate_scan, PARAMETER_RANGES)

elif page == PAGES[3]:
    page_ml_evidence(CONSTANTS, run_page4_cv_cached)

elif page == PAGES[4]:
    page_binary(CONSTANTS)

elif page == PAGES[5]:
    page_hardware(CONSTANTS)

elif page == PAGES[6]:
    page_limits(CONSTANTS)

elif page == PAGES[7]:
    page_roadmap(CONSTANTS)

elif page == PAGES[8]:
    page_teacher(CONSTANTS)

elif page == PAGES[9]:
    page_virtual(CONSTANTS, TORCH_AVAILABLE)

elif page == PAGES[10]:
    page_live_guided_scan(CONSTANTS)
