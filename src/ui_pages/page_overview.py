"""page_overview.py — Page 1: Project Overview"""
import streamlit as st
import pandas as pd
from ui_theme import show_disclaimer, warning_box, metric_card


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    st.title("🔬 Portable Multispectral Pesticide Screening")
    st.subheader("Pre-Hardware AS7265x Software Prototype")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### What This Project Is

        This dashboard demonstrates a **software simulation** of a portable
        fruit-screening device built around the **AMS AS7265x** 18-channel
        spectral sensor (£50 consumer-grade chip).

        The goal is to show, before hardware is procured:
        - That 18 fixed spectral channels can, in principle, **separate apples
          by their treatment-condition signature** (untreated / low-treatment / high-treatment)
        - That rigorous **fruit-wise cross-validation** prevents inflated results
        - That the system **admits uncertainty** rather than making false confident claims

        ### What This Project Is NOT
        """)
        warning_box(
            "⚠️ This system does <strong>not</strong> detect pesticide residues, report mg/kg "
            "concentrations, or provide MRL compliance guidance. All results are derived from "
            "synthetic channel simulation on public research datasets. Real hardware validation "
            "with AS7265x + laboratory LC-MS/MS labels is required before any product claim can be made."
        )

    with col2:
        st.markdown("### 📊 Validated Status")
        metrics = [
            ("15.0%", "Real Roomi PCA+LDA (13 fruits)"),
            ("0.0%",  "Real Roomi SVM (13 fruits)"),
            ("69.2%", "Synthetic Simulation (20 fruits)"),
            ("0",     "Physical AS7265x Scans"),
            ("0",     "LC-MS/MS Chemical Labels"),
        ]
        for val, label in metrics:
            metric_card(val, label)

    st.markdown("---")
    st.markdown("### 📋 CURRENT VALIDATED SCIENTIFIC STATUS (15 Sep 2026)")
    status_df = pd.DataFrame([
        {
            "Module Track": "Pesticide Treatment (Real Extraction)",
            "Dataset & Sample Size": "13 physical fruits / 26 scans (Roomi .bil)",
            "Architecture": "StandardScaler → PCA(5) → LDA",
            "Validated Metric": "15.00% ± 13.33% Acc (17.5% Bal Acc, 0% SVM)",
            "Scientific Status": "🔴 Below chance (33.3%). No signal established.",
        },
        {
            "Module Track": "Pesticide Treatment (Synthetic Sim)",
            "Dataset & Sample Size": "20 fruits / 240 scans (Synthetic sim)",
            "Architecture": "StandardScaler → PCA(5) → LDA",
            "Validated Metric": "69.17% ± 11.59% Acc (73.89% Bal Acc)",
            "Scientific Status": "🟡 Pipeline learnability test (Synthetic construct bias).",
        },
        {
            "Module Track": "Brix / Sweetness Prediction",
            "Dataset & Sample Size": "33 physical apples / 60 scans (Essex)",
            "Architecture": "ElasticNet / Ridge (Cultivar-aware)",
            "Validated Metric": "R² = -0.0316, RMSE = 1.19 °Bx",
            "Scientific Status": "🔴 Negative R²: predicting mean outperforms model.",
        },
        {
            "Module Track": "External Visual Quality",
            "Dataset & Sample Size": "Public apple defect dataset (train/val/test)",
            "Architecture": "MobileNetV3-Small (PyTorch)",
            "Validated Metric": "95.67% Test Accuracy",
            "Scientific Status": "🟢 High visual accuracy (Image-level split; apple grouping needed).",
        },
        {
            "Module Track": "Cross-Dataset Domain Shift (OOD)",
            "Dataset & Sample Size": "Roomi (26) vs SpectroFood (240)",
            "Architecture": "Mahalanobis Distance (LedoitWolf)",
            "Validated Metric": "100.0% Flagged OOD / Uncertain",
            "Scientific Status": "🟡 Inter-instrument domain shift (not proven general OOD).",
        },
        {
            "Module Track": "Physical Hardware Validation",
            "Dataset & Sample Size": "0 real scans / 0 LC-MS/MS samples",
            "Architecture": "AS7265x Triad + Pi 5 (Pending arrival)",
            "Validated Metric": "N/A",
            "Scientific Status": "⚪ Scaffold & driver ready; physical scans pending.",
        },
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Dataset → Module Map")
    data = {
        "Module": [
            "Treatment-condition classifier", "Binary study (narrower tasks)",
            "Domain-shift stress test",       "Camera quality",
            "Ripeness classifier",            "Brix regression",
        ],
        "Dataset now": [
            "Roomi synthetic 18-ch CSV (Zenodo 7818714)", "Roomi synthetic subset",
            "Roomi + SpectroFood (Zenodo 10301753)",       "Good/Bad apple images",
            "18-day ripeness images",                      "Sweetness Grading JPEGs",
        ],
        "Model now": [
            "PCA+LDA (GroupKFold)", "PCA+LDA + SVM",
            "Mahalanobis OOD",      "MobileNetV3-Small",
            "MobileNetV3-Small",    "Ridge / PLSR / SVR",
        ],
        "Hardware later": [
            "Real AS7265x scans",         "Real binary treatment data",
            "Real different-batch apples","Own Pi Camera images",
            "Own labelled apples",        "AS7265x + refractometer",
        ],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
