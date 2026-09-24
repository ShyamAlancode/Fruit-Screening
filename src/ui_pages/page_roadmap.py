"""page_roadmap.py — Page 8: Future Product Roadmap"""
import streamlit as st
from ui_theme import show_disclaimer


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    st.title("🗺️ Future Product Roadmap")

    stages = [
        ("V0 — Now (Pre-Hardware)", "#1e3a5f", [
            "✅ Software simulation on public HSI datasets",
            "✅ Leakage-free GroupKFold baseline: ~65% (synthetic)",
            "✅ 10-page Streamlit teacher demo",
            "✅ Roomi .bil verification + real 18-ch CSV extraction complete",
        ]),
        ("V1 — Hardware Prototype (On AS7265x arrival)", "#1b5e20", [
            "[ ] Dark/white calibration pipeline on real sensor",
            "[ ] UV405 crosstalk measurement and correction",
            "[ ] 20 apples × 3 classes × 3 scans = 180 real rows",
            "[ ] GroupKFold retrain on real data — report real accuracy",
            "[ ] Compare real vs synthetic benchmark",
        ]),
        ("V2 — Multi-Apple Product Demo", "#4a148c", [
            "[ ] ≥100 independent physical apples per class",
            "[ ] Automatic placement QC (OpenCV centring check)",
            "[ ] Pi Camera module integration (visual quality)",
            "[ ] Load-cell weight → calorie estimate",
            "[ ] Mobile dashboard (PWA or React Native)",
        ]),
        ("V3 — Research-Grade Claim", "#b71c1c", [
            "[ ] AS7265x + LC-MS/MS residue labels (laboratory)",
            "[ ] Validated Brix: AS7265x + refractometer labels",
            "[ ] MRL-compliant reporting (ONLY after lab validation)",
            "[ ] Peer-reviewed dataset publication",
            "[ ] ISO 17025 traceability chain for sensor calibration",
        ]),
    ]

    for title, colour, items in stages:
        with st.expander(title, expanded=(title.startswith("V0"))):
            for item in items:
                st.markdown(f"- {item}")
