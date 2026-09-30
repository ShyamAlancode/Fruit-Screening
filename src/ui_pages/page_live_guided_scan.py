"""
page_live_guided_scan.py
========================
Page 11: Live Guided 5-Step Scan Protocol (CFRD_REAL_IMPLEMENTATION_PLAN §6.3).

Guides the operator through the calibrated measurement protocol:
  Step 1: Calibration (PTFE White Tile Reference)
  Step 2: Dark Reference (Sensor Capped / Ambient Baseline)
  Step 3: Control Spot (Calyx / Unexposed Reference Spot on Same Fruit)
  Step 4: Target Spots (5 Perimeter Equatorial Scans)
  Step 5: Unified Result Card:
          - Sugar Content / Brix (°Bx)
          - Ripeness Grading (Spectral + Camera Multimodal)
          - Cultivar / Variety Identification
          - Agrochemical Differential Screening (3-state: Suspected / Clean / Inconclusive)
          - OOD Anomaly Detection
          - Biochemical Vegetation Indices (ARI, CI, CRI, NDVI)
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Any

from as7265x_driver import AS7265xDriver, MOCK_APPLE_REFLECTANCE, AS7265X_CENTRES_NM
from predictor import ApplePredictor
from features import compute_biochemical_indices, compute_intra_fruit_differential


def render_guided_scan_page(constants: Dict[str, Any]):
    st.markdown("## 🍏 5-Step Guided Apple Screening Protocol")
    st.markdown(
        "*Standardized field protocol designed to cancel >60% biological cuticle wax "
        "variations via intra-fruit differential reference scanning.*"
    )

    # Hardware connection mode
    col_mode, col_info = st.columns([1, 2])
    with col_mode:
        hw_mode = st.radio(
            "Acquisition Source:",
            ["🔬 Mock / Digital Twin Simulator", "🔌 Physical AS7265x (UART / I2C)"],
            index=0,
            help="Choose Mock to test the full pipeline in software, or Physical when sensor is connected.",
        )
    with col_info:
        if "Physical" in hw_mode:
            st.info("Operating in physical sensor mode via `/dev/serial0` (UART) or I2C bus 1.")
        else:
            st.success("Operating in Digital Twin Simulation mode. No physical hardware required for testing.")

    st.markdown("---")

    # Initialize session state for the 5-step workflow
    if "guided_step" not in st.session_state:
        st.session_state.guided_step = 1
    if "white_ref" not in st.session_state:
        st.session_state.white_ref = None
    if "dark_ref" not in st.session_state:
        st.session_state.dark_ref = None
    if "control_spot" not in st.session_state:
        st.session_state.control_spot = None
    if "target_spots" not in st.session_state:
        st.session_state.target_spots = []

    # Display progress
    steps = [
        "1. White Tile",
        "2. Dark Ref",
        "3. Calyx Control",
        "4. Perimeter (5 Spots)",
        "5. Unified Analysis",
    ]
    cur_step = st.session_state.guided_step

    cols_step = st.columns(5)
    for idx, (col, s_name) in enumerate(zip(cols_step, steps), start=1):
        with col:
            if idx < cur_step:
                st.markdown(f"**✅ {s_name}**")
            elif idx == cur_step:
                st.markdown(f"**👉 <u>{s_name}</u>**", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:gray;'>⏳ {s_name}</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # Step 1: White Reference
    # ---------------------------------------------------------------------------
    if cur_step == 1:
        st.markdown("### Step 1 · White Reference Calibration")
        st.markdown(
            "1. Place the **PTFE Teflon / Spectralon 40mm tile** flush against the sensor opening.<br>"
            "2. Ensure the optical chamber is closed to block ambient room lighting.<br>"
            "3. Tap **Capture White Reference** below.",
            unsafe_allow_html=True,
        )
        if st.button("⚪ Capture White Reference (100% Reflection)", type="primary"):
            try:
                driver_mode = "uart" if "Physical" in hw_mode else "mock"
                with AS7265xDriver(mode=driver_mode) as drv:
                    white_vals = drv.measure_white(n_readings=10)
                st.session_state.white_ref = white_vals
                st.session_state.guided_step = 2
                st.rerun()
            except Exception as e:
                st.error(f"Hardware error acquiring white reference: {e}")

    # ---------------------------------------------------------------------------
    # Step 2: Dark Reference
    # ---------------------------------------------------------------------------
    elif cur_step == 2:
        st.markdown("### Step 2 · Dark Current Reference")
        st.markdown(
            "1. Cover the sensor aperture completely with the **matte-black 3D printed dark cap**.<br>"
            "2. All LEDs will remain powered off to record silicon detector dark noise.<br>"
            "3. Tap **Capture Dark Reference** below.",
            unsafe_allow_html=True,
        )
        col_btn, col_back = st.columns([2, 1])
        with col_btn:
            if st.button("⚫ Capture Dark Reference (0% Reflection)", type="primary"):
                try:
                    driver_mode = "uart" if "Physical" in hw_mode else "mock"
                    with AS7265xDriver(mode=driver_mode) as drv:
                        dark_vals = drv.measure_dark(n_readings=10)
                    st.session_state.dark_ref = dark_vals
                    st.session_state.guided_step = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"Hardware error acquiring dark reference: {e}")
        with col_back:
            if st.button("⬅ Back to Step 1"):
                st.session_state.guided_step = 1
                st.rerun()

    # ---------------------------------------------------------------------------
    # Step 3: Calyx Control Spot
    # ---------------------------------------------------------------------------
    elif cur_step == 3:
        st.markdown("### Step 3 · Calyx Control Spot Reference")
        st.markdown(
            "1. Position the apple with its **calyx (bottom recess)** centered over the aperture.<br>"
            "2. The calyx receives minimal topical spray deposition and establishes the **same-fruit baseline**.<br>"
            "3. Tap **Capture Control Spot** below.",
            unsafe_allow_html=True,
        )
        col_btn, col_back = st.columns([2, 1])
        with col_btn:
            if st.button("🍎 Capture Calyx Control Spot", type="primary"):
                try:
                    driver_mode = "uart" if "Physical" in hw_mode else "mock"
                    with AS7265xDriver(mode=driver_mode) as drv:
                        drv._white_ref = st.session_state.white_ref
                        drv._dark_ref = st.session_state.dark_ref
                        cal_spec = drv.read_calibrated()
                    st.session_state.control_spot = cal_spec
                    st.session_state.guided_step = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Hardware error: {e}")
        with col_back:
            if st.button("⬅ Back to Step 2"):
                st.session_state.guided_step = 2
                st.rerun()

    # ---------------------------------------------------------------------------
    # Step 4: Equatorial Perimeter Spots
    # ---------------------------------------------------------------------------
    elif cur_step == 4:
        n_captured = len(st.session_state.target_spots)
        st.markdown(f"### Step 4 · Equatorial Perimeter Scans ({n_captured}/5 Recorded)")
        st.markdown(
            "Rotate the apple along its equator (72° increments) to capture 5 representative perimeter points."
        )

        progress_val = min(1.0, n_captured / 5.0)
        st.progress(progress_val)

        col_add, col_finish, col_reset = st.columns([2, 2, 1])
        with col_add:
            if st.button(f"📸 Record Perimeter Spot #{n_captured + 1}", disabled=(n_captured >= 5)):
                try:
                    driver_mode = "uart" if "Physical" in hw_mode else "mock"
                    with AS7265xDriver(mode=driver_mode) as drv:
                        drv._white_ref = st.session_state.white_ref
                        drv._dark_ref = st.session_state.dark_ref
                        spot_spec = drv.read_calibrated()
                    st.session_state.target_spots.append(spot_spec)
                    if len(st.session_state.target_spots) >= 5:
                        st.session_state.guided_step = 5
                    st.rerun()
                except Exception as e:
                    st.error(f"Hardware error: {e}")

        with col_finish:
            if n_captured >= 1:
                if st.button("⏩ Proceed to Analysis with Current Spots"):
                    st.session_state.guided_step = 5
                    st.rerun()

        with col_reset:
            if st.button("🔄 Clear Spots"):
                st.session_state.target_spots = []
                st.rerun()

    # ---------------------------------------------------------------------------
    # Step 5: Unified Analysis
    # ---------------------------------------------------------------------------
    elif cur_step == 5:
        st.markdown("### Step 5 · Unified Multi-Track Analysis Card")

        control_spec = st.session_state.control_spot
        target_spots = st.session_state.target_spots

        if control_spec is None or len(target_spots) == 0:
            st.warning("Incomplete measurement data. Please restart guided protocol.")
            if st.button("Start New Scan"):
                st.session_state.guided_step = 1
                st.session_state.target_spots = []
                st.rerun()
            return

        # Execute unified predictor
        predictor = ApplePredictor()
        avg_target = np.mean(target_spots, axis=0)

        pred = predictor.predict_spectrum(avg_target)
        diff_res = predictor.predict_differential(control_spec, target_spots)

        # Top summary metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Estimated Sweetness", f"{pred.brix.estimated_brix} °Bx", f"Tier: {pred.brix.sweetness_tier.split()[0]}")
        with m2:
            st.metric("Ripeness Stage", pred.ripeness.stage, f"{pred.ripeness.confidence:.1%} conf")
        with m3:
            st.metric("Cultivar Identification", pred.variety.cultivar.title(), f"{pred.variety.confidence:.1%} conf")
        with m4:
            color_stat = "🟢" if diff_res.status == "LIKELY CLEAN" else ("🟡" if diff_res.status == "INCONCLUSIVE" else "🔴")
            st.metric("Differential Screening", f"{color_stat} {diff_res.status}", f"Score: {diff_res.anomaly_score:.2f}")

        st.markdown("---")

        # Detailed breakdown
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown("#### 🔬 Intra-Fruit Differential Result")
            st.write(f"**Status:** `{diff_res.status}`")
            st.write(f"**Recommendation:** {diff_res.recommendation}")
            st.write(f"**Biological Cuticle Offset Suppressed:** `{diff_res.wax_baseline_suppression_pct:.1f}%`")
            st.write(f"**Equatorial Spots Averaged:** `{diff_res.num_spots_evaluated}`")
            st.write(f"**OOD Domain Status:** `{pred.pesticide.ood_status}`")

            st.caption(diff_res.disclaimer)

        with c_right:
            st.markdown("#### 🌿 Plant Biochemical Indices")
            if pred.biochemical_indices:
                bio_df = pd.DataFrame([
                    {"Index": k, "Value": v} for k, v in pred.biochemical_indices.items()
                ])
                st.dataframe(bio_df, use_container_width=True, hide_index=True)

        # Plot Reflectance Curves
        st.markdown("#### 📈 Spectral Curves (Control vs. Equatorial Target)")
        chart_df = pd.DataFrame({
            "Wavelength (nm)": AS7265X_CENTRES_NM,
            "Calyx Control Spot": control_spec,
            "Target Equatorial Mean": avg_target,
            "ΔR (Difference)": avg_target - control_spec,
        }).set_index("Wavelength (nm)")
        st.line_chart(chart_df)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Start New Apple Scan", type="primary"):
            st.session_state.guided_step = 1
            st.session_state.target_spots = []
            st.rerun()
