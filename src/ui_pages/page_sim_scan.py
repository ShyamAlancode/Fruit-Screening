"""page_sim_scan.py — Page 3: Simulated Hardware Scan"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from ui_theme import show_disclaimer, info_box


def render(constants: dict, simulate_scan_fn, parameter_ranges: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    centres      = constants["AS7265X_CENTRES_NM"]
    seed         = constants["RANDOM_SEED"]
    dark_def     = constants["SIM_DARK_OFFSET_DEFAULT"]
    noise_def    = constants["SIM_READ_NOISE_DEFAULT"]
    led_def      = constants["SIM_LED_SCALE_DEFAULT"]
    gain_def     = constants["SIM_GAIN_DEFAULT"]
    place_def    = constants["SIM_PLACEMENT_DEFAULT"]
    drift_def    = constants["SIM_DRIFT_DEFAULT"]
    sim_note     = constants["SIMULATOR_NOTE"]

    st.title("⚡ Simulated Hardware Scan")
    st.subheader("AS7265x Dark/White/Raw/Calibrated Pipeline")
    info_box(f"ℹ️ {sim_note}")

    col_sliders, col_output = st.columns([1, 2])

    with col_sliders:
        st.markdown("### Sensor Parameters")
        gain          = st.slider("Gain",              0.5, 2.0, gain_def,   0.05)
        led_scale     = st.slider("LED Brightness",    0.7, 1.3, led_def,    0.05)
        placement     = st.slider("Placement Scale",   0.7, 1.3, place_def,  0.05)
        noise_std     = st.slider("Read Noise Std",    0.0, 0.10, noise_def, 0.005)
        dark_offset   = st.slider("Dark Offset",       0.0, 0.10, dark_def,  0.005)
        led_drift     = st.slider("LED Drift",         0.0, 0.20, drift_def, 0.01)
        use_sat       = st.checkbox("Enable Saturation")
        sat_threshold = st.slider("Saturation Threshold", 0.5, 2.0, 1.2, 0.05) if use_sat else None

    rng_in  = np.random.default_rng(seed)
    X_true  = rng_in.uniform(0.15, 0.75, 18)
    rng_sim = np.random.default_rng(seed + 1)
    sim_result = simulate_scan_fn(
        X_true,
        gain=gain, led_scale=led_scale, placement_scale=placement,
        read_noise_std=noise_std, dark_offset=dark_offset, led_drift=led_drift,
        saturation_threshold=sat_threshold, rng=rng_sim,
    )

    with col_output:
        st.markdown("### Calibration Pipeline Output")
        panel_data = [
            ("Dark Reference",        sim_result["dark"],        "#78909c"),
            ("White Reference",       sim_result["white"],       "#ffb74d"),
            ("Raw Fruit Scan",        sim_result["raw"],         "#ef5350"),
            ("Calibrated Reflectance",sim_result["calibrated"],  "#66bb6a"),
        ]
        x_ticks = range(18)
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), facecolor="#0d2137")
        for ax, (title, values, colour) in zip(axes, panel_data):
            ax.set_facecolor("#0d2137")
            ax.bar(x_ticks, values, color=colour, alpha=0.7, edgecolor="white", linewidth=0.2)
            if title == "Calibrated Reflectance":
                ax.plot(x_ticks, X_true, "o--", color="#64b5f6", alpha=0.6,
                        markersize=3, linewidth=1, label="True X")
                ax.legend(facecolor="#1e3a5f", labelcolor="#e3f2fd", fontsize=7)
            ax.set_title(title, color="#e3f2fd", fontsize=9)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([str(c) for c in centres], rotation=45, fontsize=6.5, color="#90caf9")
            ax.tick_params(colors="#90caf9")
            ax.spines[:].set_color("#1e3a5f")
        fig.tight_layout(pad=2.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if sim_result["saturated"]:
            st.warning("⚠️ Saturation detected in one or more channels!")
        max_err = float(np.abs(sim_result["calibrated"] - X_true).max())
        col_a, col_b = st.columns(2)
        col_a.metric("Max Channel Recovery Error", f"{max_err:.4f}")
        col_b.metric("Saturated", "Yes" if sim_result["saturated"] else "No")
