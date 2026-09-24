"""page_hsi_demo.py — Page 2: HSI → 18-Channel Demo"""
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from ui_theme import show_disclaimer


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    centres   = constants["AS7265X_CENTRES_NM"]
    sigma     = constants["AS7265X_SIGMA"]
    csv_path  = constants["SYNTHETIC_CSV_PATH"]
    ch_cols   = constants["CHANNEL_COLS"]
    class_lbl = constants["CLASS_LABELS"]

    st.title("📡 HSI → 18-Channel Simulation")
    st.subheader("Gaussian Spectral Response Integration")

    st.markdown(r"""
    A full hyperspectral image cube has **141–281 spectral bands** sampled continuously.
    The AS7265x has **18 fixed-wavelength photodetectors**, each with a Gaussian response.

    The synthetic 18-channel value for channel *k* is:

    $$V_k = \frac{\sum_i R(\lambda_i) \cdot G_k(\lambda_i)}{\sum_i G_k(\lambda_i)}$$

    where $G_k(\lambda) = \exp\left(-\frac{(\lambda - c_k)^2}{2\sigma^2}\right)$
    and $\sigma = 20 / 2.35 = 8.50$ nm.
    """)

    # Gaussian response curve plot
    wl_range = np.linspace(380, 980, 500)
    fig, ax  = plt.subplots(figsize=(12, 4), facecolor="#0d2137")
    ax.set_facecolor("#0d2137")
    colours  = plt.cm.rainbow(np.linspace(0, 1, 18))
    for i, centre in enumerate(centres):
        G = np.exp(-0.5 * ((wl_range - centre) / sigma) ** 2)
        ax.fill_between(wl_range, G, alpha=0.25, color=colours[i])
        ax.plot(wl_range, G, color=colours[i], linewidth=1.0)
        ax.axvline(centre, color=colours[i], linewidth=0.4, alpha=0.5)
    ax.set_xlabel("Wavelength (nm)", color="#90caf9")
    ax.set_ylabel("Response",        color="#90caf9")
    ax.set_title("AS7265x 18-Channel Gaussian Response Curves (FWHM=20nm)", color="#e3f2fd")
    ax.tick_params(colors="#90caf9")
    ax.spines[:].set_color("#1e3a5f")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### Load a Synthetic Sample")

    if not os.path.exists(csv_path):
        st.warning(
            f"Synthetic CSV not found at:\n`{csv_path}`\n\n"
            "Download the Roomi dataset from Zenodo 7818714 and run "
            "`scripts/01_extract_roomi_18ch.py` first."
        )
        return

    df = pd.read_csv(csv_path)
    fruit_ids   = sorted(df["fruit_id"].unique())
    selected_fid = st.selectbox("Select fruit_id", fruit_ids)
    fdf  = df[df["fruit_id"] == selected_fid]
    row  = fdf.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**treatment_family:** {row['treatment_family']}")
        st.markdown(f"**level:** {row['level']}")
        st.markdown(f"**class:** {class_lbl.get(int(row['class_3']), '?')}")
    with col2:
        st.markdown(f"**Scans available:** {len(fdf)}")

    channels = row[ch_cols].values.astype(float)
    fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor="#0d2137")
    ax2.set_facecolor("#0d2137")
    ax2.bar(range(18), channels, color=colours, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax2.set_xticks(range(18))
    ax2.set_xticklabels([str(c) for c in centres], rotation=45, fontsize=8, color="#90caf9")
    ax2.set_ylabel("Calibrated Reflectance", color="#90caf9")
    ax2.set_title(f"Synthetic 18-Channel Spectrum — Fruit {selected_fid} ({row['level']})", color="#e3f2fd")
    ax2.tick_params(colors="#90caf9")
    ax2.spines[:].set_color("#1e3a5f")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
