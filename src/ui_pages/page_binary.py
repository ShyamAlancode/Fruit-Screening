"""page_binary.py — Page 5: Narrow Task & Rejection Demo"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from ui_theme import show_disclaimer, info_box


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    synth_label = constants["SYNTHETIC_LABEL"]
    csv_path    = constants["SYNTHETIC_CSV_PATH"]
    tables_dir  = constants["TABLES_DIR"]
    ch_cols     = constants["CHANNEL_COLS"]
    seed        = constants["RANDOM_SEED"]

    st.title("🎯 Narrow Task & Rejection Demo")
    st.subheader("Binary Tasks + Confidence-Threshold Coverage Curve")
    info_box(f"ℹ️ {synth_label}")

    task_name = st.selectbox("Binary Task", [
        "Control vs Treated (all)",
        "Control vs Nativo fungicide formulation",
        "Control vs Monostar insecticide formulation",
        "Control vs Nativo-high formulation",
        "Control vs Monostar-high formulation",
        "Low vs High",
    ])

    actual_binary_accs = {
        "Control vs Treated (all)":                  82.5,
        "Control vs Nativo fungicide formulation":    54.2,
        "Control vs Monostar insecticide formulation":87.5,
        "Control vs Nativo-high formulation":         97.5,
        "Control vs Monostar-high formulation":       65.8,
        "Low vs High":                                80.3,
    }
    actual_binary_stds = {
        "Control vs Treated (all)":                  15.4,
        "Control vs Nativo fungicide formulation":    25.6,
        "Control vs Monostar insecticide formulation":13.9,
        "Control vs Nativo-high formulation":          3.3,
        "Control vs Monostar-high formulation":       25.6,
        "Low vs High":                                13.3,
    }

    binary_json_path = os.path.join(tables_dir, "binary_results_latest.json")
    if os.path.exists(binary_json_path):
        try:
            with open(binary_json_path, "r", encoding="utf-8") as f:
                bj = json.load(f)
            if "accuracies" in bj:
                actual_binary_accs.update(bj["accuracies"])
            if "stds" in bj:
                actual_binary_stds.update(bj["stds"])
        except Exception:
            pass

    st.markdown(
        f"**Actual LDA result — 5-fold GroupKFold, 20-fruit synthetic dataset:** "
        f"{actual_binary_accs[task_name]}% ± {actual_binary_stds[task_name]}%"
    )

    if not os.path.exists(csv_path):
        st.warning("Synthetic CSV not yet generated. Run Phase 1 scripts first.")
        return

    from ood import run_rejection_sweep
    df = pd.read_csv(csv_path)
    y  = df["class_3"].values
    rng = np.random.default_rng(seed)
    n   = len(y)
    mock_proba = rng.dirichlet(alpha=[2, 1, 1], size=n)
    mock_pred  = mock_proba.argmax(axis=1)
    sweep = run_rejection_sweep(y, mock_pred, mock_proba, verbose=False)

    st.caption(
        "⚠ Illustrative simulation only — this parametric sweep illustrates the "
        "confidence-threshold rejection mechanism."
    )

    cov  = [r["coverage_pct"]                  for r in sweep]
    bal  = [r["accepted_balanced_accuracy_pct"] for r in sweep]
    thrs = [r["threshold"]                      for r in sweep]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d2137")
    ax.set_facecolor("#0d2137")
    ax.plot(cov, bal, "o-", color="#64b5f6", linewidth=2, markersize=7)
    for c, b, t in zip(cov, bal, thrs):
        ax.annotate(f"τ={t}", (c, b), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="#90caf9")
    ax.set_xlabel("Coverage % (accepted scans)", color="#90caf9")
    ax.set_ylabel("Balanced Accuracy % (accepted)", color="#90caf9")
    ax.set_title(f"Coverage vs Accuracy — Illustrative Simulation\n{synth_label}",
                 color="#e3f2fd", fontsize=9)
    ax.tick_params(colors="#90caf9")
    ax.spines[:].set_color("#1e3a5f")
    ax.grid(alpha=0.15)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Threshold Table")
    st.dataframe(pd.DataFrame(sweep), use_container_width=True, hide_index=True)
