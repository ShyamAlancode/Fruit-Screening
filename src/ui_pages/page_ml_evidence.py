"""page_ml_evidence.py — Page 4: Leakage-Free ML Evidence"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from ui_theme import show_disclaimer, info_box


def render(constants: dict, run_cv_cached_fn) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    class_lbl    = constants["CLASS_LABELS"]
    synth_label  = constants["SYNTHETIC_LABEL"]
    csv_path     = constants["SYNTHETIC_CSV_PATH"]

    st.title("🧪 Leakage-Free ML Evidence")
    st.subheader("GroupKFold by fruit_id — No scan-level splitting")
    info_box(f"ℹ️ {synth_label}")

    import os
    if not os.path.exists(csv_path):
        st.warning("Synthetic CSV not yet generated. Run Phase 1 scripts first.")
        return

    fold_results, oof_y, oof_pred = run_cv_cached_fn(csv_path)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    selected_fold = st.selectbox("Browse fold", [f"Fold {i+1}" for i in range(5)])
    fold_idx = int(selected_fold.split()[-1]) - 1
    row      = fold_results[fold_idx]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Train fruits:** {row['Train Fruits']}")
    with col2:
        st.markdown(f"**Test fruits:** {row['Test Fruits']}")
    st.markdown(f"**Fold accuracy:** {row['Accuracy']}")

    st.markdown("---")
    st.markdown("### Pooled Out-of-Fold Confusion Matrix")
    cm          = confusion_matrix(oof_y, oof_pred)
    class_names = [class_lbl[i] for i in sorted(class_lbl)]

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0d2137")
    ax.set_facecolor("#0d2137")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted", color="#90caf9")
    ax.set_ylabel("True",      color="#90caf9")
    ax.set_title(f"Pooled OOF Confusion Matrix  |  {synth_label}", color="#e3f2fd", fontsize=8)
    ax.tick_params(colors="#90caf9")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
