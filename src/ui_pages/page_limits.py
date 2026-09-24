"""page_limits.py — Page 7: Honest Limitations"""
import streamlit as st
from ui_theme import show_disclaimer, warning_box


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    forbidden = constants["FORBIDDEN_TERMS"]

    st.title("⚠️ Honest Limitations")
    st.subheader("What this system can and cannot claim")

    warning_box(
        "⚠️ This section must not be removed from the dashboard for visual cleanliness. "
        "It is the scientific integrity foundation of the project."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Scientifically Valid Statements")
        st.markdown("""
        - **Synthetic Pipeline Test**: "PCA+LDA achieves 69.17% ± 11.59% on synthetic data, proving that the software pipeline can recover artificial label perturbations."
        - **Real Extraction Reality**: "On real Roomi .bil cube extractions (13 fruits / 26 scans), PCA+LDA achieves 15.00% ± 13.33% and SVM achieves 0%, which is below random chance (33.3%). No treatment-condition signal is currently established."
        - **Brix Feasibility**: "18-channel digital twin yields negative R² (R² ≈ -0.03 to -0.045), showing that guessing the mean outperforms the model and 18 channels under current sampling cannot reliably predict sugar content."
        - **Visual Surface Defects**: "MobileNetV3 achieves 95.67% test accuracy on surface defects under an image-level split (apple-level grouped validation required)."
        - **Out-of-Distribution**: "SpectroFood 100% flag demonstrates cross-instrument/cross-dataset domain shift, not general-purpose unknown apple OOD."
        - **Physical Status**: "0 physical AS7265x scans and 0 LC-MS/MS chemical ground truth samples."
        """)

    with col2:
        st.markdown("### 🚫 Forbidden Claims (Scientifically Unjustified)")
        for term in forbidden:
            st.markdown(f"- ~~{term}~~")
        st.markdown("""
        - ~~"AS7265x detects pesticide treatment with 69% accuracy"~~ *(Synthetic construct bias)*
        - ~~"97.5% high-dose pesticide detection"~~ *(Synthetic construct separation only)*
        - ~~"Achieves Brix R² = -0.04"~~ *(Negative R² is a failure to predict)*
        - ~~"Proven field-ready OOD detection"~~ *(Domain shift between instruments only)*
        """)

    st.markdown("---")
    st.markdown("### 🔍 Disentangling the Three Distinct Sub-Projects")
    st.markdown("""
    | Sub-Project | Sensor / Modality | Current Evidence | Scientific Maturity |
    |---|---|---|---|
    | **Track A: Apple Visual Quality** | RGB Camera + MobileNetV3 | 95.67% test accuracy | **Moderate** (Feasible; needs fruit-level grouped evaluation) |
    | **Track B: Apple Sweetness / Brix** | AS7265x 18-ch + Regression | R² = -0.0316, RMSE = 1.19 °Bx | **Unsuccessful** (18 channels lack narrow overtone signal; N=33 too small) |
    | **Track C: Pesticide Screening** | AS7265x 18-ch + ML | Real: 15.0% (below chance) / Synthetic: 69.2% | **Unproven** (Synthetic construct bias; 0 physical scans; 0 LC-MS/MS) |
    """)

    st.markdown("---")
    st.markdown("### 🔬 Critical Methodological Distinctions")
    st.markdown("""
    1. **Synthetic Construct Bias vs Sensor Efficacy**:
       The synthetic generator introduces label-dependent shifts: `if family == 'Nativo': shift[...] += [...]`. The classifier learns to recover the synthetic rule. This proves software pipeline integrity, not that pesticides cause detectable AS7265x spectral changes on real fruit.
    2. **Treatment Condition Metadata ≠ Chemical Residue**:
       Labeling fruit as "Control", "Low", or "High" from field spray records does not verify actual chemical residue concentration. Accredited LC-MS/MS testing is mandatory before making any chemical screening claim.
    3. **Confidence Rejection Thresholding**:
       The 86.5% balanced accuracy at 28.7% coverage selected its rejection threshold on out-of-fold validation data. A frozen threshold evaluated on independent physical test apples is required for unbiased generalization.
    """)
