"""page_teacher.py — Page 9: Teacher Explanation Mode"""
import streamlit as st
from ui_theme import show_disclaimer, success_box


def render(constants: dict) -> None:
    show_disclaimer(constants["DISCLAIMER_BANNER"])
    forbidden = constants["FORBIDDEN_TERMS"]

    st.title("👨‍🏫 Teacher Explanation Mode")
    st.subheader("Click through the key concepts")

    step = st.radio("Select concept", [
        "1. Why a £50 sensor instead of a lab spectrometer?",
        "2. What is a hyperspectral cube?",
        "3. How do we simulate 18 channels from 141+ bands?",
        "4. What is GroupKFold and why does it matter?",
        "5. Why is 65% good for a pre-hardware simulation?",
        "6. What happens when hardware arrives?",
        "7. What claims are we NOT making?",
    ])

    explanations = {
        "1. Why a £50 sensor instead of a lab spectrometer?": """
**Problem:** A hyperspectral camera costs £20,000–100,000 and requires a lab.
**Goal:** A portable, on-site screening tool a farmer can use in a field or market.

The AMS AS7265x is an 18-channel multi-spectral sensor costing ~£50.
It measures light intensity at 18 fixed wavelengths (410–940nm) using 3 stacked chips.
It is not a spectrometer — it cannot resolve fine spectral features — but it may still
carry enough information to separate treatment conditions if we design the experiment correctly.

This project tests whether that is true, before spending money on hardware.
        """,
        "2. What is a hyperspectral cube?": """
A hyperspectral image is a 3D data cube: **Height × Width × Bands**.

For each spatial pixel (one point on the apple skin), there is a full reflectance
spectrum measured at 141–281 wavelength bands from 400–1000nm.

The Roomi dataset contains one cube per apple scanned. We extract a central
circular region (to avoid edge effects), average across all pixels in that region,
and get a single 141–281 point spectrum representing that apple.

The AS7265x is like compressing that 141-band spectrum into 18 values using
18 overlapping Gaussian filters — much less data, but possibly enough for classification.
        """,
        "3. How do we simulate 18 channels from 141+ bands?": """
Each AS7265x channel has a Gaussian response centred at its wavelength with FWHM=20nm.

For channel *k* centred at wavelength *c_k*:

```
V_k = Σ R(λ_i) × G_k(λ_i)  /  Σ G_k(λ_i)
where G_k(λ) = exp(-0.5 × ((λ - c_k) / σ)²)
σ = 20 / 2.35 = 8.5 nm
```

This integration is applied to the ROI-averaged HSI spectrum.
The result is 18 synthetic AS7265x values per apple.

**Important:** This is a mathematical approximation, not a measurement.
It does not account for LED spectral profile, optical path differences,
dark current, or sample-to-sensor coupling variations.
        """,
        "4. What is GroupKFold and why does it matter?": """
The Roomi dataset has **multiple scan images per apple** (different poses, lighting).

A naive train/test split by row would allow the model to see scan #1 of Apple 7
in training, and then be tested on scan #2 of Apple 7. Because both scans are of
the same physical apple, the model learns to recognise that apple, not the treatment —
and reports artificially high accuracy.

**GroupKFold** by fruit_id ensures every fold's test set contains only apples
that the model has never seen in any form during training. This gives a much more
honest estimate of how the model would behave on a completely new apple.

This is why our 65% benchmark is harder than a naively computed 80%+ result.
        """,
        "5. Why is 65% good for a pre-hardware simulation?": """
For context:
- Random chance on 3 classes = 33%
- An overfit model (leaky split) might report 80-90%, but that's meaningless
- Our 65% uses strict fruit-wise GroupKFold across 20 hand-authored synthetic fruits

Getting above chance on a leakage-free split means:
- The mathematical simulation pipeline functions correctly
- The Gaussian channel integration does not destroy all discriminating structure
- It is consistent with (but does not confirm) the AS7265x having useful discrimination on real fruit

But 65% on synthetic data is a proof-of-concept, not a product claim.
When real hardware data arrives, we will retrain and report the true device accuracy.
        """,
        "6. What happens when hardware arrives?": """
**Step 1:** Dark + white calibration (see Page 6 for full protocol)

**Step 2:** Scan own apples:
- At least 20 apples per treatment class
- 3 repositioned scans per apple
- Assign fruit_id before scanning, never change it

**Step 3:** Retrain pipeline from scratch on real data:
- StandardScaler → PCA → LDA (same pipeline)
- GroupKFold by real fruit_id
- Report real hardware accuracy separately

**Step 4:** Compare:
- Does real hardware accuracy match, exceed, or fall below 65% synthetic?
- Does the binary control vs high task still achieve ~89%?

**Under no circumstances** should real AS7265x rows be mixed with Roomi synthetic rows in one training set.
        """,
        "7. What claims are we NOT making?": (
            "The following claims **must not be made** until real hardware data is collected "
            "and validated against laboratory LC-MS/MS residue measurements:\n\n"
            + "\n".join(f'- ❌ "{t}"' for t in forbidden)
            + "\n\nThese are not formatting choices — they are scientific integrity requirements."
        ),
    }

    st.markdown(explanations[step])
    success_box(
        "✅ This explanation can be shown directly to teachers, project supervisors, or evaluators."
    )
