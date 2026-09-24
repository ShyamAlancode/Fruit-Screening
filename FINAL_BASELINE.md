# CFRD Pesticide Screening — FINAL COMPUTATIONAL BASELINE

> **Status:** FROZEN (Post-Audit)
> **Date:** September 24, 2026 (updated post third-party audit)
> **Protocol:** Evaluated on Roomi et al. Apple Hyperspectral Dataset (28 unique cubes, 5 distinct physical cohorts).
> **Audit:** Third-party AI audit accepted September 24, 2026. See [`AUDIT_CORRECTIONS.md`](AUDIT_CORRECTIONS.md).

> [!CAUTION]
> The pre-registered held-out test (P_MA1) failed. Every clean ROI was misclassified.
> This result was previously absent from this document. It is now **the primary result**.

---

## 1. Frozen Benchmark Summary (Post-Audit)

| Experiment | Metric | Value | Scientific Status & Caveats |
|:---|:---|:---:|:---|
| **Pre-registered held-out (P_MA1)** | Macro-F1 | **35–42%** | ❌ **FAILED.** Every clean ROI misclassified. Confusion matrix [[0,50],[20,80]]. This was the pre-registered frozen test — now the primary result. P_MA1 subsequently merged back into LOAO training, so no pre-registered test set remains. |
| **Random ROI CV, Full HSI (300 bands)** | Macro-F1 (Per-ROI) | **82.65%** | ⚠️ **Exploratory / Leakage-Affected.** Spot-level split across 700 ROIs mixes spots from the same physical apple across train/test. Confirms intra-fruit chemical separability only. Does not measure cross-apple generalization. Top-20 VIP bands selected across full 700 ROIs (leakage). |
| **Strict LOAO, Full HSI** | Macro-F1 (Per-ROI) | **56.14%** | ⚠️ **Rigorous but insufficient.** Primary single-spot result. Demonstrates biological wax/cuticle baseline overwhelms single-spot absolute thresholds. |
| **Strict LOAO, Full HSI** | Scan Accuracy | **75.0%** | ⚠️ **Maximum of 75 configurations.** Only 2 of 75 reach this value. Median = 57.1%. AUROC 0.59, p ≈ 0.19 (not significant). Multi-spot table is non-monotonic. |
| **Strict LOAO, AS7265x (18 channels)** | Scan Accuracy | **75.0%** | ⚠️ **Same caveat as above.** Simulated sensor. No physical hardware data. |
| **RGB Baseline** | Macro-F1 | **63.44%** | ❌ **Invalid comparison.** Uses leaky `StratifiedKFold`. Spectral LOAO (56%) is below this. "Spectral outperforms RGB" retracted. |
| **Differential Reference (intra-fruit)** | Pairwise Accuracy | **86.36%** | ⚠️ **Preliminary/Exploratory.** Intra-cohort only. LOAO differential reached 60.71% scan accuracy. 2 of 5 cohorts show expected ΔP direction. |

### Mandatory Specificity/AUROC Disclosure

| Metric | Value |
|---|---|
| Treated sensitivity | 100% (18/18) — trivially achieved by predicting "treated" for all |
| Clean specificity | **30% (3/10)** |
| Scan-level AUROC | **0.59** |
| Permutation p-value | **≈0.19** |
| 95% Wilson CI (accuracy) | **[57%, 87%]** |
| 95% Wilson CI (specificity) | **[11%, 60%]** |

---

## 2. Locked Provenance Audit (28 Unique Cubes)

Cryptographic and dimensional verification confirms that all 28 `.bil` files represent independent optical acquisitions with zero duplicate byte streams:

```
Physical Apple Cohorts:
  ├── Apple 1     : Monostar Insecticide Batch A (6 scans, 550x900x300, 283 MB)
  ├── Apple APP1  : Unknown Pesticide Trial (4 scans, 450x900x300, 232 MB)
  ├── Apple A1    : Nativo Fungicide Batch A (6 scans, 450x900x300, 232 MB)
  ├── Apple F_A1  : Nativo Fungicide Batch B / Wide (6 scans, 600x900x300, 309 MB)
  └── Apple P_MA1 : Monostar Insecticide Batch B / Wide (6 scans, 600x900x300, 309 MB)
```

- **Total Cubes:** 28
- **Unique MD5 Hashes:** 28 (100% verified unique byte streams)
- **Treated Scans:** 18 | **Clean / Fresh Scans:** 10

> [!WARNING]
> **Fruit Identity Warning (Audit Issue 5):** Unique MD5 hashes prove different byte streams,
> NOT different physical fruits. `A1` and `F_A1` share filenames (`A1_F1`…) and may be the
> same fruit rescanned under different acquisition settings. Shutter times differ between
> clean and treated scans in two cohorts. This means LOAO may leak whenever A1 and F_A1
> are split across train/test. No further modelling until fruit identity is confirmed with
> the Roomi dataset authors.

---

## 3. Mathematical & Empirical Principles (Valid)

1. **The Biological Baseline Dilemma:**
   Natural cuticle wax and background pigments shift apple reflectance by >60% between fruits, whereas surface pesticide residue contributes a <2% reflectance delta. This is a measured physical fact, not a modelling artefact.

2. **The Derivative Solution:**
   1st-derivative differences ($\Delta ch_i = ch_{i+1} - ch_i$) cancel slowly varying spectral baselines. This is standard chemometric SG-derivative preprocessing.

3. **Multi-Spot Aggregation — Corrected Interpretation:**
   Majority-vote across multiple spots reduces within-apple **random measurement noise**. It does **NOT** address the per-apple systematic wax-baseline offset (which is identical across all spots on the same apple). The multi-spot table is non-monotonic (1 spot → 67.9%, 3 spots → 64.3%), consistent with noise-floor differences of 1–2 scans.

4. **The Comparative Reference Principle (Untested Cross-Apple):**
   Intra-fruit differential scanning (Target − Control) provides the physical mechanism to eliminate biological variety shifts. This has not been validated in a cross-apple LOAO experiment.

---

## 4. Non-Pesticide Module Status (For Reference)

| Module | Architecture | Rigorous Result | Status |
|---|---|---|---|
| Brix Regression | Cultivar-Aware ElasticNet | R² = −0.03, RMSE 1.19 °Bx | 🔴 Negative R² |
| Ripeness Grading | GroupKFold RandomForest | 64.07% (vs 33.3% chance) | 🟢 Effective colour baseline |
| Cultivar Classification | GroupKFold RandomForest | 47.54% (vs 20% chance) | 🟡 Moderate, colour features |
| Visual Defect Quality | MobileNetV3 | 95.67% (image-level split) | 🟢 Promising (needs LOAO) |

> **Note on Ripeness 92% claim:** The 92% figure refers to MobileNetV3 on RGB orchard photos.
> It is **not** a sensor result. The sensor-based result is 64%.

---

*No further modifications shall be made to these baseline numbers without a new physical experiment with real hardware and confirmed chemical ground truth.*
