# CFRD — Paper Skeleton

> **Based on:** Second Technical Review, §4 (Publication Preparation)  
> **Date:** September 24, 2026  
> **Status:** Ready for manuscript development  
> **Target journals:** Computers and Electronics in Agriculture, Sensors (MDPI), Food Chemistry, Chemometrics and Intelligent Laboratory Systems

---

## Recommended Title

> **"Evaluation of an 18-Channel Multispectral Sensor Twin for Cross-Apple Optical Screening of Surface Treatment Anomalies"**

Alternative system-oriented title:
> **"From 300-Band Hyperspectral Imaging to 18-Channel Edge Multispectral Screening: A Leakage-Free Cross-Apple Study"**

---

## Abstract (Draft)

Non-destructive optical screening of agrochemical surface treatments on apple peel requires discriminating a < 2% reflectance perturbation from inter-apple biological baseline variation exceeding 60%. We evaluate whether a compact 18-channel multispectral sensor architecture, simulated from research-grade 300-band hyperspectral images using Gaussian bandpass integration, can preserve useful screening information for a low-cost edge device. Using the published Roomi et al. apple hyperspectral dataset (28 cubes, five physical cohorts, Monostar insecticide and Nativo fungicide treatments), we first demonstrate that random spatial cross-validation inflates Macro-F1 from 56% to 83% by permitting intra-fruit leakage. A strict Leave-One-Apple-Out (LOAO) protocol reduces per-region performance to 56% Macro-F1. Spatial aggregation across 25 scan positions recovers 75% scan-level accuracy for both the full 300-band derivative representation and a simulated 18-channel adjacent-difference representation. The evaluated operating point yields 100% treated-scan sensitivity with 30% clean specificity (AUROC 0.59, p ≈ 0.19). A separate held-out evaluation on one apple cohort produced Macro-F1 35–42%, confirming the screening architecture requires physical validation before deployment claims. We discuss hardware implications for the AMS AS7265x tri-spectral sensor array and present a physical validation protocol.

---

## 1. Introduction

### 1.1 Motivation
- Post-harvest fruit quality: economic scale, consumer safety.
- Existing tools: bulky HSI ($15k–$40k), destructive GC-MS, subjective RGB.
- Gap: a low-cost, non-destructive field instrument.

### 1.2 The Biological Baseline Problem
- State quantitatively: cuticle wax shifts reflectance > 60% between fruits.
- Surface pesticide residue: < 2% reflectance perturbation.
- Single-spot absolute classification on unseen varieties: inherently ill-conditioned.

### 1.3 The Research Question
> Can a compact 18-channel multispectral sensor, combined with leakage-free cross-apple validation and spatial multi-spot aggregation, approach the scan-level screening accuracy of a 300-band HSI representation?

### 1.4 Explicit Scope Boundaries
- **This paper is about screening, not detection.** No ppm-level residue quantification.
- **This paper uses simulated sensor measurements**, not physical AS7265x scans.
- **Five apple cohorts** — proof-of-concept scope, not deployment validation.

---

## 2. Related Work

### 2.1 Handheld VIS/NIR/Multispectral Sensing
- Compact spectrometers for food quality: cite relevant 3–5 papers.
- AS7265x and similar multi-channel designs: cite manufacturer + application papers.

### 2.2 Hyperspectral Apple Quality Measurement
- Brix, ripeness, cultivar: well-studied area. Cite Essex dataset paper (Tang 2025).
- Intra-fruit variability: cite 2–3 relevant papers.

### 2.3 Optical Treatment/Residue Anomaly Studies
- Roomi et al. (2023) — primary dataset reference.
- Other HSI pesticide screening studies.
- Limitations: small cohorts, intra-fruit leakage in prior work.

### 2.4 Group-Aware Validation in Biological Measurements
- GroupKFold and LOAO as methodological requirements.
- Contrast with random CV in prior food-science ML literature.

### 2.5 Edge AI Instrumentation
- Raspberry Pi + spectral sensor architectures.
- Embedded inference pipelines.

---

## 3. Materials and Methods

### 3.1 Roomi et al. Hyperspectral Dataset
- Source: Zenodo 7818714.
- 28 .bil cubes, 388.85–1026.04 nm, 300 calibrated bands.
- Treatment families: Monostar (insecticide), Nativo (fungicide, Tebuconazole + Trifloxystrobin), Unknown.
- Concentration levels: Low, High, Clean (control).

### 3.2 Cohort and Cryptographic Audit
- Five physical cohorts identified: Apple 1, APP1, A1, F_A1, P_MA1.
- 28 unique MD5 hashes verified.
- Shutter-time differences between clean and treated scans in two cohorts documented (limitation).
- A1 / F_A1 filename overlap noted — may be same physical fruit (limitation).

### 3.3 HSI Preprocessing
- Spatial ROI extraction: N spots per apple perimeter.
- Dark and white reference calibration: R_k = (S_k − D_k) / (W_k − D_k + ε).
- Savitzky-Golay 1st derivative (window=11, poly=2).
- Standard Normal Variate (SNV) normalization.

### 3.4 AS7265x Response Simulation
- Gaussian bandpass: FWHM = 20 nm at 18 center wavelengths (410–940 nm).
- Convolution formula: ch_i = ∫ R(λ) · G(λ, μ_i, σ) dλ.
- Simulation is first-order approximation — does not model stray light, electronic gain, or illumination geometry.

### 3.5 Feature Representations Evaluated
- **F1:** Full 300-band reflectance (raw).
- **F2:** 300-band + SNV.
- **F3:** 300-band SG 1st derivative + SNV. ← Best performing.
- **F4:** Simulated 18-channel raw.
- **F5:** Simulated 18-channel SNV.
- **F6:** Simulated 17 adjacent differences (Δch_i = ch_{i+1} − ch_i). ← Primary AS7265x representation.

### 3.6 Classifiers
- Logistic Regression (ElasticNet), SVM (RBF), Random Forest.
- Preprocessing: StandardScaler fitted on training folds only.

### 3.7 Random ROI Experiment (Exploratory, Leaky)
- 5-fold StratifiedKFold on 700 ROIs.
- Reported: 82.65% Macro-F1.
- Explicitly labelled as leakage-affected.

### 3.8 Leakage Audit
- Mechanism: ROIs from the same apple in both train and validation.
- Detection method: group overlap analysis.
- Consequence: ~26 percentage-point inflation.

### 3.9 Strict LOAO Evaluation
- 5 folds (one per physical apple cohort).
- All ROIs from held-out apple excluded from training.
- Scaler fitted on training data only.
- Reported: 56.14% per-ROI Macro-F1.

### 3.10 Scan-Level Spatial Aggregation
- Majority-vote across N spots per apple.
- N ∈ {1, 3, 5, 10, 25}.
- Reported: 75.0% at N=25 (non-monotonic, 1–2 scan differences at lower N).

### 3.11 Held-Out Test (Pre-Registered, Now Invalid)
- Apple P_MA1 reserved as held-out test set.
- Result: Macro-F1 35–42% (every clean ROI misclassified).
- P_MA1 subsequently merged back into LOAO training — test set no longer exists.
- **Required: a new genuinely independent external fruit cohort.**

### 3.12 Statistical Analysis
- Wilson confidence intervals for proportions.
- Permutation p-value for AUROC.
- Per-apple confusion matrices.
- Sensitivity, specificity, balanced accuracy, PPV, NPV reported for all results.

---

## 4. Results

### Table 1 — Dataset Provenance and Cohort Structure

| Cohort | Scans | Treatment | Bands | Shutter (clean/treated) |
|---|---|---|---|---|
| Apple 1 | 6 | Monostar insecticide | 300 | 18.8 / 29.3 ms ⚠️ |
| APP1 | 4 | Unknown | 300 | — |
| A1 | 6 | Nativo fungicide | 300 | 9.7 / 11.5 ms ⚠️ |
| F_A1 | 6 | Nativo fungicide (wide) | 300 | — |
| P_MA1 | 6 | Monostar insecticide (wide) | 300 | — |

⚠️ Shutter-time differences between clean and treated scans documented; may contribute to apparent separability.

### Table 2 — Random ROI Benchmark (Leaky, Exploratory)

| Representation | Classifier | Macro-F1 | Balanced Acc. | Status |
|---|---|---|---|---|
| 300-band SG deriv + SNV | ElasticNet LR | **82.65%** | 81.13% | Leakage-affected |

### Table 3 — Strict LOAO Benchmark (Per-ROI)

| Representation | Classifier | Macro-F1 | Balanced Acc. |
|---|---|---|---|
| 300-band SG deriv + SNV | ElasticNet LR | **56.14%** | 55.2% |
| 17 adj-diff + SNV (AS7265x) | ElasticNet LR | **55.01%** | 54.3% |
| ... | ... | ... | ... |

### Table 4 — Scan-Level Aggregation (LOAO)

| Spots | HSI Scan Acc. | AS7265x Scan Acc. |
|---|---|---|
| 1 | 67.86% | 67.86% |
| 3 | 64.29% | 64.29% |
| 5 | 71.43% | 71.43% |
| 10 | 71.43% | 71.43% |
| 25 | **75.00%** | **75.00%** |

**Note:** Non-monotonic; differences at low N = 1–2 scans (noise floor).

### Table 5 — Confusion Matrix and Operating Point (LOAO, 25 spots)

| | Predicted Clean | Predicted Treated |
|---|---|---|
| **Actual Clean (10)** | 3 (TN) | 7 (FP) |
| **Actual Treated (18)** | 0 (FN) | 18 (TP) |

| Metric | Value | 95% Wilson CI |
|---|---|---|
| Accuracy | 75.0% | [57%, 87%] |
| Sensitivity | 100% | [82%, 100%] |
| Specificity | 30% | [11%, 60%] |
| Balanced Accuracy | 65% | — |
| PPV | 72% | — |
| NPV | 100% | — |
| AUROC | 0.59 | — |
| Permutation p-value | ≈0.19 | Not significant |

### Table 6 — Held-Out Test (P_MA1, 150 samples)

| Representation | Macro-F1 | Balanced Acc. | Notes |
|---|---|---|---|
| 17 adj-diff + SNV | **35.62%** | 41.5% | Every clean ROI misclassified |
| 300-band SG deriv | **34.78%** | 40.0% | |
| Top-20 bands | 66.0% | — | Band selection independence not verified |

### Figure List

- **Figure 1:** Gaussian bandpass sensor twin simulation.
- **Figure 2:** Leakage mechanism — same-apple ROIs in train/test.
- **Figure 3:** Per-apple LOAO probabilities.
- **Figure 4:** Spot-count curve (non-monotonic).
- **Figure 5:** Diagnostic wavelength alignment (suggestive, not causal).
- **Figure 6:** ROC curve with permutation envelope.

---

## 5. Discussion

### 5.1 The Leakage Finding Is the Central Contribution
- 82.65% → 56.14% drop shows the field's standard random ROI CV is inadequate for cross-apple generalization claims.
- This has implications for all prior food-quality spectroscopy papers using random CV.

### 5.2 The 17-Difference Representation
- Matches 300-band performance at scan level.
- Rationale: first differences suppress broad wax baseline (low spatial-frequency component).
- **Not proof of molecular identification.** May capture pigment variation, wax variation, cultivar effects, or instrument-setting artifacts.

### 5.3 Spatial Aggregation
- Reduces random single-spot variance.
- Does NOT address per-apple systematic baseline (identical across all spots on same apple).
- Non-monotonic spot-count curve: 3-spot ≤ 1-spot. Consistent with noise floor.
- 25 spots is impractical for a handheld point sensor in commercial use.

### 5.4 The 100% Sensitivity / 30% Specificity Operating Point
- High-sensitivity screening is appropriate for a "flag for recheck" device.
- 70% false-positive rate on clean produce is commercially unacceptable without a strong recheck state.
- Majority-class baseline (predict-treated) achieves 64.3% accuracy — the 75% result adds only 10.7pp.

### 5.5 The Held-Out Test Is the Honest Calibration
- The pre-registered held-out test (Macro-F1 35–42%) shows that the five-apple LOAO scan-level result does not generalize even within the same experimental family.
- A genuinely independent external apple cohort (different acquisition, different laboratory) is required for any deployment performance claim.

### 5.6 Biological Baseline Is the Fundamental Obstacle
- The >60% cuticle wax reflectance shift is not a modelling artefact — it is a physical measurement.
- Differential reference scanning (treated − control on same fruit) is the physically motivated solution, but cross-apple validation showed limited generalization.
- Physical calibration (PTFE reference, dark correction, within-fruit reference spot) remains the most promising engineering direction.

---

## 6. Limitations

The following must be explicitly stated in the paper:

1. Five physical apple cohorts — too few for stable deployment performance estimates.
2. Public-dataset chemical exposure — not independently quantified by the authors.
3. Simulated AS7265x measurements — no physical sensor data.
4. Weak held-out result (Macro-F1 35–42%) — within same experimental family.
5. No LC-MS/MS chemical ground truth.
6. Incomplete reproducibility package — raw HSI cubes not included.
7. Possible model-selection optimism in some representations.
8. No confidence intervals in the multi-spot table (too few scan counts).
9. Non-monotonic multi-spot curve consistent with noise floor.
10. Possible fruit-identity confounding (A1 / F_A1) and shutter-time artefact in two cohorts.

---

## 7. Conclusion

> **Do not write:** "CFRD is a validated pesticide detector."

> **Write:** "CFRD demonstrates, in computational sensor-twin experiments on public apple datasets, that a compact 18-channel representation with adjacent spectral differences can match the reported scan-level accuracy of a 300-band derivative representation under a strict five-apple LOAO protocol, while revealing substantial biological baseline variation and an unfavorable specificity/sensitivity trade-off. Physical sensor performance, chemical-ground-truth correlation, and broader biological generalization remain to be established."

---

## Required Before Submission

- [ ] Package the processed ROI dataset with SHA256 hash
- [ ] Add a genuinely independent external apple cohort (not from the Roomi five)
- [ ] Freeze Brix benchmark provenance (one pipeline, one dataset version)
- [ ] Add per-apple performance table and confidence intervals
- [ ] Verify band-selection independence for the Top-20 held-out result
- [ ] Remove all "probability-delta" method claims (not implemented in supplied code)
- [ ] Separate image model results from spectral sensor results throughout
- [ ] Get LC-MS/MS confirmation on at least a subset of treated apples
