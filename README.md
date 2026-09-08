# 🍎 CFRD: Portable Multispectral Pesticide Screening & Apple Quality Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/pytest-27%2F27%20passed%20(100%25)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](https://streamlit.io/)

> **Academic & Research Notice:**  
> **Synthetic AS7265x Simulation | Research Demonstration Only — Not a Physical Hardware Measurement | Not a Pesticide Residue or MRL Decision**  
> *Pre-hardware software research demonstration. All results are synthetic until calibrated physical AMS AS7265x sensor hardware and accredited LC-MS/MS laboratory residue ground truth are collected.*

---

## 📌 Executive Summary

This repository implements the end-to-end software pipeline for **portable multispectral pesticide screening** and **apple quality evaluation**. The system models the **AMS AS7265x 18-channel optical sensor** (spanning 410 nm to 940 nm) and couples it with modern computer vision and machine learning:

1. **Chemical Treatment-Condition Screening**: GroupKFold cross-validated PCA + LDA / SVM classification separating control from commercial pesticide treatment conditions.
2. **Selective Rejection & Out-Of-Distribution (OOD) Guard**: Mahalanobis distance scoring with regularized covariance to reject abnormal, scarred, or heavily damaged apples.
3. **External Visual Quality (MobileNetV3)**: Deep transfer learning classifier diagnosing surface defects, scabs, and blemishes with **95.67% test accuracy**.
4. **Ripeness & Cultivar Grading**: Color-moment and HSV histogram ML models tracking 18-day longitudinal maturity.
5. **Interactive 10-Page Streamlit Dashboard**: Full GUI with live apple testing, virtual hardware noise sliders, calibration protocols, and teacher explanation mode.

---

## 🔬 Benchmark Results Summary

All models were evaluated under strict **leakage-free protocols** (physical `fruit_id` grouping in `GroupKFold`, with scaler and PCA fitting strictly confined to training folds).

| Module | Model Architecture | Metric | Result | Benchmark Context |
|---|---|---|---|---|
| **Treatment Screening** | PCA (5 components) + LDA | 3-Class Accuracy | **69.17% ± 11.59%** | Literature target: 60%–71% |
| **Treatment Screening** | Balanced Accuracy | 3-Class Balanced Acc | **73.89%** | Macro-F1: 69.42% |
| **High-Dose Separation** | PCA + LDA (Binary) | Control vs High Nativo | **97.50%** | Clear chemical signature separation |
| **Selective Rejection** | Confidence Rejection Sweep | Balanced Acc @ 28.75% Coverage | **86.54%** (89.86% ord. acc) | Rejects 71.25% of borderline scans (synthetic simulation only) |
| **OOD Anomaly Detection** | Mahalanobis Distance (LedoitWolf) | 240 SpectroFood Apples | **17.1% flagged OOD** | Safely detects foreign varieties & lighting |
| **Visual Surface Quality** | MobileNetV3-Small (PyTorch) | 2-Class Test Accuracy | **95.67%** | Precision: 97.24%, Recall: 94.00% |
| **Ripeness Grading** | Logistic Regression (GroupKFold) | 3-Stage Ripeness Acc | **64.07%** | Longitudinal harvest tracking |

---

## 📁 Repository Structure

```
├── app.py                     # 10-page interactive Streamlit dashboard
├── requirements.txt           # Python dependencies (torch, sklearn, streamlit, etc.)
├── pytest.ini                 # Pytest configuration
├── README.md                  # Project documentation
│
├── src/                       # Core scientific libraries
│   ├── constants.py           # 18-channel AS7265x definitions, disclaimers, schema
│   ├── sensor_simulator.py    # AS7265x optical noise, drift, and gain simulator
│   ├── calibration.py         # Dark & white reference reflectance calibration
│   ├── hsi_to_18ch.py         # Hyperspectral cube integration to 18 channels
│   └── ood.py                 # Mahalanobis distance out-of-distribution detector
│
├── scripts/                   # Training, extraction, and CLI testing scripts
│   ├── 01b_generate_synthetic_roomi_csv.py # Roomi-derived 18ch synthetic dataset
│   ├── 03_extract_spectrofood_18ch.py      # SpectroFood 240-apple 18ch extraction
│   ├── 04_train_baseline_3class.py         # 5-fold GroupKFold PCA+LDA baseline
│   ├── 05_run_binary_experiments.py        # 6 binary formulation experiments
│   ├── 06_run_coverage_experiment.py       # Confidence rejection accuracy sweep
│   ├── 07_run_ood_experiment.py            # Cross-dataset OOD Mahalanobis validation
│   ├── 08_train_camera_quality.py          # MobileNetV3 visual quality fine-tuning
│   ├── 09_train_ripeness_variety.py        # Ripeness & cultivar ML training
│   ├── 10_train_brix_regression.py         # Sweetness multispectral regression
│   ├── 11_generate_teacher_report.py       # Markdown demonstration report generator
│   └── test_single_apple.py                # CLI tool to test any uploaded apple photo
│
├── models/                    # Saved model artifacts
│   ├── camera/                # MobileNetV3 PyTorch weights (camera_quality_mobilenet.pth)
│   ├── treatment_condition/   # Scikit-learn PCA+LDA & SVM pipelines (.joblib)
│   └── quality/               # Ripeness and variety classifiers (.joblib)
│
├── data/                      # Processed 18-channel CSV datasets
│   ├── pesticide_screening/   # Roomi-derived 18-channel synthetic dataset
│   ├── natural_variation/     # SpectroFood 18-channel variation dataset
│   └── quality/               # Ripeness and sweetness feature tables
│
├── reports/                   # Academic figures and tables
│   ├── figures/               # Confusion matrices, PCA scatter, coverage curves
│   └── tables/                # Cross-validation performance summaries (Markdown)
│
└── tests/                     # Automated unit test suite (27 passing tests)
    ├── test_calibration.py
    ├── test_channel_schema.py
    ├── test_no_group_leakage.py
    └── test_sensor_simulator.py
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/ShyamAlancode/apple_pesticide.git
cd apple_pesticide
pip install -r requirements.txt
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run app.py
```

Navigate through the 10 interactive pages:
* **Page 1–3**: Sensor architecture, wavelength mapping, and dataset lineage.
* **Page 4–5**: Model benchmarks (3-class PCA+LDA, binary experiments, coverage curves).
* **Page 6**: Physical hardware calibration protocol (dark/white reference, UV405 crosstalk).
* **Page 7–9**: Honest scientific limitations, product roadmap (V0 to V3), and teacher defense guide.
* **Page 10 (`🔬 Virtual Apple Scan Demonstrator — Simulation Only`)**: Illustrative demonstration of virtual spectral response, noise/drift simulation controls, and multi-model workflow. Contains mandatory simulation disclaimers.

### 3. Test Any Apple Photo via Command Line

```bash
# Test sample apple:
python scripts/test_single_apple.py

# Test your own custom apple image:
python scripts/test_single_apple.py --image "path/to/apple.jpg"
```

### 4. Run Automated Test Suite

```bash
pytest tests/ -v
```
Output:
```
============================= 27 passed in 3.18s ==============================
```

---

## 🛡️ Scientific Integrity & Leakage Prevention

1. **Zero Group Leakage:** Physical apples (`fruit_id`) are never split across train and test folds. All cross-validation uses `GroupKFold(n_splits=5, groups=fruit_id)`.
2. **Zero Forbidden Claims:** In compliance with spectroscopic research ethics, this software strictly enforces pre-hardware disclaimer boundaries. No claims of "pesticide residue detection", "parts per million (ppm)", or "food safety/edibility certification" are made prior to accredited laboratory LC-MS/MS validation.
3. **Out-of-Distribution Rejection:** Deformed or blemished apples with irregular optical reflectance are detected via Mahalanobis distance and safely flagged as `UNCERTAIN / REJECT SCAN`.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
