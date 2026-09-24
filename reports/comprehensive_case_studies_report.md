# 🍎 CFRD: Comprehensive Case Studies & Accuracy Benchmark Report

Generated from automated evaluation harness covering all models, case studies, and inference test scenarios.

## 1. Summary of Quantitative Benchmark Accuracies

| Case Study | Modality / Task | Model Architecture | Sample Size | Accuracy / Metric | Chance Baseline | Status |
|---|---|---|---|---|---|---|
| **CS1-A: Pesticide Synthetic** | 18-channel optical | PCA(5) + LDA | 240 scans (20 fruits) | **69.17% ± 11.59% Acc** (Bal: 73.89%) | 33.33% | 🟡 Software Learnability Test |
| **CS1-B: Pesticide Real Roomi** | 18-channel optical | PCA(5) + LDA | 26 scans (13 fruits) | **15.00% ± 13.33% Acc** (Bal: 17.50%) | 33.33% | 🔴 Below Chance (No Signal) |
| **CS1-C: Binary Formulation (all)** | 18-channel optical | PCA(5) + LDA | 26 scans (13 fruits) | **50.00% ± 10.54% Acc** | 50.00% | 🔴 At Chance Level |
| **CS2: Ripeness Grading** | Appearance (RGB+HSV) | Random Forest (60 trees) | 1,077 scans (20 fruits) | **64.07% ± 2.32% Acc** (Bal: 64.05%) | 33.33% | 🟢 Effective Baseline |
| **CS3: Variety Identification** | Appearance (RGB+HSV) | Random Forest (60 trees) | 1,077 scans (20 fruits) | **47.54% ± 10.73% Acc** (Bal: 47.53%) | 20.00% | 🟡 Moderate Performance |
| **CS4: Brix Sweetness** | 18-channel digital twin | Cultivar ElasticNet | 60 scans (33 fruits) | **R² = -0.0316**, RMSE = 1.19 °Bx | Mean guess | 🔴 Negative R² (No SWIR bands) |
| **CS5: Visual Surface Defect** | RGB Camera image | MobileNetV3-Small | Public defect dataset | **95.67% Test Acc** (F1: 95.60%) | 50.00% | 🟢 Promising Prototype |
| **CS6: Domain Shift (OOD)** | Optical distribution | Mahalanobis LedoitWolf | Roomi (26) vs Spectro (240) | **100.0% Shift Detected** | N/A | 🟡 Cross-Instrument Detector |

## 2. Model Inference Answers Across 7 Test Scenarios

### Scenario 1 Apple Photo
- **Input**: `RGB Image (data/test_user_apple.jpg)`
- **Pesticide Prediction**: `Low-treatment-like` (Confidence: 91.37%, OOD Status: Uncertain / Out-of-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: 37.73%)
- **Cultivar Prediction**: `red delicious usa`
- **Estimated Brix**: `9.41`
- **Visual Defect Status**: `Good / Sound` (Defect Prob: 0.13)

### Scenario 2 Mock Hardware Scan
- **Input**: `AS7265x Mock Reflection Array (18 channels)`
- **Pesticide Prediction**: `Control-like` (Confidence: 100.0%, OOD Status: Uncertain / Out-of-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: 55.77%)
- **Cultivar Prediction**: `red delicious usa`
- **Estimated Brix**: `11.78`

### Scenario 3 Synthetic Control Spectrum
- **Input**: `Synthetic Control Apple (row 0, ground truth: Control)`
- **Pesticide Prediction**: `Low-treatment-like` (Confidence: 74.61%, OOD Status: In-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: N/A%)
- **Cultivar Prediction**: `kinnaur`
- **Estimated Brix**: `9.84 °Bx`

### Scenario 4 Synthetic High Treatment Spectrum
- **Input**: `Synthetic High Treatment Apple (ground truth: High-treatment)`
- **Pesticide Prediction**: `High-treatment-like` (Confidence: 91.64%, OOD Status: In-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: N/A%)
- **Cultivar Prediction**: `kinnaur`
- **Estimated Brix**: `8.97 °Bx`

### Scenario 5 Real Roomi HSI Spectrum
- **Input**: `Real Roomi .bil HSI Extraction (Fruit 1, untreated apple)`
- **Pesticide (Synthetic Model)**: `Control-like` (Confidence: 100.0%, OOD: Uncertain / Out-of-Distribution)
- **Pesticide (Real Model)**: `Control-like` (Confidence: 53.22%, OOD: Uncertain / Out-of-Distribution)
- **Estimated Brix**: `11.47 °Bx`

### Scenario 6 SpectroFood Natural Variation
- **Input**: `SpectroFood 18-Channel Natural Variation Scan (Domain Shift Target)`
- **Pesticide Prediction**: `High-treatment-like` (Confidence: 100.0%, OOD Status: Uncertain / Out-of-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: N/A%)
- **Cultivar Prediction**: `kinnaur`
- **Estimated Brix**: `-32332.98 °Bx`

### Scenario 7 Noisy Physical Sensor Simulation
- **Input**: `AS7265x with 30-count dark offset, 4-count noise, 8% placement drift, 4% LED drift`
- **Pesticide Prediction**: `Control-like` (Confidence: 100.0%, OOD Status: Uncertain / Out-of-Distribution)
- **Ripeness Prediction**: `Under-ripe` (Confidence: N/A%)
- **Cultivar Prediction**: `kinnaur`
- **Estimated Brix**: `4.51 °Bx`

