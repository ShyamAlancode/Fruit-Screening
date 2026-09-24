# Cultivar-Aware Dual-Benchmark Brix Regression Report

**Generated**: 2026-09-12 09:13 UTC
**Dataset**: University of Essex Hyperspectral Apple Benchmark (DOI: 10.5526/ERDR-00000228)
**Ground Truth**: ATAGO PR-32-alpha refractometer (Brix, dBx)
**Sample Size**: 60 hyperspectral scans across 33 unique physical apples
**Cultivars Tested**: Braeburn, Cox's Orange Pippin, Gala
**Validation Protocol**: Strict **GroupKFold (Zero-Leakage on Physical Apple ID)**

---

## Executive Summary & Scientific Verdict

> [!WARNING]
> **Negative Result Confirmation**: All evaluated regression models yield negative cross-validated $R^2$ values ($R^2 \in [-0.19, -0.01]$).  
> In regression, a negative $R^2$ indicates that a trivial naive baseline predicting the mean Brix of the training fold outperforms the model. The 18-channel AS7265x digital twin **does not reliably predict apple sugar content (Brix)** under current preprocessing and sample sizes.

### Key Scientific Findings:
1. **Multispectral Information Limit**: The 18 broad channels of the AS7265x (20 nm FWHM spanning 410–940 nm) are unable to resolve the subtle, narrow vibrational overtone bands of sucrose, glucose, and fructose (which typically occur in the 900–1700 nm SWIR region).
2. **Cultivar Confounding**: Cultivar peel pigmentation and cellular structure represent the dominant source of variance. While adding cultivar priors slightly reduced error (from $R^2 \approx -0.17$ to $R^2 \approx -0.03$), cross-validated predictive power remains below the zero-information threshold.
3. **Sample Size Inadequacy**: 33 physical apples (60 scans across 3 cultivars) is fundamentally insufficient for training non-destructive Brix models.
4. **Provenance Note on `cox_batch7`**: While the upstream compressed archive `cox_batch7.tar.gz` had header/truncation errors, 3 scans (`964`, `965`, `966`) were parsed from extracted folders. Whether evaluated with 60 scans or strictly filtered to 57 scans excluding `cox_batch7`, $R^2$ remains consistently negative, confirming that this data defect does not alter the fundamental finding.

---

## Performance Comparison Table

| Benchmark Track | Model Architecture | Cross-Validated $R^2$ | RMSE (dBx) | MAE (dBx) | Pearson $r$ | p-value |
|---|---|---|---|---|---|---|
| **Specim IQ 204-Band (Naive)** | HistGBR | -0.1928 | 1.279 | 0.912 | -0.0839 | 5.24e-01 |
| **AS7265x 18-Ch Twin (Naive)** | HistGBR | -0.1683 | 1.266 | 0.913 | 0.0075 | 9.55e-01 |
| **Specim IQ 204-Band (+ Cultivar)** | **ElasticNet** | **-0.0450** | **1.197** | **0.820** | **0.2511** | **5.30e-02** |
| **AS7265x 18-Ch Twin (+ Cultivar)** | **ElasticNet** | **-0.0316** | **1.190** | **0.811** | **0.2635** | **4.20e-02** |
| **AS7265x 18-Ch Twin (+ Noise + Cultivar)** | **HistGBR** | **-0.0080** | **1.176** | **0.838** | **0.2250** | **8.39e-02** |

---

## Detailed Model Breakdown

### Track A: Specim IQ 204-Band (+ Cultivar)
- **Best Model**: ElasticNet
- **R2**: -0.0450
- **RMSE**: 1.197 dBx
- **Pearson correlation**: 0.2511 ($p = 5.30e-02$)

### Track B: AS7265x 18-Channel Digital Twin (+ Cultivar)
- **Best Model**: ElasticNet
- **R2**: -0.0316
- **RMSE**: 1.190 dBx
- **Pearson correlation**: 0.2635 ($p = 4.20e-02$)

---

## Per-Apple Ground Truth Summary (33 Physical Apples)

| Apple ID | Cultivar | Scans Evaluated | Refractometer Brix (dBx) | Penetrometer Firmness (kg/cm2) |
|---|---|---|---|---|
| Braeburn_173 | Braeburn | 4 | 16.09 | 12.805 |
| Braeburn_174 | Braeburn | 4 | 12.61 | 11.075 |
| Braeburn_175 | Braeburn | 4 | 13.05 | 9.745 |
| Cox's Orange Pippin_1189 | Cox's Orange Pippin | 1 | 11.82 | 8.300 |
| Cox's Orange Pippin_1190 | Cox's Orange Pippin | 1 | 11.32 | 8.710 |
| Cox's Orange Pippin_1191 | Cox's Orange Pippin | 1 | 11.75 | 8.930 |
| Gala_1504 | Gala | 1 | 12.16 | 8.470 |
| Gala_1505 | Gala | 1 | 12.57 | 8.180 |
| Gala_1506 | Gala | 1 | 9.94 | 7.540 |
| Gala_1486 | Gala | 2 | 12.69 | 7.475 |
| Gala_1487 | Gala | 2 | 11.72 | 7.860 |
| Gala_1488 | Gala | 2 | 12.06 | 8.395 |
| Cox's Orange Pippin_1137 | Cox's Orange Pippin | 2 | 11.85 | 9.355 |
| Cox's Orange Pippin_1138 | Cox's Orange Pippin | 2 | 13.18 | 10.015 |
| Cox's Orange Pippin_1139 | Cox's Orange Pippin | 2 | 14.04 | 9.850 |
| Gala_1489 | Gala | 2 | 12.63 | 7.925 |
| Gala_1490 | Gala | 2 | 12.41 | 7.200 |
| Gala_1491 | Gala | 2 | 11.79 | 9.155 |
| Cox's Orange Pippin_1162 | Cox's Orange Pippin | 2 | 11.70 | 8.540 |
| Cox's Orange Pippin_1163 | Cox's Orange Pippin | 2 | 11.93 | 8.105 |
| Cox's Orange Pippin_1164 | Cox's Orange Pippin | 2 | 11.77 | 7.725 |
| Gala_1519 | Gala | 1 | 11.49 | 9.255 |
| Gala_1520 | Gala | 1 | 12.93 | 7.370 |
| Gala_1521 | Gala | 1 | 12.93 | 7.170 |
| Gala_1542 | Gala | 1 | 12.45 | 8.720 |
| Gala_1543 | Gala | 1 | 12.67 | 7.790 |
| Gala_1544 | Gala | 1 | 11.09 | 7.865 |
| Cox's Orange Pippin_1140 | Cox's Orange Pippin | 2 | 12.68 | 8.925 |
| Cox's Orange Pippin_1141 | Cox's Orange Pippin | 2 | 11.47 | 8.530 |
| Cox's Orange Pippin_1142 | Cox's Orange Pippin | 2 | 11.97 | 9.235 |
| Gala_1492 | Gala | 2 | 12.91 | 7.505 |
| Gala_1493 | Gala | 2 | 11.65 | 8.365 |
| Gala_1494 | Gala | 2 | 12.14 | 6.935 |
