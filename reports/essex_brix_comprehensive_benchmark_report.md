# University of Essex Specim IQ Hyperspectral Apple Brix Benchmark Report

**Date Generated**: 2026-09-24 08:24:12
**Total Samples**: 417 apple observations across 109 unique fruits
**Cultivars**: Cox's Orange Pippin, Golden Delicious, Fuji, Braeburn, Gala
**Brix Range**: 9.20 °Bx to 16.80 °Bx (Mean: 12.47 °Bx, Std: 1.42)

## 1. Executive Summary

This benchmark compares continuous non-destructive Brix sweetness estimation across two operational modalities:
- **Track A (Upper Bound)**: High-resolution Specim IQ research hyperspectral camera (204 continuous bands, 397–1004 nm).
- **Track B (Deployment Twin)**: Simulated AS7265x 18-channel optical sensor (410–940 nm, 20 nm FWHM bandpass filtering).

All evaluations use **GroupKFold (5-fold) grouped strictly by fruit ID**, preventing any multi-view leakage between training and testing sets.

## 2. Benchmark Results Table

| Track                            | Model                |     R2 |   MAE_Brix |   RMSE_Brix |   Pearson_r |   Pearson_p |   BinaryScreening_Acc_pct |   ThreeTier_Acc_pct |
|:---------------------------------|:---------------------|-------:|-----------:|------------:|------------:|------------:|--------------------------:|--------------------:|
| Track A: 204-Band Specim IQ HSI  | PLSR (n=6)           | 0.3733 |      0.887 |       1.126 |       0.612 |    2.83e-44 |                     82.01 |               65.47 |
| Track A: 204-Band Specim IQ HSI  | PLSR (n=10)          | 0.4789 |      0.774 |       1.027 |       0.697 |    7.05e-62 |                     82.73 |               74.1  |
| Track A: 204-Band Specim IQ HSI  | SVR (RBF, C=10)      | 0.4537 |      0.791 |       1.051 |       0.676 |    4.34e-57 |                     79.38 |               71.22 |
| Track A: 204-Band Specim IQ HSI  | Ridge (alpha=10)     | 0.4402 |      0.823 |       1.064 |       0.664 |    2.8e-54  |                     82.73 |               67.63 |
| Track A: 204-Band Specim IQ HSI  | Random Forest        | 0.4234 |      0.847 |       1.08  |       0.654 |    2.67e-52 |                     78.42 |               67.15 |
| Track A: 204-Band Specim IQ HSI  | HistGradientBoosting | 0.4216 |      0.839 |       1.082 |       0.664 |    2.63e-54 |                     76.74 |               66.19 |
| Track B: 18-Channel AS7265x Twin | PLSR (n=6)           | 0.35   |      0.914 |       1.146 |       0.593 |    4.93e-41 |                     82.01 |               64.51 |
| Track B: 18-Channel AS7265x Twin | PLSR (n=10)          | 0.3699 |      0.893 |       1.129 |       0.614 |    1.68e-44 |                     82.01 |               65.47 |
| Track B: 18-Channel AS7265x Twin | SVR (RBF, C=10)      | 0.4093 |      0.824 |       1.093 |       0.645 |    2.11e-50 |                     79.62 |               68.82 |
| Track B: 18-Channel AS7265x Twin | Ridge (alpha=10)     | 0.3411 |      0.913 |       1.154 |       0.584 |    1.57e-39 |                     83.21 |               65.71 |
| Track B: 18-Channel AS7265x Twin | Random Forest        | 0.4239 |      0.84  |       1.079 |       0.656 |    1.02e-52 |                     80.1  |               69.54 |
| Track B: 18-Channel AS7265x Twin | HistGradientBoosting | 0.427  |      0.843 |       1.076 |       0.665 |    1.68e-54 |                     78.66 |               69.54 |

## 3. Scientific Insights

- **Track A Best Model**: `PLSR (n=10)` achieved $R^2 = 0.4789$ with MAE = 0.77 °Bx.
- **Track B Best Model**: `HistGradientBoosting` achieved $R^2 = 0.4270$ with MAE = 0.84 °Bx.
- **Optical Information Preservation**: The 18-channel AS7265x sensor twin preserves the key spectral regions for sugar overtone absorption (900–940 nm) and chlorophyll degradation (680–705 nm), enabling high commercial screening accuracy.
