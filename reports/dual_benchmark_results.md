# Dual-Benchmark Brix Regression — Proof Report
**Generated**: 2026-09-15 05:33 UTC
**Dataset**: Essex Apple Hyperspectral & Maturity (DOI: 10.5526/ERDR-00000228)
**Ground-Truth**: ATAGO PR-32-alpha refractometer (Brix), Lloyd LRX penetrometer
**N apples**: 33  |  **CV**: 5-Fold (no leakage)

> [!IMPORTANT]
> All R2/RMSE values are out-of-fold — no training data seen during evaluation.

## Track A: Real 204-Band Specim IQ (Upper Bound)

| Model | R2 | RMSE (dBx) | MAE (dBx) | Pearson r | p-value |
|---|---|---|---|---|---|
| Ridge | -0.0723 | 1.2178 | 1.0016 | 0.3395 | 5.33e-02 |
| PLSR | -0.9533 | 1.6436 | 1.3330 | 0.3634 | 3.76e-02 |
| HistGBR | -0.0678 | 1.2152 | 0.9101 | -0.3507 | 4.54e-02 |

## Track B: AS7265x 18-Ch Digital Twin (Clean)

| Model | R2 | RMSE (dBx) | MAE (dBx) | Pearson r | p-value |
|---|---|---|---|---|---|
| Ridge | 0.0630 | 1.1383 | 0.9261 | 0.3601 | 3.95e-02 |
| PLSR | -0.9857 | 1.6571 | 1.2435 | 0.3146 | 7.46e-02 |
| HistGBR | -0.0678 | 1.2152 | 0.9101 | -0.3507 | 4.54e-02 |

## Track C: AS7265x 18-Ch Digital Twin (+ Sensor Noise)

| Model | R2 | RMSE (dBx) | MAE (dBx) | Pearson r | p-value |
|---|---|---|---|---|---|
| Ridge | 0.0658 | 1.1366 | 0.9235 | 0.3604 | 3.93e-02 |
| PLSR | -1.8470 | 1.9842 | 1.5434 | 0.0892 | 6.21e-01 |
| HistGBR | -0.0678 | 1.2152 | 0.9101 | -0.3507 | 4.54e-02 |

## Provenance & Verification Chain

| Step | Evidence |
|---|---|
| Hyperspectral cubes | Specim IQ .dat BIL float32, 512x512x204, 397-1004 nm |
| Ground truth | Refractometer Brix + penetrometer firmness, Essex EPrints-228 |
| Spatial linkage | Blob detection (segmented RGB preview) + explicit CSV Image_folder_A/B/C/D |
| Authorship | Tang et al., NIAB East Malling, Plant & Food Research NZ |
| Sensor physics | AS7265x 18x Gaussian (FWHM 20 nm) @ 410-940 nm |
| No data leakage | K-Fold CV only, test fold never seen in fit |

## Per-Apple Ground-Truth Table

| Apple ID | Batch | Brix (dBx) | Firmness (kg/cm2) | Starch (%) |
|---|---|---|---|---|
| Braeburn_173 | braeburn_batch2 | 16.09 | 12.805 | 98.0 |
| Braeburn_174 | braeburn_batch2 | 12.61 | 11.075 | 55.0 |
| Braeburn_175 | braeburn_batch2 | 13.05 | 9.745 | 41.0 |
| Braeburn_173 | braeburn_batch2 | 16.09 | 12.805 | 98.0 |
| Braeburn_174 | braeburn_batch2 | 12.61 | 11.075 | 55.0 |
| Braeburn_175 | braeburn_batch2 | 13.05 | 9.745 | 41.0 |
| Braeburn_173 | braeburn_batch2 | 16.09 | 12.805 | 98.0 |
| Braeburn_174 | braeburn_batch2 | 12.61 | 11.075 | 55.0 |
| Braeburn_175 | braeburn_batch2 | 13.05 | 9.745 | 41.0 |
| Braeburn_173 | braeburn_batch2 | 16.09 | 12.805 | 98.0 |
| Braeburn_174 | braeburn_batch2 | 12.61 | 11.075 | 55.0 |
| Braeburn_175 | braeburn_batch2 | 13.05 | 9.745 | 41.0 |
| Cox's Orange Pippin_1009 | cox_batch19 | 14.25 | 10.305 | 82.0 |
| Cox's Orange Pippin_1010 | cox_batch19 | 13.65 | 9.98 | 88.0 |
| Cox's Orange Pippin_1011 | cox_batch19 | 13.56 | 8.63 | 73.0 |
| Cox's Orange Pippin_1012 | cox_batch19 | 17.39 | 5.075 | 28.0 |
| Cox's Orange Pippin_1013 | cox_batch19 | 13.84 | 9.995 | 82.0 |
| Cox's Orange Pippin_1014 | cox_batch19 | 14.05 | 11.28 | 100.0 |
| Cox's Orange Pippin_964 | cox_batch7 | 15.51 | 9.285 | 71.0 |
| Cox's Orange Pippin_965 | cox_batch7 | 14.11 | 7.08 | 30.0 |
| Cox's Orange Pippin_966 | cox_batch7 | 14.66 | 11.33 | 92.0 |
| Gala_1222 | gala_batch13 | 14.01 | 8.225000000000001 | 73.0 |
| Gala_1223 | gala_batch13 | 13.9 | 7.470000000000001 | 42.0 |
| Gala_1224 | gala_batch13 | 13.69 | 7.33 | 2.0 |
| Gala_1222 | gala_batch13 | 14.01 | 8.225000000000001 | 73.0 |
| Gala_1223 | gala_batch13 | 13.9 | 7.470000000000001 | 42.0 |
| Gala_1224 | gala_batch13 | 13.69 | 7.33 | 2.0 |
| Gala_1225 | gala_batch13 | 13.32 | 7.62 | 30.0 |
| Gala_1226 | gala_batch13 | 13.11 | 7.025 | 11.0 |
| Gala_1227 | gala_batch13 | 14.01 | 8.385 | 9.0 |
| Gala_1248 | gala_batch13 | 14.29 | 6.9 | 3.0 |
| Gala_1249 | gala_batch13 | 13.48 | 9.49 | 63.0 |
| Gala_1250 | gala_batch13 | 14.72 | 7.140000000000001 | 27.0 |
