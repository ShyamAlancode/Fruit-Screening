# 3-Class Treatment-Condition Classifier Results (real)

Real Roomi Extraction Pilot (Below random chance 33.3%; sample size 13 fruits inadequate for product claim)

## PCA+LDA (Primary Baseline)

| Fold | Train rows | Test rows | Accuracy | Bal. Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 20 | 6 | 0.3333 | 0.3750 | 0.3000 |
| 2 | 20 | 6 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 20 | 6 | 0.1667 | 0.2500 | 0.1333 |
| 4 | 22 | 4 | 0.2500 | 0.2500 | 0.1667 |
| 5 | 22 | 4 | 0.0000 | 0.0000 | 0.0000 |
| **Mean** | — | — | **0.1500 ± 0.1333** | **0.1750** | **0.1200** |

## PCA+SVM RBF (Comparison)

| Fold | Train rows | Test rows | Accuracy | Bal. Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 1 | 20 | 6 | 0.0000 | 0.0000 | 0.0000 |
| 2 | 20 | 6 | 0.0000 | 0.0000 | 0.0000 |
| 3 | 20 | 6 | 0.0000 | 0.0000 | 0.0000 |
| 4 | 22 | 4 | 0.0000 | 0.0000 | 0.0000 |
| 5 | 22 | 4 | 0.0000 | 0.0000 | 0.0000 |
| **Mean** | — | — | **0.0000 ± 0.0000** | **0.0000** | **0.0000** |

## Summary

| Pipeline | Mean Acc | Std Acc | Bal Acc | Macro-F1 |
|---|---|---|---|---|
| PCA+LDA  | 0.1500 | 0.1333 | 0.1750 | 0.1200 |
| PCA+SVM  | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
