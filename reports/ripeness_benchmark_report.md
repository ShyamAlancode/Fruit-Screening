# 🍎 CFRD: Multi-Stage Apple Ripeness Vision Benchmark Report

Generated from automated evaluation across independent test crops (408 apples) and external RawRipe dataset (213 apples).

## 1. Executive Summary: Test Set Benchmark Accuracies

| Model Architecture | Test Accuracy | Balanced Acc | Macro-Precision | Macro-Recall | Macro-F1 | External RawRipe Acc | Status |
|---|---|---|---|---|---|---|---|
| **MobileNetV3-Small** | **91.91%** | 91.90% | 92.11% | 91.90% | **91.83%** | **86.85%** | 🟢 Exceeds 90% Target |
| **EfficientNet-B0** | **91.42%** | 91.43% | 91.62% | 91.43% | **91.29%** | **88.26%** | 🟢 Exceeds 90% Target |
| **ResNet18** | **86.52%** | 86.53% | 87.75% | 86.53% | **85.92%** | **94.37%** | 🟡 Strong Baseline |

## 2. Per-Class Confusion Matrices (Independent Test Set: 408 Crops)

Classes: `0: Immature (Green)`, `1: Semi-Mature (Transition)`, `2: Mature (Red)`

### MobileNetV3-Small

| True \ Pred | Immature | Semi-Mature | Mature |
|---|---|---|---|
| **Immature (141)** | **137** | 4 | 0 |
| **Semi-Mature (136)** | 16 | **112** | 8 |
| **Mature (131)** | 1 | 4 | **126** |

### EfficientNet-B0

| True \ Pred | Immature | Semi-Mature | Mature |
|---|---|---|---|
| **Immature (141)** | **136** | 4 | 1 |
| **Semi-Mature (136)** | 15 | **110** | 11 |
| **Mature (131)** | 1 | 3 | **127** |

### ResNet18

| True \ Pred | Immature | Semi-Mature | Mature |
|---|---|---|---|
| **Immature (141)** | **137** | 3 | 1 |
| **Semi-Mature (136)** | 34 | **88** | 14 |
| **Mature (131)** | 0 | 3 | **128** |

## 3. External Dataset Generalization: RawRipe Zero-Shot Test

Evaluates whether models trained on the 2026 orchard dataset generalize to completely different apple photographs.

### MobileNetV3-Small
- **Total External Samples**: 213 apples (105 raw, 108 ripe)
- **Exact Match Accuracy**: **86.85%**
- **Raw (Immature) Recall**: 80.00%
- **Ripe (Mature) Recall**: 93.52%
- **Predicted Distribution**: Immature: 84, Semi-Mature: 26, Mature: 103

### EfficientNet-B0
- **Total External Samples**: 213 apples (105 raw, 108 ripe)
- **Exact Match Accuracy**: **88.26%**
- **Raw (Immature) Recall**: 84.76%
- **Ripe (Mature) Recall**: 91.67%
- **Predicted Distribution**: Immature: 89, Semi-Mature: 24, Mature: 100

### ResNet18
- **Total External Samples**: 213 apples (105 raw, 108 ripe)
- **Exact Match Accuracy**: **94.37%**
- **Raw (Immature) Recall**: 88.57%
- **Ripe (Mature) Recall**: 100.00%
- **Predicted Distribution**: Immature: 93, Semi-Mature: 8, Mature: 112

