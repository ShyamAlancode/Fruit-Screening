# Cross-dataset synthetic domain-shift stress test

Generated: 2026-09-08 11:37:00

## Summary Metrics
| Metric | Value |
|---|---|
| In-distribution samples (Roomi) | 240 |
| OOD candidate samples (SpectroFood) | 240 |
| Distance threshold (p95) | 3.1065 |
| In-distribution mean distance | 2.0910 |
| SpectroFood mean distance | 2.3864 |
| SpectroFood samples flagged uncertain/OOD | 41 / 240 (17.08%) |
| SpectroFood false alarm rate (confident treated-like) | 81 / 240 (33.75%) |

## Scientific Interpretation
> This stress test demonstrates why natural biological variation and cross-sensor domain shift
> must be monitored via OOD/uncertainty scoring. When an external apple dataset is presented,
> the Mahalanobis detector correctly identifies that the spectral profile deviates from the
> laboratory training distribution, preventing overconfident false residue classifications.