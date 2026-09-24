# Cross-dataset synthetic domain-shift stress test

Generated: 2026-09-12 11:46:22

## Summary Metrics
| Metric | Value |
|---|---|
| In-distribution samples (Roomi) | 26 |
| OOD candidate samples (SpectroFood) | 240 |
| Distance threshold (p95) | 2.5040 |
| In-distribution mean distance | 1.6649 |
| SpectroFood mean distance | 8.6537 |
| SpectroFood samples flagged uncertain/OOD | 240 / 240 (100.0%) |
| SpectroFood false alarm rate (confident treated-like) | 5 / 240 (2.08%) |

## Scientific Interpretation
> This stress test demonstrates why natural biological variation and cross-sensor domain shift
> must be monitored via OOD/uncertainty scoring. When an external apple dataset is presented,
> the Mahalanobis detector correctly identifies that the spectral profile deviates from the
> laboratory training distribution, preventing overconfident false residue classifications.