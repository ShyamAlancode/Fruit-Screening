# CFRD — Physical Hardware Validation Protocol

> **Based on:** Second Technical Review, §8 (Hardware Validation Plan)  
> **Date:** September 24, 2026  
> **Status:** Pre-hardware — ready to execute when AS7265x triad arrives  
> **Target hardware:** Raspberry Pi 5 + AMS AS7265x tri-spectral sensor + PTFE reference

> [!IMPORTANT]
> No physical experiment should begin until Stage 0 is complete and all 18 channels
> return nonzero, non-saturated, reproducible values.

---

## Stage 0 — Hardware Bring-Up (Day 1, must complete before any fruit experiment)

**Goal:** Prove the sensor communicates and each channel returns plausible values.

### Procedure

```bash
# On Raspberry Pi 5 with pinned environment:
pip install -r requirements_lock.txt

# Verify I2C device presence:
i2cdetect -y 1
# Expected: address 0x49 visible

# Run bring-up tool:
python scripts/hardware_calibration_tool.py --mode check
```

### Manual Checklist

- [ ] Raspberry Pi OS fresh environment installed
- [ ] Pinned dependency set installed (`requirements_lock.txt`)
- [ ] I2C device confirmed at 0x49
- [ ] Sensor identification / status registers read successfully
- [ ] 18 channels read under stable white LED source
- [ ] All 18 channels: nonzero, non-saturated, repeatable
- [ ] Gain, integration time, LED state, temperature recorded for every scan

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Channel count | Exactly 18 valid channels |
| Zero channels | None (all > 0 under white illumination) |
| Saturated channels | None (all < 65535 ADU) |
| Repeated reads | Channel-to-channel CV < 5% within 10 consecutive reads |
| Metadata | Gain, integration_time, LED_mode, temperature logged per scan |

---

## Experiment H1 — Dark Reference Stability

**Goal:** Characterize dark current and temporal stability.

### Protocol
- Cover sensor window completely (no light leak)
- Record 30 dark frames immediately after power-on
- Repeat after 5, 30, and 60 minutes warm-up
- Record chamber temperature at each time point

### Metrics to Report
| Metric | Per-channel | Aggregate |
|---|---|---|
| Mean dark value | ✅ | Mean of means |
| Standard deviation | ✅ | Max CV across channels |
| CV (%) | ✅ Required < 5% |
| Drift over 60 min | ✅ Linear fit slope |

### Acceptance Criterion
Dark current CV < 5% per channel; no systematic drift > 2% over 60 minutes warm-up.

---

## Experiment H2 — PTFE White Reference Repeatability

**Goal:** Establish sensor measurement repeatability on a stable diffuse target.

### Protocol
- PTFE / Spectralon tile placed 20 mm ± 1 mm below sensor window
- 30 scans per position
- 3 independent repositionings of the tile (remove and replace between each block)
- Repeat on 3 separate days
- At least 2 LED current settings (12.5 mA and 50 mA)

> [!WARNING]
> Do NOT use a single 20-scan session as the only repeatability evidence.
> The previous simulated result (CV < 0.5%) was circular — the noise level matched the injected noise.
> Real repeatability must be measured across repositionings and days.

### Metrics to Report

| Metric | Requirement |
|---|---|
| Within-position CV | Report per channel (target < 1%) |
| Between-position CV | Report per channel (target < 3%) |
| Between-day CV | Report per channel (target < 5%) |
| Effect of LED current | Report response at each current level |

---

## Experiment H3 — Illumination Spectral Characterization

**Goal:** Verify that the four LED modes produce physically distinct spectral responses.

> [!CAUTION]
> The current software driver sets `current_led_mode` as a state variable but does not
> actually switch physical GPIO or sensor LED driver registers. This must be implemented
> before H3 is meaningful.

### Protocol
For each illumination source {5700K white, 875 nm IR, 405 nm UV, 365 nm UV}:

- [ ] Verify LED physically activates (measure optical output with reference photodiode or camera)
- [ ] Record 10 scans on PTFE reference
- [ ] Record dark baseline
- [ ] Record temperature
- [ ] Note LED current, duty cycle, warm-up time

### Expected Physical Behavior
- White LED: broadband response peaking in visible channels
- IR LED: dominant response in 730–940 nm channels; near-zero in 410–535 nm
- UV LEDs: potential fluorescence excitation; response depends on target material

### Acceptance Criterion
The four modes must produce statistically distinct spectral response patterns on the PTFE reference. If they are indistinguishable, the LED wiring or firmware has not been implemented correctly.

---

## Experiment H4 — Biological Baseline Study (First Physical Fruit Data)

**Goal:** Quantify inter-apple biological variance on untreated fruit.

### Protocol
- **Minimum pilot:** 20–30 untreated supermarket apples
- **Cultivars:** At least 4 different cultivars (e.g., Gala, Fuji, Granny Smith, Braeburn)
- **Spots per apple:** 10 perimeter spots, equally spaced
- **Sessions:** 2 independent sessions (different days)
- **Calibration:** Fresh dark + white reference at the start of each session

### Statistical Model
```
measurement = μ_fruit + spatial_effect + session_effect + residual
```
Estimate variance components rather than reporting only mean spectra.

### Metrics to Report

| Component | Must Report |
|---|---|
| Inter-apple variance | Variance component magnitude |
| Intra-apple (spatial) variance | Variance component per apple |
| Session effect | Day-to-day drift |
| Cultivar effect | Means and CIs per cultivar |
| Comparison to cuticle-wax shift | Quantify vs Roomi dataset baseline |

> This experiment establishes the physical signal-to-noise baseline before any treatment experiment.

---

## Experiment H5 — Controlled Treatment Study

**Goal:** Compare known-treated and untreated fruit using the physical sensor.

### Design (Factorial)
| Factor | Levels |
|---|---|
| Treatment | Untreated, Formulation A low, Formulation A high, Formulation B low, Formulation B high |
| Apples per cell | Minimum 4 independent apples |
| Measurement order | Randomized within session |
| Spots per apple | 10 perimeter spots |

- Randomize measurement order and operator where possible
- Record application concentration and method for each apple
- Photograph each apple before scanning (spatial reference)
- Store a portion of each apple for chemical analysis (H6)

### Analysis
- Apply exactly the same preprocessing as the simulation benchmark (SG derivative + SNV)
- Apply the same 17-adjacent-difference feature representation
- Run LOAO cross-validation
- Report sensitivity, specificity, balanced accuracy, AUROC, Wilson CIs

---

## Experiment H6 — Chemical Ground Truth (Subset)

**Goal:** Verify that the optical signal corresponds to actual surface chemical residue.

### Protocol
- For a subset of treated apples (recommend ≥ 5 per treatment cell):
  - Scan optically with the physical AS7265x
  - Submit to accredited laboratory for:
    - **LC-MS/MS** (liquid chromatography-mass spectrometry) — quantitative ppm
    - **GC-MS** if volatile compounds present
- Report: actual residue concentration vs optical response

> [!NOTE]
> The Roomi dataset describes "treated" apples but does not provide per-apple ppm measurements.
> Chemical ground truth is required before making any residue screening performance claim.

### Acceptance Criterion
At least a statistically significant Spearman correlation (p < 0.05) between detected optical anomaly score and independently measured residue concentration.

---

## Experiment H7 — Evaluation Protocol (Frozen Before Looking at Results)

> [!CAUTION]
> **Pre-register the evaluation protocol before looking at the held-out test results.**
> Do not select thresholds, features, or models by inspecting held-out performance.

### Required Before Running

- [ ] Preprocessing pipeline frozen and committed (one SHA256 hash)
- [ ] Feature representation frozen (17 adjacent differences OR other)
- [ ] Classifier and hyperparameters frozen (no tuning on held-out)
- [ ] Threshold selected on training data only
- [ ] GroupKFold / LOAO fold assignment pre-specified and committed

### Required Metrics (All Must Be Reported)

| Metric | Required |
|---|---|
| Sensitivity at fixed specificity (90%) | ✅ |
| Specificity at fixed sensitivity (90%) | ✅ |
| ROC curve with permutation envelope | ✅ |
| PR curve | ✅ |
| Balanced accuracy | ✅ |
| Wilson 95% CIs for all proportions | ✅ |
| Per-apple confusion matrices | ✅ |
| AUROC | ✅ |
| Permutation p-value | ✅ |
| Treatment formulation as separate factor | ✅ |
| Concentration as separate factor | ✅ |

---

## Production Acceptance Criteria

Before calling CFRD "physically validated," all of the following must be met:

| Criterion | Requirement |
|---|---|
| Physical sensor readout | ✅ 18 channels, nonzero, non-saturated |
| Calibration repeatability | ✅ CV < 3% across repositionings |
| Cross-day stability | ✅ Drift < 5% over 3+ days |
| Known-treated detection | ✅ Sensitivity > pre-declared minimum at fixed specificity |
| Clean specificity | ✅ Specificity > pre-declared minimum at fixed sensitivity |
| OOD false positives | ✅ Non-apple objects correctly flagged as OOD |
| End-to-end inference | ✅ Runs on Raspberry Pi 5 without error |
| Documented limitations | ✅ All per-experiment limitations recorded |

---

## Physical Validation 30-Day Schedule

| Week | Experiments | Deliverable |
|---|---|---|
| Week 1 | Stage 0, H1, H2, H3 | Dark/white/repeatability/LED verification |
| Week 2 | H4 — biological baseline | 20–30 clean apples, variance components |
| Week 3 | H5, H6 — treatment study | Known-treated vs clean, chemical subset |
| Week 4 | H7 — frozen evaluation | Locked benchmark, full metric disclosure |

---

## Comparison to Simulation Benchmark

After physical experiments, map results to simulation baseline:

| Metric | Simulation (current) | Physical (target) | Gap |
|---|---|---|---|
| LOAO scan accuracy | 75.0% | TBD | TBD |
| Sensitivity | 100% | TBD | TBD |
| Specificity | 30% | TBD | TBD |
| AUROC | 0.59 | TBD | TBD |
| Apple cohorts | 5 (Roomi) | TBD | TBD |
| Chemical ground truth | None | LC-MS/MS subset | TBD |
