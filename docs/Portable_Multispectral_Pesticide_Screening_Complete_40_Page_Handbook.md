# Portable Multispectral UV-Excitation Edge-AI Device for Fresh-Produce Treatment-Condition Screening
## Complete Project Handbook, Design Rationale, Research Record, and Execution Guide

 

**Core sensor:** AS7265x 18-channel VIS–NIR spectral sensor, approximately 410–940 nm  
**Primary V1 output:** Untreated-like / treated-like / uncertain—laboratory confirmation recommended  

> **Read this first:** This project is currently a scientifically disciplined software and engineering prototype plan. It has a valid synthetic-data baseline, but it does **not** yet have physical AS7265x validation, demonstrated UV fluorescence, laboratory-confirmed pesticide-residue labels, MRL validation, or real-world 90% accuracy. The next phase determines whether the concept becomes a usable instrument.

---

# Contents

1. Project in Plain Language
2. The Real Problem
3. Why Existing Methods Are Not Enough
4. The Product Vision and the V1 Scope
5. What the Device Can and Cannot Claim
6. Core Scientific Question
7. System Architecture
8. Hardware Components and Their Roles
9. Why Each Illumination Mode Exists
10. The Optical Chamber
11. Electrical Integration and Control
12. Scan Acquisition Workflow
13. Calibration Theory
14. The 405 nm Crosstalk Problem
15. Hardware Bring-Up Protocol
16. GO / NO-GO Gate
17. Existing Dataset: Roomi Apple HSI
18. From Hyperspectral Image to 18 Sensor Values
19. Synthetic Dataset: What It Proves and Does Not Prove
20. Software Pipeline Overview
21. Data Loading and Schema
22. Preprocessing and the SNV Result
23. PCA: Why It Is Used
24. LDA: Why It Won
25. Model Benchmarking Results
26. GroupKFold and Leakage Prevention
27. Narrower Binary Tasks
28. Confidence Rejection and Coverage
29. Mahalanobis Out-of-Distribution Detection
30. Digital Twin / Spectral Reconstruction
31. Real Hardware Dataset Plan
32. Confounder Controls
33. Real-Data Feature Engineering
34. Real-Data Model Training
35. Final Validation Protocol
36. Path to 90% Accuracy
37. Product Modules Beyond Screening
38. Literature Survey and Prior Art
39. Novelty and Patent Strategy
40. Funding, Publication, Timeline, and Master Checklist

---

# 1. Project in Plain Language

The project is a small optical instrument for screening apples and, later, other fruits. A person places an apple inside a dark measurement chamber. The chamber turns on several LEDs one at a time. An AS7265x sensor measures how the apple surface responds to each light source at 18 wavelength bands. Software calibrates those values, checks whether the scan is trustworthy, and uses machine learning to classify the result.

The device’s intended V1 output is not a legal food-safety certificate. It should show one of three outcomes:

1. **Untreated-like:** The scan resembles the control condition represented in the trained model.
2. **Treated-like:** The scan resembles the documented treatment condition represented in the trained model.
3. **Uncertain:** The device lacks enough evidence, the scan is poor quality, or the fruit is unlike the training data. Repeat the scan or obtain laboratory confirmation.

The long-term product vision can include apple weight, calorie estimation, external quality, bruise detection, ripeness estimates, sweetness estimates, and pesticide-related screening. However, these are separate measurement problems. They should be implemented as separate modules rather than claimed as one magical AI model.

---

# 2. The Real Problem

Pesticides are used to protect crops and maintain yield. Their use is not itself the problem. The problem is that residues can remain on food, and reliable residue testing is concentrated in laboratories.

Laboratory methods such as LC-MS/MS and GC-MS/MS can identify specific compounds and measure concentrations. They are appropriate for regulatory testing, export certification, and Maximum Residue Limit (MRL) decisions. But they require sample preparation, specialized instruments, trained analysts, consumables, and time.

At farm gates, mandis, cold-storage facilities, packhouses, small cooperatives, and retail intake points, routine laboratory testing of every fruit is impractical. Many decision points therefore have no preliminary screening at all.

This project addresses that gap. The aim is to make a low-cost preliminary checkpoint that can scan many samples and identify which samples need expensive confirmatory laboratory analysis.

The value proposition is not:

> Replace LC-MS/MS.

The value proposition is:

> Increase the number of screening checkpoints and route uncertain or suspicious samples to laboratory confirmation.

---

# 3. Why Existing Methods Are Not Enough

| Method | Strength | Limitation for field deployment |
|---|---|---|
| LC-MS/MS or GC-MS/MS | High analytical specificity and sensitivity | Laboratory-bound, expensive, destructive, slow |
| Hyperspectral imaging | Rich spectral information, many wavelengths | Camera and optical rig can be expensive and bulky |
| Portable Vis-NIR spectrometer | Useful for narrow crop–analyte tasks | Cost and task-specific calibration remain significant |
| RGB camera | Cheap and useful for visible defects | Only three broad bands; limited chemical specificity |
| Colorimetric test strips | Can be sensitive for targeted analytes | Consumables, reagents, often single-analyte |
| AS7265x multispectral sensor | Low-cost, compact, 18 spectral channels | Low spectral resolution; needs strong calibration and careful task scope |

The AS7265x device is not expected to equal a laboratory instrument. It is designed to be a lower-cost, portable screening layer.

---

# 4. The Product Vision and the V1 Scope

## Final product vision

A user places an apple into a device and receives a simple report:

```text
APPLE SCREEN REPORT

Weight: 178 g
Estimated calories: 93 kcal
External quality: Good
Visible defects: Low probability
Treatment-condition screen: Treated-like
Confidence: 0.86
Data quality: Passed
Recommendation: Laboratory confirmation recommended
```

This is a good product vision. But each item comes from a different module and has a different evidence requirement.




# 6. Core Scientific Question

The scientific question has evolved from a broad question—“Can AS7265x detect pesticides?”—to a more defensible research question:

> Can controlled multi-illumination optical measurements extract treatment-condition-related information from apple surfaces that is not available from ordinary low-cost reflectance sensing alone?

This can be split into smaller questions:

1. Is the AS7265x instrument stable and repeatable inside the chamber?
2. Do white, IR, UV365, and UV405 measurements behave reliably?
3. Is UV405 dominated by direct excitation leakage or does it contain useful apple-specific information?
4. Does controlled optical geometry improve repeatability or discrimination?
5. Does multi-mode sensing outperform white reflectance alone on unseen fruits?
6. Can a simple model classify a narrow binary physical task with useful sensitivity and specificity?
7. Can uncertainty rejection prevent false confidence on unusual scans?

---

# 7. System Architecture

```text
                 APPLE INSIDE DARK CHAMBER
                              |
        ------------------------------------------------
        |              |             |                |
      White          IR875         UV365            UV405
        |              |             |                |
        -------------------- illumination ----------------
                              |
                         AS7265x sensor
                              |
                 18 values per illumination mode
                              |
        ------------------------------------------------
        |             |              |                 |
      Dark          PTFE           Blank             Metadata
     reference     reference       chamber       gain, time, temp
        |             |              |                 |
        ------------------- calibration ------------------
                              |
               72-mode features + derived features
                              |
             Data quality / saturation / drift checks
                              |
                  PCA + LDA / regularized SVM
                              |
             confidence + OOD + rejection decisions
                              |
         Untreated-like / Treated-like / Uncertain
                              |
                 dashboard + CSV audit record
```

The architecture has four layers:

| Layer | Role |
|---|---|
| Physical sensing | LEDs, sensor, references, chamber, fruit holder |
| Edge control | Raspberry Pi or ESP32, drivers, MOSFETs, data logging |
| Machine learning | Calibration, feature engineering, PCA, LDA/SVM, OOD |
| Presentation | Display/dashboard, scan history, uncertainty explanation |

---

# 8. Hardware Components and Their Roles

| Component | Role | Why it is needed |
|---|---|---|
| AS7265x | Measures 18 spectral channels | Main optical detector |
| White LED | Broadband visible illumination | Standard reflectance baseline |
| 875 nm IR LED | Near-infrared illumination | Adds NIR response information |
| UV365 LED | UV excitation condition | Tests UV-response differences |
| UV405 LED | Second UV/violet excitation condition | Enables differential response experiments but has crosstalk risk |
| Constant-current LED drivers | Stable current supply | LED brightness must not drift unpredictably |
| MOSFET switches | Controlled LED switching | Pi GPIO cannot power LEDs directly |
| Raspberry Pi 5 | Controls device, logs data, runs dashboard | Easier development and sufficient compute |
| ESP32-S3 | Optional low-cost/compact alternative | Useful later for embedded deployment |
| PTFE tile | Diffuse white reference | Normalizes brightness and channel response |
| Dark cap | Light-blocked reference | Measures dark current and sensor offset |
| Matte-black chamber | Blocks room light and reduces reflection | Controls the measurement environment |
| Fixed fruit holder | Holds apple consistently | Reduces geometry variation |
| Temperature/humidity logging | Tracks environmental drift | Helps explain measurement variation |
| Camera, load cell, MQ135 | Optional Phase 2 modules | Product features, not V1 scientific core |

---

# 9. Why Each Illumination Mode Exists

## White illumination

White light gives a broad visible reflectance profile. It captures surface colour, broad pigment effects, and reflectance differences across the 18 sensor bands.

## IR875 illumination

Near-infrared response can vary with water content, tissue structure, and broad organic properties. It is not a magic pesticide detector, but it may provide complementary information.

## UV365 illumination

UV365 may trigger a different optical response from fruit skin, wax, organic materials, or certain chemical formulations. The actual usefulness must be measured.

## UV405 illumination

UV405 may provide a second excitation condition. Comparing UV365 and UV405 may be more informative than using either alone.

However, UV405 is risky because the AS7265x starts measuring close to 410 nm. Reflected LED light may dominate the sensor’s shortest channels.

## The multi-excitation fingerprint

Four modes create:

\[
4 \text{ modes} \times 18 \text{ channels} = 72 \text{ raw measurements}
\]

The aim is not simply to collect four separate spectra. It is to study how the surface response changes under controlled illumination changes.

---

# 10. The Optical Chamber

The chamber is one of the most important scientific components. It controls the environment between LEDs, fruit, and sensor.

It should:

- Block ambient light.
- Hold the fruit at repeatable distance and orientation.
- Hold the sensor at a repeatable position.
- Place LEDs at fixed, adjustable angles.
- Use matte-black internal walls.
- Avoid direct LED-to-sensor line of sight.
- Provide a location for PTFE reference measurements.
- Support a dark cap or dark measurement position.
- Keep UV light enclosed for safety.

## Why geometry matters

If fruit placement changes slightly, surface curvature may cause reflectance changes that are larger than the treatment-related signal. Geometry is therefore not cosmetic; it is part of the instrument.

Start with 30–45° illumination geometry, but treat it only as a starting hypothesis. Test multiple angles and distances.

---

# 11. Electrical Integration and Control

The controller must read sensor values and activate LEDs safely.

## Electrical rules

- Never power high-power LEDs directly from Pi GPIO.
- Use constant-current LED drivers.
- Use appropriate MOSFET switching circuits.
- Use common ground where electrically appropriate.
- Verify voltage levels and current limits.
- Add fuses or suitable protection.
- Prevent excessive LED heating.
- Keep sensor wiring short and stable where possible.
- Record LED current and sensor integration settings.

## Control sequence

The Pi or ESP32 sends commands:

```text
All LEDs OFF
      ↓
Dark reference read
      ↓
White LED ON → sensor read
      ↓
White OFF, IR ON → sensor read
      ↓
IR OFF, UV365 ON → sensor read
      ↓
UV365 OFF, UV405 ON → sensor read
      ↓
All LEDs OFF
      ↓
Save data
```

Add a small fixed settling delay after LED switching. Log that delay. Do not assume an LED and sensor stabilize instantaneously.

---

# 12. Scan Acquisition Workflow

A completed physical scan should proceed as follows.

1. Operator places apple in the fixed holder.
2. Operator closes the chamber.
3. System checks that lid/interlock is closed if available.
4. System reads dark reference.
5. System reads PTFE white reference.
6. System reads blank chamber when UV background checking is scheduled.
7. System activates white LED and records 18 channels.
8. System activates IR875 LED and records 18 channels.
9. System activates UV365 and records 18 channels.
10. System activates UV405 and records 18 channels.
11. System checks for saturation, missing channels, drift, and invalid references.
12. System calibrates readings.
13. System creates model features.
14. System runs classifier and uncertainty checks.
15. System displays result and saves raw + calibrated data.

For experimental work, use counterbalanced illumination order on a controlled subset. This checks whether heating or time order creates apparent effects.

---

# 13. Calibration Theory

Raw sensor counts are not directly comparable across sessions. They depend on LED brightness, sensor gain, integration time, distance, temperature, and geometry.

For each channel \(k\):

- \(S_k\): raw sample reading.
- \(D_k\): dark reading.
- \(W_k\): PTFE white-reference reading.

Calculate a calibrated response:

\[
R_k = \frac{S_k-D_k}{W_k-D_k+\epsilon}
\]

where \(\epsilon\) is a very small number to prevent division by zero.

This does not make every sensor identical or eliminate all variation. It is the first required normalization step.

## Blank chamber for UV

For UV modes, collect an empty-chamber background:\n
\[
B_{UV,k}
\]

Then calculate:

\[
U_k = (S_{UV,k}-D_k) - (B_{UV,k}-D_k)
\]

This reduces chamber-wall reflection and LED leakage background. It does not prove fluorescence emission.

---

# 14. The 405 nm Crosstalk Problem

This is the central optical risk.

The UV405 LED emits near 405 nm. The AS7265x’s first detection region begins near 410 nm. If the LED’s excitation light reaches the sensor directly or reflects strongly from the fruit/chamber, the sensor can mostly measure excitation leakage.

The measurement may become:

\[
\text{UV405 sensor signal} \approx \text{reflected LED excitation}
\]

instead of:

\[
\text{UV405 sensor signal} \approx \text{fruit-specific emission response}
\]

## Mitigation order

1. Increase LED-to-sensor separation angle.
2. Increase or optimize sensor-to-fruit/LED-to-fruit distance.
3. Add a matte-black baffle or light trap.
4. Avoid direct line of sight from LED to sensor.
5. Reduce LED current or exposure if saturation occurs.
6. Test a suitable optical long-pass or band-pass filter if needed.
7. Use blank subtraction and differential features.

## Why it may become research novelty

If a specific optical arrangement reduces bleed-through and improves signal-to-background ratio, repeatability, or held-out classification relative to ordinary reflectance, then you have a real engineering contribution.

---

# 15. Hardware Bring-Up Protocol

Do this with zero pesticide first.

## Step 1: communication

- Verify the sensor is visible on I2C.
- Confirm all 18 channels can be read.
- Confirm repeated reads do not fail.
- Save a test CSV.

## Step 2: individual LED test

For each LED separately:

- Turn it on and off.
- Verify other LEDs remain off.
- Read all 18 channels.
- Check sensor saturation.
- Check LED heat.
- Check output stability.

## Step 3: no-fruit optical matrix

| Test object | Modes | Purpose |
|---|---|---|
| Dark cap | All off | Sensor dark-current baseline |
| Empty chamber | White | Chamber reflection background |
| Empty chamber | IR | IR background |
| Empty chamber | UV365 | UV365 leakage/background |
| Empty chamber | UV405 | Critical UV405 bleed-through baseline |
| PTFE tile | All modes | Reference response and saturation |
| Black matte target | All modes | Low-reflectance control |
| Ordinary apple | All modes | Initial fruit response |

## Step 4: repeatability

- Scan one apple ten times without moving it.
- Scan one apple ten times with repositioning.
- Calculate mean, standard deviation, and coefficient of variation per channel.
- Repeat across days.

---

# 16. GO / NO-GO Gate

Before you spend time and money on treatment-condition data collection, make a decision.

| GO: proceed | NO-GO: redesign optics |
|---|---|
| No important channels saturate | UV405 saturates sensor channels |
| Dark readings stable | Dark readings drift strongly |
| PTFE response repeatable | PTFE response shifts excessively |
| Empty-chamber UV response documented and small | Empty chamber resembles apple response |
| Unmoved/repositioned variation acceptable | Placement variation dominates signal |
| Chamber blocks ambient light | Chamber leaks ambient light |
| LED output stable | LED output drifts during session |
| Calibration valid across sessions | Calibration is inconsistent |

A NO-GO outcome is useful. It saves you from collecting a large dataset with invalid measurements.

---

# 17. Existing Dataset: Roomi Apple HSI

The current software baseline uses a public hyperspectral apple dataset associated with Roomi et al. The underlying data includes hyperspectral apple images with many wavelength bands.

Key facts used in the project:

| Property | Value |
|---|---|
| Native spectral resolution | 281 bands, roughly 400–1000 nm |
| Images/hypercubes available | 617 in the source dataset |
| Synthetic rows used in project | 337 |
| Independent apples underlying current subset | 10 |
| Labels | Control, Nativo/Monostar treatment family, low/high treatment level |
| Current use | Synthetic AS7265x-like reflectance baseline |

The important point is that 337 rows do not mean 337 independent apples. The independent biological unit is the fruit.

---

# 18. From Hyperspectral Image to 18 Sensor Values

A hyperspectral image has spatial and spectral dimensions:

\[
\text{height} \times \text{width} \times 281\text{ wavelengths}
\]

Each pixel may have a 281-value spectrum. To create an AS7265x-like sample:

1. Select apple pixels and remove background.
2. Choose a region of interest.
3. Average or summarize pixel spectra into one 281-value spectrum.
4. Apply a channel-response function for each AS7265x band.
5. Produce an 18-value synthetic vector.

For sensor channel \(k\):

\[
V_k = \frac{\sum_i R(\lambda_i)G_k(\lambda_i)}{\sum_i G_k(\lambda_i)}
\]

where:

- \(R(\lambda_i)\) is the source hyperspectral reflectance.
- \(G_k(\lambda_i)\) is an approximate response curve of AS7265x channel \(k\).
- \(V_k\) is synthetic channel value \(k\).

This is how a full hyperspectral image becomes a row in a CSV with 18 columns.

---

# 19. Synthetic Dataset: What It Proves and Does Not Prove

## It proves

- The 18-channel schema is usable in software.
- GroupKFold can be applied correctly.
- PCA+LDA is a defensible baseline for small independent sample count.
- The model can find some treatment-condition-related signal in downsampled HSI data.
- Narrower tasks are easier than broad three-class tasks.
- Confidence rejection can improve performance among accepted samples.

## It does not prove

- Real AS7265x sensor accuracy.
- Real chamber repeatability.
- Real LED behavior.
- UV fluorescence response.
- UV405 leakage control.
- Pesticide concentration estimation.
- MRL classification.
- Real market performance.
- 90% physical-device accuracy.

The synthetic model is a useful baseline and software rehearsal. It is not the final product model.

---

# 20. Software Pipeline Overview

Current synthetic pipeline:

```text
Synthetic 18-channel CSV
          |
          v
Data validation
          |
          v
StandardScaler
          |
          v
PCA: 5 principal components
          |
          v
Linear Discriminant Analysis
          |
          v
Fruit-wise GroupKFold evaluation
          |
          v
Metrics, confusion matrix, OOD, report
```

Future real-hardware pipeline:

```text
Raw AS7265x readings in all illumination modes
          |
          v
Dark/PTFE/blank calibration
          |
          v
Quality flags and saturation checks
          |
          v
72-channel per-scan feature table
          |
          v
Differential feature creation
          |
          v
Fruit-wise train/validation/blind split
          |
          v
PCA+LDA / regularized SVM
          |
          v
Confidence + OOD + rejection
          |
          v
Untreated-like / treated-like / uncertain
```

---

# 21. Data Loading and Schema

## Synthetic CSV

The synthetic data uses columns similar to:

```text
filename
fruit_id
treatment_family
level
class_3
ch_410 ... ch_940
```

The 18 channels must always be in the same order.

## Real hardware raw long-format CSV

One row per illumination mode:

```text
scan_id, fruit_id, batch_id, session_id, timestamp, mode,
gain, integration_time_ms, led_current_ma,
sensor_temperature_c, chamber_temperature_c, humidity_pct,
dark_reference_id, white_reference_id, blank_reference_id,
saturation_flag, fruit_position, fruit_orientation, operator,
treatment_condition, treatment_family, treatment_level,
ch_410 ... ch_940
```

## Real hardware model wide-format CSV

One row per completed scan:

```text
scan_id, fruit_id, label,
white_ch_410 ... white_ch_940,
ir_ch_410 ... ir_ch_940,
uv365_ch_410 ... uv365_ch_940,
uv405_ch_410 ... uv405_ch_940
```

Never overwrite raw data. Save calibrated and derived data in separate files.

---

# 22. Preprocessing and the SNV Result

Current baseline uses StandardScaler:

- Each feature is centered around mean zero.
- Each feature is scaled to unit variance.

An ablation on synthetic data found:

| Preprocessing | Mean CV accuracy |
|---|---:|
| Raw + StandardScaler | 65.57% |
| SNV normalized | 52.57% |
| SNV + Savitzky–Golay derivative | 52.27% |

SNV reduced the synthetic performance. A likely reason is that it removes overall brightness magnitude, and that magnitude contained treatment-condition-related information in the pre-calibrated synthetic data.

However, do not conclude “SNV is bad.” Real hardware introduces placement and scattering effects that SNV may help correct. Re-test preprocessing on physical data.

---

# 23. PCA: Why It Is Used

AS7265x channels are correlated. Nearby wavelengths often change together. If you train directly on many correlated features with only a few independent fruits, the model can overfit.

PCA converts correlated channels into a smaller number of principal components.

Current baseline:

\[
18 \text{ channels} \rightarrow 5 \text{ principal components}
\]

For real 72-channel data, do not blindly keep five components. Test:

- PCA over all modes.
- Separate PCA per illumination mode.
- Different component counts.
- Feature subsets.

Choose inside cross-validation only.

---

# 24. LDA: Why It Won

LDA finds boundaries that separate classes by maximizing between-class separation relative to within-class variation.

It is suitable for small data because it has a simpler assumption than large tree ensembles or neural networks.

The current evidence is not that LDA is universally best. It is that LDA is best among models tested on this small synthetic dataset.

---

# 25. Model Benchmarking Results

| Model | Mean accuracy | Balanced accuracy | Macro-F1 | Interpretation |
|---|---:|---:|---:|---|
| PCA + LDA | 65.57% ± 5.34% | 65.01% | 62.37% | Best synthetic baseline |
| SVM (RBF) | 57.64% ± 5.51% | 55.70% | 54.30% | Comparison model |
| Random Forest | 46.49% ± 8.35% | 44.72% | 43.40% | High variance / likely overfit |
| XGBoost | 47.09% ± 9.72% | 45.39% | 44.71% | High variance / likely overfit |

The tree models had both lower mean accuracy and higher fold variance. This is consistent with too few independent fruits for complex flexible models.

---

# 26. GroupKFold and Leakage Prevention

The true independent unit is a fruit, not a scan.

Wrong approach:

```text
Apple A001 scan 1 → training
Apple A001 scan 2 → test
```

This leaks fruit identity.

Correct approach:

```text
All scans from Apple A001 → training OR test
Never both
```

GroupKFold groups by `fruit_id`. The current synthetic evaluation splits 10 apples into five folds, with two whole apples held out in each fold.

This is one of the strongest methodological parts of the project.

---

# 27. Narrower Binary Tasks

The synthetic experiments showed that task scope strongly affects performance.

| Synthetic task | Approximate mean accuracy |
|---|---:|
| Control / Low / High three-class | 65.57% |
| Control versus treated | ~77% |
| Low versus high | ~79–81% |
| Control versus Nativo-high | ~89% |
| Control versus Monostar-high | ~88% |

This does not mean the physical device will get these numbers. It means the data supports a strategic decision:

> Start with one narrow binary task before attempting broad multi-class classification.

---

# 28. Confidence Rejection and Coverage

A classifier always returns a class, even when it should not. A screening device should have permission to reject uncertain scans.

Define confidence as the largest predicted probability:

\[
\text{confidence} = \max(P(\text{class}_i))
\]

If confidence is below a selected threshold, the device can output uncertain.

Synthetic examples from narrow tasks showed higher accepted-scan accuracy at lower coverage. For example, a high-confidence result may show 97% accepted-scan accuracy but accept only 66% of scans.

Therefore always report both:

```text
Balanced accuracy on accepted scans: 94%
Coverage: 85%
Rejection rate: 15%
```

Never report only “94% accuracy.”

---

# 29. Mahalanobis Out-of-Distribution Detection

Mahalanobis distance estimates how unusual a scan is relative to training distribution, accounting for correlated features.

\[
D_M(x)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}
\]

Where:

- \(x\) is the transformed scan.
- \(\mu\) is training mean.
- \(\Sigma\) is training covariance.

Current synthetic threshold reference:

\[
D_M = 4.282
\]

This is not a physical-device threshold. Recalculate from real hardware training data.

The final uncertainty logic can combine:

- Low classifier confidence.
- High Mahalanobis distance.
- Saturated channels.
- Calibration failure.
- Reference drift.
- Missing modes.
- Model disagreement.

---

# 30. Digital Twin / Spectral Reconstruction

The project includes a model that maps 18 channels into an approximate 281-band spectral curve.

Current implementation:

```text
18 channels
   ↓
MLPRegressor: 64 → 32 → 64 hidden layers
   ↓
281 reconstructed values
```

Current results:

| Dataset | Reconstruction R² | RMSE |
|---|---:|---:|
| Roomi apple treatment dataset | 0.5546 | 0.03264 |
| Cucurbitaceae benchmark | 0.8654 | 0.0147 |

The apple result is moderate. Therefore, use the careful term:

> Continuous spectral reconstruction model.

Use it for:

- Dashboard visualization.
- Approximate curve display.
- Anomaly support.
- Domain adaptation research.
- Small feature enhancement.

Do not claim it recreates actual hyperspectral data or replaces a hyperspectral camera.

---

# 31. Real Hardware Dataset Plan

## Pilot A: controls only

Start with 10 untreated apples.

- One variety where possible.
- Two acquisition days.
- Three to five positions per apple.
- Multiple scans at each position.
- Record all metadata.

Goals:

- Confirm repeatability.
- Check reference drift.
- Check batch/day variation.
- Check placement sensitivity.
- Check UV background.

## Pilot B: small known-treatment-condition study

After safety approval:

- 5 controls.
- 5 documented treatment-condition apples.
- Same apple variety where possible.
- Same chamber and protocol.

This is a sanity check, not a final accuracy study.

## Main V1 dataset

| Group | Minimum | Better |
|---|---:|---:|
| Untreated apples | 30–50 independent fruits | 50–100 |
| Known-treatment-condition apples | 30–50 independent fruits | 50–100 |
| Positions per fruit | 3–5 | 5–8 |
| Acquisition days | 2–3 | 3+ |
| Batches/sources | 2–3 | 3+ |
| External blind batch | Separate source/day | Strongly preferred |

---

# 32. Confounder Controls

The model may learn non-chemical shortcuts. Test them deliberately.

| Control condition | What it tests |
|---|---|
| Normal dry apple | True baseline |
| Washed apple | Surface wax / washing effect |
| Surface-moist apple | Moisture confusion |
| Handled apple | Handling/contact effect |
| Different batch | Source/batch shortcut |
| Different ripeness | Biological variability |
| Different orientation | Geometry sensitivity |

If washed or wet apples are consistently called treated-like, the model is learning a confounder. Do not hide that result; redesign the experiment.

---

# 33. Real-Data Feature Engineering

Create features only after calibration.

## Raw mode features

- `white_ch_410 ... white_ch_940`
- `ir_ch_410 ... ir_ch_940`
- `uv365_ch_410 ... uv365_ch_940`
- `uv405_ch_410 ... uv405_ch_940`

## Derived features

\[
\Delta F_k = U_{365,k} - U_{405,k}
\]

\[
Q_{365,k} = \frac{U_{365,k}}{R_{white,k}+\epsilon}
\]

\[
Q_{405,k} = \frac{U_{405,k}}{R_{white,k}+\epsilon}
\]

Other useful non-chemical features:

- Reference drift score.
- Mean/standard deviation across repeated scans.
- Per-channel coefficient of variation.
- Saturation count.
- Position-quality flag.
- Time since calibration.

Run ablations:

| Experiment | Feature set |
|---|---|
| A | White only |
| B | White + IR |
| C | White + UV365 |
| D | White + UV405 |
| E | UV365 + UV405 |
| F | All modes |
| G | All modes + differential features |
| H | Best configuration + baffle/filter/optimized geometry |
| I | Best configuration + uncertainty rejection |

---

# 34. Real-Data Model Training

For V1, use simple models first.

## Primary models

1. StandardScaler + PCA + LDA.
2. StandardScaler + regularized SVM.
3. Optional modest tree benchmark.

## Why not deep learning first

At 30–50 independent fruits per class, deep learning can easily memorize details. The number of scan rows is not the number of independent fruits.

Use this guide:

| Independent fruits | Model recommendation |
|---:|---|
| 10–50 | PCA+LDA, regularized SVM |
| 50–150 | PCA+LDA/SVM; small tree benchmark |
| 150–500 | Compare a small 1D-CNN with classical models |
| 500+ | CNN/transfer learning may be practical |

Do not use GANs to create fake biological diversity from a tiny physical dataset.

---

# 35. Final Validation Protocol

Use a two-stage approach.

## Development stage

Within development data, use GroupKFold by fruit ID to compare:

- Feature sets.
- PCA component counts.
- Preprocessing.
- Models.
- Confidence thresholds.
- OOD thresholds.

## Locked final test

Set aside fruit IDs before final model choice.

| Partition | Purpose |
|---|---|
| 70% fruit IDs | Model development/training |
| 15% fruit IDs | Validation and threshold tuning |
| 15% fruit IDs | Locked blind final test |
| Separate batch/day | External robustness test |

Never change features, preprocessing, thresholds, or model after seeing blind-test performance.

## Metrics

Report:

- Accuracy.
- Balanced accuracy.
- Sensitivity/recall for treated condition.
- Specificity.
- Macro-F1.
- Precision.
- Confusion matrix.
- Coverage.
- Rejection rate.
- Accepted-scan performance.
- External batch result.
- Confidence intervals.

---

# 36. Path to 90% Accuracy

## What 90% should mean

A meaningful target is:

> At least 90% balanced accuracy or macro-F1 on an independent fruit-wise held-out physical test set for one narrow binary task, with coverage and rejection rate reported.

## What it should not mean

- Training accuracy.
- Random scan split accuracy.
- Synthetic-only accuracy.
- One cherry-picked threshold.
- 90% after rejecting most samples.
- A broad multi-fruit/multi-pesticide universal claim.

## Realistic stages

| Stage | Status |
|---|---|
| Synthetic 3-class baseline | 65.57% measured |
| Synthetic narrow binary | ~88–89% measured |
| Real reflectance binary | Unknown; must be measured |
| Stable chamber and geometry | Unknown; must be measured |
| UV-excitation features | Unknown; may help or not |
| 30–50+ real fruits/class | Required for meaningful estimate |
| Rejection-aware physical screening | 90% accepted-scan performance may be plausible, not guaranteed |
| LC-MS/MS linked study | Required for residue/MRL claims |

The five main accuracy levers are:

1. Better real measurement quality.
2. Narrower task definition.
3. More independent fruits.
4. Useful multi-mode/differential features.
5. Honest rejection of unreliable scans.

---

# 37. Product Modules Beyond Screening

Your professor may want a consumer-style full report. Build it as modules.

| Module | Inputs | Output | Evidence requirement |
|---|---|---|---|
| Weight | Load cell | grams | Compare to weighing scale |
| Calories | Weight + nutrition database | estimated kcal | Label as estimate, variety dependent |
| External defects | Pi camera | bruise/spot/defect score | Labelled image data |
| Ripeness | Camera + spectra | ripeness category | Real labelled fruit dataset |
| Brix/sweetness | Spectra | estimated °Brix | Refractometer labels |
| Treatment-condition screening | AS7265x + chamber | untreated-like / treated-like / uncertain | Real physical validation |
| Residue concentration | Spectra + reference lab | estimated mg/kg | LC-MS/MS/GC-MS/MS labels |
| MRL screening | crop + pesticide + measured residue | threshold decision | Full validated regulatory study |

Calories should be estimated from weight, not “measured” by AS7265x.

\[
\text{calories} = \frac{\text{weight in g}}{100} \times \text{kcal per 100g reference}
\]

---

# 38. Literature Survey and Prior Art

The literature review covers:

- Reference analytical methods.
- Vis-NIR/chemometric pesticide studies.
- Fluorescence and photo-induced fluorescence methods.
- Smartphone colorimetry.
- Low-cost multispectral ICs.
- Digital-twin/food-system work.
- TinyML/edge AI.
- Public hyperspectral datasets.

## Important prior-art correction

The project originally claimed that AS7265x had not been applied to pesticide screening. That claim is not safe. A 2024 study by Yaemsuk and Yammen reported a portable NIR spectrometer using AS7265x and ESP32 for pesticide detection on basil and chili, including carbendazim, cypermethrin, diazinon, and imidacloprid. This means generic “AS7265x + AI for pesticide detection” is not a strong novelty claim.[web:385]

## Correct research gap

A safer gap statement is:

> Existing low-cost multispectral pesticide-screening studies demonstrate promising controlled classification, but controlled multi-illumination optical geometry, excitation-bleed-through mitigation, fruit-wise independent validation, uncertainty-aware rejection, synthetic-to-real adaptation, and laboratory-grounded validation remain insufficiently integrated into a low-cost field-screening platform.

---

# 39. Novelty and Patent Strategy

## Not strong patent territory

- AS7265x used for pesticide screening.
- PCA+LDA classifier.
- GroupKFold methodology.
- Digital reconstruction algorithm alone.
- Generic “AI detects pesticides.”

## Potential stronger territory

A demonstrated physical measurement system with:

- Controlled dual-UV excitation.
- Specific sensor/LED geometry.
- Baffle or light-trap structure.
- Filter arrangement if needed.
- Dark/PTFE/blank calibration sequence.
- Differential multi-excitation feature extraction.
- Measurable technical effect.

The technical effect should be measured, for example:

- Reduced UV405 blank-chamber leakage.
- Lower saturation.
- Better signal-to-background ratio.
- Lower repositioning variation.
- Higher held-out performance than white-only sensing.

## Public disclosure warning

The project repository is public. Do not assume a public GitHub disclosure automatically preserves full patent rights. Speak with the college IP cell or a registered patent professional before publicly posting new optical geometry, filter, baffle, or experimental-result details. Indian patentability depends on novelty and inventive step; the exact disclosure history matters.[web:445]

---

# 40. Funding, Publication, Timeline, and Master Checklist

## Funding

Potential routes include institutional innovation cells, agricultural incubators, RKVY-RAFTAAR pathways, and possibly NIDHI PRAYAS depending on exact eligibility.

NIDHI PRAYAS allows individual innovators and early startups, but its guidelines also exclude funding research or student internships in academic/R&D institutions. Before investing time, ask the specific incubator whether you can apply as an individual innovator with mentor/facility support and clear IP ownership rather than as an academic internship project.[web:373][web:453]

NIDHI PRAYAS 2.0 publicly describes prototype support tiers through PRAYAS Centres and Advance PRAYAS Centres; confirm the active rules and amount with the incubator handling the application.[web:392]

## Publication

A publishable paper does not require a 99% result. It requires a rigorous question and honest evidence.

A strong paper could include:

1. Low-cost chamber design.
2. Dark/PTFE/blank calibration.
3. UV405 bleed-through measurement.
4. Geometry/baffle/filter ablation.
5. White versus IR versus UV response comparison.
6. Real independent-fruit dataset.
7. GroupKFold and blind test.
8. Confidence/coverage reporting.
9. Confounder experiments.
10. Honest negative findings.

## Timeline after parts arrive

### Week 1

- Assemble Pi/ESP32 and AS7265x.
- Verify I2C.
- Read and log 18 channels.
- Test one LED.

### Week 2

- Add all LEDs and drivers.
- Build chamber.
- Run dark/blank/PTFE/black-target tests.
- Check UV405 saturation and leakage.

### Week 3

- Repeatability tests.
- Placement tests.
- Reference drift checks.
- Geometry experiments.

### Week 4

- Control-only pilot: 10 apples over two days.
- Audit data schema and calibration.

### Month 2

- Safety clearance.
- Small known-treatment-condition pilot.
- Confounder tests.
- GO/NO-GO review.

### Month 3 onward

- Main independent-fruit dataset.
- GroupKFold development.
- Locked blind test.
- Lab-partner measurements where possible.
- Publication/funding/IP decision based on real evidence.

## Master checklist

### Before hardware arrives

- [ ] Freeze public disclosure of novel optical details until IP guidance.
- [ ] Prepare Python environment and data logger.
- [ ] Prepare scan CSV schema.
- [ ] Prepare hardware test forms.
- [ ] Contact faculty safety mentor.
- [ ] Contact possible laboratory partner.
- [ ] Contact incubator about eligibility.

### On hardware arrival

- [ ] Verify I2C communication.
- [ ] Verify all 18 channels.
- [ ] Test LEDs independently.
- [ ] Build chamber.
- [ ] Capture dark/PTFE/blank readings.
- [ ] Check saturation.
- [ ] Run UV405 crosstalk test.
- [ ] Measure repeatability.

### Before treatment data

- [ ] Pass GO/NO-GO gate.
- [ ] Obtain pesticide handling approval.
- [ ] Establish disposal procedure.
- [ ] Define one narrow treatment condition.
- [ ] Lock logging protocol.

### Before any accuracy claim

- [ ] Use real hardware data.
- [ ] Group by fruit ID.
- [ ] Use blind holdout fruits.
- [ ] Report sensitivity, specificity, balanced accuracy, macro-F1.
- [ ] Report coverage and rejection rate.
- [ ] State exact task and dataset.
- [ ] Do not make MRL/safety claims without laboratory labels.

## Final principle

> Do not make the AI smarter before making the measurement better.

The project succeeds if it becomes a repeatable, calibrated, uncertainty-aware optical screening platform with honest independent-fruit validation. Accuracy, novelty, product usefulness, research publication, funding, and potential IP all become more achievable after that foundation is real.
