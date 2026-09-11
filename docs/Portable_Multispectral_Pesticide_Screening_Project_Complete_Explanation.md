# Portable Multispectral UV-Excitation Edge-AI Device for Fresh-Produce Screening
## Complete Project Explanation, Rationale, Findings, and Execution Plan

**Project status:** Software baseline completed; physical hardware integration pending

**Primary V1 task:** Distinguish untreated apples from apples with a known, documented treatment condition using a controlled low-cost optical system. This is not yet a certified pesticide-residue, MRL, or food-safety device.

---

## 1. Project in one paragraph

This project develops a low-cost, portable instrument that scans an apple under carefully controlled light and provides a cautious screening result. The device uses an AS7265x multispectral sensor to measure the apple surface at 18 wavelengths between approximately 410 nm and 940 nm. It illuminates the fruit sequentially with white light, 875 nm infrared light, UV365, and UV405. A Raspberry Pi controls the lights, reads the sensor, calibrates the data, saves every scan, runs a machine-learning model, and shows a result.

The intended product output is not “safe” or “unsafe.” The scientifically honest V1 output is:

1. **Untreated-like**
2. **Treated-like**
3. **Uncertain — repeat the scan or obtain laboratory confirmation**

The long-term vision is a field-screening platform for farms, packhouses, cold storage, procurement centres, markets, and agricultural laboratories. It would increase the number of low-cost checkpoints in the supply chain and identify samples that deserve confirmatory LC-MS/MS or GC-MS/MS testing.

---

## 2. The problem

### 2.1 Why pesticide screening matters

Pesticides help farmers protect crops and maintain yield, but residues can remain on fruit after harvest. Regulatory agencies define Maximum Residue Limits (MRLs) for specific pesticide–crop combinations. Determining whether a fruit is truly below or above an MRL requires reliable laboratory methods and carefully controlled analytical procedures.

The problem is not that high-quality analytical methods do not exist. They do.

- **LC-MS/MS** and **GC-MS/MS** are reference-grade methods for chemical-residue analysis.
- They can identify and quantify many compounds.
- They are used for food safety, exports, and regulatory testing.

However, these methods are not easy to deploy at every place fruit changes hands. They require laboratory infrastructure, extraction/sample preparation, trained staff, reagents, time, and money.

### 2.2 The cost–accuracy–portability gap

| Method | Strength | Main limitation |
|---|---|---|
| LC-MS/MS or GC-MS/MS | High chemical specificity and quantification | Expensive, destructive, laboratory-bound, slow |
| Hyperspectral camera | Hundreds of spectral bands, non-destructive | Expensive and bulky optical setup |
| Portable Vis-NIR spectrometer | Useful for narrow crop–analyte tasks | Still costly; often narrow and controlled |
| RGB camera | Cheap and portable | Mainly sees colour/appearance, not detailed chemical spectra |
| Proposed AS7265x system | Low-cost, portable, multi-illumination screening | Limited spectral resolution; needs rigorous validation |

The project attempts to occupy the low-cost, field-deployable region of this trade-off. It does **not** try to replace laboratory testing. It tries to give a preliminary signal that helps decide which samples should be sent to a laboratory.

---

## 3. The solution

### 3.1 The key idea

An apple is placed inside a dark chamber. The device shines different types of light onto it. The apple reflects and possibly emits light differently depending on its surface, pigments, water, wax, treatment condition, damage, and other physical properties.

The AS7265x sensor converts the light it receives into numerical values.

```text
Apple inside chamber
        ↓
Controlled illumination
        ↓
Apple optical response
        ↓
AS7265x reads 18 wavelengths
        ↓
Calibration and quality checks
        ↓
Machine-learning model
        ↓
Untreated-like / Treated-like / Uncertain
```

### 3.2 Why controlled illumination matters

A sensor reading is affected by more than chemistry. It can change because of:

- Ambient room light
- Fruit distance from sensor
- Fruit surface angle
- Apple size and curvature
- LED brightness
- Sensor gain and integration time
- LED temperature
- Sensor temperature
- Surface moisture
- Wax coating
- Bruises or damage

Therefore, the system needs a closed optical chamber, fixed sample placement, stable LED current, dark reference, white reference, blank chamber reference, and a complete data log.

---

## 4. V1 scope and honest terminology

### 4.1 The correct V1 question

The first validated physical question should be:

> Can a controlled multi-illumination AS7265x optical system distinguish untreated apples from apples with a known treatment condition on completely unseen fruits?

### 4.2 Why not call it pesticide detection immediately?

A treatment-condition signal can arise from more than the active pesticide molecule. If an apple was sprayed or dipped, the sensor may respond to:

- The active ingredient
- Solvent or carrier liquid
- Surfactants and formulation additives
- Surface wetness
- Wax disturbance
- Drying effect
- Uneven residue distribution
- Handling differences

Therefore, before matched laboratory analysis, the correct term is **treatment-condition classification**, not confirmed pesticide-residue detection.

### 4.3 Terminology rules

| Do not say yet | Say instead |
|---|---|
| Detects pesticide residues | Screens for treatment-related spectral signatures |
| Detects every pesticide | Validated only for a specific V1 task |
| Safe to eat / unsafe | Preliminary screening only; laboratory confirmation recommended |
| Below MRL / above MRL | Untreated-like / treated-like until laboratory-validated |
| Pesticide percentage | No quantitative residue claim without analytical ground truth |
| Digital twin reconstructs the actual hyperspectral image | Continuous spectral reconstruction model; support/visualization tool |
| Replaces LC-MS/MS | Increases screening checkpoints and routes samples to confirmation |
| Fluorescence signal | UV-excitation response until optical bleed-through is ruled out |

---

## 5. Hardware architecture

### 5.1 Core hardware

| Component | What it does | Why it is included |
|---|---|---|
| AS7265x sensor | Measures light in 18 spectral channels | Low-cost VIS-NIR spectral measurement |
| White LED | Broadband visible illumination | Baseline reflectance response |
| 875 nm IR LED | Near-infrared illumination | Adds information related to water and broad organic/surface properties |
| UV365 LED | UV excitation condition | Tests whether this excitation gives additional useful response |
| UV405 LED | Second UV/violet excitation condition | Enables differential excitation-response study; high crosstalk risk |
| Raspberry Pi 5 | Controls LEDs, sensor, logging, model, dashboard | Easier development and standalone demonstration |
| LED drivers | Supply controlled current | LED brightness must be stable |
| MOSFETs | Switch LEDs on/off from GPIO | Enables time-multiplexed scan sequence safely |
| PTFE white tile | Known diffuse bright reference | Used to normalize optical measurements |
| Dark cap | Blocks light from detector | Measures dark current/offset |
| Dark chamber | Blocks ambient light and controls geometry | Improves repeatability |
| Fruit cradle/holder | Holds apple in same position | Reduces placement variation |

### 5.2 Optional Phase 2 hardware

The following can make the final product more useful, but should not be part of the first scientific classifier:

- Pi camera: apple presence, placement, visible defect/bruise module
- Load cell: apple weight and calorie estimate
- Touchscreen: user interface
- Temperature/humidity sensor: environment monitoring
- MQ135 gas sensor: not recommended for the first pesticide study; may be explored only as a separate freshness proxy

### 5.3 Why Raspberry Pi first?

The Raspberry Pi 5 is appropriate for initial development because it can:

- Run Python easily
- Read I2C sensors
- Control GPIO/MOSFETs
- Save CSV logs
- Run scikit-learn models
- Host a dashboard
- Interface with camera and touchscreen

An ESP32-S3 can become a lower-cost, lower-power future version after the optical system and model are validated.

---

## 6. How the optical scan works

### 6.1 Four illumination modes

Each complete scan has four modes.

| Mode | What is measured | Purpose |
|---|---|---|
| White | Visible reflectance | Baseline colour and surface reflectance |
| IR875 | Near-IR reflectance | Adds NIR response |
| UV365 | UV-excitation response | Tests whether UV365 adds useful information |
| UV405 | UV-excitation response | Tests differential response; requires crosstalk control |

The AS7265x produces 18 values for each mode. Therefore, a complete scan can provide:

\[
18\ \text{channels} \times 4\ \text{modes} = 72\ \text{raw measurements}
\]

### 6.2 Basic scan sequence

1. Place apple in fixed holder.
2. Close chamber.
3. Capture dark reference.
4. Capture PTFE white reference.
5. Capture blank-chamber reference where required.
6. Turn on white LED and record 18 channels.
7. Turn on IR LED and record 18 channels.
8. Turn on UV365 LED and record 18 channels.
9. Turn on UV405 LED and record 18 channels.
10. Save raw data and metadata.
11. Calibrate readings.
12. Run data-quality checks.
13. Run ML model only if data quality passes.
14. Show result and record it in audit log.

### 6.3 Why time multiplexing?

Only one illumination mode should be active at a time.

```text
White ON → read sensor → White OFF
IR ON → read sensor → IR OFF
UV365 ON → read sensor → UV365 OFF
UV405 ON → read sensor → UV405 OFF
```

This lets the software separate the fruit response under each excitation condition.

---

## 7. Calibration

### 7.1 Why raw values cannot go directly into ML

Raw AS7265x values are affected by sensor gain, integration time, LED brightness, temperature, distance, chamber reflection, and other instrument effects.

You need to calibrate the physical device before ML.

### 7.2 Dark reference

For each channel \(k\):

\[
D_k = \text{dark-cap reading}
\]

The dark cap blocks light. This captures dark current and electronic offset.

### 7.3 PTFE white reference

\[
W_k = \text{PTFE white-tile reading}
\]

The PTFE tile gives a known bright diffuse reference.

### 7.4 Apple reading

\[
S_k = \text{raw apple reading}
\]

### 7.5 Calibrated reflectance-like response

\[
R_k = \frac{S_k - D_k}{W_k - D_k + \epsilon}
\]

Where \(\epsilon\) is a very small number to avoid division by zero.

In plain language:

1. Subtract sensor darkness.
2. Compare fruit response against a known white reference.
3. Produce a normalized value that is more comparable across sessions.

### 7.6 Blank-chamber reference

For UV modes, also measure the chamber with LEDs on but no apple present.

\[
B_{UV,k} = \text{blank-chamber UV reading}
\]

A first blank-corrected UV value can be:

\[
U_k = (S_{UV,k}-D_k) - (B_{UV,k}-D_k)
\]

This removes some background from chamber walls, direct LED leakage, and optical reflections.

It does **not** prove fluorescence. It only gives a cleaner UV-excitation measurement.

---

## 8. The 405 nm crosstalk problem

The AS7265x shortest channel is near 410 nm. The UV405 LED emits very close to that range.

The risk is:

\[
\text{measured UV405 signal} \approx \text{reflected 405 nm excitation light}
\]

rather than:

\[
\text{measured UV405 signal} \approx \text{fruit-specific emission response}
\]

This is called excitation bleed-through or crosstalk.

### 8.1 Why it matters

If direct light leakage dominates, your ML model may learn:

- Chamber reflections
- Fruit position
- LED brightness
- Surface gloss
- Direct violet reflection

rather than treatment-related optical information.

### 8.2 Mandatory no-pesticide test matrix

| Test object | LED condition | What it tells you |
|---|---|---|
| Dark cap | All LEDs off | Dark-current baseline |
| Empty chamber | White | Chamber reflection background |
| Empty chamber | IR | IR leakage/background |
| Empty chamber | UV365 | UV365 optical background |
| Empty chamber | UV405 | Critical UV405 bleed-through baseline |
| PTFE tile | All modes | Reference response and saturation behavior |
| Black matte target | All modes | Low-reflectance control |
| Ordinary apple | All modes | Initial fruit response |

### 8.3 Crosstalk mitigation options

Test these in order:

1. Change LED angle and sensor angle.
2. Increase separation between LED and detector.
3. Add matte-black baffle/light trap.
4. Reduce UV LED current or exposure time.
5. Adjust gain/integration time to avoid saturation.
6. Add suitable optical long-pass or band-pass filtering if needed.
7. Use blank-corrected differential features.

### 8.4 Potential novelty

If you show that a specific geometry/baffle/filter:

- Reduces UV405 background,
- Improves repeatability,
- Avoids saturation, and
- Improves unseen-fruit performance,

then you have a concrete technical contribution. This is stronger than claiming novelty merely because you used AS7265x with AI.

---

## 9. Hardware bring-up plan

Do not begin with pesticides.

### Step 1: Sensor communication

- Connect AS7265x to Raspberry Pi via I2C.
- Confirm correct I2C communication.
- Read all 18 channels repeatedly.
- Record timestamps and errors.
- Verify no dropped reads.

### Step 2: LED control

For each LED:

- Turn it on through MOSFET/driver.
- Confirm only the intended LED is active.
- Record AS7265x response.
- Check heat and power stability.
- Confirm no sensor saturation.

### Step 3: Build the chamber

- Install sensor and LED mounts.
- Add matte-black surfaces.
- Add fruit holder.
- Add dark-cap position.
- Add PTFE reference location.
- Ensure chamber blocks room light.
- Ensure UV light cannot escape toward eyes/skin.

### Step 4: Run optical test matrix

Perform dark, blank, PTFE, black-target, and ordinary-apple tests.

### Step 5: Repeatability testing

#### Unmoved test

Keep one apple fixed. Scan it ten times.

\[
CV = \frac{\text{standard deviation}}{\text{mean}} \times 100
\]

Calculate CV per channel and per illumination mode.

#### Repositioned test

Remove and replace the same apple ten times. This tests real operator placement variability.

#### Session-drift test

Scan PTFE at the beginning, middle, and end of a session. Repeat over several days.

### Step 6: GO/NO-GO decision

| GO: proceed to pilot | NO-GO: redesign first |
|---|---|
| No meaningful channel saturation | UV405 saturates one or more channels |
| Stable dark readings | Empty chamber resembles apple response |
| Stable PTFE references | LED drift is large within session |
| Blank response documented and not dominant | Placement changes response excessively |
| Repeatability acceptable | Chamber leaks ambient light |
| Calibration repeatable across sessions | PTFE shifts beyond selected tolerance |

---

## 10. Literature survey: what was learned

The literature review showed that several research communities solve different parts of the challenge.

### 10.1 Laboratory methods

LC-MS/MS and GC-MS/MS provide the reference analytical standard but are expensive and laboratory-bound.

### 10.2 Hyperspectral imaging

HSI can capture hundreds of wavelengths and has shown high performance in narrow, controlled pesticide/crop experiments. However, cameras and optical rigs are expensive and not easily portable.

### 10.3 Vis-NIR chemometrics

NIR/Vis-NIR methods can classify or estimate specific pesticides on specific crops. Strong accuracy often comes from tightly controlled, narrow problems and better spectrometers than AS7265x.

### 10.4 Fluorescence methods

UV-induced fluorescence methods can be sensitive in water, extracts, or single-analyte assays. But translation to intact fruit surface scanning is not automatically proven.

### 10.5 Low-cost multispectral sensors

AS7265x and related sensors have been used for adjacent tasks such as ripeness, food quality, beverage classification, oil quality, and cold-chain monitoring.

A direct prior-art correction was required: published work already uses AS7265x and an ESP32 for pesticide-residue detection on produce. Therefore, “AS7265x + AI for pesticides” is not a defensible novelty claim by itself.

### 10.6 Corrected research gap

The current defensible gap is:

> Existing low-cost multispectral pesticide-screening studies show promising results under controlled conditions, but a low-cost platform integrating controlled multi-illumination geometry, dual-UV differential response, crosstalk mitigation, blank/dark/white calibration, uncertainty-aware rejection, synthetic-to-real research, independent fruit-wise validation, and laboratory-grounded validation remains insufficiently demonstrated.

---

## 11. The Roomi dataset

### 11.1 What it is

The Roomi apple dataset is a public hyperspectral dataset.

- 281 spectral bands.
- Approximately 400–1000 nm.
- 617 hyperspectral image cubes in the source dataset.
- In the project’s synthetic CSV, 337 records were used.
- Those records came from only 10 unique physical apples.
- Labels include control, Nativo/Monostar treatment family, and low/high treatment level.

### 11.2 Important limitation

The Roomi labels represent experimental immersion/treatment conditions, not laboratory-measured residue concentration on the fruit surface and not MRL status.

Therefore:

\[
\text{Roomi Control/Low/High} \neq \text{Below/Above MRL}
\]

### 11.3 How HSI becomes AS7265x-like values

A hyperspectral image contains many pixels and many wavelengths. For a selected apple region:

1. Select apple pixels / region of interest.
2. Average or summarize the spectrum across selected pixels.
3. Obtain a 281-band spectrum.
4. Apply the AS7265x channel-response function to simulate each 18-channel reading.

For channel \(k\):

\[
V_k = \sum_{i=1}^{281} S(\lambda_i) R_k(\lambda_i) I(\lambda_i) \Delta\lambda
\]

Where:

- \(S(\lambda_i)\) is source HSI reflectance.
- \(R_k(\lambda_i)\) is the AS7265x-like response curve.
- \(I(\lambda_i)\) is illumination weighting.
- \(V_k\) is the simulated sensor channel output.

The result becomes an 18-column synthetic CSV:

```text
fruit_id, treatment_family, level, class_3,
ch_410, ch_435, ch_460, ..., ch_940
```

### 11.4 What Roomi is useful for

- Software development.
- Testing CSV schema.
- Testing GroupKFold logic.
- Baseline ML experiments.
- Feature code development.
- Digital-twin/spectral reconstruction research.
- Comparing synthetic and real AS7265x distributions.

### 11.5 What Roomi cannot do

- Validate your physical AS7265x device.
- Train UV365/UV405 fluorescence models.
- Prove pesticide identity.
- Prove residue concentration.
- Prove MRL status.
- Replace real fruit measurements.

---

## 12. Current software baseline and findings

### 12.1 Synthetic data pipeline

```text
Roomi hyperspectral spectrum
        ↓
AS7265x Gaussian-response downsampling
        ↓
Synthetic 18-channel CSV
        ↓
StandardScaler
        ↓
PCA: 5 principal components
        ↓
LDA classifier
        ↓
Fruit-wise GroupKFold evaluation
```

### 12.2 Why GroupKFold is mandatory

If several scans come from one apple, they are related. If some scans of the same apple go into training and others into testing, the model can recognize the apple rather than generalize.

Correct rule:

```text
All scans from fruit A001 → training OR testing
Never both.
```

### 12.3 Current benchmark results

| Model | Mean accuracy | Interpretation |
|---|---:|---|
| PCA + LDA | 65.57% ± 5.34% | Best current synthetic three-class baseline |
| RBF SVM | 57.64% ± 5.51% | Lower than LDA |
| Random Forest | 46.49% ± 8.35% | High variance; likely overfitting small N |
| XGBoost | 47.09% ± 9.72% | High variance; likely overfitting small N |

### 12.4 Important findings

1. **PCA+LDA beat more complex models.** This supports using simple models when there are only 10 independent fruits.
2. **SNV harmed the synthetic baseline.** Standard Normal Variate reduced accuracy from 65.57% to approximately 52.57% on the already-calibrated synthetic data.
3. **The digital twin is not a classifier replacement.** Reconstructed spectrum alone performed worse than direct channels.
4. **Binary tasks are easier.** Control-versus-treated and low-versus-high tasks were better than the three-class task.
5. **The synthetic result is only a feasibility benchmark.** It does not predict final hardware accuracy.

### 12.5 Why SNV may change on real hardware

Synthetic data was already dark/white calibrated. Real hardware will have scattering and geometry effects. Therefore, re-test:

- No preprocessing.
- StandardScaler.
- SNV.
- MSC.
- Savitzky–Golay derivatives.
- Per-mode normalization.

Do not assume synthetic preprocessing results transfer to physical hardware.

---

## 13. Real hardware dataset structure

### 13.1 Raw long-format CSV

One row per illumination mode:

```csv
scan_id,fruit_id,batch_id,session_id,mode,gain,integration_time_ms,led_current_ma,sensor_temperature_c,ch_410,...,ch_940
A001_P1_WHITE,A001,BATCH_A,DAY1,white,64,100,20,28.1,...
A001_P1_IR,A001,BATCH_A,DAY1,ir,64,100,20,28.1,...
A001_P1_UV365,A001,BATCH_A,DAY1,uv365,64,100,15,28.1,...
A001_P1_UV405,A001,BATCH_A,DAY1,uv405,64,100,15,28.1,...
```

### 13.2 Model wide-format CSV

One row per complete fruit position scan:

```csv
scan_id,fruit_id,label,
white_ch_410,...,white_ch_940,
ir_ch_410,...,ir_ch_940,
uv365_ch_410,...,uv365_ch_940,
uv405_ch_410,...,uv405_ch_940
```

This gives 72 basic features before engineered differential features are added.

### 13.3 Metadata to save

| Field | Why it matters |
|---|---|
| fruit_id | Prevents leakage |
| batch/source | Tests batch generalization |
| date/session | Detects drift |
| illumination mode/order | Separates optical effects from time order |
| gain/integration time | Ensures scans are comparable |
| LED current/PWM | Documents illumination consistency |
| sensor/chamber temperature | Tracks drift/confounding |
| dark/white/blank reference IDs | Calibration traceability |
| saturation flag | Prevents invalid scans entering ML |
| fruit position/orientation | Tracks geometry variation |
| operator | Detects procedural effects |
| treatment details | Experimental audit trail |

---

## 14. Real dataset plan

### 14.1 Pilot A: untreated controls

Before any pesticide handling:

- Collect 10 untreated apples.
- Use one variety where possible.
- Scan across at least two days.
- Use 3–5 surface positions per apple.
- Test repeatability, placement sensitivity, day drift, and batch variation.

### 14.2 Pilot B: small treatment-condition study

After written safety approval and faculty supervision:

- Approximately 5 untreated apples.
- Approximately 5 known-treatment-condition apples.
- Same variety and documented protocol.
- Same chamber and calibration procedure.

This is a sanity-check pilot, not a publishable accuracy claim.

### 14.3 Main V1 dataset

| Item | Minimum target | Better target |
|---|---:|---:|
| Untreated independent apples | 30–50 | 50–100 |
| Known-treatment-condition apples | 30–50 | 50–100 |
| Positions per fruit | 3–5 | 5–8 |
| Sessions/days | 2–3 | 3+ |
| Fruit batches/sources | 2–3 | 3+ |
| Blind external batch | Separate source/day | Strongly recommended |

The number that matters is the number of **independent fruits**, not the number of scan rows.

---

## 15. Confounder controls

A model can appear accurate for the wrong reason. Test controls that differ in non-treatment ways.

| Control condition | What it tests |
|---|---|
| Untreated dry apple | Normal baseline |
| Washed untreated apple | Surface wax/washing effect |
| Surface-moist untreated apple | Moisture confusion |
| Handled untreated apple | Contact/handling effect |
| Different market batch | Source/batch shortcut |
| Different ripeness level | Natural biological variation |
| Known-treatment-condition apple | Intended positive condition |

If wet or washed apples are repeatedly predicted as treated-like, the model is learning a confounder. Fix the experiment before expanding the dataset.

---

## 16. ML process for real data

### 16.1 Data quality filter

Before ML, flag scans with:

- Missing channels.
- Saturated channels.
- Missing dark/PTFE/blank references.
- Invalid calibration denominator.
- LED malfunction.
- Unusual PTFE drift.
- Excessive sensor temperature.
- Bad chamber closure or placement.

Keep the raw data. Record why a scan was excluded or marked uncertain.

### 16.2 Feature sets to test

| Experiment | Features |
|---|---|
| A | White only |
| B | White + IR |
| C | White + UV365 |
| D | White + UV405 |
| E | UV365 + UV405 |
| F | All four modes |
| G | All modes + differential features |
| H | Best configuration + quality/OOD rejection |

### 16.3 Differential features

For every channel \(k\):

\[
F_{365,k} = S_{365,k} - D_k
\]

\[
F_{405,k} = S_{405,k} - D_k
\]

\[
\Delta F_k = F_{365,k} - F_{405,k}
\]

\[
Q_{365,k} = \frac{F_{365,k}}{S_{white,k}-D_k+\epsilon}
\]

\[
Q_{405,k} = \frac{F_{405,k}}{S_{white,k}-D_k+\epsilon}
\]

The objective is to test whether these features improve prediction on held-out fruits—not merely on training data.

### 16.4 Model order

For 30–50 independent fruits per class:

1. PCA + LDA.
2. Regularized SVM.
3. Optional small tree-model benchmark.

Do not default to deep learning.

| Independent physical fruits | Recommended approach |
|---:|---|
| 10–50 | PCA+LDA, regularized SVM |
| 50–150 | PCA+LDA/SVM, modest tree benchmark |
| 150–500 | Compare small 1D-CNN with classical models |
| 500+ | CNN/transfer learning becomes more reasonable |

---

## 17. Validation that survives scrutiny

### 17.1 Development validation

Use fruit-wise GroupKFold while comparing:

- Preprocessing.
- Feature sets.
- Illumination modes.
- PCA component counts.
- Model types.
- Confidence and OOD thresholds.

### 17.2 Locked blind test

Before final model decisions, split by whole fruit IDs:

| Partition | Purpose |
|---|---|
| 70% fruit IDs | Model development and training |
| 15% fruit IDs | Validation and threshold selection |
| 15% fruit IDs | Final locked blind test |
| Separate batch/day | External robustness check |

Do not change features, thresholds, models, or preprocessing after looking at the blind test.

### 17.3 Metrics

For the binary V1 task, report:

- Overall accuracy.
- Balanced accuracy.
- Sensitivity/recall for treated-like class.
- Specificity for untreated-like class.
- Macro-F1.
- Precision.
- Confusion matrix.
- Coverage/acceptance rate.
- Rejection rate.
- Accepted-scan performance.
- Confidence intervals.

### 17.4 Why sensitivity matters

A false negative means a known-treated-condition apple is predicted untreated-like. For a screening device, missed positives may be more concerning than false positives sent for laboratory confirmation.

---

## 18. Uncertainty and out-of-distribution detection

A classifier should not be forced to decide on every scan.

### 18.1 Uncertainty triggers

Return **Uncertain** when:

- Classifier confidence is low.
- Mahalanobis/OOD distance is high.
- Model variants disagree.
- Calibration failed.
- Channels saturated.
- PTFE reference drift is high.
- Scan is missing data.
- Placement quality is poor.

### 18.2 Mahalanobis distance

\[
D_M(x)=\sqrt{(x-\mu)^T\Sigma^{-1}(x-\mu)}
\]

It measures how unusual a scan is compared with training data while accounting for correlated features.

The previous synthetic threshold around 4.282 is only for the synthetic dataset. Recalculate thresholds using real hardware data after calibration.

### 18.3 Correct reporting format

Never report accepted-scan accuracy alone.

Correct example:

> Balanced accuracy: 91% on accepted scans; coverage: 88%; rejection rate: 12%; dataset: held-out real fruit IDs.

Incorrect example:

> 91% accuracy.

---

## 19. Digital twin / continuous spectral reconstruction model

### 19.1 What it does

It maps an 18-channel vector to an estimated 281-band curve:

\[
18\ \text{channels} \rightarrow 281\ \text{estimated spectral values}
\]

Current architecture:

```text
18 inputs → 64 → 32 → 64 → 281 outputs
```

Implemented as MLPRegressor with ReLU, Adam, L2 regularization, and early stopping.

### 19.2 Current results

| Dataset | Mean reconstruction R² | RMSE |
|---|---:|---:|
| Roomi apple treatment dataset | 0.5546 | 0.03264 |
| Cucurbitaceae benchmark | 0.8654 | 0.0147 |

The apple result is moderate. It means the model explains roughly 55% of the high-resolution spectral variance from only 18 channels.

### 19.3 What it is useful for

- Interactive spectral visualization.
- Comparing measured 18 values against estimated curve.
- Research on synthetic-to-real adaptation.
- Drift monitoring.
- Anomaly support.
- Modest feature enhancement.

### 19.4 What it is not

- A real hyperspectral camera.
- A replacement for physical measurement.
- The main pesticide/treatment classifier.
- Evidence of 281-band sensor performance.

The direct 18-channel classifier performed better than classification based only on reconstructed spectra.

---

## 20. Route to 90% accuracy

### 20.1 Current reality

| Current result | Meaning |
|---|---|
| 65.57% ± 5.34% | Synthetic three-class treatment-condition baseline |
| ~77% | Synthetic control vs treated task |
| ~88–89% | Synthetic narrow binary task for one treatment family/level |
| 92–99% accepted-scan accuracy | Synthetic selective prediction; must include coverage |
| Real hardware performance | Unknown until physical test |

### 20.2 What can improve real performance

1. Stable optics.
2. Dark/PTFE/blank calibration.
3. Reduced placement variation.
4. One fruit variety initially.
5. One documented treatment formulation/family initially.
6. Binary task first.
7. More independent fruits.
8. Real UV-excitation features if physically useful.
9. Correct preprocessing for real data.
10. Uncertainty rejection with transparent coverage.
11. Lab-linked labels later.

### 20.3 What not to do

- Do not inflate data with repeated scans and call them independent fruits.
- Do not use random row-level train/test splits.
- Do not train GANs on 30 apples to create fake diversity.
- Do not claim 99% after rejecting half the data without showing coverage.
- Do not jump to a Transformer because it sounds advanced.
- Do not claim real hardware performance from synthetic results.
- Do not claim UV helps before testing it.

### 20.4 Honest target

A realistic target is:

> At least 90% balanced accuracy or macro-F1 on a narrow binary physical task, evaluated on locked unseen fruit IDs, with coverage and rejection rate reported.

This is a goal, not a guarantee.

---

## 21. Product roadmap

### Prototype V1: scientific core

- AS7265x sensor.
- Controlled chamber.
- White/IR/UV modes.
- Calibration.
- Treatment-condition screening.
- Confidence and uncertain output.
- CSV logging.
- Fruit-wise validation.

### Product V1.5: useful user-facing features

- Load cell for weight.
- Calories estimated from weight and a nutrition database.
- Pi camera for apple presence and placement quality.
- Visual external defect score.
- Dashboard and scan history.

Calories are estimated, not spectrally measured:

\[
\text{estimated calories} = \frac{\text{weight in grams}}{100} \times \text{nutrition reference kcal per 100 g}
\]

### Product V2: quality models

- Bruise detection.
- External defect detection.
- Ripeness classification.
- Variety classification.
- Brix/sugar estimation if physical refractometer labels are collected.
- Water/dry-matter proxies if reference measurements are collected.

### Product V3: chemical quantification research

- Match scans to LC-MS/MS or GC-MS/MS results.
- Train selected crop–pesticide residue models.
- Estimate residue range or threshold risk.
- Study one crop–pesticide MRL problem carefully.

---

## 22. Research and publication strategy

### 22.1 Strong paper story

The strongest paper is not:

> We tried many AI algorithms.

It is:

> We developed and evaluated a low-cost controlled multi-illumination optical system, quantified UV405 crosstalk, compared white-only and differential multi-excitation features, and evaluated treatment-condition classification using fruit-wise validation and uncertainty-aware screening.

### 22.2 Core paper sections

1. Introduction: food-safety access gap.
2. Related work: laboratory, HSI, low-cost spectroscopy, prior art.
3. Hardware: chamber, sensor, LEDs, calibration.
4. Methods: dataset, labels, GroupKFold, calibration, models.
5. Optical characterization: dark/blank/PTFE/crosstalk/repeatability.
6. Ablation: white vs IR vs UV modes vs differential features.
7. Results: accuracy, sensitivity, specificity, coverage, errors.
8. Discussion: limits, confounders, synthetic-to-real gap.
9. Conclusion: what the device can and cannot claim.

### 22.3 Negative results are valuable

Report honestly if:

- UV does not improve performance.
- SNV harms the real model.
- Digital reconstruction adds little value.
- Placement variation is high.
- A confounder causes false positives.

A rigorous negative result is better research than an inflated positive result.

---

## 23. Patent strategy

### 23.1 What is not a strong patent claim

- AS7265x detects pesticide.
- PCA+LDA classifies a spectrum.
- ML model predicts treatment class.
- GroupKFold validation.
- Digital twin alone.

Prior art already exists for AS7265x + ESP32 pesticide screening on produce.

### 23.2 What may become a stronger claim

A demonstrated apparatus/method such as:

- Specific excitation/detection geometry.
- Specific baffle/light trap/filter arrangement.
- Controlled multi-mode sequence.
- Dark/white/blank calibration loop.
- Differential UV-excitation extraction procedure.
- Documented technical effect: lower bleed-through, higher SNR, more repeatable readings, or better held-out performance.

### 23.3 Public disclosure warning

The GitHub repository is already public. Before publicly posting new optical geometry, filter details, circuit details, or measured results, consult the college IP cell or a registered patent professional. A provisional filing does not guarantee eventual patent grant.

---

## 24. Funding strategy

### 24.1 Strong funding narrative

Pitch the project as:

> A low-cost, portable, uncertainty-aware produce-screening platform for identifying samples that need laboratory pesticide-residue confirmation at farm gates, packhouses, procurement centres, and markets.

Do not pitch it as:

> A student AI project that detects all pesticide residues with 99% accuracy.

### 24.2 Funding caution

NIDHI PRAYAS may support individual innovators and commercialization-track prototypes, but its rules may exclude research or student internship funding in academic institutions. Confirm eligibility directly with the selected incubator before preparing a full proposal.

Potential routes include:

- Institutional innovation cell.
- Agricultural incubator.
- RKVY-RAFTAAR/student agripreneur route.
- NIDHI PRAYAS only after eligibility clarification.
- BIRAC BIG later, after incubator/startup structure and physical proof of concept.

---

## 25. Master execution timeline

### Before components arrive

1. Prepare Raspberry Pi OS and Python environment.
2. Prepare AS7265x I2C code.
3. Prepare LED-control code.
4. Prepare raw and calibrated CSV schemas.
5. Prepare hardware test forms.
6. Draft safety SOP.
7. Contact faculty supervisor for safety and IP guidance.
8. Contact potential LC-MS/MS partner.
9. Decide on public disclosure and IP strategy.

### Week 1 after arrival

1. Connect AS7265x.
2. Read 18 channels.
3. Save readings to CSV.
4. Test I2C reliability.
5. Test one LED at a time.

### Week 2

1. Build chamber.
2. Add dark cap and PTFE tile.
3. Run dark/blank/PTFE/black-target tests.
4. Test UV405 bleed-through.
5. Check saturation.

### Week 3

1. Run unmoved repeatability.
2. Run repositioned repeatability.
3. Monitor session drift.
4. Test several geometries.
5. Select a stable configuration.

### Week 4

1. Collect 10 untreated apples across two days.
2. Analyze variation by fruit, position, day, and batch.
3. Decide GO/NO-GO.

### Month 2

1. Obtain written safety clearance.
2. Run small treatment-condition pilot.
3. Run confounder controls.
4. Begin LC-MS/MS collaboration if possible.

### Month 3 onward

1. Collect 30–50+ independent fruits per condition.
2. Train PCA+LDA/SVM models.
3. Run illumination ablation.
4. Lock blind test set.
5. Validate performance honestly.
6. Prepare research paper, funding package, and IP consultation based on actual results.

---

## 26. Final principle

The project is not about making an AI model look impressive. It is about proving that a low-cost optical instrument can produce stable, useful, and honest screening information.

The core rule is:

> **Do not make the AI smarter before making the measurement more stable, more informative, and more repeatable.**

If you follow the process—controlled optics, calibration, repeatability, independent fruits, careful labels, fruit-wise validation, uncertainty reporting, and laboratory linkage—you can build a credible project with real research value, practical product potential, and a possible future IP case.
