# CFRD Phase 4: Physical Pilot Scan Protocol (30–50 Real Apples)
**Multi-Cohort Multispectral Pesticide Screening with AS7265x Triad**  
*Document Version: 1.0 | Status: Ready for Hardware Bring-Up & Lab Execution*

---

## 1. Executive Objective & Why This Protocol Exists

The central bottleneck facing this project is sample size and biological confounding:
- In the Roomi bench audit, **$N=13$ apples** proved insufficient because natural peel variance ($\sim 15\text{--}30\%$) overwhelmed trace pesticide signals ($\sim 1\text{--}3\%$).
- In public datasets (Essex / SpectroFood), data was gathered for maturity/ripeness/Brix, **not pesticide screening**.
- **This protocol establishes the primary, unconfounded empirical dataset required for your patent claim and funding defense**: scanning 40 physical apples under identical optical geometry with the target SparkFun AS7265x Triad sensor ($410\text{--}940\text{ nm}$).

---

## 2. Experimental Cohorts (40 Apples Total)

To defeat the **"Broadband Wet Confound"** (where water-washing or surfactant films mimic pesticide absorption), apples are split into 4 balanced cohorts:

| Cohort | Name | Size ($N$) | Preparation Protocol | Target Experimental Role |
|---|---|---|---|---|
| **A** | **Clean / Organic Baseline** | 10 apples | Certified organic apples, unwashed, dry peel. | Establishes the clean biological variance baseline across natural wax/peel types. |
| **B** | **Water-Washed Control** | 10 apples | Dipped in distilled water, wiped gently, air-dried for 45 min. | Negative control: proves the sensor does not falsely flag surface water or clean-fruit wiping as chemicals. |
| **C** | **Commercial Field Dose** | 10 apples | Sprayed uniformly with commercial formulation (e.g. Nativo @ $0.5\text{ g/L}$), air-dried for 2 hours. | Validates detection of realistic post-harvest spray residues. |
| **D** | **Elevated Residue Dose** | 10 apples | Sprayed uniformly with commercial formulation (e.g. Nativo @ $2.0\text{ g/L}$), air-dried for 2 hours. | High-confidence ground truth positive: confirms spectral signature absorption depth. |

> [!IMPORTANT]
> **Prerequisite Sanity Check (Step 0): Intra-Fruit Clean Baseline & Peel Uniformity**  
> Before assuming that paired same-fruit referencing isolates pesticide signals, you MUST empirically verify the intra-fruit clean baseline noise floor:
> 1. Scan at least two clean, untreated spots on the **same untreated apple** (e.g. sun-exposed "blush" side vs shaded "ground" side).
> 2. Quantify the intra-fruit baseline differential: $\Delta R_{\text{intra-clean}} = |R_{\text{clean, A}} - R_{\text{clean, B}}|$.
> 3. Natural anthocyanin and cuticle thickness variations between blush and ground sides can create intra-apple differences of 2–5%. 
> 4. **Validation rule**: The paired differential method is only physically valid if pesticide-induced differential signal $\Delta R_{\text{pesticide}} > 3 \times \sigma(\Delta R_{\text{intra-clean}})$. Do not assume Region A and Region B have identical baselines without this empirical measurement!

> [!TIP]
> **Within-Fruit Paired Control (Golden Standard)**:  
> Scan Cohort C and D apples **before spraying** at marked spatial fiducials (as their own clean baseline), spray and dry them, and then re-scan the *exact same fiducial spots*. A paired $\Delta R = R_{\text{after}} - R_{\text{before}}$ eliminates both fruit-to-fruit and intra-fruit anatomical variance.

---

## 3. Optical Rig & Scanning Geometry

```
               ┌───────────────────────────────┐
               │    SparkFun AS7265x Triad     │
               │   (AS72651 / AS72652 / AS72653)│
               └──────────────┬────────────────┘
                              │ 45° LED Ring
                              ▼
           ═════════════════════════════════════════  Top Aperture
           │                                       │
           │       3D Printed Optical Collar       │  Matte Black PLA / Velvet
           │      (Fixed 20 mm Standoff Height)    │  Light-Tight Seal
           │                                       │
           ═════════════════════════════════════════  Bottom Flange (Soft Gasket)
                              ▼
               ( ( ( ( ( APPLE PEEL ) ) ) ) )
```

1. **Fixed Standoff Distance**: $20\text{ mm} \pm 1\text{ mm}$ maintained by the 3D-printed collar ([`chamber_enclosure.scad`](file:///d:/CFRD/project/cad/chamber_enclosure.scad)).
2. **Light-Tight Contact**: A compliant silicone or black felt gasket around the aperture rim prevents ambient fluorescent/LED light from leaking into the photodiodes.
3. **Equatorial Multi-Point Sampling**:
   - Each fruit is scanned at **4 equatorial sides** ($90^\circ$ rotations: Side A, B, C, D).
   - Plus **1 top scan (Stem end)** and **1 bottom scan (Calyx end)**.
   - Total scans per apple: **6 spectral captures**.
   - Total dataset yield: $40\text{ apples} \times 6\text{ scans} = \mathbf{240\text{ calibrated spectral vectors}}$.

---

## 4. Calibration Discipline (Every 10 Apples)

Ambient temperature and LED heating cause photodiode drift. The operator must perform a two-step reference calibration at the start of each session and every 10 apples:

1. **Dark Reference ($D_\lambda$)**:
   - Place sensor inside dark box with all LEDs disabled.
   - Record 18-channel counts: $D_\lambda \approx 0\text{--}15$ counts.
2. **White Reference ($W_\lambda$)**:
   - Seat sensor aperture firmly against a high-purity **PTFE (Teflon) calibration puck** (or Spectralon tile).
   - Enable White, IR, and UV LEDs sequentially.
   - Record 18-channel counts: $W_\lambda \approx 10,000\text{--}45,000$ counts.
3. **Real-Time Calibrated Reflectance**:
   $$R_\lambda = \frac{I_\lambda - D_\lambda}{W_\lambda - D_\lambda}$$
   Where $I_\lambda$ is the raw fruit count under illumination.

---

## 5. Strict Metadata & Logging Schema

Every scan must be logged with complete provenance to prevent the `Number` collision trap:

| Field Name | Type | Example | Purpose |
|---|---|---|---|
| `Session_ID` | String | `SESS_20261015_01` | Tracks daily session & environment conditions. |
| `Global_Apple_ID` | String | `GALA_PILOT_C_014` | **Primary Key**: Unique across all sessions. |
| `Cultivar` | String | `Royal Gala` | Apple variety. |
| `Cohort` | String | `COHORT_C_NATIVO_0.5` | Experimental treatment group. |
| `Dose_g_L` | Float | `0.50` | Exact dilution concentration. |
| `Side` | String | `Side_A` | `Side_A`, `Side_B`, `Side_C`, `Side_D`, `Stem`, `Calyx`. |
| `Rotation_Deg` | Int | `0` | `0`, `90`, `180`, `270`. |
| `Sensor_Temp_C` | Float | `27.5` | AS7265x internal die temperature. |
| `R_410` .. `R_940` | Float (18 cols) | `0.2458` | Normalized reflectance ($0.0\text{--}1.0$). |
| `Raw_410` .. `Raw_940` | Int (18 cols) | `12450` | Uncalibrated detector counts. |
| `Dark_410` .. `Dark_940` | Int (18 cols) | `12` | Session baseline dark counts. |
| `White_410` .. `White_940`| Int (18 cols) | `38900` | Teflon reference white counts. |

---

## 6. Machine Learning Cross-Validation Protocol

To guarantee patent-defensible, publication-grade results:
1. **GroupKFold (Leave-One-Fruit-Out)**:
   - Grouping key = `Global_Apple_ID`.
   - All 6 scans of `GALA_PILOT_C_014` must be strictly in Train OR strictly in Test.
   - Zero fruit leakage across folds.
2. **Preprocessing**:
   - Standard Normal Variate (SNV) to remove surface curvature and brightness differences.
3. **Target Evaluation Metrics**:
   - Primary: Binary Contamination (Clean vs. Pesticide Treated) Balanced Accuracy, ROC-AUC.
   - Secondary: 3-Class (Clean vs. Low-Dose vs. High-Dose) Confusion Matrix.
   - Mahalanobis Out-of-Distribution (OOD) distance thresholding for confidence rejection.
