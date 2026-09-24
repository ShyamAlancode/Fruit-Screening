# CFRD — Defense / Viva Preparation: 20 Hardest Questions

> **Source:** Second Technical Review, §6  
> **Date:** September 24, 2026  
> **Purpose:** Professor/examiner defense preparation. Every answer is grounded in the project's actual evidence — not in overstatement.

---

## Q1. Why did 82.65% Macro-F1 fall to 56.14%?

**Answer:**  
Because the first split was by spatial ROI, not by physical fruit. ROIs from the same apple appeared in both training and validation sets. The model therefore learned fruit-specific biological characteristics (cuticle wax, pigment baseline) rather than treatment effects. Leave-One-Apple-Out (LOAO) forced testing on a completely unseen apple and removed that source of leakage. This is documented in the leakage audit and is itself the primary methodological contribution of the project.

---

## Q2. Why do you call 75% accuracy successful if specificity is only 30%?

**Answer:**  
I do not call it a balanced classifier. The result is better interpreted as a high-sensitivity screening configuration. 18/18 treated scans were detected (100% sensitivity), but 7/10 clean scans were falsely flagged (30% specificity). Balanced accuracy is 65%. The 75% headline accuracy must always be read alongside sensitivity and specificity — which is why every summary table in this project now includes both. The device should be framed as a "flag for recheck" system, not a definitive pesticide confirmation.

---

## Q3. Why use scan-level aggregation instead of spot-level classification?

**Answer:**  
The biological baseline varies strongly between apples (>60% reflectance shift from cuticle wax). A single-spot absolute measurement cannot reliably separate this from the <2% residue signal on an unseen apple. A fruit-level screening instrument can exploit multiple spatial measurements, reducing reliance on any single absolute spectrum. However, aggregation reduces random variance, not per-apple systematic bias — so it is not a complete solution. Physical reference calibration and differential scanning are needed for that.

---

## Q4. What proves the signal is pesticide rather than apple biology?

**Answer:**  
The current computational experiments do not fully prove that. They show treatment-associated spectral discrimination under the Roomi dataset with LOAO validation. However, cross-apple generalization is limited (AUROC 0.59, p≈0.19), the final held-out test failed (Macro-F1 35–42%), and the active compounds are not identified at a molecular level. Wavelength associations with known absorption features are suggestive, but as the project correctly notes, molecular assignment requires FTIR or Raman confirmation. Physical experiments with chemical ground truth are required for stronger causal interpretation.

---

## Q5. Why simulate the AS7265x instead of measuring it directly?

**Answer:**  
The physical sensor had not arrived during the computational development phase. The simulation allowed early investigation of whether 18 finite-bandwidth channels could preserve useful screening information before hardware integration. This is called a "sensor twin" or "digital twin" study — a legitimate research methodology, but one that must be explicitly distinguished from a physical hardware measurement. All results in this project are from simulated sensor channels, not measured AS7265x scans.

---

## Q6. What is the main limitation of a Gaussian sensor-twin model?

**Answer:**  
It models spectral bandwidth correctly as a first-order approximation, but it does not capture: the full empirical spectral response curves of the real chip, electronic gain and dark current, optical chamber geometry and stray light, illumination spectral distribution, detector nonlinearity, inter-channel calibration errors, or field-of-view differences between the three sub-chips. The correct term is "physics-inspired simulation," not "physical sensor validation."

---

## Q7. Why is 100% sensitivity not enough?

**Answer:**  
Because 100% sensitivity can coexist with many false positives. In this study specificity was only 30% (3/10 clean scans correctly rejected). A screening device with 70% false positives on clean produce would be commercially unusable. Predicting "treated" for every apple achieves 64.3% accuracy and 100% sensitivity — the trivial baseline. The model adds only three additional correct clean identifications. The device must therefore communicate uncertainty and provide a "recheck" state rather than presenting every positive screen as confirmed contamination.

---

## Q8. What would invalidate your current conclusion?

**Answer:**  
A physical study with a sufficiently large number of independent apples and chemical ground truth (LC-MS/MS) showing that the 17-difference representation does not generalize outside the public Roomi dataset would invalidate the current generalization hypothesis. More proximally: the final held-out test (Macro-F1 35–42%) already substantially weakens the 75% LOAO scan-level claim. This result is now reported as the primary result in the project.

---

## Q9. Why does the final held-out test perform worse?

**Answer:**  
It exposes the instability of a small dataset (five physical apple cohorts). The held-out test uses a different subset of the same five-apple experimental family and shows that spot-level generalization is weaker than the scan-level LOAO aggregate suggests. At Macro-F1 35–42%, every clean ROI was misclassified, indicating the model has over-adapted to the training distribution. This result is now the first row of every benchmark table in this project.

---

## Q10. Is this quantitative pesticide detection?

**Answer:**  
No. The current work is screening/anomaly detection. It does not establish ppm-level concentration prediction, molecular identification of active ingredients, or regulatory MRL compliance. The project explicitly forbids terms like "pesticide detected," "residue mg/kg," "below MRL," and "safe to eat" in all outputs. The correct framing is: "this apple's spectrum is anomalous relative to the training distribution — recommend recheck."

---

## Q11. Can the system identify Monostar versus Nativo?

**Answer:**  
The experiments include different treatment families, but the current evidence does not establish robust formulation-specific identification on unseen apples. The two formulations have different active ingredients (organophosphate/pyrethroid vs triazole/strobilurin), which may produce different spectral signatures. However, the Roomi dataset cohort structure makes this analysis confounded with fruit identity. Formulation-specific identification would require a purpose-designed experiment with multiple independently randomized apples per formulation.

---

## Q12. Why is the Brix result confusing?

**Answer:**  
Two benchmark generations exist. Generation 1 (verified, Sept 15): 60 scans, 33 apples, GroupKFold — R² = −0.03 (negative). Generation 2 (unreconciled, Sept 24): 417 observations, 109 fruits, GroupKFold — R² = 0.43. The newer result may be real (more diverse cultivars, more observations), but the processed dataset is not packaged in the repository, and the "best" model was selected from 12 candidates on the same cross-validation data (selection optimism). I would present Generation 1 as authoritative until I can freeze one pipeline with one SHA256-verified dataset and external test data.

---

## Q13. Why not just use RGB?

**Answer:**  
RGB is valuable for surface appearance (defects, bruising, color grading). VIS/NIR multispectral sensing provides access to internal chemistry and surface film characteristics that are invisible to RGB. The project shows that RGB and spectral models address different information channels. However, the comparison in this project used different validation schemes (LOAO for spectral, StratifiedKFold for RGB), making a direct performance comparison invalid. Under the same LOAO protocol, we do not know whether spectral outperforms RGB — that comparison is retracted.

---

## Q14. What is the scientific contribution if the physical sensor is not built yet?

**Answer:**  
The contribution is methodological and computational: (1) leakage discovery and the LOAO framework — documenting how intra-fruit ROI splitting inflates results by ~26 percentage points; (2) finite-band sensor simulation — showing that a 17-adjacent-difference representation can match 300-band derivative performance in the LOAO benchmark; (3) the biological baseline dilemma quantification — >60% cuticle wax reflectance shift vs <2% residue signal; (4) the spatial aggregation reframing — from single-spot classifier to multi-spot scanner. These are contributions to methodology and instrumentation science regardless of whether the hardware exists.

---

## Q15. Why use adjacent differences rather than all 18 channels?

**Answer:**  
The difference transform suppresses broad spectral baseline variation (low spatial-frequency variation) and emphasizes local spectral transitions at channel boundaries. This is motivated by the biological baseline problem: cuticle wax shifts the entire spectrum up or down, but first-difference features are relatively insensitive to this. It is an empirical design choice motivated by spectroscopic principles — not a proof of universal optimality. Other representations (ratios, second differences, PCA scores) were also tested and performed less well in this benchmark.

---

## Q16. Why is the 18-channel model equal to the 300-band model in scan accuracy?

**Answer:**  
Only within the specific Roomi five-apple LOAO simulation benchmark, at the 25-spot aggregation level, with the specific ElasticNet LR classifier. Both representations reach 75% because scan-level aggregation is the dominant factor at 25 spots, not spectral resolution. It does not imply general equivalence for every dataset, every operating condition, every fruit population, or every application. Physical experiments may show different results.

---

## Q17. What is the role of OOD detection?

**Answer:**  
It is a safety mechanism for recognizing scans unlike the training distribution. The current project's OOD detector uses Mahalanobis distance and was validated as a cross-dataset domain-shift detector (Roomi vs SpectroFood) — not as a universal field OOD detector. Realistic failure modes that the current OOD detector has not been tested against include: non-apple objects, wet/condensed fruit, reflective stickers, different lighting temperatures, cultivars excluded from training, and chemical coatings other than the two tested formulations.

---

## Q18. What experiment would you run first when the hardware arrives?

**Answer:**  
Sensor readout verification and dark/white repeatability — before scanning any fruit. Specifically: (1) confirm I2C device presence and read status registers; (2) read all 18 channels under a stable white LED source and verify nonzero, non-saturated, physically plausible values; (3) perform 30 dark frames and 30 PTFE white reference frames; (4) verify LED switching by checking that the four illumination modes produce physically distinct response patterns. There is no scientific value in treatment experiments until sensor readout and calibration are trustworthy.

---

## Q19. What is the strongest evidence that your software is reproducible?

**Answer:**  
Pinned dependencies (`requirements_lock.txt`, `requirements.txt` now both pin sklearn 1.5.0), model SHA256 hashes in `model_manifest.json`, data MD5 provenance manifest, source code, and 73 passing unit tests. However, full end-to-end reproducibility is not yet achieved because: the raw Roomi HSI cubes and the complete 417-row Brix dataset are external to the package; the working tree has uncommitted modifications; and no release tag has been declared. These gaps are documented in `REPRODUCIBILITY.md`.

---

## Q20. What is your single most important next milestone?

**Answer:**  
A controlled physical experiment with real AS7265x scans, documented calibration, at least 20 independent untreated apples (biological baseline study), followed by known-treated vs untreated apples with chemical reference measurements on a subset — using exactly the same preprocessing and evaluation definitions as the simulation benchmark. Only this experiment can determine whether the simulated 18-channel equivalence holds in physical reality. Everything else in the project — the data pipeline, the feature representations, the evaluation framework, the hardware driver — exists to enable and support this one experiment.
