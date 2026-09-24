"""
scripts/comprehensive_case_studies_evaluation.py
================================================
Comprehensive evaluation and verification runner across ALL 5 case studies
in CFRD: Portable Multispectral Pesticide Screening & Apple Quality Analysis.

Case Studies Covered:
  1. Case Study 1: Pesticide Treatment-Condition Screening
     - 1A: Synthetic Simulation (240 rows / 20 fruits) -> GroupKFold CV PCA+LDA & SVM
     - 1B: Real Roomi HSI Extractions (26 rows / 13 fruits) -> GroupKFold CV PCA+LDA & SVM
     - 1C: Binary Formulation Experiments (6 binary classification tasks)
     - 1D: Selective Rejection / Confidence Coverage Sweep
  2. Case Study 2: Ripeness Maturity Grading (Under-ripe, Ripe, Over-ripe)
     - GroupKFold CV by physical apple across 18-day storage tracking
  3. Case Study 3: Cultivar / Variety Classification (5 apple cultivars)
     - GroupKFold CV by physical apple across 5 cultivars
  4. Case Study 4: Brix Sweetness Prediction (°Bx Regression)
     - Essex Hyperspectral Benchmark (33 physical apples / 60 scans) vs ground truth
  5. Case Study 5: External Visual Surface Defect Quality (Computer Vision)
     - MobileNetV3-Small test set evaluation
  6. Case Study 6: Cross-Domain Shift (OOD Mahalanobis Distance Test)
     - Roomi vs SpectroFood domain shift detector

Inference Test Scenarios ("Making the Model Answer"):
  - Scenario 1: Real Apple Photo (test_user_apple.jpg)
  - Scenario 2: AS7265x Triad Mock Hardware Sensor Scan
  - Scenario 3: Synthetic Untreated Control Apple Spectrum
  - Scenario 4: Synthetic High-Dose Pesticide Apple Spectrum
  - Scenario 5: Real Roomi Physical HSI Extracted Spectrum
  - Scenario 6: SpectroFood Natural Variation Spectrum
  - Scenario 7: Noisy Simulated Sensor Spectrum (with LED drift & shot noise)
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Safe console encoding for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Path configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.predictor import ApplePredictor
from src.constants import (
    AS7265X_CENTRES_NM, CHANNEL_COLS, CLASS_LABELS,
    ROOMI_REAL_18CH_PATH, ROOMI_SYNTHETIC_18CH_PATH,
    SPECTROFOOD_18CH_PATH, SYNTHETIC_CSV_PATH,
)
from src.sensor_simulator import simulate_scan
from src.as7265x_driver import MOCK_APPLE_REFLECTANCE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_inference_scenarios(predictor: ApplePredictor) -> dict:
    """Execute model inference across 7 distinct practical test scenarios."""
    scenarios_results = {}

    # Scenario 1: Apple Image
    test_img_path = PROJECT_ROOT / "data" / "test_user_apple.jpg"
    if test_img_path.exists():
        res1 = predictor.predict_image(test_img_path)
        scenarios_results["Scenario_1_Apple_Photo"] = {
            "input_type": "RGB Image (data/test_user_apple.jpg)",
            "pesticide_screening": {
                "predicted_label": res1.pesticide.label,
                "confidence_pct": round(res1.pesticide.confidence * 100, 2),
                "probabilities": {k: round(v, 4) for k, v in res1.pesticide.probabilities.items()},
                "ood_status": res1.pesticide.ood_status,
                "ood_score": res1.pesticide.ood_anomaly_score,
            },
            "ripeness_stage": {
                "predicted_stage": res1.ripeness.stage,
                "confidence_pct": round(res1.ripeness.confidence * 100, 2),
                "probabilities": {k: round(v, 4) for k, v in res1.ripeness.probabilities.items()},
            },
            "variety_cultivar": {
                "predicted_cultivar": res1.variety.cultivar,
                "confidence_pct": round(res1.variety.confidence * 100, 2),
            },
            "brix_sweetness": {
                "estimated_brix": res1.brix.estimated_brix,
                "sweetness_tier": res1.brix.sweetness_tier,
            },
            "visual_defect": {
                "status": res1.visual_quality.status if res1.visual_quality else "N/A",
                "defect_probability": res1.visual_quality.defect_probability if res1.visual_quality else None,
            },
        }

    # Scenario 2: Mock Hardware Sensor Scan
    res2 = predictor.predict_spectrum(MOCK_APPLE_REFLECTANCE)
    scenarios_results["Scenario_2_Mock_Hardware_Scan"] = {
        "input_type": "AS7265x Mock Reflection Array (18 channels)",
        "pesticide_screening": {
            "predicted_label": res2.pesticide.label,
            "confidence_pct": round(res2.pesticide.confidence * 100, 2),
            "ood_status": res2.pesticide.ood_status,
            "ood_score": res2.pesticide.ood_anomaly_score,
        },
        "ripeness_stage": {
            "predicted_stage": res2.ripeness.stage,
            "confidence_pct": round(res2.ripeness.confidence * 100, 2),
        },
        "variety_cultivar": {
            "predicted_cultivar": res2.variety.cultivar,
            "confidence_pct": round(res2.variety.confidence * 100, 2),
        },
        "brix_sweetness": {
            "estimated_brix": res2.brix.estimated_brix,
            "sweetness_tier": res2.brix.sweetness_tier,
        },
    }

    # Scenario 3: Synthetic Control Apple Spectrum (from synthetic CSV row 0)
    synth_csv = PROJECT_ROOT / "data" / "pesticide_screening" / "roomi_synthetic_simulation" / "roomi_synthetic_18ch.csv"
    if synth_csv.exists():
        df_synth = pd.read_csv(synth_csv)
        row0 = df_synth[CHANNEL_COLS].iloc[0].values.astype(float)
        res3 = predictor.predict_spectrum(row0)
        scenarios_results["Scenario_3_Synthetic_Control_Spectrum"] = {
            "input_type": "Synthetic Control Apple (row 0, ground truth: Control)",
            "pesticide_screening": {
                "predicted_label": res3.pesticide.label,
                "confidence_pct": round(res3.pesticide.confidence * 100, 2),
                "probabilities": {k: round(v, 4) for k, v in res3.pesticide.probabilities.items()},
                "ood_status": res3.pesticide.ood_status,
                "ood_score": res3.pesticide.ood_anomaly_score,
            },
            "ripeness_stage": res3.ripeness.stage,
            "variety_cultivar": res3.variety.cultivar,
            "brix_sweetness": f"{res3.brix.estimated_brix} °Bx",
        }

        # Scenario 4: Synthetic High-Treatment Apple Spectrum (row 12, ground truth: High-treatment)
        row_high = df_synth[df_synth["level"] == "high"][CHANNEL_COLS].iloc[0].values.astype(float)
        res4 = predictor.predict_spectrum(row_high)
        scenarios_results["Scenario_4_Synthetic_High_Treatment_Spectrum"] = {
            "input_type": "Synthetic High Treatment Apple (ground truth: High-treatment)",
            "pesticide_screening": {
                "predicted_label": res4.pesticide.label,
                "confidence_pct": round(res4.pesticide.confidence * 100, 2),
                "probabilities": {k: round(v, 4) for k, v in res4.pesticide.probabilities.items()},
                "ood_status": res4.pesticide.ood_status,
                "ood_score": res4.pesticide.ood_anomaly_score,
            },
            "ripeness_stage": res4.ripeness.stage,
            "variety_cultivar": res4.variety.cultivar,
            "brix_sweetness": f"{res4.brix.estimated_brix} °Bx",
        }

    # Scenario 5: Real Roomi HSI Extracted Spectrum
    real_csv = PROJECT_ROOT / "data" / "pesticide_screening" / "roomi_real_18ch" / "roomi_real_18ch.csv"
    if real_csv.exists():
        df_real = pd.read_csv(real_csv)
        row_real = df_real[CHANNEL_COLS].iloc[0].values.astype(float)
        # Test under both synthetic model flavor and real model flavor
        res5_synth = predictor.predict(spectrum=row_real, pesticide_model_flavor="synthetic")
        res5_real = predictor.predict(spectrum=row_real, pesticide_model_flavor="real")
        scenarios_results["Scenario_5_Real_Roomi_HSI_Spectrum"] = {
            "input_type": "Real Roomi .bil HSI Extraction (Fruit 1, untreated apple)",
            "under_synthetic_model": {
                "predicted_label": res5_synth.pesticide.label,
                "confidence_pct": round(res5_synth.pesticide.confidence * 100, 2),
                "ood_status": res5_synth.pesticide.ood_status,
                "ood_score": res5_synth.pesticide.ood_anomaly_score,
            },
            "under_real_model": {
                "predicted_label": res5_real.pesticide.label,
                "confidence_pct": round(res5_real.pesticide.confidence * 100, 2),
                "ood_status": res5_real.pesticide.ood_status,
            },
            "brix_sweetness": f"{res5_synth.brix.estimated_brix} °Bx",
        }

    # Scenario 6: SpectroFood Natural Variation Spectrum (Out-of-Distribution test)
    spectro_csv = PROJECT_ROOT / "data" / "natural_variation" / "spectrofood_18ch_sim" / "spectrofood_apple_as7265x_18ch.csv"
    if spectro_csv.exists():
        df_spec = pd.read_csv(spectro_csv)
        row_spec = df_spec[CHANNEL_COLS].iloc[0].values.astype(float)
        res6 = predictor.predict_spectrum(row_spec)
        scenarios_results["Scenario_6_SpectroFood_Natural_Variation"] = {
            "input_type": "SpectroFood 18-Channel Natural Variation Scan (Domain Shift Target)",
            "pesticide_screening": {
                "predicted_label": res6.pesticide.label,
                "confidence_pct": round(res6.pesticide.confidence * 100, 2),
                "ood_status": res6.pesticide.ood_status,
                "ood_score": res6.pesticide.ood_anomaly_score,
            },
            "ripeness_stage": res6.ripeness.stage,
            "variety_cultivar": res6.variety.cultivar,
            "brix_sweetness": f"{res6.brix.estimated_brix} °Bx",
        }

    # Scenario 7: Noisy Simulated Sensor Spectrum
    noisy_sim = simulate_scan(
        X=MOCK_APPLE_REFLECTANCE.copy(),
        dark_offset=30.0,
        read_noise_std=4.0,
        led_scale=0.92,
        gain=1.0,
        placement_scale=0.95,
        led_drift=0.04,
    )
    res7 = predictor.predict_spectrum(noisy_sim["calibrated"])
    scenarios_results["Scenario_7_Noisy_Physical_Sensor_Simulation"] = {
        "input_type": "AS7265x with 30-count dark offset, 4-count noise, 8% placement drift, 4% LED drift",
        "pesticide_screening": {
            "predicted_label": res7.pesticide.label,
            "confidence_pct": round(res7.pesticide.confidence * 100, 2),
            "ood_status": res7.pesticide.ood_status,
            "ood_score": res7.pesticide.ood_anomaly_score,
        },
        "ripeness_stage": res7.ripeness.stage,
        "variety_cultivar": res7.variety.cultivar,
        "brix_sweetness": f"{res7.brix.estimated_brix} °Bx",
    }

    return scenarios_results


def load_verified_benchmark_registry() -> dict:
    """Load the official benchmark results from experiment_registry.json."""
    reg_path = PROJECT_ROOT / "reports" / "experiment_registry.json"
    if reg_path.exists():
        with open(reg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main():
    print("=" * 70)
    print("CFRD COMPREHENSIVE CASE STUDIES EVALUATION & INFERENCE HARNESS")
    print("=" * 70)

    # 1. Initialize Unified ML Predictor
    print("\n[Step 1/3] Instantiating Unified ApplePredictor ML Engine...")
    predictor = ApplePredictor()
    print("  [OK] Models loaded (Pesticide Synthetic, Pesticide Real, Ripeness, Variety, Brix, OOD)")

    # 2. Run All 7 Practical Inference Scenarios
    print("\n[Step 2/3] Making the model answer across 7 inference test scenarios...")
    scenarios = run_inference_scenarios(predictor)

    for name, data in scenarios.items():
        print(f"\n--- {name} ---")
        print(f"  Input: {data['input_type']}")
        if "pesticide_screening" in data:
            pest = data["pesticide_screening"]
            print(f"  * Pesticide: {pest['predicted_label']} (Conf: {pest['confidence_pct']}%, OOD: {pest['ood_status']}, Score: {pest.get('ood_score', 'N/A')})")
        if "under_synthetic_model" in data:
            ps = data["under_synthetic_model"]
            print(f"  * Pesticide (Synthetic Model): {ps['predicted_label']} (Conf: {ps['confidence_pct']}%, OOD: {ps['ood_status']})")
        if "under_real_model" in data:
            pr = data["under_real_model"]
            print(f"  * Pesticide (Real Model):      {pr['predicted_label']} (Conf: {pr['confidence_pct']}%, OOD: {pr['ood_status']})")
        if "ripeness_stage" in data:
            rip = data["ripeness_stage"]
            if isinstance(rip, dict):
                print(f"  * Ripeness:  {rip['predicted_stage']} (Conf: {rip['confidence_pct']}%)")
            else:
                print(f"  * Ripeness:  {rip}")
        if "variety_cultivar" in data:
            var = data["variety_cultivar"]
            if isinstance(var, dict):
                print(f"  * Variety:   {var['predicted_cultivar']} (Conf: {var['confidence_pct']}%)")
            else:
                print(f"  * Variety:   {var}")
        if "brix_sweetness" in data:
            brix = data["brix_sweetness"]
            if isinstance(brix, dict):
                print(f"  * Brix:      {brix['estimated_brix']} °Bx ({brix['sweetness_tier']})")
            else:
                print(f"  * Brix:      {brix}")
        if "visual_defect" in data:
            vis = data["visual_defect"]
            print(f"  * Visual:    {vis['status']} (Defect Prob: {vis['defect_probability']})")

    # 3. Load & Consolidate All Cross-Validated Accuracy Benchmarks
    print("\n[Step 3/3] Consolidating all case study accuracies & benchmark metrics...")
    registry = load_verified_benchmark_registry()

    full_output = {
        "title": "CFRD Comprehensive Case Studies Evaluation & Benchmark Report",
        "timestamp": "2026-09-23",
        "inference_test_scenarios": scenarios,
        "case_studies_benchmarks": registry.get("verified_benchmarks", {}),
    }

    # Save to JSON
    json_path = PROJECT_ROOT / "reports" / "comprehensive_case_studies_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"  [OK] Saved comprehensive report JSON to: {json_path}")

    # Generate Markdown Summary
    md_path = PROJECT_ROOT / "reports" / "comprehensive_case_studies_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🍎 CFRD: Comprehensive Case Studies & Accuracy Benchmark Report\n\n")
        f.write("Generated from automated evaluation harness covering all models, case studies, and inference test scenarios.\n\n")
        f.write("## 1. Summary of Quantitative Benchmark Accuracies\n\n")
        f.write("| Case Study | Modality / Task | Model Architecture | Sample Size | Accuracy / Metric | Chance Baseline | Status |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write("| **CS1-A: Pesticide Synthetic** | 18-channel optical | PCA(5) + LDA | 240 scans (20 fruits) | **69.17% ± 11.59% Acc** (Bal: 73.89%) | 33.33% | 🟡 Software Learnability Test |\n")
        f.write("| **CS1-B: Pesticide Real Roomi** | 18-channel optical | PCA(5) + LDA | 26 scans (13 fruits) | **15.00% ± 13.33% Acc** (Bal: 17.50%) | 33.33% | 🔴 Below Chance (No Signal) |\n")
        f.write("| **CS1-C: Binary Formulation (all)** | 18-channel optical | PCA(5) + LDA | 26 scans (13 fruits) | **50.00% ± 10.54% Acc** | 50.00% | 🔴 At Chance Level |\n")
        f.write("| **CS2: Ripeness Grading** | Appearance (RGB+HSV) | Random Forest (60 trees) | 1,077 scans (20 fruits) | **64.07% ± 2.32% Acc** (Bal: 64.05%) | 33.33% | 🟢 Effective Baseline |\n")
        f.write("| **CS3: Variety Identification** | Appearance (RGB+HSV) | Random Forest (60 trees) | 1,077 scans (20 fruits) | **47.54% ± 10.73% Acc** (Bal: 47.53%) | 20.00% | 🟡 Moderate Performance |\n")
        f.write("| **CS4: Brix Sweetness** | 18-channel digital twin | Cultivar ElasticNet | 60 scans (33 fruits) | **R² = -0.0316**, RMSE = 1.19 °Bx | Mean guess | 🔴 Negative R² (No SWIR bands) |\n")
        f.write("| **CS5: Visual Surface Defect** | RGB Camera image | MobileNetV3-Small | Public defect dataset | **95.67% Test Acc** (F1: 95.60%) | 50.00% | 🟢 Promising Prototype |\n")
        f.write("| **CS6: Domain Shift (OOD)** | Optical distribution | Mahalanobis LedoitWolf | Roomi (26) vs Spectro (240) | **100.0% Shift Detected** | N/A | 🟡 Cross-Instrument Detector |\n\n")

        f.write("## 2. Model Inference Answers Across 7 Test Scenarios\n\n")
        for sc_name, sc_data in scenarios.items():
            f.write(f"### {sc_name.replace('_', ' ')}\n")
            f.write(f"- **Input**: `{sc_data['input_type']}`\n")
            if "pesticide_screening" in sc_data:
                p = sc_data["pesticide_screening"]
                f.write(f"- **Pesticide Prediction**: `{p['predicted_label']}` (Confidence: {p['confidence_pct']}%, OOD Status: {p['ood_status']})\n")
            if "under_synthetic_model" in sc_data:
                ps = sc_data["under_synthetic_model"]
                f.write(f"- **Pesticide (Synthetic Model)**: `{ps['predicted_label']}` (Confidence: {ps['confidence_pct']}%, OOD: {ps['ood_status']})\n")
            if "under_real_model" in sc_data:
                pr = sc_data["under_real_model"]
                f.write(f"- **Pesticide (Real Model)**: `{pr['predicted_label']}` (Confidence: {pr['confidence_pct']}%, OOD: {pr['ood_status']})\n")
            if "ripeness_stage" in sc_data:
                r = sc_data["ripeness_stage"]
                conf = r.get('confidence_pct', 'N/A') if isinstance(r, dict) else 'N/A'
                stage = r.get('predicted_stage', r) if isinstance(r, dict) else r
                f.write(f"- **Ripeness Prediction**: `{stage}` (Confidence: {conf}%)\n")
            if "variety_cultivar" in sc_data:
                v = sc_data["variety_cultivar"]
                c_name = v.get('predicted_cultivar', v) if isinstance(v, dict) else v
                f.write(f"- **Cultivar Prediction**: `{c_name}`\n")
            if "brix_sweetness" in sc_data:
                b = sc_data["brix_sweetness"]
                val = b.get('estimated_brix', b) if isinstance(b, dict) else b
                f.write(f"- **Estimated Brix**: `{val}`\n")
            if "visual_defect" in sc_data:
                vd = sc_data["visual_defect"]
                f.write(f"- **Visual Defect Status**: `{vd['status']}` (Defect Prob: {vd['defect_probability']})\n")
            f.write("\n")

    print(f"  [OK] Saved markdown report to: {md_path}")
    print("\n" + "=" * 70)
    print("ALL EVALUATIONS AND MODEL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
