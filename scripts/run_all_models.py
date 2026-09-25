"""
run_all_models.py  --  CFRD comprehensive model evaluation runner
Executes every available model with real inputs and reports actual outputs.
"""
import sys, json, warnings, time
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
from model_registry import get_model_path, load_manifest

SEP = "=" * 70

# -- Representative 18-channel apple spectrum (simulated Gala reflectance)
SPECTRUM_18CH = np.array([
    0.09, 0.10, 0.11, 0.14, 0.22, 0.38,
    0.44, 0.42, 0.40, 0.35, 0.22, 0.48,
    0.72, 0.76, 0.79, 0.81, 0.82, 0.83,
], dtype=np.float32).reshape(1, -1)

# Representative 17-adjacent-diff feature
ADJ_DIFF_17 = np.diff(SPECTRUM_18CH[0]).reshape(1, -1)

# Representative 36-feature colour vector (HSV + LAB + stats for ripeness/variety)
COLOUR_36 = np.array([
    # H, S, V (×3 regions), L, a, b (×3 regions), std/mean/skew per channel
    0.12, 0.78, 0.85,  0.10, 0.80, 0.82,  0.13, 0.76, 0.84,
    68.5, -15.2, 42.1,  70.1, -14.8, 41.5,  67.9, -15.5, 43.0,
    0.82, 0.03,  0.78, 0.04,  0.85, 0.02,
    0.78, 0.03,  0.76, 0.04,  0.79, 0.02,
    0.01, 0.02,  0.01, 0.01,  0.02, 0.01,
], dtype=np.float32).reshape(1, -1)

results = {}

print(SEP)
print("CFRD -- Complete Model Evaluation Report")
print(f"Python: {sys.version.split()[0]} | sklearn: ", end="")
import sklearn; print(sklearn.__version__)
print(SEP)

# -----------------------------------------------------------------------------
# 1. PESTICIDE / TREATMENT MODELS
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 1 -- PESTICIDE / TREATMENT SCREENING (AS7265x 18-ch)")
print("-"*70)

treatment_map = {0: "Clean/Control", 1: "Monostar (Insecticide)", 2: "Nativo (Fungicide)"}

for model_key, label in [("treatment_3class_synthetic", "Synthetic-trained (240 rows, PCA-LDA)"),
                          ("treatment_3class_real",      "Real-trained (26 rows, Roomi extract)")]:
    p = get_model_path(model_key)
    if p and p.exists():
        obj = joblib.load(p)
        pipe = obj["pipeline"]
        m = obj.get("metrics", {})
        pred = pipe.predict(SPECTRUM_18CH)[0]
        proba = pipe.predict_proba(SPECTRUM_18CH)[0] if hasattr(pipe, "predict_proba") else None
        print(f"\n  [{label}]")
        print(f"  File        : {p.name}")
        print(f"  Model type  : {obj.get('model_type', 'N/A')}")
        print(f"  Train acc   : {m.get('acc', m.get('accuracy', 'N/A'))}")
        print(f"  Train F1    : {m.get('f1', m.get('macro_f1', 'N/A'))}")
        print(f"  Val method  : {m.get('val_method', 'GroupKFold 5-fold by fruit_id')}")
        print(f"  -> Input     : 18-ch spectrum {list(np.round(SPECTRUM_18CH[0][:6], 2))}...")
        print(f"  -> Predicted : {treatment_map.get(pred, pred)}")
        if proba is not None:
            print(f"  -> Probas    : Clean={proba[0]:.3f}, Monostar={proba[1]:.3f}, Nativo={proba[2]:.3f}")
        results[model_key] = {"acc": m.get("acc", "N/A"), "prediction": treatment_map.get(pred, pred)}
    else:
        print(f"\n  [{label}] -- FILE NOT FOUND at {p}")

# -----------------------------------------------------------------------------
# 2. RIPENESS MODEL
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 2 -- RIPENESS CLASSIFIER (Colour/HSV features, 36-dim)")
print("-"*70)

p = get_model_path("ripeness_classifier")
if p and p.exists():
    obj = joblib.load(p)
    pipe = obj["pipeline"]
    m = obj.get("metrics", {})
    classes = obj.get("classes", ["Under-ripe", "Ripe", "Over-ripe"])
    pred_idx = pipe.predict(COLOUR_36)[0]
    proba = pipe.predict_proba(COLOUR_36)[0]
    print(f"\n  File        : {p.name}")
    print(f"  Model type  : {m.get('model_type', 'SVM/RF colour model')}")
    print(f"  Train acc   : {m.get('acc', 'N/A'):.4f}  ({m.get('acc', 0)*100:.2f}%)")
    print(f"  Balanced acc: {m.get('bal_acc', 'N/A')}")
    print(f"  Macro F1    : {m.get('f1', 'N/A')}")
    print(f"  Val method  : GroupKFold 5-fold by fruit_id")
    print(f"  Classes     : {classes}")
    print(f"  n_features  : {pipe[0].n_features_in_} (colour/HSV, NOT AS7265x spectral)")
    print(f"  ⚠ NOT the '>90%' MobileNetV3 image result -- that is a different model")
    print(f"  -> Predicted : {classes[pred_idx]} (idx={pred_idx})")
    print(f"  -> Probas    : {dict(zip(classes, np.round(proba, 3)))}")
    results["ripeness"] = {"acc": m.get("acc"), "prediction": classes[pred_idx]}
else:
    print(f"\n  Ripeness model NOT FOUND at {p}")

# -----------------------------------------------------------------------------
# 3. VARIETY / CULTIVAR MODEL
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 3 -- CULTIVAR / VARIETY CLASSIFIER (Colour/HSV features, 36-dim)")
print("-"*70)

p = get_model_path("variety_classifier")
if p and p.exists():
    obj = joblib.load(p)
    pipe = obj["pipeline"]
    m = obj.get("metrics", {})
    classes = obj.get("classes", [])
    pred_idx = pipe.predict(COLOUR_36)[0]
    proba = pipe.predict_proba(COLOUR_36)[0]
    print(f"\n  File        : {p.name}")
    print(f"  Train acc   : {m.get('acc', 'N/A'):.4f}  ({m.get('acc', 0)*100:.2f}%)")
    print(f"  Balanced acc: {m.get('bal_acc', 'N/A')}")
    print(f"  Macro F1    : {m.get('f1', 'N/A')}")
    print(f"  Classes     : {classes}")
    print(f"  n_features  : {pipe[0].n_features_in_} (colour/HSV)")
    print(f"  -> Predicted : {classes[pred_idx] if classes else pred_idx}")
    results["variety"] = {"acc": m.get("acc"), "prediction": classes[pred_idx] if classes else pred_idx}
else:
    print(f"\n  Variety model NOT FOUND at {p}")

# -----------------------------------------------------------------------------
# 4. BRIX / SWEETNESS MODELS
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 4 -- BRIX / SWEETNESS REGRESSION (AS7265x twin, 18-ch)")
print("-"*70)

brix_paths = {
    "brix_best (quality/)":  PROJECT_ROOT / "models" / "quality" / "brix_as7265x_18channel_best.joblib",
    "brix_elasticnet (brix/)": PROJECT_ROOT / "models" / "brix" / "as7265x_18ch_twin_best_elasticnet.joblib",
    "brix_ridge (brix/)":    PROJECT_ROOT / "models" / "brix" / "ridge_18ch_clean.joblib",
}

for label, p in brix_paths.items():
    if p.exists():
        obj = joblib.load(p)
        pipe = obj["pipeline"] if isinstance(obj, dict) else obj
        m = obj.get("metrics", {}) if isinstance(obj, dict) else {}
        pred = pipe.predict(SPECTRUM_18CH)[0]
        print(f"\n  [{label}]")
        print(f"  File        : {p.name}")
        print(f"  R2          : {m.get('r2', 'N/A')}")
        print(f"  RMSE        : {m.get('rmse', 'N/A')}")
        print(f"  MAE         : {m.get('mae', 'N/A')}")
        print(f"  ⚠ STATUS    : UNDER RECONCILIATION -- Gen1 R2=−0.03 (authoritative)")
        print(f"  -> Predicted : {pred:.2f} degBx")
        tier = "High" if pred >= 14 else ("Medium" if pred >= 11 else "Low")
        print(f"  -> Tier      : {tier} sweetness")
        results[f"brix_{label}"] = {"r2": m.get("r2", "N/A"), "predicted_brix": round(pred, 2)}
    else:
        print(f"\n  [{label}] -- NOT FOUND: {p.name}")

# -----------------------------------------------------------------------------
# 5. OOD DETECTOR
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 5 -- OUT-OF-DISTRIBUTION DETECTOR (Mahalanobis, PCA 5-comp)")
print("-"*70)

synth_csv = PROJECT_ROOT / "data" / "pesticide_screening" / "roomi_synthetic_simulation" / "roomi_synthetic_18ch.csv"
if synth_csv.exists():
    import pandas as pd
    from ood_detector import MahalanobisOOD
    df = pd.read_csv(synth_csv)
    ch_cols = [c for c in df.columns if c.startswith("ch_")]
    X_train = df[ch_cols].values
    ood = MahalanobisOOD(n_pca_components=5).fit(X_train)
    thr = ood.threshold_at_percentile(95.0)

    # Test: in-distribution (apple spectrum)
    score_apple = ood.score(SPECTRUM_18CH)[0]
    is_ood_apple = score_apple > thr

    # Test: OOD (random noise)
    rng = np.random.RandomState(1)
    noise = rng.rand(1, 18).astype(np.float32) * 5.0
    score_noise = ood.score(noise)[0]
    is_ood_noise = score_noise > thr

    print(f"\n  Training data: {len(X_train)} rows, 18 channels")
    print(f"  PCA components: 5  |  Threshold (95th pct): {thr:.4f}")
    print(f"  -> Apple spectrum score : {score_apple:.4f}  ({'OOD ⚠' if is_ood_apple else 'In-distribution ✓'})")
    print(f"  -> Random noise score   : {score_noise:.4f}  ({'OOD ⚠' if is_ood_noise else 'In-distribution ✓'})")
    results["ood_detector"] = {"threshold_p95": round(thr, 4), "apple_ood": is_ood_apple, "noise_ood": is_ood_noise}
else:
    print(f"\n  Synthetic CSV not found -- OOD detector skipped")

# -----------------------------------------------------------------------------
# 6. PREDICTOR INTEGRATION (end-to-end)
# -----------------------------------------------------------------------------
print("\n" + "-"*70)
print("TRACK 6 -- FULL PIPELINE: ApplePredictor end-to-end")
print("-"*70)

try:
    from predictor import ApplePredictor
    t0 = time.time()
    predictor = ApplePredictor(default_pesticide_model="synthetic")
    t_load = time.time() - t0

    t0 = time.time()
    result = predictor.predict_from_spectrum(SPECTRUM_18CH[0].tolist())
    t_infer = time.time() - t0

    print(f"\n  Model load time  : {t_load:.2f}s")
    print(f"  Inference time   : {t_infer:.4f}s")
    print(f"\n  === FULL PREDICTION OUTPUT ===")
    print(f"  Pesticide: {result.pesticide.label}  (conf={result.pesticide.confidence:.1%}, OOD={result.pesticide.ood_status})")
    print(f"  Ripeness : {result.ripeness.stage}   (conf={result.ripeness.confidence:.1%})")
    print(f"  Cultivar : {result.variety.cultivar.title()}   (conf={result.variety.confidence:.1%})")
    print(f"  Brix est : {result.brix.estimated_brix:.2f} degBx  ({result.brix.sweetness_tier})")
    print(f"  Brix note: {result.brix.scientific_warning[:80]}...")
    results["end_to_end"] = {
        "load_time_s": round(t_load, 2),
        "infer_time_s": round(t_infer, 4),
        "pesticide": result.pesticide.label,
        "ripeness": result.ripeness.stage,
        "cultivar": result.variety.cultivar,
        "brix": result.brix.estimated_brix,
    }
except Exception as e:
    print(f"\n  ApplePredictor ERROR: {e}")
    import traceback; traceback.print_exc()

# -----------------------------------------------------------------------------
# LOAO BENCHMARK SUMMARY
# -----------------------------------------------------------------------------
print("\n" + SEP)
print("LOAO BENCHMARK RESULTS (from frozen FINAL_BASELINE.md)")
print(SEP)
print("""
  Result Type              | Value   | Notes
  -------------------------|---------|----------------------------------------------
  Random ROI CV (leaky)    | 82.65%  | Exploratory/leakage-affected -- DO NOT CITE
  LOAO per-ROI Macro-F1    | 56.14%  | Primary rigorous result
  LOAO scan-level (25-spot)| 75.00%  | High sensitivity (100%), low specificity (30%)
  LOAO AUROC               |  0.59   | Permutation p≈0.19 -- not significant
  Held-out test (P_MA1)    | 35-42%  | Pre-registered -- all clean ROIs misclassified
  AS7265x 17-diff twin     | 75.00%  | Same as HSI at scan level (simulation only)
  RGB cultivar baseline     | 63.44%  | StratifiedKFold -- incomparable protocol
  Ripeness (colour SVM)    | 64.07%  | AS7265x spectral, 3-class
  Cultivar (colour SVM)    | 47.54%  | AS7265x spectral, 5-class (vs 20% chance)
  Brix R2 (Gen1, verified) | −0.03   | AUTHORITATIVE -- worse than mean prediction
  Brix R2 (Gen2, unreconciled)| 0.43 | Larger dataset, external, NOT reproducible
""")

print(SEP)
print("RESULTS SUMMARY JSON")
print(SEP)
import json as _json
print(_json.dumps(results, indent=2, default=str))
print(SEP)
print("PHYSICAL SENSOR STATUS: 0 scans taken. All data is simulation or public HSI.")
print(SEP)
