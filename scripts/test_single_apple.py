"""
test_single_apple.py

Command-line and module tool to test any single apple image (uploaded or sample).
Simulates the AS7265x optical sensor response, predicts treatment condition,
evaluates ripeness, variety, and visual defect risk without physical hardware.

Usage:
    python scripts/test_single_apple.py --image "d:\\CFRD\\DATA\\Dataset\\hyperspectral-fruit-val\\Good and bad classification of apple\\Good_\\apple_0.jpg"
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import glob
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

import numpy as np
import pandas as pd
from PIL import Image
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from constants import (
    AS7265X_CENTRES_NM,
    CHANNEL_COLS,
    CLASS_LABELS,
    DISCLAIMER_BANNER,
    SYNTHETIC_LABEL,
    SPECTROFOOD_18CH_PATH,
    SYNTHETIC_CSV_PATH,
)
from ood import MahalanobisOOD

MODELS_DIR_TREATMENT = r"d:\CFRD\project\models\treatment_condition"
MODELS_DIR_QUALITY = r"d:\CFRD\project\models\quality"


def find_latest_model(folder: str, pattern: str = "*.joblib"):
    matches = glob.glob(os.path.join(folder, pattern))
    if not matches:
        return None
    matches.sort(key=os.path.getmtime)
    return matches[-1]


def extract_visual_features_for_ripeness(im: Image.Image) -> np.ndarray:
    """Extract 36 color moments and histogram features matching ripeness classifier."""
    im_small = im.resize((128, 128)).convert("RGB")
    rgb = np.array(im_small, dtype=float) / 255.0

    rgb_means = rgb.mean(axis=(0, 1))
    rgb_stds  = rgb.std(axis=(0, 1))

    hsv = im_small.convert("HSV")
    hsv_arr = np.array(hsv, dtype=float) / 255.0
    hsv_means = hsv_arr.mean(axis=(0, 1))
    hsv_stds  = hsv_arr.std(axis=(0, 1))

    h_hist, _ = np.histogram(hsv_arr[:, :, 0], bins=8, range=(0, 1), density=True)
    s_hist, _ = np.histogram(hsv_arr[:, :, 1], bins=8, range=(0, 1), density=True)
    v_hist, _ = np.histogram(hsv_arr[:, :, 2], bins=8, range=(0, 1), density=True)

    return np.concatenate([rgb_means, rgb_stds, hsv_means, hsv_stds, h_hist, s_hist, v_hist])


def estimate_18ch_spectrum_from_image(im: Image.Image) -> np.ndarray:
    """
    Reconstructs an 18-channel AS7265x reflectance spectrum from an RGB apple photo
    using apple surface optical pigments and biological NIR priors from SpectroFood.
    """
    im_rgb = im.convert("RGB").resize((100, 100))
    arr = np.array(im_rgb, dtype=float) / 255.0

    # Central circular ROI to avoid background
    H, W, _ = arr.shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= (0.35 * min(H, W)) ** 2
    roi = arr[mask]

    r_mean = float(roi[:, 0].mean())
    g_mean = float(roi[:, 1].mean())
    b_mean = float(roi[:, 2].mean())

    # Base typical apple reflectance prior from real SpectroFood data
    # Apple curve: low blue (~0.12), small green peak (~0.25), chlorophyll dip ~680nm (~0.35),
    # steep red edge at 705nm, high NIR plateau (0.65-0.85)
    base_curve = np.array([
        0.10, 0.12, 0.14, 0.18, 0.22, 0.26,  # 410, 435, 460, 485, 510, 535 nm (Blue/Cyan/Green)
        0.28, 0.32, 0.40, 0.48, 0.42, 0.62,  # 560, 585, 610, 645, 680, 705 nm (Yellow/Orange/Red/Red-edge)
        0.72, 0.75, 0.78, 0.77, 0.72, 0.64   # 730, 760, 810, 860, 900, 940 nm (NIR plateau + water dip)
    ])

    # Modulate visible bands with actual RGB channels
    spec = base_curve.copy()
    spec[0:4]  *= (b_mean / 0.15) * 0.8 + 0.2    # Blue/Violet channels
    spec[4:8]  *= (g_mean / 0.25) * 0.8 + 0.2    # Green/Yellow channels
    spec[8:12] *= (r_mean / 0.45) * 0.8 + 0.2    # Orange/Red/Red-edge

    # NIR channels correlate with overall skin scattering & red intensity
    nir_mod = (r_mean * 0.6 + g_mean * 0.4) / 0.35
    spec[12:] *= np.clip(nir_mod, 0.7, 1.4)

    # Normalize into valid calibrated reflectance range [0.05, 1.2]
    spec = np.clip(spec, 0.05, 1.3)
    return spec


def test_apple(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print("=" * 65)
    print("🍎 CFRD APPLE TESTING SUITE (NO HARDWARE REQUIRED)")
    print(f"Target Image: {os.path.basename(image_path)}")
    print("=" * 65)

    with Image.open(image_path) as im:
        # -------------------------------------------------------------
        # 1. External Visual Quality (Defect Analysis)
        # -------------------------------------------------------------
        # Check if trained MobileNetV3 deep learning model exists
        mobilenet_path = r"d:\CFRD\project\models\camera\camera_quality_mobilenet.pth"
        used_mobilenet = False
        if TORCH_AVAILABLE and os.path.exists(mobilenet_path):
            try:
                cp = torch.load(mobilenet_path, map_location="cpu", weights_only=False)
                net = models.mobilenet_v3_small(weights=None)
                net.classifier[3] = nn.Linear(net.classifier[3].in_features, 2)
                net.load_state_dict(cp["model_state_dict"])
                net.eval()
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                t_im = tf(im.convert("RGB")).unsqueeze(0)
                with torch.no_grad():
                    logits = net(t_im)
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                defect_prob = float(probs[1])
                visual_label = "Possible defect / blemish" if defect_prob > 0.50 else "Good visual surface"
                used_mobilenet = True
            except Exception:
                used_mobilenet = False

        if not used_mobilenet:
            im_small = im.resize((100, 100)).convert("RGB")
            rgb_arr = np.array(im_small, dtype=float) / 255.0
            lum = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
            blemish_ratio = float((lum < 0.22).mean())
            defect_prob = float(np.clip(blemish_ratio * 4.5, 0.05, 0.95))
            visual_label = "Possible defect / blemish" if defect_prob > 0.40 else "Good visual surface"

        backend_tag = "(MobileNetV3 95.7% Acc)" if used_mobilenet else "(Heuristic)"
        print(f"\n[1] External Visual Quality {backend_tag}:")
        print(f"    Status:     {visual_label}")
        print(f"    Confidence: {max(defect_prob, 1.0 - defect_prob)*100:.1f}%")

        # -------------------------------------------------------------
        # 2. Ripeness & Variety Classification
        # -------------------------------------------------------------
        rip_model_path = find_latest_model(MODELS_DIR_QUALITY, "ripeness_classifier_*.joblib")
        var_model_path = find_latest_model(MODELS_DIR_QUALITY, "variety_classifier_*.joblib")

        vfeats = extract_visual_features_for_ripeness(im).reshape(1, -1)

        if rip_model_path:
            rip_obj = joblib.load(rip_model_path)
            rip_pred_idx = rip_obj["pipeline"].predict(vfeats)[0]
            rip_proba = rip_obj["pipeline"].predict_proba(vfeats)[0]
            rip_classes = rip_obj["classes"]
            print(f"\n[2] Ripeness Grading:")
            print(f"    Stage:      {rip_classes[rip_pred_idx]}")
            print(f"    Confidence: {rip_proba[rip_pred_idx]*100:.1f}%")

        if var_model_path:
            var_obj = joblib.load(var_model_path)
            var_pred_idx = var_obj["pipeline"].predict(vfeats)[0]
            var_proba = var_obj["pipeline"].predict_proba(vfeats)[0]
            var_classes = var_obj["classes"]
            print(f"\n[3] Variety Classification:")
            print(f"    Predicted:  {var_classes[var_pred_idx].title()}")
            print(f"    Confidence: {var_proba[var_pred_idx]*100:.1f}%")

        # -------------------------------------------------------------
        # 3. Virtual AS7265x Sensor Simulation & Treatment Screening
        # -------------------------------------------------------------
        sim_spec = estimate_18ch_spectrum_from_image(im)
        print(f"\n[4] AS7265x 18-Channel Optical Reconstruction:")
        print(f"    Simulated 18 channels (410-940 nm) from surface pigments.")

        treatment_model_path = find_latest_model(MODELS_DIR_TREATMENT, "treatment_3class_pca_lda_*.joblib")
        if treatment_model_path:
            t_obj = joblib.load(treatment_model_path)
            t_pipe = t_obj["pipeline"]
            
            # Predict
            pred_class = int(t_pipe.predict(sim_spec.reshape(1, -1))[0])
            pred_proba = t_pipe.predict_proba(sim_spec.reshape(1, -1))[0]
            pred_label = CLASS_LABELS.get(pred_class, f"Class {pred_class}")
            max_conf = float(pred_proba[pred_class])

            # OOD check
            df_roomi = pd.read_csv(SYNTHETIC_CSV_PATH)
            X_in = df_roomi[CHANNEL_COLS].values
            ood_scorer = MahalanobisOOD(n_pca_components=5).fit(X_in)
            ood_dist = float(ood_scorer.score(sim_spec.reshape(1, -1))[0])
            p95_thresh = ood_scorer.threshold_at_percentile(95.0)
            is_uncertain = (max_conf < 0.65) or (ood_dist > p95_thresh)

            print(f"\n[5] Treatment-Condition Model Decision:")
            print(f"    Predicted Condition: {pred_label}")
            print(f"    Decision Confidence: {max_conf * 100:.1f}%")
            print(f"    OOD Distance:        {ood_dist:.2f} (95% Thresh: {p95_thresh:.2f})")
            if is_uncertain:
                print(f"    Decision Status:     ⚠️ UNCERTAIN / REJECT SCAN (Recommend re-scanning)")
            else:
                print(f"    Decision Status:     ✅ ACCEPTED HIGH-CONFIDENCE INFERENCE")

    print("\n" + "=" * 65)
    print(f"Notice: {DISCLAIMER_BANNER}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test any apple image on CFRD screening pipeline.")
    parser.add_argument("--image", type=str, default="", help="Path to apple image file (jpg/png)")
    args = parser.parse_args()

    # Default to sample if not provided
    img_path = args.image
    if not img_path:
        good_folder = r"d:\CFRD\DATA\Dataset\hyperspectral-fruit-val\Good and bad classification of apple\Good_"
        files = [os.path.join(good_folder, f) for f in os.listdir(good_folder) if f.lower().endswith(('.jpg', '.png'))]
        if files:
            img_path = files[0]
            print(f"No --image provided. Using sample: {os.path.basename(img_path)}")
        else:
            print("ERROR: No sample image found.")
            sys.exit(1)

    test_apple(img_path)
