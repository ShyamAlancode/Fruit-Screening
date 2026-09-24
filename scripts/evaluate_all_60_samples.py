"""
evaluate_all_60_samples.py
==========================
Comprehensive benchmark on the full expanded dataset (60 scans, 33 unique physical apples)
across all extracted batches:
  - Braeburn (12 scans, 3 physical apples)
  - Cox's Orange Pippin (21 scans, 12 physical apples)
  - Gala (27 scans, 18 physical apples)

Evaluates:
  1. Specim IQ 204-Band upper bound
  2. AS7265x 18-Channel Clean Digital Twin
  3. AS7265x 18-Channel Digital Twin + Sensor Noise (dark + read + LED)
  4. Both Standard 5-Fold CV and Physical Apple GroupKFold CV
  5. Per-Apple Averaged Regression (gold-standard multi-scan physical apple level)
"""

import re, json, warnings, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
ESSEX_ROOT     = Path(os.environ.get("CFRD_ESSEX_ROOT", PROJECT_ROOT.parent / "New folder"))
EXTRACTED_ROOT = ESSEX_ROOT / "extracted"
CSV_DIR        = ESSEX_ROOT
REPORTS_DIR    = PROJECT_ROOT / "reports"
MODELS_DIR     = PROJECT_ROOT / "models" / "brix"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

AS7265X_CENTRES_NM = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
]
AS7265X_SIGMA = 20.0 / 2.35482

CULTIVAR_PALETTE = {
    "Braeburn": "#2ECC71",
    "Cox's Orange Pippin": "#E67E22",
    "Gala": "#E74C3C"
}

def parse_hdr_wavelengths(hdr_path):
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", txt, re.DOTALL)
    return np.array([float(x.strip()) for x in m.group(1).split(",") if x.strip()])

def spectrum_to_18ch(spec204, wavelengths):
    out = np.zeros(18, dtype=np.float64)
    for k, centre in enumerate(AS7265X_CENTRES_NM):
        G = np.exp(-0.5 * ((wavelengths - centre) / AS7265X_SIGMA) ** 2)
        mask = G > 1e-4
        w = G[mask].sum()
        out[k] = float(np.dot(spec204[mask], G[mask]) / w) if w > 1e-10 else 0.0
    return out

def add_sensor_noise(ch18, rng, dark=0.005, read=0.004, led=0.015):
    n = ch18.copy()
    n += dark
    n *= (1.0 + rng.normal(0.0, led, 18))
    n += rng.normal(0.0, read, 18)
    return np.clip(n, 0.0, None)

def detect_three_apple_blobs(folder_dir):
    pngs = list(folder_dir.glob("*.png"))
    if not pngs: return []
    im = Image.open(pngs[0]).convert("RGB")
    arr = np.array(im)
    gray = arr.mean(axis=2)
    mask = gray > 25
    labeled, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labeled)

    bins = {"left": [], "mid": [], "right": []}
    for i, obj in enumerate(objects):
        sz = np.sum(labeled[obj] == (i + 1))
        y_s, x_s = obj
        yc = (y_s.start + y_s.stop) / 2.0
        xc = (x_s.start + x_s.stop) / 2.0
        if sz > 2500 and 80 < yc < 360:
            item = {
                "sz": int(sz), "xc": int(xc), "yc": int(yc),
                "x_start": int(x_s.start), "x_stop": int(x_s.stop),
                "y_start": int(y_s.start), "y_stop": int(y_s.stop),
            }
            if xc < 185: bins["left"].append(item)
            elif xc < 345: bins["mid"].append(item)
            else: bins["right"].append(item)

    chosen = []
    for k in ["left", "mid", "right"]:
        if bins[k]:
            chosen.append(max(bins[k], key=lambda x: x["sz"]))
    chosen.sort(key=lambda x: x["xc"])
    return chosen

def load_all_csvs():
    dfs = {}
    for name in ["data_NZ2023.csv", "data_NZ2024.csv", "data_UK2024.csv"]:
        p = CSV_DIR / name
        if p.exists(): dfs[name] = pd.read_csv(p)
    return dfs

def match_folder_to_csv_rows(folder_name, csvs):
    for csv_name, df in csvs.items():
        img_cols = [c for c in df.columns if "Image_folder" in c]
        for c in img_cols:
            m = df[df[c].astype(str).str.contains(folder_name, na=False)]
            if len(m) > 0:
                rows = []
                for _, r in m.iterrows():
                    rows.append({
                        "ID": str(r.get("ID", "")),
                        "Cultivar": str(r.get("Cultivar", "")),
                        "Brix": float(r.get("Brix", 0.0)),
                        "Avg_firmness": float(r.get("Avg_firmness", 0.0)) if pd.notna(r.get("Avg_firmness")) else 0.0,
                        "Side": c[-1] if len(c) > 0 else "A",
                    })
                rows.sort(key=lambda x: x["ID"])
                return rows
    return []

def extract_dataset():
    csvs = load_all_csvs()
    rng = np.random.default_rng(2024)

    X_204, X_18c, X_18n = [], [], []
    y_brix, cultivars, apple_ids = [], [], []
    meta = []

    extracted_batches = sorted([b for b in EXTRACTED_ROOT.iterdir() if b.is_dir()])
    for batch_dir in extracted_batches:
        folders = sorted([f for f in batch_dir.iterdir() if f.is_dir() and not f.name.startswith("results")])
        for folder_dir in folders:
            folder_name = folder_dir.name
            csv_rows = match_folder_to_csv_rows(folder_name, csvs)
            if not csv_rows: continue

            dats = sorted((folder_dir / "results").glob("REFLECTANCE_*.dat"))
            hdrs = sorted((folder_dir / "results").glob("REFLECTANCE_*.hdr"))
            if not dats or not hdrs: continue

            blobs = detect_three_apple_blobs(folder_dir)
            if len(blobs) < len(csv_rows): continue

            wl = parse_hdr_wavelengths(hdrs[0])
            cube = np.memmap(str(dats[0]), dtype=np.float32, mode="r", shape=(512, 512, 204))

            for i in range(min(len(blobs), len(csv_rows))):
                blob, row = blobs[i], csv_rows[i]
                if row["Brix"] <= 0.0: continue

                y1, y2 = blob["y_start"], blob["y_stop"]
                x1, x2 = blob["x_start"], blob["x_stop"]
                dy, dx = int((y2 - y1) * 0.30), int((x2 - x1) * 0.30)
                cy1, cy2 = max(y1 + dy, 0), min(y2 - dy, 512)
                cx1, cx2 = max(x1 + dx, 0), min(x2 - dx, 512)
                if cy2 <= cy1 or cx2 <= cx1:
                    cy1, cy2, cx1, cx2 = y1, y2, x1, x2

                spec204 = np.array(cube[cy1:cy2, cx1:cx2, :].mean(axis=(0, 1)), dtype=np.float64)
                ch18c = spectrum_to_18ch(spec204, wl)
                ch18n = add_sensor_noise(ch18c, rng)

                X_204.append(spec204)
                X_18c.append(ch18c)
                X_18n.append(ch18n)
                y_brix.append(row["Brix"])
                cultivars.append(row["Cultivar"])
                apple_ids.append(row["ID"])
                meta.append({
                    "apple_id": row["ID"],
                    "cultivar": row["Cultivar"],
                    "brix": row["Brix"],
                    "batch": batch_dir.name,
                    "folder": folder_name,
                    "side": row["Side"]
                })
            del cube

    return {
        "X_204": np.array(X_204),
        "X_18c": np.array(X_18c),
        "X_18n": np.array(X_18n),
        "y": np.array(y_brix),
        "cult": np.array(cultivars),
        "apples": np.array(apple_ids),
        "meta": meta
    }

def main():
    print("Extracting full multi-cultivar dataset...")
    data = extract_dataset()
    X204, X18c, X18n = data["X_204"], data["X_18c"], data["X_18n"]
    y, cult, apples = data["y"], data["cult"], data["apples"]
    N = len(y)
    unique_apples = np.unique(apples)
    print(f"Loaded N={N} scans across {len(unique_apples)} physical apples.")
    for c in sorted(set(cult)):
        sub = y[cult == c]
        n_a = len(set(apples[cult == c]))
        print(f"  {c}: {len(sub)} scans, {n_a} apples, Brix {sub.min():.2f} - {sub.max():.2f} (mean {sub.mean():.2f})")

    # Cultivar One-Hot
    le = LabelEncoder()
    cult_codes = le.fit_transform(cult)
    onehot = np.eye(len(le.classes_), dtype=np.float64)[cult_codes]

    X204_c = np.hstack([X204, onehot])
    X18c_c = np.hstack([X18c, onehot])
    X18n_c = np.hstack([X18n, onehot])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n--- Tuning PLSR components on 18-ch Clean + Cultivar (5-Fold CV) ---")
    for comp in range(1, 15):
        m = PLSRegression(n_components=comp, max_iter=2000)
        Xs = StandardScaler().fit_transform(X18c_c)
        yp = cross_val_predict(m, Xs, y, cv=kf)
        r2 = r2_score(y, yp)
        rmse = np.sqrt(mean_squared_error(y, yp))
        r, p = pearsonr(y, yp)
        print(f"  comp={comp:2d}: R2={r2:+.4f}  RMSE={rmse:.3f}  r={r:.4f} (p={p:.2e})")

    print("\n--- Tuning PLSR components on 204-Band + Cultivar (5-Fold CV) ---")
    for comp in range(1, 15):
        m = PLSRegression(n_components=comp, max_iter=2000)
        Xs = StandardScaler().fit_transform(X204_c)
        yp = cross_val_predict(m, Xs, y, cv=kf)
        r2 = r2_score(y, yp)
        rmse = np.sqrt(mean_squared_error(y, yp))
        r, p = pearsonr(y, yp)
        print(f"  comp={comp:2d}: R2={r2:+.4f}  RMSE={rmse:.3f}  r={r:.4f} (p={p:.2e})")

    print("\n--- Ridge & ElasticNet on 204-Band & 18-ch ---")
    for name, alpha in [("Ridge a=0.1", 0.1), ("Ridge a=0.5", 0.5), ("Ridge a=1.0", 1.0), ("Ridge a=5.0", 5.0)]:
        Xs18 = StandardScaler().fit_transform(X18c_c)
        yp18 = cross_val_predict(Ridge(alpha=alpha), Xs18, y, cv=kf)
        r2_18 = r2_score(y, yp18)
        rmse_18 = np.sqrt(mean_squared_error(y, yp18))
        r_18, _ = pearsonr(y, yp18)

        Xs204 = StandardScaler().fit_transform(X204_c)
        yp204 = cross_val_predict(Ridge(alpha=alpha), Xs204, y, cv=kf)
        r2_204 = r2_score(y, yp204)
        rmse_204 = np.sqrt(mean_squared_error(y, yp204))
        r_204, _ = pearsonr(y, yp204)

        print(f"  {name:12s} | 18ch: R2={r2_18:+.4f} RMSE={rmse_18:.3f} r={r_18:.4f} | 204band: R2={r2_204:+.4f} RMSE={rmse_204:.3f} r={r_204:.4f}")

    print("\n--- Per-Apple Level Averaging (33 Physical Apples) ---")
    # Average the scans per physical apple
    df_meta = pd.DataFrame(data["meta"])
    unique_a = df_meta["apple_id"].unique()
    X18_avg = []
    X204_avg = []
    y_avg = []
    cult_avg = []

    for a in unique_a:
        idxs = np.where(apples == a)[0]
        X18_avg.append(X18c[idxs].mean(axis=0))
        X204_avg.append(X204[idxs].mean(axis=0))
        y_avg.append(y[idxs[0]])
        cult_avg.append(cult[idxs[0]])

    X18_avg = np.array(X18_avg)
    X204_avg = np.array(X204_avg)
    y_avg = np.array(y_avg)
    cult_avg = np.array(cult_avg)

    le_avg = LabelEncoder()
    onehot_avg = np.eye(len(le.classes_), dtype=np.float64)[le.transform(cult_avg)]

    X18_avg_c = np.hstack([X18_avg, onehot_avg])
    X204_avg_c = np.hstack([X204_avg, onehot_avg])

    kf_avg = KFold(n_splits=5, shuffle=True, random_state=42)
    print(f"Evaluating on N={len(y_avg)} averaged physical apples:")
    for comp in range(1, 10):
        m = PLSRegression(n_components=comp, max_iter=2000)
        Xs = StandardScaler().fit_transform(X18_avg_c)
        yp = cross_val_predict(m, Xs, y_avg, cv=kf_avg)
        r2 = r2_score(y_avg, yp)
        rmse = np.sqrt(mean_squared_error(y_avg, yp))
        r, p = pearsonr(y_avg, yp)
        print(f"  PLSR comp={comp} (18ch physical apple avg): R2={r2:+.4f} RMSE={rmse:.3f} r={r:.4f}")

    for comp in range(1, 10):
        m = PLSRegression(n_components=comp, max_iter=2000)
        Xs = StandardScaler().fit_transform(X204_avg_c)
        yp = cross_val_predict(m, Xs, y_avg, cv=kf_avg)
        r2 = r2_score(y_avg, yp)
        rmse = np.sqrt(mean_squared_error(y_avg, yp))
        r, p = pearsonr(y_avg, yp)
        print(f"  PLSR comp={comp} (204band physical apple avg): R2={r2:+.4f} RMSE={rmse:.3f} r={r:.4f}")

if __name__ == "__main__":
    main()
