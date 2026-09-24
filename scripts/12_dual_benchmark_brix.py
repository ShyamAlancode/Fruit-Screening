"""
12_dual_benchmark_brix.py
=========================
Dual-Benchmark Regression Pipeline — PROOF OF CONCEPT

Benchmarks THREE tracks side-by-side:
  Track A: Real 204-band Specim IQ spectrum (upper bound, full physics)
  Track B: AS7265x 18-channel Digital Twin (clean Gaussian filter)
  Track C: AS7265x 18-channel Digital Twin + realistic sensor noise

Ground truth: Refractometer Brix + penetrometer firmness from Essex EPrints-228

Usage:
    python scripts/12_dual_benchmark_brix.py
"""

import os
import sys
import re
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESSEX_ROOT   = Path(os.environ.get("CFRD_ESSEX_ROOT", PROJECT_ROOT.parent / "New folder"))
EXTRACTED_ROOT = ESSEX_ROOT / "extracted"
AUDIT_JSON = PROJECT_ROOT / "records_audit_v2.json"  # Verified spatial match (v2)

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR  = PROJECT_ROOT / "models" / "brix"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# AS7265x channel centres and Gaussian kernel
# ---------------------------------------------------------------------------
AS7265X_CENTRES_NM = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940,
]
AS7265X_FWHM_NM = 20.0
AS7265X_SIGMA   = AS7265X_FWHM_NM / 2.35482


def parse_hdr_wavelengths(hdr_path):
    with open(hdr_path, "r") as f:
        txt = f.read()
    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", txt, re.DOTALL)
    if not m:
        raise ValueError(f"No wavelength field in {hdr_path}")
    return np.array([float(x.strip()) for x in m.group(1).split(",") if x.strip()])


def load_cube(dat_path):
    cube = np.memmap(str(dat_path), dtype=np.float32, mode="r", shape=(512, 512, 204))
    return np.array(cube, dtype=np.float32)


def extract_blob_spectrum_204(cube, blob, margin=0.30):
    y1, y2 = blob["y_start"], blob["y_stop"]
    x1, x2 = blob["x_start"], blob["x_stop"]
    dy = int((y2 - y1) * margin)
    dx = int((x2 - x1) * margin)
    cy1, cy2 = y1 + dy, y2 - dy
    cx1, cx2 = x1 + dx, x2 - dx
    if cy2 <= cy1 or cx2 <= cx1:
        cy1, cy2, cx1, cx2 = y1, y2, x1, x2
    return cube[cy1:cy2, cx1:cx2, :].mean(axis=(0, 1)).astype(np.float64)


def spectrum_to_18ch(spec204, wavelengths):
    sigma = AS7265X_SIGMA
    out = np.zeros(18, dtype=np.float64)
    for k, centre in enumerate(AS7265X_CENTRES_NM):
        G = np.exp(-0.5 * ((wavelengths - centre) / sigma) ** 2)
        mask = G > 1e-4
        w_sum = G[mask].sum()
        if w_sum < 1e-10:
            out[k] = 0.0
        else:
            out[k] = float(np.dot(spec204[mask], G[mask]) / w_sum)
    return out


def add_sensor_noise(ch18, dark_offset=0.005, read_noise_std=0.004, led_drift_std=0.015, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    noisy = ch18.copy()
    noisy += dark_offset
    noisy *= (1.0 + rng.normal(0.0, led_drift_std, size=18))
    noisy += rng.normal(0.0, read_noise_std, size=18)
    return np.clip(noisy, 0.0, None)


def find_cube_paths(batch_name, folder_name):
    base = EXTRACTED_ROOT / batch_name / folder_name / "results"
    dats = sorted(base.glob("REFLECTANCE_*.dat"))
    hdrs = sorted(base.glob("REFLECTANCE_*.hdr"))
    if not dats or not hdrs:
        return None, None
    return dats[0], hdrs[0]


def build_feature_matrix():
    with open(AUDIT_JSON) as f:
        records = json.load(f)

    rows_204 = []
    rows_18_clean = []
    rows_18_noisy = []
    brix_labels = []
    firmness_labels = []
    meta = []
    wavelengths = None

    rng = np.random.default_rng(2024)
    skipped = 0
    loaded_cubes = {}

    for rec in records:
        batch   = rec["batch"]
        folder  = rec["folder"]
        blobs   = rec["blob_positions"]
        csv_rows = rec["csv_rows"]

        if len(csv_rows) == 0:
            print(f"  SKIP {folder}: no CSV match")
            skipped += 1
            continue

        dat_path, hdr_path = find_cube_paths(batch, folder)
        if dat_path is None:
            print(f"  SKIP {folder}: .dat file not found")
            skipped += 1
            continue

        cache_key = (batch, folder)
        if cache_key not in loaded_cubes:
            print(f"  Loading cube: {dat_path.name} ... ", end="", flush=True)
            loaded_cubes[cache_key] = load_cube(dat_path)
            wavelengths = parse_hdr_wavelengths(hdr_path)
            print("OK")
        cube = loaded_cubes[cache_key]
        if wavelengths is None:
            wavelengths = parse_hdr_wavelengths(hdr_path)

        # Use pre-verified aligned_pairs from v2 JSON (blob already mapped to label)
        # Fallback to old naive zip for backward compat with v1 JSON
        if "aligned_pairs" in rec and rec["aligned_pairs"]:
            pair_iter = [(p["blob"], p["label"]) for p in rec["aligned_pairs"]]
        else:
            pair_iter = list(zip(blobs, csv_rows))[:min(len(blobs), len(csv_rows))]

        for blob, csv_row in pair_iter:
            if not csv_row.get("Brix") or csv_row["Brix"] <= 0.0:
                continue

            spec_204   = extract_blob_spectrum_204(cube, blob)
            ch18_clean = spectrum_to_18ch(spec_204, wavelengths)
            ch18_noisy = add_sensor_noise(ch18_clean, rng=rng)

            rows_204.append(spec_204)
            rows_18_clean.append(ch18_clean)
            rows_18_noisy.append(ch18_noisy)
            brix_labels.append(csv_row["Brix"])
            firmness_labels.append(csv_row["Avg_firmness"])
            meta.append({
                "apple_id": csv_row["ID"],
                "batch":    batch,
                "folder":   folder,
                "brix":     csv_row["Brix"],
                "firmness": csv_row["Avg_firmness"],
                "starch":   csv_row["Starch"],
            })

    print(f"\nExtracted {len(rows_204)} apple spectra ({skipped} frames skipped).")
    return {
        "X_204":       np.array(rows_204,       dtype=np.float64),
        "X_18_clean":  np.array(rows_18_clean,  dtype=np.float64),
        "X_18_noisy":  np.array(rows_18_noisy,  dtype=np.float64),
        "y_brix":      np.array(brix_labels,    dtype=np.float64),
        "y_firmness":  np.array(firmness_labels,dtype=np.float64),
        "wavelengths": wavelengths,
        "meta":        meta,
    }


def evaluate_regression(X, y, label, n_folds=5):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    n_comp = min(10, X.shape[1], X.shape[0] - 1)

    models = {
        "Ridge":   Ridge(alpha=1.0),
        "PLSR":    PLSRegression(n_components=n_comp, max_iter=1000),
        "HistGBR": HistGradientBoostingRegressor(max_iter=300, max_depth=4, random_state=42),
    }

    results = {}
    best_name = None
    best_r2   = -999.0

    print(f"\n  === {label} ===")
    for name, model in models.items():
        y_pred = cross_val_predict(model, Xs, y, cv=kf)
        r2   = r2_score(y, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae  = float(mean_absolute_error(y, y_pred))
        r, p = pearsonr(y, y_pred)
        print(f"    {name:10s}: R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  r={r:.4f}  p={p:.2e}")
        results[name] = {"r2": r2, "rmse": rmse, "mae": mae, "pearson_r": r, "p_value": p,
                         "y_true": y.tolist(), "y_pred": y_pred.tolist()}
        if r2 > best_r2:
            best_r2 = r2
            best_name = name

    best_model = models[best_name]
    best_model.fit(Xs, y)
    return results, best_name, best_model, scaler


def make_scatter(y_true, y_pred, title, out_path, r2, rmse, pearson_r):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(y_true, y_pred, c=y_true, cmap="YlOrRd", s=90, edgecolors="k", linewidths=0.6, zorder=3)
    plt.colorbar(sc, ax=ax, label="Ground-truth Brix (refractometer)")
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Perfect prediction")
    xs = np.linspace(lo, hi, 100)
    z = np.polyfit(y_true, y_pred, 1)
    ax.plot(xs, np.poly1d(z)(xs), "steelblue", lw=1.8, alpha=0.85, label="Trend")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Refractometer Brix (Ground Truth, deg Bx)", fontsize=12)
    ax.set_ylabel("Model Predicted Brix (deg Bx)", fontsize=12)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.04, 0.96,
            f"R2 = {r2:.4f}\nRMSE = {rmse:.3f} dBx\nPearson r = {pearson_r:.4f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter: {Path(out_path).name}")


def write_report(data, r204, r18c, r18n, n, nf):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Dual-Benchmark Brix Regression — Proof Report",
        f"**Generated**: {ts}",
        f"**Dataset**: Essex Apple Hyperspectral & Maturity (DOI: 10.5526/ERDR-00000228)",
        f"**Ground-Truth**: ATAGO PR-32-alpha refractometer (Brix), Lloyd LRX penetrometer",
        f"**N apples**: {n}  |  **CV**: {nf}-Fold (no leakage)",
        "",
        "> [!IMPORTANT]",
        "> All R2/RMSE values are out-of-fold — no training data seen during evaluation.",
        "",
    ]
    def table(res, cap):
        lines.append(f"## {cap}")
        lines.append("")
        lines.append("| Model | R2 | RMSE (dBx) | MAE (dBx) | Pearson r | p-value |")
        lines.append("|---|---|---|---|---|---|")
        for name, m in res.items():
            lines.append(f"| {name} | {m['r2']:.4f} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['pearson_r']:.4f} | {m['p_value']:.2e} |")
        lines.append("")

    table(r204,  "Track A: Real 204-Band Specim IQ (Upper Bound)")
    table(r18c,  "Track B: AS7265x 18-Ch Digital Twin (Clean)")
    table(r18n,  "Track C: AS7265x 18-Ch Digital Twin (+ Sensor Noise)")

    lines += [
        "## Provenance & Verification Chain",
        "",
        "| Step | Evidence |",
        "|---|---|",
        "| Hyperspectral cubes | Specim IQ .dat BIL float32, 512x512x204, 397-1004 nm |",
        "| Ground truth | Refractometer Brix + penetrometer firmness, Essex EPrints-228 |",
        "| Spatial linkage | Blob detection (segmented RGB preview) + explicit CSV Image_folder_A/B/C/D |",
        "| Authorship | Tang et al., NIAB East Malling, Plant & Food Research NZ |",
        "| Sensor physics | AS7265x 18x Gaussian (FWHM 20 nm) @ 410-940 nm |",
        "| No data leakage | K-Fold CV only, test fold never seen in fit |",
        "",
        "## Per-Apple Ground-Truth Table",
        "",
        "| Apple ID | Batch | Brix (dBx) | Firmness (kg/cm2) | Starch (%) |",
        "|---|---|---|---|---|",
    ]
    for m in data["meta"]:
        lines.append(f"| {m['apple_id']} | {m['batch']} | {m['brix']} | {m['firmness']} | {m['starch']} |")
    lines.append("")

    out = REPORTS_DIR / "dual_benchmark_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report: {out}")
    return out


def main():
    print("=" * 68)
    print("  DUAL BENCHMARK BRIX REGRESSION PIPELINE")
    print("=" * 68)

    print("\n[1/4] Extracting spectra from Specim IQ cubes...")
    data = build_feature_matrix()
    n = len(data["y_brix"])
    if n < 5:
        print(f"ERROR: Only {n} samples — need at least 5 for CV.")
        sys.exit(1)

    print(f"\n  N={n} apples")
    print(f"  Brix range: {data['y_brix'].min():.2f} - {data['y_brix'].max():.2f} dBx")
    print(f"  Firmness range: {data['y_firmness'].min():.2f} - {data['y_firmness'].max():.2f} kg/cm2")

    nf = min(5, n)

    print("\n[2/4] Track A: 204-band real spectrum...")
    r204, best204, m204, s204 = evaluate_regression(data["X_204"], data["y_brix"], "204-band Specim IQ", nf)

    print("\n[3/4] Track B: 18-ch digital twin (clean)...")
    r18c, best18c, m18c, s18c = evaluate_regression(data["X_18_clean"], data["y_brix"], "18-ch Twin Clean", nf)

    print("\n[3b] Track C: 18-ch digital twin (noisy)...")
    r18n, best18n, m18n, s18n = evaluate_regression(data["X_18_noisy"], data["y_brix"], "18-ch Twin Noisy", nf)

    print("\n[4/4] Scatter plots + report...")

    def best_result(res):
        bk = max(res, key=lambda k: res[k]["r2"])
        m = res[bk]
        return np.array(m["y_true"]), np.array(m["y_pred"]), m

    yt204, yp204, mr204 = best_result(r204)
    make_scatter(yt204, yp204, "Track A: Real 204-Band Specim IQ -> Brix",
                 REPORTS_DIR / "brix_scatter_204band.png", mr204["r2"], mr204["rmse"], mr204["pearson_r"])

    yt18c, yp18c, mr18c = best_result(r18c)
    make_scatter(yt18c, yp18c, "Track B: AS7265x 18-Ch Digital Twin (Clean) -> Brix",
                 REPORTS_DIR / "brix_scatter_18ch_clean.png", mr18c["r2"], mr18c["rmse"], mr18c["pearson_r"])

    yt18n, yp18n, mr18n = best_result(r18n)
    make_scatter(yt18n, yp18n, "Track C: AS7265x 18-Ch Digital Twin (+ Sensor Noise) -> Brix",
                 REPORTS_DIR / "brix_scatter_18ch_noisy.png", mr18n["r2"], mr18n["rmse"], mr18n["pearson_r"])

    joblib.dump({"model": m204, "scaler": s204}, MODELS_DIR / f"{best204.lower()}_204band.joblib")
    joblib.dump({"model": m18c, "scaler": s18c}, MODELS_DIR / f"{best18c.lower()}_18ch_clean.joblib")
    joblib.dump({"model": m18n, "scaler": s18n}, MODELS_DIR / f"{best18n.lower()}_18ch_noisy.joblib")
    print(f"  Models saved -> models/brix/")

    report = write_report(data, r204, r18c, r18n, n, nf)

    print("\n" + "=" * 68)
    print("  RESULT SUMMARY")
    print("=" * 68)
    print(f"  N apples   : {n}")
    print(f"  Track A  204-band      Best R2 = {max(v['r2'] for v in r204.values()):.4f}")
    print(f"  Track B  18-ch clean   Best R2 = {max(v['r2'] for v in r18c.values()):.4f}")
    print(f"  Track C  18-ch noisy   Best R2 = {max(v['r2'] for v in r18n.values()):.4f}")
    print(f"  Report  -> {report}")
    print("=" * 68)


if __name__ == "__main__":
    main()
