"""
scripts/02_build_essex_hyperspectral_dataset.py
================================================
Automated Stream-and-Extract-and-Clean pipeline for University of Essex Specim IQ Hyperspectral dataset:
  1. Downloads target Essex batch archives sequentially
  2. Extracts calibrated reflectance cubes (512x512x204 bands, 397-1004 nm)
  3. Segments apple fruit ROIs using OpenCV contour masking
  4. Matches fruit with ground-truth refractometer Brix from data_NZ2023, data_NZ2024, data_UK2024
  5. Appends to master dataset (master_essex_hyperspectral_dataset.csv and .npz)
  6. Deletes transient multi-GB raw cubes immediately to protect disk space (free space ~8.3 GB)
  7. Runs leakage-safe GroupKFold ML benchmarking (PLSR, SVR, RF, GBR, Ridge)
"""

import os
import sys
import re
import time
import shutil
import urllib.request
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(r"d:\CFRD")
BRIX_DIR = PROJECT_ROOT / "Brixsweetness"
TMP_DOWNLOAD_DIR = BRIX_DIR / "tmp_stream_cache"
TMP_DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)

MASTER_CSV = BRIX_DIR / "master_essex_hyperspectral_dataset.csv"
MASTER_NPZ = BRIX_DIR / "master_essex_hyperspectral_dataset.npz"

# Load metadata CSVs
METADATA_FILES = {
    "NZ2023": (PROJECT_ROOT / "Ripeness" / "data_NZ2023.csv", "New Zealand", "2023"),
    "NZ2024": (PROJECT_ROOT / "Ripeness" / "data_NZ2024.csv", "New Zealand", "2024"),
    "UK2024": (PROJECT_ROOT / "Ripeness" / "data_UK2024.csv", "United Kingdom", "2024"),
}

dfs = []
for code, (p, country, season) in METADATA_FILES.items():
    if p.exists():
        df_temp = pd.read_csv(p)
        df_temp["dataset_code"] = code
        df_temp["country"] = country
        df_temp["season"] = season
        dfs.append(df_temp)
master_meta = pd.concat(dfs, ignore_index=True)
print(f"Loaded master metadata index: {len(master_meta):,} apples across 3 seasons.")


def parse_hdr(hdr_path: Path):
    """Parse ENVI header for dimensions, interleave, and wavelengths."""
    with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = int(re.search(r"lines\s*=\s*(\d+)", text).group(1))
    samples = int(re.search(r"samples\s*=\s*(\d+)", text).group(1))
    bands = int(re.search(r"bands\s*=\s*(\d+)", text).group(1))
    wl_match = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.DOTALL)
    wavelengths = [float(w.strip()) for w in wl_match.group(1).split(",") if w.strip()]
    return lines, samples, bands, np.array(wavelengths)


def extract_cube_apple_spectra(capture_dir: Path, wavelengths: np.ndarray):
    """Segment apple fruit from preview image and extract 204-band mean spectra."""
    capture_id = capture_dir.name
    
    # Locate files
    png_file = capture_dir / f"{capture_id}.png"
    if not png_file.exists():
        png_candidates = list(capture_dir.glob("*.png"))
        if png_candidates:
            png_file = png_candidates[0]
        else:
            return []
            
    # Prefer pre-calibrated reflectance in results/
    results_dir = capture_dir / "results"
    dat_files = list(results_dir.glob("REFLECTANCE_*.dat"))
    hdr_files = list(results_dir.glob("REFLECTANCE_*.hdr"))
    
    is_reflectance_float = True
    if dat_files and hdr_files:
        cube_file = dat_files[0]
        hdr_file = hdr_files[0]
    else:
        # Fallback to capture/ raw
        raw_files = list((capture_dir / "capture").glob("*.raw"))
        raw_hdrs = list((capture_dir / "capture").glob("*.hdr"))
        if not raw_files or not raw_hdrs:
            return []
        cube_file = raw_files[0]
        hdr_file = raw_hdrs[0]
        is_reflectance_float = False

    lines, samples, bands, wls = parse_hdr(hdr_file)
    
    # Load binary cube
    if is_reflectance_float:
        raw_data = np.fromfile(cube_file, dtype="<f4")
    else:
        raw_data = np.fromfile(cube_file, dtype="<u2").astype(np.float32) / 10000.0
        
    cube_bil = raw_data.reshape((lines, bands, samples))
    cube = np.transpose(cube_bil, (0, 2, 1))  # (Y, X, Lambda)
    
    # Segment apples from RGB image
    im = Image.open(png_file).convert("RGB")
    arr = np.array(im)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    apples = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 2500:  # Apple mask threshold
            x, y, w, h = cv2.boundingRect(c)
            mask = np.zeros((lines, samples), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 1, -1)
            # Erode boundary glare
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2).astype(bool)
            apples.append({"x": x, "y": y, "w": w, "h": h, "area": area, "mask": mask})
            
    apples.sort(key=lambda a: a["x"])  # Order left to right
    
    # Search matching rows in master metadata
    query_str = f"{capture_id}.png"
    matches = master_meta[
        (master_meta["image_filename_A"] == query_str) |
        (master_meta["image_filename_B"] == query_str) |
        (master_meta["image_filename_C"] == query_str) |
        (master_meta["image_filename_D"] == query_str)
    ].sort_values("Number")
    
    records = []
    for i, (_, row) in enumerate(matches.iterrows()):
        if i < len(apples):
            apple = apples[i]
            apple_voxels = cube[apple["mask"]]
            if len(apple_voxels) == 0:
                continue
            mean_spec = apple_voxels.mean(axis=0)
            median_spec = np.median(apple_voxels, axis=0)
            std_spec = apple_voxels.std(axis=0)
            
            # Determine view (A, B, C, or D)
            view = "A" if row.get("image_filename_A") == query_str else (
                "B" if row.get("image_filename_B") == query_str else (
                    "C" if row.get("image_filename_C") == query_str else "D"
                )
            )
            
            records.append({
                "apple_id": str(row["ID"]),
                "tray_number": row.get("Number"),
                "capture_id": capture_id,
                "view": view,
                "brix": float(row["Brix"]),
                "cultivar": str(row.get("Cultivar", "Unknown")),
                "firmness": float(row.get("Avg_firmness", row.get("Firmness1", np.nan))),
                "starch": float(row.get("Starch", np.nan)),
                "season": str(row.get("season", "2023")),
                "country": str(row.get("country", "New Zealand")),
                "pixel_count": int(apple["mask"].sum()),
                "spectrum_mean": mean_spec,
                "spectrum_median": median_spec,
                "spectrum_std": std_spec,
            })
            
    return records


def download_file_with_progress(url: str, dest_path: Path):
    """Download single archive file with speed and progress monitoring."""
    start_t = time.time()
    def reporthook(block_num, block_size, total_size):
        down = block_num * block_size
        if total_size > 0 and (block_num % 1000 == 0 or down >= total_size):
            pct = down / total_size * 100
            mb_down = down / (1024 * 1024)
            mb_tot = total_size / (1024 * 1024)
            el = time.time() - start_t
            spd = mb_down / el if el > 0 else 0
            print(f"  [{dest_path.name}] {mb_down:.1f}/{mb_tot:.1f} MB ({pct:.1f}%) - {spd:.2f} MB/s", end="\r", flush=True)
            
    urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
    print(f"\n  Finished downloading {dest_path.name} in {time.time()-start_t:.1f}s.")


def run_pipeline(target_archives: list):
    """Main execution loop: Download -> Extract -> Measure -> Append -> Cleanup."""
    wavelengths = None
    all_records = []
    
    # Include pre-extracted Batch 7 samples if available
    pre_extracted = Path(r"d:\CFRD\Brixsweetness\extracted_batch7")
    if pre_extracted.exists():
        for cap_dir in sorted(pre_extracted.glob("2023-*")):
            if cap_dir.is_dir():
                print(f"Processing pre-extracted capture: {cap_dir.name}...")
                hdr_f = list(cap_dir.glob("**/*.hdr"))[0]
                _, _, _, wls = parse_hdr(hdr_f)
                wavelengths = wls
                recs = extract_cube_apple_spectra(cap_dir, wavelengths)
                all_records.extend(recs)
                print(f"  Extracted {len(recs)} apples from {cap_dir.name}.")
                
    for idx, (archive_name, url) in enumerate(target_archives):
        print(f"\n[{idx+1}/{len(target_archives)}] Streaming Batch Archive: {archive_name}")
        tar_dest = TMP_DOWNLOAD_DIR / archive_name
        extract_dest = TMP_DOWNLOAD_DIR / f"unpacked_{archive_name.replace('.tar.gz', '')}"
        
        try:
            download_file_with_progress(url, tar_dest)
            extract_dest.mkdir(exist_ok=True, parents=True)
            
            print(f"  Unpacking members from {archive_name} ...")
            with tarfile.open(tar_dest, "r|gz") as tar:
                while True:
                    try:
                        m = tar.next()
                        if m is None:
                            break
                        tar.extract(m, path=extract_dest)
                    except Exception:
                        break
                        
            # Process all capture folders unpacked in this batch
            capture_dirs = [d for d in extract_dest.glob("20*") if d.is_dir() and "_" in d.name]
            print(f"  Found {len(capture_dirs)} capture folders.")
            
            for c_dir in capture_dirs:
                hdrs = list(c_dir.glob("**/*.hdr"))
                if not hdrs:
                    continue
                if wavelengths is None:
                    _, _, _, wavelengths = parse_hdr(hdrs[0])
                recs = extract_cube_apple_spectra(c_dir, wavelengths)
                if recs:
                    all_records.extend(recs)
                    print(f"    -> Extracted {len(recs)} apple views from {c_dir.name} (Total so far: {len(all_records)})")
                    
        except Exception as e:
            print(f"  Error processing {archive_name}: {e}")
        finally:
            # CLEANUP: Delete multi-GB raw binary files immediately to preserve disk space
            print(f"  Cleaning temporary cache for {archive_name} ...")
            if tar_dest.exists():
                try: os.remove(tar_dest)
                except Exception: pass
            if extract_dest.exists():
                try: shutil.rmtree(extract_dest)
                except Exception: pass

    if not all_records:
        print("No apple records extracted.")
        return

    print(f"\n========================================================")
    print(f"PIPELINE EXTRACTION COMPLETE: {len(all_records)} apple observations!")
    print(f"========================================================")
    
    # Save Master Table
    master_rows = []
    spectra_dict = {}
    
    for r in all_records:
        row_dict = {
            "apple_id": r["apple_id"],
            "tray_number": r["tray_number"],
            "capture_id": r["capture_id"],
            "view": r["view"],
            "brix": r["brix"],
            "cultivar": r["cultivar"],
            "firmness": r["firmness"],
            "starch": r["starch"],
            "season": r["season"],
            "country": r["country"],
            "pixel_count": r["pixel_count"]
        }
        for b_i in range(len(r["spectrum_mean"])):
            row_dict[f"ch_{wavelengths[b_i]:.1f}nm"] = round(float(r["spectrum_mean"][b_i]), 6)
            
        master_rows.append(row_dict)
        unique_key = f"{r['apple_id']}_{r['capture_id']}_{r['view']}"
        spectra_dict[unique_key] = r["spectrum_mean"]

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(MASTER_CSV, index=False)
    np.savez(MASTER_NPZ, wavelengths=wavelengths, **spectra_dict)
    print(f"Master CSV saved to: {MASTER_CSV} ({len(master_df)} rows)")
    print(f"Master NPZ saved to: {MASTER_NPZ}")
    
    # Execute Leakage-Safe GroupKFold Benchmarking
    run_benchmarks(master_df, wavelengths)


def run_benchmarks(df: pd.DataFrame, wavelengths: np.ndarray):
    """Benchmark 204-band models with apple-level GroupKFold."""
    print("\n--- PHASE 6: 204-BAND LEAKAGE-SAFE BENCHMARKING (GROUPED BY APPLE ID) ---")
    
    spectral_cols = [c for c in df.columns if c.startswith("ch_")]
    X = df[spectral_cols].values
    y = df["brix"].values
    groups = df["apple_id"].values
    
    unique_apples = df["apple_id"].nunique()
    print(f"Total scans: {len(df)}, Unique fruits: {unique_apples}")
    print(f"Brix range: {y.min():.2f} to {y.max():.2f} °Bx (Mean: {y.mean():.2f} ± {y.std():.2f})")
    
    gkf = GroupKFold(n_splits=min(5, unique_apples))
    
    models = {
        "Baseline (Mean Brix)": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=10.0),
        "Partial Least Squares (PLSR 6 comp)": PLSRegression(n_components=min(6, X.shape[1])),
        "Support Vector Regressor (SVR)": SVR(C=10.0, epsilon=0.1),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    }
    
    print("\n" + "="*85)
    print(f"{'Model Architecture':<36} | {'R²':<8} | {'MAE (°Bx)':<10} | {'RMSE (°Bx)':<10} | {'Pearson r':<10}")
    print("="*85)
    
    benchmark_records = []
    
    for name, model in models.items():
        y_true_all = []
        y_pred_all = []
        
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            
            if "PLSR" in name:
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_va_s).ravel()
            else:
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_va_s)
                
            y_true_all.extend(y_va)
            y_pred_all.extend(preds)
            
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        
        r2 = r2_score(y_true_all, y_pred_all)
        mae = mean_absolute_error(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        pr = pearsonr(y_true_all, y_pred_all)[0] if np.std(y_pred_all) > 1e-6 else 0.0
        
        benchmark_records.append({"Model": name, "R2": r2, "MAE": mae, "RMSE": rmse, "Pearson_r": pr})
        print(f"{name:<36} | {r2:8.4f} | {mae:10.4f} | {rmse:10.4f} | {pr:10.4f}")
        
    print("="*85)
    bench_df = pd.DataFrame(benchmark_records)
    bench_csv = BRIX_DIR / "master_essex_204band_benchmark.csv"
    bench_df.to_csv(bench_csv, index=False)
    print(f"204-band benchmark results saved to: {bench_csv}")


if __name__ == "__main__":
    # If arguments passed, can download batches, otherwise test on pre-extracted
    target_batches = []
    if "--download-gala" in sys.argv:
        target_batches = [
            ("20230303-04 Gala 110 fruit 4 side ea_batch1.tar.gz", "https://researchdata.essex.ac.uk/228/749/20230303-04%20Gala%20110%20fruit%204%20side%20ea_batch1.tar.gz"),
        ]
    run_pipeline(target_batches)
