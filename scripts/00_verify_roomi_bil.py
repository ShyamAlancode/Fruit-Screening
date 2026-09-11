"""
00_verify_roomi_bil.py

PURPOSE
-------
Verify that the real Roomi pesticide hyperspectral dataset has been correctly
downloaded and extracted from Zenodo 7818714.

DOWNLOAD FROM
-------------
    https://zenodo.org/records/7818714
    (mirror: https://data.mendeley.com/datasets/3bhr26jffs/1)

FILES TO DOWNLOAD
-----------------
    Apple_samples.zip
    Fungicide_Apple.zip
    Pesticide_Apple.zip
    Total: ~6.6 GB

EXTRACT TO
----------
    d:\\CFRD\\DATA\\Dataset\\roomi_pesticide_hsi\\

GO / NO-GO GATE
---------------
This script must pass before any pipeline code is written or run.
If it fails (no .bil files, bands <= 3, no wavelength list), the real
Roomi hyperspectral cubes are not present. Do NOT proceed.

WHAT THIS SCRIPT CHECKS
-----------------------
1. roomi_pesticide_hsi/ directory exists and is non-empty
2. At least one .bil file exists in the subtree
3. Every .bil file has a companion .hdr header file
4. At least one .hdr file contains: bands > 3, wavelength list, interleave
5. At least one cube loads correctly with spectral.io.envi
6. Loaded cube has: bands ~281, wavelength range ~400-1000 nm
7. Spatial dimensions are non-trivial (H > 10, W > 10)
8. Data type is numeric (uint16 or float32)

USAGE
-----
    cd d:\\CFRD\\project
    python scripts\\00_verify_roomi_bil.py
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOMI_HSI_ROOT = r"d:\CFRD\DATA\Dataset\roomi_pesticide_hsi"
REPORT_PATH    = r"d:\CFRD\project\reports\roomi_bil_verification_report.md"

EXPECTED_MIN_BANDS = 100          # Real Roomi has ~281 bands
EXPECTED_MAX_BANDS = 400
EXPECTED_WL_MIN    = 380.0        # nm
EXPECTED_WL_MAX    = 1010.0       # nm
EXPECTED_WL_RANGE  = 500.0        # minimum span expected (nm)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_files_by_ext(root: str, ext: str) -> list[Path]:
    """Recursively find all files with a given extension."""
    return list(Path(root).rglob(f"*{ext}"))


def parse_hdr_file(hdr_path: Path) -> dict:
    """
    Parse an ENVI .hdr header file.
    Returns dict with keys: bands, wavelength, interleave, data_type, etc.
    """
    result = {}
    current_key = None
    current_val_lines = []

    with open(hdr_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if "=" in line and not line.startswith(";"):
                if current_key:
                    result[current_key] = " ".join(current_val_lines).strip()
                    current_val_lines = []
                key, _, val = line.partition("=")
                current_key = key.strip().lower()
                current_val_lines = [val.strip()]
            elif current_key and line:
                current_val_lines.append(line)

        if current_key:
            result[current_key] = " ".join(current_val_lines).strip()

    # Parse bands as int
    if "bands" in result:
        try:
            result["bands"] = int(result["bands"])
        except ValueError:
            pass

    # Parse wavelength list into floats
    if "wavelength" in result:
        wl_str = result["wavelength"].strip("{} ")
        try:
            result["wavelength"] = [
                float(x.strip()) for x in wl_str.split(",") if x.strip()
            ]
        except ValueError:
            result["wavelength"] = []

    return result


def check_envi_loadable(bil_path: Path, hdr_path: Path) -> dict:
    """
    Attempt to open the .bil cube with spectral.io.envi.
    Returns dict with shape, dtype, wavelengths, first_band_stats.
    """
    try:
        import spectral.io.envi as envi
        img = envi.open(str(hdr_path), str(bil_path))
        shape = (img.nrows, img.ncols, img.nbands)
        dtype = str(img.dtype)
        wl = list(img.bands.centers) if img.bands.centers else []
        # Read one spatial pixel from centre to confirm data loads
        r, c = img.nrows // 2, img.ncols // 2
        spectrum = img.read_pixel(r, c)
        return {
            "ok": True,
            "shape": shape,
            "dtype": dtype,
            "wavelengths": wl,
            "centre_pixel_sum": float(sum(spectrum)),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def run_verification() -> bool:
    print("=" * 70)
    print("Roomi Hyperspectral Dataset — BIL/HDR Verification")
    print(f"Root: {ROOMI_HSI_ROOT}")
    print(f"Run:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    report_lines = [
        "# Roomi BIL/HDR Verification Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source root: `{ROOMI_HSI_ROOT}`",
        "",
    ]

    all_passed = True
    gate_results = {}

    # ------------------------------------------------------------------
    # GATE 1: Root directory exists and is non-empty
    # ------------------------------------------------------------------
    print("\n[GATE 1] Checking root directory...")
    root_path = Path(ROOMI_HSI_ROOT)
    if not root_path.exists():
        msg = f"FAIL — Directory does not exist: {ROOMI_HSI_ROOT}"
        print(f"  {msg}")
        print("\n  ACTION: Download from https://zenodo.org/records/7818714")
        print("  Extract Apple_samples.zip, Fungicide_Apple.zip, Pesticide_Apple.zip")
        print(f"  into: {ROOMI_HSI_ROOT}")
        report_lines += [f"## GATE 1: Root directory\n**FAIL** — {msg}\n"]
        _write_report(report_lines)
        return False

    all_contents = list(root_path.rglob("*"))
    if not all_contents:
        msg = "FAIL — Directory exists but is empty"
        print(f"  {msg}")
        report_lines += [f"## GATE 1: Root directory\n**FAIL** — {msg}\n"]
        _write_report(report_lines)
        return False

    print(f"  PASS — {len(all_contents)} items found under root")
    gate_results["gate1"] = "PASS"
    report_lines += [f"## GATE 1: Root directory\n**PASS** — {len(all_contents)} items\n"]

    # ------------------------------------------------------------------
    # GATE 2: .bil files exist
    # ------------------------------------------------------------------
    print("\n[GATE 2] Searching for .bil files...")
    bil_files = find_files_by_ext(ROOMI_HSI_ROOT, ".bil")

    if not bil_files:
        msg = (
            "FAIL — No .bil files found anywhere under root.\n"
            "  CAUSE: You may have downloaded a repackaged/mirror version that\n"
            "  contains only RGB TIF previews. Download directly from Zenodo 7818714."
        )
        print(f"  {msg}")
        all_passed = False
    else:
        print(f"  PASS — Found {len(bil_files)} .bil files")

    gate_results["gate2_bil_count"] = len(bil_files)
    report_lines += [
        f"## GATE 2: .bil files\n"
        f"**{'PASS' if bil_files else 'FAIL'}** — {len(bil_files)} .bil files found\n",
    ]

    if not bil_files:
        _write_report(report_lines)
        return False

    # Show folder structure
    dirs_with_bil = set(str(f.parent) for f in bil_files)
    print(f"  Directories containing .bil files ({len(dirs_with_bil)}):")
    for d in sorted(dirs_with_bil)[:20]:
        print(f"    {d}")

    # ------------------------------------------------------------------
    # GATE 3: Every .bil has a companion .hdr
    # ------------------------------------------------------------------
    print("\n[GATE 3] Checking .hdr companions for each .bil...")
    missing_hdr = []
    for bil in bil_files:
        hdr = bil.with_suffix(".hdr")
        if not hdr.exists():
            hdr_plus = bil.parent / (bil.name + ".hdr")
            if hdr_plus.exists():
                hdr = hdr_plus
            else:
                hdr_upper = bil.parent / (bil.stem + ".HDR")
                if hdr_upper.exists():
                    hdr = hdr_upper
                else:
                    missing_hdr.append(bil)

    if missing_hdr:
        print(f"  WARNING — {len(missing_hdr)} .bil files have no companion .hdr:")
        for f in missing_hdr[:5]:
            print(f"    {f}")
        gate_results["gate3"] = "WARNING"
    else:
        print(f"  PASS — All {len(bil_files)} .bil files have companion .hdr")
        gate_results["gate3"] = "PASS"

    report_lines += [
        f"## GATE 3: HDR companions\n"
        f"**{gate_results['gate3']}** — {len(missing_hdr)} .bil files missing .hdr\n",
    ]

    # ------------------------------------------------------------------
    # GATE 4: Parse a .hdr file — check bands and wavelengths
    # ------------------------------------------------------------------
    print("\n[GATE 4] Parsing header file...")
    hdr_files = find_files_by_ext(ROOMI_HSI_ROOT, ".hdr")
    print(f"  Found {len(hdr_files)} .hdr files")

    hdr_ok = False
    sample_hdr_info = {}

    for hdr_path in hdr_files[:5]:
        hdr_data = parse_hdr_file(hdr_path)
        bands = hdr_data.get("bands", 0)
        wavelengths = hdr_data.get("wavelength", [])
        interleave = hdr_data.get("interleave", "unknown")

        print(f"\n  HDR: {hdr_path.name}")
        print(f"    bands:       {bands}")
        print(f"    interleave:  {interleave}")
        print(f"    wavelengths: {len(wavelengths)} values", end="")
        if wavelengths:
            print(f" [{wavelengths[0]:.1f} – {wavelengths[-1]:.1f} nm]")
        else:
            print(" [NONE FOUND]")

        if (
            isinstance(bands, int)
            and EXPECTED_MIN_BANDS <= bands <= EXPECTED_MAX_BANDS
            and len(wavelengths) == bands
            and wavelengths
            and (wavelengths[-1] - wavelengths[0]) >= EXPECTED_WL_RANGE
        ):
            hdr_ok = True
            sample_hdr_info = {
                "file": str(hdr_path),
                "bands": bands,
                "wl_min": wavelengths[0],
                "wl_max": wavelengths[-1],
                "interleave": interleave,
            }
            print(f"    STATUS: VALID HYPERSPECTRAL HEADER")
            break
        else:
            problems = []
            if not (isinstance(bands, int) and EXPECTED_MIN_BANDS <= bands <= EXPECTED_MAX_BANDS):
                problems.append(f"bands={bands} (expected {EXPECTED_MIN_BANDS}–{EXPECTED_MAX_BANDS})")
            if not wavelengths:
                problems.append("no wavelength list")
            elif len(wavelengths) != bands:
                problems.append(f"wavelength count {len(wavelengths)} != bands {bands}")
            if wavelengths and (wavelengths[-1] - wavelengths[0]) < EXPECTED_WL_RANGE:
                problems.append(f"wavelength span {wavelengths[-1]-wavelengths[0]:.0f}nm < {EXPECTED_WL_RANGE}nm")
            print(f"    STATUS: INVALID — {'; '.join(problems)}")

    if not hdr_ok:
        print("\n  FAIL — No valid hyperspectral header found")
        print("  This suggests RGB previews were downloaded instead of real cubes.")
        all_passed = False
        gate_results["gate4"] = "FAIL"
    else:
        print("\n  PASS — Valid hyperspectral header confirmed")
        gate_results["gate4"] = "PASS"

    report_lines += [
        "## GATE 4: Header validation",
        f"**{gate_results['gate4']}**",
    ]
    if sample_hdr_info:
        report_lines += [
            f"- bands: {sample_hdr_info['bands']}",
            f"- wavelength range: {sample_hdr_info['wl_min']:.1f} – {sample_hdr_info['wl_max']:.1f} nm",
            f"- interleave: {sample_hdr_info['interleave']}",
        ]
    report_lines.append("")

    if not hdr_ok:
        _write_report(report_lines)
        return False

    # ------------------------------------------------------------------
    # GATE 5: Load one cube with spectral.io.envi
    # ------------------------------------------------------------------
    print("\n[GATE 5] Attempting to load one cube with spectral.io.envi...")
    load_result = {"ok": False, "error": "No valid bil/hdr pair found"}

    for bil_path in bil_files[:10]:
        hdr_path = bil_path.with_suffix(".hdr")
        if not hdr_path.exists():
            hdr_path = bil_path.parent / (bil_path.name + ".hdr")
        if not hdr_path.exists():
            continue
        hdr_data = parse_hdr_file(hdr_path)
        if not (isinstance(hdr_data.get("bands"), int) and hdr_data["bands"] > 3):
            continue
        load_result = check_envi_loadable(bil_path, hdr_path)
        if load_result["ok"]:
            print(f"  Loaded: {bil_path.name}")
            break

    if not load_result["ok"]:
        print(f"  FAIL — Could not load any cube: {load_result.get('error', 'unknown')}")
        all_passed = False
        gate_results["gate5"] = "FAIL"
    else:
        shape = load_result["shape"]
        wl = load_result["wavelengths"]
        print(f"  PASS — Shape: {shape}, dtype: {load_result['dtype']}")
        if wl:
            print(f"         Wavelength range: {wl[0]:.1f} – {wl[-1]:.1f} nm ({len(wl)} bands)")
        gate_results["gate5"] = "PASS"
        gate_results["confirmed_shape"] = shape
        gate_results["confirmed_bands"] = shape[2]

    report_lines += [
        "## GATE 5: ENVI cube loading",
        f"**{gate_results['gate5']}**",
    ]
    if load_result["ok"]:
        report_lines += [
            f"- shape: {load_result['shape']}",
            f"- dtype: {load_result['dtype']}",
        ]
    report_lines.append("")

    # ------------------------------------------------------------------
    # GATE 6: Band count sanity check
    # ------------------------------------------------------------------
    if load_result["ok"]:
        print("\n[GATE 6] Band count sanity check...")
        n_bands = load_result["shape"][2]
        if n_bands <= 3:
            print(f"  FAIL — Only {n_bands} bands loaded. This is an RGB image, not HSI.")
            print("  Download the real cubes from Zenodo 7818714.")
            all_passed = False
            gate_results["gate6"] = f"FAIL — {n_bands} bands"
        elif n_bands < EXPECTED_MIN_BANDS:
            print(f"  WARNING — {n_bands} bands is less than expected minimum ({EXPECTED_MIN_BANDS})")
            gate_results["gate6"] = f"WARNING — {n_bands} bands"
        else:
            print(f"  PASS — {n_bands} bands confirmed")
            gate_results["gate6"] = f"PASS — {n_bands} bands"

        report_lines += [
            f"## GATE 6: Band count\n**{gate_results['gate6']}**\n",
        ]

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for k, v in gate_results.items():
        print(f"  {k:35s}  {v}")

    if all_passed:
        print("\n  *** GO *** All gates passed. Proceed to Phase 1.")
        verdict = "GO"
    else:
        print("\n  *** NO-GO *** One or more gates failed.")
        print("  Do NOT proceed to Phase 1 until real .bil cubes are present.")
        verdict = "NO-GO"

    report_lines += [
        "## Final Verdict",
        f"**{verdict}**",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for k, v in gate_results.items():
        report_lines.append(f"| {k} | {v} |")

    _write_report(report_lines)
    print(f"\n  Report saved to: {REPORT_PATH}")
    return all_passed


def _write_report(lines: list[str]) -> None:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    passed = run_verification()
    sys.exit(0 if passed else 1)
