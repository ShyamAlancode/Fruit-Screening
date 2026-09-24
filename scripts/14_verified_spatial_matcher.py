"""
14_verified_spatial_matcher.py
==============================
Verified ground-truth matcher for University of Essex Specim IQ dataset.

THE BUG WAS:
  audit_blobs_and_labels.py matched CSV rows by substring on folder name
  (e.g. '2023-03-03_036') WITHOUT enforcing cultivar or fruit number.
  This caused Gala images to match Cox CSV rows (and vice-versa),
  corrupting the ground-truth labels for all Brix regression.

THE FIX:
  1. Parse the Specim IQ XML description field to extract:
     - Cultivar keyword  (e.g. 'Gala', 'Cox', 'bb' for Braeburn)
     - Fruit number range (e.g. '31 33' or '21-23' means fruits 31, 32, 33)
     - Side letter         (e.g. 'a', 'b', 'c', 'd')
  2. Look up CSV rows by exact (Cultivar, Number) match — never by substring.
  3. Map blobs left-to-right (ascending x_center) to fruits in ascending
     number order (they are always placed L→R by Number in the paper protocol).
  4. Output verified_records_audit_v2.json with zero cross-cultivar mismatches.

Usage:
    python scripts/14_verified_spatial_matcher.py
"""

import json, re, sys
from pathlib import Path
import pandas as pd

import os
# ---- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_DIR      = Path(os.environ.get("CFRD_ESSEX_ROOT", PROJECT_ROOT.parent / "New folder"))
AUDIT_JSON   = PROJECT_ROOT / "records_audit.json"
OUTPUT_JSON  = PROJECT_ROOT / "records_audit_v2.json"

# ---- CSV loading ----------------------------------------------------------
print("Loading ground-truth CSVs...")
csv_map = {
    "NZ2023": pd.read_csv(CSV_DIR / "data_NZ2023.csv"),
    "NZ2024": pd.read_csv(CSV_DIR / "data_NZ2024.csv"),
    "UK2024": pd.read_csv(CSV_DIR / "data_UK2024.csv"),
}
# All datasets have 'Cultivar' and 'Number' columns
for name, df in csv_map.items():
    print(f"  {name}: {len(df)} rows, cultivars: {df['Cultivar'].unique().tolist()}")

# ---- Cultivar keyword → CSV cultivar string --------------------------------
CULTIVAR_ALIASES = {
    # keyword in XML description : actual CSV Cultivar value
    "gala":    "Gala",
    "cox":     "Cox's Orange Pippin",
    "bb":      "Braeburn",
    "braeburn": "Braeburn",
    "fuji":    "Fuji",
    "golden":  "Golden Delicious",
}

def parse_xml_description(desc: str):
    """
    Extract (cultivar_str, fruit_numbers, side_letter) from Specim IQ XML description.

    Example inputs:
      'Cox 66 68 side 4 The spectral image is recorded...'  -> ('Cox', [66,67,68], '4')
      'Gala 31 33 3 The spectral image is recorded...'      -> ('Gala', [31,32,33], '3')
      'Cox 21-23 side 4 The spectral image is recorded...'  -> ('Cox', [21,22,23], '4')
      'The spectral image is recorded ... bb 19 21 a'       -> ('Braeburn', [19,20,21], 'a')
    """
    if not desc:
        return None, [], None

    desc_lower = desc.lower()

    # Detect cultivar keyword
    cultivar_raw = None
    cultivar_csv = None
    for alias, csv_name in CULTIVAR_ALIASES.items():
        if alias in desc_lower:
            cultivar_raw = alias
            cultivar_csv = csv_name
            break

    if cultivar_csv is None:
        return None, [], None

    # Extract two numbers (either "N1 N2" or "N1-N2")
    # Pattern: cultivar keyword ... NUMBER ... NUMBER
    # Remove the boilerplate text
    clean = re.sub(r'the spectral image is recorded in the default recording mode', '', desc, flags=re.IGNORECASE)
    clean = clean.strip()

    # Find integer pairs
    nums = re.findall(r'\d+', clean)
    nums = [int(n) for n in nums if 10 <= int(n) <= 9999]  # filter out side numbers (1-9)

    fruit_numbers = []
    if len(nums) >= 2:
        n1, n2 = nums[0], nums[1]
        # Generate inclusive range: n1, n1+1, ..., n2
        if n1 <= n2 and (n2 - n1) <= 10:
            fruit_numbers = list(range(n1, n2 + 1))

    # Detect side letter (a/b/c/d or 1/2/3/4)
    side_match = re.search(r'\bside\s*([1-4])\b|\b([abcd])\b', desc_lower)
    side = None
    if side_match:
        side = side_match.group(1) or side_match.group(2)
    else:
        # Look for trailing single digit or letter
        trailing = re.findall(r'\b([1-4abcd])\b', clean.lower())
        if trailing:
            side = trailing[-1]

    return cultivar_csv, fruit_numbers, side


def lookup_csv_rows(cultivar_csv, fruit_numbers, csv_src_hint=None):
    """
    Look up rows from CSV by exact (Cultivar, Number) match.
    Returns a list of dicts sorted by Number ascending (left-to-right blob order).
    """
    search_order = list(csv_map.keys())
    if csv_src_hint and csv_src_hint in csv_map:
        # Try the hinted source first
        search_order = [csv_src_hint] + [k for k in search_order if k != csv_src_hint]

    for csv_name in search_order:
        df = csv_map[csv_name]
        if "Cultivar" not in df.columns or "Number" not in df.columns:
            continue
        matched = df[
            (df["Cultivar"].astype(str).str.strip() == cultivar_csv) &
            (df["Number"].isin(fruit_numbers))
        ].copy()
        if len(matched) > 0:
            # Deduplicate: one row per fruit number (take first occurrence per Number)
            matched = matched.drop_duplicates(subset=["Number"]).sort_values("Number").reset_index(drop=True)
            # Filter to exactly requested fruit numbers
            matched = matched[matched["Number"].isin(fruit_numbers)]
            rows = []
            for _, row in matched.iterrows():
                rows.append({
                    "csv_source": csv_name,
                    "ID": str(row.get("ID", "")),
                    "Number": int(row["Number"]),
                    "Cultivar": str(row["Cultivar"]),
                    "Brix": float(row.get("Brix", 0.0)) if pd.notna(row.get("Brix")) else None,
                    "Avg_firmness": float(row.get("Avg_firmness", 0.0)) if pd.notna(row.get("Avg_firmness")) else None,
                    "Starch": float(row.get("Starch", 0.0)) if pd.notna(row.get("Starch")) else None,
                })
            return rows, csv_name

    return [], None


# ---- Main processing -------------------------------------------------------
print(f"\nLoading audit JSON: {AUDIT_JSON}")
with open(AUDIT_JSON) as f:
    records = json.load(f)

output_records = []
stats = {"total": 0, "verified_full": 0, "partial": 0, "no_match": 0,
         "cultivar_mismatch_fixed": 0, "parse_failed": 0}

print("\n=== Processing Records ===")
for rec in records:
    stats["total"] += 1
    folder    = rec["folder"]
    batch     = rec["batch"]
    xml_desc  = rec.get("xml_desc", "")
    blobs     = rec["blob_positions"]
    old_csv_src = rec.get("csv_source", "")
    old_csv_rows = rec.get("csv_rows", [])

    # Sort blobs left-to-right by x_center (ascending = left to right)
    blobs_sorted = sorted(blobs, key=lambda b: b["x_center"])

    # Parse XML description for verified match
    cultivar_csv, fruit_numbers, side = parse_xml_description(xml_desc)

    if cultivar_csv is None or not fruit_numbers:
        stats["parse_failed"] += 1
        print(f"  PARSE_FAIL  {batch}/{folder}: desc='{xml_desc}'")
        # Keep old data but mark as unverified
        rec_out = dict(rec)
        rec_out["blob_positions"] = blobs_sorted
        rec_out["verified"] = False
        rec_out["parse_cultivar"] = None
        rec_out["parse_fruit_numbers"] = []
        output_records.append(rec_out)
        continue

    # Look up exact CSV rows
    verified_rows, csv_name = lookup_csv_rows(cultivar_csv, fruit_numbers, old_csv_src)

    # Detect if old code had a cultivar mismatch
    old_cultivars = set(r.get("Cultivar", r["ID"].split("_")[0] if r.get("ID") else "?") for r in old_csv_rows)
    mismatch = cultivar_csv not in old_cultivars and bool(old_csv_rows)
    if mismatch:
        stats["cultivar_mismatch_fixed"] += 1

    if len(verified_rows) == len(fruit_numbers):
        stats["verified_full"] += 1
        status = "FULL_MATCH"
    elif len(verified_rows) > 0:
        stats["partial"] += 1
        status = f"PARTIAL ({len(verified_rows)}/{len(fruit_numbers)})"
    else:
        stats["no_match"] += 1
        status = "NO_MATCH"

    mismatch_flag = " [CULTIVAR_MISMATCH_FIXED]" if mismatch else ""
    print(f"  {status:20s} {batch}/{folder}: cultivar={cultivar_csv}, fruits={fruit_numbers}, "
          f"rows_found={len(verified_rows)}{mismatch_flag}")

    # Build aligned blob→label pairs
    n_aligned = min(len(blobs_sorted), len(verified_rows))
    aligned_pairs = []
    for i in range(n_aligned):
        aligned_pairs.append({
            "blob": blobs_sorted[i],
            "label": verified_rows[i],
        })

    rec_out = {
        "batch":           batch,
        "folder":          folder,
        "xml_desc":        xml_desc,
        "parse_cultivar":  cultivar_csv,
        "parse_fruit_numbers": fruit_numbers,
        "parse_side":      side,
        "csv_source":      csv_name,
        "verified":        len(verified_rows) == len(fruit_numbers),
        "num_blobs":       len(blobs),
        "blob_positions":  blobs_sorted,
        "csv_rows":        verified_rows,
        "aligned_pairs":   aligned_pairs,
    }
    output_records.append(rec_out)

# ---- Write output ----------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump(output_records, f, indent=2)

print(f"\n=== Verified Spatial Match Summary ===")
print(f"  Total records:             {stats['total']}")
print(f"  Full match (verified):     {stats['verified_full']}")
print(f"  Partial match:             {stats['partial']}")
print(f"  No match:                  {stats['no_match']}")
print(f"  Parse failed (no XML):     {stats['parse_failed']}")
print(f"  Cultivar mismatches FIXED: {stats['cultivar_mismatch_fixed']}")
print(f"\n  Output written to: {OUTPUT_JSON}")
