"""
scripts/01_prepare_ripeness_crops.py
====================================
Data curation and instance extraction pipeline for Apple Ripeness Computer Vision.

Extracts isolated apple instances from the 2026 ScienceDirect multi-stage dataset:
  - Parses VIA polygon annotations from train.json and test.json
  - Computes padded bounding-box crops of each apple instance
  - Filters out background noise and small occlusion artifacts (<20x20 px)
  - Maps classes to standard developmental stages:
      * 'immature apple'   -> 0 (immature / green)
      * 'semi-mature apple' -> 1 (semi_mature / transition)
      * 'mature apple'     -> 2 (mature / red)
  - Splits training data into train (85%) and val (15%) with zero image leakage
  - Indexes RawRipe (105 raw, 108 ripe) as an independent external validation set

Output Directory Structure:
  data/ripeness_multistage/
    ├── train/
    │   ├── immature/
    │   ├── semi_mature/
    │   └── mature/
    ├── val/
    │   ├── immature/
    │   ├── semi_mature/
    │   └── mature/
    ├── test/
    │   ├── immature/
    │   ├── semi_mature/
    │   └── mature/
    └── external_rawripe/
        ├── immature/
        └── mature/
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RIPENESS_RAW_DIR = Path(r"d:\CFRD\Ripeness")
DATASET_2026_DIR = RIPENESS_RAW_DIR / "dataset-20260508" / "dataset-20260508"
RAWRIPE_DIR = RIPENESS_RAW_DIR / "RawRipe" / "apples"

OUTPUT_DIR = PROJECT_ROOT / "data" / "ripeness_multistage"

CLASS_MAP = {
    "immature apple": "immature",
    "semi-mature apple": "semi_mature",
    "mature apple": "mature",
}

MIN_CROP_DIM = 20  # Minimum width/height in pixels
PADDING_RATIO = 0.08  # 8% contextual margin around bounding box


def extract_crops_from_json(json_path: Path, image_dir: Path, split_name: str) -> list[dict]:
    """Parse VIA json, extract bounding boxes, crop and return metadata list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_records = []
    print(f"Parsing {json_path.name} ({len(data)} image entries)...")

    for img_key, img_info in data.items():
        fname = img_info.get("filename")
        if not fname:
            continue
        im_path = image_dir / fname
        if not im_path.exists():
            continue

        try:
            with Image.open(im_path) as im:
                im_rgb = im.convert("RGB")
                w, h = im_rgb.size

                for idx, region in enumerate(img_info.get("regions", [])):
                    cat_name = region.get("region_attributes", {}).get("name")
                    if cat_name not in CLASS_MAP:
                        continue
                    norm_class = CLASS_MAP[cat_name]

                    shape = region.get("shape_attributes", {})
                    xs = shape.get("all_points_x", [])
                    ys = shape.get("all_points_y", [])
                    if not xs or not ys:
                        continue

                    # Bounding box
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bw = x_max - x_min
                    bh = y_max - y_min

                    if bw < MIN_CROP_DIM or bh < MIN_CROP_DIM:
                        continue

                    # Apply contextual padding margin
                    pad_x = int(bw * PADDING_RATIO)
                    pad_y = int(bh * PADDING_RATIO)

                    crop_x1 = max(0, x_min - pad_x)
                    crop_y1 = max(0, y_min - pad_y)
                    crop_x2 = min(w, x_max + pad_x)
                    crop_y2 = min(h, y_max + pad_y)

                    cropped = im_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    
                    crop_id = f"{Path(fname).stem}_crop{idx:02d}_{norm_class}.jpg"
                    extracted_records.append({
                        "crop_image": cropped,
                        "crop_filename": crop_id,
                        "original_image": fname,
                        "class_label": norm_class,
                        "split": split_name,
                        "bbox": (crop_x1, crop_y1, crop_x2, crop_y2),
                    })
        except Exception as e:
            continue

    print(f"  Extracted {len(extracted_records)} valid apple crops from {split_name}.")
    return extracted_records


def setup_output_dirs():
    """Create directory hierarchy."""
    for split in ["train", "val", "test"]:
        for c in ["immature", "semi_mature", "mature"]:
            (OUTPUT_DIR / split / c).mkdir(parents=True, exist_ok=True)

    for c in ["immature", "mature"]:
        (OUTPUT_DIR / "external_rawripe" / c).mkdir(parents=True, exist_ok=True)


def process_external_rawripe() -> dict:
    """Index and copy RawRipe apples for external validation."""
    print("\nProcessing RawRipe external validation dataset...")
    raw_images_dir = RAWRIPE_DIR / "raw" / "images"
    ripe_images_dir = RAWRIPE_DIR / "ripe" / "images"

    counts = {"immature": 0, "mature": 0}

    if raw_images_dir.exists():
        for img_p in raw_images_dir.glob("*.jpg"):
            dest = OUTPUT_DIR / "external_rawripe" / "immature" / f"rawripe_{img_p.name}"
            shutil.copy2(img_p, dest)
            counts["immature"] += 1

    if ripe_images_dir.exists():
        for img_p in ripe_images_dir.glob("*.jpg"):
            dest = OUTPUT_DIR / "external_rawripe" / "mature" / f"rawripe_{img_p.name}"
            shutil.copy2(img_p, dest)
            counts["mature"] += 1

    print(f"  RawRipe external validation set ready: {counts['immature']} immature, {counts['mature']} mature.")
    return counts


def main():
    print("=" * 70)
    print("APPLE RIPENESS DATASET CURATION & CROPPING PIPELINE")
    print("=" * 70)

    setup_output_dirs()

    # 1. Process Test Split (Independent evaluation)
    test_json = DATASET_2026_DIR / "test.json"
    test_img_dir = DATASET_2026_DIR / "test"
    test_records = extract_crops_from_json(test_json, test_img_dir, "test")

    for rec in test_records:
        dest_p = OUTPUT_DIR / "test" / rec["class_label"] / rec["crop_filename"]
        rec["crop_image"].save(dest_p, quality=95)

    # 2. Process Train Split
    train_json = DATASET_2026_DIR / "train.json"
    train_img_dir = DATASET_2026_DIR / "train"
    train_records = extract_crops_from_json(train_json, train_img_dir, "train")

    # Split train into train (85%) and val (15%) by original image to prevent leakage
    unique_orig_images = sorted(list(set(r["original_image"] for r in train_records)))
    random.seed(42)
    random.shuffle(unique_orig_images)

    n_val_imgs = int(len(unique_orig_images) * 0.15)
    val_image_set = set(unique_orig_images[:n_val_imgs])
    train_image_set = set(unique_orig_images[n_val_imgs:])

    train_saved = {"immature": 0, "semi_mature": 0, "mature": 0}
    val_saved = {"immature": 0, "semi_mature": 0, "mature": 0}

    for rec in train_records:
        if rec["original_image"] in val_image_set:
            split_target = "val"
            val_saved[rec["class_label"]] += 1
        else:
            split_target = "train"
            train_saved[rec["class_label"]] += 1

        dest_p = OUTPUT_DIR / split_target / rec["class_label"] / rec["crop_filename"]
        rec["crop_image"].save(dest_p, quality=95)

    # 3. Process External RawRipe Dataset
    rawripe_counts = process_external_rawripe()

    # Summary Statistics
    test_counts = {c: len(list((OUTPUT_DIR / "test" / c).glob("*.jpg"))) for c in ["immature", "semi_mature", "mature"]}

    print("\n" + "=" * 70)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 70)
    print(f"TRAIN SET ({sum(train_saved.values())} crops):")
    for c, cnt in train_saved.items():
        print(f"  * {c:12s}: {cnt}")
    print(f"\nVAL SET   ({sum(val_saved.values())} crops):")
    for c, cnt in val_saved.items():
        print(f"  * {c:12s}: {cnt}")
    print(f"\nTEST SET  ({sum(test_counts.values())} crops):")
    for c, cnt in test_counts.items():
        print(f"  * {c:12s}: {cnt}")
    print(f"\nEXTERNAL RAWRIPE SET ({sum(rawripe_counts.values())} apples):")
    for c, cnt in rawripe_counts.items():
        print(f"  * {c:12s}: {cnt}")

    total_instances = sum(train_saved.values()) + sum(val_saved.values()) + sum(test_counts.values()) + sum(rawripe_counts.values())
    print(f"\nTOTAL CURATED APPLE CROPS: {total_instances}")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
