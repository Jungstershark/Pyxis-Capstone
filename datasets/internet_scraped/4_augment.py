"""
Step 4: Create augmentation tiers (t0/t1/t2) from gold-corrected labels.

Prerequisite: You have manually corrected labels in LabelImg and placed them in:
    gold/images/{train, val, test}
    gold/labels/{train, val, test}

Creates:
    4_augmented/t0/  (baseline copy, no augmentation)
    4_augmented/t1/  (lighting robustness)
    4_augmented/t2/  (motion robustness)

Each tier has: images/{train,val,test} + labels/{train,val,test}
Val/test remain unchanged (gold standard) across all tiers.

Usage:
    cd Pyxis-Capstone/datasets/internet_scraped
    python 4_augment.py
"""

import sys
import shutil
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_finetuning"))

from augment import copy_gold_splits, copy_train, augment_train

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_DATASET = SCRIPT_DIR / "gold"
AUGMENTED_DIR = SCRIPT_DIR / "4_augmented"

if __name__ == "__main__":
    assert GOLD_DATASET.exists(), f"Gold dataset not found: {GOLD_DATASET}"
    assert (GOLD_DATASET / "images" / "train").exists(), "Missing gold/images/train"
    assert (GOLD_DATASET / "labels" / "train").exists(), "Missing gold/labels/train"

    summary_records = []
    tier0_counts = None

    for tier in [0, 1, 2]:
        print(f"\n=== Creating Tier {tier} dataset ===")
        output_dataset = AUGMENTED_DIR / f"t{tier}"

        if output_dataset.exists():
            shutil.rmtree(output_dataset)

        (output_dataset / "images").mkdir(parents=True, exist_ok=True)
        (output_dataset / "labels").mkdir(parents=True, exist_ok=True)

        copy_gold_splits(GOLD_DATASET, output_dataset)

        if tier == 0:
            train_images_written, empty_labels, boxes_per_class = copy_train(GOLD_DATASET, output_dataset)
        else:
            train_images_written, empty_labels, boxes_per_class = augment_train(GOLD_DATASET, output_dataset, tier)

        ladder_boxes = boxes_per_class.get(0, 0)
        person_boxes = boxes_per_class.get(1, 0)

        if tier == 0:
            tier0_counts = (ladder_boxes, person_boxes)

        t0_ladder, t0_person = tier0_counts if tier0_counts else (ladder_boxes, person_boxes)
        ladder_ratio = ladder_boxes / t0_ladder if t0_ladder > 0 else 1.0
        person_ratio = person_boxes / t0_person if t0_person > 0 else 1.0

        print(f"\n--- Tier {tier} Summary ---")
        print(f"Train images: {train_images_written}")
        print(f"Empty labels: {empty_labels}")
        print(f"Ladder boxes: {ladder_boxes} (ratio vs t0: {ladder_ratio:.3f})")
        print(f"Person boxes: {person_boxes} (ratio vs t0: {person_ratio:.3f})")

        summary_records.append({
            "tier": tier,
            "train_images": train_images_written,
            "empty_labels": empty_labels,
            "ladder_boxes": ladder_boxes,
            "person_boxes": person_boxes,
            "ladder_ratio_vs_t0": round(ladder_ratio, 4),
            "person_ratio_vs_t0": round(person_ratio, 4),
        })

    summary_csv = SCRIPT_DIR / "tier_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_records[0].keys())
        writer.writeheader()
        writer.writerows(summary_records)

    print(f"\nAll tiers created. Summary: {summary_csv}")
