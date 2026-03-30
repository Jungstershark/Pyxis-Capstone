"""
Step 6: Benchmark trained models on GOLD TEST set (final reporting only).

This script should only be run AFTER model selection is complete (using val set).
The test set is never used for model selection.

Usage:
    cd Pyxis-Capstone/datasets/singapore_river
    python 6_evaluate.py
"""

import sys
import csv
from pathlib import Path

# Add model_finetuning to path for shared evaluate module
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_finetuning"))

from evaluate import evaluate_split

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_DATASET = SCRIPT_DIR / "gold"
WEIGHTS_DIR = SCRIPT_DIR / "5_weights"
OUT_CSV = SCRIPT_DIR / "6_test_benchmark.csv"

# Add your trained models here after step 5
RUNS = [
    # ("sg_y11n_t0_960", "5_weights/sg_y11n_t0_960/best.pt", 960),
    # ("sg_y11s_t0_960", "5_weights/sg_y11s_t0_960/best.pt", 960),
    # ("sg_y11m_t0_960", "5_weights/sg_y11m_t0_960/best.pt", 960),
]

FIELDS = [
    "run_name", "model_path", "imgsz",
    "ladder_recall", "ladder_precision",
    "person_recall", "person_precision",
    "ladder_avg_pred_conf", "person_avg_pred_conf",
    "ladder_area_mean", "ladder_area_p50", "ladder_area_p90",
    "n_images"
]

if __name__ == "__main__":
    if not RUNS:
        print("No runs configured. Uncomment entries in RUNS list after training.")
        sys.exit(0)

    rows = []
    for run_name, model_path, imgsz in RUNS:
        model_path = SCRIPT_DIR / Path(model_path)
        if not model_path.exists():
            print(f"[SKIP] missing {model_path}")
            continue

        print(f"\n=== TEST EVAL: {run_name} ===")
        m = evaluate_split(
            model_path=model_path,
            dataset_dir=GOLD_DATASET,
            split="test",
            conf=0.25,
            match_iou=0.5,
            imgsz=imgsz,
        )

        row = {
            "run_name": run_name,
            "model_path": str(model_path),
            "imgsz": imgsz,
            "ladder_recall": m["ladder_recall"],
            "ladder_precision": m["ladder_precision"],
            "person_recall": m["person_recall"],
            "person_precision": m["person_precision"],
            "ladder_avg_pred_conf": m["ladder_avg_pred_conf"],
            "person_avg_pred_conf": m["person_avg_pred_conf"],
            "ladder_area_mean": m["ladder_area_mean"],
            "ladder_area_p50": m["ladder_area_p50"],
            "ladder_area_p90": m["ladder_area_p90"],
            "n_images": m["n_images"],
        }
        rows.append(row)

        print(f"  Ladder  - recall: {row['ladder_recall']:.4f}  precision: {row['ladder_precision']:.4f}")
        print(f"  Person  - recall: {row['person_recall']:.4f}  precision: {row['person_precision']:.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote test benchmarks to {OUT_CSV}")
