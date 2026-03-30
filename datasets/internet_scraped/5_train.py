"""
Step 5: Train YOLO models on internet_scraped dataset.

Reads tier datasets from:
    4_augmented/t0/, 4_augmented/t1/, 4_augmented/t2/

Gold dataset for evaluation:
    gold/

Outputs:
    5_weights/<run_name>/best.pt
    5_runs/<run_name>/
    experiments.csv

Usage:
    cd Pyxis-Capstone/datasets/internet_scraped
    python 5_train.py
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "model_finetuning"))

from ultralytics import YOLO
from evaluate import evaluate_split


# ----------------------------
# GLOBAL HYGIENE
# ----------------------------
SEED = 42

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)
DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ----------------------------
# PATHS
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "5_runs"
WEIGHTS_DIR = SCRIPT_DIR / "5_weights"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DATASET = SCRIPT_DIR / "gold"

TIER_DATASETS = {
    0: SCRIPT_DIR / "4_augmented" / "t0",
    1: SCRIPT_DIR / "4_augmented" / "t1",
    2: SCRIPT_DIR / "4_augmented" / "t2",
}

CLASS_NAMES = {0: "pilot_ladder", 1: "person"}


# ----------------------------
# WRITE DATA.YAML
# ----------------------------
def write_data_yaml(dataset_dir: Path, out_path: Path):
    text = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        f"  0: {CLASS_NAMES[0]}\n"
        f"  1: {CLASS_NAMES[1]}\n"
    )
    out_path.write_text(text, encoding="utf-8")


# ----------------------------
# EXPERIMENT DEFINITIONS
# ----------------------------
@dataclass
class Experiment:
    run_name: str
    model: str           # e.g., "yolo11s.pt"
    tier: int            # 0/1/2
    imgsz: int           # 960 or 1280
    epochs: int = 50
    batch: int = 24
    lr0: Optional[float] = None
    close_mosaic: int = 10
    conf_eval: float = 0.25
    match_iou: float = 0.5
    workers: int = 4


# ----------------------------
# CSV LOGGING
# ----------------------------
CSV_PATH = SCRIPT_DIR / "experiments.csv"
CSV_FIELDS = [
    "run_name", "model", "tier", "imgsz", "epochs", "batch", "workers",
    "ladder_recall", "ladder_precision",
    "person_recall", "person_precision",
    "ladder_avg_pred_conf", "person_avg_pred_conf",
    "ladder_area_mean", "ladder_area_p50", "ladder_area_p90",
    "gold_val_images",
    "best_ckpt_path"
]

def append_csv(row: Dict):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


# ----------------------------
# TRAIN + EVAL
# ----------------------------
def train_one(exp: Experiment) -> Path:
    set_all_seeds(SEED)

    dataset_dir = TIER_DATASETS[exp.tier]
    assert dataset_dir.exists(), f"Missing dataset tier dir: {dataset_dir}"

    run_dir = RUNS_DIR / exp.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = run_dir / "data.yaml"
    write_data_yaml(dataset_dir, data_yaml)

    (run_dir / "exp_config.json").write_text(json.dumps(asdict(exp), indent=2), encoding="utf-8")

    model = YOLO(exp.model)

    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=exp.imgsz,
        epochs=exp.epochs,
        batch=exp.batch,
        seed=SEED,
        device=DEVICE,
        workers=exp.workers,
        project=str(RUNS_DIR),
        name=exp.run_name,
        exist_ok=True,
        close_mosaic=exp.close_mosaic,
        verbose=True,
    )

    if exp.lr0 is not None:
        train_kwargs["lr0"] = exp.lr0

    model.train(**train_kwargs)

    best_path = RUNS_DIR / exp.run_name / "weights" / "best.pt"
    assert best_path.exists(), f"best.pt not found at: {best_path}"

    out_dir = WEIGHTS_DIR / exp.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    final_best = out_dir / "best.pt"
    final_best.write_bytes(best_path.read_bytes())

    return final_best


def eval_gold_val(best_ckpt: Path, exp: Experiment) -> Dict[str, float]:
    return evaluate_split(
        model_path=best_ckpt,
        dataset_dir=GOLD_DATASET,
        split="val",
        conf=exp.conf_eval,
        match_iou=exp.match_iou,
        imgsz=exp.imgsz,
    )


def run_wave(experiments: List[Experiment]) -> List[Dict]:
    rows = []
    for exp in experiments:
        print(f"\n==== TRAIN: {exp.run_name} ====")
        best_ckpt = train_one(exp)
        print(f"Best checkpoint: {best_ckpt}")

        print(f"\n==== EVAL GOLD VAL: {exp.run_name} ====")
        metrics = eval_gold_val(best_ckpt, exp)

        run_dir = RUNS_DIR / exp.run_name
        (run_dir / "gold_val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        row = {
            "run_name": exp.run_name,
            "model": exp.model,
            "tier": exp.tier,
            "imgsz": exp.imgsz,
            "epochs": exp.epochs,
            "batch": exp.batch,
            "workers": exp.workers,
            "ladder_recall": round(metrics["ladder_recall"], 6),
            "ladder_precision": round(metrics["ladder_precision"], 6),
            "person_recall": round(metrics["person_recall"], 6),
            "person_precision": round(metrics["person_precision"], 6),
            "ladder_avg_pred_conf": round(metrics["ladder_avg_pred_conf"], 6),
            "person_avg_pred_conf": round(metrics["person_avg_pred_conf"], 6),
            "ladder_area_mean": round(metrics["ladder_area_mean"], 10),
            "ladder_area_p50": round(metrics["ladder_area_p50"], 10),
            "ladder_area_p90": round(metrics["ladder_area_p90"], 10),
            "gold_val_images": int(metrics["n_images"]),
            "best_ckpt_path": str(best_ckpt),
        }

        append_csv(row)
        rows.append(row)

        print("\n---- KPI SUMMARY ----")
        for k in ["ladder_recall", "ladder_precision", "person_recall", "person_precision"]:
            print(f"{k}: {row[k]}")
        print("---------------------\n")

    return rows


# ----------------------------
# CONFIGURE YOUR EXPERIMENTS HERE
# ----------------------------
if __name__ == "__main__":
    # Wave 1: Backbone selection on Tier 0 (same as original)
    wave1 = [
        Experiment(run_name="y11n_t0_960", model="yolo11n.pt", tier=0, imgsz=960, batch=32),
        Experiment(run_name="y11s_t0_960", model="yolo11s.pt", tier=0, imgsz=960, batch=24),
        Experiment(run_name="y11m_t0_960", model="yolo11m.pt", tier=0, imgsz=960, batch=16),
        Experiment(run_name="y11l_t0_960", model="yolo11l.pt", tier=0, imgsz=960, batch=12),
        Experiment(run_name="y11x_t0_960", model="yolo11x.pt", tier=0, imgsz=960, batch=8),
    ]
    run_wave(wave1)

    print("\nWave 1 done. Review experiments.csv, then add Wave 2 experiments.")
