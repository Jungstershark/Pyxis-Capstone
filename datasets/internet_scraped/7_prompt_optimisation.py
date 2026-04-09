"""
Step 7: Prompt optimisation for GroundingDINO auto-labeling.

Benchmarks text prompts per class against gold val labels to find
the optimal prompt. Uses GroundingDINO only (no SAM) for speed.

Prompts are loaded from JSON files in 7_prompts/ so you can add
new batches without editing this script.

Usage:
    cd Pyxis-Capstone/datasets/internet_scraped

    # Run a specific batch
    python 7_prompt_optimisation.py --batch batch_1.json

    # Run all batches in 7_prompts/
    python 7_prompt_optimisation.py --all
"""

import argparse
import json
import sys
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from groundingdino.util.inference import load_model, load_image, predict

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_DIR = SCRIPT_DIR / "gold"
IMAGES_DIR = GOLD_DIR / "images" / "val"
LABELS_DIR = GOLD_DIR / "labels" / "val"

PROMPTS_DIR = SCRIPT_DIR / "7_prompts"
RESULTS_DIR = SCRIPT_DIR / "7_prompt_results"

WEIGHTS_DIR = REPO_ROOT / "data_processing" / "weights"
GDINO_CFG = WEIGHTS_DIR / "GroundingDINO_SwinT_OGC.cfg.py"
GDINO_CKPT = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.30
NMS_IOU = 0.50
MATCH_IOU = 0.50
MIN_BOX_PIXELS = 2

CLASS_CONFIG = {
    "ladder": {"class_id": 0, "max_det": 2},
    "person": {"class_id": 1, "max_det": 5},
}


# ----------------------------
# LABEL LOADING & MATCHING
# ----------------------------
def load_gold_labels(label_path, target_class):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            if cls != target_class:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x0, y0 = cx - w / 2.0, cy - h / 2.0
            x1, y1 = cx + w / 2.0, cy + h / 2.0
            boxes.append([x0, y0, x1, y1])
    return boxes


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def greedy_match(gt_boxes, pred_boxes, iou_thresh):
    matched = set()
    tp = 0
    for p in pred_boxes:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gt_boxes):
            if j in matched:
                continue
            s = iou(g, p)
            if s > best_iou:
                best_iou, best_j = s, j
        if best_iou >= iou_thresh and best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


# ----------------------------
# DETECTION (GroundingDINO only)
# ----------------------------
def clamp_and_fix(box, w, h):
    x0, y0, x1, y1 = box
    x0 = max(0, min(float(x0), w - 1))
    y0 = max(0, min(float(y0), h - 1))
    x1 = max(0, min(float(x1), w - 1))
    y1 = max(0, min(float(y1), h - 1))
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    if (x1 - x0) < MIN_BOX_PIXELS or (y1 - y0) < MIN_BOX_PIXELS:
        return None
    return [x0, y0, x1, y1]


def detect_with_prompt(gdino, image_path, prompt, max_det=5):
    image_source, image = load_image(str(image_path))
    h, w = image_source.shape[:2]

    boxes, logits, _ = predict(
        model=gdino, image=image, caption=prompt,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if len(boxes) == 0:
        return []

    boxes_xyxy, scores = [], []
    for b, score in zip(boxes, logits):
        cx, cy, bw, bh = b.tolist()
        x0 = (cx - bw / 2) * w
        y0 = (cy - bh / 2) * h
        x1 = (cx + bw / 2) * w
        y1 = (cy + bh / 2) * h
        fixed = clamp_and_fix([x0, y0, x1, y1], w, h)
        if fixed is None:
            continue
        boxes_xyxy.append(fixed)
        scores.append(float(score))

    if not boxes_xyxy:
        return []

    bt = torch.tensor(boxes_xyxy, dtype=torch.float32, device=DEVICE)
    st = torch.tensor(scores, dtype=torch.float32, device=DEVICE)
    keep = nms(bt, st, NMS_IOU)
    keep = keep[st[keep].argsort(descending=True)][:max_det]

    return [[bt[i][0].item() / w, bt[i][1].item() / h,
             bt[i][2].item() / w, bt[i][3].item() / h] for i in keep]


# ----------------------------
# BENCHMARK
# ----------------------------
def benchmark_prompts(gdino, prompts, class_id, class_name, max_det, image_paths):
    print(f"\n{'='*60}")
    print(f"Benchmarking {len(prompts)} prompts for {class_name} (class {class_id})")
    print(f"Gold val images: {len(image_paths)}")
    print(f"{'='*60}")

    results = []
    for pi, prompt in enumerate(prompts):
        total_tp, total_fp, total_fn = 0, 0, 0
        start = time.time()

        with torch.inference_mode():
            for img_path in image_paths:
                label_path = LABELS_DIR / (img_path.stem + ".txt")
                gt_boxes = load_gold_labels(label_path, class_id)
                pred_boxes = detect_with_prompt(gdino, img_path, prompt, max_det)
                tp, fp, fn = greedy_match(gt_boxes, pred_boxes, MATCH_IOU)
                total_tp += tp
                total_fp += fp
                total_fn += fn

        elapsed = time.time() - start
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "rank": 0,
            "prompt": prompt,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "time_sec": round(elapsed, 1),
        })
        print(f"  [{pi+1:2d}/{len(prompts)}] R={recall:.3f} P={precision:.3f} F1={f1:.3f} | \"{prompt}\"")

    results.sort(key=lambda r: (r["f1"], r["recall"]), reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


def write_results(results, csv_path):
    fields = ["rank", "prompt", "recall", "precision", "f1", "tp", "fp", "fn", "time_sec"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def run_batch(gdino, batch_path, image_paths):
    batch_name = batch_path.stem
    print(f"\n{'#'*60}")
    print(f"BATCH: {batch_name}")
    print(f"{'#'*60}")

    with open(batch_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    for class_name, cfg in CLASS_CONFIG.items():
        class_prompts = prompts.get(class_name, [])
        if not class_prompts:
            print(f"  No prompts for {class_name} in {batch_name}, skipping.")
            continue

        results = benchmark_prompts(
            gdino, class_prompts, cfg["class_id"], class_name, cfg["max_det"], image_paths
        )

        out_csv = RESULTS_DIR / f"{batch_name}_{class_name}.csv"
        write_results(results, out_csv)

        print(f"\n  TOP 3 ({class_name}):")
        for r in results[:3]:
            print(f"    #{r['rank']} R={r['recall']:.3f} P={r['precision']:.3f} F1={r['f1']:.3f} | \"{r['prompt']}\"")
        print(f"  Saved: {out_csv}")


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, help="Run a specific batch file (e.g. batch_1.json)")
    parser.add_argument("--all", action="store_true", help="Run all .json batches in 7_prompts/")
    args = parser.parse_args()

    if not args.batch and not args.all:
        parser.error("Specify --batch <file.json> or --all")

    assert IMAGES_DIR.exists(), f"Gold val images not found: {IMAGES_DIR}"
    assert LABELS_DIR.exists(), f"Gold val labels not found: {LABELS_DIR}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    print("Loading GroundingDINO...")
    gdino = load_model(str(GDINO_CFG), str(GDINO_CKPT)).to(DEVICE).eval()
    torch.backends.cudnn.benchmark = True

    if args.batch:
        batch_path = PROMPTS_DIR / args.batch
        assert batch_path.exists(), f"Batch file not found: {batch_path}"
        run_batch(gdino, batch_path, image_paths)
    else:
        batch_files = sorted(PROMPTS_DIR.glob("*.json"))
        if not batch_files:
            print(f"No .json files found in {PROMPTS_DIR}")
            sys.exit(1)
        for bf in batch_files:
            run_batch(gdino, bf, image_paths)

    print("\nDone. Results in 7_prompt_results/")
