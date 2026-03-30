import csv
from pathlib import Path
from evaluate import evaluate_split

DATASET = Path("dataset_split_v2_gpu_corrected")
WEIGHTS_DIR = Path("weights")  # model_finetuning/weights
OUT_CSV = Path("test_benchmark.csv")

RUNS = [
    # ("y11n_t0_960", "weights/y11n_t0_960/best.pt", 960),
    # ("y11s_t0_960", "weights/y11s_t0_960/best.pt", 960),
    # ("y11m_t0_960", "weights/y11m_t0_960/best.pt", 960),
    # ("y11l_t0_960", "weights/y11l_t0_960/best.pt", 960),
    # ("y11x_t0_960", "weights/y11x_t0_960/best.pt", 960),
    
    # ----------------------
    # Tier 1
    # ----------------------
    ("y11s_t1_960", "weights/y11s_t1_960/best.pt", 960),
    ("y11m_t1_960", "weights/y11m_t1_960/best.pt", 960),

    # ----------------------
    # Tier 2
    # ----------------------
    ("y11s_t2_960", "weights/y11s_t2_960/best.pt", 960),
    ("y11m_t2_960", "weights/y11m_t2_960/best.pt", 960),
]

FIELDS = [
    "run_name", "model_path", "imgsz",
    "ladder_recall", "ladder_precision",
    "person_recall", "person_precision",
    "ladder_avg_pred_conf", "person_avg_pred_conf",
    "ladder_area_mean", "ladder_area_p50", "ladder_area_p90",
    "n_images"
]

rows = []
for run_name, model_path, imgsz in RUNS:
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[SKIP] missing {model_path}")
        continue

    print(f"\n=== TEST EVAL: {run_name} ===")
    m = evaluate_split(
        model_path=model_path,
        dataset_dir=DATASET,
        split="test",
        conf=0.25,
        match_iou=0.5,
        imgsz=imgsz
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

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Wrote test benchmarks to {OUT_CSV}")