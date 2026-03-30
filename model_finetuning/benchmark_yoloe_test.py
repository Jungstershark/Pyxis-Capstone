import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from ultralytics import YOLO  # YOLO works; YOLOE class also exists but not required
from evaluate import clamp01_xyxy, iou_xyxy, yolo_to_xyxy, load_yolo_labels, greedy_match


DATASET = Path("dataset_split_v2_gpu_corrected")
IMAGES_DIR = DATASET / "images" / "test"
LABELS_DIR = DATASET / "labels" / "test"

OUT_CSV = Path("test_benchmark_yoloe.csv")

# IMPORTANT: order must match your dataset class IDs:
# 0 = pilot_ladder, 1 = person
# PROMPTS = ["pilot ladder", "person"] <- PROMPT
PROMPTS = ["pilot ladder, rope ladder used for pilot transfer, wooden-step ladder with side ropes, maritime boarding ladder hanging off ship hull", "person, human crew member standing or climbing"]

# YOLOE weights to benchmark (add/remove as you like)
YOLOE_MODELS = [
    ("yoloe-11s-seg", "yoloe-11s-seg.pt"),
    ("yoloe-11m-seg", "yoloe-11m-seg.pt"),
    ("yoloe-11l-seg", "yoloe-11l-seg.pt"),
]

CONF = 0.25
MATCH_IOU = 0.5
IMGSZ = 960


def eval_model_on_test(model: YOLO) -> Dict[str, float]:
    # per-class metrics: 0 ladder, 1 person
    tp = {0: 0, 1: 0}
    fp = {0: 0, 1: 0}
    fn = {0: 0, 1: 0}
    conf_sum = {0: 0.0, 1: 0.0}
    conf_n = {0: 0, 1: 0}

    ladder_area_ratios = []

    image_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    for img_path in image_paths:
        label_path = LABELS_DIR / (img_path.stem + ".txt")

        gt_cls, gt_boxes_yolo = load_yolo_labels(label_path)
        gt_xyxy = [clamp01_xyxy(yolo_to_xyxy(b)) for b in gt_boxes_yolo]

        gt_by_class = {0: [], 1: []}
        for cls, box in zip(gt_cls.tolist(), gt_xyxy):
            if cls in gt_by_class:
                gt_by_class[cls].append(box)

        # box area sanity from GT for ladder
        for cls, box_yolo in zip(gt_cls.tolist(), gt_boxes_yolo):
            if cls == 0:
                _, _, w, h = box_yolo.tolist()
                ladder_area_ratios.append(w * h)

        # Predict
        r = model.predict(source=str(img_path), imgsz=IMGSZ, conf=CONF, iou=0.7, verbose=False)[0]

        pred_by_class = {0: [], 1: []}
        conf_by_class = {0: [], 1: []}

        if r.boxes is not None and len(r.boxes) > 0:
            w_img, h_img = r.orig_shape[1], r.orig_shape[0]

            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy()

            # sort by confidence desc for stable greedy matching
            order = np.argsort(-conf_arr)

            for k in order:
                cls = int(cls_arr[k])
                if cls not in pred_by_class:
                    continue
                x0, y0, x1, y1 = xyxy[k]
                b = np.array([x0 / w_img, y0 / h_img, x1 / w_img, y1 / h_img], dtype=np.float32)
                b = clamp01_xyxy(b)

                pred_by_class[cls].append(b)
                conf_by_class[cls].append(float(conf_arr[k]))

        # Match per class
        for c in [0, 1]:
            _tp, _fp, _fn = greedy_match(gt_by_class[c], pred_by_class[c], iou_thresh=MATCH_IOU)
            tp[c] += _tp
            fp[c] += _fp
            fn[c] += _fn
            if conf_by_class[c]:
                conf_sum[c] += float(np.sum(conf_by_class[c]))
                conf_n[c] += len(conf_by_class[c])

    def prec(c): return tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
    def rec(c): return tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
    def avg_conf(c): return conf_sum[c] / conf_n[c] if conf_n[c] > 0 else 0.0

    ladder_area_mean = float(np.mean(ladder_area_ratios)) if ladder_area_ratios else 0.0
    ladder_area_p50 = float(np.percentile(np.array(ladder_area_ratios), 50)) if ladder_area_ratios else 0.0
    ladder_area_p90 = float(np.percentile(np.array(ladder_area_ratios), 90)) if ladder_area_ratios else 0.0

    return {
        "ladder_recall": rec(0),
        "ladder_precision": prec(0),
        "person_recall": rec(1),
        "person_precision": prec(1),
        "ladder_avg_pred_conf": avg_conf(0),
        "person_avg_pred_conf": avg_conf(1),
        "ladder_area_mean": ladder_area_mean,
        "ladder_area_p50": ladder_area_p50,
        "ladder_area_p90": ladder_area_p90,
        "n_images": len(list(IMAGES_DIR.iterdir())),
    }


def main():
    assert IMAGES_DIR.exists() and LABELS_DIR.exists(), "Gold test dirs not found."

    fields = [
        "run_name", "model_weight", "imgsz", "conf", "match_iou",
        "ladder_recall", "ladder_precision",
        "person_recall", "person_precision",
        "ladder_avg_pred_conf", "person_avg_pred_conf",
        "ladder_area_mean", "ladder_area_p50", "ladder_area_p90",
        "n_images"
    ]

    rows: List[Dict] = []

    for run_name, weight in YOLOE_MODELS:
        print(f"\n=== YOLOE TEST BENCH: {run_name} ({weight}) ===")

        model = YOLO(weight)  # will auto-download if not present
        # Set text prompts / classes in *your* label order
        model.set_classes(PROMPTS)

        metrics = eval_model_on_test(model)

        row = {
            "run_name": run_name,
            "model_weight": weight,
            "imgsz": IMGSZ,
            "conf": CONF,
            "match_iou": MATCH_IOU,
            **metrics
        }
        rows.append(row)

        print(f"ladder_recall={metrics['ladder_recall']:.4f} ladder_precision={metrics['ladder_precision']:.4f}")
        print(f"person_recall={metrics['person_recall']:.4f} person_precision={metrics['person_precision']:.4f}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\n✅ Wrote YOLOE test benchmarks to {OUT_CSV}")


if __name__ == "__main__":
    main()