from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


# ----------------------------
# BOX UTILS
# ----------------------------
def yolo_to_xyxy(box_yolo: np.ndarray) -> np.ndarray:
    """[cx,cy,w,h] normalized -> [x0,y0,x1,y1] normalized"""
    cx, cy, w, h = box_yolo
    x0 = cx - w / 2.0
    y0 = cy - h / 2.0
    x1 = cx + w / 2.0
    y1 = cy + h / 2.0
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def clamp01_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """Clamp xyxy to [0,1] and enforce x0<=x1, y0<=y1."""
    x0, y0, x1, y1 = xyxy.tolist()
    x0 = float(max(0.0, min(1.0, x0)))
    y0 = float(max(0.0, min(1.0, y0)))
    x1 = float(max(0.0, min(1.0, x1)))
    y1 = float(max(0.0, min(1.0, y1)))
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for normalized xyxy boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    iw = max(0.0, inter_x1 - inter_x0)
    ih = max(0.0, inter_y1 - inter_y0)
    inter = iw * ih

    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)

    denom = a_area + b_area - inter
    if denom <= 0:
        return 0.0
    return float(inter / denom)


def greedy_match(
    gt_boxes: List[np.ndarray],
    pred_boxes_sorted: List[np.ndarray],
    iou_thresh: float
) -> Tuple[int, int, int]:
    """
    Greedy matching for TP/FP/FN (single class).
    Assumes pred_boxes_sorted are ordered by descending confidence.
    Returns (tp, fp, fn).
    """
    matched_gt = set()
    tp = 0

    for p in pred_boxes_sorted:
        best_iou = 0.0
        best_j = -1
        for j, g in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            score = iou_xyxy(g, p)
            if score > best_iou:
                best_iou = score
                best_j = j

        if best_iou >= iou_thresh and best_j >= 0:
            matched_gt.add(best_j)
            tp += 1

    fp = len(pred_boxes_sorted) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn


# ----------------------------
# LABEL LOADING
# ----------------------------
def load_yolo_labels(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: (classes[N], boxes_yolo[N,4]) normalized.
    """
    if not label_path.exists():
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    classes = []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                classes.append(cls)
                boxes.append([cx, cy, w, h])
            except ValueError:
                continue

    if not boxes:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    return np.array(classes, dtype=np.int64), np.array(boxes, dtype=np.float32)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    conf_sum: float = 0.0
    conf_n: int = 0

    def precision(self) -> float:
        d = self.tp + self.fp
        return float(self.tp / d) if d > 0 else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return float(self.tp / d) if d > 0 else 0.0

    def avg_conf(self) -> float:
        return float(self.conf_sum / self.conf_n) if self.conf_n > 0 else 0.0


def percentile(x: List[float], p: float) -> float:
    if not x:
        return 0.0
    return float(np.percentile(np.array(x, dtype=np.float32), p))


def evaluate_split(
    model_path: Path,
    dataset_dir: Path,
    split: str,
    conf: float = 0.25,
    match_iou: float = 0.5,
    imgsz: int = 960,
) -> Dict[str, float]:
    """
    Evaluate model on gold labels for a split (val/test).
    Expects:
      dataset_dir/images/{split}
      dataset_dir/labels/{split}
    """
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    assert images_dir.exists(), f"Missing images dir: {images_dir}"
    assert labels_dir.exists(), f"Missing labels dir: {labels_dir}"

    model = YOLO(str(model_path))

    # per-class metrics: 0 ladder, 1 person
    m = {0: Metrics(), 1: Metrics()}

    ladder_area_ratios = []

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")

        gt_cls, gt_boxes_yolo = load_yolo_labels(label_path)
        gt_xyxy = [clamp01_xyxy(yolo_to_xyxy(b)) for b in gt_boxes_yolo]

        gt_by_class = {0: [], 1: []}
        for cls, box in zip(gt_cls.tolist(), gt_xyxy):
            if cls in gt_by_class:
                gt_by_class[cls].append(box)

        # ladder box area sanity from GT
        for cls, box_yolo in zip(gt_cls.tolist(), gt_boxes_yolo):
            if cls == 0:
                _, _, w, h = box_yolo.tolist()
                ladder_area_ratios.append(w * h)

        # Predict
        pred = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            iou=0.7,  # model internal NMS IoU
            verbose=False
        )[0]

        pred_by_class = {0: [], 1: []}
        conf_by_class = {0: [], 1: []}

        if pred.boxes is not None and len(pred.boxes) > 0:
            w_img, h_img = pred.orig_shape[1], pred.orig_shape[0]

            xyxy = pred.boxes.xyxy.cpu().numpy()
            cls_arr = pred.boxes.cls.cpu().numpy().astype(int)
            conf_arr = pred.boxes.conf.cpu().numpy()

            # ✅ Sort by confidence descending (critical)
            order = np.argsort(-conf_arr)

            for k in order:
                cls = cls_arr[k]
                if cls not in pred_by_class:
                    continue

                x0, y0, x1, y1 = xyxy[k]
                b = np.array([x0 / w_img, y0 / h_img, x1 / w_img, y1 / h_img], dtype=np.float32)

                # ✅ Clamp normalized box to [0,1] and enforce ordering
                b = clamp01_xyxy(b)

                pred_by_class[cls].append(b)
                conf_by_class[cls].append(float(conf_arr[k]))

        # Match per class (pred boxes already sorted by confidence)
        for cls in [0, 1]:
            tp, fp, fn = greedy_match(gt_by_class[cls], pred_by_class[cls], iou_thresh=match_iou)
            m[cls].tp += tp
            m[cls].fp += fp
            m[cls].fn += fn
            # average confidence logging (all preds of this class)
            if conf_by_class[cls]:
                m[cls].conf_sum += float(np.sum(conf_by_class[cls]))
                m[cls].conf_n += len(conf_by_class[cls])

    results = {
        # provenance
        "model_path": str(model_path),
        "dataset_dir": str(dataset_dir),
        "split": split,
        "conf": conf,
        "match_iou": match_iou,
        "imgsz": imgsz,

        # KPIs
        "ladder_precision": m[0].precision(),
        "ladder_recall": m[0].recall(),
        "person_precision": m[1].precision(),
        "person_recall": m[1].recall(),

        # optional calibration signals
        "ladder_avg_pred_conf": m[0].avg_conf(),
        "person_avg_pred_conf": m[1].avg_conf(),

        # box sanity from GT
        "ladder_area_mean": float(np.mean(ladder_area_ratios)) if ladder_area_ratios else 0.0,
        "ladder_area_p50": percentile(ladder_area_ratios, 50),
        "ladder_area_p90": percentile(ladder_area_ratios, 90),

        "n_images": len(image_paths),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--dataset", type=str, required=True, help="Path to gold dataset_split_v1_corrected")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match_iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()

    out = evaluate_split(
        model_path=Path(args.model),
        dataset_dir=Path(args.dataset),
        split=args.split,
        conf=args.conf,
        match_iou=args.match_iou,
        imgsz=args.imgsz,
    )

    print("\n=== GOLD EVAL RESULTS ===")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")