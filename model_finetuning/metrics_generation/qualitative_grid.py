# python metrics_generation/qualitative_grid.py \
#   --dataset dataset_split_v2_gpu_corrected \
#   --split test \
#   --yoloe yoloe-11m-seg.pt \
#   --yolov11 weights/y11m_t0_960/best.pt \
#   --out metrics_generation/out/qualitative_grid.png

from __future__ import annotations

import argparse
from pathlib import Path
import random
import math

import cv2
import numpy as np
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes(classes_path: Path) -> list[str]:
    if classes_path.exists():
        return [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return []


def yolo_label_boxes(label_path: Path):
    """
    Returns list of (cls_id, x_c, y_c, w, h) in normalized coords.
    """
    if not label_path.exists():
        return []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    out = []
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            out.append((cid, xc, yc, w, h))
        except:
            continue
    return out


def norm_to_xyxy(box, W, H):
    _, xc, yc, w, h = box
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_image_metrics(img_bgr: np.ndarray):
    """
    Returns:
      brightness (0-255),
      blur_score (variance of Laplacian; lower = blurrier)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, blur_score


def truncation_proxy_xyxy(x1, y1, x2, y2, W, H, margin_px=3):
    """
    Proxy for occlusion/truncation: bbox touches image border.
    """
    touches = 0
    if x1 <= margin_px: touches += 1
    if y1 <= margin_px: touches += 1
    if x2 >= W - margin_px: touches += 1
    if y2 >= H - margin_px: touches += 1
    return touches  # 0..4


def find_hard_cases(
    images_dir: Path,
    labels_dir: Path,
    ladder_class_id: int,
    seed: int = 42
):
    """
    Picks 4 images:
      - small ladder (min ladder area)
      - occluded/truncated ladder (bbox touches borders)
      - low contrast / dark (min brightness)
      - motion blur (min blur_score)
    Uses GT labels to compute ladder bbox size.
    """
    random.seed(seed)

    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise RuntimeError(f"No images found in {images_dir}")

    candidates = []
    for img_path in imgs:
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = yolo_label_boxes(label_path)

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # find ladder boxes
        ladder_boxes = [b for b in gt_boxes if b[0] == ladder_class_id]
        if not ladder_boxes:
            continue  # for ladder-focused qualitative panel, require ladder exists in GT

        # use the largest ladder box if multiple
        ladder_xyxys = [norm_to_xyxy(b, W, H) for b in ladder_boxes]
        areas = []
        trunc_scores = []
        for (x1, y1, x2, y2) in ladder_xyxys:
            x1c, y1c, x2c, y2c = clamp(x1, 0, W), clamp(y1, 0, H), clamp(x2, 0, W), clamp(y2, 0, H)
            area = max(0.0, (x2c - x1c) * (y2c - y1c)) / float(W * H)  # normalized
            areas.append(area)
            trunc_scores.append(truncation_proxy_xyxy(x1c, y1c, x2c, y2c, W, H))

        ladder_area = float(max(areas))
        trunc_score = int(max(trunc_scores))

        brightness, blur_score = compute_image_metrics(img)

        candidates.append({
            "img": img_path,
            "label": label_path,
            "ladder_area": ladder_area,
            "trunc_score": trunc_score,
            "brightness": brightness,
            "blur_score": blur_score,
        })

    if len(candidates) < 4:
        raise RuntimeError(f"Not enough ladder-labelled candidates found (got {len(candidates)}).")

    # Choose distinct cases (avoid duplicates)
    small = min(candidates, key=lambda d: d["ladder_area"])
    occl = max(candidates, key=lambda d: d["trunc_score"])
    dark = min(candidates, key=lambda d: d["brightness"])
    blurry = min(candidates, key=lambda d: d["blur_score"])

    picks = [small, occl, dark, blurry]

    # Ensure uniqueness: if duplicates occur, fallback to next best
    unique = []
    used = set()
    for p in picks:
        if p["img"] not in used:
            unique.append(p)
            used.add(p["img"])

    # If we lost some due to duplication, fill with random distinct candidates
    if len(unique) < 4:
        remaining = [c for c in candidates if c["img"] not in used]
        random.shuffle(remaining)
        unique.extend(remaining[: (4 - len(unique))])

    # Attach titles
    unique[0]["case"] = "Small ladder"
    unique[1]["case"] = "Occluded / truncated"
    unique[2]["case"] = "Low contrast / dark"
    unique[3]["case"] = "Motion blur"

    return unique


def render_predictions(model: YOLO, img_bgr: np.ndarray):
    """
    Runs prediction and returns a plotted BGR image.
    """
    results = model.predict(img_bgr, verbose=False)
    plotted = results[0].plot()  # returns BGR ndarray
    return plotted


def add_title_bar(img_bgr: np.ndarray, title: str):
    h, w = img_bgr.shape[:2]
    bar_h = max(32, h // 14)
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:bar_h, :] = (20, 20, 20)
    out[bar_h:, :] = img_bgr
    cv2.putText(out, title, (12, int(bar_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def hstack_pad(a, b):
    # same height
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    def pad(img, target_h):
        hi, wi = img.shape[:2]
        if hi == target_h:
            return img
        pad_h = target_h - hi
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    a2 = pad(a, h)
    b2 = pad(b, h)
    return np.hstack([a2, b2])


def vstack_pad(rows):
    widths = [r.shape[1] for r in rows]
    W = max(widths)
    out_rows = []
    for r in rows:
        h, w = r.shape[:2]
        if w < W:
            r = cv2.copyMakeBorder(r, 0, 0, 0, W-w, cv2.BORDER_CONSTANT, value=(0,0,0))
        out_rows.append(r)
    return np.vstack(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="dataset_split_v2_gpu_corrected")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--yoloe", type=str, required=True, help="Path to YOLO-E weights (.pt)")
    ap.add_argument("--yolov11", type=str, required=True, help="Path to fine-tuned YOLOv11 weights (.pt)")
    ap.add_argument("--ladder_class_name", type=str, default="pilot_ladder", help="Class name for ladder in classes.txt")
    ap.add_argument("--out", type=str, default="metrics_generation/out/qualitative_grid.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = Path(args.dataset)
    images_dir = ds / "images" / args.split
    labels_dir = ds / "labels" / args.split
    classes = load_classes(ds / "classes.txt")

    if classes and args.ladder_class_name in classes:
        ladder_id = classes.index(args.ladder_class_name)
    else:
        # fallback: assume ladder is class 0
        ladder_id = 0

    cases = find_hard_cases(images_dir, labels_dir, ladder_id, seed=args.seed)

    m_yoloe = YOLO(args.yoloe)
    m_y11 = YOLO(args.yolov11)

    rows = []
    for c in cases:
        img = cv2.imread(str(c["img"]))
        if img is None:
            continue

        pred_e = render_predictions(m_yoloe, img)
        pred_11 = render_predictions(m_y11, img)

        left = add_title_bar(pred_e, f"YOLO-E (PoC) — {c['case']}")
        right = add_title_bar(pred_11, f"YOLOv11 (Fine-tuned) — {c['case']}")

        pair = hstack_pad(left, right)
        rows.append(pair)

    grid = vstack_pad(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[OK] Wrote qualitative panel -> {out_path.resolve()}")


if __name__ == "__main__":
    main()