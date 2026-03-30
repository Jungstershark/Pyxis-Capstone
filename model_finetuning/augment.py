import shutil
import random
from pathlib import Path
import csv

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


# ----------------------------
# CONFIG
# ----------------------------
BASE_DATASET = Path("dataset_split_v2_gpu_corrected")
OUTPUT_PREFIX = Path("dataset_split_v2_gpu_corrected_t")

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
if hasattr(A, "set_seed"):
    A.set_seed(SEED)

# Class-specific minimum normalized areas
# 0: ladder (thin/far allowed), 1: person
MIN_BOX_AREA_BY_CLASS = {0: 0.00003, 1: 0.00010}
MIN_BOX_PIXELS = 2  # skip too-tiny boxes in pixel terms


# ----------------------------
# AUGMENTATION TIERS
# ----------------------------
def get_augmentor(tier: int):
    if tier == 0:
        return None

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.30,
        check_each_transform=False,  # speed
    )

    if tier == 1:
        return A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.3),
            ],
            bbox_params=bbox_params,
        )

    if tier == 2:
        return A.Compose(
            [
                A.MotionBlur(blur_limit=5, p=0.30),
                A.GaussianBlur(blur_limit=3, p=0.30),
                A.RandomBrightnessContrast(p=0.30),
                A.CoarseDropout(max_holes=3, max_height=32, max_width=32, p=0.20),
            ],
            bbox_params=bbox_params,
        )

    raise ValueError(f"Invalid tier: {tier}")


# ----------------------------
# COPY VAL/TEST UNCHANGED (GOLD)
# ----------------------------
def copy_gold_splits(src: Path, dst: Path):
    for split in ["val", "test"]:
        shutil.copytree(src / "images" / split, dst / "images" / split)
        shutil.copytree(src / "labels" / split, dst / "labels" / split)


# ----------------------------
# YOLO LABEL IO
# ----------------------------
def read_yolo_labels(label_path: Path):
    bboxes, class_labels = [], []
    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                bboxes.append([cx, cy, w, h])
                class_labels.append(cls)
            except ValueError:
                continue
    return bboxes, class_labels


def write_yolo_labels(label_path: Path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for bbox, cls in zip(bboxes, class_labels):
            cx, cy, w, h = bbox
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ----------------------------
# BOX SANITIZATION (EDGE-BASED CLAMP)
# ----------------------------
def yolo_to_xyxy(yolo_box):
    cx, cy, bw, bh = yolo_box
    x0 = cx - bw / 2.0
    y0 = cy - bh / 2.0
    x1 = cx + bw / 2.0
    y1 = cy + bh / 2.0
    return [x0, y0, x1, y1]


def xyxy_to_yolo(xyxy_box):
    x0, y0, x1, y1 = xyxy_box
    bw = x1 - x0
    bh = y1 - y0
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0
    return [cx, cy, bw, bh]


def clamp_xyxy01(xyxy_box):
    x0, y0, x1, y1 = xyxy_box
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)
    return [x0, y0, x1, y1]


def sanitize_boxes_yolo(bboxes, class_labels, img_w, img_h):
    cleaned_boxes, cleaned_labels = [], []

    for bbox, cls in zip(bboxes, class_labels):
        cx, cy, bw, bh = bbox
        if bw <= 0 or bh <= 0:
            continue

        xyxy = clamp_xyxy01(yolo_to_xyxy([cx, cy, bw, bh]))
        x0, y0, x1, y1 = xyxy
        bw2, bh2 = (x1 - x0), (y1 - y0)

        if bw2 <= 0 or bh2 <= 0:
            continue

        # pixel-size guard (uses provided shape)
        if (bw2 * img_w) < MIN_BOX_PIXELS or (bh2 * img_h) < MIN_BOX_PIXELS:
            continue

        # area guard (class-specific)
        min_area = MIN_BOX_AREA_BY_CLASS.get(int(cls), 0.00005)
        if (bw2 * bh2) < min_area:
            continue

        yolo = xyxy_to_yolo(xyxy)

        # final clamp (center/size within [0,1])
        yolo[0] = max(0.0, min(1.0, yolo[0]))
        yolo[1] = max(0.0, min(1.0, yolo[1]))
        yolo[2] = max(0.0, min(1.0, yolo[2]))
        yolo[3] = max(0.0, min(1.0, yolo[3]))

        cleaned_boxes.append(yolo)
        cleaned_labels.append(int(cls))

    return cleaned_boxes, cleaned_labels


# ----------------------------
# TIER 0: COPY TRAIN (NO RECOMPRESS)
# ----------------------------
def copy_train(src: Path, dst: Path):
    src_img_dir = src / "images" / "train"
    src_lbl_dir = src / "labels" / "train"
    dst_img_dir = dst / "images" / "train"
    dst_lbl_dir = dst / "labels" / "train"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_images_written = 0
    empty_labels = 0
    boxes_per_class = {}

    for img_path in tqdm(sorted(src_img_dir.iterdir()), desc="Copy train (Tier0)"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        shutil.copy2(img_path, dst_img_dir / img_path.name)
        train_images_written += 1

        label_path = src_lbl_dir / (img_path.stem + ".txt")
        out_label_path = dst_lbl_dir / (img_path.stem + ".txt")

        if label_path.exists():
            shutil.copy2(label_path, out_label_path)

            bboxes, labels = read_yolo_labels(label_path)
            if len(labels) == 0:
                empty_labels += 1
            for cls in labels:
                boxes_per_class[cls] = boxes_per_class.get(cls, 0) + 1
        else:
            out_label_path.write_text("", encoding="utf-8")
            empty_labels += 1

    return train_images_written, empty_labels, boxes_per_class


# ----------------------------
# AUGMENT TRAIN (Tier 1/2)
# ----------------------------
def augment_train(src: Path, dst: Path, tier: int):
    aug = get_augmentor(tier)

    src_img_dir = src / "images" / "train"
    src_lbl_dir = src / "labels" / "train"
    dst_img_dir = dst / "images" / "train"
    dst_lbl_dir = dst / "labels" / "train"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_images_written = 0
    empty_labels = 0
    boxes_per_class = {}

    for img_path in tqdm(sorted(src_img_dir.iterdir()), desc=f"Augment train (Tier{tier})"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = src_lbl_dir / (img_path.stem + ".txt")

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue  # unreadable
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = read_yolo_labels(label_path)

        # Pre-sanitize BEFORE albumentations (fix strict bbox bounds)
        h0, w0 = image_rgb.shape[:2]
        bboxes, class_labels = sanitize_boxes_yolo(bboxes, class_labels, img_w=w0, img_h=h0)
        
        if aug is not None:
            transformed = aug(image=image_rgb, bboxes=bboxes, class_labels=class_labels)
            image_rgb = transformed["image"]
            bboxes = list(transformed["bboxes"])
            class_labels = list(transformed["class_labels"])

        # Use transformed shape for pixel checks (future-proof)
        h2, w2 = image_rgb.shape[:2]
        bboxes, class_labels = sanitize_boxes_yolo(bboxes, class_labels, img_w=w2, img_h=h2)

        # Save image
        out_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(dst_img_dir / img_path.name), out_bgr)
        train_images_written += 1

        # Save labels
        out_label_path = dst_lbl_dir / (img_path.stem + ".txt")
        write_yolo_labels(out_label_path, bboxes, class_labels)

        if len(class_labels) == 0:
            empty_labels += 1

        for cls in class_labels:
            boxes_per_class[int(cls)] = boxes_per_class.get(int(cls), 0) + 1

    return train_images_written, empty_labels, boxes_per_class


# ----------------------------
# MAIN + TIER SUMMARY
# ----------------------------
if __name__ == "__main__":
    import os
    print(os.getcwd())
    summary_records = []
    tier0_counts = None

    summary_csv_path = OUTPUT_PREFIX.parent / "tier_summary.csv"
    summary_txt_path = OUTPUT_PREFIX.parent / "tier_summary.txt"

    for tier in [0, 1, 2]:
        print(f"\n=== Creating Tier {tier} dataset ===")
        output_dataset = OUTPUT_PREFIX.parent / f"{OUTPUT_PREFIX.name}{tier}"

        if output_dataset.exists():
            shutil.rmtree(output_dataset)

        (output_dataset / "images").mkdir(parents=True, exist_ok=True)
        (output_dataset / "labels").mkdir(parents=True, exist_ok=True)

        # Copy gold val/test unchanged
        copy_gold_splits(BASE_DATASET, output_dataset)

        # Train split handling
        if tier == 0:
            train_images_written, empty_labels, boxes_per_class = copy_train(BASE_DATASET, output_dataset)
        else:
            train_images_written, empty_labels, boxes_per_class = augment_train(BASE_DATASET, output_dataset, tier)

        ladder_boxes = boxes_per_class.get(0, 0)
        person_boxes = boxes_per_class.get(1, 0)

        if tier == 0:
            tier0_counts = (ladder_boxes, person_boxes)

        t0_ladder, t0_person = tier0_counts if tier0_counts else (ladder_boxes, person_boxes)

        ladder_ratio = ladder_boxes / t0_ladder if t0_ladder > 0 else 1.0
        person_ratio = person_boxes / t0_person if t0_person > 0 else 1.0

        print("\n--- Tier Summary ---")
        print(f"Tier: {tier}")
        print(f"Train images written: {train_images_written}")
        print(f"Empty train label files: {empty_labels}")
        print(f"Total boxes (class 0 ladder): {ladder_boxes}")
        print(f"Total boxes (class 1 person): {person_boxes}")
        print(f"Ladder ratio vs Tier0: {ladder_ratio:.3f}")
        print(f"Person ratio vs Tier0: {person_ratio:.3f}")
        print("--------------------\n")

        summary_records.append({
            "tier": tier,
            "train_images": train_images_written,
            "empty_labels": empty_labels,
            "ladder_boxes": ladder_boxes,
            "person_boxes": person_boxes,
            "ladder_ratio_vs_t0": round(ladder_ratio, 4),
            "person_ratio_vs_t0": round(person_ratio, 4),
        })

    # Save CSV
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_records[0].keys())
        writer.writeheader()
        writer.writerows(summary_records)

    # Save TXT
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        for row in summary_records:
            f.write(f"Tier {row['tier']}\n")
            for k, v in row.items():
                if k != "tier":
                    f.write(f"  {k}: {v}\n")
            f.write("\n")

    print("✅ All tiers created successfully.")
    print(f"📄 Summary saved to: {summary_csv_path}")