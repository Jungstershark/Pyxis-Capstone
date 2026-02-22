"""
Grounded SAM Auto-Labeler (2-class, clean + progress)

- Classes:
    0: pilot_ladder
    1: person (pilot/crew)

Pipeline per image (per class):
    DINO -> clamp/sanitize boxes -> pre-NMS -> topK -> SAM refine -> post-NMS -> YOLO labels

Adds:
- Explicit score-sorting after NMS (guaranteed top-K)
- Box clamping + x0<x1/y0<y1 enforcement + tiny-box skip (SAM-safe)
- Post-SAM NMS (removes residual duplicates after tightening)
- Granular progress updates every N images (ETA, avg time/img, class hit rates)
- Runs under torch.inference_mode() for speed
"""

from pathlib import Path
import time
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import nms

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# ----------------------------
# CONFIG
# ----------------------------
CLASSES = {
    0: {"name": "pilot_ladder", "prompt": "pilot ladder. rope ladder.", "max_det": 2},
    1: {"name": "person", "prompt": "person. pilot. crew member.", "max_det": 5},
}

# DINO thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.30

# NMS
NMS_IOU_PRE = 0.50
NMS_IOU_POST = 0.50

# Sanity filters
MIN_BOX_PIXELS = 2  # skip boxes smaller than this (width or height)

# Progress reporting
PRINT_EVERY = 10  # print stats every N images

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# UTILS
# ----------------------------
def clamp_and_fix_xyxy(box, w, h):
    """
    Clamp xyxy box into image boundaries, enforce x0<x1 and y0<y1.
    Returns a valid [x0,y0,x1,y1] or None if degenerate/tiny.
    """
    x0, y0, x1, y1 = box

    # clamp
    x0 = max(0.0, min(float(x0), w - 1.0))
    y0 = max(0.0, min(float(y0), h - 1.0))
    x1 = max(0.0, min(float(x1), w - 1.0))
    y1 = max(0.0, min(float(y1), h - 1.0))

    # enforce ordering
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0, y1 = (y0, y1) if y0 <= y1 else (y1, y0)

    # skip tiny/degenerate
    if (x1 - x0) < MIN_BOX_PIXELS or (y1 - y0) < MIN_BOX_PIXELS:
        return None

    return [x0, y0, x1, y1]


def xyxy_to_yolo_norm(box_xyxy, w, h):
    """
    Convert xyxy pixel box into YOLO normalized cx cy w h (clamped to [0,1]).
    Assumes box is already valid.
    """
    x0, y0, x1, y1 = box_xyxy

    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    cx = x0 + bw / 2.0
    cy = y0 + bh / 2.0

    cx_n = min(max(cx / w, 0.0), 1.0)
    cy_n = min(max(cy / h, 0.0), 1.0)
    bw_n = min(max(bw / w, 0.0), 1.0)
    bh_n = min(max(bh / h, 0.0), 1.0)

    return cx_n, cy_n, bw_n, bh_n


def mask_to_tight_xyxy(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def topk_after_nms(keep_idx, scores_tensor, k):
    """
    Explicitly sort keep indices by descending score and take top-k.
    """
    if keep_idx.numel() == 0:
        return keep_idx
    order = scores_tensor[keep_idx].argsort(descending=True)
    keep_idx = keep_idx[order]
    return keep_idx[:k]


# ----------------------------
# MAIN
# ----------------------------
def autolabel_folder(image_dir, label_out_dir, gdino_cfg, gdino_ckpt, sam_ckpt):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Input image_dir does not exist: {image_dir.resolve()}")

    label_out_dir = Path(label_out_dir)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GroundingDINO...")
    gdino = load_model(str(gdino_cfg), str(gdino_ckpt)).to(DEVICE).eval()

    print("Loading SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(DEVICE).eval()
    sam_predictor = SamPredictor(sam)

    # Speed niceties
    torch.backends.cudnn.benchmark = True

    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    total_images = len(images)
    print(f"[INFO] Found {total_images} images in {image_dir}")

    # Progress stats
    start_time = time.time()
    ladder_img_count = 0
    person_img_count = 0
    any_label_img_count = 0
    ladder_box_total = 0
    person_box_total = 0

    with torch.inference_mode():
        for i, img_path in enumerate(tqdm(images, desc=f"Auto-label {image_dir.name}")):
            # DINO load (image_source is RGB ndarray; image is model-ready)
            image_source, image = load_image(str(img_path))
            h, w = image_source.shape[:2]

            # SAM setup
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                # corrupted image etc.
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(rgb)

            yolo_lines = []

            # track per-image stats
            ladder_in_img = False
            person_in_img = False
            ladder_boxes_in_img = 0
            person_boxes_in_img = 0

            for class_id, cfg in CLASSES.items():
                prompt = cfg["prompt"]
                max_det = cfg["max_det"]

                boxes, logits, _phrases = predict(
                    model=gdino,
                    image=image,
                    caption=prompt,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    device=DEVICE,
                )

                if len(boxes) == 0:
                    continue

                # Convert to clamped, valid xyxy pixel boxes + scores
                boxes_xyxy = []
                scores = []

                for b, score in zip(boxes, logits):
                    cx, cy, bw, bh = b.tolist()
                    x0 = (cx - bw / 2) * w
                    y0 = (cy - bh / 2) * h
                    x1 = (cx + bw / 2) * w
                    y1 = (cy + bh / 2) * h

                    fixed = clamp_and_fix_xyxy([x0, y0, x1, y1], w, h)
                    if fixed is None:
                        continue

                    boxes_xyxy.append(fixed)
                    scores.append(float(score))

                if len(boxes_xyxy) == 0:
                    continue

                boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32, device=DEVICE)
                scores_tensor = torch.tensor(scores, dtype=torch.float32, device=DEVICE)

                # Pre-SAM NMS
                keep_idx = nms(boxes_tensor, scores_tensor, NMS_IOU_PRE)
                keep_idx = topk_after_nms(keep_idx, scores_tensor, max_det)

                # SAM refine each kept box -> produce tight boxes + carry score
                refined_boxes = []
                refined_scores = []

                for idx in keep_idx:
                    box = boxes_tensor[idx].detach().cpu().numpy().astype(np.float32)

                    # SAM refine
                    masks, mask_scores, _ = sam_predictor.predict(
                        box=box,
                        multimask_output=True,
                    )
                    best_mask = masks[int(np.argmax(mask_scores))].astype(np.uint8)

                    tight = mask_to_tight_xyxy(best_mask)
                    if tight is None:
                        tight = box.tolist()

                    tight_fixed = clamp_and_fix_xyxy(tight, w, h)
                    if tight_fixed is None:
                        continue

                    refined_boxes.append(tight_fixed)
                    refined_scores.append(float(scores_tensor[idx].item()))

                if len(refined_boxes) == 0:
                    continue

                # Post-SAM NMS
                ref_boxes_t = torch.tensor(refined_boxes, dtype=torch.float32, device=DEVICE)
                ref_scores_t = torch.tensor(refined_scores, dtype=torch.float32, device=DEVICE)

                keep2 = nms(ref_boxes_t, ref_scores_t, NMS_IOU_POST)
                keep2 = topk_after_nms(keep2, ref_scores_t, max_det)

                # Write YOLO lines
                for j in keep2:
                    bx = ref_boxes_t[j].detach().cpu().numpy().tolist()
                    cx_n, cy_n, bw_n, bh_n = xyxy_to_yolo_norm(bx, w, h)
                    yolo_lines.append(f"{class_id} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}")

                    if class_id == 0:
                        ladder_in_img = True
                        ladder_boxes_in_img += 1
                    elif class_id == 1:
                        person_in_img = True
                        person_boxes_in_img += 1

            # Always write a label file (can be empty)
            label_file = label_out_dir / f"{img_path.stem}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

            # Update stats
            if ladder_in_img:
                ladder_img_count += 1
            if person_in_img:
                person_img_count += 1
            if len(yolo_lines) > 0:
                any_label_img_count += 1

            ladder_box_total += ladder_boxes_in_img
            person_box_total += person_boxes_in_img

            # Granular progress logging
            if (i + 1) % PRINT_EVERY == 0 or (i + 1) == total_images:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (total_images - (i + 1))

                print("\n================ PROGRESS ================")
                print(f"Images: {i+1}/{total_images}")
                print(f"Avg time/img: {avg_time:.2f}s | Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m")
                print(f"Ladder imgs: {ladder_img_count} | Person imgs: {person_img_count} | Any label imgs: {any_label_img_count}")
                if ladder_img_count > 0:
                    print(f"Avg ladder boxes/img (when present): {ladder_box_total / max(ladder_img_count,1):.2f}")
                if person_img_count > 0:
                    print(f"Avg person boxes/img (when present): {person_box_total / max(person_img_count,1):.2f}")
                print("==========================================\n")


if __name__ == "__main__":
    # base = Path("dataset_split_v1")
    base = Path("dataset_split_v2_gpu")

    gdino_cfg = Path("weights") / "GroundingDINO_SwinT_OGC.cfg.py"
    gdino_ckpt = Path("weights") / "groundingdino_swint_ogc.pth"
    sam_ckpt = Path("weights") / "sam_vit_h_4b8939.pth"

    # -------------------------
    # TRAIN
    # -------------------------
    print("\n===== AUTO-LABEL TRAIN =====")
    images_train = base / "images" / "train"
    labels_train = base / "labels_auto" / "train"
    autolabel_folder(images_train, labels_train, gdino_cfg, gdino_ckpt, sam_ckpt)

    # -------------------------
    # VAL
    # -------------------------
    print("\n===== AUTO-LABEL VAL =====")
    images_val = base / "images" / "val"
    labels_val = base / "labels_auto" / "val"
    autolabel_folder(images_val, labels_val, gdino_cfg, gdino_ckpt, sam_ckpt)

    # -------------------------
    # TEST
    # -------------------------
    print("\n===== AUTO-LABEL TEST =====")
    images_test = base / "images" / "test"
    labels_test = base / "labels_auto" / "test"
    autolabel_folder(images_test, labels_test, gdino_cfg, gdino_ckpt, sam_ckpt)