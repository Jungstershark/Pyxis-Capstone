# Model Fine-Tuning & Augmentation Pipeline

## Maritime Pilot Transfer Computer Vision System

---

## 1. Project Context

This repository contains the **data augmentation, training, and evaluation pipeline** for the Computer Vision (CV) subsystem of the Maritime Pilot Transfer safety system.

The CV model supports the following critical deliverables :

* ≥95% ladder detection success rate in clear weather
* ≥90% detection under low-light/light rain
* <1 second inference + overlay latency
* Stable ladder “lock-on” capability
* Robust operation under docked and parallel cruising states

The focus of this directory is:

> Controlled fine-tuning of YOLO-based detectors with rigorous experiment hygiene and safety-oriented KPI tracking.

---

# 2. Dataset Structure

All experiments assume the following corrected dataset:

```
dataset_split_v2_gpu_corrected/
  images/
    train/
    val/   (GOLD)
    test/  (GOLD)
  labels/
    train/
    val/   (GOLD)
    test/  (GOLD)
```

### Class Definitions

| Class ID | Label        |
| -------- | ------------ |
| 0        | pilot_ladder |
| 1        | person       |

Labels follow YOLO format:

```
class_id cx cy w h   (normalized)
```

---

# 3. Augmentation Strategy

Augmentation is implemented in `augment.py`.

We use **offline augmentation tiers**, not cumulative duplication.

Each tier produces:

```
dataset_split_v2_gpu_corrected_t0/
dataset_split_v2_gpu_corrected_t1/
dataset_split_v2_gpu_corrected_t2/
```

Only the **train split** is modified.
`val` and `test` remain untouched (gold standard).

---

## Tier Definitions

### Tier 0 — Baseline

* No augmentation
* Images copied without recompression
* Establishes ground-truth distribution

### Tier 1 — Lighting Robustness

* RandomBrightnessContrast
* RandomGamma

Purpose:

* Simulate glare, brightness variability, exposure instability
* Align with low-light / light rain deliverables 

### Tier 2 — Motion Robustness

* MotionBlur
* GaussianBlur
* RandomBrightnessContrast
* Controlled dropout (with min_visibility safeguard)

Purpose:

* Simulate vibration, motion blur during parallel cruising
* Improve robustness under sea-state variability

---

# 4. Technical Safeguards Implemented

The augmentation pipeline includes strict validation:

## 4.1 RGB/BGR Correction

* cv2 loads BGR
* Albumentations expects RGB-like input
* Conversion applied before/after augmentation

## 4.2 Bounding Box Edge Clamping (Critical)

Bounding boxes are sanitized via:

1. YOLO → XYXY conversion
2. Edge-based clamping to [0,1]
3. Degenerate box removal
4. Class-specific minimum area filtering
5. Re-conversion to YOLO format

This prevents:

* Negative coordinate crashes
* Out-of-bound IoU artifacts
* Training instability

## 4.3 Class-Specific Area Thresholds

Because ladders are thin and distant:

```
MIN_BOX_AREA_BY_CLASS = {
  0: 0.00003  # ladder
  1: 0.00010  # person
}
```

This prevents over-filtering of valid ladder instances.

## 4.4 Pre- and Post-Sanitize Strategy

Boxes are sanitized:

* Before augmentation (Albumentations strict check)
* After augmentation (transform drift correction)

## 4.5 Tier Summary Diagnostics

Each augmentation run logs:

* Train images written
* Empty label count
* Box count per class
* Ratio vs Tier 0

Example output:

```
Tier: 2
Train images written: 1353
Empty train label files: 14
Total boxes (class 0 ladder): 1162
Total boxes (class 1 person): 2600
Ladder ratio vs Tier0: 0.999
Person ratio vs Tier0: 0.997
```

This prevents silent label erosion.

---

# 5. Training Strategy

Training is performed via `train.py` using the Ultralytics Python API.

Key experiment hygiene:

* Fixed random seed (42)
* Deterministic run naming
* Per-run config snapshot saved
* Best checkpoint copied to structured folder
* Full gold-val evaluation saved as JSON

---

## Wave 1 — Backbone Selection

| Run | Model   | Tier | imgsz |
| --- | ------- | ---- | ----- |
| A1  | yolo11s | 0    | 960   |
| A2  | yolo11m | 0    | 960   |

Selection based on:

1. Ladder Recall (primary KPI)
2. Ladder Precision (tie-breaker)

---

## Wave 2 — Robustness

Applied only to Wave 1 winner:

| Run | Model  | Tier | imgsz            |
| --- | ------ | ---- | ---------------- |
| B1  | winner | 1    | 960              |
| B2  | winner | 2    | 960              |
| B3  | winner | 2    | 1280 (if needed) |

---

# 6. Evaluation Protocol

Evaluation is performed using `evaluate.py`.

Metrics computed on GOLD splits only.

## Primary KPIs

* Ladder Recall
* Ladder Precision
* Person Recall
* Person Precision

Matching logic:

* Predictions sorted by confidence (descending)
* Greedy IoU matching (IoU ≥ 0.5)
* Per-class TP/FP/FN

## Additional Safety Metrics

* Ladder average predicted confidence
* Ladder bounding box area distribution (mean, p50, p90)

This guards against:

* Overconfident false locks
* Bounding box inflation
* Thin-object detection failure

---

# 7. Experimental Integrity

Important design decisions:

### Training uses auto-labels for speed

### Evaluation uses corrected gold splits only

This preserves:

* Velocity
* Credibility
* Defensible reporting

Test set is never used for model selection.

---

# 8. Known Technical Challenges Encountered

1. Albumentations strict bounding box validation
2. Floating-point negative epsilon errors (e.g. -4.99e-07)
3. CoarseDropout API differences across versions
4. RGB/BGR mismatch during augmentation
5. Ensuring no label drift during blur transforms
6. Maintaining thin-object (ladder) integrity under filtering

All mitigated via defensive sanitation and structured tier diagnostics.

---

# 9. Future Extensions (Optional)

* Sequence-based stability evaluation (5-frame lock metric)
* Negative-only dataset for false-lock stress testing
* ONNX/TensorRT export for Jetson deployment
* Confidence threshold calibration sweep

---

# 10. Summary

The fine-tuning framework prioritizes:

* Ladder recall (safety-critical)
* Controlled robustness experimentation
* Clean separation of training vs evaluation data
* Reproducibility and experiment traceability

This structure ensures that reported detection rates meaningfully support the stated deliverables  while maintaining development velocity.


Absolutely — below is a **README-ready section** (≈300 words) that you can directly insert into your `model_finetuning/README.md`.

---

## Why We Do Not Duplicate Augmented Samples (Design Rationale)

A deliberate design decision in this project was **not to duplicate the dataset offline by storing both original and augmented copies of each training image**. While duplicating data (e.g., original + blurred + brightness-augmented variants) may appear to increase dataset size and diversity, this approach is not aligned with best practices for modern YOLO-based training pipelines.

Ultralytics YOLO already performs **strong online data augmentation during training**. This includes geometric transformations, color jitter, scaling, mosaic, and mixup. These augmentations are applied dynamically per epoch, meaning each image can appear in many different transformed forms throughout training. As a result, offline duplication provides limited additional benefit while introducing several risks.

First, duplicating images increases disk usage and I/O overhead without increasing true dataset diversity. Second, it can bias the training distribution by overweighting augmented examples relative to original samples. Third, it reduces stochastic variation across epochs, since duplicated augmented images remain static once written to disk. Modern training frameworks rely on *randomized, on-the-fly augmentation* to maintain distributional richness and prevent overfitting.

Instead, this project adopts a **controlled tier-based augmentation strategy**:

* Tier 0: Baseline dataset (no offline augmentation)
* Tier 1: Lighting robustness distribution
* Tier 2: Motion robustness distribution

Each tier modifies the training distribution intentionally, without inflating dataset size. This allows structured experimentation while preserving reproducibility and preventing dataset drift.

In summary, avoiding duplication ensures:

* Cleaner experimental comparisons
* Lower risk of distribution bias
* Efficient storage and faster I/O
* Better compatibility with YOLO’s built-in augmentation

This decision improves scientific rigor while maintaining development velocity.
