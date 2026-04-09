# Methodology

## 1. Overview

The Computer Vision (CV) subsystem was developed to address the safety and operational risks associated with maritime pilot transfers. The methodology was designed to satisfy measurable system deliverables:

- **95% ladder detection success rate** under clear conditions
- **90% detection accuracy** under low-light or light rain
- **Sub-second (<1s)** visual overlay latency
- Reliable ladder **"lock-on"** capability during transfer operations

The methodology was structured around four core principles:

1. **Reproducibility** -- fixed seeds, deterministic splits, config snapshots
2. **Evaluation integrity** -- gold human-corrected test sets, strict train/test isolation
3. **Iterative robustness improvement** -- auto-label, correct, retrain cycles
4. **Alignment with deployment constraints** -- real-time inference on edge hardware

---

## 2. Dataset Preparation

### 2.1 Data Sources

Two complementary datasets were constructed:

| Dataset | Source | Raw Material | Condition Focus |
|---------|--------|-------------|-----------------|
| **Singapore River** | Field footage from Singapore River pilot ops | 7 videos + 11 photos | Docked, daytime, calm (WMO Sea State 0-3) |
| **Internet Scraped** | Diverse web-sourced pilot transfer footage | Videos + images from various maritime contexts | Mixed conditions, vessels, and angles |

The proof-of-concept baseline was limited to *docked, daytime, calm sea state* as defined in the project scope, establishing a stable baseline before extending to harsher conditions.

### 2.2 Frame Extraction Strategy

Videos were converted to individual frames at controlled extraction rates:

| Dataset | Initial Rate | Final Rate | Frames Produced | Rationale |
|---------|-------------|------------|-----------------|-----------|
| Singapore River | -- | 3 FPS | 386 | Docked scenarios with minimal motion; 3 FPS captures sufficient variation |
| Internet Scraped | 10 FPS | 2 FPS | 1,934 | 10 FPS produced ~10,000 near-identical frames with high redundancy and overfitting risk |

The reduction from 10 FPS to 2 FPS for internet-scraped footage was a deliberate decision that eliminated temporal redundancy while preserving scene diversity, reducing the dataset to a manageable ~1,900 images without sacrificing training quality.

### 2.3 Dataset Splitting

All datasets were split using the same protocol:

- **70% training** / **20% validation** / **10% test**
- Fixed random seed for reproducibility
- Test set strictly isolated and reserved for final evaluation only

Final splits:

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| Singapore River | 270 | 77 | 39 |
| Internet Scraped | 1,353 | 386 | 195 |
| Combined (merged) | 1,623 | 463 | 234 |

---

## 3. Class Design

The final object classes were restricted to:

| Class ID | Label | Purpose |
|----------|-------|---------|
| 0 | `pilot_ladder` | Guidance and lock-on detection |
| 1 | `person` | Operational safety awareness |

**Why `ship_hull` was excluded:**

- Hull bounding boxes are large and geometrically inconsistent across vessels
- They dilute gradient learning for thin ladder features during training
- Hull detection is not directly required to satisfy the 95% ladder detection KPI
- Removing this class reduced labelling noise and accelerated convergence on safety-critical metrics

Hull detection may be introduced in later phases if navigation stabilisation requires it.

---

## 4. Automated Label Generation (Grounded SAM Pipeline)

Manual labelling at scale was impractical given timeline constraints. A hybrid automated pipeline was implemented using two foundation models in sequence.

### 4.1 Pipeline Architecture

```
Raw Image
    |
    v
[GroundingDINO] -- text-guided proposals --> Coarse bounding boxes + logits
    |
    v
[Pre-SAM NMS] -- per-class IoU=0.5 --> Deduplicated proposals
    |
    v
[SAM ViT-H] -- segmentation refinement --> Tight bounding boxes from masks
    |
    v
[Post-SAM NMS] -- remove residual duplicates --> Clean labels
    |
    v
[Integrity Checks] -- clamp, filter, validate --> Final YOLO-format labels
```

### 4.2 Proposal Generation (GroundingDINO)

GroundingDINO performs zero-shot, text-guided object detection. The prompts used were determined through systematic prompt optimisation (see Section 6):

- **Ladder:** `"pilot ladder. rope ladder."`
- **Person:** `"person. pilot. crew member."`
- **Box threshold:** 0.35 | **Text threshold:** 0.30

GroundingDINO was selected for its strong zero-shot detection capability, enabling fast iteration without needing class-specific training data for the labelling stage itself.

### 4.3 Per-Class Non-Maximum Suppression (Pre-SAM)

Before SAM refinement, duplicate detections within each class were removed:

- Per-class IoU threshold: 0.5
- GroundingDINO logits used as confidence scores
- Explicit descending confidence sort enforced

Cross-class overlap was preserved -- ladder-person overlap is expected and correct in real-world pilot transfer scenes.

### 4.4 Segmentation Refinement (SAM ViT-H)

The Segment Anything Model (ViT-H variant) refines each coarse bounding box:

1. SAM generates a segmentation mask from the proposed box
2. A tight bounding box is computed from the mask contour
3. The refined box replaces the coarse proposal

This step was critical for **thin-object boundary accuracy** -- pilot ladders are narrow structures where coarse boxes introduce significant localisation noise. SAM ViT-H (the largest variant) was selected because the available RTX 5080 hardware could handle it without latency bottlenecks.

### 4.5 Post-SAM NMS

A second per-class NMS pass removes residual duplicates created when SAM tightens overlapping boxes into similar regions. This dual-NMS design (pre- and post-SAM) ensures clean, non-redundant labels.

### 4.6 Label Integrity Controls

Additional safeguards enforce geometric validity:

- **Box clamping** to image boundaries (no out-of-bounds coordinates)
- **Coordinate ordering** enforced (x0 < x1, y0 < y1)
- **Tiny box filtering** removes degenerate detections
- **Max detections per class:** ladder: 2, person: 5

These controls prevent downstream training instability from malformed labels.

### 4.7 Processing Performance

| Execution Mode | Per-Image Time | Total (1,353 images) | Speedup |
|---------------|---------------|----------------------|---------|
| CPU | ~15-17 sec | ~5+ hours | -- |
| GPU (RTX 5080, CUDA 12.8) | **0.77 sec** | **17.3 min** | ~20x |

GPU acceleration made iterative auto-labelling workflows practical within development timelines.

---

## 5. Human-in-the-Loop Correction (Gold Labels)

Auto-generated labels were reviewed and corrected using LabelImg to create **gold standard** label sets:

- **Training and validation labels:** corrected to ensure clean training signal
- **Test set labels:** meticulously corrected -- these are the ground truth for all reported metrics

The gold labels are the irreplaceable source of truth for evaluation. Corrected labels were stored in dedicated `gold/` directories per dataset.

Error patterns that required manual correction were concentrated in:
- Motion blur frames
- Partial occlusion scenarios
- Glare and low-contrast conditions
- Unusual ladder angles or partially visible ladders

---

## 6. Prompt Optimisation

A systematic prompt optimisation study was conducted on the internet-scraped dataset to determine the best GroundingDINO text prompts for auto-labelling.

### Methodology

- **Batch 1:** 15 single-term and descriptive prompts per class, evaluated against gold validation labels
- **Batch 2:** 15 multi-term combinations of top performers from Batch 1

### Key Findings

**Ladder -- Top performers:**

| Prompt | Recall | Precision | F1 |
|--------|--------|-----------|-----|
| `"pilot ladder. rope ladder."` | 0.650 | 0.703 | **0.675** |
| `"boarding ladder. ladder."` | 0.569 | 0.767 | 0.653 |
| `"rope ladder. ladder."` | 0.612 | 0.699 | 0.652 |

**Person -- Top performers:**

| Prompt | Recall | Precision | F1 |
|--------|--------|-----------|-----|
| `"person. pilot. crew member."` | 0.885 | 0.907 | **0.896** |
| `"person. human. pilot."` | 0.879 | 0.909 | 0.894 |
| `"person. human."` | 0.890 | 0.889 | 0.890 |

**Conclusions:**
- **2 concise synonyms is the sweet spot** -- adding 3+ terms degrades performance
- Descriptive/long prompts perform significantly worse (e.g. `"pilot boarding ladder on ship."` F1=0.286)
- Person detection is inherently easier than ladder detection (F1 0.896 vs 0.675)
- The baseline prompts were confirmed as optimal and used for all subsequent auto-labelling

---

## 7. Model Training

### 7.1 Training Framework

All models were trained using Ultralytics YOLO11 with the following configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Seed | 42 (fixed for reproducibility) |
| Close mosaic | 10 (disable mosaic for last 10 epochs) |
| Augmentation | YOLO built-in only (mosaic, mixup, HSV, flip, scale, translate) |
| Confidence threshold (eval) | 0.25 |
| IoU matching threshold | 0.5 |

### 7.2 Why Offline Augmentation Was Abandoned

An initial augmentation strategy using offline tiers was implemented:

- **Tier 0:** Baseline (no augmentation)
- **Tier 1:** Lighting robustness (brightness, contrast, gamma via Albumentations)
- **Tier 2:** Motion robustness (motion blur, gaussian blur, dropout)

After experimentation on the internet-scraped dataset, all three tiers produced **flat metrics** -- no meaningful improvement over Tier 0. This confirmed that YOLO's built-in online augmentation (applied dynamically per epoch) was already sufficient, and offline augmentation was adding disk overhead without benefit.

**The real bottleneck was data diversity, not data volume or augmentation.**

### 7.3 Model Configurations

Three configurations were tested per dataset to answer specific questions:

| Config | Model | Resolution | Question Answered |
|--------|-------|-----------|-------------------|
| y11s @ 960 | YOLO11-Small | 960px | Baseline performance |
| y11m @ 960 | YOLO11-Medium | 960px | Does a larger backbone help with small datasets? |
| y11s @ 1280 | YOLO11-Small | 1280px | Does higher resolution help thin ladder detection? |

### 7.4 Training Phases

Training was executed in three sequential phases:

1. **Phase 1 -- Singapore River:** 3 runs on field footage (270 training images)
2. **Phase 2 -- Internet Scraped:** 3 runs on diverse web footage (1,353 training images)
3. **Phase 3 -- Combined:** 3 runs on merged dataset (1,623 training images)

The entire pipeline was orchestrated by `run_all.py`, which runs all phases sequentially with per-step logging, 4-hour timeouts, and fault tolerance.

---

## 8. Evaluation Protocol

### 8.1 Metrics

All metrics are computed on **gold (human-corrected) test sets only**:

- **Primary:** Ladder recall and precision (safety-critical)
- **Secondary:** Person recall and precision
- **Diagnostics:** Average prediction confidence, bounding box area statistics (mean, p50, p90)

### 8.2 Matching Logic

- Predictions sorted by confidence (descending)
- Greedy IoU matching with threshold >= 0.5
- Per-class true positive / false positive / false negative counts
- Per-image confidence averaging

### 8.3 Model Selection Criteria

1. **Ladder recall** (primary -- safety-critical, must not miss ladders)
2. **Ladder precision** (tie-breaker)
3. Test set never used for model selection -- only for final reporting

---

## 9. Dataset Merging Strategy

To test whether combining datasets improves generalisation, gold labels from both datasets were merged:

- Internet-scraped files prefixed with `is_` (e.g. `is_frame_00001.jpg`)
- Singapore River files prefixed with `sg_` (e.g. `sg_frame_00001.jpg`)
- This prevented filename collisions while maintaining traceability

The combined dataset (1,623 train / 463 val / 234 test) was used for Phase 3 training.

---

## 10. Hardware and Infrastructure

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 5080 |
| CUDA | 12.8 |
| RAM | 128 GB |
| PyTorch | 2.9.0 |
| Ultralytics | 8.3.21 |
| Python | 3.10 |

Auto-labelling throughput: ~1.2-1.3 images/sec (GPU). Training: ~30 min per run (100 epochs).
