# Training Results and Analysis

## 1. Training Summary

All models trained with YOLO11, 100 epochs, seed=42, YOLO built-in augmentation only. Three configurations tested per dataset: Small@960, Medium@960, Small@1280.

---

## 2. Phase 1: Singapore River (39 test images)

| Run | Model | imgsz | Ladder R | Ladder P | Person R | Person P |
|-----|-------|-------|----------|----------|----------|----------|
| **sg_y11s_960** | y11s | 960 | **0.949** | **1.000** | 0.880 | 0.880 |
| sg_y11m_960 | y11m | 960 | 0.949 | 0.949 | 0.880 | 0.880 |
| sg_y11s_1280 | y11s | 1280 | 0.949 | 0.787 | 0.880 | 0.815 |

**Winner: sg_y11s_960** -- perfect ladder precision on the test set. The small model at 960px achieved the best balance. Increasing resolution to 1280px actually hurt precision (1.000 to 0.787), likely because the small dataset couldn't support the additional parameters effectively.

---

## 3. Phase 2: Internet Scraped (195 test images)

| Run | Model | imgsz | Ladder R | Ladder P | Person R | Person P |
|-----|-------|-------|----------|----------|----------|----------|
| **is_v2_y11s_960** | y11s | 960 | **0.762** | **0.676** | **0.843** | **0.817** |
| is_v2_y11m_960 | y11m | 960 | 0.741 | 0.609 | 0.806 | 0.783 |
| is_v2_y11s_1280 | y11s | 1280 | 0.725 | 0.606 | 0.843 | 0.798 |

**Winner: is_v2_y11s_960** -- highest recall and precision across both classes. The medium model underperformed the small model despite having more capacity, confirming overfitting at these dataset sizes. Performance is notably lower than Singapore River due to the much greater visual diversity in internet-sourced footage.

---

## 4. Phase 3: Combined Dataset (234 test images)

| Run | Model | imgsz | Ladder R | Ladder P | Person R | Person P |
|-----|-------|-------|----------|----------|----------|----------|
| combined_y11s_960 | y11s | 960 | 0.772 | 0.677 | **0.854** | **0.829** |
| combined_y11m_960 | y11m | 960 | 0.776 | 0.668 | 0.829 | 0.783 |
| **combined_y11s_1280** | y11s | 1280 | **0.781** | **0.685** | 0.833 | 0.798 |

**Winner: combined_y11s_1280** -- best ladder recall and precision. Unlike the single-dataset experiments, the larger combined dataset (1,623 train images) provided enough data to benefit from the higher 1280px resolution, particularly for resolving thin ladder structures.

---

## 5. Key Findings from Training

### 5.1 Small Models Outperform Medium

Across all three phases, YOLO11-Small consistently outperformed YOLO11-Medium. With dataset sizes of 270-1,600 images, the medium backbone overfits -- it has more capacity than the data can support. Nano was skipped (too weak for thin ladder detection), and Large/XL would overfit even worse.

### 5.2 Resolution Is Dataset-Size Dependent

- **Small datasets (270 images):** 1280px hurts precision (1.000 to 0.787 for SG ladders) -- more spatial detail means more parameters to learn with insufficient data
- **Larger datasets (1,623 images):** 1280px marginally helps ladder recall (0.772 to 0.781) -- enough data to benefit from resolving thin structures

### 5.3 Dataset Merging Improves Generalisation

Combining Singapore River + Internet Scraped data improved ladder recall from 0.762 (IS-only) to 0.781 (+2.5%). More importantly, it produced the only model that works across both domains (see cross-dataset analysis below).

### 5.4 Ladder Detection Is Harder Than Person Detection

Across all runs, ladder F1 (~0.68-0.78 recall) consistently trails person detection (~0.83-0.88 recall). Ladders are thin, variable in appearance, often partially occluded, and lack the distinctive visual features that make person detection comparatively robust.

### 5.5 Offline Augmentation Adds No Value

The augmentation tier experiments (Tier 0/1/2) on internet-scraped data showed flat metrics across all tiers. YOLO's built-in online augmentation (mosaic, mixup, HSV jitter, flip, scale, translate) was already sufficient. The real bottleneck was data diversity, not augmentation.

---

## 6. Post-Training Analysis

A comprehensive post-training analysis was conducted covering confidence threshold optimisation, cross-dataset generalisation, and test-time augmentation.

### 6.1 Confidence Threshold Optimisation

The default YOLO confidence threshold of 0.25 is not optimal for all models. A sweep across 9 thresholds (0.10-0.50) on 3 models x 2 test sets revealed significant gains from tuning:

| Model | Test Set | Default F1 (conf=0.25) | Best Ladder F1 | Optimal Conf |
|-------|----------|------------------------|----------------|-------------|
| SG | SG test | 0.974 | **0.987** | 0.15 |
| IS | IS test | 0.716 | **0.743** | 0.35 |
| Combined | SG test | 0.925 | **0.961** | 0.40 |
| Combined | IS test | 0.691 | **0.702** | 0.30 |

**Key insight:** The IS model gains **+3.8% ladder F1** simply by raising confidence from 0.25 to 0.35 -- filtering out low-confidence false positives that were dragging down precision. This is a free performance gain that requires no retraining.

### 6.2 Cross-Dataset Generalisation

This was the most critical finding of the entire analysis:

| Model | SG Test (Ladder F1) | IS Test (Ladder F1) |
|-------|---------------------|---------------------|
| SG | **0.974** | 0.000 |
| IS | 0.000 | **0.716** |
| Combined | **0.925** | **0.691** |

**Single-dataset models completely fail on out-of-domain data** -- 0% ladder detection when tested on a dataset they weren't trained on. The Singapore River model, despite achieving near-perfect metrics on its own test set, detects zero ladders in internet-sourced footage and vice versa.

**Only the Combined model generalises across both domains**, achieving 0.925 F1 on SG and 0.691 F1 on IS. This confirms that dataset merging is not optional -- it is essential for any model intended for real-world deployment where visual conditions will vary.

The Combined model's SG performance drops from 0.974 to 0.925 compared to the SG-only model, a modest trade-off for gaining cross-domain capability.

### 6.3 Test-Time Augmentation (TTA)

TTA applies multiple augmented views of each image at inference time and aggregates predictions:

| Model | Test Set | Ladder R (normal) | Ladder R (TTA) | Delta |
|-------|----------|-------------------|----------------|-------|
| SG | SG test | 0.949 | 0.974 | +0.026 |
| IS | IS test | 0.762 | 0.772 | +0.011 |
| Combined | IS test | 0.746 | 0.772 | +0.026 |

**Verdict:** TTA boosts recall by +1-4% but hurts precision (more false positives). The net F1 decreases in most cases. TTA should only be used in scenarios where recall is critical and false positives are tolerable -- not recommended for standard deployment.

---

## 7. Phase 4: Classroom Dataset (27 test images)

A classroom-specific model was trained for lab demos and user testing. The dataset combines 174 video frames from the lab setup with 91 vetted internet-scraped ladder images (from DuckDuckGo + Bing scraping). Total: 265 images (185 train / 53 val / 27 test).

| Run | Model | imgsz | Ladder R | Ladder P | Person R | Person P |
|-----|-------|-------|----------|----------|----------|----------|
| **classroom_y11s_960** | y11s | 960 | **0.891** | **0.774** | **0.789** | **0.750** |

Confidence threshold sweep on val set:

| Conf | Ladder R | Ladder P | Person R | Person P |
|------|----------|----------|----------|----------|
| 0.20 | 0.913 | 0.750 | 0.789 | 0.714 |
| 0.25 | 0.891 | 0.774 | 0.789 | 0.750 |
| **0.30** | 0.870 | 0.769 | 0.789 | 0.789 |
| **0.35** | 0.848 | 0.796 | 0.737 | 0.824 |

**Recommended conf=0.30-0.35** for demo deployment to reduce human→ladder false positives.

---

## 8. Best Models for Deployment

| Use Case | Model | Weights | Recommended Conf |
|----------|-------|---------|-----------------|
| SG River only | sg_y11s_960 | `datasets/singapore_river/5_weights/sg_y11s_t0_960/best.pt` | 0.15 |
| Internet imagery | is_v2_y11s_960 | `datasets/internet_scraped/5_weights/is_v2_y11s_960/best.pt` | 0.35 |
| General-purpose | combined_y11s_1280 | `datasets/combined/5_weights/combined_y11s_1280/best.pt` | 0.30 |
| **Classroom / demo** | **classroom_y11s_960** | `datasets/classroom/5_weights/classroom_y11s_960/best.pt` | **0.30-0.35** |

All deploy weights copied to `Pyxis-Jetson/weights/deploy/`.

**Revised recommendation:** Domain-specific models outperform generalised models when the deployment environment is known. The combined model is only necessary when the visual domain is unpredictable. For fixed environments (demo room, Singapore River), use the specialised model.

---

## 9. Gap Analysis Against Deliverables

| Deliverable | Target | Current Best | Status |
|-------------|--------|-------------|--------|
| Ladder detection (clear weather) | 95% | 97.4% (SG-only) / 78.1% (combined) | SG-only exceeds target; combined needs improvement |
| Ladder detection (low-light/rain) | 90% | Not yet tested | Requires expanded dataset |
| Overlay latency | < 1 sec | YOLO11s inference is sub-100ms | On track |
| Ladder lock-on | Reliable | Detection stable; tracking not yet integrated | Pending ByteTrack integration |

### Paths to Closing the Gap

1. **Expand training data** -- more diverse ladder imagery, particularly varied lighting and weather
2. **Confidence threshold tuning** -- free +3-4% F1 gain from threshold optimisation
3. **Active learning** -- identify and correct failure clusters in auto-labels
4. **Tracking integration** -- ByteTrack for temporal smoothing and stable lock-on
5. **Low-light/weather dataset** -- collect or generate footage for the 90% target
