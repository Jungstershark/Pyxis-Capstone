# Design Decisions, Challenges, and Constraints

## 1. Architectural Decisions

### 1.1 Two-Class Detection System

**Decision:** Restrict to `pilot_ladder` (class 0) and `person` (class 1) only.

**Alternatives considered:** A three-class system including `ship_hull` was initially explored.

**Why we chose this:**
- The primary safety KPI is 95% ladder detection -- hull detection is not required for this
- Hull bounding boxes are large and geometrically variable across vessels, introducing gradient dilution during training
- Removing hull as a class reduced label noise and let the model focus its capacity on the safety-critical targets
- Hull detection can be introduced in a later phase if navigation stabilisation requires contextual awareness

**Impact:** Faster convergence, cleaner training signal, and simpler evaluation focused on what matters.

---

### 1.2 Foundation Model Auto-Labelling (GroundingDINO + SAM)

**Decision:** Use GroundingDINO for proposal generation and SAM ViT-H for segmentation refinement, rather than manual annotation.

**Why we chose this:**
- Manual labelling of ~2,300 images across two datasets was impractical within project timelines
- GroundingDINO's zero-shot text-guided detection enabled rapid iteration -- prompts could be tuned without retraining a detector
- SAM's mask-based refinement was critical for pilot ladders, which are thin structures where coarse bounding boxes introduce significant localisation noise
- The combined pipeline achieved F1=0.675 for ladder and F1=0.896 for person on auto-labels alone, providing a strong starting point for human correction

**Trade-off:** Auto-labels require human review and correction (the gold label step), but this is far faster than labelling from scratch.

---

### 1.3 SAM ViT-H Over Lighter Variants

**Decision:** Use SAM ViT-H (2.4 GB) rather than ViT-B (375 MB) or ViT-L (1.2 GB).

**Why we chose this:**
- Pilot ladders are thin, high-aspect-ratio objects where boundary precision directly affects training quality
- The RTX 5080 with 128 GB RAM could handle ViT-H without latency issues (~0.77 sec/image)
- ViT-H produces the highest-fidelity segmentation masks, critical for extracting tight bounding boxes from thin structures

**Trade-off:** Slower than ViT-B (~3x), but the GPU made this negligible at pipeline scale (17 min for 1,353 images).

---

### 1.4 YOLO11-Small Over Medium/Large

**Decision:** Use YOLO11-Small as the primary architecture across all experiments.

**Evidence driving this decision:**
- Small consistently outperformed Medium across all three dataset phases
- With dataset sizes of 270-1,623 images, Medium's additional capacity led to overfitting
- Nano was too weak for thin ladder detection; Large/XL would overfit even worse
- Small also has lower inference latency, important for the <1s deployment target on Jetson hardware

---

### 1.5 Dual NMS (Pre-SAM and Post-SAM)

**Decision:** Apply per-class Non-Maximum Suppression twice -- once before SAM refinement and once after.

**Why both passes are needed:**
- **Pre-SAM NMS:** Removes duplicate GroundingDINO proposals before expensive SAM processing (reduces compute and prevents redundant masks)
- **Post-SAM NMS:** SAM tightening can cause previously non-overlapping boxes to converge, creating new duplicates that didn't exist before refinement

Skipping either pass produced duplicate labels in the training data, which degraded model performance.

---

### 1.6 Dataset Merging Over Single-Dataset Training

**Decision:** Merge Singapore River and Internet Scraped datasets into a combined training set.

**Evidence driving this decision:**
- Cross-dataset evaluation showed single-dataset models achieve 0% ladder detection on out-of-domain data
- SG-only model: 0.974 F1 on SG test, **0.000** on IS test
- IS-only model: 0.000 on SG test, 0.716 on IS test
- Combined model: 0.925 on SG test, 0.691 on IS test

**This was the most important finding of the project** -- without merging, no single model could generalise, making single-dataset models useless for real deployment.

**Trade-off:** The combined model's SG performance drops from 0.974 to 0.925, a modest sacrifice for gaining cross-domain capability.

---

### 1.7 Abandoning Offline Augmentation

**Decision:** Drop the offline augmentation tier system (t0/t1/t2) in favour of YOLO's built-in online augmentation only.

**Why we initially tried offline augmentation:**
- Hypothesis that lighting robustness (brightness/contrast/gamma) and motion robustness (blur/dropout) would improve detection under varied conditions
- Tiers allowed controlled A/B comparison

**Why we abandoned it:**
- All three tiers produced flat metrics -- no meaningful improvement over the unaugmented baseline
- YOLO already applies mosaic, mixup, HSV jitter, flip, scale, and translate during training, which subsumes the offline transforms
- Offline augmentation added disk I/O overhead without benefit

**Lesson learned:** The real bottleneck was data diversity (more varied scenes), not data volume or augmentation.

---

### 1.8 Domain-Specific Models Over Generalised Models

**Decision:** Deploy domain-specific models per environment rather than one generalised combined model.

**Evidence driving this decision:**
- The combined model was tested on the Jetson edge device and produced excessive human→ladder false positives in the classroom/lab environment
- Root cause: SG data (cobblestone walls), IS data (open sea), and lab data (indoor, chairs, projected images) are three fundamentally different visual domains
- When forced to learn all domains, the model learns shallow features (anything vaguely vertical + human-shaped = ladder) instead of domain-specific discriminative features
- A domain-specific classroom model trained on 265 images (lab footage + vetted internet images) achieved 89.1% ladder recall with lower false positive rates

**Revised strategy:**
- `sg_y11s_960` for Singapore River deployment
- `is_v2_y11s_960` for internet-style imagery
- `classroom_y11s_960` for lab demos and user testing
- `combined_y11s_1280` retained as a research baseline only

**Lesson learned:** For fixed deployment environments, specialisation beats generalisation. The combined model is only appropriate when the deployment domain is unpredictable.

---

## 2. Challenges Faced

### 2.1 Frame Redundancy at High Extraction Rates

**Problem:** Initial video frame extraction at 10 FPS produced ~10,000 frames with extreme temporal redundancy -- consecutive frames were near-identical.

**Impact:** Training on redundant data increased overfitting risk, slowed iteration, and inflated labelling effort without improving model performance.

**Resolution:** Reduced extraction to 2-3 FPS (dataset-dependent), cutting dataset size by 80% while preserving scene diversity. This dramatically improved iteration speed and model generalisation.

---

### 2.2 Auto-Label Quality for Thin Objects

**Problem:** GroundingDINO's coarse bounding boxes were particularly inaccurate for pilot ladders -- a thin, high-aspect-ratio object. Boxes often included significant background area.

**Impact:** Training on loose bounding boxes teaches the model imprecise localisation, reducing both precision and the usefulness of detections for lock-on guidance.

**Resolution:** SAM ViT-H segmentation refinement. By generating a pixel-level mask and extracting a tight bounding box from it, boundary accuracy improved significantly for ladder detections. This required dual NMS to handle the new duplicate patterns SAM refinement introduced.

---

### 2.3 Prompt Sensitivity in GroundingDINO

**Problem:** GroundingDINO's detection quality varies dramatically with prompt wording. Descriptive prompts like `"pilot boarding ladder on ship"` achieved only F1=0.286, while concise prompts like `"pilot ladder. rope ladder."` achieved F1=0.675.

**Impact:** Suboptimal prompts would have produced low-quality auto-labels, cascading into poor model training.

**Resolution:** Systematic prompt optimisation study testing 30 prompt variants across 2 batches. Key finding: 2 concise synonyms is the optimal structure. Longer descriptions confuse the model's text-vision alignment.

---

### 2.4 Albumentations Bounding Box Validation

**Problem:** The Albumentations library enforces strict bounding box validation. Floating-point precision errors from augmentation transforms (e.g. coordinates like -4.99e-07) triggered validation failures that crashed the pipeline.

**Impact:** Initially caused silent training data loss or pipeline crashes during augmentation.

**Resolution:** Implemented pre- and post-augmentation sanitisation:
- YOLO-to-XYXY conversion with edge clamping to [0,1]
- Degenerate box removal
- Class-specific minimum area filtering (ladder: 0.00003, person: 0.00010)
- RGB/BGR colour space correction for Albumentations compatibility

---

### 2.5 Cross-Domain Generalisation Failure

**Problem:** Models trained on a single dataset completely failed when tested on the other dataset (0% ladder detection).

**Impact:** A model trained only on Singapore River footage -- despite achieving 97.4% F1 -- was useless for any other visual context.

**Resolution:** Dataset merging with filename prefixing (`sg_`/`is_`) to avoid collisions. The combined model generalises across both domains, confirming that visual diversity in training data is non-negotiable for deployment.

---

### 2.6 Ladder Detection Difficulty vs Person Detection

**Problem:** Across all experiments, ladder detection consistently underperformed person detection (F1 ~0.68-0.78 vs ~0.83-0.88).

**Root causes:**
- Ladders are thin, high-aspect-ratio objects with variable appearance (wood, rope, metal)
- Ladders are often partially occluded by the ship hull or personnel
- Ladders lack the distinctive visual features (faces, limbs, clothing) that make person detection robust
- Fewer ladder instances per image (typically 1) vs multiple persons

**Mitigation (partial):** Higher resolution (1280px) marginally improved ladder recall on the combined dataset. Confidence threshold tuning provided a free +3-4% F1 gain. Full resolution requires more diverse ladder training data.

---

## 3. Constraints

### 3.1 Dataset Size

With 270-1,623 training images, the datasets are small by deep learning standards. This constrained:
- Model architecture choice (Small over Medium/Large to avoid overfitting)
- Resolution experiments (1280px only beneficial with the larger combined dataset)
- The upper bound on achievable metrics without additional data collection

### 3.2 Proof-of-Concept Scope

The project scope was limited to docked, daytime, calm sea state (WMO 0-3). This means:
- Low-light and adverse weather performance has not been validated
- The 90% low-light/rain target requires future data collection
- Results should be interpreted within this environmental envelope

### 3.3 Hardware Dependency

The pipeline was developed on an RTX 5080 with 128 GB RAM. This enabled SAM ViT-H and high-resolution training but:
- Deployment target is Jetson edge hardware with different performance characteristics
- ONNX/TensorRT export and Jetson-specific latency benchmarking are pending

### 3.4 Single-Frame Detection Only

The current system performs per-frame detection without temporal context:
- No tracking across frames (ByteTrack integration pending)
- No temporal smoothing for lock-on stability
- Detection jitter across consecutive frames is not yet addressed

---

## 4. Lessons Learned

1. **Data diversity beats data volume and augmentation.** Combining two visually distinct datasets produced the only deployable model. No amount of augmentation could compensate for training on a single visual domain.

2. **Smaller models win on small datasets.** The consistent superiority of YOLO11-Small over Medium was counterintuitive but clear across all experiments. Model capacity must match dataset size.

3. **Auto-labelling quality is a first-class concern.** The GroundingDINO + SAM pipeline's quality directly determined training quality. Investing in prompt optimisation and dual NMS paid dividends through cleaner labels.

4. **Confidence threshold tuning is free performance.** A simple threshold sweep (no retraining) yielded +3-4% F1 improvement. This should be standard practice for any deployed model.

5. **Test-time augmentation has limited value.** TTA boosted recall but hurt precision and net F1. It is not a general-purpose improvement.

6. **Evaluation integrity is non-negotiable.** Maintaining strictly isolated, human-corrected gold test sets ensured all reported metrics are defensible. Auto-labels for training, gold labels for evaluation.

7. **Domain-specific models outperform generalised ones for fixed deployments.** The combined model generalises but underperforms in practice due to domain confusion (human→ladder false positives). When the deployment environment is known, a specialised model is always the better choice.

8. **Internet image scraping is viable for augmenting small datasets.** Combining 174 lab video frames with 91 vetted scraped images produced a 265-image dataset sufficient for 89.1% ladder recall -- comparable to the 270-image Singapore River dataset.
