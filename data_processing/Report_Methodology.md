# Methodology

## 1. Overview

The Computer Vision (CV) subsystem was developed to address the safety and operational risks associated with maritime pilot transfers as defined in the project scope . The methodology was designed to satisfy the measurable system deliverables, particularly:

* 95% ladder detection success rate under clear conditions
* 90% detection accuracy under low-light or light rain
* Sub-second (<1s) visual overlay latency
* Reliable ladder “lock-on” capability during transfer operations 

The methodology was structured around four core principles:

1. Reproducibility
2. Evaluation integrity
3. Iterative robustness improvement
4. Alignment with deployment constraints

The full pipeline consists of:

1. Dataset preparation and structured splitting
2. Automated label generation using foundation models
3. Targeted human correction for edge-case robustness
4. YOLO-based fine-tuning and validation
5. Integration considerations for real-time deployment

---

## 2. Dataset Preparation

### 2.1 Data Sources

The dataset was constructed from:

* Still images of pilot transfer scenarios
* Video recordings of pilot transfers under varying maritime conditions

As defined in the project scope, the proof-of-concept baseline scenario was limited to:

> Docked, daytime, calm sea state (WMO Sea State 0–3) 

This constraint allowed controlled model development before extending to harsher conditions such as low visibility or rough seas.

---

### 2.2 Frame Extraction Strategy

Initial video extraction at 10 FPS produced excessive redundancy (~10,000 frames), leading to:

* High temporal duplication
* Slower labeling and training cycles
* Increased overfitting risk

To improve generalisation and iteration speed, frame extraction was reduced to:

> 2 FPS

This preserved scene diversity while eliminating near-identical frames. The resulting dataset (~1,900 images) provided sufficient variation without introducing excessive noise.

---

### 2.3 Dataset Splitting

To ensure defensible evaluation, the dataset was split into:

* 70% training
* 20% validation
* 10% test

A fixed random seed was used to ensure reproducibility.

The test set was strictly isolated and reserved for final performance reporting to maintain evaluation integrity relative to project detection requirements .

---

## 3. Class Design Strategy

The final object classes were restricted to:

* `pilot_ladder`
* `person` (pilot or crew)

This decision was made based on system-level safety objectives:

* Ladder detection is required for guidance and lock-on.
* Person detection is required for operational safety awareness.

Although `ship_hull` was initially considered, it was excluded in Phase 1 because:

* Hull boxes are large and geometrically inconsistent.
* They dilute gradient learning for thin ladder features.
* They are not directly required to satisfy ladder detection KPIs.

This class reduction improved model convergence and minimized labeling noise.

---

## 4. Automated Label Generation

Manual labeling at scale was not feasible given timeline constraints. Therefore, a hybrid automated pipeline was implemented using foundation models.

### 4.1 Proposal Generation (GroundingDINO)

GroundingDINO was used for text-guided object proposal generation.

Prompts used:

* “pilot ladder. rope ladder.”
* “person. pilot. crew member.”

GroundingDINO outputs:

* Bounding box proposals
* Confidence logits

This stage provides coarse object localization.

---

### 4.2 Per-Class Non-Maximum Suppression (Pre-Refinement)

To eliminate duplicate detections:

* Per-class IoU threshold = 0.5
* DINO logits used as confidence scores
* Explicit descending sort applied before limiting detections

Cross-class overlap was preserved, as overlap between ladder and person is expected in real-world scenes.

---

### 4.3 Segmentation-Based Refinement (SAM ViT-H)

The Segment Anything Model (SAM), specifically the ViT-H variant, was used to refine bounding boxes.

For each retained proposal:

1. SAM generates a segmentation mask.
2. A tight bounding box is computed from the mask.
3. The refined box replaces the coarse proposal.

This improves:

* Boundary precision
* Thin-object localization (critical for ladders)
* Training signal quality

SAM ViT-H was selected due to availability of high-performance GPU hardware and the need for maximal mask fidelity.

---

### 4.4 Post-Refinement NMS

A second per-class NMS pass was applied after SAM refinement to remove residual duplicates created during mask tightening.

This two-stage NMS design (pre- and post-SAM) ensures:

* Geometric validity
* Reduced redundancy
* Cleaner YOLO training labels

---

### 4.5 Label Integrity Controls

To maintain dataset cleanliness, the following safeguards were enforced:

* Bounding box clamping to image boundaries
* Enforced coordinate ordering (x0 < x1, y0 < y1)
* Tiny box filtering
* Maximum detections per class (ladder: 2, person: 5)

These controls reduce label instability and prevent degenerate training targets.

---

## 5. Human-in-the-Loop Refinement

Although automated labeling achieved strong baseline accuracy, errors were concentrated in:

* Motion blur
* Occlusion
* Glare
* Low-contrast scenarios

Rather than performing full manual relabeling, a targeted refinement strategy was adopted:

1. Train initial YOLO baseline.
2. Identify failure clusters.
3. Manually correct 30–100 high-impact edge cases.
4. Retrain model.

This approach maximizes robustness while preserving development velocity.

---

## 6. Model Training and Validation

The refined dataset was used to fine-tune a YOLOv11 detector.

Training configuration emphasized:

* High input resolution (to capture thin ladder geometry)
* Balanced class representation
* Augmentations aligned with observed failure modes (blur, brightness variation)

Validation performance was monitored using the 20% validation split.

Final performance metrics were computed exclusively on a human-corrected test set to preserve evaluation integrity relative to required KPIs .

---

## 7. Deployment Considerations

The CV pipeline was designed with real-time deployment constraints in mind:

* Sub-second inference requirement 
* Stable detection across 0.3–5m range
* Integration with LIDAR proximity system
* Compatibility with dashboard overlay system

Hardware utilized:

* NVIDIA RTX 5080
* CUDA 12.8
* PyTorch 2.10

This ensures that both auto-labeling and model training are scalable and that final inference performance can meet latency requirements.

---

## 8. Alignment with Project Deliverables

The methodology directly supports the stated deliverables  by:

* Structuring evaluation for measurable detection accuracy
* Isolating a gold test set for defensible performance claims
* Prioritizing ladder lock-on detection reliability
* Enabling future tracking integration
* Maintaining compatibility with proximity and guidance modules

The resulting CV system is engineered not only for detection accuracy, but also for operational robustness and maritime safety alignment.