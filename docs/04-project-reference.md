# Project Reference

## 1. Environment Setup

### 1.1 Python Environment

```bash
py -3.10 -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```

### 1.2 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.9.0 | Deep learning framework |
| torchvision | 0.24.0 | Vision utilities |
| ultralytics | 8.3.217 | YOLO11 training and inference |
| opencv-python | 4.12.0.88 | Image processing |
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.16.2 | Scientific computing |
| polars | 1.34.0 | Data handling (metrics CSVs) |
| PyYAML | 6.0.3 | YOLO dataset config generation |
| tqdm | 4.67.1 | Progress bars |
| matplotlib | 3.10.7 | Training curves and plots |
| clip | (git) | Ultralytics CLIP fork |

Additional packages installed via git (not in requirements.txt):
- **GroundingDINO:** `pip install git+https://github.com/IDEA-Research/GroundingDINO.git`
- **Segment Anything:** `pip install git+https://github.com/facebookresearch/segment-anything.git`
- **Albumentations:** Used for offline augmentation (deprecated in current pipeline)

### 1.3 Foundation Model Weights

Weights are stored in `data_processing/weights/` and are not tracked in git due to size.

**SAM ViT-H** (used for auto-labelling):
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" `
  -OutFile "data_processing\weights\sam_vit_h_4b8939.pth"
```
Expected size: ~2.4 GB

**SAM ViT-B** (lighter alternative, not used in final pipeline):
```powershell
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
  -OutFile "data_processing\weights\sam_vit_b_01ec64.pth"
```

**GroundingDINO SwinT-OGC:**
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/pengxian/grounding-dino/resolve/main/groundingdino_swint_ogc.pth" `
  -OutFile "data_processing\weights\groundingdino_swint_ogc.pth"
```

The GroundingDINO config file (`GroundingDINO_SwinT_OGC.cfg.py`) is also required and stored alongside the weights.

### 1.4 Hardware Used

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 5080 |
| CUDA | 12.8 |
| RAM | 128 GB |
| Python | 3.10 |

---

## 2. Complete Directory Structure

```
Pyxis-Capstone/
│
├── docs/                               # Documentation (source of truth)
│   ├── 01-methodology.md               # Full pipeline methodology
│   ├── 02-training-and-results.md      # All training results and analysis
│   ├── 03-design-decisions.md          # Design decisions, challenges, lessons
│   └── 04-project-reference.md         # This file: setup, structure, reference
│
├── datasets/                           # All datasets with numbered pipelines
│   ├── singapore_river/                # Field footage dataset
│   │   ├── raw/                        # 7 videos + 11 photos (not in git)
│   │   ├── 1_frames_3fps/             # 386 extracted frames (not in git)
│   │   ├── 2_frames_3fps_split/       # Train/val/test split with auto-labels (not in git)
│   │   ├── 4_augmented/               # Augmentation tiers (deprecated)
│   │   ├── gold/                       # Human-corrected labels (labels in git, images not)
│   │   ├── 5_runs/                     # YOLO training outputs (not in git)
│   │   ├── 5_weights/                  # Best checkpoints per run (not in git)
│   │   ├── 1_dataset_gen.py           # Step 1: extract frames at 3 FPS
│   │   ├── 2_split.py                  # Step 2: random train/val/test split
│   │   ├── 3_autolabel.py             # Step 3: GroundingDINO + SAM auto-labelling
│   │   ├── 4_augment.py               # Step 4: offline augmentation (deprecated)
│   │   ├── 5_train.py                  # Step 5: train YOLO models
│   │   └── 6_evaluate.py              # Step 6: benchmark on gold test
│   │
│   ├── internet_scraped/               # Internet-sourced dataset
│   │   ├── raw/                        # Videos + images from web (not in git)
│   │   │   ├── pilot_transfer_pictures/
│   │   │   ├── pilot_transfer_videos/
│   │   │   └── broken_pilot_ladder/
│   │   ├── 1_frames_2fps/             # 1,934 extracted frames (not in git)
│   │   ├── 2_frames_2fps_split/       # Train/val/test split with auto-labels (not in git)
│   │   ├── 4_augmented/               # Augmentation tiers t0/t1/t2 (deprecated)
│   │   ├── gold/                       # Human-corrected labels (labels in git, images not)
│   │   ├── 5_runs/                     # YOLO training outputs (not in git)
│   │   ├── 5_weights/                  # Best checkpoints per run (not in git)
│   │   ├── 7_prompts/                  # JSON prompt batch definitions
│   │   ├── 7_prompt_results/           # Prompt optimisation CSVs
│   │   ├── 1_dataset_gen.py           # Step 1: extract frames at 2 FPS
│   │   ├── 2_split.py                  # Step 2: random train/val/test split
│   │   ├── 3_autolabel.py             # Step 3: auto-labelling
│   │   ├── 4_augment.py               # Step 4: offline augmentation (deprecated)
│   │   ├── 5_train_v2.py              # Step 5: train (v2, 100 epochs)
│   │   ├── 6_evaluate_v2.py           # Step 6: benchmark on gold test
│   │   ├── 7_prompt_optimisation.py   # Step 7: benchmark DINO prompts
│   │   └── experiments_v2.csv          # Training metrics log
│   │
│   ├── combined/                       # Merged dataset
│   │   ├── gold/                       # Merged labels with sg_/is_ prefixes
│   │   ├── 5_runs/                     # Training outputs
│   │   ├── 5_weights/                  # Best checkpoints
│   │   ├── 1_merge_gold.py            # Merge SG + IS gold with filename prefixes
│   │   ├── 5_train.py                  # Train on combined data
│   │   └── 6_evaluate.py              # Final benchmark
│   │
│   └── classroom/                      # Classroom/demo dataset (265 images)
│       ├── ladder_1.mp4               # Lab video with ladder
│       ├── ladder_2.mp4               # Lab video with ladder
│       ├── no_ladder.mp4              # Lab video without ladder (negatives)
│       ├── scraped/                   # Vetted internet-scraped ladder images
│       ├── raw_frames/                # Extracted frames + scraped (not in git)
│       ├── split/                     # Train/val/test with auto-labels (not in git)
│       ├── gold/                      # Human-corrected labels (labels in git)
│       ├── 5_runs/                    # Training outputs (not in git)
│       ├── 5_weights/                 # Best checkpoints (not in git)
│       ├── 0_scrape_images.py         # DuckDuckGo image scraper
│       ├── 0_scrape_bing.py           # Bing image scraper
│       ├── 1_dataset_gen.py           # Extract frames + merge scraped images
│       ├── 2_split.py                 # Train/val/test split
│       ├── 3_autolabel.py             # GroundingDINO + SAM auto-labelling
│       └── 5_train.py                 # Train classroom_y11s_960
│
├── data_processing/                    # Shared auto-labelling module
│   ├── autolabel_grounded_sam_multiclass.py  # Full GroundingDINO + SAM pipeline
│   ├── process.py                             # Random dataset splitter
│   └── weights/                               # Foundation model checkpoints (not in git)
│       ├── sam_vit_h_4b8939.pth              # SAM ViT-H (~2.4 GB)
│       ├── groundingdino_swint_ogc.pth       # GroundingDINO SwinT
│       └── GroundingDINO_SwinT_OGC.cfg.py    # GroundingDINO config
│
├── dataset_gen/                        # Shared frame extraction module
│   └── process_data.py                # Video-to-frames converter, image copier
│
├── model_finetuning/                   # Shared training/evaluation module
│   ├── train.py                       # YOLO training harness
│   ├── evaluate.py                    # Gold test evaluation (IoU matching)
│   ├── augment.py                     # Offline augmentation (deprecated)
│   └── yolo11s.pt, yolo11m.pt        # YOLO11 base weights
│
├── _Google_finetuning/                # Post-training analysis
│   ├── run_analysis.py               # Confidence sweep, cross-dataset eval, TTA
│   ├── analysis_results.json         # Full analysis output
│   ├── gold_test/                    # Test sets copied for analysis
│   │   ├── singapore_river/          # SG gold test images + labels
│   │   └── internet_scraped/         # IS gold test images + labels
│   └── weights/                       # Best model weights copied for analysis
│       ├── sg_y11s_t0_960.pt
│       ├── is_v2_y11s_960.pt
│       └── combined_y11s_1280.pt
│
├── legacy/                            # Archived experiments
│   ├── autolabel_images/             # Early auto-label outputs
│   ├── dataset_split_v1/             # Old split format
│   └── yolo_weights/                 # Old checkpoints
│
├── run_all.py                         # Master orchestrator (all 3 phases)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Excludes weights, videos, frames, venv
└── venv310/                           # Python 3.10 virtual environment (not in git)
```

---

## 3. Git Tracking Policy

| Content Type | Tracked in Git | Reason |
|-------------|---------------|--------|
| Source code (.py) | Yes | Core pipeline logic |
| Documentation (.md) | Yes | Project documentation |
| Config files (.yaml, .json, .cfg.py) | Yes | Experiment reproducibility |
| Prompt definitions (7_prompts/*.json) | Yes | Prompt optimisation inputs |
| Prompt results CSVs | Yes | Optimisation outputs |
| Experiment metrics CSVs | Yes | Training metrics |
| Gold labels (.txt) | Yes | Irreplaceable human-corrected annotations |
| Raw videos/images | **No** | Too large; source footage |
| Extracted frames | **No** | Regenerable from raw via Step 1 |
| Auto-generated labels | **No** | Regenerable via Step 3 |
| Augmented datasets | **No** | Regenerable via Step 4 (deprecated) |
| Training run outputs | **No** | Regenerable via Step 5 |
| Model weights (.pt) | **No** | Too large; regenerable via training |
| Foundation model weights | **No** | Downloaded from official sources |
| Virtual environment | **No** | Regenerable from requirements.txt |

**Key principle:** Gold labels are the only large artefact tracked in git because they represent irreplaceable human correction work. Everything else is either regenerable or too large.

---

## 4. Pipeline Regeneration Commands

### Full Pipeline (all phases)

```bash
cd Pyxis-Capstone
python run_all.py    # Runs phases 1, 2, 3 sequentially (~2 hours)
```

### Per-Dataset Pipeline

Each dataset follows the same numbered steps. Replace `<dataset>` with `singapore_river`, `internet_scraped`, or `combined`.

```bash
cd datasets/<dataset>

# Step 1: Extract frames from raw videos
python 1_dataset_gen.py

# Step 2: Random train/val/test split (70/20/10)
python 2_split.py

# Step 3: Auto-label with GroundingDINO + SAM
python 3_autolabel.py

# (Manual: correct labels in LabelImg, copy to gold/)

# Step 5: Train YOLO models (3 configs)
python 5_train.py         # or 5_train_v2.py for internet_scraped

# Step 6: Evaluate on gold test set
python 6_evaluate.py      # or 6_evaluate_v2.py for internet_scraped
```

### Combined Dataset (Phase 3)

```bash
cd datasets/combined
python 1_merge_gold.py    # Merge SG + IS gold labels with prefixes
python 5_train.py         # Train on merged data
python 6_evaluate.py      # Final benchmark
```

### Classroom Dataset (Phase 4)

```bash
cd datasets/classroom

# Step 0: Scrape internet images (optional, already vetted in scraped/)
python 0_scrape_images.py --max 30      # DuckDuckGo
python 0_scrape_bing.py --max 40        # Bing (for remaining queries)

# Step 1: Extract video frames + merge scraped images
python 1_dataset_gen.py

# Step 2: Split 70/20/10
python 2_split.py

# Step 3: Auto-label
python 3_autolabel.py

# (Manual: correct labels in LabelImg, copy to gold/)

# Step 5: Train
python 5_train.py
```

### Post-Training Analysis

```bash
cd _Google_finetuning
python run_analysis.py    # Confidence sweep, cross-dataset eval, TTA
```

### Prompt Optimisation (internet_scraped only)

```bash
cd datasets/internet_scraped
python 7_prompt_optimisation.py --batch batch_1.json   # Single batch
python 7_prompt_optimisation.py --all                   # All batches
```

---

## 5. Shared Modules Reference

### dataset_gen/process_data.py

| Function | Description |
|----------|-------------|
| `video_to_frames(video_path, output_dir, fps)` | Extract frames at configurable FPS with continuous numbering |
| `process_dataset(raw_dir, output_dir, fps)` | Unified pipeline for mixed videos + images |
| `delete_files_by_extension(dir, ext)` | Clean output directories |
| `replace_spaces_in_filenames(dir)` | Sanitise filenames |

### data_processing/process.py

| Function | Description |
|----------|-------------|
| `random_split_dataset(src, dst, ratios, seed)` | Train/val/test split with seed-based reproducibility |

### data_processing/autolabel_grounded_sam_multiclass.py

| Function | Description |
|----------|-------------|
| `autolabel_folder(image_dir, label_dir, ...)` | Full GroundingDINO + SAM pipeline with dual NMS |

Key parameters:
- `box_threshold`: 0.35
- `text_threshold`: 0.30
- `nms_iou`: 0.5
- `max_detections`: ladder=2, person=5
- Prompts: `"pilot ladder. rope ladder."`, `"person. pilot. crew member."`

### model_finetuning/train.py

| Function | Description |
|----------|-------------|
| `train_one(config)` | Single experiment harness (trains one YOLO model) |
| `eval_gold_val(model, data_yaml, config)` | Evaluate on gold validation set |
| `run_wave(experiments, csv_path)` | Batch training + CSV logging |

Generates per-run:
- `5_runs/<run_name>/` -- full Ultralytics output (logs, curves, checkpoints)
- `5_runs/<run_name>/exp_config.json` -- experiment config snapshot
- `5_runs/<run_name>/gold_val_metrics.json` -- validation metrics
- `5_weights/<run_name>/best.pt` -- best checkpoint copy

### model_finetuning/evaluate.py

| Function | Description |
|----------|-------------|
| `evaluate_split(model, images_dir, labels_dir, conf, iou)` | Test-set benchmark with greedy IoU matching |

Outputs per-class: precision, recall, average confidence, box area statistics (mean, p50, p90).

### model_finetuning/augment.py (deprecated)

| Function | Description |
|----------|-------------|
| `copy_train(src, dst)` | Tier 0: copy without augmentation |
| `augment_train(src, dst, transforms)` | Tier 1/2: apply Albumentations transforms |

Deprecated: YOLO's built-in online augmentation proved sufficient. Retained for reference.

---

## 6. Output File Formats

### YOLO Label Format (.txt)

One line per detection, normalised to [0, 1]:
```
class_id  centre_x  centre_y  width  height
```

### Experiment Config (exp_config.json)

```json
{
  "run_name": "combined_y11s_1280",
  "model": "yolo11s.pt",
  "imgsz": 1280,
  "epochs": 100,
  "batch": 16,
  "close_mosaic": 10,
  "conf_eval": 0.25,
  "match_iou": 0.5,
  "workers": 4
}
```

### Gold Validation Metrics (gold_val_metrics.json)

```json
{
  "ladder_precision": 0.613,
  "ladder_recall": 0.683,
  "person_precision": 0.797,
  "person_recall": 0.855,
  "ladder_avg_pred_conf": 0.687,
  "person_avg_pred_conf": 0.777,
  "ladder_area_mean": 0.098,
  "ladder_area_p50": 0.059,
  "ladder_area_p90": 0.234,
  "n_images": 463
}
```

### Dataset YAML (auto-generated for YOLO)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
names:
  0: pilot_ladder
  1: person
```

---

## 7. Auto-Labelling Detection Statistics

### Internet Scraped (1,353 train images, GPU)

| Metric | Value |
|--------|-------|
| Processing speed | 0.77 sec/image, ~1.2 images/sec |
| Total runtime | 17.3 minutes |
| Ladder detected in | 1,054 / 1,353 images (78%) |
| Person detected in | 1,252 / 1,353 images (93%) |
| Any label present | 1,340 / 1,353 images (99%) |
| Avg ladder boxes (when present) | 1.10 |
| Avg person boxes (when present) | 2.08 |

### Internet Scraped (386 val images, GPU)

| Metric | Value |
|--------|-------|
| Processing speed | 0.80 sec/image |
| Total runtime | 5.1 minutes |
| Ladder detected in | 310 / 386 images (80%) |
| Person detected in | 362 / 386 images (94%) |
| Avg ladder boxes (when present) | 1.10 |
| Avg person boxes (when present) | 2.09 |

### Internet Scraped (195 test images, GPU)

| Metric | Value |
|--------|-------|
| Processing speed | 0.80 sec/image |
| Total runtime | 2.6 minutes |
| Ladder detected in | 155 / 195 images (79%) |
| Person detected in | 183 / 195 images (94%) |
| Avg ladder boxes (when present) | 1.12 |
| Avg person boxes (when present) | 2.02 |

These statistics confirm stable detection rates aligned with operational expectations: ~1 ladder per scene, ~2 persons, and minimal background-only frames.
