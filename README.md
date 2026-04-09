# Pyxis-Capstone

Computer vision pipeline for maritime pilot transfer safety -- detecting pilot ladders and personnel using YOLO models trained on real-world maritime footage.

## Deliverables

- 95% ladder detection success rate (clear weather)
- 90% detection success rate (low-light / light rain)
- < 1 second latency for dashboard visual overlay
- Reliable ladder "lock-on" capability

## Classes

```
0: pilot_ladder
1: person
```

## Project Structure

```
Pyxis-Capstone/
├── docs/                           # Detailed documentation (start here)
│   ├── 01-methodology.md           # Full pipeline methodology
│   ├── 02-training-and-results.md  # All training results and analysis
│   └── 03-design-decisions.md      # Design decisions, challenges, lessons learned
│
├── datasets/                       # All datasets, each with a numbered pipeline
│   ├── singapore_river/            # Field footage (7 videos + 11 photos, 386 frames)
│   ├── internet_scraped/           # Web-sourced footage (1,934 frames)
│   ├── combined/                   # Merged gold labels from both datasets
│   └── classroom/                  # Classroom/demo dataset (lab videos + scraped images)
│
├── data_processing/                # Shared: frame splitting, GroundingDINO+SAM auto-labeling
├── dataset_gen/                    # Shared: video-to-frames extraction
├── model_finetuning/               # Shared: training, evaluation, augmentation
├── _Google_finetuning/             # Post-training analysis (confidence tuning, cross-dataset eval, TTA)
│
├── legacy/                         # Archived / superseded experiments
├── run_all.py                      # Master orchestrator (all training phases)
└── requirements.txt
```

## Dataset Pipeline

Each dataset follows the same numbered pipeline:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_dataset_gen.py` | Extract frames from raw videos at target FPS |
| 2 | `2_split.py` | Random train/val/test split (70/20/10) |
| 3 | `3_autolabel.py` | Auto-label with GroundingDINO + SAM ViT-H |
| -- | *Manual* | Correct labels in LabelImg, copy to `gold/` |
| 5 | `5_train.py` | Train YOLO models, evaluate on gold val |
| 6 | `6_evaluate.py` | Final benchmark on gold test set |
| 7 | `7_prompt_optimisation.py` | Benchmark GroundingDINO text prompts against gold |

## Datasets

| Dataset | Source | Frames | Split (train/val/test) |
|---------|--------|--------|------------------------|
| Singapore River | Field footage, 3 FPS | 386 | 270 / 77 / 39 |
| Internet Scraped | Web sources, 2 FPS | 1,934 | 1,353 / 386 / 195 |
| Combined | Merged gold (sg_ + is_ prefixed) | 2,320 | 1,623 / 463 / 234 |
| Classroom | Lab videos + scraped internet images | 265 | 185 / 53 / 27 |

## Best Models

| Use Case | Model | Ladder R | Ladder P | Recommended Conf |
|----------|-------|----------|----------|-----------------|
| SG River deployment | sg_y11s_960 | 0.949 | 1.000 | 0.15 |
| Internet imagery | is_v2_y11s_960 | 0.762 | 0.676 | 0.35 |
| General-purpose | combined_y11s_1280 | 0.781 | 0.685 | 0.30 |
| **Classroom / demo** | **classroom_y11s_960** | **0.891** | **0.774** | **0.30-0.35** |

### Deployment Strategy

**Domain-specific models outperform generalised models** when the deployment environment is known. Cross-dataset analysis showed single-dataset models achieve 0% ladder detection on out-of-domain data, while the combined model generalises but with lower accuracy. For fixed environments (e.g. a specific demo room or deployment site), a specialised model is the correct choice.

Deploy weights: `Pyxis-Jetson/weights/deploy/`

## Key Findings

1. **Small (y11s) outperforms Medium (y11m)** -- medium overfits at these dataset sizes
2. **Domain-specific > generalised** -- specialised models outperform combined when deployment environment is known
3. **Single-dataset models fail cross-domain** -- 0% ladder detection on out-of-domain data
4. **Offline augmentation adds no value** -- YOLO's built-in augmentation is sufficient
5. **Confidence tuning gives free gains** -- +3-4% F1 from threshold optimisation alone
6. **Ladder detection is harder than person** -- thin, variable objects with fewer visual features
7. **1280px resolution not beneficial** -- marginal gains don't justify increased compute

## Documentation

For detailed documentation (presentation and report ready):

- **[Methodology](docs/01-methodology.md)** -- Data pipeline, auto-labelling, training protocol
- **[Training Results](docs/02-training-and-results.md)** -- All results, post-training analysis, gap analysis
- **[Design Decisions](docs/03-design-decisions.md)** -- Decisions, challenges, constraints, lessons learned
- **[Project Reference](docs/04-project-reference.md)** -- Setup, directory structure, regeneration commands, shared modules

## Setup

```bash
py -3.10 -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```

## Run All Training

```bash
python run_all.py
```
