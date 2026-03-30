# Pyxis-Capstone

Computer vision pipeline for maritime pilot transfer safety — detecting pilot ladders and personnel using YOLO models.

## Project Structure

```
Pyxis-Capstone/
├── datasets/                       # All datasets, each with a numbered pipeline
│   ├── internet_scraped/           # Pilot ladder dataset (internet-sourced)
│   └── singapore_river/            # Singapore River field dataset
│
├── data_processing/                # Shared: frame splitting, GroundingDINO+SAM auto-labeling
├── dataset_gen/                    # Shared: video-to-frames extraction
├── model_finetuning/               # Shared: augmentation, training, evaluation, metrics
│
├── legacy/                         # Archived / superseded files
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
| 4 | `4_augment.py` | Create augmentation tiers (t0/t1/t2) from gold |
| 5 | `5_train.py` | Train YOLO models, evaluate on gold val |
| 6 | `6_evaluate.py` | Final benchmark on gold test set |

## Classes

```
0: pilot_ladder
1: person
```

## Setup

```bash
py -3.10 -m venv venv310
venv310\Scripts\activate
pip install -r requirements.txt
```
