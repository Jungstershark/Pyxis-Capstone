"""
Step 3: Auto-label train/val/test splits using GroundingDINO + SAM.

Reads:
    split/images/{train, val, test}

Writes:
    split/labels_auto/{train, val, test}

Classes:
    0: pilot_ladder
    1: person

Usage:
    cd Pyxis-Capstone/datasets/classroom
    python 3_autolabel.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from autolabel_grounded_sam_multiclass import autolabel_folder

DATASET_DIR = Path(__file__).resolve().parent / "split"
WEIGHTS_DIR = REPO_ROOT / "data_processing" / "weights"

GDINO_CFG = WEIGHTS_DIR / "GroundingDINO_SwinT_OGC.cfg.py"
GDINO_CKPT = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
SAM_CKPT = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        print(f"\n===== AUTO-LABEL {split.upper()} =====")
        images_dir = DATASET_DIR / "images" / split
        labels_dir = DATASET_DIR / "labels_auto" / split

        autolabel_folder(images_dir, labels_dir, GDINO_CFG, GDINO_CKPT, SAM_CKPT)

    print("\nDone. Labels written to split/labels_auto/{train,val,test}")
    print("Next: manually correct labels in LabelImg, then copy to gold/")
