"""
Step 2: Split extracted frames into train/val/test sets (70/20/10).

Creates:
    2_frames_2fps_split/
        images/{train, val, test}
        labels_auto/{train, val, test}   <-- empty, populated by step 3 (autolabel)

Usage:
    cd Pyxis-Capstone/datasets/internet_scraped
    python 2_split.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data_processing"))

from process import random_split_dataset

SOURCE_DIR = Path(__file__).resolve().parent / "1_frames_2fps"
OUTPUT_DIR = Path(__file__).resolve().parent / "2_frames_2fps_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SEED = 42

if __name__ == "__main__":
    print(f"Source:  {SOURCE_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Split:   {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

    random_split_dataset(
        source_dir=str(SOURCE_DIR),
        output_dir=str(OUTPUT_DIR),
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    # Pre-create labels_auto directories for step 3
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "labels_auto" / split).mkdir(parents=True, exist_ok=True)

    print(f"\nDone. labels_auto/{{train,val,test}} directories ready for step 3.")
