"""
Step 1: Extract frames from raw videos (1 FPS) and copy images
into a single flat directory with continuous numbering.

Usage:
    cd Pyxis-Capstone/datasets/singapore_river
    python 1_dataset_gen.py
"""

import sys
from pathlib import Path

# Add dataset_gen to path so we can import the shared utilities
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dataset_gen"))

from process_data import process_dataset, delete_files_by_extension, replace_spaces_in_filenames

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "1_frames_3fps"
FPS = 3

if __name__ == "__main__":
    print(f"Raw input:  {RAW_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"FPS:        {FPS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous run (if any)
    delete_files_by_extension(str(OUTPUT_DIR), [".jpg", ".png", ".jpeg"])

    # Sanitize filenames (spaces -> underscores)
    replace_spaces_in_filenames(str(RAW_DIR), recursive=False)

    # Extract frames from videos + copy images with continuous numbering
    process_dataset(str(RAW_DIR), str(OUTPUT_DIR), frames_per_second=FPS)

    print(f"\nDone. Frames saved to: {OUTPUT_DIR}")
