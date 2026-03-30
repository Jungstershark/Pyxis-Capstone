"""
Step 1: Extract frames from raw videos (2 FPS) and copy images
into a single flat directory with continuous numbering.

Usage:
    cd Pyxis-Capstone/datasets/internet_scraped
    python 1_dataset_gen.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "dataset_gen"))

from process_data import process_dataset, delete_files_by_extension, replace_spaces_in_filenames

RAW_PICTURES = Path(__file__).resolve().parent / "raw" / "pilot_transfer_pictures"
RAW_VIDEOS = Path(__file__).resolve().parent / "raw" / "pilot_transfer_videos"
OUTPUT_DIR = Path(__file__).resolve().parent / "1_frames_2fps"
FPS = 2

if __name__ == "__main__":
    print(f"Raw pictures: {RAW_PICTURES}")
    print(f"Raw videos:   {RAW_VIDEOS}")
    print(f"Output dir:   {OUTPUT_DIR}")
    print(f"FPS:          {FPS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean previous run (if any)
    delete_files_by_extension(str(OUTPUT_DIR), [".jpg", ".png", ".jpeg"])

    # Sanitize filenames (spaces -> underscores)
    replace_spaces_in_filenames(str(RAW_PICTURES), recursive=True)
    replace_spaces_in_filenames(str(RAW_VIDEOS), recursive=True)

    # Extract frames from videos + copy images with continuous numbering
    last_counter = process_dataset(str(RAW_PICTURES), str(OUTPUT_DIR), frames_per_second=FPS, return_counter=True)
    process_dataset(str(RAW_VIDEOS), str(OUTPUT_DIR), start_count=last_counter + 1, frames_per_second=FPS)

    print(f"\nDone. Frames saved to: {OUTPUT_DIR}")
