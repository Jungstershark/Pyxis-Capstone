"""
Step 1: Merge gold labels from internet_scraped + singapore_river into combined dataset.

Copies images and labels from both datasets, prefixing filenames to avoid collisions:
    internet_scraped: is_frame_00000.jpg / is_frame_00000.txt
    singapore_river:  sg_frame_00000.jpg / sg_frame_00000.txt

Output:
    gold/images/{train, val, test}
    gold/labels/{train, val, test}

Usage:
    cd Pyxis-Capstone/datasets/combined
    python 1_merge_gold.py
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent

SOURCES = {
    "is": DATASETS_DIR / "internet_scraped" / "gold",
    "sg": DATASETS_DIR / "singapore_river" / "gold",
}

OUTPUT = SCRIPT_DIR / "gold"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def merge():
    total = {"images": 0, "labels": 0}

    for split in SPLITS:
        out_img = OUTPUT / "images" / split
        out_lbl = OUTPUT / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        split_imgs = 0
        split_lbls = 0

        for prefix, gold_dir in SOURCES.items():
            src_img = gold_dir / "images" / split
            src_lbl = gold_dir / "labels" / split

            if not src_img.exists():
                print(f"[SKIP] {src_img} does not exist")
                continue

            for img_path in sorted(src_img.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                # Prefix filename to avoid collisions
                new_name = f"{prefix}_{img_path.name}"
                shutil.copy2(img_path, out_img / new_name)
                split_imgs += 1

                # Copy matching label
                label_path = src_lbl / (img_path.stem + ".txt")
                new_label = f"{prefix}_{img_path.stem}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, out_lbl / new_label)
                    split_lbls += 1
                else:
                    # Create empty label (background image)
                    (out_lbl / new_label).write_text("", encoding="utf-8")
                    split_lbls += 1

        total["images"] += split_imgs
        total["labels"] += split_lbls
        print(f"  {split}: {split_imgs} images, {split_lbls} labels")

    print(f"\nTotal: {total['images']} images, {total['labels']} labels")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    # Clean previous merge
    for split in SPLITS:
        for sub in ["images", "labels"]:
            d = OUTPUT / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    print("Merging gold datasets...")
    for prefix, path in SOURCES.items():
        print(f"  {prefix}: {path}")
    print()

    merge()
    print("\nDone.")
