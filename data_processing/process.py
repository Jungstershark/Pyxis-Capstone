import os
import shutil
import random
from pathlib import Path


def random_split_dataset(
    source_dir,
    output_dir,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
    seed=42
):
    """
    Randomly split images into train / val / test folders.

    Parameters
    ----------
    source_dir : str
        Directory containing all processed images.

    output_dir : str
        Directory where split dataset will be created.

    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.

    seed : int
        Random seed for reproducibility.
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    image_exts = {".jpg", ".jpeg", ".png"}

    # Gather all images
    all_images = [p for p in source_dir.iterdir() if p.suffix.lower() in image_exts]

    total = len(all_images)
    print(f"Total images found: {total}")

    random.seed(seed)
    random.shuffle(all_images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_images = all_images[:train_end]
    val_images = all_images[train_end:val_end]
    test_images = all_images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # Create folder structure
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)

    # Copy files
    for split, files in splits.items():
        for file_path in files:
            dst = output_dir / "images" / split / file_path.name
            shutil.copy(file_path, dst)

    print("\nSplit Summary:")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")
    print(f"\nDataset created at: {output_dir}")


if __name__ == "__main__":
    source = "processed_data_22022026-2fps"
    destination = "dataset_split_v2_gpu"

    random_split_dataset(source, destination)