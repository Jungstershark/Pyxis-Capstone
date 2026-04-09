"""
Extract frames from classroom videos at 2 FPS.
Also pulls in scraped images from 0_scrape_images.py output.
All frames go into raw_frames/ with source prefix.
"""
import cv2, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_VIDEOS = ROOT  # videos are in the classroom folder itself
OUT_DIR = ROOT / "raw_frames"
SCRAPED_DIR = ROOT / "scraped"
TARGET_FPS = 2

VIDEOS = ["ladder_1.mp4", "ladder_2.mp4", "no_ladder.mp4"]


def extract_video_frames():
    OUT_DIR.mkdir(exist_ok=True)
    total = 0

    for vname in VIDEOS:
        vpath = RAW_VIDEOS / vname
        if not vpath.exists():
            print(f"  SKIP (not found): {vpath}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, round(src_fps / TARGET_FPS))
        prefix = vpath.stem  # e.g. ladder_1, no_ladder

        idx = 0
        saved = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                fname = f"{prefix}_frame_{saved:04d}.jpg"
                cv2.imwrite(str(OUT_DIR / fname), frame)
                saved += 1
            idx += 1

        cap.release()
        total += saved
        print(f"  {vname}: {saved} frames at {TARGET_FPS} FPS (interval={interval})")

    return total


def copy_scraped_images():
    if not SCRAPED_DIR.exists():
        print("  No scraped/ folder found — run 0_scrape_images.py first")
        return 0

    count = 0
    for f in sorted(SCRAPED_DIR.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        dest = OUT_DIR / f"scraped_{f.stem}{f.suffix.lower()}"
        # Convert to jpg if webp
        if f.suffix.lower() == ".webp":
            img = cv2.imread(str(f))
            if img is not None:
                dest = dest.with_suffix(".jpg")
                cv2.imwrite(str(dest), img)
                count += 1
        else:
            shutil.copy2(f, dest)
            count += 1

    return count


if __name__ == "__main__":
    print("Extracting video frames...")
    n_video = extract_video_frames()

    print("\nCopying scraped images...")
    n_scraped = copy_scraped_images()

    print(f"\nTotal: {n_video} video frames + {n_scraped} scraped images = {n_video + n_scraped}")
    print(f"Output: {OUT_DIR}")
