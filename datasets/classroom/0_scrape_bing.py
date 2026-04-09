"""
Scrape pilot ladder images using Bing Image Search via icrawler.
Bing is more permissive than Google/DDG for bulk image scraping.

Usage:
    python 0_scrape_bing.py              # scrape all queries
    python 0_scrape_bing.py --max 40     # limit per query
"""
import argparse, hashlib, shutil, time
from pathlib import Path
from icrawler.builtin import BingImageCrawler

# Remaining queries that DDG didn't get to
QUERIES = [
    "pilot ladder rope ladder ship hull",
    "jacob's ladder ship",
    "rope ladder ship side",
    "accommodation ladder pilot",
    "pilot ladder night",
    "pilot ladder rough sea",
    "pilot ladder container ship",
    "pilot ladder tanker",
    "pilot ladder bulk carrier",
    "pilot climbing ship ladder",
    "person climbing rope ladder ship",
    "maritime pilot boarding vessel",
    "crew member pilot ladder",
    "pilot ladder training",
    "rope ladder indoor",
    "pilot ladder safety training",
    "maritime safety ladder drill",
]

OUT_DIR = Path(__file__).resolve().parent / "scraped"


def get_file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scrape(max_per_query=30):
    OUT_DIR.mkdir(exist_ok=True)
    seen_hashes = set()
    global_idx = 0

    # Index existing files to avoid duplicates
    for f in sorted(OUT_DIR.iterdir()):
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            seen_hashes.add(get_file_hash(f))
            global_idx += 1

    print(f"Existing images: {len(seen_hashes)}")
    total_new = 0

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] Scraping: '{query}' (max {max_per_query})")

        tmp = OUT_DIR / "_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()

        try:
            crawler = BingImageCrawler(
                storage={"root_dir": str(tmp)},
                log_level="WARNING",
            )
            crawler.crawl(
                keyword=query,
                max_num=max_per_query,
                min_size=(200, 200),
            )

            new = 0
            for f in sorted(tmp.iterdir()):
                if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                # Skip tiny files (icons, thumbnails)
                if f.stat().st_size < 5000:
                    f.unlink()
                    continue
                h = get_file_hash(f)
                if h in seen_hashes:
                    f.unlink()
                    continue
                seen_hashes.add(h)
                global_idx += 1
                dest = OUT_DIR / f"scraped_{global_idx:04d}{f.suffix.lower()}"
                f.rename(dest)
                new += 1

            total_new += new
            print(f"  -> {new} new images")

        except Exception as e:
            print(f"  -> ERROR: {e}")
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)

        time.sleep(2)

    print(f"\n{'='*50}")
    print(f"Total unique images: {len(seen_hashes)}")
    print(f"New images this run: {total_new}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=30, help="Max images per query")
    args = parser.parse_args()
    scrape(max_per_query=args.max)
