"""
Scrape pilot ladder images using DuckDuckGo image search.
Downloads images into datasets/classroom/scraped/ with deduplication.

Usage:
    python 0_scrape_images.py                  # scrape all queries
    python 0_scrape_images.py --max 50         # limit per query
"""
import argparse, hashlib, time, requests
from pathlib import Path
from duckduckgo_search import DDGS

QUERIES = [
    # Core pilot ladder queries
    "pilot ladder ship",
    "pilot ladder boarding",
    "pilot transfer ladder vessel",
    "pilot climbing ladder ship",
    "maritime pilot ladder",
    "ship pilot ladder close up",
    "pilot ladder rope ladder ship hull",
    # Variation — different ladder types and conditions
    "jacob's ladder ship",
    "rope ladder ship side",
    "accommodation ladder pilot",
    "pilot ladder night",
    "pilot ladder rough sea",
    "pilot ladder container ship",
    "pilot ladder tanker",
    "pilot ladder bulk carrier",
    # Person on ladder
    "pilot climbing ship ladder",
    "person climbing rope ladder ship",
    "maritime pilot boarding vessel",
    "crew member pilot ladder",
    # Indoor / training contexts
    "pilot ladder training",
    "rope ladder indoor",
    "pilot ladder safety training",
    "maritime safety ladder drill",
]

OUT_DIR = Path(__file__).resolve().parent / "scraped"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def get_file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def download_image(url: str, timeout: int = 10) -> bytes | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        if r.status_code != 200:
            return None
        content_type = r.headers.get("content-type", "")
        if not any(t in content_type for t in ["image", "octet-stream"]):
            return None
        data = r.content
        if len(data) < 5000:  # skip tiny images / icons
            return None
        return data
    except Exception:
        return None


def scrape(max_per_query=30):
    OUT_DIR.mkdir(exist_ok=True)
    seen_hashes = set()
    global_idx = 0

    # Index existing files
    for f in OUT_DIR.iterdir():
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            seen_hashes.add(hashlib.md5(f.read_bytes()).hexdigest())
            global_idx += 1

    total_new = 0

    with DDGS() as ddgs:
        for i, query in enumerate(QUERIES, 1):
            print(f"\n[{i}/{len(QUERIES)}] Searching: '{query}'")
            new = 0

            try:
                results = list(ddgs.images(query, max_results=max_per_query))
            except Exception as e:
                print(f"  -> Search error: {e}")
                time.sleep(2)
                continue

            for result in results:
                url = result.get("image", "")
                if not url:
                    continue

                data = download_image(url)
                if data is None:
                    continue

                h = get_file_hash(data)
                if h in seen_hashes:
                    continue

                seen_hashes.add(h)
                global_idx += 1

                # Determine extension from content
                ext = ".jpg"
                if data[:4] == b"\x89PNG":
                    ext = ".png"
                elif data[:4] == b"RIFF":
                    ext = ".webp"

                dest = OUT_DIR / f"scraped_{global_idx:04d}{ext}"
                dest.write_bytes(data)
                new += 1

            total_new += new
            print(f"  -> {new} new images downloaded")
            time.sleep(5)  # rate limit — DDG throttles aggressively

    print(f"\n{'='*50}")
    print(f"Total unique images: {len(seen_hashes)}")
    print(f"New images this run: {total_new}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=30, help="Max images per query")
    args = parser.parse_args()
    scrape(max_per_query=args.max)
