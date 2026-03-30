# python metrics_generation/dataset_metrics.py --root . --json --csv
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, defaultdict
import json
import csv


SPLITS = ("train", "val", "test")


@dataclass
class SplitStats:
    split: str
    images: int = 0
    labels: int = 0
    images_missing_label: int = 0
    labels_missing_image: int = 0
    empty_label_files: int = 0
    total_boxes: int = 0
    boxes_per_class: dict[str, int] = None  # filled later


@dataclass
class DatasetStats:
    dataset_path: str
    classes: list[str]
    split_stats: dict[str, SplitStats]
    total_images: int
    total_labels: int
    total_boxes: int
    boxes_per_class_total: dict[str, int]


def read_classes(dataset_root: Path) -> list[str]:
    """
    Supports:
      - dataset_root/classes.txt (your structure)
      - dataset_root/labels/classes.txt (some variants)
    Falls back to numeric class IDs if not found.
    """
    candidates = [
        dataset_root / "classes.txt",
        dataset_root / "labels" / "classes.txt",
    ]
    for p in candidates:
        if p.exists():
            lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
            return [ln for ln in lines if ln]
    return []


def list_files(folder: Path, exts: set[str]) -> list[Path]:
    if not folder.exists():
        return []
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def label_file_to_box_classes(label_path: Path) -> list[int]:
    """
    YOLO label format per line: <class_id> <x> <y> <w> <h>
    Ignores malformed lines safely.
    """
    try:
        txt = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    if not txt:
        return []

    classes = []
    for ln in txt.splitlines():
        parts = ln.strip().split()
        if not parts:
            continue
        try:
            cid = int(float(parts[0]))
            classes.append(cid)
        except Exception:
            # skip malformed
            continue
    return classes


def compute_split_stats(dataset_root: Path, split: str, class_names: list[str]) -> SplitStats:
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    # Common image extensions
    image_files = list_files(img_dir, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    label_files = list_files(lbl_dir, {".txt"})

    # Map stems -> file paths
    img_by_stem = {p.stem: p for p in image_files}
    lbl_by_stem = {p.stem: p for p in label_files if p.name != "classes.txt"}

    # Missing pairs
    missing_label = [s for s in img_by_stem.keys() if s not in lbl_by_stem]
    missing_image = [s for s in lbl_by_stem.keys() if s not in img_by_stem]

    # Box counts
    class_counter = Counter()
    empty_label_files = 0
    total_boxes = 0

    for stem, lp in lbl_by_stem.items():
        cls_ids = label_file_to_box_classes(lp)
        if len(cls_ids) == 0:
            empty_label_files += 1
            continue
        total_boxes += len(cls_ids)
        class_counter.update(cls_ids)

    # Build boxes_per_class with readable names
    boxes_per_class = {}
    if class_names:
        for i, name in enumerate(class_names):
            boxes_per_class[name] = int(class_counter.get(i, 0))
        # Add any unexpected class ids
        unexpected = [cid for cid in class_counter.keys() if cid >= len(class_names)]
        for cid in sorted(unexpected):
            boxes_per_class[f"class_{cid}"] = int(class_counter[cid])
    else:
        # No class file found
        for cid in sorted(class_counter.keys()):
            boxes_per_class[f"class_{cid}"] = int(class_counter[cid])

    return SplitStats(
        split=split,
        images=len(image_files),
        labels=len(lbl_by_stem),
        images_missing_label=len(missing_label),
        labels_missing_image=len(missing_image),
        empty_label_files=empty_label_files,
        total_boxes=total_boxes,
        boxes_per_class=boxes_per_class,
    )


def compute_dataset_stats(dataset_root: Path) -> DatasetStats:
    class_names = read_classes(dataset_root)
    split_stats = {}
    total_images = total_labels = total_boxes = 0
    total_class_counter = Counter()

    for sp in SPLITS:
        st = compute_split_stats(dataset_root, sp, class_names)
        split_stats[sp] = st
        total_images += st.images
        total_labels += st.labels
        total_boxes += st.total_boxes

        # merge class counts
        for k, v in (st.boxes_per_class or {}).items():
            total_class_counter[k] += v

    return DatasetStats(
        dataset_path=str(dataset_root),
        classes=class_names,
        split_stats=split_stats,
        total_images=total_images,
        total_labels=total_labels,
        total_boxes=total_boxes,
        boxes_per_class_total=dict(total_class_counter),
    )


def print_human(ds: DatasetStats) -> None:
    print("=" * 80)
    print(f"DATASET: {ds.dataset_path}")
    if ds.classes:
        print(f"CLASSES ({len(ds.classes)}): {ds.classes}")
    else:
        print("CLASSES: (classes.txt not found) using class_0, class_1, ...")
    print("-" * 80)
    for sp in SPLITS:
        st = ds.split_stats.get(sp)
        if not st:
            continue
        print(f"[{sp.upper()}]")
        print(f"  images: {st.images}")
        print(f"  labels: {st.labels}")
        print(f"  images missing label: {st.images_missing_label}")
        print(f"  labels missing image: {st.labels_missing_image}")
        print(f"  empty label files: {st.empty_label_files}")
        print(f"  total boxes: {st.total_boxes}")
        print(f"  boxes per class: {st.boxes_per_class}")
        print()
    print("-" * 80)
    print(f"TOTAL images: {ds.total_images}")
    print(f"TOTAL labels: {ds.total_labels}")
    print(f"TOTAL boxes:  {ds.total_boxes}")
    print(f"TOTAL boxes per class: {ds.boxes_per_class_total}")
    print("=" * 80)


def write_json(out_path: Path, ds_list: list[DatasetStats]) -> None:
    payload = []
    for ds in ds_list:
        d = asdict(ds)
        # dataclasses -> dict includes SplitStats nested as dict already
        payload.append(d)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(out_path: Path, ds_list: list[DatasetStats]) -> None:
    """
    Writes one row per dataset per split, plus a TOTAL row per dataset.
    """
    rows = []
    for ds in ds_list:
        for sp in SPLITS:
            st = ds.split_stats.get(sp)
            if not st:
                continue
            row = {
                "dataset": ds.dataset_path,
                "split": sp,
                "images": st.images,
                "labels": st.labels,
                "images_missing_label": st.images_missing_label,
                "labels_missing_image": st.labels_missing_image,
                "empty_label_files": st.empty_label_files,
                "total_boxes": st.total_boxes,
            }
            # flatten class counts
            for cls_name, cnt in (st.boxes_per_class or {}).items():
                row[f"boxes_{cls_name}"] = cnt
            rows.append(row)

        total_row = {
            "dataset": ds.dataset_path,
            "split": "TOTAL",
            "images": ds.total_images,
            "labels": ds.total_labels,
            "images_missing_label": "",
            "labels_missing_image": "",
            "empty_label_files": "",
            "total_boxes": ds.total_boxes,
        }
        for cls_name, cnt in (ds.boxes_per_class_total or {}).items():
            total_row[f"boxes_{cls_name}"] = cnt
        rows.append(total_row)

    # determine all headers
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default=".",
        help="Path containing dataset folders (e.g. model_finetuning/).",
    )
    ap.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=[
            "dataset_split_v2_gpu_corrected",
            "dataset_split_v2_gpu_corrected_t0",
            "dataset_split_v2_gpu_corrected_t1",
            "dataset_split_v2_gpu_corrected_t2",
        ],
        help="Dataset folder names under --root.",
    )
    ap.add_argument("--out_dir", type=str, default="metrics_generation/out")
    ap.add_argument("--json", action="store_true", help="Write JSON summary")
    ap.add_argument("--csv", action="store_true", help="Write CSV summary")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_list: list[DatasetStats] = []
    for name in args.datasets:
        ds_path = root / name
        if not ds_path.exists():
            print(f"[SKIP] Not found: {ds_path}")
            continue
        ds = compute_dataset_stats(ds_path)
        ds_list.append(ds)
        print_human(ds)

    if args.json:
        write_json(out_dir / "dataset_metrics.json", ds_list)
        print(f"Wrote JSON -> {out_dir / 'dataset_metrics.json'}")

    if args.csv:
        write_csv(out_dir / "dataset_metrics.csv", ds_list)
        print(f"Wrote CSV  -> {out_dir / 'dataset_metrics.csv'}")


if __name__ == "__main__":
    main()