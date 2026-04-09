"""
Local analysis script — runs the same experiments as the Colab notebook.
1. Confidence threshold sweep (3 models × 2 test sets × 9 thresholds)
2. Cross-dataset evaluation (3 models × 2 test sets)
3. Test-time augmentation (3 models on primary test set)
"""
import json, csv, sys
import numpy as np
from pathlib import Path

# Ensure ultralytics is available
from ultralytics import YOLO
import torch

WORK = Path(__file__).resolve().parent

SG_IMAGES = WORK / 'gold_test' / 'singapore_river' / 'images'
SG_LABELS = WORK / 'gold_test' / 'singapore_river' / 'labels'
IS_IMAGES = WORK / 'gold_test' / 'internet_scraped' / 'images'
IS_LABELS = WORK / 'gold_test' / 'internet_scraped' / 'labels'

MODELS = {
    'SG': {'path': str(WORK / 'weights' / 'sg_y11s_t0_960.pt'), 'imgsz': 960},
    'IS': {'path': str(WORK / 'weights' / 'is_v2_y11s_960.pt'), 'imgsz': 960},
    'Combined': {'path': str(WORK / 'weights' / 'combined_y11s_1280.pt'), 'imgsz': 1280},
}

TEST_SETS = {
    'SG test': {'images': SG_IMAGES, 'labels': SG_LABELS},
    'IS test': {'images': IS_IMAGES, 'labels': IS_LABELS},
}

# ── helpers ──────────────────────────────────────────────────────────

def load_labels(path, cls_id):
    boxes = []
    if not path.exists():
        return boxes
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return boxes
    for line in text.split('\n'):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        if int(parts[0]) != cls_id:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x0, y0 = max(0, cx - w / 2), max(0, cy - h / 2)
        x1, y1 = min(1, cx + w / 2), min(1, cy + h / 2)
        boxes.append([x0, y0, x1, y1])
    return boxes

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    d = aa + ab - inter
    return inter / d if d > 0 else 0.0

def greedy_match(gt, pred, thresh=0.5):
    matched = set()
    tp = 0
    for p in pred:
        best_iou, best_j = 0, -1
        for j, g in enumerate(gt):
            if j in matched:
                continue
            s = iou(g, p)
            if s > best_iou:
                best_iou, best_j = s, j
        if best_iou >= thresh and best_j >= 0:
            matched.add(best_j)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp

def evaluate(model_path, images_dir, labels_dir, conf=0.25, imgsz=960, augment=False):
    model = YOLO(model_path)
    m = {0: {'tp': 0, 'fp': 0, 'fn': 0}, 1: {'tp': 0, 'fp': 0, 'fn': 0}}
    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

    for img in imgs:
        lbl = labels_dir / (img.stem + '.txt')
        gt = {0: load_labels(lbl, 0), 1: load_labels(lbl, 1)}
        pred = model.predict(str(img), imgsz=imgsz, conf=conf, iou=0.7, verbose=False, augment=augment)[0]
        pred_by_cls = {0: [], 1: []}

        if pred.boxes is not None and len(pred.boxes) > 0:
            w_img, h_img = pred.orig_shape[1], pred.orig_shape[0]
            xyxy = pred.boxes.xyxy.cpu().numpy()
            cls_arr = pred.boxes.cls.cpu().numpy().astype(int)
            conf_arr = pred.boxes.conf.cpu().numpy()
            for k in np.argsort(-conf_arr):
                c = cls_arr[k]
                if c not in pred_by_cls:
                    continue
                b = [xyxy[k][0] / w_img, xyxy[k][1] / h_img, xyxy[k][2] / w_img, xyxy[k][3] / h_img]
                pred_by_cls[c].append(b)

        for c in [0, 1]:
            tp, fp, fn = greedy_match(gt[c], pred_by_cls[c])
            m[c]['tp'] += tp
            m[c]['fp'] += fp
            m[c]['fn'] += fn

    results = {}
    for c, name in [(0, 'ladder'), (1, 'person')]:
        tp, fp, fn = m[c]['tp'], m[c]['fp'], m[c]['fn']
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0
        results[f'{name}_recall'] = round(r, 4)
        results[f'{name}_precision'] = round(p, 4)
        results[f'{name}_f1'] = round(f1, 4)
    results['n_images'] = len(imgs)
    return results


# ── main ─────────────────────────────────────────────────────────────

def main():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'GPU: {gpu}')
    print()

    out = WORK / 'analysis_results.json'
    all_results = {}

    # ── 1. Confidence threshold sweep ──
    print('=' * 70)
    print('EXPERIMENT 1: Confidence Threshold Sweep')
    print('=' * 70)
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    sweep = {}

    for model_name, mcfg in MODELS.items():
        for test_name, tcfg in TEST_SETS.items():
            key = f'{model_name} -> {test_name}'
            print(f'\n--- {key} ---')
            rows = []
            for conf in thresholds:
                r = evaluate(mcfg['path'], tcfg['images'], tcfg['labels'], conf=conf, imgsz=mcfg['imgsz'])
                r['conf'] = conf
                rows.append(r)
                print(f"  conf={conf:.2f} | Ladder R={r['ladder_recall']:.3f} P={r['ladder_precision']:.3f} F1={r['ladder_f1']:.3f} | Person R={r['person_recall']:.3f} P={r['person_precision']:.3f} F1={r['person_f1']:.3f}")
            # Find best
            best = max(rows, key=lambda x: x['ladder_f1'])
            print(f"  >> Best ladder F1={best['ladder_f1']:.3f} at conf={best['conf']}")
            sweep[key] = rows

    all_results['threshold_sweep'] = sweep

    # ── 2. Cross-dataset evaluation ──
    print()
    print('=' * 70)
    print('EXPERIMENT 2: Cross-Dataset Evaluation (conf=0.25)')
    print('=' * 70)
    cross = []

    for model_name, mcfg in MODELS.items():
        for test_name, tcfg in TEST_SETS.items():
            r = evaluate(mcfg['path'], tcfg['images'], tcfg['labels'], conf=0.25, imgsz=mcfg['imgsz'])
            r['model'] = model_name
            r['test_set'] = test_name
            cross.append(r)
            print(f"  {model_name:>10} -> {test_name}: Ladder R={r['ladder_recall']:.3f} P={r['ladder_precision']:.3f} F1={r['ladder_f1']:.3f} | Person R={r['person_recall']:.3f} P={r['person_precision']:.3f} F1={r['person_f1']:.3f}")

    all_results['cross_dataset'] = cross

    # ── 3. TTA ──
    print()
    print('=' * 70)
    print('EXPERIMENT 3: Test-Time Augmentation')
    print('=' * 70)
    tta = []

    tta_configs = [
        ('SG', 'SG test'),
        ('IS', 'IS test'),
        ('Combined', 'IS test'),
    ]

    for model_name, test_name in tta_configs:
        mcfg = MODELS[model_name]
        tcfg = TEST_SETS[test_name]
        r_normal = evaluate(mcfg['path'], tcfg['images'], tcfg['labels'], conf=0.25, imgsz=mcfg['imgsz'], augment=False)
        r_tta = evaluate(mcfg['path'], tcfg['images'], tcfg['labels'], conf=0.25, imgsz=mcfg['imgsz'], augment=True)

        lr_diff = r_tta['ladder_recall'] - r_normal['ladder_recall']
        pr_diff = r_tta['person_recall'] - r_normal['person_recall']
        verdict = 'HELPS' if lr_diff > 0.01 else 'NO EFFECT' if lr_diff > -0.01 else 'HURTS'

        print(f"\n  {model_name} model on {test_name}:")
        print(f"    Normal: Ladder R={r_normal['ladder_recall']:.3f} P={r_normal['ladder_precision']:.3f} F1={r_normal['ladder_f1']:.3f} | Person R={r_normal['person_recall']:.3f} P={r_normal['person_precision']:.3f}")
        print(f"    TTA:    Ladder R={r_tta['ladder_recall']:.3f} P={r_tta['ladder_precision']:.3f} F1={r_tta['ladder_f1']:.3f} | Person R={r_tta['person_recall']:.3f} P={r_tta['person_precision']:.3f}")
        print(f"    Delta:  Ladder R {lr_diff:+.3f} | Person R {pr_diff:+.3f} — {verdict}")

        tta.append({
            'model': model_name,
            'test_set': test_name,
            'normal': r_normal,
            'tta': r_tta,
            'ladder_recall_diff': round(lr_diff, 4),
            'person_recall_diff': round(pr_diff, 4),
            'verdict': verdict,
        })

    all_results['tta'] = tta

    # ── Save ──
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n\nAll results saved to {out}')


if __name__ == '__main__':
    main()
