# Model Fine-Tuning & Evaluation

Shared module for YOLO training, evaluation, and augmentation.

For full documentation, see [docs/01-methodology.md](../docs/01-methodology.md) (training protocol) and [docs/02-training-and-results.md](../docs/02-training-and-results.md) (all results).

## Key Files

- `train.py` -- YOLO training harness (single experiment + batch wave runner, YAML generation, config snapshots)
- `evaluate.py` -- Gold test set evaluation (greedy IoU matching, per-class precision/recall, confidence and area statistics)
- `augment.py` -- Offline augmentation tiers via Albumentations (deprecated -- YOLO built-in augmentation is sufficient)
- `yolo11s.pt`, `yolo11m.pt` -- YOLO11 base weights
