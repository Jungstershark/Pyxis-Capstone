# Step 4 Output: Augmentation Tiers (Deprecated)

- `t0/` -- Baseline (no augmentation)
- `t1/` -- Lighting robustness (brightness, contrast, gamma)
- `t2/` -- Motion robustness (blur, dropout)

Offline augmentation was tested but found to provide no benefit over YOLO's built-in online augmentation. This step is retained for reference.

Not tracked in git. Regenerate with:
```bash
python 4_augment.py
```
