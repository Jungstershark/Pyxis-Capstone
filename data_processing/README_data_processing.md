# Data Processing & Auto-Labeling

Shared module for frame splitting and GroundingDINO + SAM auto-labeling.

For full documentation, see [docs/01-methodology.md](../docs/01-methodology.md).

## Key Files

- `autolabel_grounded_sam_multiclass.py` -- Main auto-labeling pipeline (GroundingDINO proposal + SAM ViT-H refinement + dual NMS)
- `process.py` -- Random train/val/test split with seed-based reproducibility
- `weights/` -- GroundingDINO and SAM model checkpoints (see `weights/notes.md` for download instructions)
