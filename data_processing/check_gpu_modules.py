from pathlib import Path
import torch

from groundingdino.util.inference import load_model, load_image
from segment_anything import sam_model_registry

# ----------------------------
# CONFIG (adjust paths if needed)
# ----------------------------
gdino_cfg = Path("weights") / "GroundingDINO_SwinT_OGC.cfg.py"
gdino_ckpt = Path("weights") / "groundingdino_swint_ogc.pth"
sam_ckpt = Path("weights") / "sam_vit_h_4b8939.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n=== CUDA ENVIRONMENT CHECK ===")
print("torch.cuda.is_available():", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name :", torch.cuda.get_device_name(0))
    print("Current device   :", torch.cuda.current_device())
else:
    print("⚠️ CUDA not available")

print("\n=== LOADING MODELS ===")

# ----------------------------
# Load GroundingDINO
# ----------------------------
print("\nLoading GroundingDINO...")
gdino = load_model(str(gdino_cfg), str(gdino_ckpt)).to(DEVICE).eval()

print("DINO param device:", next(gdino.parameters()).device)

# ----------------------------
# Load SAM
# ----------------------------
print("\nLoading SAM...")
sam = sam_model_registry["vit_h"](checkpoint=str(sam_ckpt)).to(DEVICE).eval()

print("SAM param device :", next(sam.parameters()).device)

# ----------------------------
# Memory check
# ----------------------------
if torch.cuda.is_available():
    torch.cuda.synchronize()
    print("\n=== CUDA MEMORY STATUS ===")
    print("Allocated (MB):", torch.cuda.memory_allocated() / 1024 / 1024)
    print("Reserved  (MB):", torch.cuda.memory_reserved() / 1024 / 1024)

print("\n✅ Done.")