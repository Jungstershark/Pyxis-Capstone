Step 1 — Download weights (SAM + GroundingDINO)
1A) SAM weights (official Meta download server)

The Segment Anything repo links to the official checkpoints (ViT-H / ViT-L / ViT-B).
For speed + memory, start with vit_b (fastest). If accuracy isn’t good, swap to vit_l/h later.

Create a folder:

mkdir weights

Download SAM ViT-B:

Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "weights\sam_vit_b_01ec64.pth"

(Those official URLs are documented publicly; e.g., Kornia lists them and the Meta server URL.)

Download SAM ViT-H:

Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -OutFile "weights\sam_vit_h_4b8939.pth"

Expected size:

~2.4 GB
If it’s ~2,400,000,000 bytes → correct.

1B) GroundingDINO weights + config

We need:

weights: groundingdino_swint_ogc.pth

config: GroundingDINO_SwinT_OGC.cfg.py

Fastest reliable source (no auth): Hugging Face file hosting for both.

Download weights:

Invoke-WebRequest -Uri "https://huggingface.co/pengxian/grounding-dino/resolve/main/groundingdino_swint_ogc.pth" -OutFile "weights\groundingdino_swint_ogc.pth"

Download config:

Invoke-WebRequest -Uri "https://huggingface.co/pengxian/grounding-dino/resolve/mai

Step 2 — Install the right GroundingDINO package (important)

You installed groundingdino-py, but different wrappers expose different APIs. For the standard GroundingDINO inference utilities (the ones most tutorials assume), install the official repo package:

pip install git+https://github.com/IDEA-Research/GroundingDINO.git

(Official repo reference.)

Also make sure SAM is the official package (your pip install segment-anything might be a third-party mirror). Do this to be safe:

pip install git+https://github.com/facebookresearch/segment-anything.git