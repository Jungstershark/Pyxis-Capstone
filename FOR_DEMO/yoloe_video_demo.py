import cv2
import supervision as sv
from ultralytics import YOLO
import os

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
RAW_VIDEO_DIR = "raw_dataset/pilot_transfer_videos"
OUTPUT_DIR = "FOR_DEMO/output"

# Prompt classes
NAMES = [
    "pilot ladder, rope ladder used for pilot transfer, wooden-step ladder with side ropes, maritime boarding ladder hanging off ship hull",
    "person, human crew member standing or climbing",
    "ship hull, vessel side, exterior hull of cargo ship"
]

YOLOE_MODEL_PATH = os.path.join(os.getcwd(), "yolo_weights", "yoloe", "yoloe-11l-seg.pt")
model = YOLO(YOLOE_MODEL_PATH)
model.set_classes(NAMES, model.get_text_pe(NAMES))
model.to("cpu")

# Annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def process_video(input_path: str, output_path: str):
    """
    Runs YOLOE inference on a single video and saves the annotated version.
    """

    print(f"\n[INFO] Processing video: {input_path}")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    print("[INFO] Starting inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        print(f"  - Frame {frame_idx}")

        # YOLOE inference
        results = model.predict(frame)[0]

        # Convert to supervision format
        detections = sv.Detections.from_ultralytics(results)

        # Annotate frame
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Write to output
        out.write(annotated_frame)

    cap.release()
    out.release()

    print(f"[DONE] Saved annotated video → {output_path}")


if __name__ == "__main__":
    print("[INFO] Scanning directory:", RAW_VIDEO_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_exts = (".mp4", ".avi", ".mov", ".mkv")

    for filename in os.listdir(RAW_VIDEO_DIR):

        if not filename.lower().endswith(valid_exts):
            continue

        input_path = os.path.join(RAW_VIDEO_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")

        process_video(input_path, output_path)

    print("\n[ALL COMPLETE] All videos processed.")
