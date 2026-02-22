from ultralytics import YOLOE
import os
import cv2


def save_labelimg_yolo(result, label_path):
    """
    Save YOLOE predictions into LabelImg-style YOLO annotation format.

    Parameters
    ----------
    result : YOLOE result object for a single image
        Contains .boxes with xyxy coordinates and class IDs.
    label_path : str
        Full path of the .txt label file to write.

    Notes
    -----
    - Writes YOLO format: <class_id> <x_center> <y_center> <width> <height>
    - All coordinates are normalized (0–1).
    - Only classes in ID_TO_TRAIN_NAME are saved.
    """

    h, w = result.orig_shape
    lines = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())

        # ignore predictions outside our chosen 3 classes
        if cls_id not in ID_TO_TRAIN_NAME:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # convert XYXY → normalized XYWH
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width    = (x2 - x1) / w
        height   = (y2 - y1) / h

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if lines:
        folder = os.path.dirname(label_path)
        os.makedirs(folder, exist_ok=True)

        with open(label_path, "w") as f:
            f.write("\n".join(lines))


def auto_label_images(input_dir, output_dir, max_images=10, conf=0.35):
    """
    Run YOLOE on up to N images in input_dir and save YOLO txt files to output_dir.

    Parameters
    ----------
    input_dir : str
        Directory containing the input images.
    output_dir : str
        Directory where YOLO txt files will be saved.
    max_images : int, optional
        Maximum number of images to auto-label.
    conf : float, optional
        Confidence threshold for YOLOE predictions.

    Returns
    -------
    list[str]
        List of image file paths that were processed.
    """

    print(f"Scanning for images in: {input_dir}")

    exts = (".jpg", ".jpeg", ".png")
    image_paths = []

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                full_path = os.path.join(root, f)
                image_paths.append(full_path)

    print(f"Found {len(image_paths)} images total.")
    image_paths = image_paths[:max_images]  # take first N
    print(f"Auto-labeling first {len(image_paths)} images...")

    for img_path in image_paths:
        rel = os.path.relpath(img_path, input_dir)
        base_no_ext = os.path.splitext(rel)[0]

        label_path = os.path.join(output_dir, base_no_ext + ".txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        results = model.predict(img_path, conf=conf, verbose=False)[0]
        save_labelimg_yolo(results, label_path)

        print(f"[LABEL] {img_path} -> {label_path}")

    return image_paths


def visualize_from_txt(input_dir, labels_dir, vis_output_dir):
    """
    Visualize bounding boxes from YOLO txt files and save annotated images.

    Parameters
    ----------
    input_dir : str
        Directory containing original images.
    labels_dir : str
        Directory containing YOLO txt label files.
    vis_output_dir : str
        Directory where annotated images will be saved.

    Notes
    -----
    Draws bounding boxes using:
    - GREEN  for pilot_ladder (class 0)
    - BLUE   for ship_hull (class 1)
    - ORANGE for pilot_or_crew (class 2)
    """

    os.makedirs(vis_output_dir, exist_ok=True)

    class_colors = {
        0: (0, 255, 0),     # ladder → green
        1: (255, 0, 0),     # hull → blue
        2: (0, 128, 255),   # crew → orange
    }

    exts = (".jpg", ".jpeg", ".png")

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue

            img_path = os.path.join(root, f)
            rel = os.path.relpath(img_path, input_dir)
            label_path = os.path.join(labels_dir, os.path.splitext(rel)[0] + ".txt")

            # Skip if no labels exist
            if not os.path.exists(label_path):
                continue

            img = cv2.imread(img_path)
            h, w = img.shape[:2]

            # Read YOLO txt file
            with open(label_path, "r") as lf:
                lines = lf.readlines()

            for line in lines:
                cls, xc, yc, bw, bh = line.strip().split()
                cls = int(cls)
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

                # convert back to xyxy
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)

                cv2.rectangle(img, (x1, y1), (x2, y2), class_colors[cls], 2)
                cv2.putText(img, ID_TO_TRAIN_NAME[cls], (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_colors[cls], 2)

            # save annotated image
            out_path = os.path.join(vis_output_dir, f)
            cv2.imwrite(out_path, img)

            print(f"[VIS] Saved: {out_path}")



yoloe_model_path = os.path.join(os.getcwd(), "yolo_weights", "yoloe", "yoloe-11l-seg.pt")
model = YOLOE(yoloe_model_path)

prompt_names = [
    "pilot ladder on the side of a ship",   # class 0
    "ship hull side of a large vessel",     # class 1
    "crew or pilot wearing helmet"          # class 2
]

text_embeddings = model.get_text_pe(prompt_names)
model.set_classes(prompt_names, text_embeddings)

ID_TO_TRAIN_NAME = {
    0: "pilot_ladder",
    1: "ship_hull",
    2: "pilot_or_crew",
}



if __name__ == "__main__":
    print("Working directory:", os.getcwd())
    cur_dir = os.getcwd()

    input_dir = os.path.join(cur_dir, "autolabel_images", "test_label_yoloe")
    labels_dir = os.path.join(cur_dir, "autolabel_images", "test_label_yoloe", "labels")
    vis_dir = os.path.join(cur_dir, "autolabel_images", "test_label_yoloe", "visual")

    # 1) Auto-label N images
    auto_label_images(input_dir, labels_dir, max_images=100)

    # 2) Visualize them
    visualize_from_txt(input_dir, labels_dir, vis_dir)

    print("\nDONE.")
    print(f"- Labels saved to: {labels_dir}")
    print(f"- Visualizations saved to: {vis_dir}")

