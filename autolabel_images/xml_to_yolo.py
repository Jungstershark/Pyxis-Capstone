# Inspired by:
# https://medium.com/internet-of-technology/pascal-voc-xml-to-yolo-conversion-a-comprehensive-python-guide-eac3838d30bb

import os
import xml.etree.ElementTree as ET

def convert_bbox_to_yolo(size, box):
    """
    Convert a PASCAL VOC bounding box to YOLO normalized format.

    Parameters
    ----------
    size : tuple(int, int)
        A tuple `(width, height)` representing image size in pixels.
    box : tuple(float, float, float, float)
        A tuple `(xmin, ymin, xmax, ymax)` representing the bounding
        box coordinates in absolute pixel values.

    Returns
    -------
    tuple(float, float, float, float)
        A tuple `(x_center, y_center, width, height)` where each value
        is normalized to the range [0, 1], following YOLO format.
    """
    w, h = size
    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h

    return x_center, y_center, width, height


def xml_to_txt(input_xml, output_txt, class_mapping):
    """
    Convert a single PASCAL VOC XML annotation file to YOLO format.

    Parameters
    ----------
    input_xml : str
        Path to the input `.xml` annotation file.
    output_txt : str
        Path where the corresponding YOLO `.txt` file will be saved.
    class_mapping : dict[str, int]
        Dictionary mapping class names (strings) to numerical YOLO class IDs.

    Notes
    -----
    - Only objects whose class names exist in `class_mapping`
      will be written to the output file.
    - Bounding box coordinates are normalized to YOLO format.
    """
    tree = ET.parse(input_xml)
    root = tree.getroot()

    width = int(root.find(".//size/width").text)
    height = int(root.find(".//size/height").text)

    with open(output_txt, "w", encoding="utf-8") as f:
        for obj in root.iter("object"):
            name = obj.find("name").text

            if name not in class_mapping:
                continue

            class_id = class_mapping[name]

            xmlbox = obj.find("bndbox")
            box = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymax").text),
            )

            xc, yc, w, h = convert_bbox_to_yolo((width, height), box)

            f.write(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def main(input_dir, output_dir, class_mapping):
    """
    Convert an entire directory of PASCAL VOC XML files to YOLO format.

    This function iterates through the specified input directory,
    converts each XML annotation file, and saves the resulting YOLO
    `.txt` files to the specified output directory.

    Parameters
    ----------
    input_dir : str
        Directory containing input XML annotation files.
    output_dir : str
        Target directory where YOLO `.txt` files will be saved.
    class_mapping : dict[str, int]
        Mapping between class names and YOLO class IDs.

    Notes
    -----
    - Output directory will be created if it does not exist.
    - Only files ending with `.xml` are processed.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for xml_file in os.listdir(input_dir):
        if xml_file.endswith(".xml"):
            input_xml_path = os.path.join(input_dir, xml_file)
            output_txt_path = os.path.join(
                output_dir, xml_file.replace(".xml", ".txt")
            )
            xml_to_txt(input_xml_path, output_txt_path, class_mapping)


if __name__ == "__main__":

    # Define your class mapping for YOLO indices
    class_mapping = {
        "pilot_ladder": 0,
        "ship_hull": 1,
        "pilot_or_crew": 2,
    }

    # Directories (edit these paths as needed)
    input_dir = "test_xml_conversion"
    output_dir = "test_xml_conversion/test"

    print(f"Converting XML annotations from: {input_dir}")
    print(f"Saving YOLO labels to: {output_dir}")
    print(f"Using class mapping: {class_mapping}\n")

    main(input_dir, output_dir, class_mapping)

    print("Conversion complete.")

