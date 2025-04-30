import xml.etree.ElementTree as ET
from pathlib import Path
import shutil, random, os

#paths
ROOT = Path("FindingWaldo/object_detection/data")
XML_DIR = ROOT / "annotations"
IMG_DIR = ROOT / "images"
OUT = Path("waldo_dataset")

#make yolo layout for conversion
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

xml_files = list(XML_DIR.glob("*.xml"))
random.shuffle(xml_files)

split_idx = int(0.8 * len(xml_files))

def voc2yolo(size, box):
    w, h = size
    x_min, y_min, x_max, y_max = box
    cx = (x_min + x_max) / 2 / w
    cy = (y_min + y_max) / 2 / h
    bw = (x_max - x_min) / w
    bh = (y_max - y_min) / h
    return cx, cy, bw, bh

for i, xml_path in enumerate(xml_files):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.find("filename").text
    size  = root.find("size")
    w, h  = int(size.find("width").text), int(size.find("height").text)

    obj   = root.find("object")
    box   = obj.find("bndbox")
    xmin  = int(box.find("xmin").text)
    ymin  = int(box.find("ymin").text)
    xmax  = int(box.find("xmax").text)
    ymax  = int(box.find("ymax").text)
    cx, cy, bw, bh = voc2yolo((w, h), (xmin, ymin, xmax, ymax))

    subset = "train" if i < split_idx else "val"
    # copy image
    shutil.copy(IMG_DIR / fname, OUT / f"images/{subset}/{fname}")
    # write label
    label_path = OUT / f"labels/{subset}/{fname.rsplit('.',1)[0]}.txt"
    with open(label_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print(f"Dataset created at {OUT.absolute()}")