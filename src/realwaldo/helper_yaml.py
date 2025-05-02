from pathlib import Path

yaml_text = """\
path: ./waldo_dataset
train: images/train
val:   images/val
nc: 1
names: ["waldo"]
"""

Path("waldo_real.yaml").write_text(yaml_text)