import albumentations as A
import cv2, numpy as np, random, os
from glob import glob
from pathlib import Path
from tqdm import tqdm

BG_DIR = "histology_backgrounds"
WALDO_PATH = "cell_waldo.png"
OUT_DIR = Path("synthetic_dataset")
N_SAMPLES = 8000 
IMG_SIZE = 640

OUT_DIR.joinpath("images").mkdir(parents=True, exist_ok=True)
OUT_DIR.joinpath("labels").mkdir(parents=True, exist_ok=True)

bg_paths  = glob(os.path.join(BG_DIR, "*"))

waldo_rgba = cv2.imread(WALDO_PATH, cv2.IMREAD_UNCHANGED)

waldo_aug = A.Compose([
    A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.9),
    A.RandomScale(scale_limit=0.6, p=1.0),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
])

for i in tqdm(range(N_SAMPLES)):
    bg = cv2.imread(random.choice(bg_paths))
    h, w = bg.shape[:2]
    c = min(h, w)
    y0 = random.randint(0, h-c)
    x0 = random.randint(0, w-c)
    patch = bg[y0:y0+c, x0:x0+c]
    patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE))

    waldo = waldo_aug(image=waldo_rgba)["image"]
    wh, ww = waldo.shape[:2]

    max_x = IMG_SIZE - ww - 1
    max_y = IMG_SIZE - wh - 1
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    alpha = waldo[..., 3:] / 255.0
    patch[y:y+wh, x:x+ww] = (
        alpha * waldo[..., :3] + (1-alpha) * patch[y:y+wh, x:x+ww]
    ).astype(np.uint8)

    img_name = f"cellwaldo_{i:05d}.jpg"
    cv2.imwrite(str(OUT_DIR/"images"/img_name), patch)

    cx = (x + ww/2) / IMG_SIZE
    cy = (y + wh/2) / IMG_SIZE
    bw = ww / IMG_SIZE
    bh = wh / IMG_SIZE
    with open(OUT_DIR/"labels"/img_name.replace(".jpg", ".txt"), "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
