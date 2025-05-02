import cv2
import numpy as np
from pathlib import Path

def make_cell_waldo(out_path="cell_waldo.png",
                    radius=40,
                    cytoplasm_color=(175, 125, 255),
                    nucleus_color=(60, 0, 140)):
    r = radius
    png = np.zeros((2*r, 2*r, 4), dtype=np.uint8)

    for i in range(-r, r, 8):
        cv2.line(png, (0, i + r), (2*r, i + r),
                 cytoplasm_color + (255,), 5)

    cv2.circle(png, (r, r), r - 2, cytoplasm_color + (255,), -1)

    cv2.circle(png, (r, r), int(r * 0.55),
               nucleus_color + (255,), -1)
    cv2.circle(png, (r, r), int(r * 0.55),
               (255, 255, 255, 0), 3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, png)
    print(f"Saved → {out_path}")

if __name__ == "__main__":
    make_cell_waldo()
