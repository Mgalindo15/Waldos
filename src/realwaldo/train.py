from ultralytics import YOLO

model = YOLO("CELLWALDO/runs/detect/train3/weights/best.pt")

model.train(
    data="waldo_real.yaml",
    epochs=50,
    imgsz=1024,
    batch=4,
    lr0=3e-4,
    single_cls=True,
    patience=8
)