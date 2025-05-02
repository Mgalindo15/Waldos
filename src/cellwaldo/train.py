from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="cellwaldo.yaml",
    epochs=30,
    imgsz=640,
    lr0=1e-3,
    split=0.8
)