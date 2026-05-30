from __future__ import annotations

from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Install 'ultralytics' in your environment (pip install ultralytics)")

MODEL_PATH = "yolov8n.pt"
DATA_YAML = "example/data.yaml"
EPOCHS = 50
IMGSZ = 520


def main() -> None:
    model = YOLO(MODEL_PATH)
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMGSZ)


if __name__ == "__main__":
    main()
