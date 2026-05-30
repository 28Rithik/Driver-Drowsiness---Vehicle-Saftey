from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("The 'ultralytics' package is required. Install with 'pip install ultralytics' and retry.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO26 model on the prepared NTHU dataset.")
    parser.add_argument("--data", type=Path, default=Path("example/data.yaml"), help="Path to YOLO data YAML.")
    parser.add_argument("--model", type=str, default="yolo26n.pt", help="Pretrained model or yaml to start from.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (square).")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (adjust to your GPU).")
    parser.add_argument("--device", type=str, default=None, help="Device to use, e.g. '0' or 'cpu'.")
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Project output folder.")
    parser.add_argument("--name", type=str, default="yolo26_project", help="Run name inside project folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data YAML not found: {args.data}")

    print("Starting YOLO26 training with settings:")
    print(f"  data: {args.data}")
    print(f"  model: {args.model}")
    print(f"  epochs: {args.epochs}")
    print(f"  imgsz: {args.imgsz}")
    print(f"  batch: {args.batch}")
    print(f"  device: {args.device or 'auto'}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )


if __name__ == "__main__":
    main()
