"""Train the demonstration YOLO bottle-cap detector and install its best weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    training_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=training_dir / "dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=Path, default=training_dir / "runs")
    parser.add_argument("--name", default="demo_caps")
    parser.add_argument(
        "--install-to",
        type=Path,
        default=training_dir.parent / "gui" / "weights" / "bottle_cap_best.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=18,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    args.install_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, args.install_to)

    validation = YOLO(str(args.install_to)).val(data=str(args.data), device=device, workers=0)
    metrics = {
        "weights": str(args.install_to),
        "device": str(device),
        "map50": float(validation.box.map50),
        "map50_95": float(validation.box.map),
        "classes": ["red_cap", "cestbon_cap"],
        "purpose": "integration-demo-only",
    }
    metrics_path = args.project / args.name / "demo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
