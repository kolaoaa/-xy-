"""Validate installed demo weights and render predictions for the source photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB data to the BGR array format expected by Ultralytics."""

    return np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])


def parse_args() -> argparse.Namespace:
    training_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        type=Path,
        default=training_dir.parent / "gui" / "weights" / "bottle_cap_best.pt",
    )
    parser.add_argument("--data", type=Path, default=training_dir / "dataset.yaml")
    parser.add_argument("--red-dir", type=Path, required=True)
    parser.add_argument("--cestbon-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=training_dir / "evaluation")
    return parser.parse_args()


def render_predictions(model: YOLO, sources: list[tuple[str, Path]], output_path: Path, device) -> dict:
    tile_width = 360
    tile_height = 310
    columns = 2
    rows = (len(sources) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    summary = {"images": 0, "detections": 0, "class_counts": {}}

    for index, (expected_class, path) in enumerate(sources):
        image = Image.open(path).convert("RGB")
        result = model.predict(pil_to_bgr_array(image), conf=0.25, imgsz=640, device=device, verbose=False)[0]
        draw = ImageDraw.Draw(image)
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id])
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            color = (220, 40, 40) if class_name == "red_cap" else (20, 150, 90)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(3, image.width // 300))
            draw.text((x1, max(0, y1 - 24)), f"{class_name} {confidence:.2f}", fill=color)
            summary["detections"] += 1
            summary["class_counts"][class_name] = summary["class_counts"].get(class_name, 0) + 1

        preview = ImageOps.contain(image, (tile_width - 16, tile_height - 44))
        x = (index % columns) * tile_width + (tile_width - preview.width) // 2
        y = (index // columns) * tile_height + 4
        sheet.paste(preview, (x, y))
        caption = f"expected: {expected_class} | {path.name[:20]}"
        ImageDraw.Draw(sheet).text(((index % columns) * tile_width + 8, (index // columns) * tile_height + tile_height - 30), caption, fill="black")
        summary["images"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return summary


def main() -> None:
    args = parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))
    validation = model.val(data=str(args.data), device=device, workers=0, verbose=False)
    sources = [
        *[("red_cap", path) for path in sorted(args.red_dir.glob("*.jpg"))],
        *[("cestbon_cap", path) for path in sorted(args.cestbon_dir.glob("*.jpg"))],
    ]
    prediction_summary = render_predictions(
        model,
        sources,
        args.output_dir / "source_predictions.jpg",
        device,
    )
    metrics = {
        "weights": str(args.weights),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "names": model.names,
        "map50": float(validation.box.map50),
        "map50_95": float(validation.box.map),
        "source_prediction_summary": prediction_summary,
        "purpose": "integration-demo-only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
