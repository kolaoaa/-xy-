"""Build a real-camera YOLO dataset from an acceptance-test session."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from training.prepare_demo_dataset import CLASS_NAMES, bbox_to_yolo


@dataclass(frozen=True)
class CameraSample:
    sequence: int
    timestamp: datetime
    class_name: str
    verdict: str
    source_path: Path
    bbox: tuple[int, int, int, int]
    image_hash: int


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def remove_green_overlays(image: Image.Image) -> Image.Image:
    """Inpaint bright-green detector annotations while preserving the cap."""

    bgr = pil_to_bgr(image)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((35, 90, 80)), np.array((95, 255, 255)))
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    cleaned = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    return bgr_to_pil(cleaned)


def _circle_candidates(
    image: Image.Image,
    roi: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    bgr = pil_to_bgr(image)
    x1, y1, x2, y2 = roi
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.5)
    minimum_dimension = min(crop.shape[:2])
    min_radius = max(7, int(round(minimum_dimension * 0.035)))
    max_radius = max(min_radius + 2, int(round(minimum_dimension * 0.20)))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(24, min_radius * 2),
        param1=80,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    candidates: list[tuple[int, int, int]] = []
    if circles is not None:
        for center_x, center_y, radius in np.round(circles[0]).astype(int):
            candidates.append((center_x + x1, center_y + y1, radius))
    return candidates


def _candidate_color_score(
    bgr: np.ndarray,
    candidate: tuple[int, int, int],
    class_name: str,
) -> float:
    center_x, center_y, radius = candidate
    yy, xx = np.ogrid[: bgr.shape[0], : bgr.shape[1]]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= max(3, radius - 3) ** 2
    pixels = bgr[mask]
    if not len(pixels):
        return -1_000.0
    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    mean_h, mean_s, mean_v = hsv_pixels.mean(axis=0)
    if class_name == "cestbon_cap":
        white_fraction = np.mean((hsv_pixels[:, 1] < 65) & (hsv_pixels[:, 2] > 165))
        return float(white_fraction * 220 + mean_v * 0.55 - mean_s * 0.75)
    if class_name == "red_cap":
        hue = hsv_pixels[:, 0]
        red_fraction = np.mean(
            ((hue < 18) | (hue > 168))
            & (hsv_pixels[:, 1] > 80)
            & (hsv_pixels[:, 2] > 90)
        )
        return float(red_fraction * 260 + mean_s * 0.45 + mean_v * 0.1)
    raise ValueError(f"Unknown class: {class_name}")


def _candidate_score(
    bgr: np.ndarray,
    candidate: tuple[int, int, int],
    class_name: str,
    roi: tuple[int, int, int, int],
) -> float:
    roi_short_side = min(roi[2] - roi[0], roi[3] - roi[1])
    expected_radius = max(22.0, min(28.0, roi_short_side * 0.055))
    radius_penalty = abs(candidate[2] - expected_radius) * 4.0
    return _candidate_color_score(bgr, candidate, class_name) - radius_penalty


def _contour_candidates(
    image: Image.Image,
    class_name: str,
    roi: tuple[int, int, int, int],
) -> list[tuple[int, int, int]]:
    bgr = pil_to_bgr(image)
    x1, y1, x2, y2 = roi
    crop = bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if class_name == "cestbon_cap":
        mask = cv2.inRange(hsv, np.array((0, 0, 165)), np.array((179, 90, 255)))
    else:
        lower_red = cv2.inRange(hsv, np.array((0, 75, 80)), np.array((20, 255, 255)))
        upper_red = cv2.inRange(hsv, np.array((165, 75, 80)), np.array((179, 255, 255)))
        mask = cv2.bitwise_or(lower_red, upper_red)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 120:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if radius < 7 or radius > min(crop.shape[:2]) * 0.22:
            continue
        circle_area = math.pi * radius * radius
        if circle_area <= 0 or area / circle_area < 0.45:
            continue
        candidates.append(
            (int(round(center_x)) + x1, int(round(center_y)) + y1, int(round(radius)))
        )
    return candidates


def find_cap_bbox(
    image: Image.Image,
    class_name: str,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """Find the single expected cap and return a padded pixel bounding box."""

    cleaned = remove_green_overlays(image)
    width, height = cleaned.size
    if roi is None:
        roi = (
            int(round(width * 0.08)),
            int(round(height * 0.08)),
            int(round(width * 0.72)),
            int(round(height * 0.88)),
        )
    x1, y1, x2, y2 = roi
    roi = (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )
    candidates = _circle_candidates(cleaned, roi)
    candidates.extend(_contour_candidates(cleaned, class_name, roi))
    if not candidates:
        return None

    bgr = pil_to_bgr(cleaned)
    best = max(candidates, key=lambda item: _candidate_score(bgr, item, class_name, roi))
    if _candidate_score(bgr, best, class_name, roi) < 45:
        return None
    center_x, center_y, radius = best
    padding = max(3, int(round(radius * 0.16)))
    extent = radius + padding
    return (
        max(0, center_x - extent),
        max(0, center_y - extent),
        min(width, center_x + extent),
        min(height, center_y + extent),
    )


def difference_hash(image: Image.Image, hash_size: int = 8) -> int:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def deduplicate_samples(
    samples: list[CameraSample],
    max_hash_distance: int = 6,
    max_center_distance: float = 2.0,
    max_time_seconds: float = 1.0,
) -> list[CameraSample]:
    """Remove only adjacent near-identical frames, retaining temporal diversity."""

    retained: list[CameraSample] = []
    for sample in sorted(samples, key=lambda item: (item.timestamp, item.sequence)):
        if retained:
            previous = retained[-1]
            time_delta = (sample.timestamp - previous.timestamp).total_seconds()
            previous_center = _bbox_center(previous.bbox)
            current_center = _bbox_center(sample.bbox)
            center_distance = math.dist(previous_center, current_center)
            hash_distance = (sample.image_hash ^ previous.image_hash).bit_count()
            duplicate = (
                sample.class_name == previous.class_name
                and sample.verdict == previous.verdict
                and 0 <= time_delta <= max_time_seconds
                and center_distance <= max_center_distance
                and hash_distance <= max_hash_distance
            )
            if duplicate:
                continue
        retained.append(sample)
    return retained


def _temporal_groups(
    samples: list[CameraSample],
    max_group_size: int,
    max_gap_seconds: float,
) -> list[list[CameraSample]]:
    groups: list[list[CameraSample]] = []
    current: list[CameraSample] = []
    for sample in sorted(samples, key=lambda item: (item.timestamp, item.sequence)):
        new_group = not current
        if current:
            gap = (sample.timestamp - current[-1].timestamp).total_seconds()
            new_group = (
                sample.class_name != current[-1].class_name
                or gap > max_gap_seconds
                or len(current) >= max_group_size
            )
        if new_group:
            if current:
                groups.append(current)
            current = [sample]
        else:
            current.append(sample)
    if current:
        groups.append(current)
    return groups


def assign_temporal_splits(
    samples: list[CameraSample],
    max_group_size: int = 8,
    max_gap_seconds: float = 3.0,
) -> dict[str, list[CameraSample]]:
    """Split each class by whole temporal groups into train, validation, and test."""

    splits: dict[str, list[CameraSample]] = {"train": [], "val": [], "test": []}
    for class_name in CLASS_NAMES:
        class_samples = [sample for sample in samples if sample.class_name == class_name]
        groups = _temporal_groups(class_samples, max_group_size, max_gap_seconds)
        if not groups:
            continue
        if len(groups) < 3:
            if len(class_samples) < 3:
                splits["train"].extend(class_samples)
                continue
            train_end = max(1, min(len(class_samples) - 2, int(len(class_samples) * 0.70)))
            val_end = max(
                train_end + 1,
                min(len(class_samples) - 1, int(len(class_samples) * 0.85)),
            )
            splits["train"].extend(class_samples[:train_end])
            splits["val"].extend(class_samples[train_end:val_end])
            splits["test"].extend(class_samples[val_end:])
            continue
        val_count = max(1, int(round(len(groups) * 0.15)))
        test_count = max(1, int(round(len(groups) * 0.15)))
        while val_count + test_count >= len(groups):
            if test_count > 1:
                test_count -= 1
            elif val_count > 1:
                val_count -= 1
            else:
                break
        train_end = len(groups) - val_count - test_count
        val_end = len(groups) - test_count
        group_splits = {
            "train": groups[:train_end],
            "val": groups[train_end:val_end],
            "test": groups[val_end:],
        }
        for split, selected_groups in group_splits.items():
            splits[split].extend(item for group in selected_groups for item in group)
    for split_samples in splits.values():
        split_samples.sort(key=lambda item: (item.timestamp, item.sequence))
    return splits


def load_acceptance_samples(session_dir: Path) -> tuple[list[CameraSample], list[dict[str, str]]]:
    csv_path = session_dir / "results.csv"
    samples: list[CameraSample] = []
    skipped: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            source_path = session_dir / row["screenshot_path"]
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                cleaned = remove_green_overlays(image)
                bbox = find_cap_bbox(cleaned, row["ground_truth"])
                if bbox is None:
                    skipped.append(
                        {
                            "sequence": row["sequence"],
                            "path": str(source_path),
                            "reason": "cap_not_found",
                        }
                    )
                    continue
                samples.append(
                    CameraSample(
                        sequence=int(row["sequence"]),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        class_name=row["ground_truth"],
                        verdict=row["verdict"],
                        source_path=source_path,
                        bbox=bbox,
                        image_hash=difference_hash(cleaned),
                    )
                )
    return samples, skipped


def _write_sample(sample: CameraSample, split: str, output_dir: Path) -> None:
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    name = f"camera_{sample.sequence:04d}"
    with Image.open(sample.source_path) as source:
        cleaned = remove_green_overlays(ImageOps.exif_transpose(source).convert("RGB"))
        cleaned.save(image_dir / f"{name}.jpg", quality=94)
        normalized = bbox_to_yolo(sample.bbox, cleaned.width, cleaned.height)
    class_id = CLASS_NAMES.index(sample.class_name)
    (label_dir / f"{name}.txt").write_text(
        f"{class_id} {normalized[0]:.6f} {normalized[1]:.6f} "
        f"{normalized[2]:.6f} {normalized[3]:.6f}\n",
        encoding="ascii",
    )


def draw_preview(output_dir: Path, split: str, limit: int = 24) -> None:
    image_paths = sorted((output_dir / "images" / split).glob("*.jpg"))[:limit]
    if not image_paths:
        return
    tile_width, tile_height, columns = 320, 260, 4
    rows = math.ceil(len(image_paths) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        label_path = output_dir / "labels" / split / f"{image_path.stem}.txt"
        class_id_text, x_text, y_text, width_text, height_text = (
            label_path.read_text(encoding="ascii").strip().split()
        )
        center_x = float(x_text) * image.width
        center_y = float(y_text) * image.height
        box_width = float(width_text) * image.width
        box_height = float(height_text) * image.height
        bbox = (
            int(center_x - box_width / 2),
            int(center_y - box_height / 2),
            int(center_x + box_width / 2),
            int(center_y + box_height / 2),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(bbox, outline=(255, 60, 40), width=3)
        draw.text((bbox[0], max(0, bbox[1] - 14)), CLASS_NAMES[int(class_id_text)], fill="red")
        preview = ImageOps.contain(image, (tile_width, tile_height))
        sheet.paste(
            preview,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(preview_dir / f"{split}_preview.jpg", quality=92)


def _yaml_path(path: Path) -> str:
    return path.resolve().as_posix()


def write_dataset_yamls(
    output_dir: Path,
    base_dataset_dir: Path,
    train_yaml: Path,
    test_yaml: Path,
) -> None:
    train_yaml.write_text(
        "\n".join(
            [
                "train:",
                f"  - {_yaml_path(base_dataset_dir / 'images' / 'train')}",
                f"  - {_yaml_path(output_dir / 'images' / 'train')}",
                f"val: {_yaml_path(output_dir / 'images' / 'val')}",
                f"test: {_yaml_path(output_dir / 'images' / 'test')}",
                "names:",
                "  0: red_cap",
                "  1: cestbon_cap",
                "",
            ]
        ),
        encoding="utf-8",
    )
    test_yaml.write_text(
        "\n".join(
            [
                f"path: {_yaml_path(output_dir)}",
                "train: images/train",
                "val: images/test",
                "test: images/test",
                "names:",
                "  0: red_cap",
                "  1: cestbon_cap",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_camera_dataset(
    session_dir: Path,
    output_dir: Path,
    base_dataset_dir: Path,
    train_yaml: Path,
    test_yaml: Path,
) -> dict:
    samples, skipped = load_acceptance_samples(session_dir)
    retained = deduplicate_samples(samples)
    splits = assign_temporal_splits(retained)
    for split, split_samples in splits.items():
        for sample in split_samples:
            _write_sample(sample, split, output_dir)
        draw_preview(output_dir, split)
    write_dataset_yamls(output_dir, base_dataset_dir, train_yaml, test_yaml)
    metadata = {
        "session": str(session_dir),
        "source_rows": len(samples) + len(skipped),
        "labeled_rows": len(samples),
        "retained_rows": len(retained),
        "deduplicated_rows": len(samples) - len(retained),
        "skipped": skipped,
        "splits": {
            split: {
                "total": len(split_samples),
                "classes": {
                    class_name: sum(
                        sample.class_name == class_name for sample in split_samples
                    )
                    for class_name in CLASS_NAMES
                },
            }
            for split, split_samples in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    training_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=training_dir / "generated_dataset_camera_20260618",
    )
    parser.add_argument(
        "--base-dataset-dir",
        type=Path,
        default=training_dir / "generated_dataset_red_acceptance_neg4_20260612",
    )
    parser.add_argument(
        "--train-yaml",
        type=Path,
        default=training_dir / "dataset_camera_20260618.yaml",
    )
    parser.add_argument(
        "--test-yaml",
        type=Path,
        default=training_dir / "dataset_camera_20260618_test.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_camera_dataset(
        args.session_dir,
        args.output_dir,
        args.base_dataset_dir,
        args.train_yaml,
        args.test_yaml,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
