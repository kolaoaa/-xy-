"""Create a synthetic two-class bottle-cap dataset for integration testing."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


CLASS_NAMES = ("red_cap", "cestbon_cap")
IMAGE_SIZE = 640
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class SourceSpec:
    class_name: str
    filename: str
    center_x: float
    center_y: float
    radius: float


SOURCE_SPECS = (
    SourceSpec("red_cap", "188caad522a9850195b8177fd20b04d0.jpg", 596.4, 933.3, 177.0),
    SourceSpec("red_cap", "2e931841645c31b6e7fa6b8e23b07c92.jpg", 880.6, 586.4, 204.0),
    SourceSpec("red_cap", "73f6eabda832987d0b8b398b1fae9888.jpg", 670.3, 846.4, 168.0),
    SourceSpec("red_cap", "8891048558f981b9bd3ea63ea4b31d4e.jpg", 497.5, 952.8, 202.0),
    SourceSpec("red_cap", "9ed2ca056f15077b1743d63cee4ae3c0.jpg", 662.9, 839.3, 181.0),
    SourceSpec("cestbon_cap", "41ca39f87d1730f5fc3bbf47ba3f4e3a.jpg", 416.0, 854.0, 160.0),
    SourceSpec("cestbon_cap", "a54544d82d9d387f5fe4563148fb34ad.jpg", 407.0, 806.0, 145.0),
    SourceSpec("cestbon_cap", "b5fc615d2b42f092be7223538b4bc633.jpg", 428.0, 862.0, 158.0),
    SourceSpec("cestbon_cap", "e4f0941792b477535a4671662a287d88.jpg", 1074.0, 494.0, 215.0),
    SourceSpec("cestbon_cap", "f1b19a3b55e7fa51c39b6e4ec2de3da0.jpg", 376.0, 678.0, 154.0),
)


def clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def overlaps(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    return not (
        left[2] + margin <= right[0]
        or right[2] + margin <= left[0]
        or left[3] + margin <= right[1]
        or right[3] + margin <= left[1]
    )


def source_path(spec: SourceSpec, red_dir: Path, cestbon_dir: Path) -> Path:
    folder = red_dir if spec.class_name == "red_cap" else cestbon_dir
    return folder / spec.filename


def extract_cap(source: Path, spec: SourceSpec) -> Image.Image:
    image = Image.open(source).convert("RGB")
    padding = int(math.ceil(spec.radius * 1.08))
    left = int(round(spec.center_x - padding))
    top = int(round(spec.center_y - padding))
    right = int(round(spec.center_x + padding))
    bottom = int(round(spec.center_y + padding))
    crop = image.crop((left, top, right, bottom)).convert("RGBA")

    mask = Image.new("L", crop.size, 0)
    draw = ImageDraw.Draw(mask)
    inset = max(1, int(round(padding - spec.radius)))
    draw.ellipse((inset, inset, crop.width - inset, crop.height - inset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1.0, spec.radius * 0.015)))
    crop.putalpha(mask)
    return crop


def transform_cap(cap: Image.Image, rng: random.Random) -> Image.Image:
    transformed = ImageEnhance.Brightness(cap).enhance(rng.uniform(0.82, 1.18))
    transformed = ImageEnhance.Contrast(transformed).enhance(rng.uniform(0.88, 1.14))
    transformed = transformed.rotate(
        rng.uniform(0, 360),
        resample=Image.Resampling.BICUBIC,
        expand=True,
    )
    target_size = rng.randint(54, 142)
    transformed.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    if rng.random() < 0.16:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.2)))
    return transformed


def create_background(rng: random.Random, size: int = IMAGE_SIZE) -> Image.Image:
    base = np.asarray(
        [
            rng.randint(160, 232),
            rng.randint(160, 232),
            rng.randint(160, 232),
        ],
        dtype=np.float32,
    )
    noise = np.random.default_rng(rng.randint(0, 2**32 - 1)).normal(0, rng.uniform(2, 9), (size, size, 1))
    x_gradient = np.linspace(rng.uniform(-24, 0), rng.uniform(0, 24), size, dtype=np.float32)[None, :, None]
    y_gradient = np.linspace(rng.uniform(-14, 0), rng.uniform(0, 14), size, dtype=np.float32)[:, None, None]
    array = np.clip(base + noise + x_gradient + y_gradient, 0, 255).astype(np.uint8)
    array = np.repeat(array, 3, axis=2) if array.shape[2] == 1 else array
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    if rng.random() < 0.35:
        y = rng.randint(50, size - 50)
        draw.line((0, y, size, y + rng.randint(-18, 18)), fill=(145, 145, 145), width=rng.randint(1, 3))
    return image


def place_cap(
    background: Image.Image,
    cap: Image.Image,
    rng: random.Random,
    existing_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    for _ in range(80):
        x1 = rng.randint(12, background.width - cap.width - 12)
        y1 = rng.randint(12, background.height - cap.height - 12)
        bbox = (x1, y1, x1 + cap.width, y1 + cap.height)
        if any(overlaps(bbox, other, margin=16) for other in existing_boxes):
            continue
        background.paste(cap, (x1, y1), cap)
        return bbox
    return None


def write_label(path: Path, labels: list[tuple[int, tuple[int, int, int, int]]]) -> None:
    lines = []
    for class_id, bbox in labels:
        x_center, y_center, width, height = bbox_to_yolo(bbox, IMAGE_SIZE, IMAGE_SIZE)
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(directory)
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.lower(),
    )


def generate_split(
    split: str,
    count: int,
    sources: dict[str, list[Image.Image]],
    output_dir: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)
    image_dir = output_dir / "images" / split
    label_dir = output_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image = create_background(rng)
        labels: list[tuple[int, tuple[int, int, int, int]]] = []
        boxes: list[tuple[int, int, int, int]] = []
        object_count = rng.randint(1, 4)
        for object_index in range(object_count):
            class_id = (index + object_index + rng.randint(0, 1)) % len(CLASS_NAMES)
            class_name = CLASS_NAMES[class_id]
            cap = transform_cap(rng.choice(sources[class_name]), rng)
            bbox = place_cap(image, cap, rng, boxes)
            if bbox is None:
                continue
            boxes.append(bbox)
            labels.append((class_id, bbox))

        name = f"{split}_{index:04d}"
        image.save(image_dir / f"{name}.jpg", quality=90)
        write_label(label_dir / f"{name}.txt", labels)


def generate_context_samples(
    specs: tuple[SourceSpec, ...],
    red_dir: Path,
    cestbon_dir: Path,
    output_dir: Path,
    variants_per_source: int = 8,
) -> None:
    """Keep real background and shadow context from the provided phone photos."""

    rng = random.Random(23182712)
    image_dir = output_dir / "images" / "train"
    label_dir = output_dir / "labels" / "train"
    for source_index, spec in enumerate(specs):
        original = Image.open(source_path(spec, red_dir, cestbon_dir)).convert("RGB")
        class_id = CLASS_NAMES.index(spec.class_name)
        for variant in range(variants_per_source):
            image = ImageEnhance.Brightness(original).enhance(rng.uniform(0.82, 1.18))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.86, 1.14))
            center_x = spec.center_x
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
                center_x = image.width - center_x
            if rng.random() < 0.18:
                image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
            bbox = clamp_bbox(
                (
                    int(center_x - spec.radius),
                    int(spec.center_y - spec.radius),
                    int(center_x + spec.radius),
                    int(spec.center_y + spec.radius),
                ),
                image.width,
                image.height,
            )
            name = f"context_{source_index:02d}_{variant:02d}"
            image.save(image_dir / f"{name}.jpg", quality=90)
            normalized = bbox_to_yolo(bbox, image.width, image.height)
            (label_dir / f"{name}.txt").write_text(
                f"{class_id} {normalized[0]:.6f} {normalized[1]:.6f} {normalized[2]:.6f} {normalized[3]:.6f}\n",
                encoding="ascii",
            )


def generate_negative_samples(
    negative_dir: Path | None,
    output_dir: Path,
    train_ratio: float = 0.8,
    repeats: int = 1,
) -> dict[str, int]:
    """Add no-cap images as YOLO negative samples with empty label files."""

    if negative_dir is None:
        return {"train": 0, "val": 0, "total": 0}
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    image_paths = list_image_files(negative_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in {negative_dir}")

    shuffled = list(image_paths)
    random.Random(23182713).shuffle(shuffled)
    if len(shuffled) == 1:
        train_count = 1
    else:
        train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))

    split_images: dict[str, list[Path]] = {
        "train": [path for path in shuffled[:train_count] for _ in range(repeats)],
        "val": [path for path in shuffled[train_count:] for _ in range(repeats)],
    }
    sequence = 0
    for split, paths in split_images.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source_path_value in paths:
            name = f"negative_{sequence:04d}"
            with Image.open(source_path_value) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.save(image_dir / f"{name}.jpg", quality=92)
            (label_dir / f"{name}.txt").write_text("", encoding="ascii")
            sequence += 1

    return {
        "train": len(split_images["train"]),
        "val": len(split_images["val"]),
        "total": len(image_paths) * repeats,
    }


def find_red_cap_bbox(image: Image.Image, min_area: int = 80) -> tuple[int, int, int, int] | None:
    """Find the dominant red/orange cap blob while ignoring green GUI overlays."""

    array = np.asarray(image.convert("RGB"))
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    mask = (
        (red >= 130)
        & ((red - green) >= 35)
        & ((red - blue) >= 35)
        & (green <= 190)
        & (blue <= 180)
    ).astype(np.uint8)

    try:
        import cv2
    except ImportError:
        ys, xs = np.where(mask > 0)
        if len(xs) < min_area:
            return None
        return _padded_bbox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, image.width, image.height)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best: tuple[int, int, int, int, int] | None = None
    for label in range(1, labels_count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < min_area or width < 8 or height < 8:
            continue
        aspect = width / height
        if aspect < 0.45 or aspect > 2.2:
            continue
        if best is None or area > best[4]:
            best = (x, y, x + width, y + height, area)

    if best is None:
        return None
    return _padded_bbox(best[0], best[1], best[2], best[3], image.width, image.height)


def _padded_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    padding = max(4, int(round(max(x2 - x1, y2 - y1) * 0.18)))
    return clamp_bbox((x1 - padding, y1 - padding, x2 + padding, y2 + padding), width, height)


def generate_acceptance_red_samples(
    acceptance_dir: Path | None,
    output_dir: Path,
    train_ratio: float = 0.8,
) -> dict[str, int]:
    """Convert clean missed red-cap acceptance screenshots into YOLO samples."""

    if acceptance_dir is None:
        return {"train": 0, "val": 0, "total": 0, "skipped": 0}

    csv_path = acceptance_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    candidates: list[tuple[Path, tuple[int, int, int, int]]] = []
    skipped = 0
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("ground_truth") != "red_cap" or row.get("verdict") != "missed_detection":
                continue
            image_path = acceptance_dir / str(row.get("screenshot_path", ""))
            if not image_path.exists():
                skipped += 1
                continue
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                bbox = find_red_cap_bbox(image)
            if bbox is None:
                skipped += 1
                continue
            candidates.append((image_path, bbox))

    if not candidates:
        return {"train": 0, "val": 0, "total": 0, "skipped": skipped}

    shuffled = list(candidates)
    random.Random(23182714).shuffle(shuffled)
    if len(shuffled) == 1:
        train_count = 1
    else:
        train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * train_ratio))))

    split_samples = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count:],
    }
    sequence = 0
    for split, samples in split_samples.items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, bbox in samples:
            name = f"acceptance_red_{sequence:04d}"
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.save(image_dir / f"{name}.jpg", quality=92)
            normalized = bbox_to_yolo(bbox, image.width, image.height)
            (label_dir / f"{name}.txt").write_text(
                f"0 {normalized[0]:.6f} {normalized[1]:.6f} {normalized[2]:.6f} {normalized[3]:.6f}\n",
                encoding="ascii",
            )
            sequence += 1

    return {
        "train": len(split_samples["train"]),
        "val": len(split_samples["val"]),
        "total": len(candidates),
        "skipped": skipped,
    }


def remove_yolo_cache_files(output_dir: Path) -> None:
    for cache_path in (
        output_dir / "labels" / "train.cache",
        output_dir / "labels" / "val.cache",
    ):
        if cache_path.exists():
            cache_path.unlink()


def draw_preview(dataset_dir: Path, split: str, output_path: Path, limit: int = 16) -> None:
    images = sorted((dataset_dir / "images" / split).glob("*.jpg"))[:limit]
    tile_size = 240
    columns = 4
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), "white")
    for index, image_path in enumerate(images):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
        for line in label_path.read_text(encoding="ascii").splitlines():
            class_id_text, x_text, y_text, width_text, height_text = line.split()
            class_id = int(class_id_text)
            x_center = float(x_text) * image.width
            y_center = float(y_text) * image.height
            box_width = float(width_text) * image.width
            box_height = float(height_text) * image.height
            bbox = (
                int(x_center - box_width / 2),
                int(y_center - box_height / 2),
                int(x_center + box_width / 2),
                int(y_center + box_height / 2),
            )
            color = (220, 40, 40) if class_id == 0 else (20, 155, 90)
            draw.rectangle(bbox, outline=color, width=4)
            draw.text((bbox[0], max(0, bbox[1] - 14)), CLASS_NAMES[class_id], fill=color)
        sheet.paste(image.resize((tile_size, tile_size), Image.Resampling.LANCZOS), ((index % columns) * tile_size, (index // columns) * tile_size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def write_dataset_yaml(dataset_dir: Path, output_path: Path) -> None:
    text = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: red_cap\n"
        "  1: cestbon_cap\n"
    )
    output_path.write_text(text, encoding="utf-8")


def build_dataset(
    red_dir: Path,
    cestbon_dir: Path,
    output_dir: Path,
    negative_dir: Path | None = None,
    acceptance_red_dir: Path | None = None,
    negative_repeats: int = 1,
) -> None:
    remove_yolo_cache_files(output_dir)
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Image.Image]] = {name: [] for name in CLASS_NAMES}

    for spec in SOURCE_SPECS:
        path = source_path(spec, red_dir, cestbon_dir)
        if not path.exists():
            raise FileNotFoundError(path)
        crop = extract_cap(path, spec)
        grouped[spec.class_name].append(crop)
        crop.save(extracted_dir / f"{spec.class_name}_{path.stem}.png")

    train_sources = {name: images[:4] for name, images in grouped.items()}
    val_sources = {name: images[4:] for name, images in grouped.items()}
    generate_split("train", 640, train_sources, output_dir, seed=42)
    generate_split("val", 128, val_sources, output_dir, seed=43)
    generate_context_samples(SOURCE_SPECS, red_dir, cestbon_dir, output_dir)
    negative_counts = generate_negative_samples(negative_dir, output_dir, repeats=negative_repeats)
    acceptance_red_counts = generate_acceptance_red_samples(acceptance_red_dir, output_dir)
    draw_preview(output_dir, "train", output_dir / "previews" / "train_preview.jpg")
    draw_preview(output_dir, "val", output_dir / "previews" / "val_preview.jpg")
    write_dataset_yaml(output_dir, output_dir.parent / "dataset.yaml")
    train_images = 720 + negative_counts["train"] + acceptance_red_counts["train"]
    val_images = 128 + negative_counts["val"] + acceptance_red_counts["val"]
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "classes": list(CLASS_NAMES),
                "train_images": train_images,
                "val_images": val_images,
                "negative_images": negative_counts["total"],
                "train_negative_images": negative_counts["train"],
                "val_negative_images": negative_counts["val"],
                "acceptance_red_images": acceptance_red_counts["total"],
                "train_acceptance_red_images": acceptance_red_counts["train"],
                "val_acceptance_red_images": acceptance_red_counts["val"],
                "skipped_acceptance_red_images": acceptance_red_counts["skipped"],
                "source_images": len(SOURCE_SPECS),
                "purpose": "integration-demo-with-real-samples"
                if negative_counts["total"] or acceptance_red_counts["total"]
                else "integration-demo-only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--red-dir", type=Path, required=True)
    parser.add_argument("--cestbon-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path)
    parser.add_argument("--acceptance-red-dir", type=Path)
    parser.add_argument("--negative-repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated_dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.red_dir,
        args.cestbon_dir,
        args.output_dir,
        args.negative_dir,
        args.acceptance_red_dir,
        args.negative_repeats,
    )
    print(f"Generated dataset: {args.output_dir}")


if __name__ == "__main__":
    main()
