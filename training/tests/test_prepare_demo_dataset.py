import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from training.prepare_demo_dataset import (
    bbox_to_yolo,
    clamp_bbox,
    find_red_cap_bbox,
    generate_acceptance_red_samples,
    generate_negative_samples,
    list_image_files,
    overlaps,
)


class DatasetHelperTests(unittest.TestCase):
    def test_bbox_to_yolo_normalizes_center_and_size(self):
        label = bbox_to_yolo((100, 150, 300, 350), width=640, height=640)

        self.assertEqual(label, (0.3125, 0.390625, 0.3125, 0.3125))

    def test_clamp_bbox_keeps_coordinates_inside_image(self):
        bbox = clamp_bbox((-10, 20, 700, 680), width=640, height=640)

        self.assertEqual(bbox, (0, 20, 640, 640))

    def test_overlaps_honors_requested_margin(self):
        self.assertTrue(overlaps((10, 10, 40, 40), (45, 45, 70, 70), margin=8))
        self.assertFalse(overlaps((10, 10, 40, 40), (60, 60, 80, 80), margin=8))

    def test_list_image_files_returns_supported_images_in_name_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (12, 10), "gray").save(root / "b.JPG")
            Image.new("RGB", (12, 10), "gray").save(root / "a.png")
            (root / "notes.txt").write_text("not an image", encoding="utf-8")

            files = list_image_files(root)

        self.assertEqual([path.name for path in files], ["a.png", "b.JPG"])

    def test_generate_negative_samples_creates_empty_yolo_labels(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            negative_dir = root / "negative"
            dataset_dir = root / "dataset"
            negative_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (20 + index, 16 + index), "gray").save(
                    negative_dir / f"negative_{index}.jpg"
                )

            counts = generate_negative_samples(negative_dir, dataset_dir, train_ratio=0.75)

            self.assertEqual(counts, {"train": 3, "val": 1, "total": 4})
            for split, expected_count in (("train", 3), ("val", 1)):
                images = sorted((dataset_dir / "images" / split).glob("negative_*.jpg"))
                labels = sorted((dataset_dir / "labels" / split).glob("negative_*.txt"))
                self.assertEqual(len(images), expected_count)
                self.assertEqual(len(labels), expected_count)
                self.assertTrue(all(label.read_text(encoding="ascii") == "" for label in labels))

    def test_generate_negative_samples_can_repeat_hard_negatives(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            negative_dir = root / "negative"
            dataset_dir = root / "dataset"
            negative_dir.mkdir()
            for index in range(4):
                Image.new("RGB", (20 + index, 16 + index), "gray").save(
                    negative_dir / f"negative_{index}.jpg"
                )

            counts = generate_negative_samples(negative_dir, dataset_dir, train_ratio=0.5, repeats=3)

            self.assertEqual(counts, {"train": 6, "val": 6, "total": 12})
            self.assertEqual(
                len(list((dataset_dir / "images" / "train").glob("negative_*.jpg"))),
                6,
            )
            self.assertEqual(
                len(list((dataset_dir / "images" / "val").glob("negative_*.jpg"))),
                6,
            )

    def test_find_red_cap_bbox_ignores_green_acceptance_overlay(self):
        image = Image.new("RGB", (120, 100), "black")
        pixels = image.load()
        for x in range(24, 58):
            for y in range(31, 67):
                pixels[x, y] = (240, 70, 45)
        for x in range(70, 116):
            pixels[x, 10] = (0, 255, 0)
            pixels[x, 80] = (0, 255, 0)
        for y in range(10, 81):
            pixels[70, y] = (0, 255, 0)
            pixels[115, y] = (0, 255, 0)

        bbox = find_red_cap_bbox(image)

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 24)
        self.assertLessEqual(bbox[1], 31)
        self.assertGreaterEqual(bbox[2], 57)
        self.assertGreaterEqual(bbox[3], 66)
        self.assertLess(bbox[2], 70)

    def test_generate_acceptance_red_samples_uses_missed_red_records_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            acceptance_dir = root / "acceptance"
            screenshots = acceptance_dir / "screenshots"
            dataset_dir = root / "dataset"
            screenshots.mkdir(parents=True)

            for name in ("missed_a.jpg", "missed_b.jpg", "correct.jpg", "cestbon.jpg"):
                image = Image.new("RGB", (100, 80), (30, 40, 35))
                pixels = image.load()
                for x in range(35, 58):
                    for y in range(22, 45):
                        pixels[x, y] = (245, 75, 40)
                image.save(screenshots / name)

            (acceptance_dir / "results.csv").write_text(
                "\n".join(
                    [
                        "sequence,timestamp,ground_truth,predicted_class,confidence,detection_count,verdict,screenshot_path",
                        "1,2026-06-12T13:00:00,red_cap,,,0,missed_detection,screenshots/missed_a.jpg",
                        "2,2026-06-12T13:00:01,red_cap,,,0,missed_detection,screenshots/missed_b.jpg",
                        "3,2026-06-12T13:00:02,red_cap,red_cap,0.9,1,correct,screenshots/correct.jpg",
                        "4,2026-06-12T13:00:03,cestbon_cap,,,0,missed_detection,screenshots/cestbon.jpg",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            counts = generate_acceptance_red_samples(acceptance_dir, dataset_dir, train_ratio=0.5)

            self.assertEqual(counts, {"train": 1, "val": 1, "total": 2, "skipped": 0})
            labels = sorted((dataset_dir / "labels").glob("*/*.txt"))
            self.assertEqual(len(labels), 2)
            for label in labels:
                line = label.read_text(encoding="ascii").strip()
                self.assertTrue(line.startswith("0 "))


if __name__ == "__main__":
    unittest.main()
