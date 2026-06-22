import unittest
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw

from training.prepare_camera_dataset import (
    CameraSample,
    assign_temporal_splits,
    deduplicate_samples,
    difference_hash,
    find_cap_bbox,
    remove_green_overlays,
)


class CameraDatasetTests(unittest.TestCase):
    def test_remove_green_overlays_removes_annotation_without_erasing_cap(self):
        image = Image.new("RGB", (180, 140), (35, 55, 48))
        draw = ImageDraw.Draw(image)
        draw.ellipse((72, 50, 112, 90), fill=(245, 245, 240))
        draw.rectangle((65, 43, 119, 97), outline=(0, 255, 0), width=3)
        draw.text((65, 28), "cestbon 0.75", fill=(0, 255, 0))

        cleaned = remove_green_overlays(image)

        self.assertLess(cleaned.getpixel((66, 44))[1], 180)
        self.assertGreater(sum(cleaned.getpixel((92, 70))), 650)

    def test_find_cap_bbox_selects_white_cap_over_brown_distractor(self):
        image = Image.new("RGB", (220, 160), (28, 48, 42))
        draw = ImageDraw.Draw(image)
        draw.ellipse((82, 58, 124, 100), fill=(250, 250, 245))
        draw.ellipse((162, 42, 202, 82), fill=(155, 125, 100))

        bbox = find_cap_bbox(image, "cestbon_cap", roi=(20, 20, 210, 145))

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 82)
        self.assertLessEqual(bbox[1], 58)
        self.assertGreaterEqual(bbox[2], 124)
        self.assertGreaterEqual(bbox[3], 100)
        self.assertLess(bbox[2], 150)

    def test_find_cap_bbox_prefers_camera_sized_cap_over_small_white_marker(self):
        image = Image.new("RGB", (220, 160), (28, 48, 42))
        draw = ImageDraw.Draw(image)
        draw.ellipse((82, 58, 124, 100), fill=(235, 240, 234))
        draw.ellipse((162, 25, 190, 53), fill=(255, 255, 255))

        bbox = find_cap_bbox(image, "cestbon_cap", roi=(20, 20, 210, 145))

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLess(bbox[0], 100)
        self.assertGreater(bbox[2], 110)
        self.assertGreater(bbox[1], 45)

    def test_find_cap_bbox_selects_red_cap(self):
        image = Image.new("RGB", (220, 160), (28, 48, 42))
        draw = ImageDraw.Draw(image)
        draw.ellipse((70, 52, 116, 98), fill=(235, 72, 35))
        draw.ellipse((150, 42, 192, 84), fill=(250, 250, 245))

        bbox = find_cap_bbox(image, "red_cap", roi=(20, 20, 210, 145))

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLessEqual(bbox[0], 70)
        self.assertGreaterEqual(bbox[2], 116)
        self.assertLess(bbox[2], 140)

    def test_deduplicate_samples_drops_near_identical_adjacent_frames(self):
        base = Image.new("RGB", (64, 48), "gray")
        changed = base.copy()
        ImageDraw.Draw(changed).rectangle((5, 5, 30, 30), fill="white")
        start = datetime(2026, 6, 18, 16, 0, 0)
        samples = [
            self._sample(1, start, base, (20, 15, 40, 35)),
            self._sample(2, start + timedelta(milliseconds=400), base, (21, 15, 41, 35)),
            self._sample(3, start + timedelta(seconds=3), base, (20, 15, 40, 35)),
            self._sample(4, start + timedelta(seconds=4), changed, (20, 15, 40, 35)),
        ]

        retained = deduplicate_samples(
            samples,
            max_hash_distance=2,
            max_center_distance=3.0,
            max_time_seconds=1.0,
        )

        self.assertEqual([sample.sequence for sample in retained], [1, 3, 4])

    def test_assign_temporal_splits_keeps_whole_groups_together(self):
        start = datetime(2026, 6, 18, 16, 0, 0)
        image = Image.new("RGB", (64, 48), "gray")
        samples = [
            self._sample(index, start + timedelta(seconds=index), image, (20, 15, 40, 35))
            for index in range(1, 31)
        ]

        splits = assign_temporal_splits(samples, max_group_size=5, max_gap_seconds=3.0)

        membership = {
            sample.sequence: split
            for split, split_samples in splits.items()
            for sample in split_samples
        }
        for group_start in range(1, 31, 5):
            group_memberships = {
                membership[index]
                for index in range(group_start, min(group_start + 5, 31))
            }
            self.assertEqual(len(group_memberships), 1)
        self.assertGreater(len(splits["train"]), 0)
        self.assertGreater(len(splits["val"]), 0)
        self.assertGreater(len(splits["test"]), 0)

    def test_assign_temporal_splits_holds_out_sparse_class_samples(self):
        start = datetime(2026, 6, 18, 16, 0, 0)
        image = Image.new("RGB", (64, 48), "gray")
        samples = [
            self._sample(index, start + timedelta(seconds=index), image, (20, 15, 40, 35))
            for index in range(1, 10)
        ]

        splits = assign_temporal_splits(samples)

        for split in ("train", "val", "test"):
            self.assertGreater(len(splits[split]), 0)

    @staticmethod
    def _sample(
        sequence: int,
        timestamp: datetime,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> CameraSample:
        return CameraSample(
            sequence=sequence,
            timestamp=timestamp,
            class_name="cestbon_cap",
            verdict="missed_detection",
            source_path=Path(f"{sequence}.jpg"),
            bbox=bbox,
            image_hash=difference_hash(image),
        )


if __name__ == "__main__":
    unittest.main()
