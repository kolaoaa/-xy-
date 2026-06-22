import csv
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from acceptance_evaluator import (
    AcceptanceSessionRecorder,
    AcceptanceStats,
    evaluate_sample,
)


@dataclass(frozen=True)
class FakeDetection:
    class_name: str
    confidence: float


class AcceptanceEvaluatorTests(unittest.TestCase):
    def test_zero_detections_is_missed_detection(self):
        record = evaluate_sample(1, "2026-06-02T17:30:00", "red_cap", [])

        self.assertEqual(record.verdict, "missed_detection")
        self.assertEqual(record.predicted_class, "")
        self.assertIsNone(record.confidence)

    def test_one_matching_detection_is_correct(self):
        record = evaluate_sample(
            1,
            "2026-06-02T17:30:00",
            "red_cap",
            [FakeDetection("red_cap", 0.95)],
        )

        self.assertEqual(record.verdict, "correct")
        self.assertEqual(record.predicted_class, "red_cap")
        self.assertAlmostEqual(record.confidence, 0.95)

    def test_one_mismatching_detection_is_classification_error(self):
        record = evaluate_sample(
            1,
            "2026-06-02T17:30:00",
            "red_cap",
            [FakeDetection("cestbon_cap", 0.88)],
        )

        self.assertEqual(record.verdict, "classification_error")

    def test_multiple_detections_is_multiple_detection_anomaly(self):
        record = evaluate_sample(
            1,
            "2026-06-02T17:30:00",
            "red_cap",
            [
                FakeDetection("red_cap", 0.91),
                FakeDetection("red_cap", 0.52),
            ],
        )

        self.assertEqual(record.verdict, "multiple_detections")
        self.assertEqual(record.detection_count, 2)
        self.assertEqual(record.predicted_class, "")

    def test_stats_accumulate_each_verdict_and_accuracy(self):
        stats = AcceptanceStats()
        verdicts = [
            ("red_cap", [FakeDetection("red_cap", 0.95)]),
            ("red_cap", [FakeDetection("cestbon_cap", 0.85)]),
            ("red_cap", []),
            ("red_cap", [FakeDetection("red_cap", 0.95), FakeDetection("red_cap", 0.60)]),
        ]

        for sequence, (ground_truth, detections) in enumerate(verdicts, start=1):
            stats.add(evaluate_sample(sequence, "2026-06-02T17:30:00", ground_truth, detections))

        self.assertEqual(stats.total, 4)
        self.assertEqual(stats.correct, 1)
        self.assertEqual(stats.classification_errors, 1)
        self.assertEqual(stats.missed_detections, 1)
        self.assertEqual(stats.multiple_detections, 1)
        self.assertAlmostEqual(stats.accuracy, 0.25)

    def test_session_recorder_saves_screenshot_csv_and_statistics(self):
        times = iter(
            [
                datetime(2026, 6, 2, 17, 30, 0),
                datetime(2026, 6, 2, 17, 30, 1),
                datetime(2026, 6, 2, 17, 30, 2),
            ]
        )
        saved_paths = []

        def save_screenshot(path: Path) -> None:
            path.write_bytes(b"jpeg")
            saved_paths.append(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = AcceptanceSessionRecorder(Path(temp_dir), clock=lambda: next(times))
            first = recorder.record("red_cap", [FakeDetection("red_cap", 0.95)], save_screenshot)
            second = recorder.record("cestbon_cap", [], save_screenshot)

            session_dir = Path(temp_dir) / "20260602_173000"
            csv_path = session_dir / "results.csv"
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(first.screenshot_path, "screenshots/0001_red_cap_correct.jpg")
            self.assertEqual(second.screenshot_path, "screenshots/0002_cestbon_cap_missed_detection.jpg")
            self.assertEqual(saved_paths[0], session_dir / first.screenshot_path)
            self.assertTrue(saved_paths[0].exists())
            self.assertEqual(rows[0]["predicted_class"], "red_cap")
            self.assertEqual(rows[0]["confidence"], "0.950000")
            self.assertEqual(rows[1]["confidence"], "")
            self.assertEqual(rows[1]["verdict"], "missed_detection")
            self.assertEqual(recorder.stats.total, 2)
            self.assertEqual(recorder.stats.correct, 1)
            self.assertAlmostEqual(recorder.stats.accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
