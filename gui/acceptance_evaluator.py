"""Strict single-cap acceptance verdicts and saved evidence."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence


CSV_FIELDS = (
    "sequence",
    "timestamp",
    "ground_truth",
    "predicted_class",
    "confidence",
    "detection_count",
    "verdict",
    "screenshot_path",
)


@dataclass(frozen=True)
class AcceptanceRecord:
    sequence: int
    timestamp: str
    ground_truth: str
    predicted_class: str
    confidence: float | None
    detection_count: int
    verdict: str
    screenshot_path: str = ""

    def csv_row(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "ground_truth": self.ground_truth,
            "predicted_class": self.predicted_class,
            "confidence": "" if self.confidence is None else f"{self.confidence:.6f}",
            "detection_count": self.detection_count,
            "verdict": self.verdict,
            "screenshot_path": self.screenshot_path,
        }


@dataclass
class AcceptanceStats:
    total: int = 0
    correct: int = 0
    classification_errors: int = 0
    missed_detections: int = 0
    multiple_detections: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def add(self, record: AcceptanceRecord) -> None:
        self.total += 1
        if record.verdict == "correct":
            self.correct += 1
        elif record.verdict == "classification_error":
            self.classification_errors += 1
        elif record.verdict == "missed_detection":
            self.missed_detections += 1
        elif record.verdict == "multiple_detections":
            self.multiple_detections += 1
        else:
            raise ValueError(f"Unknown acceptance verdict: {record.verdict}")


def evaluate_sample(
    sequence: int,
    timestamp: str,
    ground_truth: str,
    detections: Sequence[object],
) -> AcceptanceRecord:
    detection_count = len(detections)
    predicted_class = ""
    confidence = None

    if detection_count == 0:
        verdict = "missed_detection"
    elif detection_count > 1:
        verdict = "multiple_detections"
    else:
        detection = detections[0]
        predicted_class = str(detection.class_name)
        confidence = float(detection.confidence)
        verdict = "correct" if predicted_class == ground_truth else "classification_error"

    return AcceptanceRecord(
        sequence=int(sequence),
        timestamp=str(timestamp),
        ground_truth=str(ground_truth),
        predicted_class=predicted_class,
        confidence=confidence,
        detection_count=detection_count,
        verdict=verdict,
    )


class AcceptanceSessionRecorder:
    """Creates one timestamped result folder and appends recorded samples."""

    def __init__(
        self,
        output_root: str | Path,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.output_root = Path(output_root)
        self.clock = clock
        self.stats = AcceptanceStats()
        self.session_dir: Path | None = None

    def record(
        self,
        ground_truth: str,
        detections: Sequence[object],
        save_screenshot: Callable[[Path], None],
    ) -> AcceptanceRecord:
        now = self.clock()
        session_dir = self._ensure_session_dir(now)
        sequence = self.stats.total + 1
        record = evaluate_sample(
            sequence=sequence,
            timestamp=now.isoformat(timespec="seconds"),
            ground_truth=ground_truth,
            detections=detections,
        )
        screenshot_path = Path("screenshots") / self._screenshot_filename(record)
        absolute_screenshot_path = session_dir / screenshot_path
        absolute_screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        save_screenshot(absolute_screenshot_path)

        record = replace(record, screenshot_path=screenshot_path.as_posix())
        self._append_csv(record)
        self.stats.add(record)
        return record

    def _ensure_session_dir(self, now: datetime) -> Path:
        if self.session_dir is None:
            self.session_dir = self.output_root / now.strftime("%Y%m%d_%H%M%S")
            self.session_dir.mkdir(parents=True, exist_ok=True)
        return self.session_dir

    @staticmethod
    def _screenshot_filename(record: AcceptanceRecord) -> str:
        ground_truth = record.ground_truth.replace("/", "_").replace("\\", "_")
        return f"{record.sequence:04d}_{ground_truth}_{record.verdict}.jpg"

    def _append_csv(self, record: AcceptanceRecord) -> None:
        if self.session_dir is None:
            raise RuntimeError("Acceptance session directory is not initialized")
        csv_path = self.session_dir / "results.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record.csv_row())

