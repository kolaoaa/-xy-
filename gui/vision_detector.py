"""YOLO bottle-cap detection normalization and target filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from camera_calibration import Bounds, HomographyCalibration


@dataclass(frozen=True)
class CapDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    pixel_x: float
    pixel_y: float
    platform_x_mm: float | None = None
    platform_y_mm: float | None = None

    def with_platform_coordinates(self, calibration: HomographyCalibration) -> "CapDetection":
        x_mm, y_mm = calibration.pixel_to_platform(self.pixel_x, self.pixel_y)
        return replace(self, platform_x_mm=x_mm, platform_y_mm=y_mm)


def _distance_mm(
    detection: CapDetection,
    point: tuple[float, float],
) -> float:
    if detection.platform_x_mm is None or detection.platform_y_mm is None:
        return math.inf
    return math.hypot(detection.platform_x_mm - point[0], detection.platform_y_mm - point[1])


def filter_detections(
    detections: Iterable[CapDetection],
    confidence_threshold: float,
    workspace: Bounds,
    processed_points: Sequence[tuple[float, float]] = (),
    processed_radius_mm: float = 12.0,
    min_spacing_mm: float = 18.0,
) -> list[CapDetection]:
    """Keep only reachable, unprocessed caps and retain the best of close neighbours."""

    candidates = []
    for detection in detections:
        if detection.confidence < confidence_threshold:
            continue
        if detection.platform_x_mm is None or detection.platform_y_mm is None:
            continue
        if not workspace.contains(detection.platform_x_mm, detection.platform_y_mm):
            continue
        if any(_distance_mm(detection, point) < processed_radius_mm for point in processed_points):
            continue
        candidates.append(detection)

    kept = []
    for detection in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(
            _distance_mm(detection, (item.platform_x_mm, item.platform_y_mm)) < min_spacing_mm
            for item in kept
        ):
            continue
        kept.append(detection)
    return kept


def order_targets_nearest_first(
    detections: Iterable[CapDetection],
    current_position: tuple[float, float],
) -> list[CapDetection]:
    return sorted(detections, key=lambda item: _distance_mm(item, current_position))


class YoloCapDetector:
    """Lazy Ultralytics wrapper so configuration and tests work without model dependencies."""

    def __init__(
        self,
        weights_path: str | Path,
        confidence_threshold: float = 0.5,
        calibration: HomographyCalibration | None = None,
    ):
        self.weights_path = str(weights_path)
        self.confidence_threshold = float(confidence_threshold)
        self.calibration = calibration
        self._model = None

    def set_calibration(self, calibration: HomographyCalibration | None) -> None:
        self.calibration = calibration

    def _ensure_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from exc
            self._model = YOLO(self.weights_path)
        return self._model

    def detect(self, frame) -> list[CapDetection]:
        model = self._ensure_model()
        results = model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        for result in results:
            names = result.names
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detection = CapDetection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, "unknown")),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    pixel_x=(x1 + x2) / 2.0,
                    pixel_y=(y1 + y2) / 2.0,
                )
                if self.calibration is not None:
                    detection = detection.with_platform_coordinates(self.calibration)
                detections.append(detection)
        return detections

    @staticmethod
    def annotate(frame, detections: Iterable[CapDetection]):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python") from exc

        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            label = f"{detection.class_name} {detection.confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (60, 220, 80), 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 220, 80),
                2,
                cv2.LINE_AA,
            )
        return annotated
