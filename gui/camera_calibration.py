"""Camera-to-platform homography calibration and sorting configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @classmethod
    def from_dict(cls, values: dict) -> "Bounds":
        return cls(
            x_min=float(values["x_min"]),
            x_max=float(values["x_max"]),
            y_min=float(values["y_min"]),
            y_max=float(values["y_max"]),
        )


@dataclass(frozen=True)
class SortingConfig:
    platform_bounds: Bounds
    pick_workspace: Bounds
    bins: dict[str, tuple[float, float]]
    camera_index: int = 0
    weights_path: str = "weights/bottle_cap_best.pt"
    calibration_path: str = "config/camera_calibration.json"
    confidence_threshold: float = 0.5
    min_spacing_mm: float = 18.0
    processed_radius_mm: float = 12.0
    move_speed_mm_s: int = 10
    pick_wait_s: float = 0.6
    release_wait_s: float = 0.6
    communication_timeout_s: float = 2.0
    dry_run: bool = True
    gripper_configured: bool = False


def load_sorting_config(path: str | Path) -> SortingConfig:
    config_path = Path(path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    bins = {
        class_name: (float(point[0]), float(point[1]))
        for class_name, point in values.get("bins_mm", {}).items()
    }
    gripper = values.get("gripper", values.get("actuator", {}))
    return SortingConfig(
        platform_bounds=Bounds.from_dict(values["platform_bounds_mm"]),
        pick_workspace=Bounds.from_dict(values["pick_workspace_mm"]),
        bins=bins,
        camera_index=int(values.get("camera_index", 0)),
        weights_path=str(values.get("weights_path", "weights/bottle_cap_best.pt")),
        calibration_path=str(values.get("calibration_path", "config/camera_calibration.json")),
        confidence_threshold=float(values.get("confidence_threshold", 0.5)),
        min_spacing_mm=float(values.get("min_spacing_mm", 18.0)),
        processed_radius_mm=float(values.get("processed_radius_mm", 12.0)),
        move_speed_mm_s=int(values.get("move_speed_mm_s", 10)),
        pick_wait_s=float(values.get("pick_wait_s", 0.6)),
        release_wait_s=float(values.get("release_wait_s", 0.6)),
        communication_timeout_s=float(values.get("communication_timeout_s", 2.0)),
        dry_run=bool(values.get("dry_run", True)),
        gripper_configured=bool(gripper.get("configured", False)),
    )


class HomographyCalibration:
    """Maps camera pixels to XY platform millimetres with a 3x3 homography."""

    def __init__(
        self,
        matrix: Iterable[Iterable[float]],
        pixel_points: Sequence[Sequence[float]] | None = None,
        platform_points: Sequence[Sequence[float]] | None = None,
    ):
        self.matrix = np.asarray(matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise ValueError("Homography matrix must have shape (3, 3)")
        if abs(float(np.linalg.det(self.matrix))) < 1e-12:
            raise ValueError("Homography matrix is singular")
        self.pixel_points = [list(map(float, point)) for point in (pixel_points or [])]
        self.platform_points = [list(map(float, point)) for point in (platform_points or [])]

    @classmethod
    def from_points(
        cls,
        pixel_points: Sequence[Sequence[float]],
        platform_points: Sequence[Sequence[float]],
    ) -> "HomographyCalibration":
        if len(pixel_points) != len(platform_points):
            raise ValueError("Pixel and platform point counts must match")
        if len(pixel_points) < 4:
            raise ValueError("At least four calibration points are required")

        rows = []
        results = []
        for pixel, platform in zip(pixel_points, platform_points):
            px, py = map(float, pixel)
            x_mm, y_mm = map(float, platform)
            rows.append([px, py, 1.0, 0.0, 0.0, 0.0, -x_mm * px, -x_mm * py])
            rows.append([0.0, 0.0, 0.0, px, py, 1.0, -y_mm * px, -y_mm * py])
            results.extend([x_mm, y_mm])

        coefficients, _, _, _ = np.linalg.lstsq(
            np.asarray(rows, dtype=float),
            np.asarray(results, dtype=float),
            rcond=None,
        )
        matrix = np.append(coefficients, 1.0).reshape(3, 3)
        return cls(matrix, pixel_points=pixel_points, platform_points=platform_points)

    def pixel_to_platform(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        return self._transform(self.matrix, pixel_x, pixel_y)

    def platform_to_pixel(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        return self._transform(np.linalg.inv(self.matrix), x_mm, y_mm)

    @staticmethod
    def _transform(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        mapped = matrix @ np.asarray([float(x), float(y), 1.0])
        if abs(float(mapped[2])) < 1e-12:
            raise ValueError("Point maps to infinity")
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "homography": self.matrix.tolist(),
                    "pixel_points": self.pixel_points,
                    "platform_points_mm": self.platform_points,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HomographyCalibration":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            values["homography"],
            pixel_points=values.get("pixel_points", []),
            platform_points=values.get("platform_points_mm", []),
        )
