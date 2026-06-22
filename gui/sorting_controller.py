"""Timer-driven bottle-cap sorting scheduler."""

from __future__ import annotations

import time
from collections import Counter
from enum import Enum
from typing import Callable

from camera_calibration import Bounds
from vision_detector import CapDetection, filter_detections, order_targets_nearest_first


class SortState(str, Enum):
    IDLE = "IDLE"
    HOMING = "HOMING"
    CAPTURE_IMAGE = "CAPTURE_IMAGE"
    DETECT_CAPS = "DETECT_CAPS"
    SELECT_TARGET = "SELECT_TARGET"
    MOVE_TO_PICK = "MOVE_TO_PICK"
    GRIPPER_CLOSE = "GRIPPER_CLOSE"
    WAIT_PICK = "WAIT_PICK"
    MOVE_TO_BIN = "MOVE_TO_BIN"
    GRIPPER_OPEN = "GRIPPER_OPEN"
    WAIT_RELEASE = "WAIT_RELEASE"
    NEXT_TARGET = "NEXT_TARGET"
    FINISHED = "FINISHED"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class ReservedGripper:
    """Placeholder gripper used when the servo hardware is not configured."""

    def is_configured(self) -> bool:
        return False

    def close(self) -> bool:
        return False

    def open(self) -> bool:
        return True

    def middle(self) -> bool:
        return True


class UsbGripper:
    """USB-backed gripper using the existing platform controller."""

    def __init__(self, motion):
        self.motion = motion

    def is_configured(self) -> bool:
        return hasattr(self.motion, "servo_open") and hasattr(self.motion, "servo_close")

    def close(self) -> bool:
        return bool(self.motion.servo_close())

    def open(self) -> bool:
        return bool(self.motion.servo_open())

    def middle(self) -> bool:
        if hasattr(self.motion, "servo_middle"):
            return bool(self.motion.servo_middle())
        return True


class SortingController:
    def __init__(
        self,
        motion,
        vision,
        gripper,
        bins: dict[str, tuple[float, float]],
        platform_bounds: Bounds,
        pick_workspace: Bounds,
        speed_mm_s: int = 10,
        confidence_threshold: float = 0.5,
        min_spacing_mm: float = 18.0,
        processed_radius_mm: float = 12.0,
        pick_wait_s: float = 0.6,
        release_wait_s: float = 0.6,
        communication_timeout_s: float = 2.0,
        move_reached_tolerance_mm: float = 2.0,
        dry_run: bool = False,
        monotonic: Callable[[], float] | None = None,
        on_update: Callable[[dict], None] | None = None,
    ):
        self.motion = motion
        self.vision = vision
        self.gripper = gripper
        self.bins = dict(bins)
        self.platform_bounds = platform_bounds
        self.pick_workspace = pick_workspace
        self.speed_mm_s = int(speed_mm_s)
        self.confidence_threshold = float(confidence_threshold)
        self.min_spacing_mm = float(min_spacing_mm)
        self.processed_radius_mm = float(processed_radius_mm)
        self.pick_wait_s = float(pick_wait_s)
        self.release_wait_s = float(release_wait_s)
        self.communication_timeout_s = float(communication_timeout_s)
        self.move_reached_tolerance_mm = float(move_reached_tolerance_mm)
        self.dry_run = bool(dry_run)
        self._monotonic = monotonic or time.monotonic
        self._on_update = on_update
        self.state = SortState.IDLE
        self.error_message = ""
        self.current_target: CapDetection | None = None
        self.last_frame = None
        self.last_detections: list[CapDetection] = []
        self.pending_targets: list[CapDetection] = []
        self.processed_points: list[tuple[float, float]] = []
        self.sorted_counts: Counter[str] = Counter()
        self._resume_state = SortState.IDLE
        self._wait_started = self._monotonic()
        self._last_communication = self._monotonic()
        self._move_target: tuple[float, float] | None = None

    def start(self) -> None:
        if not self.dry_run and not self.gripper.is_configured():
            self._fail("Gripper is not configured")
            return
        self.error_message = ""
        self.current_target = None
        self.last_frame = None
        self.last_detections = []
        self.pending_targets = []
        self.processed_points = []
        self._move_target = None
        self.sorted_counts.clear()
        self.note_communication()
        self._set_state(SortState.CAPTURE_IMAGE)

    def tick(self) -> None:
        if self.state in (SortState.IDLE, SortState.FINISHED, SortState.ERROR):
            return
        if self._monotonic() - self._last_communication > self.communication_timeout_s:
            self._fail("Communication timeout")
            return
        if self.state == SortState.PAUSED:
            return

        if self.state == SortState.HOMING and self.motion.is_idle():
            self._set_state(SortState.CAPTURE_IMAGE)
        elif self.state == SortState.CAPTURE_IMAGE:
            self.last_frame = self.vision.capture()
            if self.last_frame is None:
                self._fail("Camera capture failed")
            else:
                self._set_state(SortState.DETECT_CAPS)
        elif self.state == SortState.DETECT_CAPS:
            detected = self.vision.detect(self.last_frame)
            filtered = filter_detections(
                detected,
                confidence_threshold=self.confidence_threshold,
                workspace=self.pick_workspace,
                processed_points=self.processed_points,
                processed_radius_mm=self.processed_radius_mm,
                min_spacing_mm=self.min_spacing_mm,
            )
            self.last_detections = list(filtered)
            self.pending_targets = order_targets_nearest_first(
                filtered,
                current_position=self._current_position(),
            )
            self._set_state(SortState.SELECT_TARGET)
        elif self.state == SortState.SELECT_TARGET:
            if not self.pending_targets:
                self._set_state(SortState.FINISHED)
                return
            self.current_target = self.pending_targets.pop(0)
            if not self._move_to(self.current_target.platform_x_mm, self.current_target.platform_y_mm):
                return
            self._set_state(SortState.MOVE_TO_PICK)
        elif self.state == SortState.MOVE_TO_PICK and self._move_finished_at_target():
            self._set_state(SortState.GRIPPER_CLOSE)
        elif self.state == SortState.GRIPPER_CLOSE:
            if not self.dry_run and not self.gripper.close():
                self._fail("Failed to close gripper")
                return
            self._wait_started = self._monotonic()
            self._set_state(SortState.WAIT_PICK)
        elif self.state == SortState.WAIT_PICK:
            if self._monotonic() - self._wait_started >= self.pick_wait_s:
                bin_point = self._bin_for_current_target()
                if bin_point is None or not self._move_to(*bin_point):
                    return
                self._set_state(SortState.MOVE_TO_BIN)
        elif self.state == SortState.MOVE_TO_BIN and self._move_finished_at_target():
            self._set_state(SortState.GRIPPER_OPEN)
        elif self.state == SortState.GRIPPER_OPEN:
            if not self.dry_run and not self.gripper.open():
                self._fail("Failed to open gripper")
                return
            self._wait_started = self._monotonic()
            self._set_state(SortState.WAIT_RELEASE)
        elif self.state == SortState.WAIT_RELEASE:
            if self._monotonic() - self._wait_started >= self.release_wait_s:
                self._record_current_target()
                self._set_state(SortState.NEXT_TARGET)
        elif self.state == SortState.NEXT_TARGET:
            self.current_target = None
            self._set_state(SortState.SELECT_TARGET if self.pending_targets else SortState.CAPTURE_IMAGE)

    def pause(self) -> None:
        if self.state not in (SortState.IDLE, SortState.FINISHED, SortState.ERROR, SortState.PAUSED):
            self._resume_state = self.state
            self._set_state(SortState.PAUSED)

    def resume(self) -> None:
        if self.state == SortState.PAUSED:
            self.note_communication()
            self._set_state(self._resume_state)

    def stop(self) -> None:
        self.motion.stop()
        self.gripper.open()
        self.current_target = None
        self.pending_targets = []
        self._move_target = None
        self._set_state(SortState.IDLE)

    def note_communication(self) -> None:
        self._last_communication = self._monotonic()

    def report_platform_error(self, message: str) -> None:
        self._fail(message)

    def snapshot(self) -> dict:
        target = None
        if self.current_target is not None:
            target = (
                self.current_target.platform_x_mm,
                self.current_target.platform_y_mm,
            )
        return {
            "state": self.state.value,
            "current_target": target,
            "detections": self._detection_summaries(self.last_detections),
            "pending_targets": self._detection_summaries(self.pending_targets),
            "planned_path": self._planned_path(),
            "sorted_counts": dict(self.sorted_counts),
            "sorted_total": sum(self.sorted_counts.values()),
            "error": self.error_message,
            "dry_run": self.dry_run,
        }

    def _move_to(self, x_mm: float | None, y_mm: float | None) -> bool:
        if x_mm is None or y_mm is None:
            self._fail("Target has no platform coordinates")
            return False
        if not self.platform_bounds.contains(x_mm, y_mm):
            self._fail(f"Target is outside platform bounds: ({x_mm:.2f}, {y_mm:.2f})")
            return False
        if not self.motion.move_abs(x_mm, y_mm, self.speed_mm_s):
            self._fail("Failed to send movement command")
            return False
        self._move_target = (float(x_mm), float(y_mm))
        return True

    def _move_finished_at_target(self) -> bool:
        if not self.motion.is_idle():
            return False
        if self._move_target is None:
            return True
        current_x, current_y = self._current_position()
        target_x, target_y = self._move_target
        return (
            abs(current_x - target_x) <= self.move_reached_tolerance_mm
            and abs(current_y - target_y) <= self.move_reached_tolerance_mm
        )

    def _bin_for_current_target(self) -> tuple[float, float] | None:
        if self.current_target is None:
            self._fail("No active target")
            return None
        point = self.bins.get(self.current_target.class_name, self.bins.get("unknown"))
        if point is None:
            self._fail(f"No bin configured for class: {self.current_target.class_name}")
            return None
        return point

    def _record_current_target(self) -> None:
        if self.current_target is None:
            return
        self.sorted_counts[self.current_target.class_name] += 1
        if self.current_target.platform_x_mm is not None and self.current_target.platform_y_mm is not None:
            self.processed_points.append(
                (self.current_target.platform_x_mm, self.current_target.platform_y_mm)
            )

    @staticmethod
    def _point_for_detection(detection: CapDetection) -> tuple[float, float] | None:
        if detection.platform_x_mm is None or detection.platform_y_mm is None:
            return None
        return (float(detection.platform_x_mm), float(detection.platform_y_mm))

    def _planned_path(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = [self._current_position()]
        ordered_targets: list[CapDetection] = []
        if self.current_target is not None:
            ordered_targets.append(self.current_target)
        ordered_targets.extend(self.pending_targets)

        for target in ordered_targets:
            pick_point = self._point_for_detection(target)
            bin_point = self.bins.get(target.class_name, self.bins.get("unknown"))
            if pick_point is not None:
                points.append(pick_point)
            if bin_point is not None:
                points.append((float(bin_point[0]), float(bin_point[1])))
        return points

    def _detection_summaries(self, detections: Iterable[CapDetection]) -> list[dict]:
        summaries = []
        for detection in detections:
            point = self._point_for_detection(detection)
            if point is None:
                continue
            summaries.append(
                {
                    "class_name": detection.class_name,
                    "confidence": float(detection.confidence),
                    "x": point[0],
                    "y": point[1],
                }
            )
        return summaries

    def _current_position(self) -> tuple[float, float]:
        if hasattr(self.motion, "current_position"):
            return tuple(map(float, self.motion.current_position()))
        return (0.0, 0.0)

    def _fail(self, message: str) -> None:
        self.error_message = message
        self.motion.stop()
        self.gripper.open()
        self._move_target = None
        self._set_state(SortState.ERROR)

    def _set_state(self, state: SortState) -> None:
        self.state = state
        if self._on_update is not None:
            self._on_update(self.snapshot())
