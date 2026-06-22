"""Bottle-cap sorting operator panel integrated with the existing XY GUI."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from calibration_dialog import CalibrationDialog, frame_to_pixmap
from camera_calibration import HomographyCalibration, SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from camera_worker import CameraWorker, LatestVisionSource, VisionWorker
from sorting_controller import ReservedGripper, SortState, SortingController, UsbGripper
from vision_detector import YoloCapDetector


class SortingPanel(QWidget):
    CAMERA_OWNER = "visual_sorting"

    def __init__(
        self,
        motion_controller,
        config: SortingConfig,
        camera_ownership: CameraOwnershipCoordinator | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.motion = motion_controller
        self.config = config
        self.camera_ownership = camera_ownership or CameraOwnershipCoordinator()
        self.gui_dir = Path(__file__).resolve().parent
        self.calibration_path = self.gui_dir / config.calibration_path
        self.weights_path = self.gui_dir / config.weights_path
        self.calibration = self._load_calibration()
        self.vision_source = LatestVisionSource()
        self.detector = YoloCapDetector(
            self.weights_path,
            confidence_threshold=config.confidence_threshold,
            calibration=self.calibration,
        )
        self.camera_worker = None
        self.vision_worker = None
        self.gripper = UsbGripper(self.motion) if config.gripper_configured else ReservedGripper()
        self.sorting = SortingController(
            motion=self.motion,
            vision=self.vision_source,
            gripper=self.gripper,
            bins=config.bins,
            platform_bounds=config.platform_bounds,
            pick_workspace=config.pick_workspace,
            speed_mm_s=config.move_speed_mm_s,
            confidence_threshold=config.confidence_threshold,
            min_spacing_mm=config.min_spacing_mm,
            processed_radius_mm=config.processed_radius_mm,
            pick_wait_s=config.pick_wait_s,
            release_wait_s=config.release_wait_s,
            communication_timeout_s=config.communication_timeout_s,
            dry_run=config.dry_run,
            on_update=self._on_sorting_update,
        )
        self._build_ui()
        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self._tick)
        self.tick_timer.start(50)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        camera_group = QGroupBox("瓶盖视觉")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_view = QLabel("摄像头未启动")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(640, 260)
        self.camera_view.setStyleSheet("background: #202020; color: #dddddd;")
        camera_layout.addWidget(self.camera_view)

        camera_controls = QHBoxLayout()
        camera_controls.addWidget(QLabel("摄像头编号:"))
        self.camera_index_input = QSpinBox()
        self.camera_index_input.setRange(0, 10)
        self.camera_index_input.setValue(self.config.camera_index)
        self.camera_index_input.setToolTip("0 通常是笔记本内置摄像头，1 或更高通常是外接摄像头")
        camera_controls.addWidget(self.camera_index_input)
        self.camera_btn = QPushButton("启动摄像头")
        self.camera_btn.clicked.connect(self.toggle_camera)
        camera_controls.addWidget(self.camera_btn)
        calibrate_btn = QPushButton("标定")
        calibrate_btn.clicked.connect(self.open_calibration)
        camera_controls.addWidget(calibrate_btn)
        camera_controls.addStretch()
        self.detection_label = QLabel("检测: 0")
        camera_controls.addWidget(self.detection_label)
        camera_layout.addLayout(camera_controls)
        layout.addWidget(camera_group)

        task_group = QGroupBox("瓶盖分类")
        task_layout = QGridLayout(task_group)
        self.start_btn = QPushButton("开始分类")
        self.start_btn.clicked.connect(self.start_sorting)
        task_layout.addWidget(self.start_btn, 0, 0)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        task_layout.addWidget(self.pause_btn, 0, 1)
        stop_btn = QPushButton("停止")
        stop_btn.clicked.connect(self.stop_sorting)
        task_layout.addWidget(stop_btn, 0, 2)
        task_layout.addWidget(QLabel("模式:"), 1, 0)
        task_layout.addWidget(QLabel("移动演练（夹子未接入）" if self.config.dry_run else "真实抓取"), 1, 1, 1, 2)
        task_layout.addWidget(QLabel("任务状态:"), 2, 0)
        self.task_state_label = QLabel(SortState.IDLE.value)
        task_layout.addWidget(self.task_state_label, 2, 1, 1, 2)
        task_layout.addWidget(QLabel("当前目标:"), 3, 0)
        self.target_label = QLabel("-")
        task_layout.addWidget(self.target_label, 3, 1, 1, 2)
        task_layout.addWidget(QLabel("已分类:"), 4, 0)
        self.total_label = QLabel("0")
        task_layout.addWidget(self.total_label, 4, 1, 1, 2)
        task_layout.addWidget(QLabel("分类统计:"), 5, 0)
        self.stats_label = QLabel("-")
        task_layout.addWidget(self.stats_label, 5, 1, 1, 2)
        task_layout.addWidget(QLabel("异常:"), 6, 0)
        self.error_label = QLabel("-")
        self.error_label.setWordWrap(True)
        task_layout.addWidget(self.error_label, 6, 1, 1, 2)
        layout.addWidget(task_group)

        gripper_group = QGroupBox("夹子")
        gripper_layout = QHBoxLayout(gripper_group)
        gripper_layout.addWidget(QLabel("SERVO1 / PA0" if self.config.gripper_configured else "尚未绑定硬件"))
        gripper_layout.addStretch()
        gripper_open = QPushButton("张开")
        gripper_open.setEnabled(self.config.gripper_configured)
        gripper_open.clicked.connect(self.gripper.open)
        gripper_layout.addWidget(gripper_open)
        gripper_close = QPushButton("闭合")
        gripper_close.setEnabled(self.config.gripper_configured)
        gripper_close.clicked.connect(self.gripper.close)
        gripper_layout.addWidget(gripper_close)
        gripper_middle = QPushButton("中位")
        gripper_middle.setEnabled(self.config.gripper_configured)
        gripper_middle.clicked.connect(self.gripper.middle)
        gripper_layout.addWidget(gripper_middle)
        layout.addWidget(gripper_group)

    def toggle_camera(self) -> None:
        if self.camera_worker is not None:
            self.stop_camera()
            return
        if not self.camera_ownership.acquire(self.CAMERA_OWNER):
            QMessageBox.warning(self, "摄像头正忙", "请先停止另一个页面中的摄像头")
            return
        self.camera_index_input.setEnabled(False)
        self.camera_worker = CameraWorker(self._selected_camera_index(), self)
        self.vision_worker = VisionWorker(self.detector, self)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.camera_status.connect(self._on_camera_status)
        self.camera_worker.error_occurred.connect(self._show_error)
        self.vision_worker.result_ready.connect(self._on_detection_result)
        self.vision_worker.error_occurred.connect(self._show_error)
        self.vision_worker.start()
        self.camera_worker.start()

    def stop_camera(self) -> None:
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker = None
        if self.vision_worker is not None:
            self.vision_worker.stop()
            self.vision_worker = None
        self.camera_ownership.release(self.CAMERA_OWNER)
        self.camera_index_input.setEnabled(True)
        self.camera_btn.setText("启动摄像头")
        self.camera_view.setText("摄像头未启动")

    def open_calibration(self) -> None:
        frame = self.vision_source.capture()
        if frame is None:
            QMessageBox.warning(self, "无法标定", "请先启动摄像头并等待画面")
            return
        dialog = CalibrationDialog(
            frame,
            platform_points=self.calibration_platform_points(),
            on_move=lambda x, y: self.motion.move_abs(x, y, self.config.move_speed_mm_s),
            on_save=self._save_calibration,
            parent=self,
        )
        dialog.exec_()

    def calibration_platform_points(self) -> list[tuple[float, float]]:
        workspace = self.config.pick_workspace
        return [
            (workspace.x_min, workspace.y_min),
            (workspace.x_max, workspace.y_min),
            (workspace.x_max, workspace.y_max),
            (workspace.x_min, workspace.y_max),
        ]

    def start_readiness(self) -> tuple[bool, str]:
        if hasattr(self.motion, "is_zeroed") and not self.motion.is_zeroed():
            return False, "请先回零或手动设零"
        if hasattr(self.motion, "is_idle") and not self.motion.is_idle():
            return False, "请等待平台空闲。回零或手动设零后，请确认平台状态已经变为“空闲”。"
        if self.calibration is None:
            return False, "请先完成相机标定。能识别到物体不等于已经完成像素到平台坐标的转换。"
        if self.vision_source.capture() is None:
            return False, "请先启动摄像头并等待画面"
        return True, ""

    def start_sorting(self) -> None:
        ready, reason = self.start_readiness()
        if not ready:
            self._reject_start(reason)
            return
        self.sorting.start()

    def _reject_start(self, reason: str) -> None:
        self.error_label.setText(reason)
        self._log_sorting_event(f"无法开始: {reason}")
        QMessageBox.warning(self, "无法开始", reason)

    def toggle_pause(self) -> None:
        if self.sorting.state == SortState.PAUSED:
            self.sorting.resume()
            self.pause_btn.setText("暂停")
        else:
            self.sorting.pause()
            self.pause_btn.setText("继续")

    def stop_sorting(self) -> None:
        self.sorting.stop()
        self.pause_btn.setText("暂停")

    def on_platform_status(self, status_info: dict) -> None:
        self.sorting.note_communication()
        error_code = int(status_info.get("error", 0))
        status = int(status_info.get("status", 0))
        if status == 0xFF or error_code != 0:
            self.sorting.report_platform_error(f"平台错误，代码: {error_code}")

    def shutdown(self) -> None:
        self.sorting.stop()
        self.stop_camera()

    def _tick(self) -> None:
        if self.sorting.state not in (SortState.IDLE, SortState.FINISHED, SortState.ERROR):
            self.motion.query_status(silent=True)
        self.sorting.tick()

    def _on_frame_ready(self, frame) -> None:
        self.vision_source.update_frame(frame)
        self._set_preview(frame)
        if self.vision_worker is not None:
            self.vision_worker.submit_frame(frame)

    def _on_detection_result(self, annotated, detections) -> None:
        self.vision_source.update_detections(detections)
        self.detection_label.setText(f"检测: {len(detections)}")
        self._set_preview(annotated)
        self._publish_overlay(detections=detections)
        if detections:
            items = []
            for detection in detections:
                if detection.platform_x_mm is None or detection.platform_y_mm is None:
                    continue
                items.append(
                    f"{detection.class_name} ({detection.platform_x_mm:.1f}, {detection.platform_y_mm:.1f}) "
                    f"{detection.confidence:.2f}"
                )
            if items:
                self._log_sorting_event("识别: " + "; ".join(items))

    def _on_camera_status(self, started: bool) -> None:
        self.camera_index_input.setEnabled(not started)
        self.camera_btn.setText("停止摄像头" if started else "启动摄像头")
        if not started:
            self.camera_ownership.release(self.CAMERA_OWNER)
            if self.vision_worker is not None:
                self.vision_worker.stop()
                self.vision_worker = None
            self.camera_worker = None

    def _selected_camera_index(self) -> int:
        return int(self.camera_index_input.value())

    def _save_calibration(self, pixel_points, platform_points) -> None:
        self.calibration = HomographyCalibration.from_points(pixel_points, platform_points)
        self.calibration.save(self.calibration_path)
        self.detector.set_calibration(self.calibration)
        QMessageBox.information(self, "标定完成", f"标定参数已保存到 {self.calibration_path}")

    def _load_calibration(self):
        if not self.calibration_path.exists():
            return None
        try:
            return HomographyCalibration.load(self.calibration_path)
        except Exception:
            return None

    def _set_preview(self, frame) -> None:
        pixmap = frame_to_pixmap(frame)
        self.camera_view.setPixmap(
            pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)

    def _on_sorting_update(self, snapshot: dict) -> None:
        self.task_state_label.setText(snapshot["state"])
        target = snapshot["current_target"]
        self.target_label.setText("-" if target is None else f"({target[0]:.1f}, {target[1]:.1f}) mm")
        self.total_label.setText(str(snapshot["sorted_total"]))
        stats = snapshot["sorted_counts"]
        self.stats_label.setText(", ".join(f"{name}: {count}" for name, count in sorted(stats.items())) or "-")
        self.error_label.setText(snapshot["error"] or "-")
        self._publish_overlay(
            detections=snapshot.get("detections", []),
            planned_path=snapshot.get("planned_path", []),
            current_target=snapshot.get("current_target"),
        )
        path = snapshot.get("planned_path", [])
        if path:
            self._log_sorting_event(f"状态: {snapshot['state']} 预定路径点数: {len(path)}")

    def _publish_overlay(self, detections=None, planned_path=None, current_target=None) -> None:
        window = self.window()
        if hasattr(window, "update_sorting_overlay"):
            window.update_sorting_overlay(
                detections=detections,
                planned_path=planned_path,
                bins=self.config.bins,
                current_target=current_target,
            )

    def _log_sorting_event(self, message: str) -> None:
        window = self.window()
        if hasattr(window, "on_log_message"):
            window.on_log_message(message)
