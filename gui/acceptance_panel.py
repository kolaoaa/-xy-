"""Computer-webcam acceptance page for the demonstration bottle-cap detector."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from acceptance_evaluator import AcceptanceRecord, AcceptanceSessionRecorder
from calibration_dialog import frame_to_pixmap
from camera_calibration import SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from camera_worker import CameraWorker, VisionWorker
from vision_detector import YoloCapDetector


CLASS_LABELS = {
    "red_cap": "红色瓶盖",
    "cestbon_cap": "怡宝瓶盖",
}

VERDICT_LABELS = {
    "correct": "正确",
    "classification_error": "分类错误",
    "missed_detection": "漏检",
    "multiple_detections": "多框异常",
}


class AcceptancePanel(QWidget):
    CAMERA_OWNER = "acceptance_test"

    def __init__(
        self,
        config: SortingConfig,
        camera_ownership: CameraOwnershipCoordinator,
        recorder: AcceptanceSessionRecorder | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.config = config
        self.camera_ownership = camera_ownership
        self.gui_dir = Path(__file__).resolve().parent
        self.weights_path = self.gui_dir / config.weights_path
        self.detector = YoloCapDetector(
            self.weights_path,
            confidence_threshold=config.confidence_threshold,
        )
        self.recorder = recorder or AcceptanceSessionRecorder(self.gui_dir / "acceptance_results")
        self.camera_worker = None
        self.vision_worker = None
        self._annotated_frame = None
        self._detections = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        body.addWidget(self._build_preview_group(), 3)
        body.addWidget(self._build_statistics_group(), 1)
        layout.addLayout(body, 1)
        layout.addWidget(self._build_recent_group())

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("电脑摄像头实时识别")
        layout = QVBoxLayout(group)

        self.camera_view = QLabel("摄像头未启动")
        self.camera_view.setAlignment(Qt.AlignCenter)
        self.camera_view.setMinimumSize(720, 420)
        self.camera_view.setStyleSheet("background: #202020; color: #dddddd;")
        layout.addWidget(self.camera_view, 1)

        camera_controls = QHBoxLayout()
        camera_controls.addWidget(QLabel("摄像头编号:"))
        self.camera_index_input = QSpinBox()
        self.camera_index_input.setRange(0, 10)
        self.camera_index_input.setValue(self.config.camera_index)
        self.camera_index_input.setToolTip("0 通常是笔记本内置摄像头，1 或更高通常是外接摄像头")
        camera_controls.addWidget(self.camera_index_input)
        self.camera_start_btn = QPushButton("启动摄像头")
        self.camera_start_btn.clicked.connect(self.start_camera)
        camera_controls.addWidget(self.camera_start_btn)
        self.camera_stop_btn = QPushButton("停止摄像头")
        self.camera_stop_btn.clicked.connect(self.stop_camera)
        self.camera_stop_btn.setEnabled(False)
        camera_controls.addWidget(self.camera_stop_btn)
        camera_controls.addStretch()
        layout.addLayout(camera_controls)

        truth_group = QGroupBox("真实类别")
        truth_layout = QHBoxLayout(truth_group)
        self.red_cap_radio = QRadioButton("红色瓶盖")
        self.cestbon_cap_radio = QRadioButton("怡宝瓶盖")
        truth_layout.addWidget(self.red_cap_radio)
        truth_layout.addWidget(self.cestbon_cap_radio)
        truth_layout.addStretch()
        layout.addWidget(truth_group)

        self.record_btn = QPushButton("记录当前帧")
        self.record_btn.clicked.connect(self.record_current_frame)
        self.record_btn.setMinimumHeight(42)
        layout.addWidget(self.record_btn)
        return group

    def _build_statistics_group(self) -> QGroupBox:
        group = QGroupBox("验收统计")
        layout = QGridLayout(group)
        layout.addWidget(QLabel("摄像头状态:"), 0, 0)
        self.camera_status_label = QLabel("未启动")
        layout.addWidget(self.camera_status_label, 0, 1)
        layout.addWidget(QLabel("当前识别:"), 1, 0)
        self.current_detection_label = QLabel("-")
        self.current_detection_label.setWordWrap(True)
        layout.addWidget(self.current_detection_label, 1, 1)
        layout.addWidget(QLabel("最近判定:"), 2, 0)
        self.latest_verdict_label = QLabel("-")
        self.latest_verdict_label.setWordWrap(True)
        layout.addWidget(self.latest_verdict_label, 2, 1)
        layout.addWidget(QLabel("累计样本:"), 3, 0)
        self.total_label = QLabel("0")
        layout.addWidget(self.total_label, 3, 1)
        layout.addWidget(QLabel("准确率:"), 4, 0)
        self.accuracy_label = QLabel("0.0%")
        layout.addWidget(self.accuracy_label, 4, 1)
        layout.addWidget(QLabel("正确:"), 5, 0)
        self.correct_label = QLabel("0")
        layout.addWidget(self.correct_label, 5, 1)
        layout.addWidget(QLabel("漏检:"), 6, 0)
        self.missed_label = QLabel("0")
        layout.addWidget(self.missed_label, 6, 1)
        layout.addWidget(QLabel("多框异常:"), 7, 0)
        self.multiple_label = QLabel("0")
        layout.addWidget(self.multiple_label, 7, 1)
        layout.addWidget(QLabel("分类错误:"), 8, 0)
        self.classification_error_label = QLabel("0")
        layout.addWidget(self.classification_error_label, 8, 1)
        layout.addWidget(QLabel("异常提示:"), 9, 0)
        self.error_label = QLabel("-")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label, 9, 1)
        layout.setRowStretch(10, 1)
        return group

    def _build_recent_group(self) -> QGroupBox:
        group = QGroupBox("最近记录")
        layout = QVBoxLayout(group)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(110)
        layout.addWidget(self.recent_list)
        self.result_path_label = QLabel("保存目录: 尚未记录样本")
        self.result_path_label.setWordWrap(True)
        layout.addWidget(self.result_path_label)
        return group

    def start_camera(self) -> None:
        if self.camera_worker is not None:
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
        self.camera_start_btn.setEnabled(True)
        self.camera_stop_btn.setEnabled(False)
        self.camera_status_label.setText("未启动")
        self.camera_view.setText("摄像头未启动")

    def record_current_frame(self) -> None:
        if self.camera_worker is None:
            QMessageBox.warning(self, "无法记录", "请先启动摄像头")
            return
        if self._annotated_frame is None:
            QMessageBox.warning(self, "无法记录", "请等待识别画面")
            return
        ground_truth = self._selected_ground_truth()
        if ground_truth is None:
            QMessageBox.warning(self, "无法记录", "请先选择真实类别")
            return

        frame = self._annotated_frame.copy()
        try:
            record = self.recorder.record(
                ground_truth=ground_truth,
                detections=list(self._detections),
                save_screenshot=lambda path: self._save_screenshot(path, frame),
            )
        except Exception as exc:
            self._show_error(str(exc))
            QMessageBox.critical(self, "保存失败", str(exc))
            return
        self._refresh_after_record(record)

    def shutdown(self) -> None:
        self.stop_camera()

    def _on_frame_ready(self, frame) -> None:
        self._set_preview(frame)
        if self.vision_worker is not None:
            self.vision_worker.submit_frame(frame)

    def _on_detection_result(self, annotated, detections) -> None:
        self._annotated_frame = annotated.copy()
        self._detections = list(detections)
        self.current_detection_label.setText(self._describe_detections(self._detections))
        self._set_preview(annotated)

    def _on_camera_status(self, started: bool) -> None:
        self.camera_index_input.setEnabled(not started)
        self.camera_start_btn.setEnabled(not started)
        self.camera_stop_btn.setEnabled(started)
        self.camera_status_label.setText("运行中" if started else "未启动")
        if not started:
            self.camera_ownership.release(self.CAMERA_OWNER)
            if self.vision_worker is not None:
                self.vision_worker.stop()
                self.vision_worker = None
            self.camera_worker = None

    def _selected_camera_index(self) -> int:
        return int(self.camera_index_input.value())

    def _selected_ground_truth(self) -> str | None:
        if self.red_cap_radio.isChecked():
            return "red_cap"
        if self.cestbon_cap_radio.isChecked():
            return "cestbon_cap"
        return None

    @staticmethod
    def _describe_detections(detections) -> str:
        if not detections:
            return "未检测到瓶盖"
        if len(detections) > 1:
            return f"检测到 {len(detections)} 个框"
        detection = detections[0]
        label = CLASS_LABELS.get(detection.class_name, detection.class_name)
        return f"{label}，置信度 {detection.confidence:.2f}"

    def _refresh_after_record(self, record: AcceptanceRecord) -> None:
        verdict = VERDICT_LABELS[record.verdict]
        truth = CLASS_LABELS.get(record.ground_truth, record.ground_truth)
        predicted = CLASS_LABELS.get(record.predicted_class, record.predicted_class or "-")
        self.latest_verdict_label.setText(f"{verdict}：真实 {truth}，预测 {predicted}")

        stats = self.recorder.stats
        self.total_label.setText(str(stats.total))
        self.accuracy_label.setText(f"{stats.accuracy * 100:.1f}%")
        self.correct_label.setText(str(stats.correct))
        self.missed_label.setText(str(stats.missed_detections))
        self.multiple_label.setText(str(stats.multiple_detections))
        self.classification_error_label.setText(str(stats.classification_errors))
        self.error_label.setText("-")

        self.recent_list.insertItem(0, f"#{record.sequence:04d}  {truth}  {verdict}")
        while self.recent_list.count() > 8:
            self.recent_list.takeItem(self.recent_list.count() - 1)
        self.result_path_label.setText(f"保存目录: {self.recorder.session_dir}")

    def _set_preview(self, frame) -> None:
        pixmap = frame_to_pixmap(frame)
        self.camera_view.setPixmap(
            pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)

    @staticmethod
    def _save_screenshot(path: Path, frame) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV 未安装，无法保存截图") from exc
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("截图编码失败")
        encoded.tofile(str(path))
