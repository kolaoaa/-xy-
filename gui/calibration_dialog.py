"""Click-based camera homography calibration dialog."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def frame_to_pixmap(frame) -> QPixmap:
    rgb = frame[:, :, ::-1].copy()
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ClickableImageLabel(QLabel):
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self._source_pixmap = None
        self._display_pixmap = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._refresh_scaled_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh_scaled_pixmap()

    def mousePressEvent(self, event) -> None:
        if self._source_pixmap is None or self._display_pixmap is None:
            return
        left = (self.width() - self._display_pixmap.width()) / 2.0
        top = (self.height() - self._display_pixmap.height()) / 2.0
        x = event.pos().x() - left
        y = event.pos().y() - top
        if not (0 <= x < self._display_pixmap.width() and 0 <= y < self._display_pixmap.height()):
            return
        source_x = x * self._source_pixmap.width() / self._display_pixmap.width()
        source_y = y * self._source_pixmap.height() / self._display_pixmap.height()
        self.clicked.emit(float(source_x), float(source_y))

    def _refresh_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self._display_pixmap = self._source_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(self._display_pixmap)


class CalibrationDialog(QDialog):
    def __init__(self, frame, platform_points, on_move, on_save, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相机标定")
        self.resize(900, 680)
        self.platform_points = list(platform_points)
        self.pixel_points = []
        self.on_move = on_move
        self.on_save = on_save

        layout = QVBoxLayout(self)
        self.instruction = QLabel()
        layout.addWidget(self.instruction)

        self.image_label = ClickableImageLabel()
        self.image_label.setMinimumSize(720, 480)
        self.image_label.set_source_pixmap(frame_to_pixmap(frame))
        self.image_label.clicked.connect(self._on_image_clicked)
        layout.addWidget(self.image_label, 1)

        controls = QHBoxLayout()
        self.move_btn = QPushButton("移动到当前标定点")
        self.move_btn.clicked.connect(self._move_current_point)
        controls.addWidget(self.move_btn)
        reset_btn = QPushButton("重新标定")
        reset_btn.clicked.connect(self._reset)
        controls.addWidget(reset_btn)
        controls.addStretch()
        save_btn = QPushButton("保存标定")
        save_btn.clicked.connect(self._save)
        controls.addWidget(save_btn)
        layout.addLayout(controls)
        self._refresh_instruction()

    def _move_current_point(self) -> None:
        if len(self.pixel_points) >= len(self.platform_points):
            return
        self.on_move(*self.platform_points[len(self.pixel_points)])

    def _on_image_clicked(self, pixel_x: float, pixel_y: float) -> None:
        if len(self.pixel_points) >= len(self.platform_points):
            return
        self.pixel_points.append((pixel_x, pixel_y))
        self._refresh_instruction()

    def _reset(self) -> None:
        self.pixel_points.clear()
        self._refresh_instruction()

    def _save(self) -> None:
        if len(self.pixel_points) < 4:
            QMessageBox.warning(self, "标定点不足", "至少需要采集 4 个标定点")
            return
        self.on_save(self.pixel_points, self.platform_points[: len(self.pixel_points)])
        self.accept()

    def _refresh_instruction(self) -> None:
        index = len(self.pixel_points)
        if index >= len(self.platform_points):
            self.instruction.setText("标定点已采集完成，请保存标定参数")
            self.move_btn.setEnabled(False)
            return
        x_mm, y_mm = self.platform_points[index]
        self.instruction.setText(
            f"第 {index + 1}/{len(self.platform_points)} 点：先移动平台到 "
            f"({x_mm:.1f}, {y_mm:.1f}) mm，等待停止后点击图像中的末端中心"
        )
        self.move_btn.setEnabled(True)
