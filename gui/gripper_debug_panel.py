"""Manual gripper debug panel."""

from __future__ import annotations

from PyQt5.QtWidgets import QGridLayout, QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget


class GripperDebugPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        group = QGroupBox("夹子调试")
        grid = QGridLayout(group)

        grid.addWidget(QLabel("PWM"), 0, 0)
        grid.addWidget(QLabel("J9 SERVO1 / PA0"), 0, 1, 1, 2)

        open_btn = QPushButton("张开夹子")
        open_btn.clicked.connect(self.open_gripper)
        grid.addWidget(open_btn, 1, 0)

        close_btn = QPushButton("闭合夹子")
        close_btn.clicked.connect(self.close_gripper)
        grid.addWidget(close_btn, 1, 1)

        middle_btn = QPushButton("中位")
        middle_btn.clicked.connect(self.middle_gripper)
        grid.addWidget(middle_btn, 1, 2)

        self.status_label = QLabel("-")
        self.status_label.setWordWrap(True)
        grid.addWidget(QLabel("状态"), 2, 0)
        grid.addWidget(self.status_label, 2, 1, 1, 2)

        layout.addWidget(group)
        layout.addStretch()

    def open_gripper(self) -> None:
        self._run_command("张开", self.controller.servo_open)

    def close_gripper(self) -> None:
        self._run_command("闭合", self.controller.servo_close)

    def middle_gripper(self) -> None:
        self._run_command("中位", self.controller.servo_middle)

    def _run_command(self, label: str, command) -> None:
        ok = bool(command())
        self.status_label.setText(f"{label}: {'已发送' if ok else '发送失败'}")
