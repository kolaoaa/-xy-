import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
# Match the Windows GUI bootstrap order: PyTorch must load before Qt widgets.
import torch  # noqa: F401
from PyQt5.QtWidgets import QApplication

from acceptance_evaluator import AcceptanceSessionRecorder
from acceptance_panel import AcceptancePanel
from camera_calibration import Bounds, SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from vision_detector import CapDetection


def build_config() -> SortingConfig:
    return SortingConfig(
        platform_bounds=Bounds(0, 260, 0, 260),
        pick_workspace=Bounds(10, 250, 10, 250),
        bins={},
    )


def detection(class_name: str, confidence: float) -> CapDetection:
    return CapDetection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=(10, 10, 40, 40),
        pixel_x=25,
        pixel_y=25,
    )


class AcceptancePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_records_latest_inference_and_updates_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = AcceptanceSessionRecorder(
                Path(temp_dir),
                clock=lambda: datetime(2026, 6, 2, 17, 30, 0),
            )
            panel = AcceptancePanel(
                config=build_config(),
                camera_ownership=CameraOwnershipCoordinator(),
                recorder=recorder,
            )
            panel.camera_worker = object()
            panel._annotated_frame = np.zeros((80, 120, 3), dtype=np.uint8)
            panel._detections = [detection("red_cap", 0.95)]
            panel.red_cap_radio.setChecked(True)

            panel.record_current_frame()

            self.assertEqual(recorder.stats.total, 1)
            self.assertEqual(recorder.stats.correct, 1)
            self.assertEqual(panel.total_label.text(), "1")
            self.assertEqual(panel.accuracy_label.text(), "100.0%")
            self.assertIn("正确", panel.latest_verdict_label.text())
            self.assertTrue((Path(temp_dir) / "20260602_173000" / "results.csv").exists())
            panel.camera_worker = None
            panel.shutdown()


if __name__ == "__main__":
    unittest.main()
