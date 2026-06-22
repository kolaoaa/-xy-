import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from camera_calibration import Bounds, SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from sorting_panel import SortingPanel


class FakeMotion:
    def __init__(self, zeroed=True, idle=True):
        self._zeroed = zeroed
        self._idle = idle

    def is_zeroed(self):
        return self._zeroed

    def is_idle(self):
        return self._idle

    def query_status(self, silent=False):
        return True

    def stop(self):
        return True


class SortingStartReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def build_panel(self, *, zeroed=True, idle=True, calibrated=True, camera_ready=True):
        config = SortingConfig(
            platform_bounds=Bounds(0, 300, 0, 300),
            pick_workspace=Bounds(10, 290, 10, 290),
            bins={},
        )
        panel = SortingPanel(
            motion_controller=FakeMotion(zeroed=zeroed, idle=idle),
            config=config,
            camera_ownership=CameraOwnershipCoordinator(),
        )
        panel.calibration = object() if calibrated else None
        panel.vision_source.capture = lambda: object() if camera_ready else None
        return panel

    def test_readiness_reports_platform_not_idle_after_zero(self):
        panel = self.build_panel(zeroed=True, idle=False)

        ready, reason = panel.start_readiness()

        self.assertFalse(ready)
        self.assertIn("平台空闲", reason)
        panel.shutdown()

    def test_readiness_reports_missing_calibration_even_when_camera_has_frame(self):
        panel = self.build_panel(calibrated=False, camera_ready=True)

        ready, reason = panel.start_readiness()

        self.assertFalse(ready)
        self.assertIn("相机标定", reason)
        panel.shutdown()

    def test_failed_start_shows_exact_reason_in_error_label(self):
        panel = self.build_panel(zeroed=True, idle=False)

        with patch("sorting_panel.QMessageBox.warning") as warning:
            panel.start_sorting()

        self.assertIn("平台空闲", panel.error_label.text())
        self.assertIn("平台空闲", warning.call_args.args[2])
        panel.shutdown()


if __name__ == "__main__":
    unittest.main()
