import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from camera_calibration import Bounds, SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from sorting_panel import SortingPanel


class FakeMotion:
    def is_zeroed(self):
        return True

    def is_idle(self):
        return True

    def query_status(self, silent=False):
        return True

    def stop(self):
        return True


class SortingPanelCalibrationPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_calibration_points_follow_300mm_pick_workspace_corners(self):
        config = SortingConfig(
            platform_bounds=Bounds(0, 300, 0, 300),
            pick_workspace=Bounds(10, 290, 10, 290),
            bins={},
        )
        panel = SortingPanel(
            motion_controller=FakeMotion(),
            config=config,
            camera_ownership=CameraOwnershipCoordinator(),
        )

        self.assertEqual(
            panel.calibration_platform_points(),
            [(10.0, 10.0), (290.0, 10.0), (290.0, 290.0), (10.0, 290.0)],
        )
        panel.shutdown()


if __name__ == "__main__":
    unittest.main()
