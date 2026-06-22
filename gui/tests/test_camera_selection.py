import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from acceptance_panel import AcceptancePanel
from camera_calibration import Bounds, SortingConfig
from camera_ownership import CameraOwnershipCoordinator
from sorting_panel import SortingPanel


def build_config(camera_index=0) -> SortingConfig:
    return SortingConfig(
        platform_bounds=Bounds(0, 260, 0, 260),
        pick_workspace=Bounds(10, 250, 10, 250),
        bins={},
        camera_index=camera_index,
    )


class SignalSpy:
    def __init__(self):
        self.callbacks = []

    def connect(self, callback):
        self.callbacks.append(callback)


class FakeCameraWorker:
    instances = []

    def __init__(self, camera_index, parent=None):
        self.camera_index = int(camera_index)
        self.parent = parent
        self.frame_ready = SignalSpy()
        self.camera_status = SignalSpy()
        self.error_occurred = SignalSpy()
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeVisionWorker:
    def __init__(self, detector, parent=None):
        self.detector = detector
        self.parent = parent
        self.result_ready = SignalSpy()
        self.error_occurred = SignalSpy()
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeMotion:
    def __init__(self, zeroed=True, idle=True):
        self.zeroed = zeroed
        self.idle = idle

    def is_zeroed(self):
        return self.zeroed

    def is_idle(self):
        return self.idle

    def query_status(self, silent=False):
        return True

    def stop(self):
        return True


class CameraSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        FakeCameraWorker.instances.clear()

    def test_acceptance_panel_starts_selected_camera_index(self):
        with (
            patch("acceptance_panel.CameraWorker", FakeCameraWorker),
            patch("acceptance_panel.VisionWorker", FakeVisionWorker),
        ):
            panel = AcceptancePanel(
                config=build_config(camera_index=0),
                camera_ownership=CameraOwnershipCoordinator(),
            )
            panel.camera_index_input.setValue(1)

            panel.start_camera()

            self.assertEqual(FakeCameraWorker.instances[-1].camera_index, 1)
            self.assertFalse(panel.camera_index_input.isEnabled())
            panel.stop_camera()

    def test_sorting_panel_starts_selected_camera_index(self):
        with (
            patch("sorting_panel.CameraWorker", FakeCameraWorker),
            patch("sorting_panel.VisionWorker", FakeVisionWorker),
        ):
            panel = SortingPanel(
                motion_controller=FakeMotion(),
                config=build_config(camera_index=0),
                camera_ownership=CameraOwnershipCoordinator(),
            )
            panel.camera_index_input.setValue(2)

            panel.toggle_camera()

            self.assertEqual(FakeCameraWorker.instances[-1].camera_index, 2)
            self.assertFalse(panel.camera_index_input.isEnabled())
            panel.shutdown()

    def test_sorting_panel_requires_manual_zero_before_start(self):
        panel = SortingPanel(
            motion_controller=FakeMotion(zeroed=False),
            config=build_config(camera_index=0),
            camera_ownership=CameraOwnershipCoordinator(),
        )
        panel.calibration = object()
        panel.vision_source.capture = lambda: object()

        with patch("sorting_panel.QMessageBox.warning") as warning:
            panel.start_sorting()

        self.assertEqual(panel.sorting.state.value, "IDLE")
        self.assertTrue(warning.called)
        panel.shutdown()

    def test_sorting_panel_requires_idle_platform_before_start(self):
        panel = SortingPanel(
            motion_controller=FakeMotion(zeroed=True, idle=False),
            config=build_config(camera_index=0),
            camera_ownership=CameraOwnershipCoordinator(),
        )
        panel.calibration = object()
        panel.vision_source.capture = lambda: object()

        with patch("sorting_panel.QMessageBox.warning") as warning:
            panel.start_sorting()

        self.assertEqual(panel.sorting.state.value, "IDLE")
        self.assertTrue(warning.called)
        panel.shutdown()


if __name__ == "__main__":
    unittest.main()
