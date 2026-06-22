import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui import XYPlotCanvas
from PyQt5.QtWidgets import QApplication
from vision_detector import CapDetection


class TrajectoryOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_canvas_accepts_detections_planned_path_and_bins(self):
        canvas = XYPlotCanvas()

        canvas.set_sorting_overlay(
            detections=[
                {"class_name": "red_cap", "x": 30.0, "y": 40.0, "confidence": 0.91},
                {"class_name": "cestbon_cap", "x": 100.0, "y": 120.0, "confidence": 0.88},
            ],
            planned_path=[(0.0, 0.0), (30.0, 40.0), (20.0, 180.0)],
            bins={"red_cap": (20.0, 180.0), "cestbon_cap": (140.0, 180.0)},
            current_target=(30.0, 40.0),
        )

        self.assertEqual(list(canvas.planned_path_line.get_xdata()), [0.0, 30.0, 20.0])
        self.assertEqual(list(canvas.planned_path_line.get_ydata()), [0.0, 40.0, 180.0])
        self.assertEqual(canvas.detected_points.get_offsets().tolist(), [[30.0, 40.0], [100.0, 120.0]])
        self.assertEqual(canvas.bin_points.get_offsets().tolist(), [[20.0, 180.0], [140.0, 180.0]])
        self.assertEqual(list(canvas.active_target_point.get_xdata()), [30.0])
        self.assertEqual(list(canvas.active_target_point.get_ydata()), [40.0])

    def test_canvas_accepts_cap_detection_objects_from_vision_worker(self):
        canvas = XYPlotCanvas()

        canvas.set_sorting_overlay(
            detections=[
                CapDetection(
                    class_id=0,
                    class_name="red_cap",
                    confidence=0.9,
                    bbox=(0, 0, 10, 10),
                    pixel_x=5,
                    pixel_y=5,
                    platform_x_mm=55.0,
                    platform_y_mm=66.0,
                )
            ],
        )

        self.assertEqual(canvas.detected_points.get_offsets().tolist(), [[55.0, 66.0]])


if __name__ == "__main__":
    unittest.main()
