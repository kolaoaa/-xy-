import unittest

from camera_calibration import Bounds
from camera_worker import LatestVisionSource
from vision_detector import CapDetection, filter_detections, order_targets_nearest_first


def cap(class_name, confidence, x, y):
    return CapDetection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=(x - 3, y - 3, x + 3, y + 3),
        pixel_x=x,
        pixel_y=y,
        platform_x_mm=float(x),
        platform_y_mm=float(y),
    )


class VisionDetectorTests(unittest.TestCase):
    def test_latest_vision_source_returns_snapshot_copies(self):
        source = LatestVisionSource()
        frame = [[1, 2], [3, 4]]
        detection = cap("red_cap", 0.9, 20, 20)

        source.update_frame(frame)
        source.update_detections([detection])
        captured = source.capture()
        detected = source.detect(captured)
        detected.clear()

        self.assertEqual(captured, frame)
        self.assertEqual(source.detect(captured), [detection])

    def test_filters_low_confidence_out_of_bounds_processed_and_close_targets(self):
        detections = [
            cap("red_cap", 0.90, 20, 20),
            cap("blue_cap", 0.80, 24, 23),
            cap("green_cap", 0.30, 80, 80),
            cap("red_cap", 0.95, 300, 80),
            cap("blue_cap", 0.85, 120, 120),
        ]

        filtered = filter_detections(
            detections=detections,
            confidence_threshold=0.50,
            workspace=Bounds(10, 250, 10, 250),
            processed_points=[(121, 121)],
            processed_radius_mm=5,
            min_spacing_mm=10,
        )

        self.assertEqual([(item.platform_x_mm, item.platform_y_mm) for item in filtered], [(20.0, 20.0)])

    def test_orders_targets_from_current_platform_position(self):
        ordered = order_targets_nearest_first(
            [cap("red_cap", 0.9, 200, 200), cap("blue_cap", 0.9, 30, 40)],
            current_position=(0, 0),
        )

        self.assertEqual(ordered[0].class_name, "blue_cap")


if __name__ == "__main__":
    unittest.main()
