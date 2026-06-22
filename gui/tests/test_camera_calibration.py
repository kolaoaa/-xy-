import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from camera_calibration import HomographyCalibration, load_sorting_config


class HomographyCalibrationTests(unittest.TestCase):
    def test_maps_pixel_points_to_platform_coordinates(self):
        calibration = HomographyCalibration.from_points(
            pixel_points=[(10, 20), (110, 20), (110, 220), (10, 220)],
            platform_points=[(0, 0), (200, 0), (200, 100), (0, 100)],
        )

        x_mm, y_mm = calibration.pixel_to_platform(60, 120)

        self.assertAlmostEqual(x_mm, 100.0, places=4)
        self.assertAlmostEqual(y_mm, 50.0, places=4)

    def test_saves_and_loads_calibration_json(self):
        calibration = HomographyCalibration.from_points(
            pixel_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            platform_points=[(0, 0), (200, 0), (200, 200), (0, 200)],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "camera_calibration.json"
            calibration.save(path)

            loaded = HomographyCalibration.load(path)

        self.assertTrue(np.allclose(calibration.matrix, loaded.matrix))

    def test_loads_bounds_from_sorting_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sorting_config.json"
            path.write_text(
                json.dumps(
                    {
                        "platform_bounds_mm": {
                            "x_min": 0,
                            "x_max": 260,
                            "y_min": 0,
                            "y_max": 260,
                        },
                        "pick_workspace_mm": {
                            "x_min": 10,
                            "x_max": 250,
                            "y_min": 10,
                            "y_max": 250,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_sorting_config(path)

        self.assertTrue(config.platform_bounds.contains(0, 260))
        self.assertTrue(config.pick_workspace.contains(250, 10))
        self.assertFalse(config.pick_workspace.contains(0, 0))


if __name__ == "__main__":
    unittest.main()

