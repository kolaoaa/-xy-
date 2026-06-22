import unittest
from pathlib import Path

from camera_calibration import load_sorting_config


class SortingConfig300mmTests(unittest.TestCase):
    def test_default_sorting_config_uses_300mm_workspace_and_real_gripper(self):
        gui_dir = Path(__file__).resolve().parents[1]
        config = load_sorting_config(gui_dir / "config" / "sorting_config.json")

        self.assertEqual((config.platform_bounds.x_min, config.platform_bounds.x_max), (0.0, 300.0))
        self.assertEqual((config.platform_bounds.y_min, config.platform_bounds.y_max), (0.0, 300.0))
        self.assertEqual((config.pick_workspace.x_min, config.pick_workspace.x_max), (10.0, 290.0))
        self.assertEqual((config.pick_workspace.y_min, config.pick_workspace.y_max), (10.0, 290.0))
        self.assertFalse(config.dry_run)
        self.assertTrue(config.gripper_configured)


if __name__ == "__main__":
    unittest.main()
