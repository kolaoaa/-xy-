import unittest

import numpy as np
from PIL import Image

from training.evaluate_demo_model import pil_to_bgr_array


class EvaluationHelperTests(unittest.TestCase):
    def test_pil_to_bgr_array_swaps_rgb_channels_for_ultralytics(self):
        image = Image.new("RGB", (1, 1), (240, 20, 10))

        frame = pil_to_bgr_array(image)

        self.assertEqual(frame.tolist(), [[[10, 20, 240]]])
        self.assertTrue(frame.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
