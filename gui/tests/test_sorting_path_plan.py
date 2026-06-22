import unittest

from camera_calibration import Bounds
from sorting_controller import SortingController, UsbGripper
from vision_detector import CapDetection


class FakeMotion:
    def __init__(self):
        self.commands = []
        self.position = (5.0, 5.0)
        self.idle = True

    def move_abs(self, x, y, speed):
        self.commands.append(("move_abs", x, y, speed))
        self.idle = False
        return True

    def stop(self):
        self.commands.append(("stop",))
        self.idle = True
        return True

    def is_idle(self):
        return self.idle

    def current_position(self):
        return self.position

    def servo_close(self):
        self.commands.append(("servo_close",))
        return True

    def servo_open(self):
        self.commands.append(("servo_open",))
        return True

    def servo_middle(self):
        self.commands.append(("servo_middle",))
        return True


class FakeVision:
    def __init__(self, detections):
        self.detections = detections

    def capture(self):
        return object()

    def detect(self, frame):
        return list(self.detections)


def detection(class_name, x, y, confidence=0.9):
    return CapDetection(
        class_id=0,
        class_name=class_name,
        confidence=confidence,
        bbox=(0, 0, 10, 10),
        pixel_x=5,
        pixel_y=5,
        platform_x_mm=x,
        platform_y_mm=y,
    )


class SortingPathPlanTests(unittest.TestCase):
    def test_snapshot_contains_nearest_first_pick_and_bin_path(self):
        motion = FakeMotion()
        controller = SortingController(
            motion=motion,
            vision=FakeVision(
                [
                    detection("red_cap", 240.0, 240.0),
                    detection("cestbon_cap", 30.0, 40.0),
                ]
            ),
            gripper=UsbGripper(motion),
            bins={"red_cap": (20.0, 180.0), "cestbon_cap": (140.0, 180.0), "unknown": (280.0, 180.0)},
            platform_bounds=Bounds(0, 300, 0, 300),
            pick_workspace=Bounds(10, 290, 10, 290),
            speed_mm_s=10,
            pick_wait_s=0,
            release_wait_s=0,
            communication_timeout_s=2,
        )

        controller.start()
        controller.tick()
        controller.tick()

        self.assertEqual(controller.pending_targets[0].class_name, "cestbon_cap")
        self.assertEqual(
            controller.snapshot()["planned_path"],
            [(5.0, 5.0), (30.0, 40.0), (140.0, 180.0), (240.0, 240.0), (20.0, 180.0)],
        )

    def test_usb_gripper_delegates_to_motion_controller(self):
        motion = FakeMotion()
        gripper = UsbGripper(motion)

        self.assertTrue(gripper.is_configured())
        self.assertTrue(gripper.close())
        self.assertTrue(gripper.open())
        self.assertTrue(gripper.middle())
        self.assertEqual(motion.commands, [("servo_close",), ("servo_open",), ("servo_middle",)])


if __name__ == "__main__":
    unittest.main()
