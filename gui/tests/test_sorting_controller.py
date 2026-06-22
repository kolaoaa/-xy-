import unittest

from camera_calibration import Bounds
from sorting_controller import SortState, SortingController
from vision_detector import CapDetection


class FakeMotion:
    def __init__(self):
        self.commands = []
        self.idle = True
        self.position = (0.0, 0.0)

    def home(self):
        self.commands.append(("home",))
        self.idle = False
        return True

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


class LazyIdleMotion(FakeMotion):
    def move_abs(self, x, y, speed):
        self.commands.append(("move_abs", x, y, speed))
        return True


class FakeVision:
    def __init__(self, detections):
        self.detections = detections

    def capture(self):
        return object()

    def detect(self, frame):
        return list(self.detections)


class FakeGripper:
    def __init__(self, configured=True):
        self.configured = configured
        self.commands = []

    def is_configured(self):
        return self.configured

    def close(self):
        self.commands.append("close")
        return True

    def open(self):
        self.commands.append("open")
        return True


def make_detection():
    return CapDetection(
        class_id=0,
        class_name="red_cap",
        confidence=0.95,
        bbox=(10, 10, 20, 20),
        pixel_x=15,
        pixel_y=15,
        platform_x_mm=40.0,
        platform_y_mm=50.0,
    )


def build_controller(gripper=None, dry_run=False, monotonic=None):
    motion = FakeMotion()
    vision = FakeVision([make_detection()])
    gripper = gripper or FakeGripper()
    controller = SortingController(
        motion=motion,
        vision=vision,
        gripper=gripper,
        bins={"red_cap": (20.0, 180.0), "unknown": (200.0, 180.0)},
        platform_bounds=Bounds(0, 260, 0, 260),
        pick_workspace=Bounds(10, 250, 10, 250),
        speed_mm_s=10,
        pick_wait_s=0,
        release_wait_s=0,
        communication_timeout_s=2,
        dry_run=dry_run,
        monotonic=monotonic,
    )
    return controller, motion, gripper


class SortingControllerTests(unittest.TestCase):
    def test_start_uses_current_zero_without_homing(self):
        controller, motion, _ = build_controller()

        controller.start()

        self.assertEqual(controller.state, SortState.CAPTURE_IMAGE)
        self.assertEqual(motion.commands, [])

    def test_requires_configured_gripper_for_real_sorting(self):
        controller, _, _ = build_controller(gripper=FakeGripper(configured=False))

        controller.start()

        self.assertEqual(controller.state, SortState.ERROR)
        self.assertIn("gripper", controller.error_message.lower())

    def test_dry_run_completes_one_target_without_gripper_hardware(self):
        controller, motion, gripper = build_controller(
            gripper=FakeGripper(configured=False),
            dry_run=True,
        )
        controller.start()
        motion.idle = True

        for _ in range(30):
            controller.tick()
            if controller.state in (SortState.MOVE_TO_PICK, SortState.MOVE_TO_BIN):
                if motion.commands and motion.commands[-1][0] == "move_abs":
                    _, x, y, _ = motion.commands[-1]
                    motion.position = (x, y)
                motion.idle = True
            if controller.state == SortState.FINISHED:
                break

        self.assertEqual(controller.state, SortState.FINISHED)
        self.assertEqual(controller.sorted_counts["red_cap"], 1)
        self.assertEqual(gripper.commands, [])

    def test_pause_and_resume_preserve_state(self):
        controller, motion, _ = build_controller()
        controller.start()
        motion.idle = True
        controller.tick()
        saved_state = controller.state

        controller.pause()
        controller.tick()
        controller.resume()

        self.assertEqual(controller.state, saved_state)

    def test_waits_for_position_update_before_closing_gripper(self):
        motion = LazyIdleMotion()
        vision = FakeVision([make_detection()])
        gripper = FakeGripper()
        controller = SortingController(
            motion=motion,
            vision=vision,
            gripper=gripper,
            bins={"red_cap": (20.0, 180.0), "unknown": (200.0, 180.0)},
            platform_bounds=Bounds(0, 260, 0, 260),
            pick_workspace=Bounds(10, 250, 10, 250),
            speed_mm_s=10,
            pick_wait_s=0,
            release_wait_s=0,
            communication_timeout_s=2,
        )

        controller.start()
        for _ in range(4):
            controller.tick()

        self.assertEqual(controller.state, SortState.MOVE_TO_PICK)
        self.assertEqual(gripper.commands, [])
        self.assertEqual(motion.commands, [("move_abs", 40.0, 50.0, 10)])

        motion.position = (40.0, 50.0)
        controller.tick()

        self.assertEqual(controller.state, SortState.GRIPPER_CLOSE)

    def test_stop_stops_motion_and_returns_to_idle(self):
        controller, motion, gripper = build_controller()
        controller.start()

        controller.stop()

        self.assertEqual(controller.state, SortState.IDLE)
        self.assertEqual(motion.commands[-1], ("stop",))
        self.assertEqual(gripper.commands[-1], "open")

    def test_communication_timeout_enters_error(self):
        now = [0.0]
        controller, motion, gripper = build_controller(monotonic=lambda: now[0])
        controller.start()
        controller.note_communication()
        now[0] = 3.0

        controller.tick()

        self.assertEqual(controller.state, SortState.ERROR)
        self.assertIn("timeout", controller.error_message.lower())
        self.assertEqual(motion.commands[-1], ("stop",))
        self.assertEqual(gripper.commands[-1], "open")

    def test_platform_error_stops_task_immediately(self):
        controller, motion, gripper = build_controller()
        controller.start()

        controller.report_platform_error("Positive limit triggered")

        self.assertEqual(controller.state, SortState.ERROR)
        self.assertIn("limit", controller.error_message.lower())
        self.assertEqual(motion.commands[-1], ("stop",))
        self.assertEqual(gripper.commands[-1], "open")


if __name__ == "__main__":
    unittest.main()
