import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class XYPlatformControllerZeroTests(unittest.TestCase):
    def test_home_marks_zeroed_only_after_idle_status_response(self):
        gui_dir = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            import struct
            from gui import XYPlatformController
            from protocol import CommandType, PlatformStatus, ProtocolFrame

            class FakeSignal:
                def __init__(self):
                    self.values = []

                def emit(self, *args):
                    self.values.append(args)

            class FakeEmitter:
                def __init__(self):
                    self.error_occurred = FakeSignal()
                    self.log_message = FakeSignal()
                    self.status_updated = FakeSignal()

            class FakeCommunicator:
                def __init__(self):
                    self.sent = []

                def is_connected(self):
                    return True

                def send_data(self, data):
                    self.sent.append(data)
                    return True

            def status_frame(status, error=0):
                data = struct.pack("<ffBB", 0.0, 0.0, int(status), error)
                return ProtocolFrame().pack(CommandType.STATUS_RESPONSE, data)

            controller = XYPlatformController(FakeCommunicator(), FakeEmitter())
            assert not controller.is_zeroed()
            controller.home()
            assert not controller.is_zeroed()
            controller.handle_response(status_frame(PlatformStatus.HOMING))
            assert not controller.is_zeroed()
            controller.handle_response(status_frame(PlatformStatus.IDLE))
            assert controller.is_zeroed()
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=gui_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_move_command_marks_platform_busy_before_status_response(self):
        gui_dir = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(
            """
            from gui import XYPlatformController

            class FakeSignal:
                def __init__(self):
                    self.values = []

                def emit(self, *args):
                    self.values.append(args)

            class FakeEmitter:
                def __init__(self):
                    self.error_occurred = FakeSignal()
                    self.log_message = FakeSignal()
                    self.status_updated = FakeSignal()

            class FakeCommunicator:
                def __init__(self):
                    self.sent = []

                def is_connected(self):
                    return True

                def send_data(self, data):
                    self.sent.append(data)
                    return True

            controller = XYPlatformController(FakeCommunicator(), FakeEmitter())
            assert controller.is_idle()
            assert controller.move_abs(40.0, 50.0, 10)
            assert not controller.is_idle()
            """
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=gui_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
