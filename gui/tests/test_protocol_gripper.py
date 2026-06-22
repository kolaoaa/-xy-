import unittest

from protocol import CommandBuilder, CommandType, ProtocolFrame


class GripperProtocolTests(unittest.TestCase):
    def test_gripper_commands_replace_actuator_commands(self):
        self.assertEqual(CommandType.SET_ZERO, 0x08)
        self.assertEqual(CommandType.SERVO_OPEN, 0x09)
        self.assertEqual(CommandType.SERVO_CLOSE, 0x0A)
        self.assertEqual(CommandType.SERVO_MIDDLE, 0x0B)

    def test_gripper_frames_have_no_payload(self):
        self.assertEqual(CommandBuilder.servo_open(), bytes.fromhex("AA 09 00 09 FF"))
        self.assertEqual(CommandBuilder.servo_close(), bytes.fromhex("AA 0A 00 0A FF"))
        self.assertEqual(CommandBuilder.servo_middle(), bytes.fromhex("AA 0B 00 0B FF"))

    def test_gripper_frames_unpack(self):
        cmd, data = ProtocolFrame.unpack(CommandBuilder.servo_open())

        self.assertEqual(cmd, CommandType.SERVO_OPEN)
        self.assertEqual(data, b"")


if __name__ == "__main__":
    unittest.main()
