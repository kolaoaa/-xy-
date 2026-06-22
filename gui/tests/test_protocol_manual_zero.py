import unittest

from protocol import CommandBuilder, CommandType, ProtocolFrame


class ManualZeroProtocolTests(unittest.TestCase):
    def test_set_zero_command_packs_empty_payload_frame(self):
        frame = CommandBuilder.set_zero()

        self.assertEqual(CommandType.SET_ZERO, 0x08)
        self.assertEqual(frame, bytes([0xAA, 0x08, 0x00, 0x08, 0xFF]))
        cmd, data = ProtocolFrame.unpack(frame)
        self.assertEqual(cmd, CommandType.SET_ZERO)
        self.assertEqual(data, b"")


if __name__ == "__main__":
    unittest.main()
