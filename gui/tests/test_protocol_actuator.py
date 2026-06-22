import unittest

from protocol import CommandType


class RemovedActuatorProtocolTests(unittest.TestCase):
    def test_electromagnet_actuator_commands_are_removed(self):
        self.assertFalse(hasattr(CommandType, "ACTUATOR_ON"))
        self.assertFalse(hasattr(CommandType, "ACTUATOR_OFF"))


if __name__ == "__main__":
    unittest.main()
