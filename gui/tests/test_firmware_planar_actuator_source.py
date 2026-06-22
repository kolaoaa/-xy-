import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FirmwarePlanarActuatorSourceTests(unittest.TestCase):
    def test_firmware_has_pa0_servo_gripper_without_electromagnet_commands(self):
        servo_h = PROJECT_ROOT / "MDK-ARM" / "Drivers" / "z_servo.h"
        servo_cpp = PROJECT_ROOT / "MDK-ARM" / "Drivers" / "z_servo.cpp"
        xusb_h = PROJECT_ROOT / "MDK-ARM" / "Drivers" / "xusb.h"
        xusb_cpp = PROJECT_ROOT / "MDK-ARM" / "Drivers" / "xusb.cpp"
        my_rtos_cpp = PROJECT_ROOT / "MDK-ARM" / "App" / "my_rtos.cpp"
        my_config_h = PROJECT_ROOT / "MDK-ARM" / "App" / "my_config.h"
        freertos_c = PROJECT_ROOT / "Core" / "Src" / "freertos.c"
        project_file = PROJECT_ROOT / "MDK-ARM" / "XY_Platform_Motion_Control.uvprojx"

        self.assertTrue(servo_h.exists())
        self.assertTrue(servo_cpp.exists())

        self.assertIn("servo_open", servo_h.read_text(encoding="utf-8"))
        servo_source = servo_cpp.read_text(encoding="utf-8")
        self.assertIn("SERVO_PWM_GPIO_PORT", servo_source)
        self.assertIn("SERVO_PWM_GPIO_PIN", servo_source)

        config = my_config_h.read_text(encoding="utf-8")
        self.assertIn("SERVO_PWM_GPIO_PORT GPIOA", config)
        self.assertIn("SERVO_PWM_GPIO_PIN GPIO_PIN_0", config)
        self.assertIn("SERVO_OPEN_POS 500", config)
        self.assertIn("SERVO_CLOSE_POS 1700", config)

        protocol = xusb_h.read_text(encoding="utf-8")
        self.assertIn("CMD_SERVO_OPEN", protocol)
        self.assertIn("CMD_SERVO_CLOSE", protocol)
        self.assertIn("CMD_SERVO_MIDDLE", protocol)
        self.assertNotIn("CMD_ACTUATOR_", protocol)
        self.assertNotIn("CMD_Z_", protocol)

        handler = xusb_cpp.read_text(encoding="utf-8")
        self.assertIn("case CMD_SERVO_OPEN", handler)
        self.assertIn("case CMD_SERVO_CLOSE", handler)
        self.assertIn("case CMD_SERVO_MIDDLE", handler)
        self.assertIn("servo_open();", handler)
        self.assertNotIn("actuator_", handler)

        startup = my_rtos_cpp.read_text(encoding="utf-8")
        self.assertIn("StartServoTask", startup)
        self.assertIn("ConfigureWorkspace(0.0f, 300.0f, 0.0f, 300.0f)", startup)

        self.assertIn("servoTaskHandle", freertos_c.read_text(encoding="utf-8"))
        project_text = project_file.read_text(encoding="utf-8")
        self.assertIn("z_servo.cpp", project_text)
        self.assertNotIn("actuator.cpp", project_text)


if __name__ == "__main__":
    unittest.main()
