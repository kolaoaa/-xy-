import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gui import MainWindow
from PyQt5.QtWidgets import QApplication, QTabWidget


class MainWindowAcceptanceIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_creates_acceptance_panel(self):
        window = MainWindow()

        self.assertTrue(hasattr(window, "acceptance_panel"))

        window.close()
        self.app.processEvents()

    def test_main_window_creates_gripper_debug_tab(self):
        window = MainWindow()

        self.assertTrue(hasattr(window, "gripper_debug_panel"))
        tab_texts = []
        for tabs in window.findChildren(QTabWidget):
            tab_texts.extend(tabs.tabText(index) for index in range(tabs.count()))
        self.assertIn("调试夹子", tab_texts)

        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
