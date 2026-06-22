import subprocess
import sys
import unittest
from pathlib import Path


class GuiRuntimeBootstrapTests(unittest.TestCase):
    def test_gui_import_preloads_torch_before_qt_widgets(self):
        gui_dir = Path(__file__).resolve().parents[1]

        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import gui; import torch; print(torch.__version__)",
            ],
            cwd=gui_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
