"""H1 (2026-09-23 audit) — requirements.txt must describe the environment
that actually runs.

The old file pinned numpy==2.2.6 over the shipped 1.26.4 ABI (compiled
wheels fail to import under numpy 2.x), pinned pywinauto DOWN, and omitted
~12 imported runtime packages. It is now a pip-freeze of the working venv.
"""
import re
import unittest
from pathlib import Path

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"


class RequirementsTests(unittest.TestCase):
    def setUp(self):
        self.text = REQ.read_text(encoding="utf-8")
        self.pins = {}
        for line in self.text.splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line.strip())
            if m:
                self.pins[m.group(1).lower().replace("_", "-")] = m.group(2)

    def test_numpy_pin_matches_the_working_abi(self):
        self.assertEqual(self.pins.get("numpy"), "1.26.4")

    def test_pywinauto_is_not_pinned_down(self):
        self.assertEqual(self.pins.get("pywinauto"), "0.6.9")

    def test_every_previously_missing_runtime_package_is_pinned(self):
        for pkg in ("faster-whisper", "ctranslate2", "playwright", "pillow",
                    "av", "pydub", "webrtcvad", "keyboard", "mss", "httpx",
                    "ddgs", "pytest", "simpleaudio"):
            self.assertIn(pkg, self.pins, pkg)


if __name__ == "__main__":
    unittest.main()
