"""H3 + H7 (2026-09-23 audit) — UIA acting safety.

H3: set_focus failure must abort typing (never type into whatever holds
focus) and legacy bare-wrapper cache entries must never act (no liveness
metadata).
H7: every UIA entry initializes the thread's COM apartment first.
"""
import time
import unittest
from unittest.mock import MagicMock, patch

from backend.services import screen_ui_elements as uie


class SetFocusTests(unittest.TestCase):
    def test_type_aborts_when_set_focus_fails(self):
        wrapper = MagicMock()
        wrapper.set_focus.side_effect = RuntimeError("focus refused")
        entry = {"wrapper": wrapper, "created_at": time.monotonic(),
                 "hwnd": 123}
        with patch.object(uie, "_element_cache", {"u1": entry}):
            ok = uie.invoke_element("u1", action="type", text="secret",
                                    hwnd=123)
        self.assertFalse(ok)
        wrapper.type_keys.assert_not_called()

    def test_type_proceeds_when_set_focus_works(self):
        wrapper = MagicMock()
        entry = {"wrapper": wrapper, "created_at": time.monotonic(),
                 "hwnd": 123}
        with patch.object(uie, "_element_cache", {"u1": entry}):
            ok = uie.invoke_element("u1", action="type", text="hello",
                                    hwnd=123)
        self.assertTrue(ok)
        wrapper.type_keys.assert_called_once()


class BareWrapperTests(unittest.TestCase):
    def test_legacy_bare_wrapper_is_refused(self):
        wrapper = MagicMock()
        with patch.object(uie, "_element_cache", {"u1": wrapper}), \
             patch.object(uie, "find_element_by_runtime_id", return_value=None):
            ok = uie.invoke_element("u1", action="type", text="x", hwnd=123)
        self.assertFalse(ok)
        wrapper.type_keys.assert_not_called()
        wrapper.set_focus.assert_not_called()


class ComInitTests(unittest.TestCase):
    def test_ensure_com_returns_bool_and_calls_comtypes(self):
        fake_comtypes = MagicMock()
        with patch.dict("sys.modules", {"comtypes": fake_comtypes}):
            self.assertTrue(uie._ensure_com())
        fake_comtypes.CoInitialize.assert_called()

    def test_ensure_com_fails_closed_on_com_error(self):
        fake_comtypes = MagicMock()
        fake_comtypes.CoInitialize.side_effect = OSError(
            "[WinError -2147417850] Cannot change thread mode after it is set")
        with patch.dict("sys.modules", {"comtypes": fake_comtypes}):
            self.assertFalse(uie._ensure_com())


if __name__ == "__main__":
    unittest.main()
