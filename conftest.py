"""Test-session safety net for the developer's untracked ``.env``.

``.env`` holds the real API keys and is gitignored, so anything that deletes
it destroys data that cannot be recovered from version control. A test must
NEVER operate on it: to test config parsing in isolation, neutralise the
loader instead —

    with patch("dotenv.load_dotenv"):   # reload reads no .env file
        importlib.reload(config)

(see ``PayloadKnobTests`` in ``backend/tests/test_browser_agent.py``).

This guard is belt-and-braces. It snapshots ``.env`` once per session and:

* restores it after any single test that changed or deleted it, and fails
  that test loudly so the offending code is fixed rather than tolerated;
* restores it again at interpreter exit as a last resort.

It is intentionally read-only with respect to a healthy session: if nothing
touches ``.env``, the snapshot is never written back.
"""

import atexit
from pathlib import Path

import pytest

_ENV = Path(__file__).resolve().parent / ".env"
_SNAPSHOT = None
_REASON = (
    "this test modified or deleted the repository .env, which holds the "
    "developer's real API keys and is not in version control. Tests must "
    "never touch it — isolate config parsing with "
    "patch('dotenv.load_dotenv') instead."
)


def _read():
    """Current bytes of .env, or None when the file is absent/unreadable."""
    try:
        return _ENV.read_bytes() if _ENV.exists() else None
    except OSError:
        return None


def _restore():
    """Put the snapshot back if anything removed or changed .env."""
    if _SNAPSHOT is None:
        return False
    if _read() == _SNAPSHOT:
        return False
    try:
        _ENV.write_bytes(_SNAPSHOT)
    except OSError:
        return False
    return True


def pytest_configure(config):
    global _SNAPSHOT
    _SNAPSHOT = _read()
    if _SNAPSHOT is not None:
        atexit.register(_restore)


@pytest.fixture(autouse=True)
def guard_the_real_env():
    """Fail and self-heal if a test leaves the real ``.env`` changed."""
    yield
    if _SNAPSHOT is None:
        return
    if _read() != _SNAPSHOT:
        _restore()
        raise AssertionError(_REASON)
