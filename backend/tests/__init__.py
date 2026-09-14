"""Test-package import hook: a last-resort safety net for the real ``.env``.

``.env`` holds the developer's live API keys and is gitignored, so anything
that deletes it destroys data that cannot be recovered from version control.
That is not hypothetical: ``PayloadKnobTests`` used to ``unlink()`` the real
file and restore it from memory in a ``finally``, so crashing, a hard timeout,
Ctrl+C or a killed run between those two steps lost every key for good.

Any test that needs config isolation must neutralise the loader instead of
touching the file::

    with patch("dotenv.load_dotenv"):   # reload reads no .env file
        importlib.reload(config)

``conftest.py`` at the repo root enforces that under pytest. The suite is
normally run with ``python -m unittest``, where a conftest is inert, so this
module does the same job for that path: snapshot ``.env`` once when the test
package is imported and restore it at interpreter exit if anything removed or
changed it. A healthy run never writes to ``.env``.
"""

import atexit
import sys
from pathlib import Path

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
_SNAPSHOT = None


def _read_env():
    """Current bytes of ``.env``, or None when it is absent/unreadable."""
    try:
        return _ENV_PATH.read_bytes() if _ENV_PATH.exists() else None
    except OSError:
        return None


def _restore_env():
    """Put the snapshot back if the run removed or changed ``.env``."""
    if _SNAPSHOT is None:
        return
    current = _read_env()
    if current == _SNAPSHOT:
        return
    try:
        _ENV_PATH.write_bytes(_SNAPSHOT)
    except OSError as exc:
        print("[ENV GUARD] could not restore %s: %s" % (_ENV_PATH, exc),
              file=sys.stderr)
        return
    print(
        "[ENV GUARD] the test run DELETED or MODIFIED %s; the original file "
        "has been restored from the session snapshot. A test must never "
        "operate on the real .env — isolate config parsing with "
        "patch('dotenv.load_dotenv') instead (see "
        "backend/tests/test_browser_agent.py::PayloadKnobTests)."
        % _ENV_PATH,
        file=sys.stderr,
    )


def _install_guard():
    global _SNAPSHOT
    if _SNAPSHOT is not None:
        return
    _SNAPSHOT = _read_env()
    if _SNAPSHOT is not None:
        atexit.register(_restore_env)


_install_guard()
