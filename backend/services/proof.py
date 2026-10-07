"""RANK 4 — PROVE IT BEFORE CLAIMING IT: per-tool postcondition proofs.

A tool's "ok" is only a CLAIM. This module checks the world for the proof a
claim needs, with the smallest possible rules per tool:

  * ``code.write_file`` — the file exists on disk AND its content hash matches
    what was written (or, at minimum, it is a non-empty file when the content
    was not echoed back to us).
  * ``code.create_folder`` — the path exists and is a directory.
  * ``code.apply_patch`` — the patched file is present (and non-empty).

Tools without a proof rule return ``None`` and keep today's behaviour; the
point is that a proofable tool can never publish a "done" the filesystem
contradicts. All checks are stdlib-only filesystem reads.
"""

import os
from typing import Any, Dict, Optional

from backend.services import code_grants

PROVED = "proved"
FAILED = "failed"

#: Tools whose success claim has a deterministic postcondition to verify.
PROOF_TOOLS = frozenset({
    "code.write_file", "code.create_folder", "code.apply_patch",
})


def _path_from(args: Any, structured: Any) -> str:
    """The path a step claimed to touch, from its result first, then args."""
    for source in (structured, args):
        if isinstance(source, dict):
            path = source.get("path")
            if isinstance(path, str) and path.strip():
                return path.strip()
    return ""


def _file_size(path: str) -> Optional[int]:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def prove_step(tool: str, args: Any, structured: Any) -> Optional[Dict[str, str]]:
    """Verify the postcondition of a step that reported success.

    Returns ``{"state": "proved"|"failed", "evidence": str}`` for a proofable
    tool, or ``None`` when the tool has no proof rule.
    """
    if tool not in PROOF_TOOLS:
        return None
    path = _path_from(args, structured)
    if not path:
        return {"state": FAILED,
                "evidence": "the step did not name a path to check"}
    if not os.path.exists(path):
        return {"state": FAILED,
                "evidence": "%s is missing after the step" % path}
    if tool == "code.create_folder":
        if os.path.isdir(path):
            return {"state": PROVED, "evidence": "folder exists: %s" % path}
        return {"state": FAILED,
                "evidence": "%s is not a folder" % path}
    if tool == "code.write_file":
        return _prove_write(path, args, structured)
    # code.apply_patch — a patch leaves a changed file at the path.
    if os.path.isfile(path):
        size = _file_size(path)
        return {"state": PROVED,
                "evidence": "patched file present: %s (%s bytes)"
                            % (path, size)}
    return {"state": FAILED,
            "evidence": "%s is not a file after the patch" % path}


def _prove_write(path: str, args: Any, structured: Any) -> Dict[str, str]:
    """A write is proved by content, not by the tool's own summary."""
    if not os.path.isfile(path):
        return {"state": FAILED,
                "evidence": "%s is not a file after the write" % path}
    actual = code_grants.content_hash(path=path)
    expected = None
    content = args.get("content") if isinstance(args, dict) else None
    if isinstance(content, str):
        expected = code_grants.content_hash(text=content)
    if expected is None and isinstance(structured, dict):
        after = structured.get("after_hash")
        if isinstance(after, str) and after:
            expected = after
    if expected:
        if actual == expected:
            return {"state": PROVED,
                    "evidence": "file content verified on disk: %s "
                                "(sha256 %s)" % (path, str(actual)[:12])}
        return {
            "state": FAILED,
            "evidence": ("%s content does not match what was written "
                         "(expected %s, found %s)"
                         % (path, str(expected)[:12], str(actual)[:12])),
        }
    size = _file_size(path)
    if size:
        return {"state": PROVED,
                "evidence": "file exists on disk: %s (%d bytes)"
                            % (path, size)}
    return {"state": FAILED,
            "evidence": "%s is empty after the write" % path}
