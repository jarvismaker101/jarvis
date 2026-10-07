"""Rank 1 — entity ledger + reference resolver ("this one", "it", "that file").

Pure stdlib; imports nothing from the codebase, so ``brain`` and the
workers can use it without import cycles. Thread-safe.

The ledger is the live notebook of every object Jarvis has seen,
created, or been told about: files, folders, and (later) videos,
websites, jobs. The resolver binds a mention in a turn to one ledger
entry using deterministic scoring — never the nearest noun, never a
guess. Close ties ask ONE short question; a "no, not that one" rejects
the bound entity (negative evidence) so the re-resolution never offers
it again.
"""

import os
import re
import threading
import time

_lock = threading.Lock()
_entities = []          # entity dicts; most-recently-touched last
_entities_by_id = {}   # id -> entity dict (same objects as in _entities)
_seq = 0
_ENTITIES_MAX = 30

_last_binding = None   # {"mention","entity_id","candidates","expected_kind",
                       #  "constraints","at"} — what "not that one" revises.

_rejected = {}         # entity_id -> rejected_at (negative evidence)
_REJECTION_TTL = 600   # a rejection stops counting after 10 minutes


#: Kinds that may substitute for each other when filtering candidates.
_KIND_SUBSTITUTES = {
    "file": {"file", "document"},
    "document": {"document", "file"},
    "video": {"video", "movie", "topic"},
    "movie": {"movie", "video", "topic"},
    "website": {"website", "url", "site", "topic"},
    "url": {"url", "website", "site", "topic"},
    "site": {"site", "website", "url", "topic"},
    "topic": {"topic", "video", "movie", "website", "url", "site"},
}

#: How much each provenance source is trusted (spec §3b.3).
_PROVENANCE_TRUST = {
    "user_said": 1.0,
    "created": 1.0,
    "created_by_job": 1.0,
    "listed": 0.9,
    "observed": 0.7,
    "screen": 0.7,
    "research": 0.6,
}

#: Verbs that tell us what KIND "it"/"that" must be.
_FILE_VERBS_RE = re.compile(
    r"\b(delete|remove|rename|open|read|edit|play|move|copy|save|"
    r"write|append|create)\b",
    re.IGNORECASE,
)
_FOLDER_VERBS_RE = re.compile(
    r"\b(list|inspect|inside|into|within|there)\b",
    re.IGNORECASE,
)

_MENTION_NOUN_KIND = {
    "file": "file", "files": "file",
    "folder": "folder", "folders": "folder",
    "directory": "folder", "directories": "folder",
    "video": "video", "movie": "movie",
    "website": "website", "site": "site", "webpage": "website",
    "report": "file", "document": "document", "photo": "file",
    "image": "file", "song": "file",
    "stream": "topic", "streamer": "topic", "livestream": "topic",
    "creator": "topic", "youtuber": "topic", "channel": "topic",
    "content": "topic", "topic": "topic",
}

_DEMONSTRATIVE_RE = re.compile(
    r"\b(this|that)\s+(?:(?:youtube|yt|the)\s+)?"
    r"(one|file|files|folder|folders|directory|"
    r"directories|video|movie|website|site|report|document|photo|"
    r"image|song|stream|streamer|livestream|creator|youtuber|"
    r"channel|content|topic)\b",
    re.IGNORECASE,
)
_BARE_PRONOUN_RE = re.compile(r"\b(it|that|them)\b", re.IGNORECASE)
_DEFINTE_CREATED_RE = re.compile(
    r"\bthe\s+(file|folder|directory|video|movie|website|report|"
    r"document)\s+(?:that\s+)?you\s+(?:just\s+)?"
    r"(made|created|listed|mentioned|showed|found|opened|saved|"
    r"wrote|checked)\b",
    re.IGNORECASE,
)
_QUOTED_NAME_RE = re.compile(r"""["']([^"'"]{1,80})["']""")
_NAMED_RE = re.compile(
    r"\b(?:named|called|by the name of)\s+([A-Za-z0-9 _.\-]{1,80})",
    re.IGNORECASE,
)

_BIND_THRESHOLD = 0.50   # top score below this = unresolvable
_ASK_MARGIN = 0.15       # top two within this (both strong) = ask once


def _looks_like_path(ref):
    text = str(ref or "")
    return bool(os.path.isabs(text) or os.sep in text or "/" in text
                or re.match(r"^[A-Za-z]:", text))


def record_entity(name, ref, kind="file", source="observed",
                  confidence=None, identifiers=None):
    """Upsert an entity; returns its dict. Never raises."""
    global _seq
    try:
        kind = str(kind or "file").lower()
        text_ref = str(ref or "")
        canon = os.path.normpath(text_ref) if _looks_like_path(text_ref) else text_ref
        key = (kind, os.path.normcase(canon))
        with _lock:
            ent = None
            for existing in _entities:
                ekey = (str(existing.get("kind") or ""),
                        os.path.normcase(str(existing.get("canon") or "")))
                if ekey == key:
                    ent = existing
                    break
            now = time.time()
            if ent is None:
                _seq += 1
                ent = {
                    "id": "ent-%d" % _seq,
                    "kind": kind,
                    "display_name": str(name or os.path.basename(canon) or canon),
                    "ref": text_ref,
                    "canon": canon,
                    "provenance": str(source or "observed"),
                    "confidence": (float(confidence) if confidence is not None
                                   else _PROVENANCE_TRUST.get(str(source or ""), 0.7)),
                    "identifiers": dict(identifiers or {}),
                    "salience": 0,
                    "version": 1,
                    "user_confirmed": False,
                    "observed_at": now,
                    "last_mentioned_at": 0.0,
                }
                _entities.append(ent)
                _entities_by_id[ent["id"]] = ent
                while len(_entities) > _ENTITIES_MAX:
                    dropped = _entities.pop(0)
                    _entities_by_id.pop(dropped["id"], None)
            else:
                ent["version"] = int(ent.get("version") or 1) + 1
                ent["observed_at"] = now
                if name:
                    ent["display_name"] = str(name)
                try:
                    _entities.remove(ent)
                except ValueError:
                    pass
                _entities.append(ent)
            return dict(ent)
    except Exception:
        return {}


def mention_entity(entity_id):
    """Mark an entity mentioned/focused (recency + salience). Never raises."""
    try:
        with _lock:
            ent = _entities_by_id.get(str(entity_id))
            if not ent:
                return
            ent["salience"] = int(ent.get("salience") or 0) + 1
            ent["last_mentioned_at"] = time.time()
            try:
                _entities.remove(ent)
            except ValueError:
                pass
            _entities.append(ent)
    except Exception:
        pass


def reject_entity(entity_id):
    """Negative evidence: the user said "not that one". Never raises."""
    try:
        with _lock:
            if str(entity_id) in _entities_by_id:
                _rejected[str(entity_id)] = time.time()
    except Exception:
        pass


def rejected_ids():
    """Unexpired rejected entity ids. Never raises."""
    try:
        now = time.time()
        with _lock:
            live = {eid for eid, at in _rejected.items()
                    if now - float(at or 0) < _REJECTION_TTL}
            for eid in set(_rejected) - live:
                _rejected.pop(eid, None)
            return live
    except Exception:
        return set()


def focus_head(kind=None):
    """Most recently mentioned/observed entity (optionally of one kind)."""
    try:
        with _lock:
            entries = list(_entities)
        allowed = None
        if kind:
            allowed = _KIND_SUBSTITUTES.get(str(kind).lower(), {str(kind).lower()})
        best, best_at = None, -1.0
        for ent in entries:
            if allowed and str(ent.get("kind") or "") not in allowed:
                continue
            stamp = float(ent.get("last_mentioned_at") or 0.0) or float(
                ent.get("observed_at") or 0.0)
            if stamp >= best_at:
                best, best_at = ent, stamp
        return dict(best) if best else None
    except Exception:
        return None


def set_last_binding(mention, entity_id, candidates=None,
                     expected_kind=None, constraints=None, text=None):
    """Remember what "not that one" should revise. Never raises."""
    global _last_binding
    try:
        with _lock:
            _last_binding = {
                "mention": str(mention or ""),
                # Raw user text re-resolves cleanly; the display label may
                # carry quote characters that would parse as a quoted name.
                "text": str(text or mention or ""),
                "entity_id": str(entity_id or ""),
                "candidates": [str(c) for c in (candidates or [])],
                "expected_kind": expected_kind,
                "constraints": dict(constraints or {}),
                "at": time.time(),
            }
    except Exception:
        pass


def get_last_binding():
    """The last recorded mention binding, or None."""
    try:
        with _lock:
            return dict(_last_binding) if _last_binding else None
    except Exception:
        return None


def reset():
    """Clear the ledger (tests only)."""
    global _last_binding, _seq
    with _lock:
        _entities[:] = []
        _entities_by_id.clear()
        _rejected.clear()
        _last_binding = None
        _seq = 0


def snapshot():
    """One consistent read of entities + last binding (diagnostics/tests)."""
    with _lock:
        return {
            "entities": [dict(e) for e in _entities],
            "last_binding": dict(_last_binding) if _last_binding else None,
        }


def kind_hint_from_verb(text):
    """"delete it" -> file; "list it" -> folder; else None. Never raises."""
    try:
        lowered = str(text or "").lower()
        if _FILE_VERBS_RE.search(lowered):
            return "file"
        if _FOLDER_VERBS_RE.search(lowered):
            return "folder"
    except Exception:
        pass
    return None


def _recency_score(ent, now):
    stamp = float(ent.get("last_mentioned_at") or 0.0) or float(
        ent.get("observed_at") or 0.0)
    age = max(0.0, now - stamp)
    return 1.0 / (1.0 + age / 120.0)


def _name_score(ent, surface):
    if not surface:
        return None
    surface = str(surface).strip().lower()
    names = {str(ent.get("display_name") or "").lower()}
    canon = str(ent.get("canon") or "")
    if canon:
        names.add(os.path.basename(canon).lower())
        names.add(canon.lower())
    names.discard("")
    for name in names:
        if name == surface:
            return 1.0
    for name in names:
        if surface and (surface in name or name in surface):
            return 0.7
    return 0.0


def _constraint_gate(ent, constraints):
    """Hard gates: location / provenance / name_contains must ALL match."""
    try:
        constraints = constraints or {}
        location = str(constraints.get("location") or "").lower()
        if location:
            home = os.path.expanduser("~")
            roots = {
                "desktop": os.path.join(home, "Desktop"),
                "documents": os.path.join(home, "Documents"),
                "downloads": os.path.join(home, "Downloads"),
                "home": home,
            }
            root = roots.get(location)
            canon = str(ent.get("canon") or "")
            if not root or not canon:
                return False
            if (os.path.normcase(os.path.normpath(canon)) !=
                    os.path.normcase(os.path.normpath(root)) and
                    not os.path.normcase(canon).startswith(
                        os.path.normcase(root) + os.sep)):
                return False
        provenance = str(constraints.get("provenance") or "").lower()
        if provenance and provenance not in str(ent.get("provenance") or "").lower():
            if not (provenance == "created"
                    and "creat" in str(ent.get("provenance") or "").lower()):
                return False
        needle = str(constraints.get("name_contains") or "").lower()
        if needle:
            hay = (str(ent.get("display_name") or "") + " " +
                   str(ent.get("canon") or "")).lower()
            if needle not in hay:
                return False
    except Exception:
        return False
    return True


def _liveness_factor(ent):
    try:
        canon = str(ent.get("canon") or "")
        if canon and _looks_like_path(canon) and not os.path.exists(canon):
            return 0.7
    except Exception:
        pass
    return 1.0


def _score(ent, surface, now):
    name = _name_score(ent, surface)
    recency = _recency_score(ent, now)
    salience = min(int(ent.get("salience") or 0), 5) / 5.0
    trust = _PROVENANCE_TRUST.get(str(ent.get("provenance") or ""), 0.7)
    if name is None:
        total = 0.45 * recency + 0.25 * salience + 0.30 * trust
    else:
        total = (0.40 * name + 0.30 * recency + 0.15 * salience
                 + 0.15 * trust)
    return total * _liveness_factor(ent)


def _spoken(ent):
    name = str(ent.get("display_name") or "") or os.path.basename(
        str(ent.get("canon") or "")) or str(ent.get("canon") or "")
    kind = str(ent.get("kind") or "")
    if kind and not name.lower().startswith(("the ", "a ")):
        return "the %s '%s'" % (kind, name) if kind in (
            "file", "folder", "video", "movie", "website") else name
    return name


def resolve_mention(text, expected_kind=None, constraints=None):
    """Bind a mention in *text* to one ledger entity.

    Returns ("bound", entity) | ("ask", {"question", "options"}) |
    ("none", "") — never raises, never guesses. A detected mention with
    no viable candidate asks (once, upstream) instead of binding.
    """
    try:
        return _resolve(text, expected_kind, constraints)
    except Exception:
        return "none", ""


def _resolve(text, expected_kind, constraints):
    lowered = str(text or "")
    if not lowered.strip():
        return "none", ""
    constraints = dict(constraints or {})

    surface, noun_kind, provenance_hint = None, None, None
    created_m = _DEFINTE_CREATED_RE.search(lowered)
    if created_m:
        noun_kind = _MENTION_NOUN_KIND.get(created_m.group(1).lower())
        verb = created_m.group(2).lower()
        provenance_hint = ("created" if verb in (
            "made", "created", "saved", "wrote") else verb)
    else:
        demo_m = _DEMONSTRATIVE_RE.search(lowered)
        if demo_m:
            word = demo_m.group(2).lower()
            if word == "one":
                surface = None
            else:
                noun_kind = _MENTION_NOUN_KIND.get(word)
        else:
            quoted_m = _QUOTED_NAME_RE.search(lowered)
            if quoted_m:
                surface = quoted_m.group(1).strip()
            else:
                named_m = _NAMED_RE.search(lowered)
                if named_m:
                    surface = named_m.group(1).strip()
    if surface is None and noun_kind is None and not _BARE_PRONOUN_RE.search(lowered):
        return "none", ""

    want_kind = expected_kind or noun_kind or kind_hint_from_verb(lowered)
    if provenance_hint:
        constraints = dict(constraints)
        constraints.setdefault("provenance", provenance_hint)

    now = time.time()
    try:
        rejected = rejected_ids()
    except Exception:
        rejected = set()
    allowed = (_KIND_SUBSTITUTES.get(str(want_kind).lower(), {str(want_kind).lower()})
               if want_kind else None)

    scored = []
    try:
        with _lock:
            entries = list(_entities)
    except Exception:
        entries = []
    for ent in entries:
        try:
            if str(ent.get("id") or "") in rejected:
                continue
            if allowed and str(ent.get("kind") or "") not in allowed:
                continue
            if not _constraint_gate(ent, constraints):
                continue
            scored.append((_score(ent, surface, now), ent))
        except Exception:
            continue
    scored.sort(key=lambda pair: pair[0], reverse=True)

    mention_label = (surface or noun_kind or
                     ("'%s'" % lowered.strip()[:60]))
    raw_text = str(text or "").strip()[:160]
    if not scored or scored[0][0] < _BIND_THRESHOLD:
        question = ("Sir, which %s do you mean?" % want_kind
                    if want_kind else "Sir, which one do you mean?")
        return "ask", {"question": question, "options": []}
    if (len(scored) > 1 and scored[1][0] >= _BIND_THRESHOLD
            and scored[0][0] - scored[1][0] < _ASK_MARGIN):
        top_two = [dict(scored[0][1]), dict(scored[1][1])]
        question = ("Sir, by %s do you mean %s or %s?" % (
            mention_label, _spoken(top_two[0]), _spoken(top_two[1])))
        set_last_binding(mention_label, top_two[0].get("id"),
                         [e.get("id") for e in top_two],
                         expected_kind or noun_kind, constraints,
                         text=raw_text)
        return "ask", {"question": question, "options": top_two}
    best = dict(scored[0][1])
    mention_entity(best.get("id"))
    cands = [str(e.get("id") or "") for _, e in scored[:3]]
    set_last_binding(mention_label, best.get("id"), cands,
                     expected_kind or noun_kind, constraints,
                     text=raw_text)
    return "bound", best
