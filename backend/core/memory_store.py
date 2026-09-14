"""Persistent memory & continuity store.

Fable-5 audit G9:
  F06 — retrievable personal memory: every independent fact has its own
        stable key (subject+attribute, or a content-addressed key for
        free-form notes, so unrelated notes coexist instead of
        superseding each other) with provenance, confidence, sensitivity
        and a per-fact revision (superseded-by) chain; explicit
        corrections resolve ONE fact by key/alias/subject; aliases resolve
        for retrieval and correction; FTS5 retrieval over every indexed
        column returns ACTIVE rows with metadata and falls back to an
        escaped LIKE scan whenever FTS is unavailable or matches only
        inactive rows; literal ``%``/``_`` forgetting criteria are escaped;
        model proposals are stored for review and applied only on an
        explicit approval call. Bounded ``memory_context`` injection for
        the chat prompt.
  F07 — remember work, not just chat: one bounded, secret-masked event
        store (chat exchanges, task results, background replies) so
        "use the report you just researched" / "where did you save that
        file?" are grounded follow-ups. The 20-message recent-chat
        projection stays in backend/core/memory.py.
  F10 — persistent commitments: an explicitly-armed commitment store with
        a scheduler that delivers through the existing event/UI/speech
        path. It NEVER silently acquires broader computer control — the
        only supported triggers are explicit deadlines and completed
        background jobs (calendar/file-watch triggers are assumption-
        flagged by the audit and deliberately rejected).
  F09 — skills from verified runs: completed-task TRACES are captured as
        candidate declarative skills with their real steps, postconditions
        and param_schema. A candidate is trusted only by a version-bound
        approval PLUS a verified replay, a trusted version keeps working
        until its replacement is promoted, and replay failures invalidate
        applicability (changed selectors) immediately. Webpage instructions
        and generated code are never promoted into trusted skills.

Pure data module: stdlib + backend.config/backend.services.tool_policy
(both import-safe from brain) — no brain/browser_agent imports, no cycles.

Everything is gated by config.MEMORY_ENABLED; every call is a safe no-op
when the store is disabled, and every write is bounded and secret-masked.
"""

import hashlib
import itertools
import json
import logging
import os
import re
import sqlite3
import threading
import time

try:
    from backend import config as _config
except Exception:  # pragma: no cover - config import fallback
    _config = None

try:
    from backend.services.tool_policy import mask_secrets
    from backend.services.tool_policy import redact_for_egress
except Exception:  # pragma: no cover - tool_policy import fallback

    def mask_secrets(text, limit=None):
        return text if isinstance(text, str) else str(text)

    def redact_for_egress(payload):
        return mask_secrets(payload)


def _env_flag(name, default="1"):
    try:
        return os.getenv(name, default) != "0"
    except Exception:
        return default != "0"


#: Global kill switch (F06-F10 all route through this).
MEMORY_ENABLED = _env_flag("JARVIS_MEMORY_ENABLED", "1")

# ── Bounded writes (audit: redacted structured results, not transcripts) ──
FACT_SUBJECT_MAX = 120
FACT_VALUE_MAX = 400
EVENT_SUMMARY_MAX = 300
EVENT_DETAIL_MAX = 1500
COMMITMENT_TEXT_MAX = 400
SKILL_GOAL_MAX = 300

# ── Bounded reads ──
CONTEXT_BUDGET_CHARS = 600
RECALL_BUDGET_CHARS = 500
RELEVANT_LIMIT = 8

#: Commitment trigger kinds the audit explicitly authorises today.
SUPPORTED_TRIGGERS = frozenset(("deadline", "job_completed"))

_SKILL_INVALIDATE_FAILURES = 3

# ── F09: captured procedures (bounded, real steps — never the goal sentence) ──
SKILL_PROCEDURE_STEPS_MAX = 12
SKILL_PROCEDURE_TEXT_MAX = 200
SKILL_POSTCONDITIONS_MAX = 8
SKILL_PARAM_MAX = 12
SKILL_REPLAY_EVIDENCE_MAX = 300

#: Step args that bind a procedure to a CONCRETE target — the things whose
#: disappearance makes a captured skill no longer applicable.
SKILL_TARGET_ARG_KEYS = frozenset((
    "selector", "locator", "xpath", "url", "path", "name", "target",
    "query", "text",
))

#: Edit-distance-free applicability signals: text that says a recorded target
#: no longer resolves.
SKILL_APPLICABILITY_SIGNALS = (
    "not found", "no longer", "does not exist", "doesn't exist", "no such",
    "not visible", "not present", "unable to locate", "cannot find",
    "can't find", "unresolved", "gone", "missing", "404", "no element",
    "timed out", "timeout", "changed",
)

#: Signals that an observation CONTRADICTS a stored postcondition.
SKILL_POSTCONDITION_CONTRADICTIONS = (
    "does not contain", "doesn't contain", "assertion failed",
    "postcondition failed", "does not match", "doesn't match", "mismatch",
    "expected", "contradict", "no longer holds",
)

_db_lock = threading.Lock()
_db_gen = 0
_db_path = None
_local = threading.local()
_fts_ok = None  # tri-state: None = untested, True/False after first open
_delivery_cb = None
_scheduler_thread = None
_scheduler_stop = threading.Event()
#: F50 — whether this process has declared itself the store's durable writer.
#: A refusal (another live owner) is remembered so it is logged once, not on
#: every connection.
_ownership_declared = False

#: F10 — the acknowledged-outbox knobs. A reminder that cannot be delivered is
#: retried with exponential backoff and then marked 'failed' (visible), never
#: silently recorded as delivered.
COMMITMENT_MAX_ATTEMPTS = 4
COMMITMENT_RETRY_BACKOFF = 30.0  # seconds; doubles per attempt
COMMITMENT_CLAIM_TIMEOUT = 300.0  # a claim older than this is re-armed

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL DEFAULT 'is',
    value TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'user',
    source TEXT NOT NULL DEFAULT 'chat',
    confidence REAL NOT NULL DEFAULT 1.0,
    sensitivity TEXT NOT NULL DEFAULT 'normal',
    created_at REAL NOT NULL,
    superseded_by INTEGER,
    forgotten INTEGER NOT NULL DEFAULT 0,
    forgotten_at REAL
);
CREATE TABLE IF NOT EXISTS fact_aliases (
    alias TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_proposals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL DEFAULT 'is',
    value TEXT NOT NULL,
    rationale TEXT,
    source TEXT NOT NULL DEFAULT 'model',
    review_state TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL,
    reviewed_at REAL,
    applied_fact_id INTEGER,
    review_note TEXT
);
CREATE TABLE IF NOT EXISTS entities (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL DEFAULT 'thing',
    display_name TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    detail TEXT,
    refs TEXT,
    request_id TEXT
);
CREATE TABLE IF NOT EXISTS work_requests (
    request_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    route TEXT,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    closed_at REAL,
    summary TEXT,
    artifacts TEXT,
    evidence TEXT,
    provenance TEXT,
    updated_at REAL
);
CREATE TABLE IF NOT EXISTS commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    trigger_kind TEXT NOT NULL,
    trigger_data TEXT,
    context TEXT,
    prep TEXT NOT NULL DEFAULT 'none',
    notify TEXT NOT NULL DEFAULT 'once',
    due_at REAL,
    expires_at REAL,
    status TEXT NOT NULL DEFAULT 'armed',
    last_fired_at REAL,
    delivered_at REAL,
    cancel_reason TEXT,
    created_at REAL NOT NULL,
    request_id TEXT
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'candidate',
    app TEXT NOT NULL DEFAULT 'browser',
    goal TEXT NOT NULL,
    param_schema TEXT,
    preconditions TEXT,
    steps TEXT,
    postconditions TEXT,
    permissions TEXT,
    outcome TEXT,
    failure_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    approved_at REAL,
    validated_at REAL,
    retired_at REAL,
    retire_reason TEXT,
    replay_evidence TEXT
);
"""


def claim_memory_ownership(owner="backend"):
    """F50 — the BACKEND is the sole designated writer of the intelligence store.

    The audit found durable state written from whichever surface happened to
    have the path. Ownership is a lease in the shared intelligence-state
    registry: a second LIVE owner is refused, and every write path re-asserts
    it before touching the file.
    """
    try:
        from backend.services import intelligence_state as _is
    except Exception:
        return True
    try:
        _is.durable.claim(_is.RESOURCE_MEMORY, owner)
        return True
    except Exception as exc:
        logging.warning("[MEMORY] not the designated store writer: %s", exc)
        return False


def memory_owner():
    """F50 — the label of the live designated writer ("" when unowned)."""
    try:
        from backend.services import intelligence_state as _is
    except Exception:
        return "backend"
    return _is.durable.owner_of(_is.RESOURCE_MEMORY)


def assert_writer(owner="backend"):
    """F50 — fail closed when another live surface owns the store."""
    try:
        from backend.services import intelligence_state as _is
    except Exception:
        return True
    _is.durable.assert_writer(_is.RESOURCE_MEMORY, owner)
    return True


def configure(path):
    """Point the store at *path* (tests use a tmp file). Reopens lazily."""
    global _db_path, _db_gen, _fts_ok, _ownership_declared
    close()
    with _db_lock:
        _db_path = str(path)
        _db_gen += 1
        _fts_ok = None
        _ownership_declared = False
        # Per-thread cached connections carry the old generation number and
        # are reopened on the next _conn() call in that thread.
    # F50: configuring the store (i.e. the backend opening it) also claims
    # sole write ownership — this is the one place the backend adopts it.
    _ownership_declared = claim_memory_ownership()


def close():
    """Close the current thread's cached connection (test teardown hook —
    releases the SQLite file lock on Windows)."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        try:
            _local.conn = None
        except Exception:
            pass


def default_db_path():
    if _db_path:
        return _db_path
    if _config is not None and getattr(_config, "MEMORY_DB_PATH", None):
        return str(_config.MEMORY_DB_PATH)
    import pathlib
    return str(pathlib.Path(__file__).resolve().parent.parent.parent
               / "data" / "jarvis_memory.db")


def _conn():
    global _ownership_declared
    if getattr(_local, "gen", None) != _db_gen or getattr(_local, "conn", None) is None:
        path = default_db_path()
        parent = os.path.dirname(path)
        if parent:
            try:
                os.makedirs(parent, exist_ok=True)
            except Exception:
                pass
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        conn.execute("PRAGMA busy_timeout=5000")
        _ensure_schema(conn)
        # F50: whoever OPENS the intelligence store is its designated writer.
        # Declared once per process (a refusal by another live owner is
        # remembered rather than retried on every connection).
        if not _ownership_declared:
            _ownership_declared = claim_memory_ownership()
        _local.conn = conn
        _local.gen = _db_gen
    return _local.conn


def _migrate_commitments(conn):
    """F10 — the acknowledged-outbox columns for existing commitment DBs.

    "Delivered" used to be written BEFORE the callback ran, so a failed
    notification was lost forever and two concurrent ticks could both deliver
    the same reminder. Delivery is now claimed atomically, acknowledged only
    on success, and retried (with a bounded attempt count) instead.
    """
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(commitments)").fetchall()}
        for name, ddl in (
            ("attempts", "ALTER TABLE commitments ADD COLUMN attempts "
                         "INTEGER NOT NULL DEFAULT 0"),
            ("claimed_at", "ALTER TABLE commitments ADD COLUMN claimed_at REAL"),
            ("last_attempt_at", "ALTER TABLE commitments ADD COLUMN "
                                "last_attempt_at REAL"),
            ("next_attempt_at", "ALTER TABLE commitments ADD COLUMN "
                                "next_attempt_at REAL"),
            ("last_error", "ALTER TABLE commitments ADD COLUMN last_error TEXT"),
        ):
            if name not in cols:
                conn.execute(ddl)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_commitments_due "
            "ON commitments(status, due_at, next_attempt_at)")
    except Exception as exc:  # pragma: no cover - migration must never break
        logging.warning("[MEMORY] commitments schema migration failed: %s", exc)


def _migrate_facts(conn):
    """Add the F06 per-fact ``key`` column to an existing DB and backfill it.

    Legacy rows were keyed implicitly by (subject, predicate) and every
    generic note shared one key; the derived key is stable per independent
    fact, so unrelated notes stop tombstoning each other.
    """
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "key" not in cols:
            conn.execute("ALTER TABLE facts ADD COLUMN key TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(key)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_active "
            "ON facts(forgotten, superseded_by, created_at)")
    except Exception as exc:  # pragma: no cover - migration must never break
        logging.warning("[MEMORY] facts schema migration failed: %s", exc)
    try:
        rows = conn.execute(
            "SELECT id, subject, predicate, value FROM facts "
            "WHERE key IS NULL OR key = ''").fetchall()
        for r in rows:
            conn.execute(
                "UPDATE facts SET key = ? WHERE id = ?",
                (fact_key(r["subject"], r["predicate"], r["value"]), r["id"]))
    except Exception as exc:  # pragma: no cover - backfill is best effort
        logging.warning("[MEMORY] facts key backfill failed: %s", exc)


def _migrate_skills(conn):
    """Add the F09 ``replay_evidence`` column to an existing skills table.

    Promotion is replay-validated, and the evidence that validated it is kept
    next to the row (an approval that cannot name its verification is not one).
    """
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(skills)").fetchall()}
        if "replay_evidence" not in cols:
            conn.execute("ALTER TABLE skills ADD COLUMN replay_evidence TEXT")
    except Exception as exc:  # pragma: no cover - migration must never break
        logging.warning("[MEMORY] skills schema migration failed: %s", exc)


def _ensure_schema(conn):
    global _fts_ok
    conn.executescript(_SCHEMA)
    _migrate_facts(conn)
    _migrate_skills(conn)
    _migrate_commitments(conn)
    if _fts_ok is None:
        # FTS5 is optional in some SQLite builds — probe once, degrade to LIKE.
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts "
                "USING fts5(subject, predicate, value, "
                "content='facts', content_rowid='id')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS events_fts "
                "USING fts5(summary, content='events', content_rowid='id')"
            )
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts "
                "USING fts5(name, goal, content='skills', content_rowid='id')"
            )
            _fts_ok = True
        except Exception as exc:
            logging.warning("[MEMORY] FTS5 unavailable — LIKE fallback: %s", exc)
            _fts_ok = False
        if _fts_ok:
            # F06: earlier builds never actually populated the FTS index
            # (arity bug above); rebuild once per DB open so an existing DB
            # is searchable rather than silently falling back to LIKE.
            for fts_table in ("facts_fts", "events_fts", "skills_fts"):
                try:
                    conn.execute("INSERT INTO %s(%s) VALUES('rebuild')"
                                 % (fts_table, fts_table))
                except Exception as exc:
                    logging.debug("[MEMORY] fts rebuild failed: %s", exc)
    conn.commit()


_FTS_TABLES = {
    "facts": ("facts_fts", ["subject", "predicate", "value"]),
    "events": ("events_fts", ["summary"]),
    "skills": ("skills_fts", ["name", "goal"]),
}


def _fts_insert(conn, table, rowid, row):
    """Mirror one indexed row into its fts5 shadow table (best effort).

    F06: one placeholder per indexed column PLUS the explicit rowid — the
    old arity was one short, every insert raised, and the FTS index stayed
    permanently empty (so only the LIKE fallback ever matched anything).
    """
    if not _fts_ok:
        return
    fts_table, cols = _FTS_TABLES[table]
    try:
        placeholders = ", ".join(["?"] * (len(cols) + 1))
        conn.execute(
            "INSERT INTO %s(rowid, %s) VALUES (%s)"
            % (fts_table, ", ".join(cols), placeholders),
            [rowid] + [row[c] for c in cols],
        )
    except Exception as exc:
        logging.debug("[MEMORY] fts sync failed: %s", exc)


def _fts_query(conn, table, query, limit, extra_where="", extra_args=()):
    """FTS5 MATCH over *table*; returns rowids ranked best-first, or None
    when FTS is unavailable / the query is unmatchable (caller falls back
    to LIKE).

    F06: the MATCH target is the FTS table itself (every indexed column),
    not just the first column — matching only ``subject`` could not find a
    fact whose *value* held the queried term.
    """
    if not _fts_ok:
        return None
    fts_table, _cols = _FTS_TABLES[table]
    # Quote every token so user text can never break the MATCH syntax; a
    # trailing "*" makes long queries prefix-tolerant.
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", query or "")
    if not tokens:
        return None
    match = " ".join('"%s"*' % t for t in tokens[:8])
    sql = (
        "SELECT rowid FROM %s WHERE %s MATCH ? %s "
        "ORDER BY rank LIMIT ?" % (fts_table, fts_table, extra_where)
    )
    try:
        rows = conn.execute(sql, [match] + list(extra_args) + [limit]).fetchall()
        return [r[0] for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] fts query failed: %s", exc)
        return None


def _like_tokens(text):
    tokens = [t for t in re.findall(r"[A-Za-z0-9_]{3,}", text or "") if len(t) >= 3]
    return tokens[:6]


def _like_escape(text):
    """Escape a user-supplied LIKE criterion so ``%``/``_``/``\\`` are
    literal. F06: an unescaped ``%`` criterion used to act as a wildcard
    and could forget the whole store."""
    s = str(text or "")
    return (s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_"))


def _like_contains(text):
    return "%" + _like_escape(text) + "%"


def _row(row):
    return dict(row) if row is not None else None


# ── F06: per-fact stable keys & aliases ──────────────────────────────────
#: Subjects that are only a display container for free-form notes — their
#: value defines the identity of the fact (content-addressed key), so two
#: unrelated notes coexist instead of superseding each other.
GENERIC_NOTE_SUBJECTS = frozenset((
    "note", "notes", "user note", "user notes", "misc", "misc note",
))

#: Possessive/article prefixes that never change a fact's identity.
_KEY_STOPWORDS = frozenset((
    "my", "our", "your", "his", "her", "their", "the", "a", "an", "this",
    "that", "these", "those", "sir",
))


def _key_slug(text, max_words=8):
    words = [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())]
    words = [w for w in words if w not in _KEY_STOPWORDS] or words
    return "-".join(words[:max_words]) or "unknown"


def fact_key(subject, predicate="is", value=None):
    """Derive the stable key of one independent fact.

    * independent subject+attribute facts -> ``subject-slug:predicate-slug``
    * free-form notes (generic container subject) -> ``note:<content hash>``
      so unrelated notes are independent facts rather than revisions of
      the same shared key.
    """
    subject_txt = str(subject or "").strip().lower()
    if subject_txt in GENERIC_NOTE_SUBJECTS or not subject_txt:
        digest = hashlib.sha1(
            ("%s|%s" % (subject_txt, str(value or ""))).encode(
                "utf-8", "replace")).hexdigest()[:16]
        return "note:%s" % digest
    return "%s:%s" % (_key_slug(subject_txt), _key_slug(predicate or "is", 3))


def _canonical_key(subject, predicate="is", value=None):
    """The key a fact is stored/superseded under — alias-resolved.

    An explicitly supplied key (``"note:<hash>"`` / ``"project:is"``, as
    returned by retrieval metadata) is honoured verbatim when it names an
    existing fact, so callers can correct/history by key.
    """
    target = _alias_target(subject)
    if target:
        return target
    text = str(subject or "").strip()
    if ":" in text and _key_exists(text.lower()):
        return text.lower()
    return fact_key(text, predicate, value)


def _key_exists(key):
    if not key:
        return False
    try:
        return _conn().execute(
            "SELECT 1 FROM facts WHERE key = ? LIMIT 1",
            (key,)).fetchone() is not None
    except Exception:
        return False


def _alias_target(name):
    """Resolve *name* to its canonical fact key (None when not an alias)."""
    alias = _key_slug(name, 8)
    if not alias or alias == "unknown":
        return None
    try:
        row = _conn().execute(
            "SELECT target FROM fact_aliases WHERE alias = ?", (alias,)
        ).fetchone()
        return row["target"] if row else None
    except Exception as exc:
        logging.debug("[MEMORY] alias lookup failed: %s", exc)
        return None


def add_alias(alias, target):
    """Record that *alias* refers to the fact identified by *target*
    (a subject text or an explicit fact key). Returns the normalized alias
    or None. Aliases are retrieval/correction sugar — they never create a
    fact of their own."""
    if not MEMORY_ENABLED:
        return None
    alias = _key_slug(alias, 8)
    if not alias or alias == "unknown":
        return None
    target_txt = str(target or "").strip()
    if not target_txt:
        return None
    target_key = (target_txt.lower() if ":" in target_txt
                  else _canonical_key(target_txt))
    if alias == target_key or alias == target_key.split(":", 1)[0]:
        return None  # an alias for itself is not an alias
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO fact_aliases (alias, target, created_at) "
            "VALUES (?, ?, ?) ON CONFLICT(alias) DO UPDATE SET "
            "target = excluded.target",
            (alias, target_key, time.time()),
        )
        conn.commit()
        return alias
    except Exception as exc:
        logging.debug("[MEMORY] add_alias failed: %s", exc)
        return None


def resolve_alias(alias):
    """Public read: the canonical fact key *alias* points at, or None."""
    if not MEMORY_ENABLED:
        return None
    return _alias_target(alias)


def list_aliases(limit=50):
    if not MEMORY_ENABLED:
        return []
    try:
        return [_row(r) for r in _conn().execute(
            "SELECT * FROM fact_aliases ORDER BY created_at DESC LIMIT ?",
            (int(limit),)).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] list_aliases failed: %s", exc)
        return []


def remove_alias(alias):
    if not MEMORY_ENABLED:
        return 0
    try:
        cur = _conn().execute(
            "DELETE FROM fact_aliases WHERE alias = ?", (_key_slug(alias, 8),))
        _conn().commit()
        return cur.rowcount or 0
    except Exception as exc:
        logging.debug("[MEMORY] remove_alias failed: %s", exc)
        return 0


def _alias_expansions(query):
    """Extra search terms contributed by aliases mentioned in *query* —
    'phoenix' finds the fact stored as 'project'."""
    terms = []
    text = " " + _key_slug(query, 16).replace("-", " ") + " "
    if text.strip() == "unknown":
        return terms
    try:
        rows = _conn().execute(
            "SELECT alias, target FROM fact_aliases").fetchall()
    except Exception:
        return terms
    for r in rows:
        alias_words = (r["alias"] or "").replace("-", " ")
        if alias_words and alias_words in text:
            subj = (r["target"] or "").split(":", 1)[0].replace("-", " ")
            if subj and subj not in text:
                terms.append(subj)
    return terms[:3]


# ── F06: facts ────────────────────────────────────────────────────────────
def _insert_fact(subject, value, predicate="is", provenance="user",
                 source="chat", confidence=1.0, sensitivity="normal",
                 key=None):
    """One INSERT + FTS mirror + supersede of the prior ACTIVE revision of
    the SAME fact key. Returns the new fact id (or None)."""
    now = time.time()
    key = key or _canonical_key(subject, predicate, value)
    try:
        conn = _conn()
        cur = conn.execute(
            "INSERT INTO facts (key, subject, predicate, value, provenance, "
            "source, confidence, sensitivity, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, subject, predicate, value, str(provenance or "user")[:40],
             str(source or "chat")[:40], float(confidence),
             str(sensitivity or "normal")[:20], now),
        )
        fact_id = cur.lastrowid
        _fts_insert(conn, "facts", fact_id, {
            "subject": subject, "predicate": predicate, "value": value,
        })
        # Revision chain, scoped to THIS key only: unrelated independent
        # facts (other keys — including other free-form notes) are never
        # touched. History is preserved, nothing is dropped.
        conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE key = ? "
            "AND forgotten = 0 AND superseded_by IS NULL AND id != ?",
            (fact_id, key, fact_id),
        )
        upsert_entity(subject, kind="subject")
        conn.commit()
        return fact_id
    except Exception as exc:
        logging.warning("[MEMORY] remember failed: %s", exc)
        return None


def remember(subject, value, predicate="is", provenance="user",
             source="chat", confidence=1.0, sensitivity="normal"):
    """Store one independent fact; only the previous ACTIVE revision of the
    SAME fact key is superseded, never silently overwritten.

    Facts are keyed per independent fact (subject+attribute; free-form notes
    by content hash) so unrelated notes coexist. Returns the new fact id (or
    None when disabled / the input is empty).
    """
    if not MEMORY_ENABLED:
        return None
    subject = mask_secrets(str(subject or "").strip())[:FACT_SUBJECT_MAX]
    value = mask_secrets(str(value or "").strip())[:FACT_VALUE_MAX]
    predicate = str(predicate or "is").strip()[:40] or "is"
    if not subject or not value:
        return None
    return _insert_fact(subject, value, predicate=predicate,
                        provenance=provenance, source=source,
                        confidence=confidence, sensitivity=sensitivity)


def _active_fact_for_key(key):
    try:
        return _conn().execute(
            "SELECT * FROM facts WHERE key = ? AND forgotten = 0 "
            "AND superseded_by IS NULL ORDER BY id DESC LIMIT 1",
            (key,)).fetchone()
    except Exception as exc:
        logging.debug("[MEMORY] active fact lookup failed: %s", exc)
        return None


def _active_fact_for_subject(subject):
    """Fallback resolution by subject/value when no derived key matches
    (e.g. a legacy row whose key never got backfilled)."""
    needle = _like_contains(subject)
    try:
        return _conn().execute(
            "SELECT * FROM facts WHERE forgotten = 0 "
            "AND superseded_by IS NULL AND (subject LIKE ? ESCAPE '\\' "
            "OR key LIKE ? ESCAPE '\\') ORDER BY id DESC LIMIT 1",
            (needle, needle)).fetchone()
    except Exception as exc:
        logging.debug("[MEMORY] subject fact lookup failed: %s", exc)
        return None


def resolve_fact(name):
    """Resolve *name* (subject, alias or explicit fact key) to the ACTIVE
    row of exactly one fact. Public read helper (returns a decorated row
    with metadata, or None)."""
    if not MEMORY_ENABLED or not str(name or "").strip():
        return None
    row = _active_fact_for_key(_canonical_key(name))
    if row is None:
        row = _active_fact_for_subject(name)
    if row is None:
        return None
    return _fact_meta(row, _conn())


def correct_fact(subject, predicate=None, value=None, source="chat",
                 provenance="user_correction"):
    """F06 scoped correction: resolve the ONE fact the correction is about
    (by fact key, then alias, then subject) and supersede only that one,
    recording a new revision with correction provenance.

    Never a silent overwrite of anything else: unrelated facts — including
    other free-form notes — are untouched. Returns the new fact id, or None
    when nothing currently stored matches the correction target.
    """
    if not MEMORY_ENABLED:
        return None
    # Legacy positional shape correct_fact(subject, predicate, value).
    if value is None and predicate is not None:
        value, predicate = predicate, None
    subject = mask_secrets(str(subject or "").strip())[:FACT_SUBJECT_MAX]
    value = mask_secrets(str(value or "").strip())[:FACT_VALUE_MAX]
    if not subject or not value:
        return None
    key = _canonical_key(subject, predicate or "is")
    row = _active_fact_for_key(key)
    if row is None:
        row = _active_fact_for_subject(subject)
    if row is None:
        return None  # a correction needs an existing fact to correct
    key = row["key"] or key
    new_predicate = (str(predicate).strip()[:40] if predicate
                     else (row["predicate"] or "is"))
    if (row["value"] or "") == value and new_predicate == row["predicate"]:
        return row["id"]  # already the current revision: nothing to record
    return _insert_fact(row["subject"] or subject, value,
                        predicate=new_predicate, provenance=provenance,
                        source=source, confidence=row["confidence"],
                        sensitivity=row["sensitivity"], key=key)


def forget(subject=None, predicate=None, fact_id=None, source=None, key=None):
    """F06 scoped forgetting — tombstones matching ACTIVE facts.

    Every criterion is matched LITERALLY: ``%``/``_`` in a criterion are
    escaped, so a criterion of exactly ``%`` forgets nothing (it can only
    match a key that literally contains a percent sign). With no criterion
    this forgets nothing (blanket clears stay with backend/core/memory.py's
    explicit clear triggers). Returns the count.
    """
    if not MEMORY_ENABLED:
        return 0
    where, args = ["forgotten = 0"], []
    if fact_id is not None:
        where.append("id = ?")
        args.append(int(fact_id))
    if key:
        where.append("key = ?")
        args.append(str(key).strip())
    if subject:
        # A subject-target match lands on subject OR value OR key ("forget my
        # project" still works when the fact was stored under "project").
        needle = _like_contains(subject)
        where.append("(subject LIKE ? ESCAPE '\\' OR value LIKE ? ESCAPE '\\' "
                     "OR key LIKE ? ESCAPE '\\')")
        args.extend([needle, needle, needle])
    if predicate:
        where.append("predicate = ?")
        args.append(str(predicate).strip())
    if source:
        where.append("source = ?")
        args.append(str(source).strip())
    if len(where) == 1:
        return 0
    now = time.time()
    try:
        conn = _conn()
        cur = conn.execute(
            "UPDATE facts SET forgotten = 1, forgotten_at = ? WHERE %s"
            % " AND ".join(where),
            [now] + args,
        )
        conn.commit()
        return cur.rowcount or 0
    except Exception as exc:
        logging.warning("[MEMORY] forget failed: %s", exc)
        return 0


def _fact_meta(row, conn=None):
    """Attach retrieval metadata (key, revision, active flag, timestamps)
    to a facts row so callers never have to re-query the chain."""
    d = _row(row)
    if d is None:
        return None
    d["active"] = (not d.get("forgotten")
                   and d.get("superseded_by") is None)
    d["forgotten"] = bool(d.get("forgotten"))
    revision = 1
    try:
        conn = conn or _conn()
        if d.get("key"):
            revision = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE key = ? AND id <= ?",
                (d["key"], d["id"])).fetchone()[0] or 1
    except Exception:
        revision = 1
    d["revision"] = int(revision)
    return d


def fact_revisions(subject_or_key, limit=20):
    """The full revision history of ONE fact key, newest first (tombstoned
    and superseded revisions included) — the latest revision is what
    ``relevant_facts`` / ``resolve_fact`` return."""
    if not MEMORY_ENABLED or not str(subject_or_key or "").strip():
        return []
    key = _canonical_key(subject_or_key)
    try:
        conn = _conn()
        rows = conn.execute(
            "SELECT * FROM facts WHERE key = ? ORDER BY id DESC LIMIT ?",
            (key, int(limit))).fetchall()
        return [_fact_meta(r, conn) for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] fact_revisions failed: %s", exc)
        return []


def relevant_facts(query, limit=RELEVANT_LIMIT):
    """Retrieval over ACTIVE (never-superseded, not-forgotten) facts,
    newest first, with metadata.

    F06: FTS is consulted first over every indexed column, but whenever it
    yields fewer usable ACTIVE rows than requested — FTS unavailable, the
    rows unindexed, or the match landing only on inactive/tombstoned rows —
    a LIKE scan over the ACTIVE rows runs as a real fallback, so a query
    never comes back empty merely because FTS matched only inactive rows.
    Aliases mentioned in *query* are expanded to their canonical subject.
    """
    if not MEMORY_ENABLED:
        return []
    try:
        conn = _conn()
        base = (
            "SELECT * FROM facts WHERE forgotten = 0 AND superseded_by IS NULL"
        )
        terms = [str(query or "")] + _alias_expansions(query)
        rows, seen = [], set()

        def _add(row):
            if row is None or row["id"] in seen:
                return
            seen.add(row["id"])
            rows.append(row)

        for term in terms[:3]:
            for fid in (_fts_query(conn, "facts", term, limit * 3) or []):
                if len(rows) >= limit:
                    break
                raw = conn.execute(
                    "SELECT * FROM facts WHERE id = ?", (fid,)).fetchone()
                if raw is None or raw["forgotten"]:
                    continue
                if raw["superseded_by"] is None:
                    _add(raw)
                    continue
                # FTS matched a HISTORICAL revision: surface the latest
                # ACTIVE revision of the same independent fact instead of
                # coming back empty because only inactive rows matched.
                if raw["key"]:
                    _add(_active_fact_for_key(raw["key"]))
            if len(rows) >= limit:
                break
        if len(rows) < limit:
            # LIKE fallback over ACTIVE rows: literal, escaped patterns.
            for term in terms[:3]:
                tokens = _like_tokens(term) or (
                    [term.strip()] if str(term).strip() else [])
                for token in tokens:
                    if len(rows) >= limit:
                        break
                    like = _like_contains(token)
                    for r in conn.execute(
                        base + " AND (subject LIKE ? ESCAPE '\\' "
                        "OR value LIKE ? ESCAPE '\\' "
                        "OR key LIKE ? ESCAPE '\\') "
                        "ORDER BY created_at DESC LIMIT ?",
                        (like, like, like, limit),
                    ).fetchall():
                        _add(r)
        rows.sort(key=lambda r: (r["created_at"] or 0, r["id"] or 0),
                  reverse=True)
        return [_fact_meta(r, conn) for r in rows[:limit]]
    except Exception as exc:
        logging.warning("[MEMORY] relevant_facts failed: %s", exc)
        return []


def memory_context(query, budget=CONTEXT_BUDGET_CHARS):
    """The bounded injection block for the chat prompt (empty when the
    store has nothing relevant — zero prompt change until then)."""
    if not MEMORY_ENABLED:
        return ""
    rows = relevant_facts(query)
    if not rows:
        return ""
    lines = ["Known context from memory:"]
    total = len(lines[0])
    for r in rows:
        line = "- %s %s: %s" % (r["subject"], r["predicate"], r["value"])
        if total + len(line) > budget:
            break
        lines.append(line)
        total += len(line)
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


# ── F06: reviewed model proposals ────────────────────────────────────────
def propose_fact(subject, value, predicate="is", rationale=None,
                 source="model"):
    """Record a MODEL-proposed fact for REVIEW. It is never applied here:
    the proposal stays ``pending`` until an explicit
    :func:`approve_fact_proposal` call, and approval then goes through the
    same independent-fact / correction path as a user fact.

    Returns the proposal id (or None when disabled/empty).
    """
    if not MEMORY_ENABLED:
        return None
    subject = mask_secrets(str(subject or "").strip())[:FACT_SUBJECT_MAX]
    value = mask_secrets(str(value or "").strip())[:FACT_VALUE_MAX]
    predicate = str(predicate or "is").strip()[:40] or "is"
    if not subject or not value:
        return None
    key = _canonical_key(subject, predicate)
    try:
        conn = _conn()
        existing = conn.execute(
            "SELECT id FROM fact_proposals WHERE key = ? AND value = ? "
            "AND review_state = 'pending' ORDER BY id DESC LIMIT 1",
            (key, value)).fetchone()
        if existing:
            return existing["id"]
        cur = conn.execute(
            "INSERT INTO fact_proposals (key, subject, predicate, value, "
            "rationale, source, review_state, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (key, subject, predicate, value,
             mask_secrets(str(rationale or "").strip())[:300] or None,
             str(source or "model")[:20], time.time()),
        )
        conn.commit()
        return cur.lastrowid
    except Exception as exc:
        logging.debug("[MEMORY] propose_fact failed: %s", exc)
        return None


def list_fact_proposals(status="pending", limit=10):
    if not MEMORY_ENABLED:
        return []
    try:
        sql = "SELECT * FROM fact_proposals"
        args = []
        if status:
            sql += " WHERE review_state = ?"
            args.append(str(status)[:20])
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(int(limit))
        return [_row(r) for r in _conn().execute(sql, args).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] list_fact_proposals failed: %s", exc)
        return []


def approve_fact_proposal(proposal_id=None, source="chat"):
    """EXPLICIT approval of one pending model proposal: apply it through
    the same fact/correction path (superseding only its own fact key) and
    mark it applied. Returns a dict with the applied fact id, else None."""
    if not MEMORY_ENABLED:
        return None
    try:
        conn = _conn()
        if proposal_id is None:
            row = conn.execute(
                "SELECT * FROM fact_proposals WHERE review_state = 'pending' "
                "ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM fact_proposals WHERE id = ?",
                (int(proposal_id),)).fetchone()
        if row is None or row["review_state"] != "pending":
            return None
        key = row["key"] or _canonical_key(row["subject"], row["predicate"])
        if _active_fact_for_key(key) is not None:
            fact_id = correct_fact(row["subject"], value=row["value"],
                                   predicate=row["predicate"], source=source,
                                   provenance="user_approved_proposal")
        else:
            fact_id = _insert_fact(
                row["subject"], row["value"], predicate=row["predicate"],
                provenance="user_approved_proposal", source=source, key=key)
        if not fact_id:
            return None
        conn.execute(
            "UPDATE fact_proposals SET review_state = 'applied', "
            "reviewed_at = ?, applied_fact_id = ? WHERE id = ? "
            "AND review_state = 'pending'",
            (time.time(), fact_id, row["id"]))
        conn.commit()
        out = _row(row)
        out.update({"review_state": "applied", "applied_fact_id": fact_id})
        return out
    except Exception as exc:
        logging.warning("[MEMORY] approve_fact_proposal failed: %s", exc)
        return None


def reject_fact_proposal(proposal_id=None, reason=None):
    """Review outcome: the proposal is dropped, never applied."""
    if not MEMORY_ENABLED:
        return False
    try:
        conn = _conn()
        if proposal_id is None:
            row = conn.execute(
                "SELECT id FROM fact_proposals WHERE review_state = 'pending' "
                "ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
            if row is None:
                return False
            proposal_id = row["id"]
        cur = conn.execute(
            "UPDATE fact_proposals SET review_state = 'rejected', "
            "reviewed_at = ?, review_note = ? WHERE id = ? "
            "AND review_state = 'pending'",
            (time.time(), mask_secrets(str(reason or ""))[:200] or None,
             int(proposal_id)))
        conn.commit()
        return bool(cur.rowcount)
    except Exception as exc:
        logging.debug("[MEMORY] reject_fact_proposal failed: %s", exc)
        return False


def propose_facts_from_events(limit=5):
    """Episodic derivation: turn recent VERIFIED work events into PENDING
    fact proposals (review required — nothing is applied or auto-trusted).
    Returns the list of proposal ids."""
    if not MEMORY_ENABLED:
        return []
    proposals = []
    for ev in recent_events(limit=int(limit), kind="task_result"):
        detail = {}
        if ev.get("detail"):
            try:
                parsed = json.loads(ev["detail"])
                if isinstance(parsed, dict):
                    detail = parsed
            except Exception:
                detail = {}
        task = str(detail.get("task") or "").strip()
        summary = str(detail.get("summary") or ev.get("summary") or "").strip()
        if not task or not summary or summary.startswith("["):
            continue
        pid = propose_fact(
            task, summary, predicate="last_outcome",
            rationale="derived from completed work event %s (%s)"
                      % (ev.get("id"), detail.get("status") or "unknown"),
            source="episodic")
        if pid:
            proposals.append(pid)
    return proposals


# ── F06: entities (explicit normalized-key matching) ─────────────────────
def upsert_entity(key, kind="thing", display_name=None):
    if not MEMORY_ENABLED:
        return None
    key = str(key or "").strip().lower()[:80]
    if not key:
        return None
    now = time.time()
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO entities (key, kind, display_name, first_seen, "
            "last_seen) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET last_seen = excluded.last_seen",
            (key, str(kind or "thing")[:20],
             str(display_name or key)[:120], now, now),
        )
        conn.commit()
        return key
    except Exception as exc:
        logging.debug("[MEMORY] entity upsert failed: %s", exc)
        return None


def lookup_entity(key):
    if not MEMORY_ENABLED:
        return None
    key = str(key or "").strip().lower()[:80]
    if not key:
        return None
    try:
        return _row(_conn().execute(
            "SELECT * FROM entities WHERE key = ?", (key,)).fetchone())
    except Exception as exc:
        logging.debug("[MEMORY] entity lookup failed: %s", exc)
        return None


# ── F07: work-event store ────────────────────────────────────────────────
def record_event(kind, summary, request_id=None, detail=None, refs=None):
    """Record one bounded, secret-masked event. *detail* may be any
    JSON-serialisable dict (stored redacted and clipped); *refs* is a
    short list of artifact references. Returns the event id."""
    if not MEMORY_ENABLED:
        return None
    # F50: only the designated writer of the store may add durable state.
    assert_writer()
    kind = str(kind or "event").strip()[:40] or "event"
    summary = mask_secrets(str(summary or "").strip())[:EVENT_SUMMARY_MAX]
    if not summary:
        return None
    try:
        detail_json = None
        if isinstance(detail, dict):
            # F21: redact by KEY as well as by shape, and at any depth, before
            # the detail is serialised. Text-shape masking alone let a
            # {"password": "hunter2"} through into permanent memory.
            detail_json = mask_secrets(json.dumps(
                redact_for_egress(detail), ensure_ascii=False,
                default=str))[:EVENT_DETAIL_MAX]
        elif detail:
            detail_json = mask_secrets(str(detail))[:EVENT_DETAIL_MAX]
        refs_json = None
        if refs:
            # F21: artifact references are stored verbatim by design, and a
            # reference is exactly where a token gets smuggled into a URL.
            refs_json = json.dumps(
                [mask_secrets(str(r))[:160] for r in refs][:8],
                ensure_ascii=False)
        conn = _conn()
        cur = conn.execute(
            "INSERT INTO events (ts, kind, summary, detail, refs, "
            "request_id) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), kind, summary, detail_json, refs_json,
             str(request_id or "")[:60] or None),
        )
        conn.commit()
        _fts_insert(conn, "events", cur.lastrowid, {"summary": summary})
        conn.commit()
        return cur.lastrowid
    except Exception as exc:
        logging.debug("[MEMORY] record_event failed: %s", exc)
        return None


def recent_events(limit=20, kind=None):
    if not MEMORY_ENABLED:
        return []
    try:
        sql = "SELECT * FROM events"
        args = []
        if kind:
            sql += " WHERE kind = ?"
            args.append(str(kind)[:40])
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        args.append(int(limit))
        return [_row(r) for r in _conn().execute(sql, args).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] recent_events failed: %s", exc)
        return []


def find_events(query, limit=10, kind=None):
    """FTS/LIKE search over event summaries (grounding lookups)."""
    if not MEMORY_ENABLED:
        return []
    try:
        conn = _conn()
        ids = _fts_query(conn, "events", query, limit * 2)
        where, args = [], []
        if ids:
            where.append("id IN (%s)" % ", ".join(["?"] * len(ids)))
            args.extend(ids)
        else:
            for token in _like_tokens(query):
                where.append("summary LIKE ?")
                args.append("%" + token + "%")
        if kind:
            where.append("kind = ?")
            args.append(str(kind)[:40])
        if not where:
            return []
        sql = (
            "SELECT * FROM events WHERE %s ORDER BY ts DESC, id DESC LIMIT ?"
            % " AND ".join(where)
        )
        rows = conn.execute(sql, args + [int(limit)]).fetchall()
        return [_row(r) for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] find_events failed: %s", exc)
        return []


# ── F10: commitments & scheduler ─────────────────────────────────────────
def add_commitment(text, trigger_kind="deadline", due_at=None,
                   expires_at=None, context=None, request_id=None,
                   notify="once"):
    """Arm one commitment. EXPLICIT authorisation only — the caller must
    have matched an explicit user phrase (or a completed job the user
    started). Calendar/file-watch triggers are assumption-flagged by the
    audit and deliberately rejected rather than silently accepted."""
    if not MEMORY_ENABLED:
        return None
    if trigger_kind not in SUPPORTED_TRIGGERS:
        raise ValueError(
            "unsupported commitment trigger %r — only %s are explicit "
            "authorisation sources today"
            % (trigger_kind, sorted(SUPPORTED_TRIGGERS)))
    text = mask_secrets(str(text or "").strip())[:COMMITMENT_TEXT_MAX]
    if not text:
        return None
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO commitments (text, trigger_kind, trigger_data, "
        "context, due_at, expires_at, status, created_at, request_id, "
        "notify) VALUES (?, ?, ?, ?, ?, ?, 'armed', ?, ?, ?)",
        (text, trigger_kind,
         json.dumps({"due_at": due_at, "expires_at": expires_at},
                    default=str) if due_at or expires_at else None,
         mask_secrets(str(context or "").strip())[:200] or None,
         due_at, expires_at, time.time(),
         str(request_id or "")[:60] or None, str(notify or "once")[:10]),
    )
    conn.commit()
    _start_scheduler()
    return cur.lastrowid


def due_commitments(now=None):
    """Armed commitments whose moment has come AND whose retry is due.

    F10: a failed delivery is retried on its backoff schedule instead of being
    silently lost, so ``next_attempt_at`` participates in the scan.
    """
    now = now if now is not None else time.time()
    try:
        rows = _conn().execute(
            "SELECT * FROM commitments WHERE status = 'armed' "
            "AND due_at IS NOT NULL AND due_at <= ? "
            "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
            "AND (expires_at IS NULL OR expires_at > ?)",
            (now, now, now),
        ).fetchall()
        return [_row(r) for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] due_commitments failed: %s", exc)
        return []


def claim_commitment(commitment_id, now=None):
    """F10 — atomically claim ONE due commitment for delivery.

    Two scheduler ticks (or a tick racing a manual dispatch) can both see the
    same due row; only the one that wins this conditional UPDATE may deliver
    it, so a reminder can never be announced twice.
    """
    at = now if now is not None else time.time()
    try:
        conn = _conn()
        cur = conn.execute(
            "UPDATE commitments SET status = 'delivering', claimed_at = ? "
            "WHERE id = ? AND status = 'armed'",
            (at, int(commitment_id)),
        )
        conn.commit()
        return (cur.rowcount or 0) == 1
    except Exception as exc:
        logging.debug("[MEMORY] claim_commitment failed: %s", exc)
        return False


def release_claim(commitment_id, error="", now=None):
    """F10 — acknowledge FAILURE: arm the row again with bounded backoff.

    Returns True when the row will be retried, False once it is exhausted
    (status 'failed') — an undeliverable reminder is visible, not forgotten.
    """
    at = now if now is not None else time.time()
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT attempts FROM commitments WHERE id = ?",
            (int(commitment_id),)).fetchone()
        attempts = int(row["attempts"] or 0) if row else 0
        attempts += 1
        if attempts >= COMMITMENT_MAX_ATTEMPTS:
            conn.execute(
                "UPDATE commitments SET status = 'failed', attempts = ?, "
                "last_attempt_at = ?, last_error = ? "
                "WHERE id = ? AND status = 'delivering'",
                (attempts, at, str(error or "")[:200], int(commitment_id)))
            conn.commit()
            return False
        delay = COMMITMENT_RETRY_BACKOFF * (2 ** (attempts - 1))
        conn.execute(
            "UPDATE commitments SET status = 'armed', attempts = ?, "
            "last_attempt_at = ?, next_attempt_at = ?, last_error = ? "
            "WHERE id = ? AND status = 'delivering'",
            (attempts, at, at + delay, str(error or "")[:200],
             int(commitment_id)))
        conn.commit()
        return True
    except Exception as exc:
        logging.debug("[MEMORY] release_claim failed: %s", exc)
        return False


def reclaim_stale_claims(now=None, timeout=None):
    """F10 — a crash mid-delivery must not strand a reminder forever.

    A row stuck in 'delivering' past the timeout is armed again (its attempt
    count stands, so a poison reminder still exhausts and stops).
    """
    at = now if now is not None else time.time()
    stale = COMMITMENT_CLAIM_TIMEOUT if timeout is None else float(timeout)
    try:
        conn = _conn()
        cur = conn.execute(
            "UPDATE commitments SET status = 'armed' "
            "WHERE status = 'delivering' AND claimed_at IS NOT NULL "
            "AND claimed_at <= ?",
            (at - stale,))
        conn.commit()
        return cur.rowcount or 0
    except Exception as exc:
        logging.debug("[MEMORY] reclaim_stale_claims failed: %s", exc)
        return 0


def expire_commitments(now=None):
    """Lapse armed commitments past their expiry. Returns the count."""
    now = now if now is not None else time.time()
    try:
        cur = _conn().execute(
            "UPDATE commitments SET status = 'expired' "
            "WHERE status = 'armed' AND expires_at IS NOT NULL "
            "AND expires_at <= ?",
            (now,),
        )
        _conn().commit()
        return cur.rowcount or 0
    except Exception as exc:
        logging.debug("[MEMORY] expire_commitments failed: %s", exc)
        return 0


def mark_delivered(commitment_id, at=None):
    at = at if at is not None else time.time()
    try:
        conn = _conn()
        conn.execute(
            "UPDATE commitments SET status = 'delivered', last_fired_at = ?, "
            "delivered_at = ?, next_attempt_at = NULL "
            "WHERE id = ? AND status IN ('armed', 'delivering')",
            (at, at, int(commitment_id)),
        )
        conn.commit()
    except Exception as exc:
        logging.debug("[MEMORY] mark_delivered failed: %s", exc)


def commitment_tick(now=None):
    """F10 — one scheduler pass over the ACKNOWLEDGED delivery outbox.

    Order: expire lapses, reclaim stranded claims, then claim and deliver every
    due commitment. Each row is claimed atomically (so concurrent ticks notify
    once) and marked delivered ONLY after the callback succeeds; a failure
    re-arms it with backoff, so a transient delivery error is retried instead
    of being recorded as delivered. Returns the ids actually delivered.
    """
    if not MEMORY_ENABLED:
        return []
    now = now if now is not None else time.time()
    expire_commitments(now)
    reclaim_stale_claims(now)
    delivered = []
    for c in due_commitments(now):
        cid = c["id"]
        if not claim_commitment(cid, now):
            # Another tick got there first — never a second notification.
            continue
        cb = _delivery_cb
        if cb is None:
            # No delivery surface registered (headless/test): leave it armed
            # rather than claiming a delivery that never happened.
            release_claim(cid, "no delivery callback registered", now)
            continue
        try:
            result = cb(c)
            if result is False:
                raise RuntimeError("delivery callback reported failure")
        except Exception as exc:
            logging.warning("[MEMORY] commitment delivery failed: %s", exc)
            release_claim(cid, str(exc), now)
            continue
        mark_delivered(cid, now)
        delivered.append(cid)
    return delivered


def get_commitment(commitment_id):
    """F10 — one commitment by id (None when it does not exist)."""
    if not MEMORY_ENABLED:
        return None
    try:
        row = _conn().execute(
            "SELECT * FROM commitments WHERE id = ?",
            (int(commitment_id),)).fetchone()
        return _row(row) if row else None
    except Exception as exc:
        logging.debug("[MEMORY] get_commitment failed: %s", exc)
        return None


def list_commitments(status=None, limit=20):
    if not MEMORY_ENABLED:
        return []
    try:
        sql = "SELECT * FROM commitments"
        args = []
        if status:
            sql += " WHERE status = ?"
            args.append(str(status)[:12])
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(int(limit))
        return [_row(r) for r in _conn().execute(sql, args).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] list_commitments failed: %s", exc)
        return []


def cancel_commitment(commitment_id=None, reason="cancelled by user"):
    """Cancel one commitment by id, or the most recent armed one."""
    if not MEMORY_ENABLED:
        return 0
    try:
        conn = _conn()
        if commitment_id is None:
            row = conn.execute(
                "SELECT id FROM commitments WHERE status = 'armed' "
                "ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
            if not row:
                return 0
            commitment_id = row["id"]
        cur = conn.execute(
            "UPDATE commitments SET status = 'cancelled', cancel_reason = ? "
            "WHERE id = ? AND status IN ('armed', 'delivering')",
            (str(reason)[:120], int(commitment_id)),
        )
        conn.commit()
        return cur.rowcount or 0
    except Exception as exc:
        logging.debug("[MEMORY] cancel_commitment failed: %s", exc)
        return 0


def _commitment_tokens(text):
    """Content words that identify a commitment, for named cancellation."""
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    stop = {"the", "a", "an", "to", "at", "in", "on", "of", "for", "and",
            "me", "my", "remind", "reminder", "about", "that", "this",
            "please", "jarvis", "cancel", "forget", "delete", "remove"}
    return {w for w in words if len(w) > 1 and w not in stop}


def resolve_commitment(query, statuses=("armed", "delivering")):
    """F10 — identify a commitment by CONTENT, not by recency.

    The audit found that naming a reminder ("cancel the dentist reminder")
    cancelled whichever one happened to be newest. The best match by shared
    content words is returned (None when nothing is close enough).
    """
    if not MEMORY_ENABLED:
        return None
    wanted = _commitment_tokens(query)
    if not wanted:
        return None
    try:
        placeholders = ",".join("?" for _ in statuses)
        rows = _conn().execute(
            "SELECT * FROM commitments WHERE status IN (%s) "
            "ORDER BY created_at DESC, id DESC LIMIT 50" % placeholders,
            tuple(statuses)).fetchall()
    except Exception as exc:
        logging.debug("[MEMORY] resolve_commitment failed: %s", exc)
        return None
    best, best_score = None, 0.0
    for r in rows:
        row = _row(r)
        tokens = _commitment_tokens(row.get("text"))
        if not tokens:
            continue
        score = len(wanted & tokens) / float(len(wanted | tokens))
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= 0.25 else None


def cancel_commitment_by_text(query, reason="cancelled by user"):
    """F10 — cancel the NAMED commitment only; other reminders survive."""
    match = resolve_commitment(query)
    if not match:
        return 0
    return cancel_commitment(match["id"], reason)


def set_commitment_delivery(cb):
    """Register the delivery callback — brain wires this to its
    event/UI/speech path (``_notify_async_reply``). Never executes
    computer control: the callback only delivers a notification."""
    global _delivery_cb
    _delivery_cb = cb


def _scheduler_loop():
    while not _scheduler_stop.wait(5.0):
        try:
            commitment_tick()
        except Exception as exc:
            logging.debug("[MEMORY] scheduler tick failed: %s", exc)


def _start_scheduler():
    """Idempotent: spawns the daemon tick thread on the first commitment."""
    global _scheduler_thread
    if not MEMORY_ENABLED or _scheduler_thread is not None:
        return
    _scheduler_stop.clear()
    _scheduler_thread = threading.Thread(
        target=_scheduler_loop, name="jarvis-commitments", daemon=True)
    _scheduler_thread.start()


def start_scheduler():
    """F10 — start (or join) the commitment scheduler under BACKEND ownership.

    The audit found the tick thread only ever started as a side effect of
    ADDING a reminder, so a restart with reminders already stored never
    delivered them. The backend calls this at startup; the first tick then
    picks up every overdue 'armed' row.
    """
    _start_scheduler()
    return scheduler_running()


def scheduler_running():
    """F10 — is the backend-owned scheduler thread alive?"""
    thread = _scheduler_thread
    return bool(thread is not None and thread.is_alive())


def pending_commitments(now=None):
    """F10 — armed rows that are due now (including failed retries)."""
    return due_commitments(now)


def stop_scheduler():
    """Test/teardown hook — stops the daemon thread if it is running."""
    global _scheduler_thread
    _scheduler_stop.set()
    _scheduler_thread = None


# ── F09: skills from verified runs ───────────────────────────────────────
def _slugify(text):
    words = re.findall(r"[a-z0-9]+", (text or "").lower())[:6]
    return "-".join(words) or "task"


def skill_procedure_from_trace(trace, request_text=""):
    """F09: turn one VERIFIED run trace into a real PROCEDURE.

    The audit found capture storing the goal sentence instead of what the run
    actually did. Here every committed action becomes a step rendered as
    ``tool: key=value`` (e.g. ``browser.click_locator: selector=#pay``), each
    step's observation becomes a postcondition, and the args that track the
    request text become the parameterised slots of a ``param_schema``.

    Returns ``{"steps": [...], "postconditions": [...], "param_schema": {...}}``
    with empty containers when the trace proves no procedure.
    """
    request_lower = str(request_text or "").lower()
    steps, postconditions, properties, required = [], [], {}, []
    for entry in list(trace or [])[:40]:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool") or "").strip()[:60]
        if not tool:
            continue
        # Only COMMITTED actions belong to the procedure: a step that reported
        # a failure is not part of what worked.
        if entry.get("ok") is False:
            continue
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        rendered = []
        for key, value in list(args.items())[:SKILL_PARAM_MAX]:
            name = str(key)[:40]
            if isinstance(value, (list, tuple)):
                value_text = ", ".join(str(v) for v in list(value)[:8])
            else:
                value_text = str(value)
            value_text = mask_secrets(value_text)[:SKILL_PROCEDURE_TEXT_MAX]
            rendered.append("%s=%s" % (name, value_text))
            if name not in properties:
                parameterised = bool(value_text) and \
                    value_text.lower() in request_lower
                properties[name] = {
                    "type": "string",
                    "example": value_text,
                    "parameterised": parameterised,
                }
                if parameterised:
                    required.append(name)
        steps.append(("%s: %s" % (tool, ", ".join(rendered))
                      if rendered else tool)[:SKILL_PROCEDURE_TEXT_MAX])
        observation = mask_secrets(
            str(entry.get("observation") or "").strip())[:SKILL_PROCEDURE_TEXT_MAX]
        if observation:
            first_line = observation.splitlines()[0].strip()
            if first_line and first_line not in postconditions:
                postconditions.append(first_line)
        if len(steps) >= SKILL_PROCEDURE_STEPS_MAX:
            break
    param_schema = {}
    if properties:
        param_schema = {
            "type": "object",
            "properties": properties,
            # A slot that the request text filled is what a replay must vary;
            # a painted-in constant stays in the step itself.
            "required": required or list(properties),
        }
    return {
        "steps": steps,
        "postconditions": postconditions[:SKILL_POSTCONDITIONS_MAX],
        "param_schema": param_schema,
    }


def _skill_get(skill, key, default=None):
    """Read one field from a dict OR a sqlite3.Row skill record."""
    try:
        if isinstance(skill, dict):
            return skill.get(key, default)
        return skill[key]
    except Exception:
        return default


def _skill_json_list(value):
    """A skills row's list column → list (values were stored as JSON)."""
    if isinstance(value, (list, tuple)):
        return [v for v in value]
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return [value]
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    return [value]


def skill_procedure(skill):
    """F09: render one skill row's ACTUAL procedure (steps + postconditions).

    Retrieval used to print only name/version/app/goal, so a captured skill was
    indistinguishable from a remembered sentence. Bounded and empty when the
    row carries no procedure.
    """
    if skill is None:
        return ""
    steps = _skill_json_list(_skill_get(skill, "steps"))
    posts = _skill_json_list(_skill_get(skill, "postconditions"))
    parts = []
    if steps:
        rendered = " | ".join(
            "%d) %s" % (i + 1, mask_secrets(str(s))[:SKILL_PROCEDURE_TEXT_MAX])
            for i, s in enumerate(steps[:SKILL_PROCEDURE_STEPS_MAX]))
        parts.append("Steps: " + rendered)
    if posts:
        parts.append("Postconditions: " + "; ".join(
            mask_secrets(str(p))[:SKILL_PROCEDURE_TEXT_MAX]
            for p in posts[:SKILL_POSTCONDITIONS_MAX]))
    return " ".join(parts)


def record_skill_candidate(name, app, goal, steps=None, postconditions=None,
                           preconditions=None, permissions=None, outcome=None,
                           param_schema=None, source_task_id=None):
    """Capture one verified-run trace as a CANDIDATE skill (never trusted
    until a user approves it AND a replay validates it).

    F09: a new capture supersedes a prior CANDIDATE (old row retired,
    version bumps). A prior PROMOTED (trusted) version is left working —
    it is retired only when its replacement is itself promoted, so
    re-capturing a changed procedure never strips the trusted one.
    """
    if not MEMORY_ENABLED:
        return None
    name = _slugify(name or goal)
    goal = mask_secrets(str(goal or "").strip())[:SKILL_GOAL_MAX]
    if not goal:
        return None
    try:
        conn = _conn()
        prior = conn.execute(
            "SELECT id, version, status FROM skills WHERE name = ? AND app = ? "
            "AND status IN ('candidate', 'promoted') "
            "ORDER BY version DESC LIMIT 1", (name, app)).fetchone()
        version = (prior["version"] + 1) if prior else 1
        cur = conn.execute(
            "INSERT INTO skills (name, version, status, app, goal, "
            "param_schema, preconditions, steps, postconditions, "
            "permissions, outcome, created_at) "
            "VALUES (?, ?, 'candidate', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (name, version, str(app or "browser")[:20], goal,
             json.dumps(param_schema, default=str) if param_schema else None,
             json.dumps(preconditions, default=str) if preconditions else None,
             json.dumps([str(s)[:SKILL_PROCEDURE_TEXT_MAX]
                         for s in steps][:SKILL_PROCEDURE_STEPS_MAX],
                        default=str) if steps else None,
             json.dumps([str(p)[:SKILL_PROCEDURE_TEXT_MAX]
                         for p in postconditions][:SKILL_POSTCONDITIONS_MAX],
                        default=str) if postconditions else None,
             json.dumps(permissions, default=str) if permissions else None,
             mask_secrets(str(outcome or "").strip())[:300] or None,
             time.time()),
        )
        sid = cur.lastrowid
        _fts_insert(conn, "skills", sid, {"name": name, "goal": goal})
        if prior and prior["status"] == "candidate":
            conn.execute(
                "UPDATE skills SET status = 'retired', retired_at = ?, "
                "retire_reason = 'superseded by newer verified run' "
                "WHERE id = ? AND status = 'candidate'",
                (time.time(), prior["id"]),
            )
        conn.commit()
        return sid
    except Exception as exc:
        logging.debug("[MEMORY] record_skill_candidate failed: %s", exc)
        return None


def find_skills(query, status="promoted", limit=3):
    """Retrieve skills of *status* matching *query* (planner grounding)."""
    if not MEMORY_ENABLED:
        return []
    try:
        conn = _conn()
        where, args = ["status = ?"], [str(status)[:12]]
        ids = _fts_query(conn, "skills", query, limit * 2)
        if ids:
            where.append("id IN (%s)" % ", ".join(["?"] * len(ids)))
            args.extend(ids)
        else:
            toks = _like_tokens(query)
            if not toks:
                return []
            likes = []
            for token in toks:
                likes.append("(name LIKE ? OR goal LIKE ?)")
                args.extend(["%" + token + "%", "%" + token + "%"])
            where.append("(" + " OR ".join(likes) + ")")
        sql = ("SELECT * FROM skills WHERE %s ORDER BY created_at DESC "
               "LIMIT ?" % " AND ".join(where))
        rows = conn.execute(sql, args + [int(limit)]).fetchall()
        return [_row(r) for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] find_skills failed: %s", exc)
        return []


def promote_skill(skill_id, approved_version=None, replay_evidence=None):
    """Promote a candidate — version-bound approval PLUS replay validation.

    F09: approval alone is not trust. Promotion is refused unless
      * ``approved_version`` equals the row's CURRENT ``version`` (a
        re-captured procedure needs fresh approval), and
      * a verified replay exists — the row already carries ``validated_at``
        from a successful replay, or a truthy ``replay_evidence`` is supplied
        (stored on the row).

    The promoted row then REPLACES any older promoted version of the same
    (name, app), which is retired with an explicit reason.
    """
    if not MEMORY_ENABLED:
        return False
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ?", (int(skill_id),)).fetchone()
        if row is None or row["status"] != "candidate":
            return False
        if approved_version is None \
                or int(approved_version) != int(row["version"]):
            return False
        evidence = mask_secrets(
            str(replay_evidence or "").strip())[:SKILL_REPLAY_EVIDENCE_MAX]
        if not row["validated_at"] and not evidence:
            return False
        if not evidence:
            evidence = "verified replay (validated)"
        now = time.time()
        cur = conn.execute(
            "UPDATE skills SET status = 'promoted', approved_at = ?, "
            "replay_evidence = ? "
            "WHERE id = ? AND status = 'candidate' AND version = ?",
            (now, evidence, int(skill_id), int(row["version"])),
        )
        if not cur.rowcount:
            conn.rollback()
            return False
        conn.execute(
            "UPDATE skills SET status = 'retired', retired_at = ?, "
            "retire_reason = ? "
            "WHERE name = ? AND app = ? AND id != ? AND status = 'promoted'",
            (now, "replaced by promoted v%d" % int(row["version"]),
             row["name"], row["app"], int(skill_id)),
        )
        conn.commit()
        return True
    except Exception as exc:
        logging.debug("[MEMORY] promote_skill failed: %s", exc)
        return False


def retire_skill(skill_id, reason="retired by user"):
    if not MEMORY_ENABLED:
        return False
    try:
        cur = _conn().execute(
            "UPDATE skills SET status = 'retired', retired_at = ?, "
            "retire_reason = ? WHERE id = ? AND status != 'retired'",
            (time.time(), str(reason)[:120], int(skill_id)),
        )
        _conn().commit()
        return bool(cur.rowcount)
    except Exception as exc:
        logging.debug("[MEMORY] retire_skill failed: %s", exc)
        return False


def _skill_step_targets(skill):
    """The concrete targets a captured procedure is bound to.

    Steps are rendered ``tool: key=value, key2=value2``; only the args that
    name a TARGET (selector/url/path/name/...) can be invalidated by a page or
    filesystem change.
    """
    targets = []
    for line in _skill_json_list(_skill_get(skill, "steps")):
        text = str(line)
        _head, _sep, rest = text.partition(":")
        for chunk in rest.split(","):
            key, eq, value = chunk.strip().partition("=")
            if not eq:
                continue
            value = value.strip().strip("'\"")
            if value and key.strip().lower() in SKILL_TARGET_ARG_KEYS:
                targets.append((key.strip().lower(), value))
    return targets


def _skill_applicability_failure(skill, reason, observed):
    """F09: is this replay failure an APPLICABILITY failure?

    Returns a reason naming the changed selector/target when a recorded step
    target no longer resolves, or when an observation contradicts a stored
    postcondition — otherwise "" (a generic failure only counts after N).
    """
    try:
        if isinstance(observed, dict):
            observed = json.dumps(observed, default=str)
        elif isinstance(observed, (list, tuple)):
            observed = "; ".join(str(v) for v in observed)
        text = ("%s %s" % (reason or "", observed or "")).lower()
        if not text.strip():
            return ""
        signal = next(
            (s for s in SKILL_APPLICABILITY_SIGNALS if s in text), "")
        contradiction = any(
            c in text for c in SKILL_POSTCONDITION_CONTRADICTIONS)
        for key, value in _skill_step_targets(skill):
            if value.lower() not in text:
                continue
            if signal:
                return ("changed %s %r no longer resolves (%s)"
                        % (key, value, signal))
            if contradiction:
                return ("%s %r contradicts the recorded postcondition"
                        % (key, value))
        for post in _skill_json_list(_skill_get(skill, "postconditions")):
            tokens = re.findall(r"[a-z0-9]{4,}", str(post).lower())
            if tokens and all(t in text for t in tokens[:3]) and contradiction:
                return "postcondition no longer holds: %s" % str(post)[:80]
        return ""
    except Exception:
        return ""


def note_skill_replay(skill_id, success, reason="", observed=None):
    """F09: replay feedback for one skill, end to end.

    A verified success validates the row (``validated_at``) — the replay
    evidence promotion requires. A failure increments failure bookkeeping and,
    when it is an APPLICABILITY failure (a recorded selector/step target no
    longer resolves, or an observation contradicts a stored postcondition),
    invalidates the skill IMMEDIATELY with a reason naming the changed target.

    Returns the invalidation reason when the skill was invalidated, else None.
    """
    if not MEMORY_ENABLED:
        return None
    reason = mask_secrets(str(reason or "").strip())
    try:
        conn = _conn()
        row = conn.execute(
            "SELECT * FROM skills WHERE id = ?", (int(skill_id),)).fetchone()
        if row is None:
            return None
        now = time.time()
        if success:
            conn.execute(
                "UPDATE skills SET success_count = success_count + 1, "
                "validated_at = ?, replay_evidence = ? WHERE id = ?",
                (now,
                 (reason or "verified replay")[:SKILL_REPLAY_EVIDENCE_MAX],
                 int(skill_id)))
            conn.commit()
            return None
        conn.execute(
            "UPDATE skills SET failure_count = failure_count + 1 "
            "WHERE id = ?", (int(skill_id),))
        invalidation = _skill_applicability_failure(row, reason, observed)
        if not invalidation:
            count = int(row["failure_count"] or 0) + 1
            if count >= _SKILL_INVALIDATE_FAILURES:
                invalidation = "invalidated after %d failures" % count
        if invalidation and row["status"] != "retired":
            conn.execute(
                "UPDATE skills SET status = 'invalidated', retired_at = ?, "
                "retire_reason = ? WHERE id = ?",
                (now, mask_secrets(invalidation)[:120], int(skill_id)))
        conn.commit()
        return invalidation or None
    except Exception as exc:
        logging.debug("[MEMORY] note_skill_replay failed: %s", exc)
        return None


def note_skill_outcome(skill_id, success):
    """Back-compatible alias of :func:`note_skill_replay` (no reason given)."""
    return note_skill_replay(skill_id, success)


#: The event summary format written by ``record_task_outcome`` is
#: ``[engine/status] text``. Verification reads the STATUS FIELD, never a
#: substring of the whole summary ("[x/not completed]" is not a completion).
_EVENT_STATUS_RE = re.compile(r"^\s*\[[^\]/]*/([^\]]*)\]")


def _event_status(summary):
    """The exact status token of a ``[engine/status] text`` event summary."""
    match = _EVENT_STATUS_RE.match(str(summary or ""))
    return match.group(1).strip().lower() if match else ""


def recall_skills_for(command, limit=2):
    """The bounded grounding block for the planner / browser handoff:
    user-approved skills — WITH their actual procedures — and recently
    VERIFIED similar task outcomes. Empty string when nothing matches (zero
    prompt change until then)."""
    if not MEMORY_ENABLED:
        return ""
    blocks = []
    for s in find_skills(command, status="promoted", limit=limit):
        head = ("Approved skill '%s' (v%d, %s): %s"
                % (s["name"], s["version"], s["app"], s["goal"]))
        procedure = skill_procedure(s)
        block = ("%s %s" % (head, procedure)).strip() if procedure else head
        blocks.append(block[:RECALL_BUDGET_CHARS])
    for ev in find_events(command, limit=limit, kind="task_result"):
        if _event_status(ev["summary"]) == "completed":
            blocks.append("Verified prior outcome: %s" % ev["summary"])
    if not blocks:
        return ""
    lines = ["Relevant memory of prior work:"]
    total = 0
    for b in blocks[:limit * 2]:
        if total + len(b) > RECALL_BUDGET_CHARS:
            break
        lines.append("- " + b)
        total += len(b)
    return "\n".join(lines) if len(lines) > 1 else ""


# ── Explicit-phrase operations (deterministic, no model decision) ────────
_REMEMBER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remember|note)\s+that\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_FORGET_RE = re.compile(
    r"^\s*forget\s+(?:about\s+)?(?:the\s+)?(.+)$", re.IGNORECASE | re.DOTALL)
_RECALL_RE = re.compile(
    r"^\s*what\s+do\s+you\s+remember\s+about\s+(.+)$",
    re.IGNORECASE | re.DOTALL)
_REMIND_RE = re.compile(
    r"^\s*remind\s+me\s+(?:to|about|that)\s+(.+)$",
    re.IGNORECASE | re.DOTALL)
_REMINDERS_LIST_RE = re.compile(
    r"^\s*(?:what|which)\s+reminders?\b.*$|"
    r"^\s*(?:what|which)\s+commitments?\b.*$|"
    r"^\s*list\s+(?:my\s+)?reminders?\b.*$",
    re.IGNORECASE | re.DOTALL)
_REMINDER_CANCEL_RE = re.compile(
    r"^\s*cancel\s+(?:the\s+)?reminder\b.*$", re.IGNORECASE | re.DOTALL)
_SKILL_APPROVE_RE = re.compile(
    r"^\s*(?:approve|promote)\s+(?:the\s+)?(?P<name>[\w\s-]{1,40}?)\s*skill\b",
    re.IGNORECASE)
_SKILL_RETIRE_RE = re.compile(
    r"^\s*(?:retire|drop|forget)\s+(?:the\s+)?"
    r"(?P<name>[\w\s-]{1,40}?)\s*skill\b", re.IGNORECASE)
_SKILLS_LIST_RE = re.compile(
    r"^\s*what\s+skills?\s+do\s+you\s+know\b.*$", re.IGNORECASE)

# F06 — explicit correction / alias / reviewed-proposal phrasing. These are
# deterministic and scoped: a correction names ONE fact, an alias binds one
# name to one fact target, and a model proposal is only ever APPLIED by an
# explicit approval phrase.
_CORRECT_RE = re.compile(
    r"^\s*(?:please\s+)?(?:correct|update|fix)\s+(?:that\s+)?"
    r"(?P<subject>.{1,120}?)\s+(?:to|as)\s+(?P<value>.{1,400})$",
    re.IGNORECASE | re.DOTALL)
_ALIAS_ADD_RE = re.compile(
    r"^\s*(?:please\s+)?(?:add|create|remember)\s+(?:an?\s+)?alias\s+"
    r"(?P<alias>.{1,80}?)\s+(?:for|to)\s+(?P<target>.{1,120})$",
    re.IGNORECASE | re.DOTALL)
_ALIAS_KNOWN_RE = re.compile(
    r"^\s*(?P<target>.{1,120}?)\s+(?:is|are)\s+also\s+"
    r"(?:known|called)\s+as\s+(?P<alias>.{1,80})$",
    re.IGNORECASE | re.DOTALL)
_PROPOSAL_LIST_RE = re.compile(
    r"^\s*(?:what|which|list)\s+(?:my\s+)?(?:memory\s+|fact\s+|pending\s+)?"
    r"proposals?\b.*$", re.IGNORECASE)
_PROPOSAL_APPROVE_RE = re.compile(
    r"^\s*(?:approve|accept|apply)\s+(?:the\s+)?(?:memory\s+|fact\s+|"
    r"pending\s+|latest\s+)?proposals?\b.*$", re.IGNORECASE)
_PROPOSAL_REJECT_RE = re.compile(
    r"^\s*(?:reject|decline|dismiss|discard)\s+(?:the\s+)?(?:memory\s+|"
    r"fact\s+|pending\s+|latest\s+)?proposals?\b.*$", re.IGNORECASE)

# Screen-control negation guard words — a "forget ..." that names a screen
# operation is NEVER a memory op (screen confirmations keep their priority).
_SCREEN_GUARD_RE = re.compile(
    r"\b(screen|display|region|browser|tab|window|action|click|type)\b",
    re.IGNORECASE)

_IN_DURATION_RE = re.compile(
    r"\bin\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
    re.IGNORECASE)
_AT_TIME_RE = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
                         re.IGNORECASE)
_TONIGHT_RE = re.compile(r"\btonight\b", re.IGNORECASE)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)

_DUR_UNITS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
    "day": 86400, "days": 86400,
}


def parse_deadline(text, now=None):
    """Extract an explicit due time from *text*.

    Supported (explicit, auditable phrasings): 'in N <unit>', 'at HH[:MM]
    [am|pm]', 'tonight', 'tomorrow', and the COMBINATION 'tomorrow at HH[:MM]'.

    F10: the day qualifier used to be ignored whenever a clock time was
    present, so "tomorrow at 9" became today-or-tomorrow 9:00 depending on
    when it was spoken. The qualifier is now resolved with the time.
    Returns ``(due_ts, expires_ts, cleaned_text)`` or None.
    """
    if not text:
        return None
    now = now if now is not None else time.time()
    import datetime

    tomorrow = bool(_TOMORROW_RE.search(text))
    due = None
    m = _IN_DURATION_RE.search(text)
    if m:
        unit = m.group(2).lower()
        due = now + int(m.group(1)) * _DUR_UNITS.get(
            unit if unit in _DUR_UNITS else unit + "s", 60)
    if due is None:
        m = _AT_TIME_RE.search(text)
        if m:
            hour = int(m.group(1)) % 24
            minute = int(m.group(2) or 0)
            amp = (m.group(3) or "").lower()
            if amp == "pm" and hour < 12:
                hour += 12
            elif amp == "am" and hour == 12:
                hour = 0
            local = datetime.datetime.fromtimestamp(now)
            day = local.date() + datetime.timedelta(days=1 if tomorrow else 0)
            target = datetime.datetime.combine(
                day, datetime.time(hour=hour, minute=minute))
            ts = target.timestamp()
            # "at 9" spoken after 9 means the NEXT 9 o'clock; "tomorrow at 9"
            # is already pinned to tomorrow by the qualifier.
            if ts <= now:
                ts += 86400
            due = ts
    if due is None and tomorrow:
        # Bare "tomorrow" keeps its long-standing meaning: 24 hours from now.
        due = now + 86400
    if due is None and _TONIGHT_RE.search(text):
        due = now + 3600 * 6  # conservative default: +6h
    if due is None:
        return None
    cleaned = re.sub(
        r"\b(in\s+\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)"
        r"|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|tonight|tomorrow)\b",
        "", text, flags=re.IGNORECASE).strip(" ,.")
    # Default expiry: 24h after due — stale reminders never fire silently.
    return (due, due + 86400, cleaned or text)


def list_skills(limit=5):
    """Active (candidate or promoted) skills, newest first — the plain
    listing behind 'what skills do you know'."""
    if not MEMORY_ENABLED:
        return []
    try:
        rows = _conn().execute(
            "SELECT * FROM skills WHERE status IN ('candidate', 'promoted') "
            "ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        return [_row(r) for r in rows]
    except Exception as exc:
        logging.debug("[MEMORY] list_skills failed: %s", exc)
        return []


def handle_memory_phrase(msg):
    """Deterministic scoped memory/commitment/skill operations.

    Returns a reply string when the message is one of the explicit
    operations (the caller returns it straight to the user), else None so
    routing continues untouched. NEVER blanket-forgets, NEVER executes
    computer control.
    """
    if not MEMORY_ENABLED or not msg or not msg.strip():
        return None
    raw = msg.strip()
    if raw.lower() in ("forget it", "forget that", "never mind"):
        return None  # negation idiom, not a memory op

    m = _REMEMBER_RE.match(raw)
    if m:
        clause = m.group(1).strip()
        fm = re.match(r"^(.{1,120}?)\s+(is|are)\s+(.+)$", clause,
                      re.IGNORECASE | re.DOTALL)
        if fm:
            remember(fm.group(1).strip(), fm.group(3).strip(),
                     predicate=fm.group(2).lower(), provenance="user",
                     source="chat")
            return "Noted, sir — I'll remember that."
        remember("user note", clause, predicate="about", provenance="user",
                 source="chat")
        return "Noted, sir."

    m = _CORRECT_RE.match(raw)
    if m:
        subject = re.sub(r"^(my|our|his|her|the|that|this)\s+", "",
                         m.group("subject").strip(), flags=re.IGNORECASE)
        new_id = correct_fact(subject, value=m.group("value").strip(),
                              source="chat")
        if new_id:
            return ("Corrected, sir — I updated what I had stored about "
                    "%s." % subject)
        # "fix the header to blue" is a task, not a memory correction:
        # when no stored fact matches, routing continues untouched.
        return None

    m = _ALIAS_ADD_RE.match(raw) or _ALIAS_KNOWN_RE.match(raw)
    if m:
        target = re.sub(r"^(my|our|his|her|the|that|this)\s+", "",
                        m.group("target").strip(), flags=re.IGNORECASE)
        alias = m.group("alias").strip()
        if add_alias(alias, target):
            return ("Noted, sir — I'll treat '%s' as another name for %s."
                    % (alias, target))
        return None

    m = _PROPOSAL_APPROVE_RE.match(raw)
    if m:
        applied = approve_fact_proposal()
        if not applied:
            return "I have no pending memory proposal to approve, sir."
        return ("Approved, sir — I now remember %s %s: %s."
                % (applied["subject"], applied["predicate"],
                   applied["value"]))

    m = _PROPOSAL_REJECT_RE.match(raw)
    if m:
        if reject_fact_proposal(reason="rejected by user"):
            return "Rejected, sir — I won't store that proposal."
        return "I have no pending memory proposal to reject, sir."

    if _PROPOSAL_LIST_RE.match(raw):
        pending = list_fact_proposals(status="pending", limit=5)
        if not pending:
            return "No memory proposals awaiting review, sir."
        lines = ["Memory proposals awaiting your review, sir:"]
        for p in pending:
            lines.append("- %s %s: %s" % (p["subject"], p["predicate"],
                                          p["value"]))
        return "\n".join(lines)

    m = _RECALL_RE.match(raw)
    if m:
        rows = relevant_facts(m.group(1))
        if not rows:
            return "I don't have anything stored about that, sir."
        lines = ["Here's what I remember, sir:"]
        for r in rows[:5]:
            lines.append("- %s %s: %s" % (r["subject"], r["predicate"],
                                          r["value"]))
        return "\n".join(lines)

    m = _FORGET_RE.match(raw)
    if m and not _SCREEN_GUARD_RE.search(m.group(1)):
        target = m.group(1).strip().rstrip(".")
        # "forget my project" / "forget the report" -> "project" / "report".
        target = re.sub(r"^(my|our|his|her|the|that|this)\s+", "", target,
                        flags=re.IGNORECASE)
        if not target:
            return None
        if forget(subject=target):
            return "Forgotten, sir."
        return "I don't have that stored, sir."

    m = _REMIND_RE.match(raw)
    if m:
        parsed = parse_deadline(m.group(1))
        if not parsed:
            return "When should I remind you, sir?"
        due, expires, cleaned = parsed
        add_commitment(cleaned or m.group(1), trigger_kind="deadline",
                       due_at=due, expires_at=expires)
        when = "in %d minutes" % max(1, int((due - time.time()) / 60))
        return "Understood, sir — I'll remind you %s." % when

    if _REMINDERS_LIST_RE.match(raw):
        armed = list_commitments(status="armed")
        if not armed:
            return "No reminders armed, sir."
        lines = ["Armed reminders, sir:"]
        for c in armed[:5]:
            due = ""
            if c["due_at"]:
                due = " (due in %d min)" % max(
                    0, int((c["due_at"] - time.time()) / 60))
            lines.append("- %s%s" % (c["text"], due))
        return "\n".join(lines)

    if _REMINDER_CANCEL_RE.match(raw):
        if cancel_commitment():
            return "Reminder cancelled, sir."
        return "No armed reminder to cancel, sir."

    m = _SKILL_APPROVE_RE.match(raw)
    if m:
        name = _slugify(m.group("name"))
        rows = find_skills(name, status="candidate", limit=1)
        if not rows:
            return "I don't have a pending skill by that name, sir."
        row = rows[0]
        # F09: approval is version-bound AND replay-validated — the approval
        # phrase alone can never make an unvalidated run trusted.
        evidence = ("verified replay (validated)"
                    if row["validated_at"] else "")
        if promote_skill(row["id"], approved_version=row["version"],
                         replay_evidence=evidence):
            return ("Approved, sir — the '%s' skill is now trusted and will "
                    "ground similar tasks." % row["name"])
        return ("I can't trust the '%s' skill yet, sir — it has not been "
                "replayed successfully since it was captured. Run the task "
                "once more and I'll verify the procedure first." % row["name"])

    m = _SKILL_RETIRE_RE.match(raw)
    if m:
        name = _slugify(m.group("name"))
        retired = 0
        for row in (find_skills(name, status="promoted", limit=5)
                    + find_skills(name, status="candidate", limit=5)):
            if retire_skill(row["id"], reason="retired by user"):
                retired += 1
        if retired:
            return "Retired, sir — the skill will no longer be used."
        return "No active skill by that name, sir."

    if _SKILLS_LIST_RE.match(raw):
        promoted = list_skills(limit=5)
        if not promoted:
            return ("No skills approved yet, sir — completed tasks are "
                    "captured as candidates you can approve.")
        lines = ["Skills I know, sir:"]
        for s in promoted:
            lines.append("- %s (v%d, %s)" % (s["name"], s["version"],
                                             s["app"]))
        return "\n".join(lines)

    return None


def record_task_outcome(engine, status, task_description, summary="",
                        evidence=None, request_id=None, capture_skill=True,
                        trace=None, verification=None):
    """One call site for the task engines' terminal results (F07 + F09):
    a bounded task_result event always; a candidate-skill capture when the
    run was VERIFIED complete AND left a real trace.

    F09: the candidate is the PROCEDURE derived from the verified trace
    (steps/postconditions/param_schema), never the goal sentence. A run with
    no committed steps (failed, unvalidated, or traceless) produces no
    candidate at all — an unvalidated run can never become trusted.
    """
    if not MEMORY_ENABLED:
        return
    rid = request_id or current_request_id()
    record_event(
        "task_result",
        "[%s/%s] %s" % (engine, status, summary or task_description),
        request_id=rid,
        detail={"task": task_description[:300], "status": status,
                "summary": (summary or "")[:300],
                "evidence": [str(e)[:160] for e in (evidence or [])][:6]},
        refs=[task_description[:160]],
    )
    if rid:
        close_request(rid, status, summary=summary, evidence=evidence,
                      engine=engine)
    if status != "completed" or not capture_skill or not task_description:
        return
    procedure = skill_procedure_from_trace(trace, task_description)
    if not procedure["steps"]:
        return
    postconditions = list(procedure["postconditions"])
    for item in (verification or []):
        text = mask_secrets(str(item).strip())[:SKILL_PROCEDURE_TEXT_MAX]
        if text and text not in postconditions:
            postconditions.append(text)
    record_skill_candidate(
        name=task_description,
        app="browser" if engine == "browser_agent" else "shell",
        goal=task_description,
        steps=procedure["steps"],
        postconditions=postconditions[:SKILL_POSTCONDITIONS_MAX],
        param_schema=procedure["param_schema"],
        outcome=summary or "",
        permissions=["browser" if engine == "browser_agent" else "shell"],
    )


# ── F07: identified work events ───────────────────────────────────────────
# One central, identified contract for WORK (not just chat): every request gets
# an id, every terminal result links back to it, artifacts/sources are stored
# as STRUCTURE (never as a clipped blob), and both the chat and the episodic
# projection are derived from these rows.

#: Fields kept from one artifact/source record, in order. Anything else is
#: dropped rather than serialised (an artifact can carry a page's whole DOM).
_ARTIFACT_FIELDS = ("kind", "path", "url", "title", "source", "name",
                    "line", "lines", "sha1", "request_id")
_ARTIFACT_TEXT_MAX = 300
_ARTIFACT_MAX = 24

#: Per-thread identity of the request being handled. Deliberately NOT the
#: connection thread-local declared near line 98 — that one owns the DB handle.
_work_local = threading.local()

#: Collision-proof sequence. (time.monotonic() alone repeats on Windows — its
#: resolution is ~15ms, so two requests inside one tick shared an id.)
_REQUEST_SEQ = itertools.count(1)


def new_request_id(prefix="req"):
    digest = hashlib.sha1(
        ("%s|%s|%s|%s" % (prefix, os.getpid(), time.time_ns(),
                          next(_REQUEST_SEQ))).encode(
            "utf-8", "replace")).hexdigest()
    return "%s-%s" % (prefix, digest[:12])


def _artifact_rows(artifacts):
    """Normalise artifacts to redacted, field-capped STRUCTURE (F07).

    The audit found serialised detail being CLIPPED, which breaks the
    structure a follow-up needs (a half-written JSON array yields no path).
    Here each field is masked and capped individually and the list is capped
    as a list, so the result always round-trips as valid JSON.
    """
    rows = []
    for item in list(artifacts or [])[:_ARTIFACT_MAX]:
        if isinstance(item, str):
            rows.append({"kind": "ref",
                         "value": mask_secrets(item)[:_ARTIFACT_TEXT_MAX]})
            continue
        if not isinstance(item, dict):
            continue
        row = {}
        for field in _ARTIFACT_FIELDS:
            value = item.get(field)
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                row[field] = [mask_secrets(str(v))[:_ARTIFACT_TEXT_MAX]
                              for v in list(value)[:8]]
            else:
                row[field] = mask_secrets(str(value))[:_ARTIFACT_TEXT_MAX]
        if row:
            rows.append(row)
    return rows


def _json_artifact_rows(artifacts):
    if not artifacts:
        return None
    return json.dumps(redact_for_egress(_artifact_rows(artifacts)),
                      ensure_ascii=False, default=str)


def record_request(text, route=None, request_id=None, provenance=None,
                   source="chat"):
    """Record one IDENTIFIED request (F07). Returns its request_id."""
    if not MEMORY_ENABLED:
        return request_id
    # F50: only the designated writer of the store may add durable state.
    assert_writer()
    rid = str(request_id or new_request_id())[:60]
    body = mask_secrets(str(text or "").strip())[:EVENT_SUMMARY_MAX]
    try:
        conn = _conn()
        conn.execute(
            "INSERT OR REPLACE INTO work_requests (request_id, ts, route, "
            "text, status, provenance, updated_at) VALUES (?, ?, ?, ?, "
            "'open', ?, ?)",
            (rid, time.time(), str(route or "")[:60] or None, body,
             mask_secrets(str(provenance or source))[:120], time.time()),
        )
        conn.commit()
    except Exception as exc:
        logging.debug("[MEMORY] record_request failed: %s", exc)
    record_event("work_request", body or "(empty request)", request_id=rid,
                 detail={"route": route, "provenance": provenance,
                         "source": source})
    _work_local.request_id = rid
    return rid


def current_request_id():
    """The request id of the turn being handled on THIS thread (F07)."""
    return getattr(_work_local, "request_id", None)


def begin_request(text, route=None, provenance=None):
    """Start an identified request for the current thread (F07)."""
    # F50: only the designated writer of the store may add durable state.
    assert_writer()
    return record_request(text, route=route, provenance=provenance)


def record_result(request_id, status, summary="", artifacts=None,
                  evidence=None, engine=None, provenance=None):
    """Record a terminal result and LINK it to its request (F07)."""
    if not MEMORY_ENABLED:
        return None
    rid = str(request_id or current_request_id() or "")[:60] or None
    event_id = record_event(
        "work_result" if status != "needs_input" else "work_suspension",
        "[%s] %s" % (status, summary or ""),
        request_id=rid,
        detail={"status": status, "engine": engine,
                "summary": mask_secrets(str(summary or ""))[:300],
                "evidence": [mask_secrets(str(e))[:160]
                             for e in (evidence or [])][:6],
                "provenance": provenance},
        )
    if rid:
        close_request(rid, status, summary=summary, evidence=evidence,
                      artifacts=artifacts, engine=engine,
                      provenance=provenance)
    return event_id


def record_suspension(request_id, question, checkpoint_id=None):
    """Record a SUSPENSION: the request stays open, waiting for the user (F07)."""
    if not MEMORY_ENABLED:
        return None
    rid = str(request_id or current_request_id() or "")[:60] or None
    event_id = record_event(
        "work_suspension", mask_secrets(str(question or ""))[:EVENT_SUMMARY_MAX],
        request_id=rid,
        detail={"checkpoint_id": checkpoint_id, "status": "needs_input"})
    if rid:
        try:
            conn = _conn()
            conn.execute(
                "UPDATE work_requests SET status = 'suspended', "
                "summary = ?, updated_at = ? WHERE request_id = ?",
                (mask_secrets(str(question or ""))[:300], time.time(), rid))
            conn.commit()
        except Exception as exc:
            logging.debug("[MEMORY] record_suspension failed: %s", exc)
    return event_id


def close_request(request_id, status, summary="", evidence=None,
                  artifacts=None, engine=None, provenance=None):
    """Mark a request terminal and store its structured artifacts (F07).

    Artifacts are merged, not overwritten: a run can report its report path
    first and its sources later, and a follow-up needs both.
    """
    rid = str(request_id or "")[:60]
    if not rid or not MEMORY_ENABLED:
        return False
    try:
        conn = _conn()
        existing = conn.execute(
            "SELECT artifacts FROM work_requests WHERE request_id = ?",
            (rid,)).fetchone()
        merged = []
        if existing and existing["artifacts"]:
            try:
                merged = json.loads(existing["artifacts"]) or []
            except Exception:
                merged = []
        known = {json.dumps(r, sort_keys=True, default=str) for r in merged}
        for row in _artifact_rows(artifacts):
            signature = json.dumps(row, sort_keys=True, default=str)
            if signature not in known:
                known.add(signature)
                merged.append(row)
        conn.execute(
            "UPDATE work_requests SET status = ?, closed_at = ?, summary = ?, "
            "artifacts = ?, evidence = ?, provenance = ?, updated_at = ? "
            "WHERE request_id = ?",
            (str(status or "unknown")[:40], time.time(),
             mask_secrets(str(summary or ""))[:600],
             json.dumps(merged[:_ARTIFACT_MAX], ensure_ascii=False,
                        default=str) if merged else None,
             json.dumps([mask_secrets(str(e))[:160]
                         for e in (evidence or [])][:8], ensure_ascii=False)
             if evidence else None,
             mask_secrets(str(provenance or engine or ""))[:120] or None,
             time.time(), rid),
        )
        conn.commit()
        return True
    except Exception as exc:
        logging.debug("[MEMORY] close_request failed: %s", exc)
        return False


def _work_row(row):
    data = _row(row)
    for field in ("artifacts", "evidence"):
        raw = data.get(field)
        if raw:
            try:
                data[field] = json.loads(raw)
            except Exception:
                data[field] = []
        else:
            data[field] = []
    return data


def get_work_request(request_id):
    """One identified request row, or None (F07)."""
    if not MEMORY_ENABLED or not request_id:
        return None
    try:
        row = _conn().execute(
            "SELECT * FROM work_requests WHERE request_id = ?",
            (str(request_id)[:60],)).fetchone()
        return _work_row(row) if row else None
    except Exception as exc:
        logging.debug("[MEMORY] get_work_request failed: %s", exc)
        return None


def work_events(request_id=None, kind=None, limit=50):
    """The event timeline, optionally for one identified request (F07)."""
    if not MEMORY_ENABLED:
        return []
    try:
        sql = "SELECT * FROM events"
        clauses, args = [], []
        if request_id:
            clauses.append("request_id = ?")
            args.append(str(request_id)[:60])
        if kind:
            clauses.append("kind = ?")
            args.append(str(kind)[:40])
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY ts ASC, id ASC LIMIT ?"
        args.append(int(limit))
        return [_row(r) for r in _conn().execute(sql, args).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] work_events failed: %s", exc)
        return []


def recent_work_requests(limit=5, statuses=None):
    """Most recent identified requests, newest first (F07)."""
    if not MEMORY_ENABLED:
        return []
    try:
        sql = "SELECT * FROM work_requests"
        args = []
        if statuses:
            sql += " WHERE status IN (%s)" % ",".join(
                "?" for _ in statuses)
            args.extend(str(s)[:40] for s in statuses)
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(limit))
        return [_work_row(r) for r in _conn().execute(sql, args).fetchall()]
    except Exception as exc:
        logging.debug("[MEMORY] recent_work_requests failed: %s", exc)
        return []


#: Follow-ups that ask about EARLIER WORK rather than requesting something new.
_WORK_FOLLOWUP_RE = re.compile(
    r"\b(?:that|the|this|last|previous|earlier|same)\s+"
    r"(?:report|task|job|run|search|research|file|document|sheet|result|"
    r"thing|one)\b|\b(?:where is|what happened to|show me|send me|"
    r"open (?:it|that|the report)|its? (?:path|sources?|links?))\b",
    re.IGNORECASE)


def looks_like_work_followup(text):
    return bool(text and _WORK_FOLLOWUP_RE.search(str(text)))


def retrieve_work(query=None, request_id=None, limit=3):
    """Unified retrieval for normal chat (F07).

    Returns {"request":…, "result":…, "artifacts":[…], "sources":[…],
    "events":[…]} for the identified request being asked about: the newest
    one, or the newest one whose text overlaps the query.
    """
    if not MEMORY_ENABLED:
        return {}
    request = get_work_request(request_id) if request_id else None
    if request is None:
        rows = recent_work_requests(limit=max(int(limit), 1) * 4)
        if query:
            terms = {t for t in re.findall(r"[a-z0-9]+", str(query).lower())
                     if len(t) > 2}
            def score(row):
                text = ("%s %s" % (row.get("text") or "",
                                   row.get("summary") or "")).lower()
                words = set(re.findall(r"[a-z0-9]+", text))
                return len(terms & words) if terms else 0
            scored = sorted(rows, key=score, reverse=True)
            request = scored[0] if scored and score(scored[0]) else (
                rows[0] if rows else None)
        elif rows:
            request = rows[0]
    if not request:
        return {}
    rid = request.get("request_id")
    events = work_events(request_id=rid, limit=50)
    result = None
    for event in reversed(events):
        if event.get("kind") in ("work_result", "work_suspension",
                                 "task_result"):
            result = event
            break
    artifacts = list(request.get("artifacts") or [])
    sources = [a for a in artifacts
               if isinstance(a, dict) and (a.get("url") or a.get("source"))]
    return {"request": request, "result": result, "artifacts": artifacts,
            "sources": sources, "events": events}


def work_context(query, budget=600):
    """Bounded prompt block for a follow-up about earlier work (F07).

    Reads the persisted rows, so it survives a restart. Empty when the query is
    not a work follow-up or nothing relevant is stored.
    """
    if not MEMORY_ENABLED or not query:
        return ""
    try:
        if not looks_like_work_followup(query):
            return ""
        work = retrieve_work(query)
    except Exception:
        return ""
    request = (work or {}).get("request")
    if not request:
        return ""
    lines = ["Earlier work you can refer to:"]
    total = len(lines[0])
    text = (request.get("text") or "").strip()
    if text:
        lines.append("- Request: %s" % text[:160])
    status = request.get("status")
    summary = (request.get("summary") or "").strip()
    if status or summary:
        lines.append("- Outcome: %s%s" % (status or "unknown",
                                          (" — %s" % summary[:160]) if summary else ""))
    for artifact in (work.get("artifacts") or [])[:6]:
        location = artifact.get("path") or artifact.get("url") or \
            artifact.get("value")
        if not location:
            continue
        kind = artifact.get("kind") or "artifact"
        lines.append("- %s: %s" % (kind, location))
    for source in (work.get("sources") or [])[:6]:
        url = source.get("url") or source.get("source")
        title = source.get("title") or ""
        if url and url not in "\n".join(lines):
            lines.append("- source: %s%s" % (url, (" (%s)" % title) if title else ""))
    for line in list(lines):
        if total + len(line) > budget:
            lines.remove(line)
            continue
        total += len(line)
    return "\n".join(lines) if len(lines) > 1 else ""


def chat_projection(limit=20):
    """The chat history DERIVED from work events (F07).

    Chat is a projection of the work-event store, not a second, separate
    record: the same rows answer "what did we do" and "what was said".
    """
    if not MEMORY_ENABLED:
        return []
    try:
        rows = _conn().execute(
            "SELECT ts, kind, summary, request_id FROM events "
            "WHERE kind IN ('work_request', 'chat_exchange', 'work_result', "
            "'work_suspension', 'task_result') "
            "ORDER BY ts DESC, id DESC LIMIT ?", (int(limit),)).fetchall()
        messages = []
        for row in reversed(rows):
            kind = row["kind"]
            if kind in ("work_request", "chat_exchange"):
                role = "user"
            else:
                role = "assistant"
            messages.append({"role": role, "content": row["summary"],
                             "request_id": row["request_id"], "kind": kind,
                             "ts": row["ts"]})
        return messages
    except Exception as exc:
        logging.debug("[MEMORY] chat_projection failed: %s", exc)
        return []


def episodic_projection(limit=20):
    """Verified WORK episodes for episodic derivation (F07 + F06).

    Only terminal events with an identified request are returned, so a
    removed/forgotten request cannot resurrect as a fact.
    """
    if not MEMORY_ENABLED:
        return []
    try:
        rows = _conn().execute(
            "SELECT w.request_id, w.route, w.text, w.status, w.summary, "
            "w.artifacts, w.closed_at FROM work_requests w "
            "WHERE w.status NOT IN ('open', 'suspended') "
            "ORDER BY w.closed_at DESC, w.ts DESC LIMIT ?",
            (int(limit),)).fetchall()
        episodes = []
        for row in rows:
            artifacts = []
            if row["artifacts"]:
                try:
                    artifacts = json.loads(row["artifacts"]) or []
                except Exception:
                    artifacts = []
            episodes.append({
                "request_id": row["request_id"], "route": row["route"],
                "request": row["text"], "status": row["status"],
                "summary": row["summary"], "artifacts": artifacts,
                "closed_at": row["closed_at"],
            })
        return episodes
    except Exception as exc:
        logging.debug("[MEMORY] episodic_projection failed: %s", exc)
        return []












