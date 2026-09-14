"""Fable-5 audit G5 / F48 — one shared provenance vocabulary.

Jarvis mixes things it has genuinely established with things it merely
assumed, and used to present both the same way. Every claim now carries where
it came from, using the SAME words on every path — screen Q&A, quick search
and deep research — so the overlay and the spoken answer can say "I saw this"
versus "I inferred this" versus "I looked this up".

The vocabulary:

``observed``
    Read directly off a source Jarvis opened, or seen on the user's screen.
``quoted``
    Lifted verbatim from a source — the strongest form of observed.
``inferred``
    The model's own reasoning. Useful, but NOT a fact anyone stated.
``secondary``
    An AI-generated summary of other people's sources. It is a restatement,
    never independent corroboration of anything (F48).
``externally_checked``
    A real lookup actually ran and answered the question. Nothing may carry
    this label unless a lookup tool was invoked (F48).

ROUND 2 (the audit's remaining F48 defects, fixed here):

  * "Missing labels become observed" — an absent or unrecognised label now
    degrades to ``inferred`` (uncertainty-preserving default), never to a
    stronger claim than the evidence supports;
  * "generic prefixes masquerade as supporting spans" — a span must VERBATIM
    appear in its source and actually overlap the claim; boilerplate and
    navigation prefixes are rejected explicitly (:func:`span_supports_claim`);
  * "boilerplate counts as corroboration" — mirrors, syndicated copies and
    AI summaries are collapsed/counted separately, so two copies of one
    article can never read as two independent sources
    (:func:`independent_corroboration`);
  * "Synthesis lacks claim/source contracts ... APIs lose fields" — a
    claim-level contract (:class:`Claim`, :class:`SourceSpan`) with lookup
    references and lossless ``to_dict``/``from_dict`` round trips
    (:func:`claim_to_evidence` / :func:`evidence_to_claim`);
  * "UTC-named timestamps are naive local time" — :func:`utc_now_iso` returns
    an AWARE UTC timestamp and :func:`parse_timestamp` never hands back a
    naive datetime.
"""

import re
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from urllib.parse import urlsplit

PROVENANCE_OBSERVED = "observed"
PROVENANCE_QUOTED = "quoted"
PROVENANCE_INFERRED = "inferred"
PROVENANCE_SECONDARY = "secondary"
PROVENANCE_EXTERNALLY_CHECKED = "externally_checked"

#: Every accepted value — used to reject a model's invented labels.
PROVENANCE_VALUES = (
    PROVENANCE_OBSERVED,
    PROVENANCE_QUOTED,
    PROVENANCE_INFERRED,
    PROVENANCE_SECONDARY,
    PROVENANCE_EXTERNALLY_CHECKED,
)

#: Labels that assert something was established from a source. Anything else
#: (including a missing or invented label) must not be promoted to one of
#: these — F48: "uncertainty-preserving defaults".
STRONG_PROVENANCE = (
    PROVENANCE_OBSERVED,
    PROVENANCE_QUOTED,
    PROVENANCE_EXTERNALLY_CHECKED,
)

#: Human-readable wording for the overlay and the spoken answer.
PROVENANCE_LABELS = {
    PROVENANCE_OBSERVED: "observed",
    PROVENANCE_QUOTED: "quoted from the source",
    PROVENANCE_INFERRED: "inferred — not stated anywhere on screen",
    PROVENANCE_SECONDARY: "AI summary — secondary source",
    PROVENANCE_EXTERNALLY_CHECKED: "externally checked",
}

#: What we say when nothing corroborated a claim.
UNCERTAINTY_UNVERIFIED = "unverified"
UNCERTAINTY_SINGLE_SOURCE = "single source — not independently verified"
UNCERTAINTY_CORROBORATED = "independently corroborated"

#: Boilerplate that appears on nearly every page and supports no claim.
_BOILERPLATE_PATTERNS = (
    r"^\s*(skip to (main )?content|jump to (main )?content)\s*$",
    r"^\s*(accept|manage) (all )?cookies?\s*$",
    r"^\s*cookie(s)? (policy|settings|preferences)\s*$",
    r"^\s*(privacy|terms|legal)( (policy|notice|of service))?\s*$",
    r"^\s*(sign|log) ?(in|up|out)\s*$",
    r"^\s*subscribe( to (our|the) newsletter)?\s*$",
    r"^\s*(home|about|contact|menu|search|share|advertisement)\s*$",
    r"^\s*(all rights reserved|copyright ©?\s*\d{4})\s*$",
    r"^\s*(enable javascript|javascript is (required|disabled))\s*$",
)

_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9]{3,}")

#: Words too common to prove that a span supports a claim.
_STOPWORDS = frozenset((
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "its", "his", "her", "their", "they",
    "you", "your", "our", "out", "into", "over", "than", "then", "them",
    "there", "here", "what", "which", "who", "when", "where", "why", "how",
    "all", "any", "can", "will", "would", "should", "could", "about", "more",
    "most", "some", "such", "only", "also", "been", "being", "does", "did",
))


def normalise_provenance(value, default=PROVENANCE_INFERRED):
    """Coerce a model-supplied provenance into the vocabulary.

    A model that invents ``"definitely true"`` must not get it printed — an
    unknown label falls back to *default*. F48: the default is now
    ``inferred``, because a MISSING label is not evidence that something was
    observed (previously an unlabelled item was presented as "seen on
    screen").
    """
    text = str(value or "").strip().lower()
    return text if text in PROVENANCE_VALUES else default


def provenance_label(value):
    """Wording for *value* suitable for showing to the user."""
    return PROVENANCE_LABELS.get(
        normalise_provenance(value), PROVENANCE_LABELS[PROVENANCE_INFERRED])


def utc_now_iso():
    """AWARE UTC timestamp for *retrieved_at* / *observed_at* fields.

    F48: the name said UTC while the value was naive LOCAL time — an
    ambiguous instant that two machines (or a log vs. an API payload) could
    read differently. The offset is now explicit.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value, assume_utc=True):
    """Parse *value* into an AWARE datetime (None when unparseable).

    A naive input is given UTC tzinfo rather than silently treated as local
    time, so comparisons and round trips cannot shift an observation.
    """
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        if not assume_utc:
            return None
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def is_aware_timestamp(value):
    """True when *value* names an unambiguous instant."""
    parsed = parse_timestamp(value)
    return parsed is not None and parsed.tzinfo is not None


def normalise_text(text):
    """Whitespace/case-normalized text for verbatim span comparison."""
    return " ".join(str(text or "").split()).lower()


def content_words(text):
    """Claim-bearing words: length-3+ tokens without stopwords."""
    return {word for word in _WORD_RE.findall(str(text or "").lower())
            if word not in _STOPWORDS}


def is_boilerplate(text):
    """True for navigation/consent/footer text that supports no claim."""
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return True
    if len(stripped) <= 24 and _BOILERPLATE_RE.match(stripped):
        return True
    return bool(_BOILERPLATE_RE.match(stripped))


def source_domain(url):
    """Registrable-ish domain of *url* ('' when there is no host).

    Mirrors are different HOSTS carrying the same content — the host is what
    "independent source" is counted on, so it is derived here, once.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        host = urlsplit(text if "//" in text else "//" + text).hostname or ""
    except Exception:
        return ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def spans_support(quote, claim, source_text=None, min_overlap=1):
    """``(supported, reason)`` — does *quote* actually support *claim*?

    F48: a span is only a supporting span when it
      * is non-empty and not boilerplate/navigation text,
      * appears VERBATIM in its source (when the source text is available),
      * and shares claim-bearing vocabulary with the claim — a generic page
        prefix ("Home About Contact") must not masquerade as support.
    """
    text = " ".join(str(quote or "").split())
    if not text:
        return False, "empty span"
    if is_boilerplate(text):
        return False, "boilerplate/navigation text, not a supporting span"
    if source_text is not None:
        haystack = normalise_text(source_text)
        if haystack and normalise_text(text) not in haystack:
            return False, "span does not appear verbatim in its source"
    claim_words = content_words(claim)
    span_words = content_words(text)
    if claim_words and len(span_words & claim_words) < max(1, min_overlap):
        prefix_only = bool(span_words) and len(text) <= 120
        if prefix_only:
            return False, ("generic prefix — shares no claim-bearing words "
                           "with the claim")
        return False, "span shares no claim-bearing words with the claim"
    return True, "verbatim span overlapping the claim"


# ── claim / source contract (F48) ──────────────────────────────────────────

@dataclass(frozen=True)
class SourceSpan:
    """One claim-level source span: where a statement came from, verbatim."""

    url: str = ""
    quote: str = ""
    selector: str = ""
    publisher: str = ""
    retrieved_at: str = ""
    independent: bool = True

    def to_dict(self):
        return {
            "url": self.url,
            "quote": self.quote,
            "selector": self.selector,
            "publisher": self.publisher,
            "retrieved_at": self.retrieved_at,
            "independent": bool(self.independent),
            "domain": source_domain(self.url),
        }

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        return cls(
            url=str(data.get("url") or ""),
            quote=str(data.get("quote") or ""),
            selector=str(data.get("selector") or ""),
            publisher=str(data.get("publisher") or ""),
            retrieved_at=str(data.get("retrieved_at") or ""),
            independent=bool(data.get("independent", True)),
        )


@dataclass(frozen=True)
class Claim:
    """A claim with its provenance, supporting spans and lookup reference.

    ``to_dict``/``from_dict`` are LOSSLESS (F48: "APIs lose fields",
    "metadata survives round trips"): every field an answer carries —
    including spans, the lookup reference, uncertainty and aware timestamps —
    survives a JSON round trip unchanged.
    """

    text: str = ""
    provenance: str = PROVENANCE_INFERRED
    spans: tuple = ()
    lookup: dict = dataclass_field(default_factory=dict)
    uncertainty: str = UNCERTAINTY_UNVERIFIED
    observed_at: str = ""
    retrieved_at: str = ""
    corroboration: dict = dataclass_field(default_factory=dict)
    relevance: float = 0.0

    def to_dict(self):
        return {
            "text": self.text,
            "provenance": normalise_provenance(self.provenance),
            "spans": [span.to_dict() if isinstance(span, SourceSpan)
                      else SourceSpan.from_dict(span).to_dict()
                      for span in (self.spans or ())],
            "lookup": dict(self.lookup or {}),
            "uncertainty": self.uncertainty or UNCERTAINTY_UNVERIFIED,
            "observed_at": self.observed_at,
            "retrieved_at": self.retrieved_at,
            "corroboration": dict(self.corroboration or {}),
            "relevance": float(self.relevance or 0.0),
        }

    @classmethod
    def from_dict(cls, data):
        data = data if isinstance(data, dict) else {}
        spans = tuple(
            SourceSpan.from_dict(span) for span in (data.get("spans") or ())
        )
        lookup = data.get("lookup")
        corroboration = data.get("corroboration")
        return cls(
            text=str(data.get("text") or ""),
            provenance=normalise_provenance(data.get("provenance")),
            spans=spans,
            lookup=dict(lookup) if isinstance(lookup, dict) else {},
            uncertainty=str(data.get("uncertainty") or UNCERTAINTY_UNVERIFIED),
            observed_at=str(data.get("observed_at") or ""),
            retrieved_at=str(data.get("retrieved_at") or ""),
            corroboration=(dict(corroboration)
                           if isinstance(corroboration, dict) else {}),
            relevance=float(data.get("relevance") or 0.0),
        )


def claim_to_evidence(claim, title="", url=""):
    """The overlay/API evidence shape for a :class:`Claim` — all fields kept.

    Kept in the shared vocabulary module so the screen path, quick search and
    deep research all produce the SAME shape (F48: "lossless API/synthesis
    propagation"). Never drops spans, lookup, uncertainty or timestamps.
    """
    if not isinstance(claim, Claim):
        claim = Claim.from_dict(claim)
    data = claim.to_dict()
    primary = claim.spans[0] if claim.spans else None
    return {
        "result_title": title or (primary.publisher if primary else ""),
        "url": url or (primary.url if primary else ""),
        "provenance": data["provenance"],
        "summary": claim.text,
        "uncertainty": data["uncertainty"],
        "retrieved_at": (claim.retrieved_at or claim.observed_at),
        "spans": data["spans"],
        "lookup": data["lookup"],
        "corroboration": data["corroboration"],
        "relevance": data["relevance"],
        "observed_at": data["observed_at"],
    }


def evidence_to_claim(item):
    """Inverse of :func:`claim_to_evidence` — nothing silently dropped."""
    item = item if isinstance(item, dict) else {}
    spans = tuple(SourceSpan.from_dict(span)
                  for span in (item.get("spans") or ()))
    if not spans and (item.get("url") or item.get("quote")):
        spans = (SourceSpan(url=str(item.get("url") or ""),
                            quote=str(item.get("quote") or ""),
                            publisher=str(item.get("result_title") or "")),)
    lookup = item.get("lookup")
    if not isinstance(lookup, dict):
        lookup = {}
        if item.get("url"):
            lookup["url"] = str(item["url"])
    corroboration = item.get("corroboration")
    return Claim(
        text=str(item.get("summary") or item.get("text") or ""),
        provenance=normalise_provenance(item.get("provenance")),
        spans=spans,
        lookup=dict(lookup),
        uncertainty=str(item.get("uncertainty") or UNCERTAINTY_UNVERIFIED),
        observed_at=str(item.get("observed_at") or ""),
        retrieved_at=str(item.get("retrieved_at") or ""),
        corroboration=(dict(corroboration)
                       if isinstance(corroboration, dict) else {}),
        relevance=float(item.get("relevance") or 0.0),
    )


def build_claim(text, sources=(), provenance=PROVENANCE_OBSERVED, lookup=None,
                observed_at="", relevance=0.0):
    """One claim with VALIDATED spans and an explicit corroboration verdict.

    F48: this is the claim/source contract the synthesis paths share. A span
    that is boilerplate, not verbatim, or not actually about the claim is
    REJECTED (and the rejection is reported), and a claim labelled
    observed/quoted/externally-checked with no surviving span degrades to
    ``inferred`` — a label is never stronger than its evidence.
    """
    claim_text = " ".join(str(text or "").split())
    spans = []
    rejected = []
    for source in sources or ():
        item = source if isinstance(source, dict) else {"quote": str(source)}
        span = SourceSpan(
            url=str(item.get("url") or ""),
            quote=" ".join(str(item.get("quote") or item.get("summary")
                               or item.get("text") or "").split()),
            selector=str(item.get("selector") or ""),
            publisher=str(item.get("publisher") or item.get("result_title") or ""),
            retrieved_at=str(item.get("retrieved_at") or ""),
            independent=bool(item.get("independent", True)),
        )
        supported, reason = spans_support(
            span.quote, claim_text, item.get("source_text"))
        if supported:
            spans.append(span)
        else:
            rejected.append({"url": span.url, "domain": source_domain(span.url),
                             "reason": reason})
    corroboration = independent_corroboration(sources)
    if rejected:
        corroboration["rejected_spans"] = rejected
    label = normalise_provenance(provenance)
    if label in STRONG_PROVENANCE and not spans and sources:
        label = PROVENANCE_INFERRED
    lookup = dict(lookup or {})
    return Claim(
        text=claim_text,
        provenance=label,
        spans=tuple(spans),
        lookup=lookup,
        uncertainty=corroboration["level"],
        observed_at=str(observed_at or ""),
        retrieved_at=str(lookup.get("retrieved_at") or ""),
        corroboration=corroboration,
        relevance=float(relevance or 0.0),
    )


def independent_corroboration(sources):
    """Count INDEPENDENT corroboration across *sources* (F48).

    A source is independent only when it is a primary (non-``secondary``, i.e.
    not an AI summary) report on its own host, with its own text. Mirrors —
    the same content republished on another host — and boilerplate collapse
    into the origin, so "three sites said it" can never be produced by one
    syndicated article. Returns the verdict plus the explicit accounting.
    """
    independent_domains = {}
    seen_quotes = set()
    mirrors = []
    secondary = []
    boilerplate = []
    for index, source in enumerate(sources or ()):
        item = source if isinstance(source, dict) else {"quote": str(source)}
        provenance = normalise_provenance(item.get("provenance"))
        quote = str(item.get("quote") or item.get("summary")
                    or item.get("text") or "")
        url = str(item.get("url") or "")
        domain = source_domain(url) or str(item.get("publisher") or "").strip()
        label = domain or ("source#%d" % index)
        if provenance == PROVENANCE_SECONDARY:
            secondary.append(label)
            continue
        if quote and is_boilerplate(quote):
            boilerplate.append(label)
            continue
        key = normalise_text(quote)[:400] if quote else ""
        if key and key in seen_quotes:
            # Same words on another host: a mirror/syndicated copy, not an
            # independent report.
            mirrors.append(label)
            continue
        if domain and domain in independent_domains:
            mirrors.append(label)
            continue
        if key:
            seen_quotes.add(key)
        independent_domains[domain or label] = {
            "key": key, "domain": domain, "quote": quote[:300]}
    independent = len(independent_domains)
    if independent >= 2:
        level = UNCERTAINTY_CORROBORATED
    elif independent == 1:
        level = UNCERTAINTY_SINGLE_SOURCE
    else:
        level = UNCERTAINTY_UNVERIFIED
    return {
        "independent": independent,
        "corroborated": independent >= 2,
        "level": level,
        "domains": sorted(independent_domains),
        "mirrors": mirrors,
        "secondary_sources": secondary,
        "boilerplate": boilerplate,
        "considered": len(list(sources or ())),
    }
