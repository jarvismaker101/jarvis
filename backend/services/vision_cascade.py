"""One eligible-provider vision cascade (Fable-5 audit F37).

CURRENT (audit F37): the unrelated Groq-key guard is gone, but availability
was still decided per call site and per branch:

  * only the registry-selected provider (or openrouter/fireworks/groq special
    cases) was treated as a *primary*; every other configured provider could
    only be reached through a hard-coded ``gemini -> groq`` tail, so a
    supported single-provider installation still ended in an unconditional
    Groq attempt with no Groq key;
  * the final Groq fallback sat OUTSIDE any try/except, so an adapter
    exception escaped the cascade instead of advancing to the next provider;
  * any NONEMPTY text counted as success (``_extract_*_content(result) is not
    None``), so malformed JSON from the first provider ended the cascade;
  * the selected provider could be attempted twice (primary branch + the same
    provider again as the tail fallback);
  * the caller only learned "no response" — never which providers were
    actually attempted.

CHANGE: one place decides (a) which providers are ELIGIBLE for a role and in
what order, (b) how each attempt is dispatched (caller-supplied adapters),
(c) whether a response passes the caller's schema validator, and (d) the
ACTUAL-ATTEMPT metadata. Every attempt is wrapped, every provider is
attempted at most once, the total number of attempts is bounded, and an
ineligible provider is never dispatched at all (so a provider's unrelated key
is never required).

Pure decision/dispatch module: provider registry reads are lazy imports, so
importing this module touches no network and no configuration. Callers own
the adapters and the response-shape validation — this module owns eligibility,
ordering, bounding and the attempt record.
"""

import logging

#: The registry role whose capability requirements gate this cascade.
VISION_ROLE = "vision"
#: The capability every provider in this cascade must be able to serve.
REQUIRED_CAPABILITY = "vision_input"

#: Fallback order AFTER the registry-selected provider. Deterministic (never
#: dictated by dict ordering) and bounded: a vision call is on the user's
#: critical path, so the cascade never walks an unbounded provider list.
FALLBACK_ORDER = ("gemini", "fireworks", "groq", "openrouter")

#: Hard bound on DISPATCHED attempts per call (skips are free — an ineligible
#: provider is rejected before any adapter runs).
MAX_ATTEMPTS = 3

#: Attempt outcomes, named once so callers/tests can rely on them.
OUTCOME_SERVED = "served"
OUTCOME_EMPTY = "empty"
OUTCOME_MALFORMED = "malformed"
OUTCOME_ERROR = "error"
OUTCOME_SKIPPED = "skipped"


def _normalize(provider):
    return str(provider or "").strip().lower()


def vision_role_providers():
    """Provider ids the registry allows for the vision role.

    Falls back to the shipped vision providers when the registry cannot be
    read (a corrupt settings file must not make the cascade blind).
    """
    try:
        from backend.services import model_registry
        allowed = model_registry.get_allowed_providers_for_role(VISION_ROLE)
        if allowed:
            return [str(p).strip().lower() for p in allowed if str(p).strip()]
    except Exception as exc:  # noqa: BLE001 - degrade, never crash a vision call
        logging.warning("[VISION] role provider discovery failed: %s", exc)
    return list(FALLBACK_ORDER)


def provider_eligible(provider, role_providers=None):
    """``(eligible, reason)`` — may *provider* serve a vision call at all?

    Three independent gates, all of which must pass:
      1. the provider is allowed for the vision role (registry allowlist);
      2. the provider declares the role's required capability (F49);
      3. a credential/daemon for it is actually configured.
    A provider that fails any gate is NEVER dispatched: that is what keeps a
    single-provider installation free of every other provider's key.
    """
    pid = _normalize(provider)
    if not pid:
        return False, "no provider selected"
    allowed = role_providers if role_providers is not None else vision_role_providers()
    if pid not in [ _normalize(p) for p in allowed ]:
        return False, "not allowed for the vision role"
    try:
        from backend.services import model_registry
        model_registry.validate_role_capabilities(VISION_ROLE, pid, "")
    except Exception as exc:  # noqa: BLE001 - any failure means "not eligible"
        return False, "lacks the vision capability (%s)" % (exc,)
    if not provider_available(pid):
        return False, "no credentials configured"
    return True, "eligible"


def provider_available(provider):
    """True when *provider* has a usable credential/daemon right now.

    Read live on every call (never cached at import) so a key added or
    removed while Jarvis runs changes eligibility immediately. Gemini is
    asked through its own client (it owns the availability rule); every other
    vision provider is a keyed HTTP adapter, so the registry's masked
    credential record is the single source of truth for "is it configured".
    A REGISTERED custom provider is ready only with BOTH its stored key and
    its stored base URL — a half-written record must never look ready.
    """
    pid = _normalize(provider)
    if not pid:
        return False
    if pid == "gemini":
        try:
            from backend.services import gemini_client
            return bool(gemini_client.is_available())
        except Exception:
            return False
    try:
        from backend.services import model_registry
        env_ids = {_normalize(p) for p in (model_registry.ENV_PROVIDERS or {})}
        custom_ids = {_normalize(p)
                      for p in (model_registry.custom_provider_ids() or [])}
        if pid not in env_ids and pid not in custom_ids:
            return False
        key, base_url = model_registry.get_provider_credentials(pid)
        # Configured means "has a usable key" - a base URL alone must not make
        # a keyless provider look ready (env base URLs are canonical constants
        # now, so an or-condition here would dispatch an unconfigured
        # provider). Custom providers always store their key WITH the URL.
        if pid in custom_ids:
            return bool(key) and bool(base_url)
        return bool(key)
    except Exception:
        return False


def _make_custom_dispatcher(provider_id):
    """One ``prompt, image, model, max_tokens, response_format`` adapter bound
    to a registered custom provider id (the dispatcher map's contract)."""
    def dispatch(prompt, image_data_url, model, max_completion_tokens,
                 response_format):
        try:
            from backend.services import model_registry
            from backend.services.openai_compat_client import (
                ask_openai_compat_vision)
            api_key, base_url = model_registry.get_provider_credentials(
                provider_id)
            if not api_key or not base_url:
                return {}
            return ask_openai_compat_vision(
                prompt, image_data_url, model, base_url, api_key,
                max_completion_tokens=max_completion_tokens,
                response_format=response_format)
        except Exception as exc:  # noqa: BLE001 - one bad provider advances
            logging.warning("[VISION] custom provider '%s' failed: %s",
                            provider_id, exc)
            return {}
    return dispatch


def custom_vision_dispatchers():
    """``{provider id: adapter}`` for every REGISTERED custom provider.

    A user-added OpenAI-compatible provider rides ONE generic vision adapter
    (the standard ``image_url`` content part) at every call site, so a call
    site merges this map instead of special-casing custom ids. Credentials
    resolve lazily per call through the registry — a provider removed while
    Jarvis runs dispatches to nothing rather than to stale keys. Call sites
    keep ownership of their shipped adapters and their response-shape checks.
    """
    try:
        from backend.services import model_registry
        ids = model_registry.custom_provider_ids()
    except Exception as exc:  # noqa: BLE001 - degrade, never crash a vision call
        logging.warning("[VISION] custom provider discovery failed: %s", exc)
        return {}
    return {pid: _make_custom_dispatcher(pid) for pid in ids}


def vision_provider_order(selected=None):
    """The deterministic provider order for one call: selected first, then
    FALLBACK_ORDER (deduplicated). Nothing here is dispatched blindly — every
    entry still has to pass the eligibility gates below."""
    if isinstance(selected, dict):
        selected_provider = _normalize(selected.get("provider"))
    else:
        selected_provider = _normalize(selected)
    order = []
    if selected_provider:
        order.append(selected_provider)
    for candidate in FALLBACK_ORDER:
        if candidate not in order:
            order.append(candidate)
    return order


def provider_candidates(selected=None, availability=None):
    """``(eligible, skipped)`` for one call.

    *eligible* is the ordered list of dispatchable ``{"provider", "model",
    "reason"}`` candidates. *skipped* is ``[{"provider", "reason"}, ...]`` for
    every provider in the order that failed a gate — recorded so the attempt
    report can distinguish "we tried it and it failed" from "it was never
    eligible", which is exactly what a single-provider installation needs.
    """
    if isinstance(selected, dict):
        selected_provider = _normalize(selected.get("provider"))
        selected_model = str(selected.get("model") or "").strip() or None
    else:
        selected_provider, selected_model = _normalize(selected), None

    role_providers = vision_role_providers()
    eligible = []
    skipped = []
    for pid in vision_provider_order(selected):
        ok, reason = _candidate_eligible(pid, role_providers, availability)
        if not ok:
            skipped.append({"provider": pid, "reason": reason})
            continue
        eligible.append({
            "provider": pid,
            "model": selected_model if pid == selected_provider else None,
            "reason": reason,
        })
    return eligible, skipped


def eligible_vision_providers(selected=None, availability=None):
    """Ordered ELIGIBLE (provider, model) candidates for one vision call.

    The registry-selected provider comes first (when it is eligible at all);
    the rest of FALLBACK_ORDER follows, filtered by the same gates. *selected*
    is the ``{"provider", "model"}`` pair the caller resolved; *availability*
    is an optional callable/mapping used by tests to pin credential state.

    Returns a list of ``{"provider", "model", "reason"}`` dicts. The model is
    the SELECTED model only for the selected provider — a fallback provider
    uses its own adapter default rather than being sent another provider's
    model name (the previous cascade's cross-provider model leak).
    """
    return provider_candidates(selected=selected, availability=availability)[0]


def _candidate_eligible(pid, role_providers, availability):
    if availability is not None:
        try:
            available = (availability.get(pid) if hasattr(availability, "get")
                         else availability(pid))
        except Exception:
            available = None
        if available is None:
            # Unknown in the injected snapshot: fall back to the live check.
            return provider_eligible(pid, role_providers)
        if not available:
            return False, "no credentials configured"
        if pid not in [_normalize(p) for p in role_providers]:
            return False, "not allowed for the vision role"
        try:
            from backend.services import model_registry
            model_registry.validate_role_capabilities(VISION_ROLE, pid, "")
        except Exception as exc:  # noqa: BLE001
            return False, "lacks the vision capability (%s)" % (exc,)
        return True, "eligible"
    return provider_eligible(pid, role_providers)


def ask_vision_with_fallback(prompt, image_data_url, dispatchers, validate,
                             selected=None, availability=None,
                             max_completion_tokens=800, response_format=None,
                             max_attempts=MAX_ATTEMPTS, attempts_out=None):
    """Run the bounded eligible-provider cascade for one vision request.

    *dispatchers* maps provider id -> ``callable(prompt, image_data_url,
    model, max_completion_tokens, response_format)`` returning the provider's
    raw result dict (``{}``/``None`` when the provider produced nothing).

    *validate* maps a raw result to ``(usable, reason)``; a result that is not
    usable ADVANCES the cascade instead of ending it (F37: malformed nonempty
    output no longer counts as an answer).

    Returns ``(result, report)``. *report* is the actual-attempt record::

        {"usable": bool, "provider": str|None, "model": str|None,
         "attempted": [provider, ...], "attempts": [{provider, model,
         outcome, detail}], "unavailable_reason": str}

    ``result`` is the serving provider's result when one served. When every
    eligible provider advanced but at least one returned nonempty-but-invalid
    output, the LAST such result is returned with ``report["usable"]`` False
    and ``report["degraded"]`` True — the caller keeps its own last-resort
    handling (e.g. using the raw text) while knowing it is not schema-valid.
    """
    candidates, skipped = provider_candidates(selected=selected,
                                              availability=availability)
    report = {
        "usable": False,
        "provider": None,
        "model": None,
        "attempted": [],
        "attempts": [],
        "degraded": False,
        "unavailable_reason": "",
    }
    # Ineligible providers are recorded (so the report can say exactly why a
    # provider was never called) but are NEVER dispatched — that is what keeps
    # another provider's missing key from mattering at all.
    for entry in skipped:
        report["attempts"].append({
            "provider": entry["provider"],
            "model": None,
            "outcome": OUTCOME_SKIPPED,
            "detail": entry["reason"],
        })

    dispatched = 0
    malformed_result = None
    for candidate in candidates:
        if dispatched >= max(1, int(max_attempts)):
            report["attempts"].append({
                "provider": candidate["provider"],
                "model": candidate["model"],
                "outcome": OUTCOME_SKIPPED,
                "detail": "attempt budget exhausted",
            })
            continue
        pid = candidate["provider"]
        dispatcher = (dispatchers or {}).get(pid)
        if dispatcher is None:
            report["attempts"].append({
                "provider": pid, "model": candidate["model"],
                "outcome": OUTCOME_SKIPPED,
                "detail": "no adapter is wired for this provider",
            })
            continue
        dispatched += 1
        report["attempted"].append(pid)
        try:
            result = dispatcher(
                prompt,
                image_data_url,
                candidate["model"],
                max_completion_tokens,
                response_format,
            )
        except Exception as exc:  # noqa: BLE001 - an adapter error advances
            logging.warning("[VISION] %s adapter error: %s", pid, exc)
            report["attempts"].append({
                "provider": pid, "model": candidate["model"],
                "outcome": OUTCOME_ERROR, "detail": str(exc)[:200],
            })
            continue
        if not result:
            report["attempts"].append({
                "provider": pid, "model": candidate["model"],
                "outcome": OUTCOME_EMPTY, "detail": "provider returned nothing",
            })
            continue
        usable, reason = _safe_validate(validate, result)
        if not usable:
            malformed_result = result
            report["attempts"].append({
                "provider": pid, "model": candidate["model"],
                "outcome": OUTCOME_MALFORMED,
                "detail": reason or "response did not match the expected schema",
            })
            continue
        report.update({
            "usable": True, "provider": pid, "model": candidate["model"],
            "attempts": report["attempts"] + [{
                "provider": pid, "model": candidate["model"],
                "outcome": OUTCOME_SERVED, "detail": "",
            }],
        })
        _publish(report, attempts_out)
        return result, report

    if malformed_result is not None:
        last = report["attempts"][-1] if report["attempts"] else {}
        report["degraded"] = True
        # NOTE: ``provider``/``model`` mean "served this request" and stay
        # None — nothing schema-valid served it. The degraded fallback is
        # named separately so callers cannot report it as the answering
        # provider.
        report["degraded_provider"] = last.get("provider")
        report["degraded_model"] = last.get("model")
    report["unavailable_reason"] = unavailable_reason(report)
    _publish(report, attempts_out)
    return malformed_result, report


def _safe_validate(validate, result):
    if validate is None:
        return True, ""
    try:
        usable, reason = validate(result)
    except Exception as exc:  # noqa: BLE001 - a broken validator must not serve
        return False, "validator failed: %s" % (exc,)
    return bool(usable), str(reason or "")


def _publish(report, attempts_out):
    """Copy the attempt record into a caller-supplied container, when given.

    Callers hand in a dict/list so the raw provider result keeps the exact
    shape its adapter returned (many call sites compare it directly), while
    the attempt metadata still travels back to whoever needs to report it.
    """
    if attempts_out is None:
        return
    try:
        if isinstance(attempts_out, dict):
            attempts_out.update(report)
        elif isinstance(attempts_out, list):
            attempts_out.extend(report["attempts"])
    except Exception:
        pass


def attempted_providers(report):
    """Provider ids actually dispatched, in attempt order."""
    if not report:
        return []
    return [str(p) for p in (report.get("attempted") or [])]


def unavailable_reason(report):
    """One honest sentence naming exactly what was and was not tried.

    F37 acceptance: total unavailability must be accurate and name the ACTUAL
    attempted providers — never a hard-coded provider that was never called
    and never a bare "the model failed".
    """
    attempts = list((report or {}).get("attempts") or [])
    attempted = attempted_providers(report)
    details = []
    for attempt in attempts:
        provider = attempt.get("provider") or "?"
        outcome = attempt.get("outcome") or "?"
        detail = str(attempt.get("detail") or "").strip()
        details.append("%s: %s%s" % (provider, outcome,
                                     " (%s)" % detail if detail else ""))
    if attempted:
        head = "no vision provider served the request (tried: %s)" % ", ".join(attempted)
    else:
        head = ("no vision provider is eligible right now "
                "(none was attempted)")
    if details:
        return head + " — " + "; ".join(details)
    return head
