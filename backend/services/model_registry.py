"""Runtime model registry for Jarvis text-chat responses.

Which (provider, model) answers chat messages must be switchable at runtime
from the UI with no backend restart, so it cannot live in the import-time
constants of the individual clients (gemini_client.GEMINI_CHAT_MODEL,
fireworks_client.DEFAULT_MODEL). This registry is the single runtime source
of truth: brain.py asks it per message.

Selection state persists to data/jarvis_settings.json (the data/ directory
is gitignored — custom provider API keys live ONLY there, never in .env,
never in git). Writes are atomic (tmp + os.replace), and every mutation
holds the module lock across the ENTIRE read-modify-write so concurrent
changes on different keys cannot interleave and clobber each other. Reads
are best-effort: a missing or corrupt file simply means "use the env
default".

Hard rule: API keys never leave this module. Everything returned to callers
(routes / UI) is masked — has_key booleans only.

F49 — capability-aware selection. A (provider, model) pair is only usable for
a role when the capabilities the ROLE requires are actually established for
that model, from one of four explicit sources: the provider's adapter
declaration (env providers only), the provider record's declared capability
set (custom providers, declared at add time), model-level rules for families
whose name states the capability, and capability metadata the provider itself
published in its model list (recorded by list_provider_models). Anything
unknown FAILS CLOSED: an unregistered provider has no capabilities at all, and
a model that cannot be shown to support a required capability is refused
rather than silently degraded. Resolution is validated at use time and
snapshotted from a single locked read, so a persisted selection that no longer
validates can never run and an in-flight call cannot see a half-applied
configuration change.
"""

import copy
import json
import logging
import os
import re
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.config import BASE_DIR, FIREWORKS_API_KEY, GEMINI_API_KEY
from backend.config import FISH_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY
from backend.services.gemini_client import GEMINI_CHAT_MODEL, GEMINI_MODEL
# [F56] The local Ollama endpoint has ONE owner (ollama_client, the native
# /api client the accessibility agent uses); this registry only appends the
# /v1 suffix the generic OpenAI-compatible adapter speaks.
from backend.services.ollama_client import OLLAMA_BASE_URL as _OLLAMA_BASE_URL
from backend import config as _config

SETTINGS_FILE = BASE_DIR / "data" / "jarvis_settings.json"

#: [F56] Ollama's OpenAI-compatible surface — what the generic chat adapter
#: (and the intent classifier's local hop) talk to.
OLLAMA_OPENAI_BASE_URL = str(_OLLAMA_BASE_URL or "").rstrip("/") + "/v1"

# Providers wired through environment keys (.env), not the settings file.
ENV_PROVIDERS = {
    "gemini": {"name": "Google Gemini"},
    "fireworks": {"name": "Fireworks AI"},
    "groq": {"name": "Groq"},
    "fish": {"name": "Fish Audio"},
    "gtts": {"name": "Google TTS (free)"},
    "openrouter": {"name": "OpenRouter"},
    "whisper": {"name": "Local Whisper"},
    "inworld": {"name": "Inworld STT"},
    # [F56] The local Ollama server: models run on THIS machine. It needs no
    # .env entry and no account, so it is registered as an env-style provider
    # whose credential is a placeholder (see _credentials_from) and whose
    # model list is whatever the local daemon reports.
    "ollama": {"name": "Ollama (local)"},
}

# Per-role allowlist (env providers). Custom providers are allowed for every
# LLM-backed role — the roles whose calls ride the OpenAI-compatible chat
# completions adapter (chat / vision / browser_tool / planner). The voice
# roles (tts / listening) drive dedicated audio engines with their own wire
# APIs, so a base-url + key provider cannot serve them.
_ROLE_ALLOWED_ENV = {
    # openrouter: same Gemini-family models over Cloudflare-fronted endpoints.
    # Added for chat after the 2026-09-23 incident: the direct Gemini API was
    # returning 7-45s+ (often >45s read timeouts) through the user's VPN relay
    # while openrouter answered the same model in ~1.4s, and the chat chain's
    # other leg (fireworks) is suspended. vision/browser_tool already allowed it.
    # [F56] ollama: fully local chat (llama3.2 3B / qwen3 1.7B) — no key, no
    # network, no cost, and immune to every cloud outage in this list.
    "chat": {"gemini", "fireworks", "openrouter", "ollama"},
    # Two interchangeable voice engines: Fish (metered, high quality) and
    # Google Translate TTS (free, key-less) as the zero-cost fallback.
    "tts": {"fish", "gtts"},
    "vision": {"gemini", "fireworks", "groq", "openrouter"},
    "browser_tool": {"gemini", "fireworks", "groq", "openrouter"},
    "listening": {"whisper", "inworld"},
    # F49 (G8): the planner drives the native tool-use orchestrator; it
    # MUST be a model with native tool calling + structured output, so the
    # allowlist is narrow and capability-validated below.
    "planner": {"fireworks"},
    # [F56] The intent classifier. Every message on the critical path goes
    # through it, so it is the one role where a LOCAL model is a real win:
    # these are the providers the router can already dispatch, plus ollama.
    # Deliberately NOT custom-provider capable: a user-added gateway has no
    # business on the first hop of every message, and the shipped cloud chain
    # (see services/intent.py) stays as the fallback either way.
    "intent": {"gemini", "fireworks", "groq", "openrouter", "ollama"},
}
_ROLE_ALLOWS_CUSTOM = {"chat", "vision", "browser_tool", "planner"}

_GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_FIREWORKS_MODELS_URL = "https://api.fireworks.ai/inference/v1/models"
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Static Fish TTS models (no public list API)
_FISH_TTS_STATIC_MODELS = [
    "s1",
    "s1-mini",
    "s2",
    "s2.1-pro",
    "s2.1-pro-free",
    "fish-speech-1.5",
]

# Roles that can be switched at runtime
VALID_ROLES = {"chat", "tts", "vision", "browser_tool", "listening", "planner",
               "intent"}

_ROLE_STORAGE_KEY = {
    "chat": "chat_model",
    "tts": "tts_model",
    "vision": "vision_model",
    "browser_tool": "browser_tool_model",
    "listening": "listening_model",
    "planner": "planner_model",
    "intent": "intent_model",
}

# ── F49 (G8): capability-aware selection ───────────────────────────────────
# What each role REQUIRES of the model.
ROLE_CAPABILITIES = {
    "chat": frozenset(("streaming",)),
    "tts": frozenset(("audio_output",)),
    "vision": frozenset(("vision_input",)),
    # F49: browser_tool used to require tools/schema only, so a TEXT-ONLY
    # model passed selection and the browser agent silently acted blind
    # whenever it took a screenshot. Vision is a hard requirement now.
    "browser_tool": frozenset(("tool_calling", "structured_output",
                               "vision_input")),
    "listening": frozenset(("speech_input",)),
    "planner": frozenset(("tool_calling", "structured_output", "streaming")),
    # [F56] The classifier's verdict IS a JSON object the router parses, so
    # structured output is the hard requirement. Streaming is deliberately NOT
    # required: classification is one short non-streaming call.
    "intent": frozenset(("structured_output",)),
}

# Capability names this module can reason about (used to validate explicit
# declarations so a typo can never widen a selection).
KNOWN_CAPABILITIES = frozenset((
    "streaming", "vision_input", "tool_calling", "structured_output",
    "audio_output", "speech_input",
))

# Adapter-level capability floors for the providers wired into the shipped
# clients. These are ADAPTER facts (the code path that will carry the call),
# not measurements of one deployed model: the per-model sources below narrow
# them whenever the model's family or the provider's own metadata says a
# capability is absent. [ASSUMPTION — the audit's flagged assumption: a
# provider that publishes no capability metadata cannot be measured from
# here; the adapter floor is the conservative, documented floor for models
# that carry no narrowing evidence.]
PROVIDER_CAPABILITIES = {
    "gemini": frozenset(("vision_input", "tool_calling", "structured_output",
                         "streaming")),
    "fireworks": frozenset(("vision_input", "tool_calling", "structured_output",
                            "streaming")),
    "groq": frozenset(("vision_input", "tool_calling", "structured_output",
                       "streaming")),
    "openrouter": frozenset(("vision_input", "tool_calling", "structured_output",
                             "streaming")),
    "fish": frozenset(("audio_output",)),
    # Google Translate TTS: free, key-less voice synthesis. Same adapter
    # capability as Fish — it is another audio_output engine for the tts
    # role, just one that needs no account.
    "gtts": frozenset(("audio_output",)),
    "whisper": frozenset(("speech_input",)),
    "inworld": frozenset(("speech_input",)),
    # [F56] The local Ollama server speaks the OpenAI-compatible dialect, so
    # the generic adapter carries it: streaming text, native tool calls and
    # JSON structured output are adapter facts. VISION is deliberately NOT
    # part of this floor — every model installed here today (llama3.2 3B,
    # qwen3 1.7B, deepseek-coder) is text-only, and the ollama provider is not
    # in the vision role's allowlist either, so a local text model can never
    # be selected to answer a screen question.
    "ollama": frozenset(("tool_calling", "structured_output", "streaming")),
}

# The capability set recorded with a user-added OpenAI-compatible provider
# when the caller does not declare one explicitly. It is stored IN the
# provider record (see add_custom_provider) so the authorization is explicit
# and reviewable — the shipped OpenAI-compatible adapter carries chat,
# tools, JSON schema and image parts, and the provider was added with a
# live-validated key. It is NEVER used as an implicit fallback for an
# unregistered provider id (F49: unknown providers inherit nothing).
_CUSTOM_PROVIDER_DEFAULT_CAPABILITIES = frozenset((
    "vision_input", "tool_calling", "structured_output", "streaming"))

# Model-level capability rules: (provider | "*", model-id regex,
# {capability: bool}). Applied in order (last match wins) on top of the
# provider declaration. They cover the two cases a provider's own metadata
# cannot: families whose NAME states the capability outright, and providers
# (Fireworks) whose model list publishes no tool/vision metadata at all.
# Capability metadata observed from the provider's list API always wins over
# these (see _capabilities_for).
_MODEL_CAPABILITY_RULES = (
    # Non-generative utility families: no chat, no tools, no vision.
    ("*", r"(^|[/_.-])(embed|embedding|rerank|reranker|guard|moderation|classif)",
     {"streaming": False, "vision_input": False, "tool_calling": False,
      "structured_output": False}),
    # Speech-to-text models serve the listening role only.
    ("*", r"(^|[/_.-])(whisper|stt|speech[-_.]?to[-_.]?text)",
     {"speech_input": True, "streaming": False, "vision_input": False,
      "tool_calling": False, "structured_output": False}),
    # Voice-synthesis models serve the tts role only.
    ("*", r"(^|[/_.-])(tts|text[-_.]?to[-_.]?speech|voice[-_.]?synth)",
     {"audio_output": True, "streaming": False, "vision_input": False,
      "tool_calling": False, "structured_output": False}),
    # Text-only LLama generations on Fireworks (vision variants keep vision).
    ("fireworks", r"llama-(v3p1|v3p3|3\.1|3\.3)", {"vision_input": False}),
    ("fireworks", r"llama[^/]*vision", {"vision_input": True}),
    ("*", r"gemma-2", {"vision_input": False}),
)

# Conservative per-model limits used when neither the provider's metadata nor
# a caller-supplied declaration establishes them.
_DEFAULT_LIMITS = {"max_input_tokens": 128000, "max_output_tokens": 8192}

# Models on Fireworks that run their own default reasoning: never send them a
# reasoning control (matches the browser agent's shipped carve-out).
_REASONING_OWN_DEFAULT_MODELS = ("minimax", "glm")

# Persistent session with automatic retries on low-level connection errors
# (same pattern as gemini_client, but for GET model-list calls).
_MAX_RETRIES = 2
_session = requests.Session()
_retry_strategy = Retry(
    total=_MAX_RETRIES,
    backoff_factor=0.5,          # 0s, 0.5s between retries
    status_forcelist=[429, 502, 503, 504],
    allowed_methods=["GET"],
)
_session.mount("https://", HTTPAdapter(max_retries=_retry_strategy))

_lock = threading.Lock()


class ModelRegistryError(ValueError):
    """User-facing registry error (bad provider / key / model input)."""


#: [P1-08] The parsed settings, keyed on (st_mtime_ns, st_size). The audio path
#: reaches this file several times PER CHUNK, and a file read plus a JSON parse
#: there is real work on the hot path. A stat is cheap; a parse is not. The key
#: keeps the project's promise that a change from the UI takes effect on the
#: very next phrase with NO restart — a long-lived cache without the key would
#: silently break live switching.
#:
#: ``data`` is the LAST GOOD copy: a transient read/parse failure falls back to
#: it rather than to {}, which used to silently discard the user's settings and
#: flip the whole system to env defaults. It is kept PER PATH — the path is part
#: of the key — so a "last good copy" can never be served for a different file.
_settings_cache = {"path": None, "key": None, "data": None, "loaded": False}


def _settings_key_unlocked():
    """(path, st_mtime_ns, st_size) for the settings file, or None if absent."""
    try:
        stat = SETTINGS_FILE.stat()
    except OSError:
        return None
    return (str(SETTINGS_FILE), stat.st_mtime_ns, stat.st_size)


def _load_unlocked():
    """The parsed settings; {} only when there has never been a good read.

    Caller must already hold _lock (read-only callers use _load_settings).
    Never returns the cached object itself: callers do read-modify-write on the
    dict they get, and a shared object would corrupt the cache.
    """
    key = _settings_key_unlocked()
    if key is None:
        # No settings file at all is a legitimate empty configuration (nothing
        # has been selected yet), not a failure.
        return {}
    if key == _settings_cache["key"] and _settings_cache["loaded"]:
        return copy.deepcopy(_settings_cache["data"])
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("settings root is not an object")
    except Exception as exc:
        # [P1-08] Keep the last good copy. Returning {} here silently reverted
        # every role to its env default without telling anyone.
        same_file = _settings_cache["path"] == str(SETTINGS_FILE)
        if _settings_cache["loaded"] and same_file:
            logging.warning(
                "[MODEL REGISTRY] Could not read settings (%s); keeping the "
                "last successfully loaded copy", exc)
            return copy.deepcopy(_settings_cache["data"])
        logging.warning(
            "[MODEL REGISTRY] Could not read settings (%s); falling back to "
            "environment defaults for every role", exc)
        return {}
    _settings_cache["path"] = str(SETTINGS_FILE)
    _settings_cache["key"] = key
    _settings_cache["data"] = copy.deepcopy(data)
    _settings_cache["loaded"] = True
    return data


def _forget_cached_settings_unlocked():
    """Drop the cache so the next read hits the file. For writers/tests."""
    _settings_cache["path"] = None
    _settings_cache["key"] = None
    _settings_cache["data"] = None
    _settings_cache["loaded"] = False


def _save_unlocked(settings):
    """Atomically persist the full settings dict (tmp + os.replace).

    Caller must already hold _lock across the ENTIRE read-modify-write:
    loading, mutating and saving as one critical section is what prevents
    two concurrent mutations on different keys from interleaving and the
    later full-file write silently clobbering the earlier change.

    F49: every mutation bumps the file's monotonic revision, so a caller can
    tell whether two configuration snapshots came from the same state.
    """
    try:
        settings["revision"] = int(settings.get("revision") or 0) + 1
    except (TypeError, ValueError):
        settings["revision"] = 1
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)
    tmp.replace(SETTINGS_FILE)
    # [P1-08] What we just wrote IS the parsed settings: adopt it instead of
    # making the next reader re-parse the file we already have in memory.
    _settings_cache["path"] = str(SETTINGS_FILE)
    _settings_cache["key"] = _settings_key_unlocked()
    _settings_cache["data"] = copy.deepcopy(settings)
    _settings_cache["loaded"] = True


def _load_settings():
    """Locked read for read-only callers; {} when missing or corrupt."""
    with _lock:
        return _load_unlocked()


def _custom_providers_from(settings):
    """Custom providers listed in an already-loaded settings dict.

    For use inside a locked read-modify-write — takes no lock itself.
    """
    providers = settings.get("custom_providers")
    if not isinstance(providers, list):
        return []
    return [p for p in providers if isinstance(p, dict) and p.get("id")]


def _custom_providers():
    return _custom_providers_from(_load_settings())


def _mask_provider(provider):
    """Public shape of a stored custom provider — no api_key ever.

    F49: the capability set that was explicitly declared for this provider is
    part of the public shape, so the UI (and the user) can see exactly which
    roles the provider is authorized for.
    """
    declared = provider.get("capabilities")
    if isinstance(declared, list):
        capabilities = sorted({str(c).strip() for c in declared if str(c).strip()})
        capabilities_source = str(
            provider.get("capabilities_source") or "custom_declared")
    else:
        capabilities = sorted(_CUSTOM_PROVIDER_DEFAULT_CAPABILITIES)
        capabilities_source = "custom_default"
    masked = {
        "id": provider.get("id"),
        "name": provider.get("name") or provider.get("id"),
        "kind": provider.get("kind") or "openai_compatible",
        "has_key": bool(provider.get("api_key")),
        "source": "custom",
        "capabilities": capabilities,
        "capabilities_source": capabilities_source,
    }
    base = provider.get("base_url")
    if base:
        # never echo a query/fragment: some gateways accept key-in-URL
        masked["base_url"] = str(base).split("?", 1)[0].split("#", 1)[0]
    return masked


def _env_default_for_role(role):
    """Env default (provider, model) for a role; reads live config values."""
    if role == "chat":
        return {"provider": "gemini", "model": GEMINI_CHAT_MODEL}
    if role == "tts":
        # Fish Audio is the TTS engine; model from FISH_MODEL env.
        try:
            m = getattr(_config, "FISH_MODEL", None) or "s2.1-pro-free"
        except Exception:
            m = "s2.1-pro-free"
        return {"provider": "fish", "model": str(m).strip() or "s2.1-pro-free"}
    if role == "vision":
        return {"provider": "gemini", "model": GEMINI_MODEL}
    if role == "browser_tool":
        try:
            prov = str(getattr(_config, "BROWSER_AGENT_PROVIDER", "") or "fireworks").strip() or "fireworks"
            mod = str(getattr(_config, "BROWSER_AGENT_MODEL", "") or "accounts/fireworks/models/qwen3p7-plus").strip() or "accounts/fireworks/models/qwen3p7-plus"
        except Exception:
            prov, mod = "fireworks", "accounts/fireworks/models/qwen3p7-plus"
        return {"provider": prov, "model": mod}
    if role == "listening":
        # Conversation STT default = Inworld (matches the shipped chain);
        # model id read live exactly like transcription.py does.
        try:
            m = str(os.getenv("INWORLD_STT_MODEL", "") or "").strip() or "inworld/inworld-stt-1"
        except Exception:
            m = "inworld/inworld-stt-1"
        return {"provider": "inworld", "model": m}
    if role == "planner":
        # F49 (G8): the native tool-use orchestrator's planner defaults to
        # the same Fireworks qwen3p7-plus the browser agent already uses
        # (the audit's named model) — resolved LIVE from config so env
        # overrides work without a restart.
        try:
            mod = str(getattr(_config, "BROWSER_AGENT_MODEL", "") or
                      "accounts/fireworks/models/qwen3p7-plus").strip() or \
                "accounts/fireworks/models/qwen3p7-plus"
        except Exception:
            mod = "accounts/fireworks/models/qwen3p7-plus"
        return {"provider": "fireworks", "model": mod}
    if role == "intent":
        # [F56] The registry default for the classifier must be the SHIPPED
        # router's first hop (see services/intent.py), otherwise an install
        # that never touched the new selector would start routing through a
        # different hop. Imported lazily: intent.py reads this module.
        try:
            from backend.services.intent import DEFAULT_OPENROUTER_MODEL as _m
            mod = str(_m or "").strip()
        except Exception:
            mod = ""
        return {"provider": "openrouter",
                "model": mod or "google/gemini-2.5-flash-lite"}
    return {"provider": "gemini", "model": GEMINI_CHAT_MODEL}


def _model_key(provider, model):
    """Settings key for one (provider, model). Provider ids never contain a
    slash (see add_custom_provider), so the first slash is unambiguous."""
    return "%s/%s" % (str(provider or "").strip(), str(model or "").strip())


def _provider_declaration(provider, settings):
    """Capability declaration for a REGISTERED provider; None when unknown.

    Returns {"capabilities": frozenset, "source": str, "private": bool}.

    F49: the old code handed every unknown provider id the OpenAI-compatible
    floor, so a typo, a removed provider or an arbitrary string silently
    inherited tool + vision + schema support. Now a provider that is not
    registered (env or stored custom) has NO capabilities at all, and a
    custom provider's set is the one explicitly declared with its record.
    """
    pid = str(provider or "").strip()
    if pid in ENV_PROVIDERS:
        declared = PROVIDER_CAPABILITIES.get(pid)
        if declared is None:
            return None
        return {"capabilities": frozenset(declared),
                "source": "env_adapter", "private": False}
    for p in _custom_providers_from(settings):
        if str(p.get("id") or "").strip() != pid:
            continue
        stored = p.get("capabilities")
        if isinstance(stored, list):
            capabilities = frozenset(
                str(c).strip() for c in stored if str(c).strip())
            source = str(p.get("capabilities_source") or "custom_declared")
        else:
            # Legacy record written before F49: it was authorized at add time
            # with the shipped OpenAI-compatible floor. Keep it usable, but
            # label the source so it is distinguishable from a declaration.
            capabilities = _CUSTOM_PROVIDER_DEFAULT_CAPABILITIES
            source = "custom_default"
        return {"capabilities": capabilities, "source": source, "private": True}
    return None


def _capabilities_for(provider, model, settings, declaration=None):
    """Effective capability set + provenance for one (provider, model).

    Order (later wins): provider declaration → model-name rules → capability
    metadata the provider published → explicit per-model declaration. An
    unregistered provider resolves to the EMPTY set (fail closed).
    """
    if declaration is None:
        declaration = _provider_declaration(provider, settings)
    if declaration is None:
        return frozenset(), ["unknown_provider"]
    capabilities = set(declaration["capabilities"])
    sources = [declaration["source"]]
    pid = str(provider or "").strip()
    model_id = str(model or "")
    for provider_pattern, model_pattern, deltas in _MODEL_CAPABILITY_RULES:
        if provider_pattern != "*" and provider_pattern != pid:
            continue
        if not re.search(model_pattern, model_id, re.IGNORECASE):
            continue
        for capability, enabled in deltas.items():
            if enabled:
                capabilities.add(capability)
            else:
                capabilities.discard(capability)
        sources.append("model_rule:%s" % model_pattern)
    observed = (settings.get("observed_capabilities") or {}).get(
        _model_key(provider, model))
    if isinstance(observed, dict):
        for capability, enabled in (observed.get("capabilities") or {}).items():
            if capability not in KNOWN_CAPABILITIES:
                continue
            if enabled:
                capabilities.add(capability)
            else:
                capabilities.discard(capability)
        sources.append("observed")
    declared = (settings.get("model_capabilities") or {}).get(
        _model_key(provider, model))
    if isinstance(declared, dict) and isinstance(declared.get("capabilities"), list):
        capabilities = {str(c).strip() for c in declared["capabilities"]
                        if str(c).strip()}
        sources.append("declared")
    return frozenset(capabilities), sources


def model_capabilities_for(provider, model=None):
    """Effective capability set for a (provider[, model]).

    F49: an unregistered provider has NO capabilities (fail closed), instead
    of inheriting the OpenAI-compatible floor.
    """
    with _lock:
        settings = _load_unlocked()
        capabilities, _sources = _capabilities_for(provider, model, settings)
        return capabilities


def validate_role_capabilities(role, provider, model=None):
    """F49: validate a (role, provider[, model]) pairing BEFORE use.

    Raises ModelRegistryError when the provider is unknown, is not allowed
    for the role, or cannot be shown to support a capability the role
    requires. Returns the validation detail on success.
    """
    r = str(role or "").strip()
    required = ROLE_CAPABILITIES.get(r)
    if required is None:
        raise ModelRegistryError("unknown role '%s'" % (role,))
    provider_id = str(provider or "").strip()
    model_id = str(model or "").strip()
    with _lock:
        settings = _load_unlocked()
        declaration = _provider_declaration(provider_id, settings)
        if declaration is None:
            raise ModelRegistryError(
                "unknown provider '%s' — its capabilities cannot be "
                "established, so the selection is refused" % (provider_id,))
        custom_ids = [str(p.get("id") or "").strip()
                      for p in _custom_providers_from(settings)]
        if not _is_provider_allowed_for_role(provider_id, r, custom_ids):
            raise ModelRegistryError(
                "provider '%s' is not allowed for role '%s'" % (provider_id, r))
        capabilities, sources = _capabilities_for(
            provider_id, model_id, settings, declaration)
    missing = sorted(required - capabilities)
    if missing:
        raise ModelRegistryError(
            "model '%s' on provider '%s' lacks capabilities %s required for "
            "role '%s'" % (model_id, provider_id, ", ".join(missing), r))
    return {
        "role": r,
        "provider": provider_id,
        "model": model_id,
        "required": sorted(required),
        "capabilities": sorted(capabilities),
        "capability_sources": sources,
    }


def _limits_for(provider, model, settings):
    """Per-model request limits: defaults, overridden by provider metadata."""
    limits = dict(_DEFAULT_LIMITS)
    observed = (settings.get("observed_capabilities") or {}).get(
        _model_key(provider, model))
    if isinstance(observed, dict):
        for key in ("max_input_tokens", "max_output_tokens"):
            value = (observed.get("limits") or {}).get(key)
            if isinstance(value, (int, float)) and value > 0:
                limits[key] = int(value)
    return limits


def _reasoning_for(provider, model, capabilities):
    """Whether a reasoning control may be sent to this model, and which one.

    Two adapters understand ``reasoning_effort``: Fireworks, and the generic
    OpenAI-compatible path when it points at a LOCAL Ollama server. Some
    Fireworks models (MiniMax / GLM) run their own default reasoning and
    reject the field — the same carve-out the browser agent ships.
    """
    pid = str(provider or "").strip().lower()
    model_id = str(model or "").lower()
    if pid == "ollama":
        # [F56] Measured on Ollama 0.34.4: the OpenAI-compatible surface
        # IGNORES the native ``think`` field (and a ``/no_think`` marker) but
        # honours reasoning_effort — "none" disables thinking (qwen3 1.7B:
        # 2.2s -> 0.4s, and the first ANSWER token goes 1.41s -> 0.06s),
        # while any enabling value turns it on. A model that cannot think
        # (llama3.2) ACCEPTS "none" and 400s on every enabling value, so
        # "none" is the only value that is safe for every installed model.
        return {"supported": True, "param": "reasoning_effort", "effort": "none",
                "reason": "local Ollama endpoint: reasoning_effort=none "
                          "disables thinking"}
    if pid != "fireworks":
        return {"supported": False, "param": None, "effort": None,
                "reason": "adapter does not carry reasoning controls"}
    if any(token in model_id for token in _REASONING_OWN_DEFAULT_MODELS):
        return {"supported": False, "param": None, "effort": None,
                "reason": "model runs its own default reasoning"}
    if not capabilities & {"streaming", "tool_calling"}:
        return {"supported": False, "param": None, "effort": None,
                "reason": "model is not a generative chat model"}
    try:
        effort = str(getattr(_config, "BROWSER_AGENT_REASONING_EFFORT", "")
                     or "").strip() or None
    except Exception:
        effort = None
    return {"supported": True, "param": "reasoning_effort", "effort": effort,
            "reason": "fireworks reasoning_effort supported"}


def _adapter_for(provider):
    """The code path that will carry the call for *provider*."""
    if str(provider or "").strip() == "gemini":
        return "gemini_native"
    return "openai_compatible"


# Env providers whose shipped adapter is OpenAI-compatible, and the config
# attribute naming its chat-completions URL.
_ENV_CHAT_URL_ATTRS = {
    "fireworks": "FIREWORKS_API_URL",
    "groq": "GROQ_API_URL",
    "openrouter": "OPENROUTER_API_URL",
    "cline": "CLINE_API_URL",
}


def _base_url_for(provider, settings):
    """Chat base URL for *provider* (no /chat/completions suffix); None when
    the provider has no OpenAI-compatible HTTP endpoint (native gemini, local
    whisper, fish/inworld audio APIs)."""
    pid = str(provider or "").strip()
    if pid == "ollama":
        # [F56] Local server, fixed endpoint (OLLAMA_OPENAI_BASE_URL derives
        # from ollama_client's single source of truth).
        return OLLAMA_OPENAI_BASE_URL
    if pid in _ENV_CHAT_URL_ATTRS:
        raw = str(getattr(_config, _ENV_CHAT_URL_ATTRS[pid], "") or "").strip()
        if raw.endswith("/chat/completions"):
            raw = raw[: -len("/chat/completions")]
        return raw.rstrip("/") or None
    for p in _custom_providers_from(settings):
        if str(p.get("id") or "").strip() == pid:
            return str(p.get("base_url") or "").strip().rstrip("/") or None
    return None


def _credentials_from(provider, settings):
    """(api_key, base_url) for *provider* from ONE settings snapshot.

    Same shape as get_provider_credentials, but never re-reads the settings
    file: a resolution stays a single atomic read (F49).
    """
    pid = str(provider or "").strip()
    if pid == "gemini":
        return (GEMINI_API_KEY or None, None)
    if pid == "fireworks":
        return (FIREWORKS_API_KEY or None, None)
    if pid == "groq":
        return (GROQ_API_KEY or None, None)
    if pid == "fish":
        return (FISH_API_KEY or None, None)
    if pid == "gtts":
        # Google Translate TTS is key-less and free: there is no credential
        # to find, and its absence must NOT be read as "misconfigured" (the
        # same reason the local whisper daemon reports (None, None)).
        return (None, None)
    if pid == "openrouter":
        return (OPENROUTER_API_KEY or None, None)
    if pid == "whisper":
        # Local daemon — no API key exists.
        return (None, None)
    if pid == "inworld":
        return (os.getenv("INWORLD_STT_API_KEY") or None, None)
    if pid == "ollama":
        # [F56] The local Ollama server has NO credential — but the generic
        # OpenAI-compatible adapter (and brain's "a provider with no usable
        # credentials fails closed" rule) needs a truthy key to send the
        # request at all. Ollama ignores the Authorization value, so a
        # placeholder keeps every other invariant intact instead of punching
        # a keyless special case through the chat path.
        return ("ollama-local", None)
    for p in _custom_providers_from(settings):
        if str(p.get("id") or "").strip() == pid:
            return (p.get("api_key") or None, _base_url_for(pid, settings))
    return (None, None)


def _snapshot_from_settings(role, settings, override=None,
                            selection_source=None, with_credentials=False):
    """Build the validated configuration snapshot for one role.

    Pure function of ONE already-loaded settings dict (the caller holds
    _lock): provider resolution, allowlist, capabilities, credentials,
    endpoint, limits and reasoning all come from that single read, so an
    in-flight call can never observe a half-applied configuration change.
    Raises ModelRegistryError for anything it cannot validate.
    """
    r = str(role or "").strip()
    required = ROLE_CAPABILITIES.get(r)
    if required is None:
        raise ModelRegistryError("unknown role '%s'" % (r,))
    if override is None:
        stored = settings.get(_ROLE_STORAGE_KEY[r])
        if (isinstance(stored, dict)
                and str(stored.get("provider") or "").strip()
                and str(stored.get("model") or "").strip()):
            override = (str(stored["provider"]).strip(),
                        str(stored["model"]).strip())
            selection_source = selection_source or "persisted"
    if override is None:
        default = _env_default_for_role(r)
        override = (str(default.get("provider") or "").strip(),
                    str(default.get("model") or "").strip())
        selection_source = selection_source or "env_default"
    provider = str(override[0] or "").strip()
    model = str(override[1] or "").strip()
    if not provider or not model:
        raise ModelRegistryError(
            "role '%s' has an incomplete model selection "
            "(provider=%r model=%r)" % (r, provider, model))
    declaration = _provider_declaration(provider, settings)
    if declaration is None:
        raise ModelRegistryError(
            "unknown provider '%s' selected for role '%s' — refusing to run "
            "it" % (provider, r))
    custom_ids = [str(p.get("id") or "").strip()
                  for p in _custom_providers_from(settings)]
    if not _is_provider_allowed_for_role(provider, r, custom_ids):
        allowed = sorted(_ROLE_ALLOWED_ENV.get(r, set()))
        if r in _ROLE_ALLOWS_CUSTOM:
            allowed = sorted(set(allowed) | {"custom providers"})
        raise ModelRegistryError(
            "provider '%s' is not allowed for role '%s' (allowed: %s)"
            % (provider, r, ", ".join(allowed)))
    capabilities, capability_sources = _capabilities_for(
        provider, model, settings, declaration)
    missing = sorted(required - capabilities)
    if missing:
        raise ModelRegistryError(
            "model '%s' on provider '%s' lacks capabilities %s required for "
            "role '%s' — selection refused"
            % (model, provider, ", ".join(missing), r))
    api_key, base_url = _credentials_from(provider, settings)
    if declaration["private"] and (not api_key or not base_url):
        raise ModelRegistryError(
            "private provider '%s' has no usable stored credentials — "
            "refusing to run role '%s' and refusing to send its work to "
            "another provider" % (provider, r))
    if base_url is None:
        # Env providers keep their adapter's configured endpoint (the shipped
        # clients read it from config; get_provider_credentials deliberately
        # reports no base_url for them — see its docstring).
        base_url = _base_url_for(provider, settings)
    endpoint = {
        "base_url": base_url,
        "has_credentials": bool(api_key),
        "credential_source": ("provider_settings" if declaration["private"]
                              else "environment"),
    }
    if with_credentials:
        # Internal callers only: resolve_call_config never reaches the UI.
        endpoint["api_key"] = api_key
    return {
        "role": r,
        "provider": provider,
        "model": model,
        "selection_source": selection_source or "unknown",
        "adapter": _adapter_for(provider),
        "endpoint": endpoint,
        "required": sorted(required),
        "capabilities": sorted(capabilities),
        "capability_sources": capability_sources,
        "tools": "tool_calling" in capabilities,
        "vision": "vision_input" in capabilities,
        "schema": "structured_output" in capabilities,
        "streaming": "streaming" in capabilities,
        "limits": _limits_for(provider, model, settings),
        "reasoning": _reasoning_for(provider, model, capabilities),
        "settings_revision": int(settings.get("revision") or 0),
        "resolved_at": time.time(),
    }


def _fallback_authorized(role, from_provider, settings):
    """The ONLY fallback F49 authorizes for a rejected selection.

    It is the role's own declared env default (from config/env, i.e. an
    explicitly configured default — never an arbitrary substitute), and it is
    only authorized away from an ENV provider. A private (custom) provider's
    work is never silently rerouted: that fails closed instead. Returns
    {"provider","model"} or None.
    """
    if str(from_provider or "").strip() not in ENV_PROVIDERS:
        return None
    declared = _env_default_for_role(str(role or "").strip())
    provider = str(declared.get("provider") or "").strip()
    model = str(declared.get("model") or "").strip()
    if not provider or not model:
        return None
    return {"provider": provider, "model": model}


def authorized_fallback_for(role, provider):
    """F49: the authorized fallback for *role* when *provider* is unusable.

    None when no fallback is authorized (a private or unknown provider).
    """
    r = str(role or "").strip()
    if r not in VALID_ROLES:
        raise ModelRegistryError("unknown role '%s'" % (r,))
    with _lock:
        return _fallback_authorized(r, provider, _load_unlocked())


def _resolve_role(role, strict=True, allow_env_fallback=False,
                  with_credentials=False):
    """Resolve one role from a SINGLE locked settings read.

    strict=True (get_model_config / resolve_call_config): any validation
    failure raises — an invalid persisted selection can never run.
    strict=False (get_model_for_role): an invalid persisted selection on an
    ENV provider is replaced by the authorized, re-validated role default;
    anything touching a private/unknown provider still fails closed.
    """
    r = str(role or "").strip()
    if r not in VALID_ROLES:
        raise ModelRegistryError("unknown role '%s'" % (r,))
    with _lock:
        settings = _load_unlocked()
        try:
            return _snapshot_from_settings(
                r, settings, with_credentials=with_credentials)
        except ModelRegistryError as exc:
            if strict or not allow_env_fallback:
                raise
            stored = settings.get(_ROLE_STORAGE_KEY[r])
            rejected = ""
            if isinstance(stored, dict):
                rejected = str(stored.get("provider") or "").strip()
            fallback = _fallback_authorized(r, rejected, settings)
            if fallback is None:
                raise
            logging.warning(
                "[MODEL REGISTRY] persisted %s selection rejected (%s); "
                "using the authorized default %s/%s",
                r, exc, fallback["provider"], fallback["model"])
            return _snapshot_from_settings(
                r, settings,
                override=(fallback["provider"], fallback["model"]),
                selection_source="authorized_fallback",
                with_credentials=with_credentials)


def get_model_config(role):
    """F49: the validated per-call configuration snapshot for *role*.

    The snapshot carries the model, adapter, endpoint shape, capability set
    (+provenance), limits, reasoning support and the settings revision, all
    from ONE locked read (an in-flight call cannot see a half-applied
    change). Raises ModelRegistryError when the resolved configuration is
    invalid — invalid persisted settings cannot run.
    """
    return _resolve_role(role, strict=True)


def resolve_call_config(role):
    """F49: like get_model_config, but WITH the resolved credentials.

    Internal callers only (orchestrator): the snapshot is what makes one
    call's provider/model/endpoint consistent for its whole lifetime, and it
    must never be serialized to the UI.
    """
    return _resolve_role(role, strict=True, with_credentials=True)


def get_model_for_role(role):
    """(provider, model) for a role: settings override first, else env default.

    F49: resolution is VALIDATED at use time. A persisted selection that no
    longer validates never runs — for an env provider the role's declared
    default (the only authorized fallback, itself re-validated) is used; for
    a private or unknown provider the call fails closed so private work is
    never silently sent to a different provider. Raises ModelRegistryError on
    unknown role. Degrades to the env default on a missing/corrupt file.
    """
    snapshot = _resolve_role(role, strict=False, allow_env_fallback=True)
    return {"provider": snapshot["provider"], "model": snapshot["model"]}


def validate_request_limits(snapshot, prompt_tokens=None, max_tokens=None):
    """F49: validate a request against the limits of the model it will run on.

    *snapshot* is a config snapshot (or a role name, for convenience). Raises
    ModelRegistryError when the request cannot fit; True otherwise.
    """
    if isinstance(snapshot, str):
        snapshot = get_model_config(snapshot)
    snapshot = snapshot or {}
    limits = snapshot.get("limits") or {}
    max_input = limits.get("max_input_tokens")
    max_output = limits.get("max_output_tokens")
    model = snapshot.get("model")
    if max_tokens is not None and max_output and int(max_tokens) > int(max_output):
        raise ModelRegistryError(
            "requested max_tokens=%s exceeds the output limit %s of model '%s'"
            % (max_tokens, max_output, model))
    if prompt_tokens is not None and max_input and int(prompt_tokens) > int(max_input):
        raise ModelRegistryError(
            "prompt tokens %s exceed the input limit %s of model '%s'"
            % (prompt_tokens, max_input, model))
    if (prompt_tokens is not None and max_tokens is not None and max_input
            and int(prompt_tokens) + int(max_tokens) > int(max_input)):
        raise ModelRegistryError(
            "request (%s + %s tokens) exceeds the context limit %s of model '%s'"
            % (prompt_tokens, max_tokens, max_input, model))
    return True


def declare_model_capabilities(provider, model, capabilities, source="user"):
    """F49: EXPLICITLY declare/authorize one model's capability set.

    This is the only way an otherwise unknown model can be authorized for a
    role, and the declaration is persisted in the settings file so the
    authorization is reviewable and survives restarts. The declared set
    replaces every inferred source for that model.
    """
    provider_id = str(provider or "").strip()
    model_id = str(model or "").strip()
    if not provider_id or not model_id:
        raise ModelRegistryError("provider and model are required")
    if isinstance(capabilities, str):
        capabilities = [capabilities]
    try:
        declared = sorted({str(c).strip() for c in (capabilities or [])
                           if str(c).strip()})
    except TypeError:
        raise ModelRegistryError("capabilities must be a list of names")
    unknown = [c for c in declared if c not in KNOWN_CAPABILITIES]
    if unknown:
        raise ModelRegistryError(
            "unknown capability name(s): %s" % ", ".join(unknown))
    with _lock:
        settings = _load_unlocked()
        if _provider_declaration(provider_id, settings) is None:
            raise ModelRegistryError("unknown provider '%s'" % (provider_id,))
        store = settings.get("model_capabilities")
        if not isinstance(store, dict):
            store = {}
        entry = {
            "provider": provider_id,
            "model": model_id,
            "capabilities": declared,
            "source": str(source or "user"),
            "declared_at": time.time(),
        }
        store[_model_key(provider_id, model_id)] = entry
        settings["model_capabilities"] = store
        _save_unlocked(settings)
    return dict(entry)


def _is_provider_allowed_for_role(provider, role, custom_ids=None):
    """True if provider may be used for role per allowlist."""
    if provider in ENV_PROVIDERS:
        return provider in _ROLE_ALLOWED_ENV.get(role, set())
    # custom provider
    if custom_ids is None:
        custom_ids = [p.get("id") for p in _custom_providers()]
    if provider in custom_ids:
        return role in _ROLE_ALLOWS_CUSTOM
    return False


def get_allowed_providers_for_role(role):
    """Sorted list of provider ids allowed for role (env + applicable customs)."""
    r = str(role or "").strip()
    if r not in VALID_ROLES:
        raise ModelRegistryError(f"unknown role '{r}'")
    allowed = set(_ROLE_ALLOWED_ENV.get(r, set()))
    if r in _ROLE_ALLOWS_CUSTOM:
        for p in _custom_providers():
            allowed.add(p.get("id"))
    return sorted(allowed)


def get_role_allowed_map():
    """{role: [provider ids]} for all roles — for GET /settings."""
    return {r: get_allowed_providers_for_role(r) for r in VALID_ROLES}


def set_model_for_role(role, provider, model):
    """Persist the model for a role; immediate effect, no restart.

    Raises ModelRegistryError on bad role/provider/model."""
    r = str(role or "").strip()
    if r not in VALID_ROLES:
        raise ModelRegistryError(f"unknown role '{r}'")
    provider = str(provider or "").strip()
    model = str(model or "").strip()
    if not provider:
        raise ModelRegistryError("provider is required")
    if not model:
        raise ModelRegistryError("model is required")
    key = _ROLE_STORAGE_KEY[r]
    with _lock:
        settings = _load_unlocked()
        custom_ids = [p.get("id") for p in _custom_providers_from(settings)]
        if provider not in ENV_PROVIDERS and provider not in custom_ids:
            raise ModelRegistryError(f"unknown provider '{provider}'")
        if not _is_provider_allowed_for_role(provider, r, custom_ids):
            allowed = sorted(_ROLE_ALLOWED_ENV.get(r, set()))
            if r in _ROLE_ALLOWS_CUSTOM:
                allowed = sorted(set(allowed) | {"custom providers"})
            raise ModelRegistryError(
                f"provider '{provider}' is not allowed for role '{r}' (allowed: {', '.join(sorted(allowed))})"
            )
        # F49: the CANDIDATE selection is validated (allowlist, capabilities,
        # credentials) from the same locked read that persists it — a
        # selection that would silently lose images, tools, schema support or
        # a private provider's endpoint is rejected at the boundary.
        _snapshot_from_settings(r, settings, override=(provider, model))
        settings[key] = {"provider": provider, "model": model}
        _save_unlocked(settings)
    return {"provider": provider, "model": model}


def get_default_chat_model():
    """(provider, model) for chat replies: settings override first, else
    the env default (gemini, GEMINI_BRAIN_MODEL)."""
    return get_model_for_role("chat")


def set_default_chat_model(provider, model):
    """Persist the default chat model; effect is immediate (per-message
    resolution) with no restart. Raises ModelRegistryError on bad input."""
    return set_model_for_role("chat", provider, model)


#: Canonical OpenAI-compatible chat endpoints for env providers that have
#: one AND whose calls ride the generic openai-compat dispatch (openrouter,
#: groq, ollama). The dedicated clients know their URLs privately; exposing
#: them here lets the brain chat / browser tool carry an env-provider
#: selection too — without moving the key out of .env into a custom provider
#: record. Fireworks keeps base_url=None deliberately: it has its own full
#: client and its tests pin that contract. [added 2026-09-23: chat via
#: openrouter after the direct Gemini API became unusably slow behind the
#: user's VPN relay; ollama added 2026-10-01 for local models]
_ENV_PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": OLLAMA_OPENAI_BASE_URL,
}


def get_provider_credentials(provider_id):
    """(api_key, base_url) for a provider; env providers return their
    canonical OpenAI-compatible base URL when they have one (see
    _ENV_PROVIDER_BASE_URLS), else None.

    Returns (None, None) for unknown providers. F49: resolved from a single
    locked settings read (see _credentials_from) so callers that ALSO use a
    config snapshot cannot observe two different configuration states.
    """
    with _lock:
        api_key, base_url = _credentials_from(provider_id, _load_unlocked())
    if base_url is None and provider_id in _ENV_PROVIDER_BASE_URLS:
        base_url = _ENV_PROVIDER_BASE_URLS[provider_id]
    return api_key, base_url


def list_providers():
    """All selectable providers, MASKED — api keys never leave this module.

    F49: every provider reports the capability set that authorizes it, so the
    UI can show exactly which roles a selection is valid for.
    """
    providers = []
    for pid, meta in ENV_PROVIDERS.items():
        key, _base = get_provider_credentials(pid)
        providers.append({
            "id": pid,
            "name": meta["name"],
            "kind": "env",
            "has_key": bool(key),
            "source": "env",
            "capabilities": sorted(PROVIDER_CAPABILITIES.get(pid, frozenset())),
            "capabilities_source": "env_adapter",
        })
    for p in _custom_providers():
        providers.append(_mask_provider(p))
    return providers


def list_custom_providers():
    """Stored custom providers, masked."""
    return [_mask_provider(p) for p in _custom_providers()]


def custom_provider_ids():
    """Ids of the stored custom providers (for adapter dispatch maps)."""
    return sorted(str(p.get("id") or "").strip()
                  for p in _custom_providers() if p.get("id"))


def roles_allowing_custom():
    """Roles a user-added OpenAI-compatible provider may serve.

    Part of GET /settings so the UI can show the per-functionality
    "add custom provider" entry exactly where it will be accepted.
    """
    return sorted(_ROLE_ALLOWS_CUSTOM)


def test_custom_provider(base_url, api_key):
    """Live-check a candidate custom provider WITHOUT storing anything.

    The "Test" button of the add-provider form lands here: the exact
    (base_url, api_key) pair gets one model-list round trip, so the user
    learns whether the provider works before anything is persisted. A junk
    key or unreachable endpoint raises a scrubbed ModelRegistryError and
    nothing is written — same validation add_custom_provider enforces,
    minus the write.
    """
    key = str(api_key or "").strip()
    base = str(base_url or "").strip().rstrip("/")
    if not key:
        raise ModelRegistryError("api_key is required")
    if not (base.startswith("http://") or base.startswith("https://")):
        raise ModelRegistryError(
            "base_url must start with http:// or https://")
    try:
        models = _list_openai_compat_models(base, key)
    except ModelRegistryError:
        raise
    except Exception:
        raise ModelRegistryError(
            "could not reach the provider to validate the API key"
        )
    if not models:
        raise ModelRegistryError(
            "API key validated but the provider returned no models"
        )
    return models


# Key-material echo patterns for scrubbing provider error text: key=value
# pairs, Bearer headers, and long token-like runs. Over-redaction is the
# goal — losing a bit of diagnostic detail is always cheaper than leaking
# a key into a UI-facing error.
_KEY_VALUE_RE = re.compile(
    r"\b(api[_-]?key|key|token|secret|password)\s*[=:]\s*[^\s,;&'\">]+",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\bbearer\s+[a-z0-9._~+/-]{8,}", re.IGNORECASE)
_TOKEN_RUN_RE = re.compile(r"\b[a-z0-9][a-z0-9._~+/-]{23,}", re.IGNORECASE)


def _scrub_secrets(text):
    """Redact potential key material from provider error text before it can
    reach a UI-facing error detail."""
    text = _KEY_VALUE_RE.sub(r"\1=REDACTED", text)
    text = _BEARER_RE.sub("Bearer REDACTED", text)
    text = _TOKEN_RUN_RE.sub("REDACTED", text)
    return text


def _http_get_json(url, params=None, headers=None, timeout=(5, 15)):
    try:
        resp = _session.get(url, params=params, headers=headers, timeout=timeout)
    except Exception:
        # Never surface the exception text: for Gemini the API key rides in
        # the request URL, and requests embeds the URL in connection errors.
        raise ModelRegistryError("model list request failed (network error)")
    if resp.status_code != 200:
        detail = ""
        try:
            # Scrub the FULL body first, then truncate: a raw token cut
            # mid-run by the slice could otherwise leak a key fragment.
            detail = " " + _scrub_secrets(resp.text or "")[:160]
        except Exception:
            pass
        raise ModelRegistryError(
            f"model list request failed (HTTP {resp.status_code}){detail}"
        )
    try:
        return resp.json()
    except Exception:
        raise ModelRegistryError("model list response was not valid JSON")


def _observation_from_item(item):
    """Capability/limit metadata a provider published for one model.

    Returns {"model", "capabilities", "limits"} or None when the provider
    said nothing measurable about it. OpenRouter's documented model objects
    carry ``architecture.input_modalities``/``modality``,
    ``supported_parameters``, ``context_length`` and
    ``top_provider.max_completion_tokens``; other OpenAI-compatible gateways
    may carry the same fields. Only fields actually present are recorded —
    an absent field is never turned into a capability claim.
    """
    mid = str(item.get("id") or "").strip()
    if not mid:
        return None
    capabilities = {}
    limits = {}
    params = item.get("supported_parameters")
    if isinstance(params, list):
        lowered = {str(p).strip().lower() for p in params}
        capabilities["tool_calling"] = "tools" in lowered
        capabilities["structured_output"] = bool(lowered & {
            "response_format", "structured_outputs", "structured_output",
            "json_schema",
        })
    arch = item.get("architecture")
    if isinstance(arch, dict):
        modalities = arch.get("input_modalities")
        if isinstance(modalities, list) and modalities:
            capabilities["vision_input"] = any(
                "image" in str(m).lower() for m in modalities)
        else:
            modality = arch.get("modality")
            if isinstance(modality, str) and modality.strip():
                capabilities["vision_input"] = "image" in modality.lower()
    for source, target in (("context_length", "max_input_tokens"),
                           ("max_input_tokens", "max_input_tokens"),
                           ("max_completion_tokens", "max_output_tokens")):
        value = item.get(source)
        if isinstance(value, (int, float)) and value > 0:
            limits[target] = int(value)
    top = item.get("top_provider")
    if isinstance(top, dict):
        value = top.get("max_completion_tokens")
        if isinstance(value, (int, float)) and value > 0:
            limits["max_output_tokens"] = int(value)
    if not capabilities and not limits:
        return None
    return {"model": mid, "capabilities": capabilities, "limits": limits}


def _record_observed_capabilities(provider, observations):
    """Persist the capabilities a provider published for its models.

    F49: this is what makes capability checks about the actual model instead
    of a provider-wide assumption — a gateway that reports a text-only model
    (no image input, no tool support) cannot be selected for a vision/tool
    role even if its provider advertises them. Best effort: a listing must
    never fail because the cache could not be written.
    """
    clean = {}
    for entry in observations or []:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("model") or "").strip()
        capabilities = {k: bool(v) for k, v in
                        (entry.get("capabilities") or {}).items()
                        if k in KNOWN_CAPABILITIES}
        limits = {k: int(v) for k, v in (entry.get("limits") or {}).items()
                  if isinstance(v, (int, float)) and v > 0}
        if not model_id or not (capabilities or limits):
            continue
        clean[_model_key(provider, model_id)] = {
            "capabilities": capabilities,
            "limits": limits,
            "observed_at": time.time(),
        }
    if not clean:
        return 0
    try:
        with _lock:
            settings = _load_unlocked()
            store = settings.get("observed_capabilities")
            if not isinstance(store, dict):
                store = {}
            for key, entry in clean.items():
                merged = store.get(key) if isinstance(store.get(key), dict) else {}
                capabilities = dict(merged.get("capabilities") or {})
                capabilities.update(entry["capabilities"])
                limits = dict(merged.get("limits") or {})
                limits.update(entry["limits"])
                store[key] = {
                    "capabilities": capabilities,
                    "limits": limits,
                    "observed_at": entry["observed_at"],
                }
            settings["observed_capabilities"] = store
            _save_unlocked(settings)
        return len(clean)
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning(
            "[MODEL REGISTRY] could not record observed capabilities: %s", exc)
        return 0


def _list_gemini_models(api_key):
    """Chat-capable Gemini models (generateContent only), paginated."""
    models = []
    observations = []
    page_token = None
    for _ in range(5):  # hard cap on pagination
        params = {"key": api_key, "pageSize": 100}
        if page_token:
            params["pageToken"] = page_token
        data = _http_get_json(_GEMINI_MODELS_URL, params=params)
        for m in data.get("models", []) or []:
            methods = m.get("supportedGenerationMethods") or []
            name = str(m.get("name") or "")
            # chat URLs are models/{id}:streamGenerateContent, so the bare
            # id (name minus the models/ prefix) is what gets passed around
            mid = name[len("models/"):] if name.startswith("models/") else name
            if not mid:
                continue
            # F49: generateContent/supportedGenerationMethods and the token
            # limits are real per-model capability metadata — record them.
            observation = {
                "model": mid,
                "capabilities": {
                    "streaming": "generateContent" in methods,
                },
                "limits": {},
            }
            for source, target in (("inputTokenLimit", "max_input_tokens"),
                                   ("outputTokenLimit", "max_output_tokens")):
                value = m.get(source)
                if isinstance(value, (int, float)) and value > 0:
                    observation["limits"][target] = int(value)
            observations.append(observation)
            if "generateContent" not in methods:
                continue  # embeddings / tts / imagen cannot answer chat
            models.append({"id": mid, "display": m.get("displayName") or mid})
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    _record_observed_capabilities("gemini", observations)
    return models


def _list_fireworks_models(api_key):
    """Fireworks models — ids are FULL accounts/fireworks/models/... paths."""
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _http_get_json(_FIREWORKS_MODELS_URL, headers=headers)
    models = []
    observations = []
    for item in data.get("data", []) or []:
        mid = str(item.get("id") or "").strip()
        if mid:
            display = mid.rsplit("/", 1)[-1] if "/" in mid else mid
            models.append({"id": mid, "display": display})
        observation = _observation_from_item(item)
        if observation:
            observations.append(observation)
    _record_observed_capabilities("fireworks", observations)
    return models


def _list_openai_compat_models(base_url, api_key):
    """Models from an OpenAI-compatible gateway: GET {base_url}/models."""
    url = str(base_url).rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = _http_get_json(url, headers=headers)
    models = []
    for item in data.get("data", []) or []:
        mid = str(item.get("id") or "").strip()
        if mid:
            models.append({"id": mid, "display": mid})
    return models


def _list_fish_models(api_key):
    """Fish Audio TTS models — static list (no public list API)."""
    if not api_key:
        raise ModelRegistryError("no Fish Audio API key configured")
    return [{"id": mid, "display": mid} for mid in _FISH_TTS_STATIC_MODELS]


#: Fallback voice list, used only if the engine module cannot be imported.
#: The live list lives in ``google_tts.LANGUAGES`` so there is exactly one
#: source of truth for what the engine can actually speak.
_GOOGLE_TTS_FALLBACK_VOICES = ("en", "hi", "es", "fr", "de", "it", "pt", "ja")


def _list_google_tts_models(api_key=None):
    """Google Translate TTS voices — static list, no key and no list API.

    The selectable "models" are real languages (the endpoint serves one
    voice per language, so regional accents would be a fake distinction).
    Deliberately does NOT require a key: this engine is the free fallback,
    and a missing credential is its normal state, not an error.
    """
    try:
        from backend.services.google_tts import LANGUAGES as _langs
        langs = tuple(_langs) or _GOOGLE_TTS_FALLBACK_VOICES
    except Exception:
        langs = _GOOGLE_TTS_FALLBACK_VOICES
    return [{"id": lang, "display": lang} for lang in langs]


def _list_openrouter_models(api_key):
    """OpenRouter models — GET https://openrouter.ai/api/v1/models.

    The endpoint is public; a key is sent when present but not required.
    Responses are {data: [{id, architecture: {input_modalities, modality},
    supported_parameters, context_length, top_provider}]}.
    When an architecture image-capability field exists we filter to
    image-capable models; otherwise we list everything (public list without
    that field). F49: the published metadata (modalities, supported
    parameters, context/completion limits) is recorded per model so a
    selection is validated against the model rather than the provider.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = _http_get_json(_OPENROUTER_MODELS_URL, headers=headers or None)
    models = []
    observations = []
    for item in data.get("data", []) or []:
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        # Record BEFORE the image filter: a text-only model is exactly the
        # selection that must be refused for a vision role.
        observation = _observation_from_item(item)
        if observation:
            observations.append(observation)
        arch = item.get("architecture")
        if isinstance(arch, dict):
            # Determine if an image-capability field exists
            input_mods = arch.get("input_modalities")
            if isinstance(input_mods, list) and len(input_mods) > 0:
                # field exists — keep only image-capable
                has_image = any("image" in str(m).lower() for m in input_mods)
                if not has_image:
                    continue
            else:
                modality = arch.get("modality")
                if isinstance(modality, str) and modality.strip():
                    if "image" not in modality.lower():
                        continue
        display = mid.rsplit("/", 1)[-1] if "/" in mid else mid
        # Preserve full id but show short display
        models.append({"id": mid, "display": display})
    _record_observed_capabilities("openrouter", observations)
    return models


def _list_ollama_models():
    """Models installed in the LOCAL Ollama daemon (GET /api/tags).

    [F56] The list is what this machine actually has — the UI's promise is
    "pick a model that will run", and this is the only honest source. Models
    the daemon reports but that cannot answer a chat/intent call are dropped
    (an embedding model may well be installed for another Jarvis feature; it
    must not appear as a selectable brain). No key, no cloud call: a stopped
    Ollama server raises a clean ModelRegistryError instead of an empty list.
    """
    from backend.services.ollama_client import OLLAMA_BASE_URL
    data = _http_get_json(
        str(OLLAMA_BASE_URL).rstrip("/") + "/api/tags", timeout=(2, 5))
    models = []
    for item in data.get("models", []) or []:
        mid = str(item.get("name") or item.get("model") or "").strip()
        if not mid:
            continue
        usable = model_capabilities_for("ollama", mid) & {
            "streaming", "structured_output", "tool_calling"}
        if not usable:
            continue
        models.append({"id": mid, "display": mid})
    if not models:
        raise ModelRegistryError(
            "the local Ollama server reports no chat-capable models installed")
    return models


def _list_groq_models(api_key):
    """Groq models — static known list (no public list API)."""
    if not api_key:
        raise ModelRegistryError("no Groq API key configured")
    try:
        from backend.services.grok_client import VISION_MODEL as _gv, DEFAULT_MODEL as _gd
        models = []
        for mid in [_gv, _gd]:
            if mid and mid not in [m["id"] for m in models]:
                models.append({"id": mid, "display": mid})
        return models
    except Exception:
        # H8: "llama-3.3-70b-versatile" is RETIRED at Groq (404) - dropped from
        # the offered catalog so the UI can no longer select it.
        return [
            {"id": "qwen/qwen3.6-27b", "display": "qwen/qwen3.6-27b"},
        ]


def list_provider_models(provider_id):
    """Live model list for a provider. Raises ModelRegistryError on a bad
    key or unreachable endpoint so routes can answer with a clean 4xx."""
    pid = str(provider_id or "").strip()
    if pid == "gemini":
        key, _base = get_provider_credentials(pid)
        if not key:
            raise ModelRegistryError("no Gemini API key configured")
        return _list_gemini_models(key)
    if pid == "fireworks":
        key, _base = get_provider_credentials(pid)
        if not key:
            raise ModelRegistryError("no Fireworks API key configured")
        return _list_fireworks_models(key)
    if pid == "groq":
        key, _base = get_provider_credentials(pid)
        return _list_groq_models(key)
    if pid == "fish":
        key, _base = get_provider_credentials(pid)
        return _list_fish_models(key)
    if pid == "gtts":
        # Free, key-less engine: listing never depends on a credential, so
        # the option stays visible exactly when the user needs it most (i.e.
        # after the paid engine's credits run out).
        return _list_google_tts_models()
    if pid == "openrouter":
        key, _base = get_provider_credentials(pid)
        # OpenRouter list is public; missing key is allowed but if listing
        # fails (network/auth) we surface a clean 400. Let _list handle empty-key case.
        try:
            return _list_openrouter_models(key)
        except ModelRegistryError:
            raise
        except Exception as exc:
            raise ModelRegistryError(str(exc)[:160] or "openrouter list failed")
    if pid == "whisper":
        # Local whisper daemon: one static option, no list API and no key.
        return [{"id": "whisper-local", "display": "whisper-local"}]
    if pid == "inworld":
        # Static env-driven option (same read as transcription.py); no
        # network needed to list, so a missing key still lists fine — the
        # UI gates selectability on the masked has_key flag.
        m = str(os.getenv("INWORLD_STT_MODEL", "") or "").strip() or "inworld/inworld-stt-1"
        return [{"id": m, "display": m}]
    if pid == "ollama":
        # [F56] Local, key-less, live: whatever is installed right now.
        return _list_ollama_models()
    for p in _custom_providers():
        if p.get("id") == pid:
            if not p.get("api_key") or not p.get("base_url"):
                raise ModelRegistryError(
                    f"provider '{pid}' has no stored credentials"
                )
            return _list_openai_compat_models(p["base_url"], p["api_key"])
    raise ModelRegistryError(f"unknown provider '{pid}'")


def add_custom_provider(provider_id, name, api_key, base_url, capabilities=None):
    """Store a user-added OpenAI-compatible provider.

    The key is validated LIVE (a model-list call with that exact key must
    succeed) before anything is persisted, so junk keys never reach disk.

    F49: the provider's capability set is stored WITH the record as an
    explicit, reviewable authorization. Pass ``capabilities`` to authorize a
    narrower or wider set than the shipped OpenAI-compatible default; an
    unsupported name is rejected rather than silently ignored.
    """
    pid = str(provider_id or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,31}", pid or ""):
        raise ModelRegistryError(
            "provider id must be 1-32 chars: lowercase letters, digits, - or _"
        )
    if pid in ENV_PROVIDERS:
        raise ModelRegistryError(f"provider id '{pid}' is reserved")
    display_name = str(name or "").strip() or pid
    key = str(api_key or "").strip()
    base = str(base_url or "").strip().rstrip("/")
    if not key:
        raise ModelRegistryError("api_key is required")
    if not (base.startswith("http://") or base.startswith("https://")):
        raise ModelRegistryError("base_url must start with http:// or https://")
    if capabilities is None:
        declared_capabilities = sorted(_CUSTOM_PROVIDER_DEFAULT_CAPABILITIES)
        capabilities_source = "custom_default"
    else:
        if isinstance(capabilities, str):
            capabilities = [capabilities]
        try:
            declared_capabilities = sorted(
                {str(c).strip() for c in capabilities if str(c).strip()})
        except TypeError:
            raise ModelRegistryError("capabilities must be a list of names")
        unsupported = [
            c for c in declared_capabilities if c not in KNOWN_CAPABILITIES]
        if unsupported:
            raise ModelRegistryError(
                "unknown capability name(s): %s" % ", ".join(unsupported))
        capabilities_source = "custom_declared"

    # Fast-fail duplicate check BEFORE the (network) live validation, so a
    # duplicate never costs a round-trip. The authoritative check runs
    # inside the locked read-modify-write below.
    if any(p.get("id") == pid for p in _custom_providers()):
        raise ModelRegistryError(f"provider '{pid}' already exists")

    try:
        probe = _list_openai_compat_models(base, key)
    except ModelRegistryError:
        raise
    except Exception:
        raise ModelRegistryError(
            "could not reach the provider to validate the API key"
        )
    if not probe:
        raise ModelRegistryError(
            "API key validated but the provider returned no models"
        )

    stored = {
        "id": pid,
        "name": display_name,
        "kind": "openai_compatible",
        "base_url": base,
        "api_key": key,
        "capabilities": declared_capabilities,
        "capabilities_source": capabilities_source,
    }
    # Hold _lock across the ENTIRE read-modify-write (the network validation
    # already happened outside — no I/O under the lock) so a concurrent
    # mutation (e.g. the default chat model being switched) survives this
    # full-file write instead of being silently clobbered.
    with _lock:
        settings = _load_unlocked()
        providers = [
            p for p in settings.get("custom_providers", [])
            if isinstance(p, dict)
        ]
        if any(p.get("id") == pid for p in providers):
            # raced in after the fast-fail check above
            raise ModelRegistryError(f"provider '{pid}' already exists")
        providers.append(stored)
        settings["custom_providers"] = providers
        _save_unlocked(settings)
    return _mask_provider(stored)
