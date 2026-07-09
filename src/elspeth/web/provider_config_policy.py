"""Web-authored transform configuration policy helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.plugins.transforms.llm.providers.openrouter import (
    OPENROUTER_BASE_URL,
    normalize_openrouter_base_url,
)

WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS: Final[int] = 30

MANAGED_IDENTITY_POLICY_ERROR: Final[str] = (
    "Azure Search managed identity is a server credential and cannot be enabled "
    "from web-authored rag_retrieval provider_config. Use api_key authentication "
    "or an operator-controlled named connector/allowlist before enabling managed identity."
)
LLM_BASE_URL_POLICY_ERROR: Final[str] = (
    "Web-authored OpenRouter LLM nodes may not override base_url. The api_key is "
    "resolved server-side, so a custom base_url — a loopback/private address or any "
    "non-canonical host — would direct the server-held bearer credential to an "
    "author-chosen destination (a credential-egress / SSRF path). Omit base_url to use "
    f"the canonical OpenRouter endpoint ({OPENROUTER_BASE_URL}); a private OpenAI-compatible "
    "gateway requires an operator-controlled runtime outside the web composer."
)
LLM_RETRY_BUDGET_POLICY_ERROR: Final[str] = (
    "Web-authored sequential multi-query LLM nodes must explicitly set "
    f"max_capacity_retry_seconds <= {WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS}. "
    "The LLM transform default is one hour, which can monopolize the web execution worker. "
    "Set max_capacity_retry_seconds to a small positive value or use pool_size > 1 for pooled retry handling."
)

_FALSE_LITERALS: Final[frozenset[str]] = frozenset({"", "0", "false", "f", "no", "n", "off"})
_TRUE_LITERALS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
_INT_ADAPTER: Final[TypeAdapter[int]] = TypeAdapter(int)


@trust_boundary(
    tier=3,
    source="web-authored provider_config use_managed_identity value (untrusted scalar)",
    source_param="value",
    suppresses=("R5",),
    invariant=(
        "recognized false-y forms return False, recognized truthy forms return True, and any "
        "ambiguous present value fails closed to True (policy error fires); never raises"
    ),
    non_raising=True,
)
def _provider_config_enables_managed_identity(value: object) -> bool:
    """Return whether a raw web-authored value enables managed identity.

    Pydantic accepts common bool-like strings for bool fields. This helper
    mirrors that permissiveness for recognized values and otherwise fails
    closed when the sensitive key is present with an ambiguous truthy value.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _FALSE_LITERALS:
            return False
        if normalized in _TRUE_LITERALS:
            return True
        return True
    return bool(value)


@trust_boundary(
    tier=3,
    source="web-authored RAG provider config (untrusted composer-author options mapping)",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant=(
        "returns MANAGED_IDENTITY_POLICY_ERROR only when a well-formed azure_search "
        "provider_config enables managed identity; any missing or malformed key fails "
        "closed to None (no policy error) and never raises"
    ),
    non_raising=True,
)
def web_rag_provider_config_policy_error(options: Mapping[str, Any]) -> str | None:
    """Reject web-authored RAG Azure Search configs that enable managed identity."""
    if options.get("provider") != "azure_search":
        return None

    provider_config = options.get("provider_config")
    if not isinstance(provider_config, Mapping):
        return None

    if _provider_config_enables_managed_identity(provider_config.get("use_managed_identity")):
        return MANAGED_IDENTITY_POLICY_ERROR

    return None


def _positive_int_or_none(value: object) -> int | None:
    try:
        parsed = _INT_ADAPTER.validate_python(value)
    except PydanticValidationError:
        return None
    if parsed > 0:
        return parsed
    return None


@trust_boundary(
    tier=3,
    source="web/composer-authored LLM transform options (untrusted author-supplied mapping)",
    source_param="options",
    suppresses=("R1",),
    invariant=(
        "absent 'queries' means no multi-query retry-budget policy applies (None); a "
        "malformed or unbounded retry budget returns LLM_RETRY_BUDGET_POLICY_ERROR; "
        "never raises on malformed options"
    ),
    non_raising=True,
)
def web_llm_retry_budget_policy_error(plugin: str | None, options: Mapping[str, Any]) -> str | None:
    """Reject web-authored sequential multi-query LLM configs with unbounded local retries."""
    if plugin != "llm":
        return None
    if options.get("queries") is None:
        return None

    pool_size = _positive_int_or_none(options.get("pool_size", 1))
    if pool_size is None:
        return LLM_RETRY_BUDGET_POLICY_ERROR
    if pool_size > 1:
        return None

    max_retry_seconds = _positive_int_or_none(options.get("max_capacity_retry_seconds"))
    if max_retry_seconds is None or max_retry_seconds > WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS:
        return LLM_RETRY_BUDGET_POLICY_ERROR

    return None


@trust_boundary(
    tier=3,
    source="web-authored pipeline LLM provider config (untrusted author-supplied options mapping)",
    source_param="options",
    suppresses=("R1", "R5"),
    invariant=(
        "omitted base_url falls back to the canonical OpenRouter endpoint (None); an "
        "explicit base_url is rejected with LLM_BASE_URL_POLICY_ERROR unless it "
        "normalises to the canonical endpoint; never raises on malformed options"
    ),
    non_raising=True,
)
def web_llm_base_url_policy_error(plugin: str | None, options: Mapping[str, Any]) -> str | None:
    """Reject web-authored OpenRouter LLM configs that override base_url.

    The OpenRouter provider sends ``Authorization: Bearer <api_key>`` to whatever
    ``base_url`` names. In a web-authored pipeline the ``api_key`` is resolved
    server-side (a ``{"secret_ref": ...}`` against the deployment secret
    inventory, which may be a *server*-scoped credential the author cannot read),
    while ``base_url`` is set by the untrusted pipeline author. That asymmetry
    turns a custom base_url into a credential-egress / SSRF vector — the author
    can direct the server's bearer to a loopback service, a private host, or an
    attacker-controlled public host.

    The plugin config-validator deliberately tolerates HTTP loopback so the
    *CLI* dev examples (local ChaosLLM at ``http://127.0.0.1:8199/v1``) run; that
    single-machine threat model does not hold for a hosted server, so the web
    execution boundary pins base_url here. An unset base_url falls back to the
    canonical OpenRouter endpoint and is always allowed; an explicit base_url is
    allowed only when it normalises to that same canonical endpoint. Private
    OpenAI-compatible gateways are an operator-controlled runtime concern, not a
    web-author option — mirroring the managed-identity and web_scrape network
    policies.
    """
    if plugin != "llm":
        return None
    base_url = options.get("base_url")
    if base_url is None:
        return None
    if not isinstance(base_url, str):
        # Non-string base_url is rejected at config construction (pydantic); the
        # network-policy gate only adjudicates author-chosen string endpoints.
        return None
    if normalize_openrouter_base_url(base_url.strip()) == normalize_openrouter_base_url(OPENROUTER_BASE_URL):
        return None
    return LLM_BASE_URL_POLICY_ERROR
