"""Web-authored transform configuration policy helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import TypeAdapter
from pydantic import ValidationError as PydanticValidationError

WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS: Final[int] = 30

MANAGED_IDENTITY_POLICY_ERROR: Final[str] = (
    "Azure Search managed identity is a server credential and cannot be enabled "
    "from web-authored rag_retrieval provider_config. Use api_key authentication "
    "or an operator-controlled named connector/allowlist before enabling managed identity."
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
    if (
        max_retry_seconds is None
        or max_retry_seconds > WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS
    ):
        return LLM_RETRY_BUDGET_POLICY_ERROR

    return None
