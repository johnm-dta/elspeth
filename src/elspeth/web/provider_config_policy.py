"""Web-authored provider configuration policy helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

MANAGED_IDENTITY_POLICY_ERROR: Final[str] = (
    "Azure Search managed identity is a server credential and cannot be enabled "
    "from web-authored rag_retrieval provider_config. Use api_key authentication "
    "or an operator-controlled named connector/allowlist before enabling managed identity."
)

_FALSE_LITERALS: Final[frozenset[str]] = frozenset({"", "0", "false", "f", "no", "n", "off"})
_TRUE_LITERALS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})


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
