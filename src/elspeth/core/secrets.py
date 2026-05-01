"""Secret resolution helpers — tree-walk and error types.

Layer: L1 (core). Imports from L0 (contracts) only.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from typing import Any

from elspeth.contracts.secrets import ResolvedSecret, WebSecretResolver

_EXACT_ENV_VAR_REF_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}")

# Field-name heuristic for credential-bearing options.
#
# This is the closed list of field names that the runtime (config fingerprinting)
# treats as carrying credentials, and that the composer's fabrication-aware
# `secret_refs` validator (web.execution.validation) treats as requiring a
# wired `{secret_ref: ...}` or inventory env-marker.  Keeping the predicate in
# core.secrets gives runtime and validate a single source of truth — divergence
# would re-open the validator/runtime parity gap that issue elspeth-72d1dccd44
# was filed to close.
SECRET_FIELD_NAMES = frozenset(
    {
        "api_key",
        "api-key",
        "authorization",
        "connection_string",
        "credential",
        "password",
        "secret",
        "token",
        "x-api-key",
    }
)

SECRET_FIELD_SUFFIXES = ("_secret", "_key", "_token", "_password", "_credential", "_connection_string")


def is_secret_field(field_name: str) -> bool:
    """Return True when a field name represents a credential-bearing option.

    Case-insensitive: matches an exact name in ``SECRET_FIELD_NAMES`` or any
    suffix in ``SECRET_FIELD_SUFFIXES``.
    """
    normalized = field_name.lower()
    return normalized in SECRET_FIELD_NAMES or normalized.endswith(SECRET_FIELD_SUFFIXES)


class SecretResolutionError(Exception):
    """Raised when one or more secret refs cannot be resolved."""

    def __init__(self, missing: list[str]) -> None:
        self.missing = missing
        names = ", ".join(missing)
        super().__init__(f"Cannot resolve secret references: {names}")


def resolve_secret_refs(
    config: dict[str, Any],
    resolver: WebSecretResolver,
    user_id: str,
    *,
    env_ref_names: Collection[str] = frozenset(),
) -> tuple[dict[str, Any], list[ResolvedSecret]]:
    """Walk a config dict tree and replace {"secret_ref": "NAME"} with resolved values.

    Exact ``${NAME}`` / ``${NAME:-default}`` strings are also treated as
    secret refs when ``NAME`` is supplied in ``env_ref_names``. This lets
    web-authored pipelines route known secret inventory names through the
    same resolver/audit path instead of falling through to blind config
    env-var expansion. Embedded strings such as ``"prefix-${NAME}"`` are
    intentionally left alone.

    Returns (resolved_config, list_of_resolutions).
    Raises SecretResolutionError listing ALL missing refs (not one at a time).
    The returned config is a deep copy — the original is not mutated.
    """
    result = deepcopy(config)
    resolutions: list[ResolvedSecret] = []
    missing: list[str] = []
    _walk(result, resolver, user_id, resolutions, missing, env_ref_names)
    if missing:
        raise SecretResolutionError(missing)
    return result, resolutions


def _is_secret_ref(value: Any) -> str | None:
    """If value is {"secret_ref": "NAME"}, return NAME. Else None."""
    if isinstance(value, Mapping) and len(value) == 1 and "secret_ref" in value:
        ref = value["secret_ref"]
        if isinstance(ref, str):
            return ref
    return None


def _is_secret_env_ref(value: Any, env_ref_names: Collection[str]) -> str | None:
    """If value is an exact ${NAME} string for a declared secret, return NAME."""
    if not isinstance(value, str) or not env_ref_names:
        return None
    match = _EXACT_ENV_VAR_REF_PATTERN.fullmatch(value)
    if match is None:
        return None
    ref_name = match.group(1)
    if ref_name in env_ref_names:
        return ref_name
    return None


def secret_env_ref_name(value: Any, env_ref_names: Collection[str]) -> str | None:
    """Return a declared secret name from an exact env marker, if present."""
    return _is_secret_env_ref(value, env_ref_names)


def _walk(
    obj: Any,
    resolver: WebSecretResolver,
    user_id: str,
    resolutions: list[ResolvedSecret],
    missing: list[str],
    env_ref_names: Collection[str],
) -> None:
    """Recursively walk and replace secret refs in-place.

    Uses Mapping for isinstance checks to cover dict, MappingProxyType,
    OrderedDict, etc. After deepcopy(), MappingProxyType becomes dict,
    so in-place mutation via obj[key] is safe at runtime.
    """
    if isinstance(obj, Mapping):
        for key in list(obj.keys()):
            ref_name = _is_secret_ref(obj[key])
            if ref_name is None:
                ref_name = _is_secret_env_ref(obj[key], env_ref_names)
            if ref_name is not None:
                resolved = resolver.resolve(user_id, ref_name)
                if resolved is None:
                    missing.append(ref_name)
                else:
                    obj[key] = resolved.value  # type: ignore[index]  # safe: deepcopy produces dict
                    resolutions.append(resolved)
            else:
                _walk(obj[key], resolver, user_id, resolutions, missing, env_ref_names)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            ref_name = _is_secret_ref(item)
            if ref_name is None:
                ref_name = _is_secret_env_ref(item, env_ref_names)
            if ref_name is not None:
                resolved = resolver.resolve(user_id, ref_name)
                if resolved is None:
                    missing.append(ref_name)
                else:
                    obj[i] = resolved.value
                    resolutions.append(resolved)
            else:
                _walk(item, resolver, user_id, resolutions, missing, env_ref_names)
