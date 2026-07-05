"""Secret resolution helpers — tree-walk and error types.

Layer: L1 (core). Imports from L0 (contracts) only.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from copy import deepcopy
from typing import Any

from elspeth.contracts.secrets import ResolvedSecret, SecretRefPlacementViolation, WebSecretResolver

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

# Exact-name exemptions from the suffix heuristic: structural (non-credential)
# fields whose names happen to end in a secret suffix. The suffix set must
# stay broad — real credential fields such as Langfuse tracing's
# ``secret_key``/``public_key`` are matched only by the bare ``_key`` suffix,
# so narrowing it would silently drop those from audit fingerprinting.
# Exemptions are therefore exact lowercase names, never suffixes, and each
# entry cites the plugin field it exists for. Failure asymmetry: a missing
# exemption is a visible ``fabricated_secret`` block (add the name here); a
# wrong exemption leaks a credential into the audit trail — keep this list
# short and literal.
STRUCTURAL_FIELD_EXEMPTIONS = frozenset(
    {
        # JSONSource.data_key / azure_blob_source data_key: the key that names
        # the array to extract from a JSON document (e.g. "results").
        "data_key",
        # DataverseSink.alternate_key: the alternate-key *column name* used to
        # route upserts (e.g. "crabc_code").
        "alternate_key",
    }
)


def is_secret_field(field_name: str) -> bool:
    """Return True when a field name represents a credential-bearing option.

    Case-insensitive: matches an exact name in ``SECRET_FIELD_NAMES`` or any
    suffix in ``SECRET_FIELD_SUFFIXES``, unless the name is an exact
    structural exemption in ``STRUCTURAL_FIELD_EXEMPTIONS``.
    """
    normalized = field_name.lower()
    if normalized in STRUCTURAL_FIELD_EXEMPTIONS:
        return False
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


# A non-empty, obviously-non-secret stand-in substituted for wired
# ``{secret_ref: NAME}`` markers when a config is validated WITHOUT a secret
# resolver (the YAML-export preflight, which deliberately withholds the
# resolver so resolved secret values can never reach plugin error prose).
# Long enough not to trip a plausible credential-field min-length check, and
# self-describing so it can never be mistaken for a real credential.
SECRET_REF_VALIDATION_PLACEHOLDER = "elspeth-preflight-secret-placeholder"


def redact_secret_refs_for_validation(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``config`` with every wired ``{secret_ref: NAME}``
    marker replaced by :data:`SECRET_REF_VALIDATION_PLACEHOLDER`.

    For validation paths that run without a secret resolver. An unresolved
    marker is *valid wiring*, not a config error — but plugin config models
    type credential fields as ``str`` (e.g. OpenRouter ``api_key: str``), so an
    unsubstituted marker dict fails instantiation with "Input should be a valid
    string". Substituting a placeholder lets such a path validate pipeline
    *structure* without the real secret. The original ``config`` is not mutated;
    the caller serialises this copy for the settings loader only.
    """
    result = deepcopy(config)
    _walk_redact(result)
    return result


def _walk_redact(obj: Any) -> None:
    """Recursively replace ``{secret_ref: NAME}`` markers with the placeholder.

    Mirrors :func:`_walk`'s traversal. After ``deepcopy`` every Mapping is a
    plain ``dict``, so in-place ``obj[key] = ...`` assignment is safe.
    """
    if isinstance(obj, Mapping):
        for key in list(obj.keys()):
            if _is_secret_ref(obj[key]) is not None:
                obj[key] = SECRET_REF_VALIDATION_PLACEHOLDER  # type: ignore[index]  # safe: deepcopy produces dict
            else:
                _walk_redact(obj[key])
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if _is_secret_ref(item) is not None:
                obj[i] = SECRET_REF_VALIDATION_PLACEHOLDER
            else:
                _walk_redact(item)


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


def is_secret_ref_marker(value: Any) -> bool:
    """Return True when value is exactly a wired ``{"secret_ref": NAME}`` marker."""
    return _is_secret_ref(value) is not None


def is_wired_secret_value(value: Any, env_ref_names: Collection[str] = frozenset()) -> bool:
    """Return True when value uses an approved deferred-secret syntax.

    Valid wired forms are:
    - ``{"secret_ref": "NAME"}``
    - exact ``${NAME}`` / ``${NAME:-default}`` strings where NAME is present
      in the caller-provided secret inventory.
    """
    if is_secret_ref_marker(value):
        return True
    if isinstance(value, str):
        return _is_secret_env_ref(value, env_ref_names) is not None
    return False


def collect_credential_field_violations(
    options: Any,
    env_ref_names: Collection[str] = frozenset(),
    *,
    additional_credential_fields: Collection[str] = frozenset(),
) -> list[str]:
    """Return credential-bearing field names that contain literal strings.

    The returned list intentionally names fields only, never values. It uses
    the same field-name predicate as runtime fingerprinting, and treats only
    deferred-secret markers as provisioned. Missing, empty, ``None``, and
    non-string values are left for plugin config validation to classify.
    Callers may pass plugin-specific exact field names, such as the database
    sink's whole-DSN ``url`` field.
    """
    credential_exact = frozenset(field.lower() for field in additional_credential_fields)
    return _collect_credential_field_violations(
        options,
        env_ref_names,
        credential_exact,
    )


def _collect_credential_field_violations(
    options: Any,
    env_ref_names: Collection[str],
    additional_credential_fields: Collection[str],
) -> list[str]:
    violations: list[str] = []
    if isinstance(options, Mapping):
        if is_secret_ref_marker(options):
            return violations
        for key, value in options.items():
            if isinstance(key, str) and _field_allows_secret_ref(key, additional_credential_fields):
                if is_wired_secret_value(value, env_ref_names):
                    continue
                if value is None or value == "":
                    continue
                if isinstance(value, str):
                    violations.append(key)
                    continue
                continue
            violations.extend(_collect_credential_field_violations(value, env_ref_names, additional_credential_fields))
    elif isinstance(options, (list, tuple)):
        for item in options:
            violations.extend(_collect_credential_field_violations(item, env_ref_names, additional_credential_fields))
    return violations


def collect_disallowed_secret_ref_markers(
    options: Any,
    env_ref_names: Collection[str] = frozenset(),
    *,
    additional_allowed_fields: Collection[str] = frozenset(),
) -> list[SecretRefPlacementViolation]:
    """Return secret-ref markers placed outside credential-bearing fields.

    The returned objects name field paths and secret names only. They never
    include secret values. A marker is allowed when its immediate containing
    field is credential-bearing according to ``is_secret_field`` or is listed in
    ``additional_allowed_fields`` by the caller for plugin-specific credentials
    such as the database sink's whole-DSN ``url`` field.
    """
    allowed_exact = frozenset(field.lower() for field in additional_allowed_fields)
    violations: list[SecretRefPlacementViolation] = []
    _collect_disallowed_secret_ref_markers(
        options,
        env_ref_names,
        allowed_exact,
        path=(),
        violations=violations,
    )
    return violations


def _field_allows_secret_ref(field_name: str, additional_allowed_fields: Collection[str]) -> bool:
    return is_secret_field(field_name) or field_name.lower() in additional_allowed_fields


def _collect_disallowed_secret_ref_markers(
    obj: Any,
    env_ref_names: Collection[str],
    additional_allowed_fields: Collection[str],
    *,
    path: tuple[str, ...],
    violations: list[SecretRefPlacementViolation],
) -> None:
    ref_name = _is_secret_ref(obj)
    if ref_name is None:
        ref_name = _is_secret_env_ref(obj, env_ref_names)
    if ref_name is not None:
        field_name = path[-1] if path else ""
        if not _field_allows_secret_ref(field_name, additional_allowed_fields):
            violations.append(
                SecretRefPlacementViolation(
                    field_path=".".join(path) if path else "<root>",
                    secret_name=ref_name,
                )
            )
        return

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            next_path = (*path, key) if isinstance(key, str) else (*path, f"<{type(key).__name__}>")
            _collect_disallowed_secret_ref_markers(
                value,
                env_ref_names,
                additional_allowed_fields,
                path=next_path,
                violations=violations,
            )
    elif isinstance(obj, (list, tuple)):
        for index, item in enumerate(obj):
            _collect_disallowed_secret_ref_markers(
                item,
                env_ref_names,
                additional_allowed_fields,
                path=(*path, f"[{index}]"),
                violations=violations,
            )


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
