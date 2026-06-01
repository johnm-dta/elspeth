"""Curated server-secret inventory backed by environment variables.

Only exposes secrets whose names are in the configured allowlist.
Does NOT dump arbitrary env vars to the browser.
"""

from __future__ import annotations

import os

from elspeth.contracts.secrets import FingerprintKeyMissingError, SecretInventoryItem
from elspeth.core.security.secret_loader import SecretNotFoundError, SecretRef
from elspeth.web.secrets.user_store import _compute_fingerprint, _fingerprint_key_available
from elspeth.web.validation import validate_secret_name

_RESERVED_PREFIX = "ELSPETH_"
"""Server secrets are for third-party API keys, not ELSPETH internals.

Any env var whose name starts with ELSPETH_ is an internal secret
(fingerprint key, JWT signing key, audit passphrase, etc.) and must
never be exposed through the server secret store.
"""


def _is_reserved(name: str) -> bool:
    return name.startswith(_RESERVED_PREFIX)


def _env_secret_value(name: str) -> str | None:
    """Read a server-secret env var, treating set-but-empty as absent.

    ``os.environ`` is a Tier-3 boundary (the process environment is authored by
    the deployment operator, not by us).  An unset env var is an expected
    external state, and a set-but-empty (``""``) env var is a non-crashing
    anomaly that must NOT be exposed as a resolvable secret.  Both collapse to
    ``None`` (honest absence) here so callers never have to re-derive the
    "set AND non-empty" predicate — there is exactly one place to get it right.
    """
    if name not in os.environ:
        return None
    value = os.environ[name]
    return value if value else None


class ServerSecretStore:
    """Env-var secret store restricted to an explicit allowlist.

    The allowlist is set from ``WebSettings.server_secret_allowlist``
    so that only operator-approved names are ever readable.
    """

    def __init__(self, allowlist: tuple[str, ...]) -> None:
        validated = tuple(validate_secret_name(name, field_name="Server secret allowlist entry") for name in allowlist)
        reserved = [n for n in validated if _is_reserved(n)]
        if reserved:
            raise ValueError(f"Server secret allowlist contains ELSPETH internal names that must never be exposed: {sorted(reserved)}")
        self._allowlist = validated

    def has_secret(self, name: str) -> bool:
        """Check if an allowlisted env var secret is resolvable.

        Returns True only when the name is allowlisted, the env var is
        set, AND the fingerprint key is available.  This aligns with
        get_secret() on the success path.

        Reserved (ELSPETH_*) names are never resolvable through this
        store and return False.  get_secret() continues to raise
        SecretNotFoundError for them — the asymmetry is deliberate:
        has_secret() participates in WebSecretService.has_ref()'s boolean
        composition (user-scope OR server-scope), and raising here would
        turn a probe into a 500 whenever the user-scope lookup returns
        False for a reserved name.  On the resolve path, get_secret()'s
        raise is caught by WebSecretService.resolve() (symmetric with
        allowlist/env-var misses), keeping the has_ref == True ⟺
        resolve() != None invariant.
        """
        if _is_reserved(name):
            return False
        return name in self._allowlist and _env_secret_value(name) is not None and _fingerprint_key_available()

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Resolve an allowlisted env var.

        Raises:
            FingerprintKeyMissingError: If ``ELSPETH_FINGERPRINT_KEY`` is
                unset.  Checked first so deployment misconfiguration
                surfaces as 503 at the HTTP boundary rather than being
                indistinguishable from "secret not in allowlist".
            SecretNotFoundError: If *name* is reserved, not in the
                allowlist, or the env var is unset / empty.  These cases
                are deliberately indistinguishable to callers — a
                non-allowlisted probe must not reveal whether the name
                matches a real env var.
        """
        # Fingerprint-availability check intentionally precedes the reserved
        # / allowlist / env-var checks: deployment misconfiguration is a
        # global state and the typed exception carries no per-secret
        # information that a probing caller could exploit.  This also
        # aligns with user_store.get_secret which fails fast on the same
        # deployment issue.
        if not _fingerprint_key_available():
            raise FingerprintKeyMissingError(f"Secret {name!r} is not resolvable — ELSPETH_FINGERPRINT_KEY is not set")
        if _is_reserved(name):
            raise SecretNotFoundError(name)
        if name not in self._allowlist:
            raise SecretNotFoundError(name)
        value = _env_secret_value(name)  # Tier 3: env vars are external input
        if value is None:
            raise SecretNotFoundError(name)
        fp = _compute_fingerprint(name, value)
        return value, SecretRef(name=name, fingerprint=fp, source="env")

    def list_secrets(self) -> list[SecretInventoryItem]:
        """Return metadata for every allowlisted name (never exposes values).

        The ``available`` flag requires both the env var being set AND
        the fingerprint key being configured — without the latter,
        get_secret() would raise FingerprintKeyMissingError.

        Reason precedence is **fingerprint-key-first**: when
        ``ELSPETH_FINGERPRINT_KEY`` is unset, every entry reports
        ``fingerprint_resolver_not_configured`` regardless of env-var
        state.  Rationale: the fingerprint-key gap is global deployment
        misconfiguration; reporting per-secret env-var details under it
        would mislead operators into chasing the wrong cause.  The same
        precedence is mirrored in ``UserSecretStore.list_secrets`` so
        an operator sees one consistent reason taxonomy across both
        scopes.
        """
        can_fingerprint = _fingerprint_key_available()
        items: list[SecretInventoryItem] = []
        for name in self._allowlist:
            if _is_reserved(name):
                continue
            if not can_fingerprint:
                items.append(
                    SecretInventoryItem(
                        name=name,
                        scope="server",
                        available=False,
                        source_kind="env",
                        reason="fingerprint_resolver_not_configured",
                    )
                )
                continue
            if _env_secret_value(name) is None:
                items.append(
                    SecretInventoryItem(
                        name=name,
                        scope="server",
                        available=False,
                        source_kind="env",
                        reason="env_var_not_set",
                    )
                )
                continue
            items.append(
                SecretInventoryItem(
                    name=name,
                    scope="server",
                    available=True,
                    source_kind="env",
                )
            )
        return items
