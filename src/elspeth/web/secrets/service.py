"""WebSecretService -- composes user + server stores; ScopedSecretResolver adapts it to WebSecretResolver."""

from __future__ import annotations

import threading
import time

import structlog

from elspeth.contracts.secrets import (
    CreateSecretResult,
    FingerprintKeyMissingError,
    ResolvedSecret,
    SecretDecryptionError,
    SecretInventoryItem,
)
from elspeth.core.security.secret_loader import SecretNotFoundError
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.user_store import UserSecretStore

_slog = structlog.get_logger()

# Rate-limit deployment-error breadcrumbs so an unconfigured environment
# remains visible to operators without flooding logs on every secret
# lookup. The resolve path intentionally swallows
# ``FingerprintKeyMissingError`` to preserve its "secret absent"
# contract, so this emission is the operator-visible signal.
_FINGERPRINT_MISSING_LOG_INTERVAL_SECONDS = 60.0
_fingerprint_missing_last_logged_at: float | None = None
_fingerprint_missing_suppressed = 0
_fingerprint_missing_state_lock = threading.Lock()
_SECRET_DECRYPTION_LOG_INTERVAL_SECONDS = 60.0
_secret_decryption_last_logged_at: float | None = None
_secret_decryption_suppressed = 0
_secret_decryption_state_lock = threading.Lock()


def _rate_limit_monotonic() -> float:
    """Return the monotonic clock used by rate-limited secret breadcrumbs."""
    return time.monotonic()


def _log_fingerprint_missing_rate_limited() -> None:
    """Emit a rate-limited breadcrumb when the resolve path swallows
    FingerprintKeyMissingError.

    This is a web-layer deployment-error slog emission, matching the
    precedent set by the app-level ``FingerprintKeyMissingError`` HTTP
    handler in :mod:`elspeth.web.app` (``http_fingerprint_key_missing``).
    The resolve path cannot raise — it supports the pipeline-validation
    aggregation invariant (all misses bucketed into a single
    ``SecretResolutionError``) — so the typed signal is converted into an
    operational breadcrumb instead of a per-request 503.

    Per CLAUDE.md ``logging-telemetry-policy``: this is a deployment
    misconfiguration event, not pipeline activity.  The audit trail
    (Landscape) records WHAT the pipeline did; it does not have a slot
    for "server-wide secrets subsystem is misconfigured."  Telemetry
    would be preferred but the web layer does not yet have an
    operational-metric emitter distinct from slog.
    """
    global _fingerprint_missing_last_logged_at, _fingerprint_missing_suppressed

    with _fingerprint_missing_state_lock:
        now_monotonic = _rate_limit_monotonic()
        if (
            _fingerprint_missing_last_logged_at is not None
            and now_monotonic - _fingerprint_missing_last_logged_at < _FINGERPRINT_MISSING_LOG_INTERVAL_SECONDS
        ):
            _fingerprint_missing_suppressed += 1
            return

        suppressed_since_last_emit = _fingerprint_missing_suppressed
        _fingerprint_missing_last_logged_at = now_monotonic
        _fingerprint_missing_suppressed = 0
    _slog.error(
        "secret_resolve_fingerprint_key_missing",
        detail=(
            "ELSPETH_FINGERPRINT_KEY is unset or misconfigured; every "
            "call to WebSecretService.resolve() will return None until "
            "the deployment environment is fixed."
        ),
        emit_interval_seconds=_FINGERPRINT_MISSING_LOG_INTERVAL_SECONDS,
        suppressed_since_last_emit=suppressed_since_last_emit,
    )


def _log_secret_decryption_rate_limited() -> None:
    """Emit a rate-limited breadcrumb when resolve() swallows SecretDecryptionError.

    Mirrors ``_log_fingerprint_missing_rate_limited``: the pipeline-path
    contract still requires ``resolve()`` to collapse unresolvable refs
    into ``None``, but operators need a deployment-visible signal when
    stored secrets have become unreadable after key rotation or corruption.
    """
    global _secret_decryption_last_logged_at, _secret_decryption_suppressed

    with _secret_decryption_state_lock:
        now_monotonic = _rate_limit_monotonic()
        if (
            _secret_decryption_last_logged_at is not None
            and now_monotonic - _secret_decryption_last_logged_at < _SECRET_DECRYPTION_LOG_INTERVAL_SECONDS
        ):
            _secret_decryption_suppressed += 1
            return

        suppressed_since_last_emit = _secret_decryption_suppressed
        _secret_decryption_last_logged_at = now_monotonic
        _secret_decryption_suppressed = 0
    _slog.error(
        "secret_resolve_decryption_failed",
        detail=(
            "A stored web secret exists but cannot be decrypted with the current "
            "server key material; affected calls to WebSecretService.resolve() "
            "will return None until the secret is re-saved or the deployment "
            "key configuration is restored."
        ),
        emit_interval_seconds=_SECRET_DECRYPTION_LOG_INTERVAL_SECONDS,
        suppressed_since_last_emit=suppressed_since_last_emit,
    )


class WebSecretService:
    """Chained secret resolution: user -> server.

    All methods require ``auth_provider_type`` to scope user secrets by
    auth provider — the web layer treats (user_id, auth_provider_type) as
    the ownership boundary.  Server secrets are deployment-scoped and do
    not use auth_provider_type.

    Also exposes ``set_user_secret`` / ``delete_user_secret`` for the
    REST API layer.
    """

    def __init__(self, user_store: UserSecretStore, server_store: ServerSecretStore) -> None:
        self._user_store = user_store
        self._server_store = server_store

    # -- Resolution methods ------------------------------------------------

    def list_refs(self, user_id: str, *, auth_provider_type: str) -> list[SecretInventoryItem]:
        """Merge user and server inventories; user scope wins on name clash."""
        user_items = {item.name: item for item in self._user_store.list_secrets(user_id=user_id, auth_provider_type=auth_provider_type)}
        server_items = {item.name: item for item in self._server_store.list_secrets()}
        merged = {**server_items, **user_items}  # user scope wins
        return sorted(merged.values(), key=lambda x: x.name)

    def has_ref(self, user_id: str, name: str, *, auth_provider_type: str) -> bool:
        """Check whether *name* is resolvable in either scope.

        User scope shadows server scope by name: if the user has stored a
        row for ``name``, that row controls availability even when it is
        currently undecryptable.  This keeps validation aligned with
        list_refs(), where user scope also wins on name clash.
        """
        if self._user_store.has_secret_record(name, user_id=user_id, auth_provider_type=auth_provider_type):
            return self._user_store.has_secret(name, user_id=user_id, auth_provider_type=auth_provider_type)
        return self._server_store.has_secret(name)

    def resolve(self, user_id: str, name: str, *, auth_provider_type: str) -> ResolvedSecret | None:
        """Resolve a secret, trying user scope first then server.

        Returns None for any condition that makes the ref unresolvable —
        genuine absence, fingerprint-key misconfiguration, or
        decryption failure.  Callers (``resolve_secret_refs`` in
        ``core/secrets.py``) batch all misses into a single
        ``SecretResolutionError`` so a pipeline-validation pass reports
        every bad ref in one round trip.

        The typed exceptions from the stores (``FingerprintKeyMissingError``,
        ``SecretDecryptionError``) are intentionally swallowed **here** to
        preserve that aggregation contract.  Callers that DO want the
        typed signal (e.g., the ``/api/secrets/{name}/validate`` HTTP
        route and the ``set_user_secret`` write path) use
        :meth:`check_user_ref_resolvable` or :meth:`set_user_secret` —
        both of which propagate the typed errors so HTTP handlers can
        map them to 503 / 409 responses.

        TOCTOU note: the fetch sequence below opens a bounded number of
        independent reads (user scope: one record probe + one row fetch;
        server scope: one env read).  A concurrent ``DELETE`` or env-clear
        landing between the probe and the fetch is absorbed into the
        "absent" bucket.
        """
        try:
            # User scope first: the record-existence probe distinguishes
            # "no user-scope row, try server" from "user-scope row exists
            # — its state controls availability (shadowing rule)".  We do
            # NOT call has_secret as a pre-check here because it eats
            # decryption failures as False; the shadowing invariant
            # requires a failing user row to NOT fall through to server.
            if self._user_store.has_secret_record(name, user_id=user_id, auth_provider_type=auth_provider_type):
                value, ref = self._user_store.get_secret(name, user_id=user_id, auth_provider_type=auth_provider_type)
                return ResolvedSecret(name=name, value=value, scope="user", fingerprint=ref.fingerprint)
            # Server fallback — get_secret raises typed errors for each
            # failure class; all are absorbed into "None" below.
            value, ref = self._server_store.get_secret(name)
            return ResolvedSecret(name=name, value=value, scope="server", fingerprint=ref.fingerprint)
        except FingerprintKeyMissingError:
            # Deployment misconfiguration breadcrumb (rate limited) —
            # see ``_log_fingerprint_missing_rate_limited`` docstring.
            # Returning None preserves the pipeline-validation
            # aggregation contract (misses bucketed into a single
            # SecretResolutionError), but without this emission the
            # deployment error would be invisible to operators until
            # they hit the /validate HTTP route or the
            # ``set_user_secret`` write path.
            _log_fingerprint_missing_rate_limited()
            return None
        except SecretDecryptionError:
            _log_secret_decryption_rate_limited()
            return None
        except SecretNotFoundError:
            return None

    def check_user_ref_resolvable(
        self,
        user_id: str,
        name: str,
        *,
        auth_provider_type: str,
    ) -> bool:
        """Typed-error variant of :meth:`has_ref` for the HTTP validate route.

        Returns True if the ref resolves right now, False if genuinely
        absent.  **Propagates** typed store errors so HTTP handlers can
        translate them into actionable status codes rather than reporting
        a silent ``available=False`` that hides the deployment issue:

        Raises
        ------
        FingerprintKeyMissingError
            ``ELSPETH_FINGERPRINT_KEY`` is unset — mapped to 503 by the
            app-level handler.
        SecretDecryptionError
            Row exists but its ciphertext cannot be decrypted (key
            rotation, corruption) — mapped to 409 with re-save guidance.

        Distinct from :meth:`resolve` because the contracts differ:
        ``resolve`` supports the pipeline-path aggregation invariant
        (all misses bucketed into a single ``SecretResolutionError``),
        while this method surfaces per-condition typed signals the HTTP
        layer needs to give an API consumer something to act on.
        """
        if self._user_store.has_secret_record(name, user_id=user_id, auth_provider_type=auth_provider_type):
            # Exercises the full get_secret path so both
            # FingerprintKeyMissingError and SecretDecryptionError can
            # surface.  The plaintext is immediately discarded — this
            # method is security-equivalent to has_ref (no value leak).
            try:
                self._user_store.get_secret(name, user_id=user_id, auth_provider_type=auth_provider_type)
            except SecretNotFoundError:
                return False
            return True
        # No user-scope row — check server scope.  A server-scope row
        # with fingerprint key unset will raise FingerprintKeyMissingError
        # (propagates); allowlist / env misses raise SecretNotFoundError
        # (absorbed into False below).
        try:
            self._server_store.get_secret(name)
        except SecretNotFoundError:
            return False
        return True

    # -- REST API helpers (not part of WebSecretResolver) --------------------

    def set_user_secret(self, user_id: str, name: str, value: str, *, auth_provider_type: str) -> CreateSecretResult:
        """Create or update a user-scoped secret.

        Returns a ``CreateSecretResult`` describing the persisted state.
        Eager-fingerprint semantics (see :meth:`UserSecretStore.set_secret`)
        guarantee that a successful return implies the secret is both
        persisted AND immediately resolvable — no TOCTOU window against
        a separate ``has_ref`` probe.

        Raises
        ------
        FingerprintKeyMissingError
            Propagated from the store when ``ELSPETH_FINGERPRINT_KEY`` is
            unset.  No row is written.  HTTP handlers map to 503.
        """
        fingerprint = self._user_store.set_secret(name, value=value, user_id=user_id, auth_provider_type=auth_provider_type)
        return CreateSecretResult(
            name=name,
            scope="user",
            fingerprint=fingerprint,
        )

    def delete_user_secret(self, user_id: str, name: str, *, auth_provider_type: str) -> bool:
        """Delete a user-scoped secret. Returns True if deleted."""
        return self._user_store.delete_secret(name, user_id=user_id, auth_provider_type=auth_provider_type)


class ScopedSecretResolver:
    """Binds auth_provider_type into the WebSecretResolver protocol.

    The ``WebSecretResolver`` protocol (L0) uses ``(user_id, name)`` as
    the resolution key.  The web layer scopes user secrets by
    ``(user_id, auth_provider_type, name)``.  This adapter bridges the
    gap so ``resolve_secret_refs()`` in L1 can resolve secrets without
    knowing about auth providers — preserving layer boundaries.

    ``auth_provider_type`` comes from ``WebSettings.auth_provider`` and
    is deployment-level (the same for all requests to a given server).
    """

    def __init__(self, service: WebSecretService, auth_provider_type: str) -> None:
        self._service = service
        self._auth_provider_type = auth_provider_type

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        return self._service.list_refs(user_id, auth_provider_type=self._auth_provider_type)

    def has_ref(self, user_id: str, name: str) -> bool:
        return self._service.has_ref(user_id, name, auth_provider_type=self._auth_provider_type)

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None:
        return self._service.resolve(user_id, name, auth_provider_type=self._auth_provider_type)
