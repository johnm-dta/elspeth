"""Tests for WebSecretService -- chained user -> server resolution."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest
import sqlalchemy as sa

from elspeth.contracts.secrets import (
    CreateSecretResult,
    FingerprintKeyMissingError,
    SecretDecryptionError,
)
from elspeth.core.security.secret_loader import SecretNotFoundError
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import WebSecretService
from elspeth.web.secrets.user_store import UserSecretStore
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture(autouse=True)
def _ensure_fingerprint_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide ELSPETH_FINGERPRINT_KEY — required by _compute_fingerprint()."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-service-fp-key")


@pytest.fixture()
def engine() -> sa.engine.Engine:
    """In-memory SQLite engine migrated to head."""
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


@pytest.fixture()
def user_store(engine: sa.engine.Engine) -> UserSecretStore:
    """UserSecretStore backed by in-memory SQLite."""
    return UserSecretStore(engine=engine, master_key="test-master-key-32chars-minimum!")


@pytest.fixture()
def server_store(monkeypatch: pytest.MonkeyPatch) -> ServerSecretStore:
    """ServerSecretStore with TEST_KEY in allowlist."""
    monkeypatch.setenv("TEST_KEY", "env-value")
    return ServerSecretStore(allowlist=("TEST_KEY",))


@pytest.fixture()
def service(user_store: UserSecretStore, server_store: ServerSecretStore) -> WebSecretService:
    return WebSecretService(user_store=user_store, server_store=server_store)


class TestResolve:
    def test_user_secret_takes_priority_over_server(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the same secret exists in both user and server stores, user wins."""
        monkeypatch.setenv("TEST_KEY", "server-value")
        user_store.set_secret("TEST_KEY", value="user-value", user_id="user-1", auth_provider_type="local")

        result = service.resolve("user-1", "TEST_KEY", auth_provider_type="local")

        assert result is not None
        assert result.value == "user-value"
        assert result.scope == "user"

    def test_server_fallback_when_user_missing(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When user has no secret, falls back to server store."""
        monkeypatch.setenv("TEST_KEY", "env-value")

        result = service.resolve("user-1", "TEST_KEY", auth_provider_type="local")

        assert result is not None
        assert result.value == "env-value"
        assert result.scope == "server"

    def test_resolve_returns_none_when_missing(self, service: WebSecretService) -> None:
        """Returns None when secret is not in either store."""
        result = service.resolve("user-1", "NONEXISTENT", auth_provider_type="local")

        assert result is None

    def test_has_ref_checks_both_scopes(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """has_ref returns True for secrets in either scope."""
        monkeypatch.setenv("TEST_KEY", "env-value")
        user_store.set_secret("USER_ONLY", value="val", user_id="user-1", auth_provider_type="local")

        # Server scope
        assert service.has_ref("user-1", "TEST_KEY", auth_provider_type="local") is True
        # User scope
        assert service.has_ref("user-1", "USER_ONLY", auth_provider_type="local") is True
        # Neither
        assert service.has_ref("user-1", "NONEXISTENT", auth_provider_type="local") is False


class TestListRefs:
    def test_list_refs_merges_scopes(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """list_refs returns items from both user and server stores."""
        monkeypatch.setenv("TEST_KEY", "env-value")
        user_store.set_secret("MY_SECRET", value="user-val", user_id="user-1", auth_provider_type="local")

        refs = service.list_refs("user-1", auth_provider_type="local")
        names = [r.name for r in refs]

        assert "TEST_KEY" in names
        assert "MY_SECRET" in names

    def test_list_refs_deduplicates_by_name(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When same name exists in both scopes, user scope wins in listing."""
        monkeypatch.setenv("TEST_KEY", "env-value")
        user_store.set_secret("TEST_KEY", value="user-value", user_id="user-1", auth_provider_type="local")

        refs = service.list_refs("user-1", auth_provider_type="local")
        test_key_items = [r for r in refs if r.name == "TEST_KEY"]

        assert len(test_key_items) == 1
        assert test_key_items[0].scope == "user"


class TestCrud:
    def test_set_and_delete_user_secret(
        self,
        service: WebSecretService,
    ) -> None:
        """CRUD pass-through to user store works correctly."""
        # Set
        service.set_user_secret("user-1", "MY_KEY", "my-value", auth_provider_type="local")

        result = service.resolve("user-1", "MY_KEY", auth_provider_type="local")
        assert result is not None
        assert result.value == "my-value"
        assert result.scope == "user"

        # Delete
        deleted = service.delete_user_secret("user-1", "MY_KEY", auth_provider_type="local")
        assert deleted is True

        # Gone
        result = service.resolve("user-1", "MY_KEY", auth_provider_type="local")
        assert result is None

        # Double delete
        deleted = service.delete_user_secret("user-1", "MY_KEY", auth_provider_type="local")
        assert deleted is False

    def test_set_user_secret_returns_create_result(self, service: WebSecretService) -> None:
        """set_user_secret returns a CreateSecretResult with fingerprint only.

        Eager-fingerprint design: if this call returns (instead of raising
        FingerprintKeyMissingError), the secret is persisted AND immediately
        resolvable — no TOCTOU window against a subsequent has_ref probe.
        """
        result = service.set_user_secret("user-1", "FRESH", "my-value", auth_provider_type="local")

        assert isinstance(result, CreateSecretResult)
        assert result.name == "FRESH"
        assert result.scope == "user"
        assert not hasattr(result, "available")
        assert len(result.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in result.fingerprint)

    def test_set_user_secret_raises_fingerprint_missing(self, service: WebSecretService, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_user_secret propagates FingerprintKeyMissingError from the store.

        HTTP handlers map this to 503 so API consumers see deployment
        guidance instead of a generic 500.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        with pytest.raises(FingerprintKeyMissingError):
            service.set_user_secret("user-1", "FP_GONE", "val", auth_provider_type="local")


class TestCheckUserRefResolvable:
    """check_user_ref_resolvable(): typed-error variant of has_ref for HTTP.

    ``resolve()`` must keep swallowing typed errors so the pipeline-path
    aggregation invariant (``TestHasRefResolveInvariant``) is preserved —
    all misses bucket into a single ``SecretResolutionError``.

    The HTTP ``/api/secrets/{name}/validate`` endpoint needs the opposite:
    surface fingerprint-missing as 503 and decryption-failure as 409 so
    API consumers get actionable status codes.  ``check_user_ref_resolvable``
    is the adapter that provides that separate contract.
    """

    def test_returns_true_for_resolvable_user_secret(self, service: WebSecretService, user_store: UserSecretStore) -> None:
        user_store.set_secret("RESOLVABLE", value="v", user_id="u1", auth_provider_type="local")
        assert service.check_user_ref_resolvable("u1", "RESOLVABLE", auth_provider_type="local") is True

    def test_returns_false_for_absent_ref(self, service: WebSecretService) -> None:
        assert service.check_user_ref_resolvable("u1", "ABSENT", auth_provider_type="local") is False

    def test_returns_true_for_resolvable_server_secret(self, service: WebSecretService, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_KEY", "env-value")
        assert service.check_user_ref_resolvable("u1", "TEST_KEY", auth_provider_type="local") is True

    def test_raises_fingerprint_missing_for_user_scope(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """User-scope row + fingerprint key unset → typed error propagates."""
        user_store.set_secret("FP_USR", value="v", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        with pytest.raises(FingerprintKeyMissingError):
            service.check_user_ref_resolvable("u1", "FP_USR", auth_provider_type="local")

    def test_raises_fingerprint_missing_for_server_scope(self, service: WebSecretService, monkeypatch: pytest.MonkeyPatch) -> None:
        """Server-scope path also surfaces FingerprintKeyMissingError."""
        monkeypatch.setenv("TEST_KEY", "val")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        with pytest.raises(FingerprintKeyMissingError):
            service.check_user_ref_resolvable("u1", "TEST_KEY", auth_provider_type="local")

    def test_raises_decryption_error_on_key_rotation(
        self,
        user_store: UserSecretStore,
        engine: sa.engine.Engine,
    ) -> None:
        """Key rotation surfaces as typed SecretDecryptionError (→ HTTP 409)."""
        user_store.set_secret("ROT", value="v", user_id="u1", auth_provider_type="local")
        rotated = UserSecretStore(engine=engine, master_key="rotated-master-key")
        rotated_service = WebSecretService(user_store=rotated, server_store=ServerSecretStore(()))

        with pytest.raises(SecretDecryptionError):
            rotated_service.check_user_ref_resolvable("u1", "ROT", auth_provider_type="local")

    def test_resolve_still_returns_none_for_absent_row(self, service: WebSecretService) -> None:
        """Contract preserved: resolve() still returns None for genuinely missing refs.

        This is the companion assertion — resolve() must NOT learn the
        typed-error behaviour, because that would break the pipeline-path
        aggregation invariant in TestHasRefResolveInvariant.
        """
        assert service.resolve("u1", "NEVER_EXISTED", auth_provider_type="local") is None

    def test_returns_false_when_user_secret_deleted_between_probe_and_fetch(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """Delete race is absence, not an exception, on the typed validate path."""
        user_store.set_secret("RACE", value="v", user_id="u1", auth_provider_type="local")

        original_probe = user_store.has_secret_record

        def has_record_then_delete(*args: object, **kwargs: object) -> bool:
            result = original_probe(*args, **kwargs)  # type: ignore[arg-type]
            user_store.delete_secret("RACE", user_id="u1", auth_provider_type="local")
            return result

        with patch.object(user_store, "has_secret_record", side_effect=has_record_then_delete):
            assert service.check_user_ref_resolvable("u1", "RACE", auth_provider_type="local") is False


class TestServerStore:
    def test_allowlist_restricts_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ServerSecretStore only exposes allowlisted env vars."""
        monkeypatch.setenv("ALLOWED", "yes")
        monkeypatch.setenv("FORBIDDEN", "no")
        store = ServerSecretStore(allowlist=("ALLOWED",))

        value, ref = store.get_secret("ALLOWED")
        assert value == "yes"
        assert ref.source == "env"

        with pytest.raises(SecretNotFoundError):
            store.get_secret("FORBIDDEN")

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Allowlisted but unset env var raises SecretNotFoundError."""
        monkeypatch.delenv("MISSING_KEY", raising=False)
        store = ServerSecretStore(allowlist=("MISSING_KEY",))

        with pytest.raises(SecretNotFoundError):
            store.get_secret("MISSING_KEY")

    def test_list_secrets_shows_availability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_secrets shows which allowlisted vars are set vs missing."""
        monkeypatch.setenv("SET_KEY", "val")
        monkeypatch.delenv("UNSET_KEY", raising=False)
        store = ServerSecretStore(allowlist=("SET_KEY", "UNSET_KEY"))

        items = store.list_secrets()
        by_name = {i.name: i for i in items}

        assert by_name["SET_KEY"].available is True
        assert by_name["UNSET_KEY"].available is False
        assert by_name["SET_KEY"].scope == "server"
        assert by_name["SET_KEY"].source_kind == "env"


class TestScopedSecretResolver:
    """ScopedSecretResolver binds auth_provider_type for the protocol."""

    def test_satisfies_protocol(
        self,
        service: WebSecretService,
    ) -> None:
        """ScopedSecretResolver must satisfy WebSecretResolver protocol."""
        from elspeth.contracts.secrets import WebSecretResolver
        from elspeth.web.secrets.service import ScopedSecretResolver

        resolver = ScopedSecretResolver(service, auth_provider_type="local")
        assert isinstance(resolver, WebSecretResolver)

    def test_binds_auth_provider_for_has_ref(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """has_ref through scoped resolver uses the bound auth_provider_type."""
        from elspeth.web.secrets.service import ScopedSecretResolver

        user_store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="oidc")
        oidc_resolver = ScopedSecretResolver(service, auth_provider_type="oidc")
        local_resolver = ScopedSecretResolver(service, auth_provider_type="local")

        assert oidc_resolver.has_ref("u1", "KEY") is True
        assert local_resolver.has_ref("u1", "KEY") is False

    def test_binds_auth_provider_for_resolve(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """resolve through scoped resolver uses the bound auth_provider_type."""
        from elspeth.web.secrets.service import ScopedSecretResolver

        user_store.set_secret("KEY", value="secret-val", user_id="u1", auth_provider_type="oidc")
        oidc_resolver = ScopedSecretResolver(service, auth_provider_type="oidc")
        local_resolver = ScopedSecretResolver(service, auth_provider_type="local")

        result = oidc_resolver.resolve("u1", "KEY")
        assert result is not None
        assert result.value == "secret-val"

        assert local_resolver.resolve("u1", "KEY") is None

    def test_binds_auth_provider_for_list_refs(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """list_refs through scoped resolver uses the bound auth_provider_type."""
        from elspeth.web.secrets.service import ScopedSecretResolver

        user_store.set_secret("A", value="v", user_id="u1", auth_provider_type="local")
        user_store.set_secret("B", value="v", user_id="u1", auth_provider_type="oidc")
        local_resolver = ScopedSecretResolver(service, auth_provider_type="local")

        refs = local_resolver.list_refs("u1")
        user_names = [r.name for r in refs if r.scope == "user"]
        assert "A" in user_names
        assert "B" not in user_names


class TestHasRefResolveInvariant:
    """Contract invariant: has_ref()==True implies resolve()!=None.

    This is the load-bearing contract of the availability/resolvability
    alignment fix.  If has_ref reports available but resolve fails, the
    pipeline will pass validation then fail at execution time.
    """

    def test_user_scope_has_ref_implies_resolve(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """has_ref==True for a user secret must mean resolve returns a result."""
        user_store.set_secret("CONTRACT", value="val", user_id="u1", auth_provider_type="local")
        assert service.has_ref("u1", "CONTRACT", auth_provider_type="local") is True
        result = service.resolve("u1", "CONTRACT", auth_provider_type="local")
        assert result is not None
        assert result.value == "val"

    def test_server_scope_has_ref_implies_resolve(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """has_ref==True for a server secret must mean resolve returns a result."""
        monkeypatch.setenv("TEST_KEY", "server-val")
        assert service.has_ref("u1", "TEST_KEY", auth_provider_type="local") is True
        result = service.resolve("u1", "TEST_KEY", auth_provider_type="local")
        assert result is not None
        assert result.value == "server-val"

    def test_missing_fingerprint_key_breaks_both(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When fingerprint key is missing, has_ref must return False (not True with resolve failing)."""
        user_store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        # has_ref must be False — secret exists but is not resolvable
        assert service.has_ref("u1", "KEY", auth_provider_type="local") is False

    def test_has_ref_false_implies_resolve_none_or_raises(
        self,
        service: WebSecretService,
    ) -> None:
        """has_ref==False for a missing secret must mean resolve returns None."""
        assert service.has_ref("u1", "NONEXISTENT", auth_provider_type="local") is False
        result = service.resolve("u1", "NONEXISTENT", auth_provider_type="local")
        assert result is None

    def test_resolve_returns_none_when_user_secret_exists_but_fingerprint_key_missing(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """resolve() must return None (not crash) when fingerprint key is missing.

        Regression: get_secret() used to raise RuntimeError via
        _compute_fingerprint() — resolve() only caught SecretNotFoundError,
        so the RuntimeError propagated uncaught through the pipeline.
        """
        user_store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")

        assert service.has_ref("u1", "KEY", auth_provider_type="local") is False
        result = service.resolve("u1", "KEY", auth_provider_type="local")
        assert result is None

    def test_resolve_returns_none_when_server_secret_exists_but_fingerprint_key_missing(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Server-scope resolve() must also return None when fingerprint key is missing."""
        monkeypatch.setenv("TEST_KEY", "env-value")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")

        assert service.has_ref("u1", "TEST_KEY", auth_provider_type="local") is False
        result = service.resolve("u1", "TEST_KEY", auth_provider_type="local")
        assert result is None

    def test_resolve_re_emits_after_rate_limit_window_when_fingerprint_key_missing(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A missing fingerprint key must re-emit after the cooldown window.

        The resolve path still cannot raise, but operators tailing logs
        after the first failure need a periodic breadcrumb that the
        deployment is still unhealthy.
        """
        from elspeth.web.secrets import service as service_module

        user_store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        monkeypatch.setattr(service_module, "_fingerprint_missing_last_logged_at", None)
        monkeypatch.setattr(service_module, "_fingerprint_missing_suppressed", 0)
        monotonic_values = iter((100.0, 100.0, 161.0))
        monkeypatch.setattr(service_module, "_rate_limit_monotonic", lambda: next(monotonic_values))

        with patch.object(service_module, "_slog") as mock_slog:
            service.resolve("u1", "KEY", auth_provider_type="local")
            service.resolve("u1", "KEY", auth_provider_type="local")
            service.resolve("u1", "KEY", auth_provider_type="local")

        assert mock_slog.error.call_count == 2
        args, kwargs = mock_slog.error.call_args_list[0]
        assert args[0] == "secret_resolve_fingerprint_key_missing"
        assert "ELSPETH_FINGERPRINT_KEY" in kwargs["detail"]
        assert kwargs["suppressed_since_last_emit"] == 0

        args, kwargs = mock_slog.error.call_args_list[1]
        assert args[0] == "secret_resolve_fingerprint_key_missing"
        assert kwargs["suppressed_since_last_emit"] == 1

    def test_resolve_suppresses_repeated_logs_within_rate_limit_window(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated misses inside the cooldown window must not flood logs."""
        from elspeth.web.secrets import service as service_module

        user_store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        monkeypatch.setattr(service_module, "_fingerprint_missing_last_logged_at", None)
        monkeypatch.setattr(service_module, "_fingerprint_missing_suppressed", 0)
        monkeypatch.setattr(service_module, "_rate_limit_monotonic", lambda: 100.0)

        with patch.object(service_module, "_slog") as mock_slog:
            service.resolve("u1", "KEY", auth_provider_type="local")
            service.resolve("u1", "KEY", auth_provider_type="local")
            service.resolve("u1", "OTHER", auth_provider_type="local")

        assert mock_slog.error.call_count == 1
        assert service_module._fingerprint_missing_suppressed == 2

    def test_resolve_emits_breadcrumb_only_for_fingerprint_missing_not_generic_miss(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A plain SecretNotFoundError on resolve must NOT fire the
        fingerprint-key breadcrumb.  The breadcrumb is specific to a
        deployment-error class; confusing it with a genuine cache miss
        would cause operators to chase ghost misconfigurations every
        time a pipeline references an unknown secret.
        """
        from elspeth.web.secrets import service as service_module

        # Fingerprint key IS set (the autouse fixture provides it).  The
        # lookup just misses — this is a normal validation condition.
        monkeypatch.setattr(service_module, "_fingerprint_missing_last_logged_at", None)
        monkeypatch.setattr(service_module, "_fingerprint_missing_suppressed", 0)

        with patch.object(service_module, "_slog") as mock_slog:
            result = service.resolve("u1", "NONEXISTENT", auth_provider_type="local")

        assert result is None
        assert mock_slog.error.call_count == 0

    def test_resolve_re_emits_after_rate_limit_window_when_secret_cannot_be_decrypted(
        self,
        engine: sa.engine.Engine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A swallowed SecretDecryptionError must emit a periodic breadcrumb."""
        from elspeth.web.secrets import service as service_module

        writer_store = UserSecretStore(engine=engine, master_key="test-master-key-32chars-minimum!")
        writer_store.set_secret("ROT", value="user-val", user_id="u1", auth_provider_type="local")

        rotated_user_store = UserSecretStore(engine=engine, master_key="rotated-master-key")
        rotated_service = WebSecretService(
            user_store=rotated_user_store,
            server_store=ServerSecretStore(()),
        )

        monkeypatch.setattr(service_module, "_secret_decryption_last_logged_at", None, raising=False)
        monkeypatch.setattr(service_module, "_secret_decryption_suppressed", 0, raising=False)
        monotonic_values = iter((100.0, 100.0, 161.0))
        monkeypatch.setattr(service_module, "_rate_limit_monotonic", lambda: next(monotonic_values))

        with patch.object(service_module, "_slog") as mock_slog:
            assert rotated_service.resolve("u1", "ROT", auth_provider_type="local") is None
            assert rotated_service.resolve("u1", "ROT", auth_provider_type="local") is None
            assert rotated_service.resolve("u1", "ROT", auth_provider_type="local") is None

        assert mock_slog.error.call_count == 2
        args, kwargs = mock_slog.error.call_args_list[0]
        assert args[0] == "secret_resolve_decryption_failed"
        assert "cannot be decrypted" in kwargs["detail"]
        assert kwargs["suppressed_since_last_emit"] == 0

        args, kwargs = mock_slog.error.call_args_list[1]
        assert args[0] == "secret_resolve_decryption_failed"
        assert kwargs["suppressed_since_last_emit"] == 1

    def test_resolve_emits_decryption_breadcrumb_only_for_secret_decryption_error(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A plain SecretNotFoundError must not emit the decryption breadcrumb."""
        from elspeth.web.secrets import service as service_module

        monkeypatch.setattr(service_module, "_secret_decryption_last_logged_at", None, raising=False)
        monkeypatch.setattr(service_module, "_secret_decryption_suppressed", 0, raising=False)

        with patch.object(service_module, "_slog") as mock_slog:
            result = service.resolve("u1", "NONEXISTENT", auth_provider_type="local")

        assert result is None
        assert mock_slog.error.call_count == 0

    def test_fingerprint_missing_breadcrumb_counts_all_suppressed_calls_under_concurrency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Concurrent suppressed calls must not lose count updates."""
        import time as stdlib_time

        from elspeth.web.secrets import service as service_module

        class _SlowCounter(int):
            def __new__(cls, value: int, delay_seconds: float):
                obj = int.__new__(cls, value)
                obj._delay_seconds = delay_seconds
                return obj

            def __add__(self, other: int) -> int:
                stdlib_time.sleep(self._delay_seconds)
                return int(self) + other

        monkeypatch.setattr(service_module, "_fingerprint_missing_last_logged_at", 100.0)
        monkeypatch.setattr(service_module, "_fingerprint_missing_suppressed", _SlowCounter(0, 0.05))
        monkeypatch.setattr(service_module, "_rate_limit_monotonic", lambda: 100.0)

        start = threading.Barrier(6)

        def _worker() -> None:
            start.wait()
            service_module._log_fingerprint_missing_rate_limited()

        with patch.object(service_module, "_slog") as mock_slog:
            threads = [threading.Thread(target=_worker) for _ in range(5)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join()

        assert mock_slog.error.call_count == 0
        assert service_module._fingerprint_missing_suppressed == 5

    def test_has_ref_returns_false_for_reserved_name_when_user_has_no_row(
        self,
        service: WebSecretService,
    ) -> None:
        """Probing a reserved (ELSPETH_*) name that the user has not stored
        must return False, not propagate SecretNotFoundError.

        Regression: the server store's reserved-name guard used to raise
        inside has_secret(), which turned has_ref()'s boolean composition
        (user OR server) into an uncaught 500 whenever user-scope returned
        False for a reserved name.  The boolean contract is load-bearing
        for /api/secrets/validate, composer wire-secret-ref, and pipeline
        validation's missing_refs walk.
        """
        assert service.has_ref("u1", "ELSPETH_FOO", auth_provider_type="local") is False
        assert service.resolve("u1", "ELSPETH_FOO", auth_provider_type="local") is None

    def test_has_ref_returns_true_when_user_has_reserved_name_row(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """User-scope stores have no reserved-prefix guard — a user may
        store 'ELSPETH_FOO' as a user secret.  When they do, has_ref must
        return True (user-scope short-circuits the OR before reaching the
        server store's reserved-name branch).
        """
        user_store.set_secret("ELSPETH_FOO", value="user-val", user_id="u1", auth_provider_type="local")
        assert service.has_ref("u1", "ELSPETH_FOO", auth_provider_type="local") is True
        result = service.resolve("u1", "ELSPETH_FOO", auth_provider_type="local")
        assert result is not None
        assert result.value == "user-val"
        assert result.scope == "user"

    def test_has_ref_reserved_name_invariant_with_missing_fingerprint_key(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reviewer's specific scenario: reserved-name probe with
        ELSPETH_FINGERPRINT_KEY unset.

        user_store.has_secret returns False early (no fingerprint key),
        the OR falls through to server_store.has_secret, which must
        return False (not raise) for the reserved name.  resolve() also
        returns None under both conditions.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        assert service.has_ref("u1", "ELSPETH_FINGERPRINT_KEY", auth_provider_type="local") is False
        assert service.resolve("u1", "ELSPETH_FINGERPRINT_KEY", auth_provider_type="local") is None

    def test_has_ref_reserved_name_with_user_row_but_fingerprint_unset(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Closes the 3x2 matrix corner: user *has* a reserved-name row
        stored, but ELSPETH_FINGERPRINT_KEY becomes unset at probe time.

        user_store.has_secret's fingerprint-key gate fires first
        (returns False regardless of the stored row — audit trail is
        broken without the key).  The OR must then fall through to
        server_store.has_secret, which must return False (not raise)
        for the reserved name.  This locks in that the fingerprint gate
        is independent of the server store's reserved-name branch —
        neither is allowed to shortcut the other into an exception.
        """
        user_store.set_secret(
            "ELSPETH_FOO",
            value="user-val",
            user_id="u1",
            auth_provider_type="local",
        )
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")

        assert service.has_ref("u1", "ELSPETH_FOO", auth_provider_type="local") is False
        assert service.resolve("u1", "ELSPETH_FOO", auth_provider_type="local") is None

    def test_resolve_returns_none_when_user_secret_deleted_between_has_and_get(
        self,
        service: WebSecretService,
        user_store: UserSecretStore,
    ) -> None:
        """Regression for the TOCTOU race in WebSecretService.resolve.

        The refactored resolve() opens two independent reads on the user
        path (``has_secret_record`` then ``get_secret``).  A concurrent
        DELETE landing between the two makes ``get_secret`` raise
        ``SecretNotFoundError`` — resolve() must absorb it into ``None``
        so the pipeline-validation ``missing_refs`` walk can aggregate
        the miss instead of the walk itself 500-ing.

        The race window is narrower than the pre-refactor design (which
        called ``has_secret`` first, an extra read that participated in
        the race); this test still pins the "race → None" contract.
        """
        user_store.set_secret("RACE", value="v", user_id="u1", auth_provider_type="local")

        original_probe = user_store.has_secret_record

        def has_record_then_delete(*args: object, **kwargs: object) -> bool:
            result = original_probe(*args, **kwargs)  # type: ignore[arg-type]
            # Simulate a concurrent DELETE /api/secrets/{name} landing
            # inside the TOCTOU window between has_secret_record (which
            # observed the row) and get_secret (which will not).
            user_store.delete_secret("RACE", user_id="u1", auth_provider_type="local")
            return result

        with patch.object(user_store, "has_secret_record", side_effect=has_record_then_delete):
            result = service.resolve("u1", "RACE", auth_provider_type="local")

        assert result is None

    def test_resolve_returns_none_when_server_env_cleared_mid_resolve(
        self,
        service: WebSecretService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Server-scope env clear must still absorb into None.

        The refactored resolve() calls ``server_store.get_secret``
        directly with no preceding ``has_secret`` probe, so the prior
        LBYL double-read race window is gone.  What remains is the
        simpler invariant: a server secret unavailable at the moment
        ``get_secret`` runs resolves to None rather than propagating a
        500.  Exercised by un-setting the env var BEFORE the call.
        """
        monkeypatch.setenv("TEST_KEY", "server-val")

        # Verify resolve succeeds when env is set.
        assert service.resolve("u1", "TEST_KEY", auth_provider_type="local") is not None

        # Clear env — matches what an "env cleared between reads" race
        # would leave behind by the time the next resolve call runs.
        monkeypatch.delenv("TEST_KEY", raising=False)
        assert service.resolve("u1", "TEST_KEY", auth_provider_type="local") is None

    def test_unresolvable_user_secret_shadows_server_secret_of_same_name(
        self,
        engine: sa.engine.Engine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A broken user-scoped row must not fall through to a server-scoped secret.

        list_refs() already lets user scope win on name clashes, so has_ref()
        and resolve() must keep the same shadowing rule when the user copy
        becomes undecryptable after a web secret_key rotation.
        """
        monkeypatch.setenv("TEST_KEY", "server-val")
        writer_store = UserSecretStore(engine=engine, master_key="test-master-key-32chars-minimum!")
        writer_store.set_secret("TEST_KEY", value="user-val", user_id="u1", auth_provider_type="local")

        rotated_user_store = UserSecretStore(engine=engine, master_key="rotated-master-key")
        service = WebSecretService(
            user_store=rotated_user_store,
            server_store=ServerSecretStore(allowlist=("TEST_KEY",)),
        )

        refs = service.list_refs("u1", auth_provider_type="local")
        [item] = [ref for ref in refs if ref.name == "TEST_KEY"]
        assert item.scope == "user"
        assert item.available is False
        assert service.has_ref("u1", "TEST_KEY", auth_provider_type="local") is False
        assert service.resolve("u1", "TEST_KEY", auth_provider_type="local") is None
