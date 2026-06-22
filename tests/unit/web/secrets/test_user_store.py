"""Tests for UserSecretStore — Fernet-encrypted user secret persistence.

Verifies:
- Round-trip encrypt/decrypt integrity
- User-scoped isolation (user A cannot see user B's secrets)
- Provider-scoped isolation (same user, different auth_provider_type)
- Upsert semantics (second set overwrites first, atomic via ON CONFLICT)
- Delete lifecycle
- List returns metadata only (no values)
- SecretNotFoundError on missing secrets
- Graceful degradation when ELSPETH_FINGERPRINT_KEY is unset
"""

from __future__ import annotations

import os
import threading
from typing import Any

import pytest
import sqlalchemy as sa
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.pool import StaticPool

from elspeth.contracts.secrets import (
    FingerprintKeyMissingError,
    SecretDecryptionError,
    SecretInventoryItem,
)
from elspeth.core.security.secret_loader import SecretNotFoundError
from elspeth.web.secrets.user_store import UserSecretStore, _derive_fernet_key
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import user_secrets_table
from elspeth.web.sessions.schema import initialize_session_schema

TEST_MASTER_KEY = "test-master-key-for-encryption"


@pytest.fixture()
def db_engine():
    """In-memory SQLite engine with all session tables created."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    return engine


@pytest.fixture()
def store(db_engine) -> UserSecretStore:
    return UserSecretStore(engine=db_engine, master_key=TEST_MASTER_KEY)


@pytest.fixture(autouse=True)
def _ensure_fingerprint_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide ELSPETH_FINGERPRINT_KEY for all tests.

    _compute_fingerprint() now crashes if the key is unset (it's a
    deployment requirement). Tests that verify the missing-key path
    override this via their own monkeypatch.delenv().
    """
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-default-fp-key")


class TestUserSecretStore:
    def test_store_and_retrieve_roundtrip(self, store: UserSecretStore) -> None:
        """Set a secret then get it back — value must match."""
        store.set_secret("API_KEY", value="sk-secret-123", user_id="user-1", auth_provider_type="local")
        value, ref = store.get_secret("API_KEY", user_id="user-1", auth_provider_type="local")

        assert value == "sk-secret-123"
        assert ref.name == "API_KEY"
        assert ref.source == "user"
        assert len(ref.fingerprint) == 64

    def test_different_users_isolated(self, store: UserSecretStore) -> None:
        """User A's secret must not be visible to user B."""
        store.set_secret("API_KEY", value="alice-key", user_id="alice", auth_provider_type="local")
        store.set_secret("API_KEY", value="bob-key", user_id="bob", auth_provider_type="local")

        alice_val, _ = store.get_secret("API_KEY", user_id="alice", auth_provider_type="local")
        bob_val, _ = store.get_secret("API_KEY", user_id="bob", auth_provider_type="local")

        assert alice_val == "alice-key"
        assert bob_val == "bob-key"

    def test_upsert_updates_existing(self, store: UserSecretStore) -> None:
        """Setting the same name twice must overwrite the first value."""
        store.set_secret("TOKEN", value="old-value", user_id="user-1", auth_provider_type="local")
        store.set_secret("TOKEN", value="new-value", user_id="user-1", auth_provider_type="local")

        value, _ = store.get_secret("TOKEN", user_id="user-1", auth_provider_type="local")
        assert value == "new-value"

    def test_delete_removes_secret(self, store: UserSecretStore) -> None:
        """Delete then get must raise SecretNotFoundError."""
        store.set_secret("TEMP", value="ephemeral", user_id="user-1", auth_provider_type="local")
        result = store.delete_secret("TEMP", user_id="user-1", auth_provider_type="local")
        assert result is True

        with pytest.raises(SecretNotFoundError):
            store.get_secret("TEMP", user_id="user-1", auth_provider_type="local")

    def test_delete_nonexistent_returns_false(self, store: UserSecretStore) -> None:
        """Deleting a secret that doesn't exist returns False."""
        result = store.delete_secret("NOPE", user_id="user-1", auth_provider_type="local")
        assert result is False

    def test_list_returns_metadata_only(self, store: UserSecretStore) -> None:
        """List must return names and scope, never values."""
        store.set_secret("KEY_A", value="val-a", user_id="user-1", auth_provider_type="local")
        store.set_secret("KEY_B", value="val-b", user_id="user-1", auth_provider_type="local")

        items = store.list_secrets(user_id="user-1", auth_provider_type="local")
        assert len(items) == 2

        names = {item.name for item in items}
        assert names == {"KEY_A", "KEY_B"}

        for item in items:
            assert isinstance(item, SecretInventoryItem)
            assert item.scope == "user"
            assert item.available is True
            assert item.source_kind == "user_store"
            # SecretInventoryItem has no value field — metadata only by design

    def test_list_secrets_does_not_decrypt_each_row(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inventory listing must not perform per-secret PBKDF/decrypt work."""
        store.set_secret("KEY_A", value="val-a", user_id="user-1", auth_provider_type="local")
        store.set_secret("KEY_B", value="val-b", user_id="user-1", auth_provider_type="local")

        def fail_if_decrypted(*args: Any, **kwargs: Any) -> bool:
            raise AssertionError("list_secrets must not decrypt user secret rows")

        monkeypatch.setattr(store, "_row_is_resolvable", fail_if_decrypted)

        items = store.list_secrets(user_id="user-1", auth_provider_type="local")

        assert [item.name for item in items] == ["KEY_A", "KEY_B"]
        assert all(item.available for item in items)
        assert all(item.reason is None for item in items)

    def test_list_secrets_selects_no_secret_material_columns(self, store: UserSecretStore, db_engine) -> None:
        """Inventory listing must not read encrypted values or salts."""
        store.set_secret("KEY_A", value="val-a", user_id="user-1", auth_provider_type="local")

        statements: list[str] = []

        def capture_statement(
            _conn: Any,
            _cursor: Any,
            statement: str,
            _parameters: Any,
            _context: Any,
            _executemany: bool,
        ) -> None:
            if "user_secrets" in statement:
                statements.append(statement)

        sa.event.listen(db_engine, "before_cursor_execute", capture_statement)
        try:
            items = store.list_secrets(user_id="user-1", auth_provider_type="local")
        finally:
            sa.event.remove(db_engine, "before_cursor_execute", capture_statement)

        assert [item.name for item in items] == ["KEY_A"]
        list_query = statements[-1].lower()
        assert "encrypted_value" not in list_query
        assert "salt" not in list_query

    def test_get_nonexistent_raises(self, store: UserSecretStore) -> None:
        """Getting a secret that doesn't exist must raise SecretNotFoundError."""
        with pytest.raises(SecretNotFoundError):
            store.get_secret("MISSING", user_id="user-1", auth_provider_type="local")

    def test_has_secret_true(self, store: UserSecretStore) -> None:
        """has_secret returns True for existing, decryptable secrets."""
        store.set_secret("EXISTS", value="val", user_id="user-1", auth_provider_type="local")
        assert store.has_secret("EXISTS", user_id="user-1", auth_provider_type="local") is True

    def test_has_secret_false(self, store: UserSecretStore) -> None:
        """has_secret returns False for non-existent secrets."""
        assert store.has_secret("NOPE", user_id="user-1", auth_provider_type="local") is False

    def test_has_secret_user_scoped(self, store: UserSecretStore) -> None:
        """has_secret is user-scoped — user B cannot see user A's secret."""
        store.set_secret("SCOPED", value="val", user_id="alice", auth_provider_type="local")
        assert store.has_secret("SCOPED", user_id="alice", auth_provider_type="local") is True
        assert store.has_secret("SCOPED", user_id="bob", auth_provider_type="local") is False

    def test_fingerprint_populated_when_key_set(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_secret returns a non-empty fingerprint when ELSPETH_FINGERPRINT_KEY is set."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fp-key")
        store.set_secret("FP_TEST", value="my-secret", user_id="user-1", auth_provider_type="local")
        _, ref = store.get_secret("FP_TEST", user_id="user-1", auth_provider_type="local")
        assert len(ref.fingerprint) == 64
        assert all(c in "0123456789abcdef" for c in ref.fingerprint)

    def test_fingerprint_missing_key_raises(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_secret raises FingerprintKeyMissingError when ELSPETH_FINGERPRINT_KEY is unset.

        Typed exception (not SecretNotFoundError) so HTTP handlers can map
        deployment misconfiguration to 503 rather than treating the row as
        absent (which would be 404 / ``available=False`` territory).
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "temp-for-set")
        store.set_secret("FP_EMPTY", value="val", user_id="user-1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        with pytest.raises(FingerprintKeyMissingError, match="ELSPETH_FINGERPRINT_KEY"):
            store.get_secret("FP_EMPTY", user_id="user-1", auth_provider_type="local")

    # -- Regression: provider namespace isolation (Bug 4) --

    def test_different_providers_isolated(self, store: UserSecretStore) -> None:
        """Secrets scoped by auth_provider_type — same user_id, different providers."""
        store.set_secret("API_KEY", value="local-key", user_id="alice", auth_provider_type="local")
        store.set_secret("API_KEY", value="oidc-key", user_id="alice", auth_provider_type="oidc")

        local_val, _ = store.get_secret("API_KEY", user_id="alice", auth_provider_type="local")
        oidc_val, _ = store.get_secret("API_KEY", user_id="alice", auth_provider_type="oidc")

        assert local_val == "local-key"
        assert oidc_val == "oidc-key"

    def test_has_secret_respects_provider(self, store: UserSecretStore) -> None:
        """has_secret is provider-scoped — different providers see different secrets."""
        store.set_secret("KEY", value="val", user_id="alice", auth_provider_type="local")
        assert store.has_secret("KEY", user_id="alice", auth_provider_type="local") is True
        assert store.has_secret("KEY", user_id="alice", auth_provider_type="oidc") is False

    def test_list_secrets_filtered_by_provider(self, store: UserSecretStore) -> None:
        """list_secrets only returns secrets matching the provider."""
        store.set_secret("A", value="v", user_id="alice", auth_provider_type="local")
        store.set_secret("B", value="v", user_id="alice", auth_provider_type="oidc")
        items = store.list_secrets(user_id="alice", auth_provider_type="local")
        assert len(items) == 1
        assert items[0].name == "A"

    def test_delete_secret_provider_scoped(self, store: UserSecretStore) -> None:
        """Deleting from one provider does not affect another."""
        store.set_secret("KEY", value="v1", user_id="alice", auth_provider_type="local")
        store.set_secret("KEY", value="v2", user_id="alice", auth_provider_type="oidc")
        deleted = store.delete_secret("KEY", user_id="alice", auth_provider_type="local")
        assert deleted is True
        # OIDC copy still exists
        val, _ = store.get_secret("KEY", user_id="alice", auth_provider_type="oidc")
        assert val == "v2"

    # -- Regression: concurrent upsert must not raise IntegrityError (Bug 3) --

    def test_concurrent_set_secret_no_integrity_error(self, tmp_path) -> None:
        """Simultaneous set_secret() for same key must not raise IntegrityError.

        Uses a file-backed SQLite database instead of in-memory + StaticPool
        because the latter shares a single connection across threads and
        raises InterfaceError on concurrent access — which is a test
        infrastructure issue, not a code bug.
        """
        db_path = tmp_path / "test_concurrent.db"
        engine = create_session_engine(f"sqlite:///{db_path}")
        initialize_session_schema(engine)
        store = UserSecretStore(engine=engine, master_key=TEST_MASTER_KEY)
        errors: list[Exception] = []

        def writer(value: str) -> None:
            try:
                store.set_secret("RACE", value=value, user_id="u1", auth_provider_type="local")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(f"v{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent writes failed: {errors}"
        # One value should have won
        val, _ = store.get_secret("RACE", user_id="u1", auth_provider_type="local")
        assert val.startswith("v")

    @pytest.mark.parametrize("dialect_name", ["mysql", "mariadb"])
    def test_mysql_family_dialects_use_atomic_upsert_statements(self, dialect_name: str) -> None:
        """MySQL-family dialects must keep startup and write paths usable.

        The regression happened at construction time (`create_app()`
        instantiates `UserSecretStore` unconditionally), but correctness also
        depends on the eventual write statement being the MySQL-family atomic
        upsert form rather than SQLite/PostgreSQL's `ON CONFLICT`.
        """
        from sqlalchemy.dialects.mysql import dialect as mysql_dialect

        class _CaptureConnection:
            def __init__(self) -> None:
                self.statements: list[Any] = []

            def execute(self, stmt: Any) -> None:
                self.statements.append(stmt)

        class _CaptureBegin:
            def __init__(self, conn: _CaptureConnection) -> None:
                self._conn = conn

            def __enter__(self) -> _CaptureConnection:
                return self._conn

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        class _CaptureEngine:
            def __init__(self, dialect_name: str) -> None:
                self.dialect = type("_Dialect", (), {"name": dialect_name})()
                self.connection = _CaptureConnection()

            def begin(self) -> _CaptureBegin:
                return _CaptureBegin(self.connection)

        engine = _CaptureEngine(dialect_name)
        store = UserSecretStore(engine=engine, master_key=TEST_MASTER_KEY)

        store.set_secret("API_KEY", value="secret", user_id="user-1", auth_provider_type="local")

        assert len(engine.connection.statements) == 1
        compiled = str(engine.connection.statements[0].compile(dialect=mysql_dialect()))
        assert "ON DUPLICATE KEY UPDATE" in compiled
        assert "ON CONFLICT" not in compiled

    # -- Regression: fingerprint key availability (Bugs 1 & 2) --

    def test_has_secret_false_when_fingerprint_key_missing(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """has_secret returns False when ELSPETH_FINGERPRINT_KEY is unset."""
        store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        assert store.has_secret("KEY", user_id="u1", auth_provider_type="local") is False

    def test_list_secrets_unavailable_when_fingerprint_key_missing(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_secrets marks available=False with structural reason when fingerprint key is missing."""
        store.set_secret("KEY", value="val", user_id="u1", auth_provider_type="local")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        items = store.list_secrets(user_id="u1", auth_provider_type="local")
        assert len(items) == 1
        assert items[0].available is False
        assert items[0].reason == "fingerprint_resolver_not_configured"

    def test_rotation_makes_existing_secret_unresolvable_for_resolution(self, db_engine) -> None:
        """A master-key rotation must make explicit resolution unavailable.

        Typed ``SecretDecryptionError`` (not ``SecretNotFoundError``) so HTTP
        handlers can map to 409 Conflict with re-save guidance — the row
        exists but is in conflict with current server configuration.

        Inventory listing is intentionally metadata-only and does not decrypt
        every row, so row-level decryption failures are detected by
        ``has_secret`` / ``get_secret`` / validate rather than ``list_secrets``.
        """
        writer_store = UserSecretStore(engine=db_engine, master_key=TEST_MASTER_KEY)
        writer_store.set_secret("ROTATED", value="val", user_id="u1", auth_provider_type="local")

        rotated_store = UserSecretStore(engine=db_engine, master_key="rotated-master-key")

        assert rotated_store.has_secret("ROTATED", user_id="u1", auth_provider_type="local") is False
        items = rotated_store.list_secrets(user_id="u1", auth_provider_type="local")
        assert len(items) == 1
        assert items[0].name == "ROTATED"
        assert items[0].available is True
        assert items[0].reason is None
        with pytest.raises(SecretDecryptionError, match="cannot be decrypted"):
            rotated_store.get_secret("ROTATED", user_id="u1", auth_provider_type="local")

    def test_corrupt_ciphertext_is_not_resolvable(self, store: UserSecretStore, db_engine) -> None:
        """Ciphertext corruption must be caught by explicit resolution APIs."""
        store.set_secret("CORRUPT", value="val", user_id="u1", auth_provider_type="local")

        with db_engine.begin() as conn:
            conn.execute(
                sa.update(user_secrets_table)
                .where(
                    sa.and_(
                        user_secrets_table.c.name == "CORRUPT",
                        user_secrets_table.c.user_id == "u1",
                        user_secrets_table.c.auth_provider_type == "local",
                    )
                )
                .values(encrypted_value=b"corrupt-token")
            )

        assert store.has_secret("CORRUPT", user_id="u1", auth_provider_type="local") is False
        items = store.list_secrets(user_id="u1", auth_provider_type="local")
        assert len(items) == 1
        assert items[0].available is True
        assert items[0].reason is None
        with pytest.raises(SecretDecryptionError, match="cannot be decrypted"):
            store.get_secret("CORRUPT", user_id="u1", auth_provider_type="local")

    def test_get_secret_accepts_memoryview_binary_columns(
        self,
        store: UserSecretStore,
        db_engine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Drivers may return LargeBinary columns as memoryview rather than bytes."""
        store.set_secret("MEMORYVIEW", value="val", user_id="u1", auth_provider_type="local")

        with db_engine.connect() as conn:
            row = conn.execute(
                sa.select(user_secrets_table.c.encrypted_value, user_secrets_table.c.salt).where(
                    sa.and_(
                        user_secrets_table.c.name == "MEMORYVIEW",
                        user_secrets_table.c.user_id == "u1",
                        user_secrets_table.c.auth_provider_type == "local",
                    )
                )
            ).first()
        assert row is not None
        memoryview_row = type(
            "_MemoryviewSecretRow",
            (),
            {
                "encrypted_value": memoryview(row.encrypted_value),
                "salt": memoryview(row.salt),
            },
        )()
        monkeypatch.setattr(store, "_fetch_secret_row", lambda *args, **kwargs: memoryview_row)

        value, ref = store.get_secret("MEMORYVIEW", user_id="u1", auth_provider_type="local")

        assert value == "val"
        assert ref.name == "MEMORYVIEW"

    def test_has_secret_accepts_memoryview_binary_columns(
        self,
        store: UserSecretStore,
        db_engine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Availability checks must use the same binary normalization as get_secret."""
        store.set_secret("MEMORYVIEW", value="val", user_id="u1", auth_provider_type="local")

        with db_engine.connect() as conn:
            row = conn.execute(
                sa.select(user_secrets_table.c.encrypted_value, user_secrets_table.c.salt).where(
                    sa.and_(
                        user_secrets_table.c.name == "MEMORYVIEW",
                        user_secrets_table.c.user_id == "u1",
                        user_secrets_table.c.auth_provider_type == "local",
                    )
                )
            ).first()
        assert row is not None
        memoryview_row = type(
            "_MemoryviewSecretRow",
            (),
            {
                "encrypted_value": memoryview(row.encrypted_value),
                "salt": memoryview(row.salt),
            },
        )()
        monkeypatch.setattr(store, "_fetch_secret_row", lambda *args, **kwargs: memoryview_row)

        assert store.has_secret("MEMORYVIEW", user_id="u1", auth_provider_type="local") is True

    # -- Eager fingerprint / write-time atomicity (A+B best-practice design) --

    def test_set_secret_raises_fingerprint_missing_when_key_unset(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_secret eager-computes the fingerprint BEFORE encrypting+writing.

        When ELSPETH_FINGERPRINT_KEY is unset the write must fail with
        FingerprintKeyMissingError so HTTP handlers map to 503 rather than
        returning 201 ``available=False`` (the old silent half-success).
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")
        with pytest.raises(FingerprintKeyMissingError, match="ELSPETH_FINGERPRINT_KEY"):
            store.set_secret("EAGER", value="val", user_id="u1", auth_provider_type="local")

    def test_set_secret_atomic_no_row_left_on_fingerprint_missing(
        self,
        store: UserSecretStore,
        db_engine,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transactional guarantee: a failed set_secret must leave no DB row.

        CLAUDE.md's audit-primacy rule requires that unfingerprinted secrets
        are never persisted. Regression test: the fingerprint check must
        precede the upsert so the DB stays consistent with the audit trail.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY")

        with pytest.raises(FingerprintKeyMissingError):
            store.set_secret("ATOMIC", value="val", user_id="u1", auth_provider_type="local")

        # Direct DB check — NOT via has_secret/get_secret, which themselves
        # check the fingerprint key and would report False for unrelated
        # reasons. We want to prove the row itself does not exist.
        with db_engine.connect() as conn:
            rows = conn.execute(
                sa.select(user_secrets_table.c.name).where(
                    sa.and_(
                        user_secrets_table.c.name == "ATOMIC",
                        user_secrets_table.c.user_id == "u1",
                        user_secrets_table.c.auth_provider_type == "local",
                    )
                )
            ).fetchall()
        assert rows == [], "Fingerprint pre-check must prevent any row from being written"

    def test_set_secret_returns_fingerprint_on_success(self, store: UserSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_secret returns the computed fingerprint for audit correlation."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "eager-fp-key")
        fp = store.set_secret("RETURN_FP", value="val", user_id="u1", auth_provider_type="local")
        assert isinstance(fp, str)
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

        # Fingerprint returned by set_secret matches the one computed on read.
        _, ref = store.get_secret("RETURN_FP", user_id="u1", auth_provider_type="local")
        assert ref.fingerprint == fp

    def test_get_secret_raises_secret_decryption_error_on_rotation(self, db_engine) -> None:
        """Decryption failure surfaces as SecretDecryptionError (typed).

        Complement to test_rotation_makes_existing_secret_unresolvable —
        explicit contract that callers can catch SecretDecryptionError
        specifically (e.g., HTTP layer mapping to 409 with re-save
        guidance).
        """
        writer = UserSecretStore(engine=db_engine, master_key=TEST_MASTER_KEY)
        writer.set_secret("TYPED_ROT", value="val", user_id="u1", auth_provider_type="local")

        rotated = UserSecretStore(engine=db_engine, master_key="different-master-key")

        with pytest.raises(SecretDecryptionError, match="cannot be decrypted"):
            rotated.get_secret("TYPED_ROT", user_id="u1", auth_provider_type="local")


class TestDeriveFernetKey:
    """Direct tests for _derive_fernet_key — key derivation function."""

    def test_different_salts_produce_different_keys(self) -> None:
        """Same master key + different salts must produce different Fernet keys."""
        salt_a = b"\x00" * 16
        salt_b = b"\x01" * 16
        key_a = _derive_fernet_key("master", salt_a)
        key_b = _derive_fernet_key("master", salt_b)
        assert key_a != key_b

    def test_different_master_keys_produce_different_keys(self) -> None:
        """Different master keys + same salt must produce different Fernet keys."""
        salt = b"\x00" * 16
        key_a = _derive_fernet_key("master-a", salt)
        key_b = _derive_fernet_key("master-b", salt)
        assert key_a != key_b

    def test_wrong_key_cannot_decrypt(self) -> None:
        """Data encrypted with one derived key cannot be decrypted with another."""
        salt = os.urandom(16)
        key_correct = _derive_fernet_key("correct-master", salt)
        key_wrong = _derive_fernet_key("wrong-master", salt)

        ciphertext = Fernet(key_correct).encrypt(b"sensitive data")
        with pytest.raises(InvalidToken):
            Fernet(key_wrong).decrypt(ciphertext)

    def test_deterministic_for_same_inputs(self) -> None:
        """Same master key + same salt must produce the same Fernet key."""
        salt = b"\xab\xcd" * 8
        key_a = _derive_fernet_key("my-master", salt)
        key_b = _derive_fernet_key("my-master", salt)
        assert key_a == key_b
