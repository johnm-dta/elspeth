"""Tests for ServerSecretStore — env-var allowlist enforcement.

Verifies:
- Allowlist boundary: non-allowlisted names always rejected
- Env-var presence: allowlisted but unset names raise SecretNotFoundError
- get_secret returns (value, SecretRef) with correct fingerprint source
- list_secrets exposes metadata only, never values
- Empty allowlist blocks everything
"""

from __future__ import annotations

import pytest

from elspeth.contracts.secrets import FingerprintKeyMissingError, SecretInventoryItem
from elspeth.core.security.secret_loader import SecretNotFoundError, SecretRef
from elspeth.web.secrets.server_store import ServerSecretStore


@pytest.fixture()
def _fingerprint_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ELSPETH_FINGERPRINT_KEY is set for _compute_fingerprint."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fp-key")


@pytest.fixture()
def store(_fingerprint_key: None) -> ServerSecretStore:
    """Store with a two-item allowlist."""
    return ServerSecretStore(allowlist=("ALLOWED_KEY_A", "ALLOWED_KEY_B"))


@pytest.fixture()
def empty_store(_fingerprint_key: None) -> ServerSecretStore:
    """Store with an empty allowlist."""
    return ServerSecretStore(allowlist=())


class TestHasSecret:
    """Allowlist + env-var presence gate."""

    def test_allowlisted_and_set_returns_true(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "some-value")
        assert store.has_secret("ALLOWED_KEY_A") is True

    def test_allowlisted_but_unset_returns_false(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALLOWED_KEY_A", raising=False)
        assert store.has_secret("ALLOWED_KEY_A") is False

    def test_allowlisted_but_empty_returns_false(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "")
        assert store.has_secret("ALLOWED_KEY_A") is False

    def test_not_allowlisted_returns_false_even_if_set(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NOT_ALLOWED", "secret-value")
        assert store.has_secret("NOT_ALLOWED") is False

    def test_empty_allowlist_blocks_everything(self, empty_store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "value")
        assert empty_store.has_secret("ALLOWED_KEY_A") is False


class TestGetSecret:
    """Value retrieval with allowlist enforcement."""

    def test_returns_value_and_ref(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "my-secret-value")
        value, ref = store.get_secret("ALLOWED_KEY_A")
        assert value == "my-secret-value"
        assert isinstance(ref, SecretRef)
        assert ref.name == "ALLOWED_KEY_A"
        assert ref.source == "env"
        assert len(ref.fingerprint) == 64  # HMAC-SHA256 hex digest

    def test_not_allowlisted_raises(self, store: ServerSecretStore) -> None:
        with pytest.raises(SecretNotFoundError):
            store.get_secret("NOT_ALLOWED")

    def test_allowlisted_but_unset_raises(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALLOWED_KEY_B", raising=False)
        with pytest.raises(SecretNotFoundError):
            store.get_secret("ALLOWED_KEY_B")

    def test_allowlisted_but_empty_raises(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "")
        with pytest.raises(SecretNotFoundError):
            store.get_secret("ALLOWED_KEY_A")

    def test_fingerprint_is_deterministic(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "same-value")
        _, ref1 = store.get_secret("ALLOWED_KEY_A")
        _, ref2 = store.get_secret("ALLOWED_KEY_A")
        assert ref1.fingerprint == ref2.fingerprint

    def test_different_values_different_fingerprints(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "value-one")
        _, ref1 = store.get_secret("ALLOWED_KEY_A")
        monkeypatch.setenv("ALLOWED_KEY_A", "value-two")
        _, ref2 = store.get_secret("ALLOWED_KEY_A")
        assert ref1.fingerprint != ref2.fingerprint


class TestListSecrets:
    """Inventory metadata without exposing values."""

    def test_lists_all_allowlisted_names(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "val-a")
        monkeypatch.delenv("ALLOWED_KEY_B", raising=False)
        items = store.list_secrets()
        assert len(items) == 2
        by_name = {item.name: item for item in items}
        assert by_name["ALLOWED_KEY_A"].available is True
        assert by_name["ALLOWED_KEY_A"].reason is None
        assert by_name["ALLOWED_KEY_B"].available is False
        assert by_name["ALLOWED_KEY_B"].reason == "env_var_not_set"

    def test_inventory_items_have_correct_fields(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALLOWED_KEY_A", "val")
        items = store.list_secrets()
        item = next(i for i in items if i.name == "ALLOWED_KEY_A")
        assert isinstance(item, SecretInventoryItem)
        assert item.scope == "server"
        assert item.source_kind == "env"

    def test_empty_allowlist_returns_empty(self, empty_store: ServerSecretStore) -> None:
        assert empty_store.list_secrets() == []

    def test_values_never_exposed(self, store: ServerSecretStore, monkeypatch: pytest.MonkeyPatch) -> None:
        """SecretInventoryItem has no value field — verify structurally."""
        monkeypatch.setenv("ALLOWED_KEY_A", "super-secret")
        items = store.list_secrets()
        assert items  # Ensure at least one secret is listed
        assert "value" not in SecretInventoryItem.__slots__


class TestReservedSecretNames:
    """ELSPETH internal secrets must never be exposed through the server store."""

    def test_construction_rejects_fingerprint_key(self, _fingerprint_key: None) -> None:
        with pytest.raises(ValueError, match="ELSPETH internal"):
            ServerSecretStore(allowlist=("ELSPETH_FINGERPRINT_KEY",))

    def test_construction_rejects_any_elspeth_prefix(self, _fingerprint_key: None) -> None:
        with pytest.raises(ValueError, match="ELSPETH_WEB__SECRET_KEY"):
            ServerSecretStore(allowlist=("VALID_KEY", "ELSPETH_WEB__SECRET_KEY"))

    def test_construction_rejects_audit_key(self, _fingerprint_key: None) -> None:
        with pytest.raises(ValueError, match="ELSPETH_AUDIT_KEY"):
            ServerSecretStore(allowlist=("ELSPETH_AUDIT_KEY",))

    def test_construction_accepts_non_reserved(self, _fingerprint_key: None) -> None:
        store = ServerSecretStore(allowlist=("OPENAI_API_KEY",))
        assert store._allowlist == ("OPENAI_API_KEY",)

    def test_reserved_prefix_is_case_sensitive(self, _fingerprint_key: None) -> None:
        """Lowercase 'elspeth_' is NOT reserved — only uppercase ELSPETH_ is.

        This documents the intentional case-sensitivity decision.  If
        case-insensitive blocking is ever required, this test should be
        updated to reflect the new behaviour.
        """
        store = ServerSecretStore(allowlist=("elspeth_lowercase_key",))
        assert store._allowlist == ("elspeth_lowercase_key",)

    def test_has_secret_returns_false_for_reserved_even_if_in_env(
        self,
        empty_store: ServerSecretStore,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """has_secret() returns False (not raise) for reserved names.

        WebSecretService.has_ref() composes user-scope OR server-scope
        has_secret() calls — if this path raised, a reserved-name probe
        with no matching user row would propagate as a 500 through the
        /api/secrets/validate, wire-secret-ref, and pipeline-validation
        paths.  get_secret() still raises for reserved names
        (test_get_secret_rejects_reserved_name).
        """
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "secret-fp-key")
        assert empty_store.has_secret("ELSPETH_FINGERPRINT_KEY") is False

    def test_get_secret_rejects_reserved_name(self, store: ServerSecretStore) -> None:
        with pytest.raises(SecretNotFoundError):
            store.get_secret("ELSPETH_FINGERPRINT_KEY")

    def test_list_secrets_excludes_reserved(self, _fingerprint_key: None, monkeypatch: pytest.MonkeyPatch) -> None:
        store = ServerSecretStore(allowlist=("VALID_KEY",))
        monkeypatch.setenv("VALID_KEY", "val")
        items = store.list_secrets()
        names = {item.name for item in items}
        assert "ELSPETH_FINGERPRINT_KEY" not in names
        assert "VALID_KEY" in names


class TestAllowlistNameValidation:
    """Server secret names must satisfy the shared secret-name contract."""

    @pytest.mark.parametrize(
        ("name", "match"),
        [
            ("", "must not be empty"),
            ("A/B", "must match"),
            ("A" * 257, "must be <= 256"),
        ],
    )
    def test_construction_rejects_malformed_allowlist_name(
        self,
        _fingerprint_key: None,
        name: str,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ServerSecretStore(allowlist=(name,))


class TestFingerprintKeyAvailability:
    """has_secret and list_secrets must reflect fingerprint key readiness."""

    def test_has_secret_false_when_fingerprint_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """has_secret returns False even if env var is set when fingerprint key is missing."""
        monkeypatch.setenv("ALLOWED_KEY", "value")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        assert store.has_secret("ALLOWED_KEY") is False

    def test_has_secret_true_when_fingerprint_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """has_secret returns True when both env var and fingerprint key are set."""
        monkeypatch.setenv("ALLOWED_KEY", "value")
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-key")
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        assert store.has_secret("ALLOWED_KEY") is True

    def test_list_secrets_unavailable_when_fingerprint_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_secrets marks available=False with structural reason when fingerprint key is missing."""
        monkeypatch.setenv("ALLOWED_KEY", "value")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        items = store.list_secrets()
        item = next(i for i in items if i.name == "ALLOWED_KEY")
        assert item.available is False
        assert item.reason == "fingerprint_resolver_not_configured"

    def test_list_secrets_available_when_fingerprint_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_secrets marks available=True (and reason=None) when both env var and fingerprint key set."""
        monkeypatch.setenv("ALLOWED_KEY", "value")
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-key")
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        items = store.list_secrets()
        item = next(i for i in items if i.name == "ALLOWED_KEY")
        assert item.available is True
        assert item.reason is None

    def test_list_secrets_fingerprint_key_takes_precedence_over_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When BOTH the fingerprint key AND the env var are unset, the global
        deployment gap is reported — not the per-secret env-var gap.

        The precedence is deliberate: fixing the env var without first fixing
        the fingerprint key would not make the secret resolvable, so reporting
        ``env_var_not_set`` first would mislead operators into chasing the
        wrong cause.  Locking this in a test so the choice survives future
        refactors.
        """
        monkeypatch.delenv("ALLOWED_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        items = store.list_secrets()
        item = next(i for i in items if i.name == "ALLOWED_KEY")
        assert item.available is False
        assert item.reason == "fingerprint_resolver_not_configured"

    def test_list_secrets_reason_is_closed_list_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audit-hygiene: a sentinel value placed in another env var must
        never appear in any inventory item — the reason field is structural,
        never derived from candidate-secret material.

        Probes the structural property directly: if the reason were ever
        widened to a free-form string interpolating env content, this
        assertion would fail.
        """
        sentinel = "DO-NOT-LEAK-THIS-VALUE-1234"
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "fp-key")
        monkeypatch.setenv("ALLOWED_KEY_A", sentinel)
        monkeypatch.delenv("ALLOWED_KEY_B", raising=False)
        store = ServerSecretStore(allowlist=("ALLOWED_KEY_A", "ALLOWED_KEY_B"))
        items = store.list_secrets()
        for item in items:
            assert sentinel not in (item.reason or "")
            assert sentinel not in (item.source_kind or "")
            assert sentinel not in item.name

    def test_get_secret_raises_fingerprint_missing_typed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_secret surfaces FingerprintKeyMissingError (typed) when key is unset.

        Replaces the prior SecretNotFoundError carrying the
        "ELSPETH_FINGERPRINT_KEY is not set" message.  HTTP handlers
        map the typed exception to 503 so operators see actionable
        deployment guidance; the pipeline resolution path fails fast
        for the same reason.
        """
        monkeypatch.setenv("ALLOWED_KEY", "value")
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        store = ServerSecretStore(allowlist=("ALLOWED_KEY",))
        with pytest.raises(FingerprintKeyMissingError, match="ELSPETH_FINGERPRINT_KEY"):
            store.get_secret("ALLOWED_KEY")
