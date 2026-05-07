"""Tests for secret contract types — security invariants."""

from __future__ import annotations

import pytest

from elspeth.contracts.secrets import CreateSecretResult, ResolvedSecret, SecretInventoryItem

_VALID_FINGERPRINT = "a" * 64


class TestResolvedSecret:
    def test_repr_does_not_contain_value(self):
        """SECURITY: __repr__ must never expose plaintext."""
        rs = ResolvedSecret(name="API_KEY", value="sk-secret-123", scope="user", fingerprint=_VALID_FINGERPRINT)
        repr_str = repr(rs)
        assert "sk-secret-123" not in repr_str
        assert "API_KEY" in repr_str

    def test_str_does_not_contain_value(self):
        rs = ResolvedSecret(name="API_KEY", value="sk-secret-123", scope="user", fingerprint=_VALID_FINGERPRINT)
        assert "sk-secret-123" not in str(rs)

    def test_fields_accessible(self):
        rs = ResolvedSecret(name="KEY", value="val", scope="server", fingerprint=_VALID_FINGERPRINT)
        assert rs.name == "KEY"
        assert rs.value == "val"
        assert rs.scope == "server"
        assert rs.fingerprint == _VALID_FINGERPRINT

    def test_frozen(self):
        rs = ResolvedSecret(name="KEY", value="val", scope="server", fingerprint=_VALID_FINGERPRINT)
        with pytest.raises(AttributeError):
            rs.value = "new"

    def test_invalid_scope_rejected(self) -> None:
        with pytest.raises(ValueError, match="scope must be one of"):
            ResolvedSecret(
                name="KEY",
                value="val",
                scope="bogus",  # type: ignore[arg-type]
                fingerprint=_VALID_FINGERPRINT,
            )

    @pytest.mark.parametrize("fingerprint", ["abc123", "A" * 64, "g" * 64])
    def test_invalid_fingerprint_rejected(self, fingerprint: str) -> None:
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            ResolvedSecret(name="KEY", value="val", scope="server", fingerprint=fingerprint)


class TestCreateSecretResult:
    def test_valid_construction(self) -> None:
        result = CreateSecretResult(name="KEY", scope="org", fingerprint=_VALID_FINGERPRINT)
        assert result.scope == "org"
        assert result.fingerprint == _VALID_FINGERPRINT

    def test_invalid_scope_rejected(self) -> None:
        with pytest.raises(ValueError, match="scope must be one of"):
            CreateSecretResult(
                name="KEY",
                scope="bogus",  # type: ignore[arg-type]
                fingerprint=_VALID_FINGERPRINT,
            )

    @pytest.mark.parametrize("fingerprint", ["nothex", "A" * 64, "g" * 64])
    def test_invalid_fingerprint_rejected(self, fingerprint: str) -> None:
        with pytest.raises(ValueError, match="64-char lowercase hex"):
            CreateSecretResult(name="KEY", scope="user", fingerprint=fingerprint)


class TestSecretInventoryItem:
    def test_no_value_field(self):
        """Inventory items must not carry secret values."""
        item = SecretInventoryItem(name="KEY", scope="user", available=True)
        assert not hasattr(item, "value")

    def test_available_item_has_no_reason(self) -> None:
        item = SecretInventoryItem(name="KEY", scope="user", available=True)
        assert item.reason is None

    def test_unavailable_item_carries_reason(self) -> None:
        item = SecretInventoryItem(
            name="KEY",
            scope="server",
            available=False,
            source_kind="env",
            reason="env_var_not_set",
        )
        assert item.name == "KEY"
        assert item.scope == "server"
        assert item.available is False
        assert item.source_kind == "env"
        assert item.reason == "env_var_not_set"

    def test_invalid_scope_rejected(self) -> None:
        with pytest.raises(ValueError, match="scope must be one of"):
            SecretInventoryItem(name="KEY", scope="bogus", available=True)  # type: ignore[arg-type]

    def test_available_with_reason_rejected(self) -> None:
        """An available secret with a reason is incoherent."""
        with pytest.raises(ValueError, match="reason must be None when available=True"):
            SecretInventoryItem(
                name="KEY",
                scope="user",
                available=True,
                reason="env_var_not_set",
            )

    def test_unavailable_without_reason_rejected(self) -> None:
        """The shape this field exists to eliminate: false-with-no-explanation."""
        with pytest.raises(ValueError, match="reason is required when available=False"):
            SecretInventoryItem(name="KEY", scope="server", available=False, source_kind="env")

    def test_unknown_reason_rejected(self) -> None:
        """The Literal contract is enforced at construction time."""
        with pytest.raises(ValueError, match="reason must be one of"):
            SecretInventoryItem(
                name="KEY",
                scope="server",
                available=False,
                reason="something_else",  # type: ignore[arg-type]
            )
