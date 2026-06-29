"""Regression tests for audit-safe database URL query credential handling."""

import pytest

from elspeth.contracts.url import SanitizedDatabaseUrl


def test_database_url_query_password_removed_from_audit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query-string database passwords are removed before audit storage."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "postgresql://user@host:5432/db?password=supersecret&sslmode=require"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "supersecret" not in result.sanitized_url
    assert "password=" not in result.sanitized_url
    assert result.sanitized_url == "postgresql://user@host:5432/db?sslmode=require"
    assert result.fingerprint is not None


def test_database_url_sslpassword_query_param_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """libpq sslpassword query params are credentials and must be scrubbed."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "postgresql://user@host/db?sslpassword=pkpass&sslmode=verify-full"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "pkpass" not in result.sanitized_url
    assert "sslpassword=" not in result.sanitized_url
    assert result.sanitized_url == "postgresql://user@host/db?sslmode=verify-full"
    assert result.fingerprint is not None


def test_database_url_fragment_password_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fragment credentials are removed from database URLs."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "postgresql://user@host/db?sslmode=require#password=supersecret&keep=me"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "supersecret" not in result.sanitized_url
    assert "password=" not in result.sanitized_url
    assert result.sanitized_url == "postgresql://user@host/db?sslmode=require#keep=me"
    assert result.fingerprint is not None


def test_database_url_direct_construction_rejects_query_password() -> None:
    """Direct construction rejects sensitive query parameters."""
    with pytest.raises(ValueError, match="sensitive query parameters"):
        SanitizedDatabaseUrl(
            sanitized_url="postgresql://user@host/db?password=leaked",
            fingerprint=None,
        )


def test_database_url_direct_construction_rejects_fragment_password() -> None:
    """Direct construction rejects sensitive fragment parameters."""
    with pytest.raises(ValueError, match="sensitive fragment parameters"):
        SanitizedDatabaseUrl(
            sanitized_url="postgresql://user@host/db#sslpassword=leaked",
            fingerprint=None,
        )
