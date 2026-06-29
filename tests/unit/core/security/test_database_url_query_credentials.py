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


def test_database_url_preserves_empty_authority_sqlite_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stripping a query credential must not collapse the ``//`` of a no-host DSN.

    ``urlunparse`` drops the authority introducer for schemes outside urllib's
    ``uses_netloc`` set (sqlite, postgresql) when the netloc is empty. The
    sanitizer must preserve ``sqlite:///`` so the audited DSN keeps its shape.
    """
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "sqlite:///:memory:?password=supersecret&timeout=1"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "supersecret" not in result.sanitized_url
    assert "password=" not in result.sanitized_url
    assert result.sanitized_url == "sqlite:///:memory:?timeout=1"
    assert result.fingerprint is not None


def test_database_url_preserves_empty_authority_postgresql_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host-in-query postgresql socket DSN keeps its ``///`` after scrubbing."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "postgresql:///db?password=supersecret&host=/var/run/postgresql"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "supersecret" not in result.sanitized_url
    assert result.sanitized_url == "postgresql:///db?host=/var/run/postgresql"
    assert result.fingerprint is not None


def test_database_url_preserves_absolute_path_slash_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absolute-path sqlite DSN must keep all four slashes.

    ``sqlite:////var/lib/app.db`` is an *absolute* path; collapsing one slash to
    ``sqlite:///var/lib/app.db`` silently reinterprets it as a relative path.
    """
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "sqlite:////var/lib/app.db?password=supersecret"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "supersecret" not in result.sanitized_url
    assert result.sanitized_url == "sqlite:////var/lib/app.db"
    assert result.fingerprint is not None


def test_database_url_no_host_sanitized_output_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    """The sanitized no-host DSN is a stable, accepted ``SanitizedDatabaseUrl`` input."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "sqlite:///:memory:?password=supersecret&timeout=1"

    once = SanitizedDatabaseUrl.from_raw_url(url)
    # Re-sanitizing the output must not mutate it further (idempotent) and must
    # pass __post_init__ (no ValueError).
    twice = SanitizedDatabaseUrl.from_raw_url(once.sanitized_url)

    assert twice.sanitized_url == once.sanitized_url == "sqlite:///:memory:?timeout=1"


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
