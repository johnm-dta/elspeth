"""Regression tests for audit-safe database URL query credential handling."""

from urllib.parse import parse_qs, urlparse

import pytest

from elspeth.contracts.security import secret_fingerprint
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


def test_database_url_odbc_connect_password_removed_from_audit_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """ODBC passthrough DSNs embed PWD inside the odbc_connect value."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BSQL+Server%7D%3BPWD%3Dsecret123%3BServer%3Dhost"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "secret123" not in result.sanitized_url
    assert "PWD" not in result.sanitized_url
    parsed_query = parse_qs(urlparse(result.sanitized_url).query)
    assert parsed_query["odbc_connect"] == ["DRIVER={SQL Server};Server=host"]
    assert result.fingerprint == secret_fingerprint("secret123")


def test_database_url_odbc_connect_braced_password_with_semicolon_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Braced ODBC password values may contain semicolons and must be fully removed."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    url = "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BSQL+Server%7D%3BPWD%3D%7Bp%3Bass%7D%3BServer%3Dhost"

    result = SanitizedDatabaseUrl.from_raw_url(url)

    assert "p%3Bass" not in result.sanitized_url
    assert "p;ass" not in result.sanitized_url
    parsed_query = parse_qs(urlparse(result.sanitized_url).query)
    assert parsed_query["odbc_connect"] == ["DRIVER={SQL Server};Server=host"]
    assert result.fingerprint == secret_fingerprint("p;ass")


def test_database_url_odbc_connect_preserves_non_password_attribute_names() -> None:
    """Only PWD/Password attributes are secret; longer attribute names are preserved."""
    url = "mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BSQL+Server%7D%3BAPWD%3Dkeep%3BNotPassword%3Dalso_keep"

    result = SanitizedDatabaseUrl.from_raw_url(url, fail_if_no_key=False)

    parsed_query = parse_qs(urlparse(result.sanitized_url).query)
    assert parsed_query["odbc_connect"] == ["DRIVER={SQL Server};APWD=keep;NotPassword=also_keep"]
    assert result.fingerprint is None


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


def test_database_url_direct_construction_rejects_odbc_connect_password() -> None:
    """Direct construction rejects ODBC passwords embedded in odbc_connect."""
    with pytest.raises(ValueError, match="sensitive query parameters"):
        SanitizedDatabaseUrl(
            sanitized_url="mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BSQL+Server%7D%3BPWD%3Dleaked",
            fingerprint=None,
        )


def test_database_url_direct_construction_rejects_braced_odbc_connect_password() -> None:
    """Direct construction rejects braced ODBC password values containing semicolons."""
    with pytest.raises(ValueError, match="sensitive query parameters"):
        SanitizedDatabaseUrl(
            sanitized_url="mssql+pyodbc:///?odbc_connect=DRIVER%3D%7BSQL+Server%7D%3BPWD%3D%7Bp%3Bass%7D",
            fingerprint=None,
        )
