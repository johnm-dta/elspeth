"""Secret-scrubbing helper for DeclarationContractViolation payloads."""

from __future__ import annotations

import base64

import pytest

from elspeth.contracts.freeze import deep_freeze
from elspeth.contracts.secret_scrub import scrub_payload_for_audit, scrub_text_for_audit

REDACTED = "<redacted-secret>"


def test_plain_values_pass_through() -> None:
    p = {"field": "name", "count": 3, "flag": True}
    assert scrub_payload_for_audit(p) == {"field": "name", "count": 3, "flag": True}


def test_plain_text_passes_through() -> None:
    assert scrub_text_for_audit("plain audit message") == "plain audit message"


def test_api_key_like_value_is_redacted() -> None:
    p = {"api_key": "sk-1234567890abcdef1234567890abcdef"}  # secret-scan: allow-this-line
    out = scrub_payload_for_audit(p)
    assert out["api_key"] == REDACTED


def test_secret_like_text_is_redacted() -> None:
    assert scrub_text_for_audit("secret sk-1234567890abcdef1234567890abcdef leaked") == REDACTED


def test_openrouter_key_like_text_is_redacted() -> None:
    assert scrub_text_for_audit("secret sk-or-v1-abcdefghijklmnopqrstuvwxyz123456 leaked") == REDACTED


@pytest.mark.parametrize(
    ("credential_shape", "secret_text"),
    (
        ("github_fine_grained_pat", "github_pat_" + ("A" * 22) + "_" + ("B" * 59)),
        ("github_oauth_token", "gho_" + ("A" * 36)),
        ("github_server_token", "ghs_" + ("A" * 36)),
        ("github_refresh_token", "ghr_" + ("A" * 36)),
        ("google_api_key", "AIza" + ("A" * 35)),
        ("slack_app_token", "xapp-" + ("1-" * 8) + "abcde"),
        ("slack_xoxe_token", "xoxe-" + ("1-" * 8) + "abcde"),
        ("azure_storage_account_key", base64.b64encode(b"a" * 64).decode("ascii")),
    ),
)
def test_common_credential_shapes_in_freeform_text_are_redacted(credential_shape: str, secret_text: str) -> None:
    audit_text = f"plugin diagnostics exposed {credential_shape} {secret_text}"

    assert scrub_text_for_audit(audit_text) == REDACTED


def test_aws_access_key_redacted() -> None:
    p = {"note": "AKIAIOSFODNN7EXAMPLE in log"}  # secret-scan: allow-this-line
    out = scrub_payload_for_audit(p)
    assert "AKIA" not in out["note"]


def test_nested_mapping_scrubbed() -> None:
    p = {"outer": {"inner_key": "sk-abcdef1234567890abcdef1234567890"}}
    out = scrub_payload_for_audit(p)
    assert out["outer"]["inner_key"] == REDACTED


def test_sequence_values_scrubbed() -> None:
    p = {"secrets": ["sk-abcdef1234567890abcdef1234567890", "normal"]}
    out = scrub_payload_for_audit(p)
    assert out["secrets"] == [REDACTED, "normal"]


def test_set_values_scrubbed_after_freeze() -> None:
    p = deep_freeze({"secrets": {"sk-abcdef1234567890abcdef1234567890", "normal"}})
    out = scrub_payload_for_audit(p)
    assert out["secrets"] == [REDACTED, "normal"]


def test_frozenset_values_scrubbed() -> None:
    p = {"secrets": frozenset({"sk-abcdef1234567890abcdef1234567890", "normal"})}
    out = scrub_payload_for_audit(p)
    assert out["secrets"] == [REDACTED, "normal"]


def test_secret_named_tuple_value_redacted_after_freeze() -> None:
    p = deep_freeze({"api_key": ["dev-token-123", "dev-token-456"]})
    out = scrub_payload_for_audit(p)
    assert out["api_key"] == REDACTED


def test_secret_named_mapping_value_redacted_after_freeze() -> None:
    p = deep_freeze({"authorization": {"scheme": "Bearer", "credentials": "opaque-dev-token"}})
    out = scrub_payload_for_audit(p)
    assert out["authorization"] == REDACTED


# -----------------------------------------------------------------------------
# H5 Layer 2 — pattern list expansion (issue elspeth-3956044fb7)
# -----------------------------------------------------------------------------
#
# The closed-set _PATTERNS and _SECRET_KEY_NAMES tables missed four live
# secret formats that can appear in DeclarationContractViolation payloads:
#
# 1. Azure SAS tokens  — the `sig=` parameter in a SAS query string.
# 2. Database connection strings — ODBC-style Password= and URL-style
#    postgres://user:pass@host / mysql://user:pass@host.  # secret-scan: allow-this-line
# 3. Basic-auth URLs  — https://user:pass@host/path.
# 4. Bearer/session tokens under keys other than `authorization`
#    (session_token, access_token, refresh_token, auth_cookie, sas_token,
#    connection_string, conn_string).
#
# Whole-string replacement is load-bearing: partial redaction of
# `Server=x;Password=y;Database=z` would leak Database=z, which is often
# PII-adjacent. Every one of these tests asserts `== REDACTED`, not absence
# of the specific substring.


# ----- Azure SAS tokens -----


def test_azure_sas_token_sig_param_redacted() -> None:
    p = {"uri": "https://acct.blob.core.windows.net/ctr/blob?sv=2021-06-08&sig=abcdef1234567890ABCDEF%2FxyZ%3D&se=2030-01-01T00%3A00%3A00Z"}
    out = scrub_payload_for_audit(p)
    assert out["uri"] == REDACTED


def test_azure_sas_key_name_redacted() -> None:
    p = {"sas_token": "sv=2021-06-08&sig=abcdef1234567890ABCDEF%2FxyZ%3D"}
    out = scrub_payload_for_audit(p)
    assert out["sas_token"] == REDACTED


# ----- Database connection strings -----


def test_odbc_password_param_redacted() -> None:
    p = {"conn": "Server=prod-db.example.com;Database=audit;Uid=service;Password=p@ssw0rd-xyz;Encrypt=yes"}
    out = scrub_payload_for_audit(p)
    assert out["conn"] == REDACTED


def test_postgres_url_with_credentials_redacted() -> None:
    p = {"dsn": "postgresql://dbuser:dbpass-xyz@db.prod.example.com:5432/audit"}  # secret-scan: allow-this-line
    out = scrub_payload_for_audit(p)
    assert out["dsn"] == REDACTED


def test_postgres_short_scheme_redacted() -> None:
    p = {"dsn": "postgres://dbuser:dbpass-xyz@db.prod.example.com/audit"}  # secret-scan: allow-this-line
    out = scrub_payload_for_audit(p)
    assert out["dsn"] == REDACTED


def test_mysql_url_with_credentials_redacted() -> None:
    p = {"dsn": "mysql://root:toor@mysql-ci.internal:3306/metrics"}  # secret-scan: allow-this-line
    out = scrub_payload_for_audit(p)
    assert out["dsn"] == REDACTED


def test_connection_string_key_name_redacted() -> None:
    p = {"connection_string": "Server=x;Database=y;Uid=z;Encrypt=yes"}
    out = scrub_payload_for_audit(p)
    assert out["connection_string"] == REDACTED


def test_conn_string_key_name_redacted() -> None:
    p = {"conn_string": "Server=x;Database=y"}
    out = scrub_payload_for_audit(p)
    assert out["conn_string"] == REDACTED


# ----- Basic-auth URLs -----


def test_https_basic_auth_url_redacted() -> None:
    p = {"endpoint": "https://user:s3cret-pass@api.example.com/v1/resource"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


def test_http_basic_auth_url_redacted() -> None:
    p = {"endpoint": "http://admin:changeme@legacy.internal/health"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


def test_plain_https_url_without_credentials_passes_through() -> None:
    # The basic-auth regex must not fire on a credential-free URL — otherwise
    # we would redact innocent endpoint URLs (e.g. a Landscape resource URI)
    # and the audit trail would lose triage value.
    p = {"endpoint": "https://api.example.com/v1/resource?search=foo"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == "https://api.example.com/v1/resource?search=foo"


def test_https_username_only_userinfo_is_redacted() -> None:
    p = {"endpoint": "https://tokenonlysecret@api.example.com/webhook"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


def test_https_token_query_param_is_redacted() -> None:
    p = {"endpoint": "https://api.example.com/webhook?token=opaque-dev-token"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


def test_https_api_key_query_param_is_redacted() -> None:
    p = {"endpoint": "https://api.example.com/webhook?api_key=opaque-dev-token"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


def test_https_access_token_fragment_is_redacted() -> None:
    p = {"endpoint": "https://api.example.com/callback#access_token=opaque-dev-token"}
    out = scrub_payload_for_audit(p)
    assert out["endpoint"] == REDACTED


# ----- Bearer/session tokens in non-Authorization payload keys -----


def test_session_token_key_name_redacted() -> None:
    p = {"session_token": "abc123notarealtoken"}
    out = scrub_payload_for_audit(p)
    assert out["session_token"] == REDACTED


def test_access_token_key_name_redacted() -> None:
    p = {"access_token": "abc123notarealtoken"}
    out = scrub_payload_for_audit(p)
    assert out["access_token"] == REDACTED


def test_refresh_token_key_name_redacted() -> None:
    p = {"refresh_token": "abc123notarealtoken"}
    out = scrub_payload_for_audit(p)
    assert out["refresh_token"] == REDACTED


def test_auth_cookie_key_name_redacted() -> None:
    p = {"auth_cookie": "abc123notarealtoken"}
    out = scrub_payload_for_audit(p)
    assert out["auth_cookie"] == REDACTED


def test_client_secret_and_private_key_names_redacted() -> None:
    p = {
        "client_secret": "opaque-client-secret",
        "private_key": "opaque-private-key",
    }
    out = scrub_payload_for_audit(p)
    assert out["client_secret"] == REDACTED
    assert out["private_key"] == REDACTED


def test_key_value_secret_text_is_redacted() -> None:
    assert scrub_text_for_audit("contract failed with client_secret=opaque-client-secret") == REDACTED
    assert scrub_text_for_audit("contract failed with Authorization: Bearer opaque-token") == REDACTED


# ----- Key-name match is case-insensitive -----


def test_key_name_match_is_case_insensitive() -> None:
    # Existing behaviour claim — the docstring says key matching is
    # case-insensitive. Pin it so future refactors don't silently break it.
    p = {"Session_Token": "x", "REFRESH_TOKEN": "y", "Auth_Cookie": "z"}
    out = scrub_payload_for_audit(p)
    assert out["Session_Token"] == REDACTED
    assert out["REFRESH_TOKEN"] == REDACTED
    assert out["Auth_Cookie"] == REDACTED
