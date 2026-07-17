"""Embedded URL query/fragment secret handling for audit text scrubber."""

import pytest

from elspeth.contracts.secret_scrub import scrub_payload_for_audit, scrub_text_for_audit

REDACTED = "<redacted-secret>"


@pytest.mark.parametrize(
    "param_name",
    [
        "access_token",
        "api_key",
        "token",
        "signature",
        "client_secret",
        "authorization",
        "x-api-key",
        "x-amz-signature",
        "x-amz-credential",
        "x-amz-security-token",
    ],
)
def test_embedded_url_sensitive_query_param_redacts_whole_audit_text(param_name: str) -> None:
    text = f"GET https://api.example.com/data?{param_name}=opaque_value&format=json failed with 403"

    assert scrub_text_for_audit(text) == REDACTED


def test_embedded_url_sensitive_query_param_redacts_payload_string_value() -> None:
    payload = {"message": "request to https://api.example.com/data?access_token=opaque_value failed"}

    assert scrub_payload_for_audit(payload)["message"] == REDACTED


def test_embedded_presigned_s3_url_redacts_whole_audit_text() -> None:
    text = (
        "GetObject failed for https://bucket.s3.ap-southeast-2.amazonaws.com/key"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA0000EXAMPLE0000%2F20260717"
        "&X-Amz-Signature=deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    )

    assert scrub_text_for_audit(text) == REDACTED


def test_embedded_url_sensitive_fragment_param_redacts_whole_audit_text() -> None:
    text = "redirected to https://api.example.com/callback#access_token=opaque_value before failure"

    assert scrub_text_for_audit(text) == REDACTED


def test_embedded_url_without_sensitive_params_passes_through() -> None:
    text = "GET https://api.example.com/data?format=json failed with 404"

    assert scrub_text_for_audit(text) == text
