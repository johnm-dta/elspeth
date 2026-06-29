"""Secret key-name normalization coverage for audit payload scrubbing."""

from elspeth.contracts.secret_scrub import scrub_payload_for_audit

REDACTED = "<redacted-secret>"


def test_hyphenated_api_key_name_redacted() -> None:
    payload = {"api-key": "opaque_literal_value"}

    assert scrub_payload_for_audit(payload)["api-key"] == REDACTED


def test_header_style_x_api_key_name_redacted() -> None:
    payload = {"x-api-key": "opaque_literal_value"}

    assert scrub_payload_for_audit(payload)["x-api-key"] == REDACTED


def test_nested_header_style_secret_names_redacted() -> None:
    payload = {
        "headers": {
            "X-API-Key": "opaque_literal_value",
            "X-Auth-Token": "opaque_literal_value",
            "Proxy-Authorization": "opaque_literal_value",
        }
    }

    assert scrub_payload_for_audit(payload)["headers"] == {
        "X-API-Key": REDACTED,
        "X-Auth-Token": REDACTED,
        "Proxy-Authorization": REDACTED,
    }
