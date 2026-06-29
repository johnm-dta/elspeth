"""Known path-secret handling for sanitized webhook URLs."""

import json as json_module

import pytest

from elspeth.contracts.url import SanitizedWebhookUrl
from elspeth.core.security import secret_fingerprint


def test_slack_webhook_path_secret_removed_and_fingerprinted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known Slack incoming-webhook path secrets are removed."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    secret = "opaque_path_segment_value"
    url = f"https://hooks.slack.com/services/T00000000/B00000000/{secret}"

    result = SanitizedWebhookUrl.from_raw_url(url)

    assert result.sanitized_url == "https://hooks.slack.com/services/T00000000/B00000000/REDACTED"
    assert secret not in result.sanitized_url
    assert result.fingerprint == secret_fingerprint(json_module.dumps([secret], separators=(",", ":")))


def test_unknown_path_token_preserved_because_paths_are_not_generic_secret_boundary() -> None:
    """Arbitrary path tokens are not generically distinguishable from routing IDs."""
    url = "https://api.example.com/services/T00000000/B00000000/opaque-routing-id"

    result = SanitizedWebhookUrl.from_raw_url(url, fail_if_no_key=False)

    assert result.sanitized_url == url
    assert result.fingerprint is None


def test_docstrings_scope_webhook_path_secret_coverage() -> None:
    """Webhook docs must not promise generic path-secret removal."""
    docs = "\n".join(
        doc or ""
        for doc in (
            SanitizedWebhookUrl.__doc__,
            SanitizedWebhookUrl.from_raw_url.__doc__,
        )
    )

    assert "userinfo, sensitive query parameters, sensitive fragments" in docs
    assert "Known Slack incoming-webhook path tokens are redacted" in docs
    assert "Other path-borne secrets are not generically redacted" in docs


def test_direct_constructor_rejects_slack_path_secret() -> None:
    """Direct construction with known Slack path secrets is rejected."""
    with pytest.raises(ValueError, match="known webhook path secret"):
        SanitizedWebhookUrl(
            sanitized_url="https://hooks.slack.com/services/T00000000/B00000000/opaque_path_segment_value",
            fingerprint=None,
        )
