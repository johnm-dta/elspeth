"""Regression tests for webhook URL fragment sanitization."""

import pytest

from elspeth.contracts.url import SanitizedWebhookUrl


def test_plain_fragment_preserved_when_basic_auth_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain fragment anchors stay literal when another URL part is sanitized."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    userinfo = "user:password-placeholder"
    url = f"https://{userinfo}@example.com/page#section-header"

    result = SanitizedWebhookUrl.from_raw_url(url)

    assert result.sanitized_url == "https://example.com/page#section-header"
    assert result.fingerprint is not None


def test_plain_fragment_part_preserved_when_sensitive_fragment_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-param fragment parts stay literal after sensitive fragment params are stripped."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
    sensitive_fragment = "access_token=" + "secret-placeholder"
    url = f"https://example.com/callback#{sensitive_fragment}&section-header"

    result = SanitizedWebhookUrl.from_raw_url(url)

    assert result.sanitized_url == "https://example.com/callback#section-header"
    assert result.fingerprint is not None
