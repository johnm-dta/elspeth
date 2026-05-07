"""Tests for PluginRetryableError base class and error hierarchy."""

from elspeth.contracts.errors import PluginRetryableError
from elspeth.plugins.infrastructure.clients.llm import LLMClientError
from elspeth.plugins.transforms.web_scrape_errors import WebScrapeError


def test_plugin_retryable_error_has_status_code():
    err = PluginRetryableError("test", retryable=False, status_code=404)
    assert err.retryable is False
    assert err.status_code == 404


def test_plugin_retryable_error_status_code_defaults_none():
    err = PluginRetryableError("test", retryable=True)
    assert err.status_code is None


def test_llm_client_error_is_plugin_retryable_error():
    err = LLMClientError("test", retryable=True)
    assert isinstance(err, PluginRetryableError)
    assert err.retryable is True
    assert err.status_code is None


def test_web_scrape_error_is_plugin_retryable_error():
    err = WebScrapeError("test", retryable=True)
    assert isinstance(err, PluginRetryableError)
    assert err.retryable is True
    assert err.status_code is None


def test_llm_subclasses_still_work():
    """Re-parenting must not break existing subclass behavior."""
    from elspeth.plugins.infrastructure.clients.llm import NetworkError, RateLimitError

    rate_err = RateLimitError("rate limited")
    assert isinstance(rate_err, LLMClientError)
    assert isinstance(rate_err, PluginRetryableError)
    assert rate_err.retryable is True

    net_err = NetworkError("timeout")
    assert isinstance(net_err, LLMClientError)
    assert isinstance(net_err, PluginRetryableError)
    assert net_err.retryable is True


def test_web_scrape_subclasses_still_work():
    """Re-parenting must not break existing subclass behavior."""
    from elspeth.plugins.transforms.web_scrape_errors import (
        NotFoundError,
    )
    from elspeth.plugins.transforms.web_scrape_errors import (
        RateLimitError as WebRateLimitError,
    )
    from elspeth.plugins.transforms.web_scrape_errors import (
        ServerError as WebServerError,
    )

    rate_err = WebRateLimitError("rate limited")
    assert isinstance(rate_err, WebScrapeError)
    assert isinstance(rate_err, PluginRetryableError)
    assert rate_err.retryable is True

    server_err = WebServerError("500")
    assert isinstance(server_err, WebScrapeError)
    assert isinstance(server_err, PluginRetryableError)
    assert server_err.retryable is True

    not_found = NotFoundError("404")
    assert isinstance(not_found, WebScrapeError)
    assert isinstance(not_found, PluginRetryableError)
    assert not_found.retryable is False


def test_all_plugin_errors_share_retryable_interface():
    """All plugin error types have a consistent retryable interface."""
    errors = [
        LLMClientError("test", retryable=True),
        WebScrapeError("test", retryable=True),
        PluginRetryableError("test", retryable=True),
    ]
    for err in errors:
        assert isinstance(err, PluginRetryableError)
        assert err.retryable is True


def test_web_scrape_error_requires_retryable_keyword():
    """WebScrapeError must not have a default for retryable — matches PluginRetryableError contract."""
    import pytest

    with pytest.raises(TypeError):
        WebScrapeError("missing retryable keyword")
