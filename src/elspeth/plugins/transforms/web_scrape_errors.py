"""Error hierarchy for web scraping transform.

Follows LLM plugin pattern: retryable errors are re-raised for engine
RetryManager, non-retryable errors return TransformResult.error().
"""

from elspeth.contracts.errors import PluginRetryableError


class WebScrapeError(PluginRetryableError):
    """Base error for web scrape transform."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message, retryable=retryable)


# Retryable errors (re-raise for engine retry)


class RateLimitError(WebScrapeError):
    """HTTP 429 or rate limit exceeded."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class NetworkError(WebScrapeError):
    """Network/connection errors (DNS, timeout, connection refused)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ServerError(WebScrapeError):
    """HTTP 5xx server errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


# Non-retryable errors (return TransformResult.error())


class NotFoundError(WebScrapeError):
    """HTTP 404 Not Found."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class ForbiddenError(WebScrapeError):
    """HTTP 403 Forbidden."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class UnauthorizedError(WebScrapeError):
    """HTTP 401 Unauthorized."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class InvalidURLError(WebScrapeError):
    """Malformed or invalid URL."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


class SSRFBlockedError(WebScrapeError):
    """URL resolves to blocked IP range (private, loopback, cloud metadata)."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)
