# src/elspeth/plugins/pooling/errors.py
"""Capacity error classification for pooled API transforms.

Capacity errors are transient overload conditions that should be retried
with AIMD throttling. They are distinct from "normal" errors
(auth failures, malformed requests) which use standard retry limits.

HTTP Status Codes:
- 429: Too Many Requests (universal)
- 503: Service Unavailable (universal)
- 529: Overloaded (Azure, some other providers)
"""

from __future__ import annotations

# HTTP status codes that indicate capacity/rate limiting
# These trigger AIMD throttle and capacity retry
CAPACITY_ERROR_CODES: frozenset[int] = frozenset({429, 503, 529})


def is_capacity_error(status_code: int) -> bool:
    """Check if HTTP status code indicates a capacity error.

    Capacity errors are transient overload conditions that should trigger
    AIMD throttle backoff and be retried with increasing delays.

    Args:
        status_code: HTTP status code

    Returns:
        True if this is a capacity error, False otherwise
    """
    return status_code in CAPACITY_ERROR_CODES


class CapacityError(Exception):
    """Exception for capacity/rate limit errors.

    Raised when an API call fails due to capacity limits.
    These errors trigger AIMD throttle and are retried until
    max_capacity_retry_seconds is exceeded.

    Attributes:
        status_code: HTTP status code that triggered this error
        retryable: Always True for capacity errors
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize capacity error.

        Args:
            status_code: HTTP status code (429, 503, or 529)
            message: Error message
        """
        super().__init__(message)
        self.status_code = status_code
        self.retryable = True
