# src/elspeth/contracts/engine.py
"""Engine-related type contracts."""

from typing import TypedDict


class RetryPolicy(TypedDict, total=False):
    """Schema for retry configuration from plugin policies.

    All fields are optional - from_policy() applies defaults.

    Attributes:
        max_attempts: Maximum number of attempts (minimum 1)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Random jitter to add to delays in seconds
    """

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float
