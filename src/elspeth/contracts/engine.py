# src/elspeth/contracts/engine.py
"""Engine-related type contracts."""

from dataclasses import dataclass
from typing import TypedDict

from elspeth.contracts.enums import RowOutcome


@dataclass(frozen=True, slots=True)
class PendingOutcome:
    """Pending token outcome waiting for sink durability confirmation.

    This dataclass carries outcome information through the pending_tokens queue
    to be recorded AFTER sink durability is achieved.

    The key insight: token outcomes must only be recorded after sink write + flush
    complete successfully. Recording before durability creates audit trail entries
    that claim data was written when it may not have been.

    Attributes:
        outcome: The terminal outcome (COMPLETED, ROUTED, QUARANTINED, etc.)
        error_hash: Required for QUARANTINED/FAILED outcomes - hash of error details.
                   For other outcomes, this is None.

    Fix for: P1-2026-01-31-quarantine-outcome-before-durability
    """

    outcome: RowOutcome
    error_hash: str | None = None


class RetryPolicy(TypedDict, total=False):
    """Schema for retry configuration from plugin policies.

    All fields are optional - from_policy() applies defaults.

    Attributes:
        max_attempts: Maximum number of attempts (minimum 1)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Random jitter to add to delays in seconds
        exponential_base: Exponential backoff multiplier (default 2.0)
    """

    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float
    exponential_base: float
