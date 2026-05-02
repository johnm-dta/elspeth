"""Engine-related type contracts."""

import math
from dataclasses import dataclass
from typing import ClassVar, TypedDict

from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.freeze import require_int


@dataclass(frozen=True, slots=True)
class BufferEntry[T]:
    """Entry emitted from the reorder buffer with timing metadata.

    This is the contracts-layer type for reorder buffer results, used by
    both the pooling subsystem (plugins/) and audit context types
    (contracts/node_state_context.py).

    Attributes:
        submit_index: Order in which item was submitted (0-indexed)
        complete_index: Order in which item completed (may differ from submit)
        result: The actual result value
        submit_timestamp: time.perf_counter() when submitted
        complete_timestamp: time.perf_counter() when completed
        buffer_wait_ms: Time spent waiting in buffer after completion
    """

    submit_index: int
    complete_index: int
    result: T
    submit_timestamp: float
    complete_timestamp: float
    buffer_wait_ms: float

    def __post_init__(self) -> None:
        require_int(self.submit_index, "BufferEntry.submit_index", min_value=0)
        require_int(self.complete_index, "BufferEntry.complete_index", min_value=0)
        if not math.isfinite(self.submit_timestamp) or self.submit_timestamp < 0:
            raise ValueError(f"BufferEntry.submit_timestamp must be non-negative and finite, got {self.submit_timestamp}")
        if not math.isfinite(self.complete_timestamp) or self.complete_timestamp < 0:
            raise ValueError(f"BufferEntry.complete_timestamp must be non-negative and finite, got {self.complete_timestamp}")
        if not math.isfinite(self.buffer_wait_ms) or self.buffer_wait_ms < 0:
            raise ValueError(f"BufferEntry.buffer_wait_ms must be non-negative and finite, got {self.buffer_wait_ms}")


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

    Quarantine outcomes are recorded after sink durability, not before.
    """

    # Outcomes that require error_hash on PendingOutcome (the misleading
    # legacy name "_FAILURE_OUTCOMES" is preserved for callsite stability;
    # the actual semantic is "outcomes requiring error_hash"). ROUTED_ON_ERROR
    # joins this set because it is the first outcome that both routes through
    # the pending-sink pipeline AND requires error_hash for single-hop audit
    # attributability — see docs/contracts/token-outcomes/00-token-outcome-contract.md.
    _FAILURE_OUTCOMES: ClassVar[frozenset[RowOutcome]] = frozenset(
        {
            RowOutcome.QUARANTINED,
            RowOutcome.FAILED,
            RowOutcome.ROUTED_ON_ERROR,
        }
    )

    outcome: RowOutcome
    error_hash: str | None = None

    def __post_init__(self) -> None:
        """Validate outcome/error_hash consistency.

        QUARANTINED, FAILED, and ROUTED_ON_ERROR outcomes MUST have an
        error_hash — the audit trail needs to reference the error record.
        ROUTED_ON_ERROR is unique among the three in that it ALSO writes
        to a sink (via the pending pipeline); QUARANTINED and FAILED are
        recorded synchronously without sink writes. Other outcomes must NOT
        have an error_hash (an error_hash on COMPLETED would be nonsensical).
        """
        if self.outcome in self._FAILURE_OUTCOMES and (self.error_hash is None or not self.error_hash.strip()):
            raise ValueError(f"PendingOutcome with {self.outcome.name} outcome must have a non-empty error_hash")
        if self.outcome not in self._FAILURE_OUTCOMES and self.error_hash is not None:
            raise ValueError(f"PendingOutcome with {self.outcome.name} outcome must not have error_hash")


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
