# src/elspeth/plugins/results.py
"""Result types for plugin operations.

These types define the contracts between plugins and the SDA engine.
All fields needed for Phase 3 Landscape/OpenTelemetry integration are
included here, even if not used until Phase 3.
"""

from enum import Enum


class RowOutcome(Enum):
    """Terminal states for rows in the pipeline.

    INVARIANT: Every row reaches exactly one terminal state.
    No silent drops.
    """

    COMPLETED = "completed"           # Reached output sink
    ROUTED = "routed"                 # Sent to named sink by gate (move mode)
    FORKED = "forked"                 # Split into child tokens (parent terminates)
    CONSUMED_IN_BATCH = "consumed_in_batch"  # Fed into aggregation
    COALESCED = "coalesced"           # Merged with other tokens
    QUARANTINED = "quarantined"       # Failed, stored for investigation
    FAILED = "failed"                 # Failed, not recoverable
