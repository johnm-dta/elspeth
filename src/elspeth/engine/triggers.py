"""Trigger evaluation for aggregation batches.

Per plugin-protocol.md: Multiple triggers can be combined (first one to fire wins).
The TriggerEvaluator evaluates all configured triggers with OR logic.

The engine creates one evaluator per aggregation and calls should_trigger()
after each accept. When should_trigger() returns True, which_triggered()
indicates which trigger fired (for audit trail).

Trigger types:
- count: Fires when batch_count >= threshold
- timeout: Fires when batch_age_seconds >= timeout_seconds
- condition: Fires when custom expression evaluates to True
- end_of_source: Implicit - engine handles at source exhaustion (not in TriggerConfig)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from elspeth.contracts.enums import TriggerType
from elspeth.core.config import TriggerConfig
from elspeth.engine.clock import DEFAULT_CLOCK
from elspeth.engine.expression_parser import ExpressionParser

if TYPE_CHECKING:
    from elspeth.engine.clock import Clock


class TriggerEvaluator:
    """Evaluates trigger conditions for an aggregation batch.

    Per plugin-protocol.md: Triggers are combinable (first to fire wins).
    All configured triggers are evaluated with OR logic.

    Created by engine for each aggregation. Tracks batch state (count, age)
    and evaluates whether ANY configured trigger condition is met.

    Example:
        evaluator = TriggerEvaluator(TriggerConfig(count=100, timeout_seconds=60))

        for row in rows:
            if aggregation.accept(row).accepted:
                evaluator.record_accept()
                if evaluator.should_trigger():
                    print(f"Triggered by: {evaluator.which_triggered()}")
                    aggregation.flush()
                    evaluator.reset()
    """

    def __init__(self, config: TriggerConfig, clock: Clock | None = None) -> None:
        """Initialize evaluator with trigger configuration.

        Args:
            config: Trigger configuration from AggregationSettings
            clock: Optional clock for time access. Defaults to system clock.
                   Inject MockClock for deterministic testing.
        """
        self._config = config
        self._clock = clock if clock is not None else DEFAULT_CLOCK
        self._batch_count = 0
        self._first_accept_time: float | None = None
        self._last_triggered: Literal["count", "timeout", "condition"] | None = None

        # Track when each trigger first fired (for "first to fire wins" semantics)
        # Per plugin-protocol.md:1211: "Multiple triggers can be combined (first one to fire wins)"
        self._count_fire_time: float | None = None
        self._condition_fire_time: float | None = None

        # Pre-parse condition expression if applicable
        self._condition_parser: ExpressionParser | None = None
        if config.condition is not None:
            self._condition_parser = ExpressionParser(config.condition)

    @property
    def batch_count(self) -> int:
        """Current number of accepted rows in batch."""
        return self._batch_count

    @property
    def batch_age_seconds(self) -> float:
        """Seconds since first accept in this batch."""
        if self._first_accept_time is None:
            return 0.0
        return self._clock.monotonic() - self._first_accept_time

    def get_age_seconds(self) -> float:
        """Get elapsed time since first accept (alias for batch_age_seconds).

        This method exists for clarity when checkpointing - it returns the
        elapsed time that should be stored in checkpoint state for timeout
        preservation across resume.

        Returns:
            Elapsed seconds since first accept, or 0.0 if no accepts yet
        """
        return self.batch_age_seconds

    def record_accept(self) -> None:
        """Record that a row was accepted into the batch.

        Call this after each successful accept. Updates batch_count,
        starts the timer on first accept, and tracks when triggers first fire.

        Per plugin-protocol.md:1211: "Multiple triggers can be combined (first one to fire wins)"
        We track when each trigger first becomes true so we can report the earliest.
        """
        current_time = self._clock.monotonic()
        self._batch_count += 1

        if self._first_accept_time is None:
            self._first_accept_time = current_time

        # Track when count threshold was first reached
        if self._count_fire_time is None and self._config.count is not None and self._batch_count >= self._config.count:
            self._count_fire_time = current_time

        # Track when condition first became true
        if self._condition_fire_time is None and self._condition_parser is not None:
            context = {
                "batch_count": self._batch_count,
                "batch_age_seconds": current_time - self._first_accept_time,
            }
            result = self._condition_parser.evaluate(context)
            # P2-2026-01-31: Defense-in-depth - reject non-boolean at runtime
            # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
            if not isinstance(result, bool):
                raise TypeError(
                    f"Trigger condition must return bool, got {type(result).__name__}: {result!r}. "
                    f"Expression: {self._condition_parser.expression!r}"
                )
            if result:
                self._condition_fire_time = current_time

    def should_trigger(self) -> bool:
        """Evaluate whether ANY trigger condition is met (OR logic).

        Per plugin-protocol.md:1211: "Multiple triggers can be combined (first one to fire wins)"
        When multiple triggers are satisfied, we report the one that fired EARLIEST,
        not the one checked first in code order.

        Returns:
            True if any configured trigger should fire, False otherwise.

        Side effect:
            Sets _last_triggered to the trigger type that fired first.
        """
        self._last_triggered = None
        current_time = self._clock.monotonic()

        # Collect all triggers that have fired with their fire times
        # Format: (fire_time, trigger_name)
        candidates: list[tuple[float, Literal["count", "timeout", "condition"]]] = []

        # Timeout: fire time is deterministic (first_accept_time + timeout_seconds)
        if self._config.timeout_seconds is not None and self._first_accept_time is not None:
            timeout_fire_time = self._first_accept_time + self._config.timeout_seconds
            if current_time >= timeout_fire_time:
                candidates.append((timeout_fire_time, "timeout"))

        # Count: fire time tracked in record_accept()
        if self._count_fire_time is not None:
            candidates.append((self._count_fire_time, "count"))

        # Condition: Re-evaluate since time-dependent conditions (batch_age_seconds)
        # may have become true after time passed, not just when rows were accepted.
        if self._condition_parser is not None and self._first_accept_time is not None:
            batch_age = current_time - self._first_accept_time
            context = {
                "batch_count": self._batch_count,
                "batch_age_seconds": batch_age,
            }
            result = self._condition_parser.evaluate(context)
            # P2-2026-01-31: Defense-in-depth - reject non-boolean at runtime
            # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
            if not isinstance(result, bool):
                raise TypeError(
                    f"Trigger condition must return bool, got {type(result).__name__}: {result!r}. "
                    f"Expression: {self._condition_parser.expression!r}"
                )
            if result:
                # Condition is true now
                if self._condition_fire_time is None:
                    # First time detecting condition is true - set fire time now.
                    # For conditions that became true due to time passing (not row accepts),
                    # we use current_time as a conservative estimate. We can't know exactly
                    # when it became true without parsing the expression.
                    self._condition_fire_time = current_time
                candidates.append((self._condition_fire_time, "condition"))

        if not candidates:
            return False

        # First to fire wins - sort by fire time and take earliest
        candidates.sort(key=lambda x: x[0])
        self._last_triggered = candidates[0][1]
        return True

    def which_triggered(self) -> Literal["count", "timeout", "condition"] | None:
        """Return which trigger fired on the last should_trigger() call.

        Returns:
            "count", "timeout", or "condition" if a trigger fired.
            None if no trigger fired.

        Note:
            This is used for the audit trail (TriggerType.COUNT, etc.)
        """
        return self._last_triggered

    def get_trigger_type(self) -> TriggerType | None:
        """Get TriggerType enum for the trigger that fired.

        Returns:
            TriggerType enum if a trigger fired, None otherwise.
        """
        if self._last_triggered == "count":
            return TriggerType.COUNT
        elif self._last_triggered == "timeout":
            return TriggerType.TIMEOUT
        elif self._last_triggered == "condition":
            return TriggerType.CONDITION
        return None

    # --- Checkpoint/Restore API (P2-2026-02-01) ---

    def get_count_fire_offset(self) -> float | None:
        """Get the offset from first_accept_time when count trigger fired.

        Returns:
            Seconds after first accept when count fired, or None if not fired.
            Used by checkpoint to preserve "first to fire wins" ordering on resume.
        """
        if self._count_fire_time is None or self._first_accept_time is None:
            return None
        return self._count_fire_time - self._first_accept_time

    def get_condition_fire_offset(self) -> float | None:
        """Get the offset from first_accept_time when condition trigger fired.

        Returns:
            Seconds after first accept when condition fired, or None if not fired.
            Used by checkpoint to preserve "first to fire wins" ordering on resume.
        """
        if self._condition_fire_time is None or self._first_accept_time is None:
            return None
        return self._condition_fire_time - self._first_accept_time

    def restore_from_checkpoint(
        self,
        batch_count: int,
        elapsed_age_seconds: float,
        count_fire_offset: float | None,
        condition_fire_offset: float | None,
    ) -> None:
        """Restore evaluator state from checkpoint data.

        This method restores the evaluator to a state equivalent to having
        processed batch_count rows, with the specified elapsed time and
        trigger fire times preserved.

        P2-2026-02-01: This fixes the bug where record_accept() was used
        during restore, which set fire times to current clock time instead
        of preserving the original ordering.

        Args:
            batch_count: Number of rows in the restored batch
            elapsed_age_seconds: Time elapsed since first accept (for timeout)
            count_fire_offset: Offset from first_accept when count fired, or None
            condition_fire_offset: Offset from first_accept when condition fired, or None
        """
        current_time = self._clock.monotonic()

        # Restore batch count
        self._batch_count = batch_count

        # Restore first_accept_time by rewinding from current time
        # This preserves the batch_age_seconds for timeout calculation
        self._first_accept_time = current_time - elapsed_age_seconds

        # Restore fire times as absolute times (offset from restored first_accept_time)
        if count_fire_offset is not None:
            self._count_fire_time = self._first_accept_time + count_fire_offset
        else:
            self._count_fire_time = None

        if condition_fire_offset is not None:
            self._condition_fire_time = self._first_accept_time + condition_fire_offset
        else:
            self._condition_fire_time = None

    def reset(self) -> None:
        """Reset state for a new batch.

        Call this after flush completes to prepare for the next batch.
        """
        self._batch_count = 0
        self._first_accept_time = None
        self._last_triggered = None
        self._count_fire_time = None
        self._condition_fire_time = None
