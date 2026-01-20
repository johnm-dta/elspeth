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

import time
from typing import Literal

from elspeth.contracts.enums import TriggerType
from elspeth.core.config import TriggerConfig
from elspeth.engine.expression_parser import ExpressionParser


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

    def __init__(self, config: TriggerConfig) -> None:
        """Initialize evaluator with trigger configuration.

        Args:
            config: Trigger configuration from AggregationSettings
        """
        self._config = config
        self._batch_count = 0
        self._first_accept_time: float | None = None
        self._last_triggered: Literal["count", "timeout", "condition"] | None = None

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
        return time.monotonic() - self._first_accept_time

    def record_accept(self) -> None:
        """Record that a row was accepted into the batch.

        Call this after each successful accept. Updates batch_count and
        starts the timer on first accept.
        """
        self._batch_count += 1
        if self._first_accept_time is None:
            self._first_accept_time = time.monotonic()

    def should_trigger(self) -> bool:
        """Evaluate whether ANY trigger condition is met (OR logic).

        Returns:
            True if any configured trigger should fire, False otherwise.

        Side effect:
            Sets _last_triggered to the trigger type that fired.
        """
        self._last_triggered = None

        # Check count trigger
        if self._config.count is not None and self._batch_count >= self._config.count:
            self._last_triggered = "count"
            return True

        # Check timeout trigger
        if self._config.timeout_seconds is not None and self.batch_age_seconds >= self._config.timeout_seconds:
            self._last_triggered = "timeout"
            return True

        # Check condition trigger
        if self._condition_parser is not None:
            # ExpressionParser.evaluate() accepts a dict that becomes "row" in expressions.
            # So row['batch_count'] accesses this dict directly.
            context = {
                "batch_count": self._batch_count,
                "batch_age_seconds": self.batch_age_seconds,
            }
            result = self._condition_parser.evaluate(context)
            if bool(result):
                self._last_triggered = "condition"
                return True

        return False

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

    def reset(self) -> None:
        """Reset state for a new batch.

        Call this after flush completes to prepare for the next batch.
        """
        self._batch_count = 0
        self._first_accept_time = None
        self._last_triggered = None
