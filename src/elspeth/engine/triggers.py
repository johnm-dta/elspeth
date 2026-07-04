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

import bisect
import math
from typing import TYPE_CHECKING, Literal

from elspeth.contracts.enums import TriggerType
from elspeth.core.config import TriggerConfig
from elspeth.core.expression_parser import ExpressionParser
from elspeth.engine.clock import DEFAULT_CLOCK

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
        # Sorted durable arrival instants of every member fed through
        # record_accept (elspeth-eed319ed3d). ADR-030 §E.2 intake adopts rows
        # in (barrier_key, ingest_sequence, work_item_id) order — NOT
        # blocked-at order — so a later accept may carry an EARLIER durable
        # arrival. The count/condition latches recompute over this set so
        # they are pure functions of durable state, invariant under adoption
        # order (§H doctrine), matching the timeout's min-anchor. Empty (and
        # deliberately incomplete) for checkpoint-restored batches — the
        # recompute is guarded on len == batch_count.
        self._member_accept_times: list[float] = []
        # Provenance of the condition latch: True when should_trigger()
        # latched at OBSERVATION time (sampled-at-evaluation,
        # elspeth-06df383e4a) or the latch was checkpoint-restored. The
        # durable replay only governs latches record_accept itself derived.
        self._condition_fire_observed = False

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

    def record_accept(self, accept_time: float | None = None) -> None:
        """Record that a row was accepted into the batch.

        Call this after each successful accept. Updates batch_count,
        starts the timer on first accept, and tracks when triggers first fire.

        Per plugin-protocol.md:1211: "Multiple triggers can be combined (first one to fire wins)"
        We track when each trigger first becomes true so we can report the earliest.

        ``accept_time`` (ADR-030 §E.2 backdated accept timing): an explicit
        monotonic anchor for this accept — the journal-first intake passes the
        row's durable ``barrier_blocked_at`` arrival converted onto this
        clock's monotonic scale. Because intake adoption iterates in
        (barrier_key, ingest_sequence, work_item_id) order — NOT blocked-at
        order — a later accept may carry an EARLIER durable arrival: the
        first-accept anchor min-rewinds, the count latch recomputes as the
        N-th smallest durable arrival, and the condition latch replays over
        the members in durable order (elspeth-eed319ed3d). All three latches
        are therefore pure functions of durable state + config, invariant
        under adoption order and leader takeover (§H pinned doctrine).
        Checkpoint-restored batches carry no member arrival times, so their
        restored latches are preserved and post-restore accepts fall back to
        latch-once semantics. ``None`` preserves the live-clock anchor.
        """
        current_time = accept_time if accept_time is not None else self._clock.monotonic()
        self._batch_count += 1
        insert_index = bisect.bisect_left(self._member_accept_times, current_time)
        self._member_accept_times.insert(insert_index, current_time)
        # A checkpoint-restored batch has fewer member times than members;
        # recomputes are forbidden there (they would derive latches from a
        # partial member set and corrupt the restored instants).
        members_complete = len(self._member_accept_times) == self._batch_count
        appended_in_order = insert_index == len(self._member_accept_times) - 1

        if self._first_accept_time is None or current_time < self._first_accept_time:
            # min-anchor: §H doctrine pins the timeout fire time to the OLDEST
            # member's durable arrival; a backdated adoption arriving out of
            # blocked-at order must still rewind the anchor.
            self._first_accept_time = current_time

        # Count latch: the N-th smallest durable arrival — recomputed on every
        # accept so a backdated adoption lands the same latch as in-order
        # adoption. (For in-order feeds this equals the historical latch-once
        # instant.)
        if self._config.count is not None and self._batch_count >= self._config.count:
            if members_complete:
                self._count_fire_time = self._member_accept_times[self._config.count - 1]
            elif self._count_fire_time is None:
                self._count_fire_time = current_time

        # Condition latch (durable provenance only — observation latches from
        # should_trigger() are sampled-at-evaluation and never rewritten here).
        if self._condition_parser is not None and not self._condition_fire_observed:
            if members_complete and not appended_in_order:
                # The inserted member changed earlier prefix contexts (count
                # and, on an anchor rewind, ages): replay the whole member
                # sequence in durable order and latch the first true instant.
                self._condition_fire_time = self._replay_condition_over_members(self._condition_parser)
            elif self._condition_fire_time is None:
                context = {
                    "batch_count": self._batch_count,
                    "batch_age_seconds": current_time - self._first_accept_time,
                }
                result = self._condition_parser.evaluate(context)
                # Defense-in-depth: reject non-boolean at runtime
                # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
                if not isinstance(result, bool):
                    raise TypeError(
                        f"Trigger condition must return bool, got {type(result).__name__}: {result!r}. "
                        f"Expression: {self._condition_parser.expression!r}"
                    )
                if result:
                    self._condition_fire_time = current_time

    def _replay_condition_over_members(self, parser: ExpressionParser) -> float | None:
        """First durable instant the condition is true over the member sequence.

        A pure function of the sorted member arrival times: evaluates the
        condition at each prefix (batch_count = k, age = arrival_k - arrival_1)
        and returns the k-th arrival of the first true prefix, or None when no
        prefix satisfies it (a window condition can genuinely UNLATCH when an
        anchor rewind grows every age past its window — the durable latch
        follows the durable state).
        """
        times = self._member_accept_times
        for member_count, instant in enumerate(times, start=1):
            context = {
                "batch_count": member_count,
                "batch_age_seconds": instant - times[0],
            }
            result = parser.evaluate(context)
            # Defense-in-depth: reject non-boolean at runtime
            # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
            if not isinstance(result, bool):
                raise TypeError(
                    f"Trigger condition must return bool, got {type(result).__name__}: {result!r}. Expression: {parser.expression!r}"
                )
            if result:
                return instant
        return None

    def should_trigger(self) -> bool:
        """Evaluate whether ANY trigger condition is met (OR logic).

        Per plugin-protocol.md:1211: "Multiple triggers can be combined (first one to fire wins)"
        When multiple triggers are satisfied, we report the one that fired EARLIEST,
        not the one checked first in code order.

        Condition fire times are sampled-at-evaluation: a condition's fire
        time is the first instant it was OBSERVED true (bounded by poll
        cadence), not the unknowable exact crossing instant. Accepted
        residual: a condition truly crossing at t=15 but first observed at a
        t=20 poll loses to a timeout firing at t=18 — that is the defined
        semantic, not a bug.

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

        # Condition: Once fired (latched), always honor the fire time.
        # Re-evaluate only when not yet fired, since time-dependent conditions
        # (batch_age_seconds) may have become true after time passed.
        # Window-based conditions (e.g., batch_age_seconds < 0.5) could "unfire"
        # if re-evaluated after the window closed. Latching fixes this.
        if self._condition_parser is not None and self._first_accept_time is not None:
            if self._condition_fire_time is not None:
                # Already latched — honor the recorded fire time unconditionally
                candidates.append((self._condition_fire_time, "condition"))
            else:
                batch_age = current_time - self._first_accept_time
                context = {
                    "batch_count": self._batch_count,
                    "batch_age_seconds": batch_age,
                }
                result = self._condition_parser.evaluate(context)
                # Defense-in-depth: reject non-boolean at runtime
                # Per CLAUDE.md: "if bool(result)" coercion is forbidden for our data
                if not isinstance(result, bool):
                    raise TypeError(
                        f"Trigger condition must return bool, got {type(result).__name__}: {result!r}. "
                        f"Expression: {self._condition_parser.expression!r}"
                    )
                if result:
                    # Sampled-at-evaluation semantic (elspeth-06df383e4a): a
                    # poll-driven engine only KNOWS condition truth at
                    # observation time, so the fire time is the first instant
                    # the condition was actually observed true — never a
                    # backdated known-false check time or first accept, which
                    # could steal a win from a timeout that genuinely fired
                    # first and corrupt the TriggerType audit value.
                    self._condition_fire_time = current_time
                    # Observation provenance: record_accept's durable replay
                    # must never rewrite a sampled-at-evaluation latch.
                    self._condition_fire_observed = True
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

    # --- Checkpoint/Restore API ---

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

        This fixes a bug where record_accept() was used during restore,
        which set fire times to current clock time instead of preserving
        the original ordering.

        Args:
            batch_count: Number of rows in the restored batch
            elapsed_age_seconds: Time elapsed since first accept (for timeout)
            count_fire_offset: Offset from first_accept when count fired, or None
            condition_fire_offset: Offset from first_accept when condition fired, or None
        """
        if batch_count < 0:
            raise ValueError(f"batch_count must be non-negative, got {batch_count}")
        if elapsed_age_seconds < 0 or not math.isfinite(elapsed_age_seconds):
            raise ValueError(f"elapsed_age_seconds must be non-negative and finite, got {elapsed_age_seconds}")
        if count_fire_offset is not None and (count_fire_offset < 0 or not math.isfinite(count_fire_offset)):
            raise ValueError(f"count_fire_offset must be non-negative and finite, got {count_fire_offset}")
        if condition_fire_offset is not None and (condition_fire_offset < 0 or not math.isfinite(condition_fire_offset)):
            raise ValueError(f"condition_fire_offset must be non-negative and finite, got {condition_fire_offset}")

        current_time = self._clock.monotonic()

        # Restore batch count
        self._batch_count = batch_count

        # Restored members carry NO durable arrival times (the checkpoint
        # persists only scalars), so the member list stays empty: the
        # incomplete-member guard in record_accept keeps every restored latch
        # as-is and post-restore accepts use latch-once semantics.
        self._member_accept_times = []
        self._condition_fire_observed = False

        # Restore first_accept_time by rewinding from current time
        # This preserves the batch_age_seconds for timeout calculation
        self._first_accept_time = current_time - elapsed_age_seconds

        # Restore fire times as absolute times (offset from restored first_accept_time)
        if count_fire_offset is not None:
            self._count_fire_time = self._first_accept_time + count_fire_offset
        elif self._config.count is not None and batch_count >= self._config.count:
            # Backward compatibility for checkpoints that predate count fire offsets:
            # the restored batch already satisfied the count trigger, so latch it
            # conservatively instead of waiting for another accepted row.
            self._count_fire_time = self._first_accept_time
        else:
            self._count_fire_time = None

        if condition_fire_offset is not None:
            self._condition_fire_time = self._first_accept_time + condition_fire_offset
            # A restored latch is not record_accept-derived; the durable
            # replay must never rewrite it.
            self._condition_fire_observed = True
        else:
            # Not yet fired at checkpoint time. On resume, the first
            # should_trigger() that observes the condition true latches the
            # observation instant (sampled-at-evaluation) — same semantic as
            # live evaluation.
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
        self._member_accept_times = []
        self._condition_fire_observed = False
