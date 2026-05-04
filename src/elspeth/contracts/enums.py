"""All status codes, modes, and kinds used across subsystem boundaries.

CRITICAL: Every plugin MUST declare a Determinism value at registration.
There is no "unknown" - undeclared determinism crashes at registration time.
This is per ELSPETH's principle: "I don't know what happened" is never acceptable.
"""

from enum import StrEnum


class RunStatus(StrEnum):
    """Status of a pipeline run.

    Stored in the database (runs.status).

    The four-value terminal taxonomy (COMPLETED / COMPLETED_WITH_FAILURES /
    FAILED / EMPTY) was introduced in Phase 2.2 (elspeth-0de989c56d) so an
    operator scanning ``/api/runs/{rid}`` can distinguish "ran cleanly" from
    "ran but no row succeeded" without reading diagnostics.  RUNNING is
    non-terminal; INTERRUPTED is signal-bounded (SIGINT/SIGTERM).

    The presence-indicator predicate that maps row-count shapes to status
    values is enforced in :class:`elspeth.contracts.run_result.RunResult`'s
    ``__post_init__`` (the in-memory engine record carries the row counters).
    The web sessions DB and the Pydantic API schemas mirror the same
    invariant; the Landscape audit ``Run`` dataclass has no row-count
    fields so the enum widening alone is sufficient at that layer.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    FAILED = "failed"
    EMPTY = "empty"
    INTERRUPTED = "interrupted"


class NodeStateStatus(StrEnum):
    """Status of a node processing a token.

    Stored in database (node_states.status).
    """

    OPEN = "open"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ExportStatus(StrEnum):
    """Status of run export operation.

    Stored in the database.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchStatus(StrEnum):
    """Status of an aggregation batch.

    Stored in database (batches.status).
    """

    DRAFT = "draft"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class TriggerType(StrEnum):
    """Type of trigger that caused an aggregation batch to execute.

    Stored in database (batches.trigger_type).

    Values:
        COUNT: Batch reached configured row count threshold
        TIMEOUT: Batch reached configured time limit
        CONDITION: Custom condition expression evaluated to true
        END_OF_SOURCE: Source exhausted, flush remaining rows
    """

    COUNT = "count"
    TIMEOUT = "timeout"
    CONDITION = "condition"
    END_OF_SOURCE = "end_of_source"


class NodeType(StrEnum):
    """Type of node in the execution graph.

    Stored in database (nodes.node_type).
    """

    SOURCE = "source"
    TRANSFORM = "transform"
    GATE = "gate"
    AGGREGATION = "aggregation"
    COALESCE = "coalesce"
    SINK = "sink"


class Determinism(StrEnum):
    """Plugin determinism classification for reproducibility.

    Every plugin MUST declare one of these at registration. No default.
    Undeclared determinism = crash at registration time.

    Each value tells you what to do for replay/verify:
    - DETERMINISTIC: Just re-run, expect identical output
    - SEEDED: Capture seed, replay with same seed
    - IO_READ: Capture what was read (time, files, env)
    - IO_WRITE: Be careful - has side effects on replay
    - EXTERNAL_CALL: Record request/response for replay
    - NON_DETERMINISTIC: Must record output, cannot reproduce

    Stored in database (nodes.determinism).
    """

    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    EXTERNAL_CALL = "external_call"
    NON_DETERMINISTIC = "non_deterministic"


class RoutingKind(StrEnum):
    """Kind of routing action from a gate.

    Stored in routing_events.
    """

    CONTINUE = "continue"
    ROUTE = "route"
    FORK_TO_PATHS = "fork_to_paths"


class RoutingMode(StrEnum):
    """Mode for routing edges.

    MOVE: Token exits current path, goes to destination only
    COPY: Token clones to destination AND continues on current path
    DIVERT: Token is diverted from normal flow to error/quarantine sink.
            Like MOVE, but semantically distinct: represents failure handling,
            not intentional routing. Used for source quarantine and transform
            on_error edges. These are structural markers in the DAG — rows
            reach these sinks via exception handling, not by traversing the edge.

    Stored in the database.
    """

    MOVE = "move"
    COPY = "copy"
    DIVERT = "divert"


class RowOutcome(StrEnum):
    """Outcome for a token in the pipeline.

    These outcomes are explicitly recorded in the `token_outcomes` table
    (AUD-001) at determination time. The (StrEnum) base allows direct
    database storage via .value.

    Most outcomes are TERMINAL - the token's journey is complete:
    - COMPLETED: Reached output sink successfully
    - ROUTED: Sent to named sink by gate (intentional MOVE)
    - ROUTED_ON_ERROR: Successful sink write capturing an upstream
      transform error (DIVERT — operationally distinct from FAILED;
      the row reached an error sink with the originating error_hash)
    - FORKED: Split into multiple parallel paths (parent token)
    - FAILED: Processing failed, not recoverable
    - QUARANTINED: Failed validation, stored for investigation
    - DIVERTED: Sink write failed for this row, diverted to failsink
    - CONSUMED_IN_BATCH: Absorbed into aggregate (single/transform mode)
    - DROPPED_BY_FILTER: Transform intentionally emitted zero rows
    - COALESCED: Merged in join from parallel paths
    - EXPANDED: Deaggregated into child tokens (parent token)

    One outcome is NON-TERMINAL - the token will reappear:
    - BUFFERED: Held for batch processing in passthrough mode

    The terminal/non-terminal partition is enforced at module-import
    time by the closed-set assertions following this class definition;
    a future enum addition that fails to land in exactly one of
    ``_TERMINAL_ROW_OUTCOMES`` / ``_NON_TERMINAL_ROW_OUTCOMES`` will
    raise ``AssertionError`` before any code executes.
    """

    # Terminal outcomes
    COMPLETED = "completed"
    ROUTED = "routed"
    ROUTED_ON_ERROR = "routed_on_error"
    FORKED = "forked"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    DIVERTED = "diverted"
    CONSUMED_IN_BATCH = "consumed_in_batch"
    DROPPED_BY_FILTER = "dropped_by_filter"
    COALESCED = "coalesced"
    EXPANDED = "expanded"

    # Non-terminal outcomes
    BUFFERED = "buffered"

    @property
    def is_terminal(self) -> bool:
        """Check if this outcome represents a final state for the token.

        Terminal outcomes mean the token's journey is complete - it won't
        appear again in results. Non-terminal outcomes (BUFFERED) mean
        the token is temporarily held and will reappear with a final outcome.

        Closed-set membership (rather than ``!= BUFFERED``): under the
        negative-logic predicate, a future non-terminal addition would
        silently classify as terminal and corrupt the Tier 1 cross-check
        at ``core/landscape/model_loaders.py:539`` (DB-stored
        ``is_terminal`` vs enum-derived ``is_terminal``).  Closed-set
        membership combined with the module-level exhaustiveness
        assertions below forces every new value to be classified
        deliberately at definition time.
        """
        return self in _TERMINAL_ROW_OUTCOMES


# elspeth-879f6de6bd-followup (S-2): closed-set partition of RowOutcome
# members.  Mirrors the LEGAL_RUN_TRANSITIONS pattern in
# web/sessions/protocol.py:62-80 — module-level assertions raise
# AssertionError at import time if the partition drifts from the canonical
# enum, so a future RowOutcome addition cannot ship without explicit
# classification.
#
# Update discipline: when adding a new RowOutcome value, also add it to
# exactly one of these sets in this same edit.  The exhaustiveness check
# below will fail loudly otherwise.
_TERMINAL_ROW_OUTCOMES: frozenset[RowOutcome] = frozenset(
    {
        RowOutcome.COMPLETED,
        RowOutcome.ROUTED,
        RowOutcome.ROUTED_ON_ERROR,
        RowOutcome.FORKED,
        RowOutcome.FAILED,
        RowOutcome.QUARANTINED,
        RowOutcome.DIVERTED,
        RowOutcome.CONSUMED_IN_BATCH,
        RowOutcome.DROPPED_BY_FILTER,
        RowOutcome.COALESCED,
        RowOutcome.EXPANDED,
    }
)

_NON_TERMINAL_ROW_OUTCOMES: frozenset[RowOutcome] = frozenset(
    {
        RowOutcome.BUFFERED,
    }
)

# Exhaustiveness: every RowOutcome value MUST appear in exactly one of
# the two sets.  An unclassified value would silently miscount via the
# ``case _:`` arms in resume aggregation (engine/orchestrator/core.py)
# and corrupt the audit integrity invariant in
# core/landscape/model_loaders.py:539.
_all_row_outcomes = frozenset(RowOutcome)
_classified_row_outcomes = _TERMINAL_ROW_OUTCOMES | _NON_TERMINAL_ROW_OUTCOMES
if _classified_row_outcomes != _all_row_outcomes:
    _unclassified = _all_row_outcomes - _classified_row_outcomes
    raise AssertionError(
        f"RowOutcome members {sorted(m.name for m in _unclassified)} are not "
        f"classified into _TERMINAL_ROW_OUTCOMES or _NON_TERMINAL_ROW_OUTCOMES "
        f"in contracts/enums.py — every new RowOutcome value must be added to "
        f"exactly one of these sets."
    )

# Mutual exclusion: a value cannot be both terminal and non-terminal.
# This guards against a copy-paste error that lands a value in both sets.
_overlap = _TERMINAL_ROW_OUTCOMES & _NON_TERMINAL_ROW_OUTCOMES
if _overlap:
    raise AssertionError(
        f"RowOutcome members {sorted(m.name for m in _overlap)} appear in BOTH "
        f"_TERMINAL_ROW_OUTCOMES and _NON_TERMINAL_ROW_OUTCOMES — these sets must "
        f"be disjoint."
    )


# ADR-019 (two-axis terminal model): TerminalOutcome and TerminalPath split the
# single-axis ``RowOutcome`` into a lifecycle answer (outcome) and a provenance
# answer (path).  See ``docs/architecture/adr/019-two-axis-terminal-model.md``
# § "Counter derivation contract — public API field names preserved" (round-4
# amendment, 2026-05-04) for the normative ``(outcome, path) → counter
# increment`` mapping that the migration's Stage 2/3 PR enforces.
#
# Stage 1 (this commit) introduces these enums ALONGSIDE the existing
# ``RowOutcome``.  No producer/recorder/accumulator/test reads or writes the
# new fields yet; ``RowOutcome`` continues to drive the audit trail.  Stage 5
# removes ``RowOutcome`` once Stages 2-4 have flipped every reader and
# producer.  Until then, the two-axis closed-set partition below runs in
# parallel with the ``_TERMINAL_ROW_OUTCOMES`` / ``_NON_TERMINAL_ROW_OUTCOMES``
# partition above; both must succeed at module-import time.
class TerminalOutcome(StrEnum):
    """Lifecycle answer for a row that has reached a terminal state.

    ADR-019 § Decision: when ``completed=True``, ``outcome`` is one of three
    values; when ``completed=False`` (only ``BUFFERED`` today), ``outcome`` is
    NULL.  ``SUCCESS`` and ``FAILURE`` are predicate inputs to
    ``RunResult.__post_init__``'s ``RunStatus`` derivation; ``TRANSIENT`` is
    explicitly NOT a predicate input — it marks parent-token bookkeeping
    (``FORK_PARENT``, ``EXPAND_PARENT``), batch absorption (``BATCH_CONSUMED``),
    and sink-fallback-to-failsink absorptions whose lifecycle answers live on
    a paired ``token_outcomes`` row, ``node_state``, or ``artifacts`` row
    elsewhere.

    See ADR-019 § "Why TRANSIENT exists as a third outcome value" for the
    rationale that admits this third value.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    TRANSIENT = "transient"


class TerminalPath(StrEnum):
    """Provenance answer for a row's terminal — how did it get there?

    Producer-declared, producer-emitted; never inferred from graph topology
    or counter context.  See ADR-019 § "Classification is producer-declared,
    not topology-derivable" — ``ON_ERROR_ROUTED`` and
    ``SINK_FALLBACK_TO_FAILSINK`` are structurally identical at the audit
    layer (both write a paired ``NodeStateStatus.COMPLETED`` ``node_state``
    plus an ``artifacts`` row at a different node), so only the producer
    knows whether the lifecycle answer is FAILURE (transform threw, on-error
    sink received) or TRANSIENT (sink-write fallback for visibility).

    Stored alongside ``TerminalOutcome`` in the post-Stage-2 ``token_outcomes``
    schema.  ``BUFFERED`` is the only non-terminal path — it pairs with
    ``outcome IS NULL`` to mark a row that hasn't decided yet.
    """

    DEFAULT_FLOW = "default_flow"
    GATE_ROUTED = "gate_routed"
    ON_ERROR_ROUTED = "on_error_routed"
    FILTER_DROPPED = "filter_dropped"
    COALESCED = "coalesced"
    UNROUTED = "unrouted"
    QUARANTINED_AT_SOURCE = "quarantined_at_source"
    SINK_FALLBACK_TO_FAILSINK = "sink_fallback_to_failsink"
    SINK_DISCARDED = "sink_discarded"
    FORK_PARENT = "fork_parent"
    EXPAND_PARENT = "expand_parent"
    BATCH_CONSUMED = "batch_consumed"
    BUFFERED = "buffered"


# Closed-set partition over the cross-product of TerminalOutcome and
# TerminalPath per the ADR-019 mapping table at lines 99-115.  Every legal terminal
# pair is enumerated below; the assertion that follows verifies every
# ``TerminalPath`` is either covered by a legal terminal pair OR present in
# ``_NON_TERMINAL_PATHS``.  This mirrors the existing
# ``_TERMINAL_ROW_OUTCOMES`` / ``_NON_TERMINAL_ROW_OUTCOMES`` partition above
# (which protects the unchanged ``RowOutcome``); both run in parallel until
# Stage 5 deletes ``RowOutcome``.
#
# NOTE: ``DIVERTED`` (the single ``RowOutcome`` value) maps to TWO legal pairs
# under the two-axis model — ``(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)`` for
# the failsink-mode case (paired ``NodeStateStatus.COMPLETED`` + ``artifacts``
# row at the failsink node) and ``(FAILURE, SINK_DISCARDED)`` for the
# discard-mode case (no failsink, primary node_state at FAILED).  See ADR-019
# § Sub-decisions Resolved by Panel Review (verdict 5).
_LEGAL_TERMINAL_PAIRS: frozenset[tuple[TerminalOutcome, TerminalPath]] = frozenset(
    {
        (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW),
        (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED),
        (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED),
        (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED),
        (TerminalOutcome.SUCCESS, TerminalPath.COALESCED),
        (TerminalOutcome.FAILURE, TerminalPath.UNROUTED),
        (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE),
        (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK),
        (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED),
        (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT),
        (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT),
        (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED),
    }
)

_NON_TERMINAL_PATHS: frozenset[TerminalPath] = frozenset(
    {
        TerminalPath.BUFFERED,
    }
)


# Exhaustiveness: every TerminalPath value MUST be covered by either a legal
# terminal pair (paired with some TerminalOutcome) or the non-terminal set.
# An unclassified path would silently land in the ``case _:`` arm of any
# future (outcome, path) match in the recorder/accumulator and corrupt the
# audit-integrity invariant the way an unclassified ``RowOutcome`` would.
_paths_in_terminal_pairs: frozenset[TerminalPath] = frozenset(path for _, path in _LEGAL_TERMINAL_PAIRS)
_all_terminal_paths: frozenset[TerminalPath] = frozenset(TerminalPath)
_classified_terminal_paths: frozenset[TerminalPath] = _paths_in_terminal_pairs | _NON_TERMINAL_PATHS
if _classified_terminal_paths != _all_terminal_paths:
    _unclassified_paths = _all_terminal_paths - _classified_terminal_paths
    raise AssertionError(
        f"TerminalPath members {sorted(p.name for p in _unclassified_paths)} are "
        f"not classified into _LEGAL_TERMINAL_PAIRS or _NON_TERMINAL_PATHS in "
        f"contracts/enums.py — every new TerminalPath value must be added to "
        f"exactly one (paired with a TerminalOutcome in _LEGAL_TERMINAL_PAIRS, "
        f"or listed alone in _NON_TERMINAL_PATHS)."
    )

# Mutual exclusion: a path cannot be both terminal-paired and non-terminal.
_paths_overlap = _paths_in_terminal_pairs & _NON_TERMINAL_PATHS
if _paths_overlap:
    raise AssertionError(
        f"TerminalPath members {sorted(p.name for p in _paths_overlap)} appear in "
        f"BOTH _LEGAL_TERMINAL_PAIRS and _NON_TERMINAL_PATHS — these sets must be "
        f"disjoint (a path is either terminal-paired or non-terminal, never both)."
    )

# Outcome exhaustiveness: every TerminalOutcome value MUST be the lifecycle
# answer for at least one legal terminal pair.  An unused outcome would mean
# the enum has dead values that no producer can emit — drift from the ADR.
_outcomes_in_terminal_pairs: frozenset[TerminalOutcome] = frozenset(outcome for outcome, _ in _LEGAL_TERMINAL_PAIRS)
_all_terminal_outcomes: frozenset[TerminalOutcome] = frozenset(TerminalOutcome)
if _outcomes_in_terminal_pairs != _all_terminal_outcomes:
    _orphaned_outcomes = _all_terminal_outcomes - _outcomes_in_terminal_pairs
    raise AssertionError(
        f"TerminalOutcome members {sorted(o.name for o in _orphaned_outcomes)} "
        f"do not appear in any pair in _LEGAL_TERMINAL_PAIRS in contracts/enums.py "
        f"— every TerminalOutcome value must be the lifecycle answer for at least "
        f"one legal (outcome, path) pair per the ADR-019 mapping table."
    )


class CallType(StrEnum):
    """Type of external call (Phase 6).

    Stored in database (calls.call_type).
    """

    LLM = "llm"
    HTTP = "http"
    HTTP_REDIRECT = "http_redirect"
    SQL = "sql"
    VECTOR = "vector"
    FILESYSTEM = "filesystem"


class CallStatus(StrEnum):
    """Status of an external call (Phase 6).

    Stored in database (calls.status).
    """

    SUCCESS = "success"
    ERROR = "error"


class RunMode(StrEnum):
    """Pipeline execution mode for live/replay/verify behavior.

    Stored in database (runs.run_mode).

    Values:
        LIVE: Make real API calls, record everything
        REPLAY: Use recorded responses, skip live calls
        VERIFY: Make real calls, compare to recorded
    """

    LIVE = "live"
    REPLAY = "replay"
    VERIFY = "verify"


class TelemetryGranularity(StrEnum):
    """Granularity of telemetry events emitted by the TelemetryManager.

    Values:
        LIFECYCLE: Only run start/complete/failed events (minimal overhead)
        ROWS: Lifecycle + row-level events (row_started, row_completed, etc.)
        FULL: Rows + external call events (LLM requests, HTTP calls, etc.)
    """

    LIFECYCLE = "lifecycle"
    ROWS = "rows"
    FULL = "full"


class BackpressureMode(StrEnum):
    """How to handle backpressure when telemetry exporters can't keep up.

    Values:
        BLOCK: Block the pipeline until exporters catch up (safest, may slow pipeline)
        DROP: Drop events when buffer is full (lossy, no pipeline impact)
        SLOW: Adaptive rate limiting (not yet implemented)
    """

    BLOCK = "block"
    DROP = "drop"
    SLOW = "slow"


# Backpressure modes that are currently implemented.
# Used by RuntimeTelemetryConfig.from_settings() to fail fast on unimplemented modes.
_IMPLEMENTED_BACKPRESSURE_MODES = frozenset({BackpressureMode.BLOCK, BackpressureMode.DROP})


class ReproducibilityGrade(StrEnum):
    """Reproducibility levels for a completed run.

    Grades:
    - FULL_REPRODUCIBLE: All nodes are deterministic or seeded. The run can be
      fully re-executed with identical results (given the same seed).
    - REPLAY_REPRODUCIBLE: At least one node is nondeterministic (e.g., LLM calls).
      Results can only be replayed using recorded external call responses.
    - ATTRIBUTABLE_ONLY: Payloads have been purged. We can verify what happened
      via hashes, but cannot replay the run.

    Stored in database (runs.reproducibility_grade).
    """

    FULL_REPRODUCIBLE = "full_reproducible"
    REPLAY_REPRODUCIBLE = "replay_reproducible"
    ATTRIBUTABLE_ONLY = "attributable_only"


class OutputMode(StrEnum):
    """Output mode for aggregation batches.

    Stored in database.

    Values:
        PASSTHROUGH: Emit buffered rows unchanged after flush
        TRANSFORM: Emit transformed output from aggregation plugin
    """

    PASSTHROUGH = "passthrough"
    TRANSFORM = "transform"


def error_edge_label(transform_id: str) -> str:
    """Canonical label for a transform error DIVERT edge.

    Shared between DAG construction (dag.py) and error-routing audit recording
    (executors.py, processor.py) to prevent label drift.

    Args:
        transform_id: Stable transform name for error-route labels.
    """
    return f"__error_{transform_id}__"
