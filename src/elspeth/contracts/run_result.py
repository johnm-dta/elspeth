"""Pipeline run result — pure data type for pipeline execution outcomes.

Moved to L0 (contracts/) because it has no dependencies above L0: uses only
RunStatus (L0), freeze_fields (L0), and stdlib types. This placement allows
PipelineRunner protocol (also L0) to reference it without a layer violation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.freeze import deep_thaw, freeze_fields, require_int


@dataclass(frozen=True, slots=True)
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0
    rows_coalesce_failed: int = 0  # Coalesce failures (quorum_not_met, incomplete_branches)
    rows_expanded: int = 0  # Deaggregation parent tokens
    rows_buffered: int = 0  # Passthrough mode buffered tokens
    rows_diverted: int = 0  # Rows diverted to failsink during sink write
    routed_destinations: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if not isinstance(self.status, RunStatus):
            raise TypeError(f"RunResult.status must be a RunStatus enum, got {type(self.status).__name__}: {self.status!r}")
        require_int(self.rows_processed, "rows_processed", min_value=0)
        require_int(self.rows_succeeded, "rows_succeeded", min_value=0)
        require_int(self.rows_failed, "rows_failed", min_value=0)
        require_int(self.rows_routed, "rows_routed", min_value=0)
        require_int(self.rows_quarantined, "rows_quarantined", min_value=0)
        require_int(self.rows_forked, "rows_forked", min_value=0)
        require_int(self.rows_coalesced, "rows_coalesced", min_value=0)
        require_int(self.rows_coalesce_failed, "rows_coalesce_failed", min_value=0)
        require_int(self.rows_expanded, "rows_expanded", min_value=0)
        require_int(self.rows_buffered, "rows_buffered", min_value=0)
        require_int(self.rows_diverted, "rows_diverted", min_value=0)
        freeze_fields(self, "routed_destinations")
        self._check_status_invariant()

    def _check_status_invariant(self) -> None:
        """Phase 2.2 (elspeth-0de989c56d) — biconditional invariant linking
        ``status`` to the row-count shape using the presence-indicator
        predicate documented on the issue (comment 693).

        Non-terminal (``RUNNING``) and signal-bounded (``INTERRUPTED``)
        statuses bypass the predicate — partial-progress shapes are
        intentionally representable for the resume / SIGINT paths.

        Failure indicators are ``rows_failed``, ``rows_quarantined``, and
        ``rows_coalesce_failed``; ``rows_routed`` is excluded because the
        engine's counter conflates DIVERT (failure-handling) with MOVE
        (intentional gate routing).  See ``elspeth-obs-abc8baa1cd`` for
        the structural-counter follow-up.
        """
        has_failures = self.rows_failed > 0 or self.rows_quarantined > 0 or self.rows_coalesce_failed > 0

        match (self.status, self.rows_processed, self.rows_succeeded, has_failures):
            case (RunStatus.RUNNING, _, _, _):
                return
            case (RunStatus.INTERRUPTED, _, _, _):
                return
            case (RunStatus.COMPLETED, _, s, False) if s > 0:
                # rows_processed > 0 AND rows_succeeded > 0 is the typical
                # success shape; rows_processed == 0 AND rows_succeeded > 0
                # is the resume / coalesce-continuation shape (the resume
                # ingested no NEW source rows but flushed a restored
                # coalesce that produced successful tokens).  Both are
                # operationally COMPLETED.
                return
            case (RunStatus.COMPLETED, _, s, _) if s == 0:
                raise ValueError(
                    f"RunResult: status=COMPLETED requires rows_succeeded > 0, got rows_succeeded={s} "
                    f"(use status=FAILED when no row reached the success path)"
                )
            case (RunStatus.COMPLETED, _, _, True):
                raise ValueError(
                    f"RunResult: status=COMPLETED requires no failures "
                    f"(rows_failed={self.rows_failed}, rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed}); use status=COMPLETED_WITH_FAILURES "
                    f"when at least one row reached a failure terminal state"
                )
            case (RunStatus.COMPLETED_WITH_FAILURES, p, s, True) if p > 0 and s > 0:
                return
            case (RunStatus.COMPLETED_WITH_FAILURES, _, s, _) if s == 0:
                raise ValueError(
                    f"RunResult: status=COMPLETED_WITH_FAILURES requires rows_succeeded > 0, "
                    f"got rows_succeeded={s} (use status=FAILED when no row reached the success path)"
                )
            case (RunStatus.COMPLETED_WITH_FAILURES, _, _, False):
                raise ValueError(
                    f"RunResult: status=COMPLETED_WITH_FAILURES requires at least one failure indicator "
                    f"(rows_failed > 0 or rows_quarantined > 0 or rows_coalesce_failed > 0); "
                    f"got rows_failed={self.rows_failed}, rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed} (use status=COMPLETED for clean runs)"
                )
            case (RunStatus.FAILED, _, _, _):
                # FAILED has two semantic origins under Phase 2.2:
                #
                #   (1) The presence-indicator predicate decided the run
                #       had no successful rows (rows_succeeded == 0).
                #   (2) An out-of-band exception bounded the run mid-flight
                #       (orchestrator's _RunFailedWithPartialResultError
                #       path).  Partial counters at the moment of the
                #       exception may include rows_succeeded > 0.
                #
                # ``derive_terminal_run_status`` only ever picks FAILED for
                # case (1) — case (2) reaches FAILED via the orchestrator's
                # explicit failure ceremony, where the row-count shape is a
                # snapshot of work-in-progress rather than a terminal-state
                # decomposition.  The biconditional therefore tolerates any
                # ``rows_succeeded`` under FAILED; the predicate enforces
                # the structural distinction at the engine's success-path
                # decision site.
                return
            case (RunStatus.EMPTY, 0, 0, False):
                return
            case (RunStatus.EMPTY, p, _, _) if p > 0:
                raise ValueError(f"RunResult: status=EMPTY requires rows_processed == 0, got rows_processed={p}")
            case (RunStatus.EMPTY, _, s, _) if s > 0:
                raise ValueError(f"RunResult: status=EMPTY requires rows_succeeded == 0, got rows_succeeded={s}")
            case (RunStatus.EMPTY, _, _, True):
                raise ValueError(
                    f"RunResult: status=EMPTY requires no failures "
                    f"(rows_failed={self.rows_failed}, rows_quarantined={self.rows_quarantined}, "
                    f"rows_coalesce_failed={self.rows_coalesce_failed}); use status=FAILED when "
                    f"the run encountered failures with no successful rows"
                )
            case _:
                raise ValueError(
                    f"RunResult: unhandled status/row-count shape: status={self.status!r}, "
                    f"rows_processed={self.rows_processed}, rows_succeeded={self.rows_succeeded}, "
                    f"has_failures={has_failures}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON export.

        Replaces ``dataclasses.asdict()`` which cannot deep-copy
        ``MappingProxyType`` fields (raises ``TypeError: cannot pickle
        'mappingproxy' object``).
        """
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "rows_processed": self.rows_processed,
            "rows_succeeded": self.rows_succeeded,
            "rows_failed": self.rows_failed,
            "rows_routed": self.rows_routed,
            "rows_quarantined": self.rows_quarantined,
            "rows_forked": self.rows_forked,
            "rows_coalesced": self.rows_coalesced,
            "rows_coalesce_failed": self.rows_coalesce_failed,
            "rows_expanded": self.rows_expanded,
            "rows_buffered": self.rows_buffered,
            "rows_diverted": self.rows_diverted,
            "routed_destinations": deep_thaw(self.routed_destinations),
        }


def derive_terminal_run_status(
    *,
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_quarantined: int,
    rows_coalesce_failed: int,
) -> RunStatus:
    """Phase 2.2 (elspeth-0de989c56d) — pick a terminal RunStatus from row counts.

    Implements the presence-indicator predicate documented on the issue
    (comment 693).  Used by the orchestrator at run completion to set the
    persisted ``runs.status`` and the operator-facing API status without
    duplicating the predicate at every call site.

    The result is constrained to the four-value terminal taxonomy
    (``COMPLETED`` / ``COMPLETED_WITH_FAILURES`` / ``FAILED`` / ``EMPTY``);
    callers that need ``INTERRUPTED`` or ``RUNNING`` set those values
    directly.

    ``rows_routed`` is excluded from the predicate because the engine's
    counter conflates DIVERT (failure-handling) with MOVE (intentional gate
    routing).  See ``elspeth-obs-abc8baa1cd`` for the structural-counter
    follow-up.

    Args:
        rows_processed: Total rows ingested from the source.
        rows_succeeded: Rows that reached a success terminal state.
        rows_failed: Rows that reached the FAILED terminal state.
        rows_quarantined: Rows quarantined at the source / processor.
        rows_coalesce_failed: Coalesce-quorum failures.

    Returns:
        A terminal RunStatus consistent with the row-count shape.
    """
    has_failures = rows_failed > 0 or rows_quarantined > 0 or rows_coalesce_failed > 0
    if rows_processed == 0 and rows_succeeded == 0:
        # An empty source with no failures is EMPTY; if the engine took
        # the failed path before any source iteration but recorded a
        # failure counter (defensive case), prefer FAILED so the audit
        # trail names the failure rather than the absence.
        return RunStatus.FAILED if has_failures else RunStatus.EMPTY
    if rows_succeeded == 0:
        # rows_processed > 0 AND rows_succeeded == 0 — the run ingested
        # rows but none reached the success path (S1A / S1B-msg2 shapes).
        return RunStatus.FAILED
    if has_failures:
        return RunStatus.COMPLETED_WITH_FAILURES
    # rows_succeeded > 0 with no failures is COMPLETED regardless of
    # rows_processed.  Resume / coalesce-continuation runs may have
    # rows_processed == 0 with rows_succeeded > 0 (no new source rows
    # ingested, but a restored coalesce flushed successful tokens).
    return RunStatus.COMPLETED
