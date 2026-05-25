"""Tests for RunResult — pipeline execution outcome contract.

Validates __post_init__ guards: empty run_id rejection, require_int on all
numeric fields (negative rejection, bool rejection, float rejection),
and freeze_fields on routed_destinations.
"""

from types import MappingProxyType
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.run_result import RunResult, derive_terminal_run_status
from tests.fixtures.factories import make_run_result


class TestRunResultValidation:
    """__post_init__ guards on RunResult."""

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            RunResult(
                run_id="",
                status=RunStatus.COMPLETED,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        ],
    )
    def test_negative_value_rejected(self, field: str) -> None:
        """Every numeric field must be >= 0."""
        kwargs: dict[str, Any] = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed_success": 0,
            "rows_routed_failure": 0,
            field: -1,
        }
        with pytest.raises(ValueError, match=field):
            RunResult(**kwargs)

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        ],
    )
    def test_bool_rejected(self, field: str) -> None:
        """Bool must not be accepted as int (Python subclass trap)."""
        kwargs: dict[str, Any] = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed_success": 0,
            "rows_routed_failure": 0,
            field: True,
        }
        with pytest.raises(TypeError):
            RunResult(**kwargs)

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
        ],
    )
    def test_float_rejected(self, field: str) -> None:
        """Float must not be silently accepted for int fields."""
        kwargs: dict[str, Any] = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed_success": 0,
            "rows_routed_failure": 0,
            field: 1.5,
        }
        with pytest.raises(TypeError):
            RunResult(**kwargs)


class TestRunResultImmutability:
    """Frozen dataclass + freeze_fields on routed_destinations."""

    def test_routed_destinations_frozen(self) -> None:
        """Dict passed to routed_destinations must be deep-frozen."""
        result = make_run_result(routed_destinations={"sink_a": 5, "sink_b": 3})
        assert isinstance(result.routed_destinations, MappingProxyType)

    def test_routed_destinations_default_is_empty_frozen(self) -> None:
        """Default routed_destinations must be an empty frozen mapping."""
        result = make_run_result()
        assert isinstance(result.routed_destinations, MappingProxyType)
        assert len(result.routed_destinations) == 0

    def test_routed_destinations_mutation_blocked(self) -> None:
        """Callers must not be able to mutate routed_destinations after creation."""
        result = make_run_result(routed_destinations={"sink_a": 5})
        with pytest.raises(TypeError):
            result.routed_destinations["sink_b"] = 10  # type: ignore[index]


class TestRunResultSerialization:
    """to_dict() must produce JSON-serializable output."""

    def test_to_dict_returns_plain_dict(self) -> None:
        result = make_run_result(routed_destinations={"sink_a": 5, "sink_b": 3})
        d = result.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["routed_destinations"], dict)
        assert d["routed_destinations"] == {"sink_a": 5, "sink_b": 3}

    def test_to_dict_status_is_string(self) -> None:
        result = make_run_result()
        d = result.to_dict()
        assert isinstance(d["status"], str)
        assert d["status"] == "completed"

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        result = make_run_result(routed_destinations={"sink_a": 5})
        json.dumps(result.to_dict())  # must not raise

    def test_to_dict_empty_routed_destinations(self) -> None:
        result = make_run_result()
        d = result.to_dict()
        assert d["routed_destinations"] == {}


class TestRunResultFactory:
    """Tests for the make_run_result factory (ensures factory is usable)."""

    def test_factory_defaults_produce_valid_result(self) -> None:
        result = make_run_result()
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10

    def test_factory_accepts_all_overrides(self) -> None:
        result = make_run_result(
            run_id="custom-run",
            status=RunStatus.FAILED,
            rows_processed=100,
            rows_succeeded=5,
            rows_failed=100,
            rows_routed_success=5,
            rows_routed_failure=0,
            rows_quarantined=2,
            rows_forked=3,
            rows_coalesced=1,
            rows_coalesce_failed=1,
            rows_expanded=4,
            rows_buffered=2,
            routed_destinations={"x": 5},
        )
        assert result.run_id == "custom-run"
        assert result.status == RunStatus.FAILED
        assert result.rows_failed == 100
        assert result.routed_destinations["x"] == 5


class TestRunResultStatusInvariant:
    """Phase 2.2 (elspeth-0de989c56d) — biconditional invariant linking
    ``status`` to row-count shape.

    The presence-indicator predicate (see issue comment 693) maps row-count
    shapes onto the four-value terminal taxonomy:

    +-------------------------+----------------------------------------------+
    | Status                  | Required shape                               |
    +=========================+==============================================+
        | ``COMPLETED``           | rows_succeeded>0, rows_failed=0,             |
        |                         | rows_coalesce_failed=0                       |
    +-------------------------+----------------------------------------------+
        | ``COMPLETED_WITH_FAIL`` | rows_succeeded>0 and at least one of         |
        | ``URES``                | rows_failed / rows_coalesce_failed > 0       |
    +-------------------------+----------------------------------------------+
    | ``FAILED``              | rows_succeeded == 0 (rows_processed any)     |
    +-------------------------+----------------------------------------------+
        | ``EMPTY``               | rows_processed==0, rows_succeeded=0,         |
        |                         | rows_failed=0,                               |
    |                         | rows_coalesce_failed=0                       |
    +-------------------------+----------------------------------------------+

    Non-terminal (``RUNNING``) and signal-bounded (``INTERRUPTED``) statuses
    bypass the biconditional — partial-progress shapes are intentionally
    representable for the resume / SIGINT / SIGTERM control-flow paths.
    """

    @staticmethod
    def _build(**overrides: Any) -> RunResult:
        kwargs: dict[str, Any] = {
            "run_id": "test-run",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed_success": 0,
            "rows_routed_failure": 0,
            "rows_quarantined": 0,
        }
        kwargs.update(overrides)
        return RunResult(**kwargs)

    # -- Legal terminal shapes -----------------------------------------------

    def test_completed_clean_run(self) -> None:
        """Healthy linear pipeline: rows_processed == rows_succeeded, no failures."""
        result = self._build(status=RunStatus.COMPLETED, rows_processed=10, rows_succeeded=10)
        assert result.status == RunStatus.COMPLETED

    def test_completed_aggregation_run(self) -> None:
        """batch_stats aggregation: 6 source rows -> 1 emitted aggregated row.

        rows_processed > rows_succeeded is intentional (CONSUMED_IN_BATCH on
        source rows is non-counting; the flush emits one COMPLETED).  The
        presence-indicator predicate accepts this shape because rows_failed
        == 0 — i.e., there were no failures, just consumption-into-batch.
        Equality predicates would falsely report this as
        completed_with_failures.
        """
        result = self._build(status=RunStatus.COMPLETED, rows_processed=6, rows_succeeded=1)
        assert result.status == RunStatus.COMPLETED

    def test_completed_with_failures_mixed_run(self) -> None:
        """Some rows succeeded, some failed."""
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_completed_with_failures_via_quarantine(self) -> None:
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
            rows_quarantined=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_completed_with_failures_via_coalesce_failed(self) -> None:
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=7,
            rows_coalesce_failed=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_failed_s1b_msg2_reproducer(self) -> None:
        """S1B msg2 reproducer: rows_succeeded=0, rows_failed=6, no on_error."""
        result = self._build(status=RunStatus.FAILED, rows_processed=6, rows_succeeded=0, rows_failed=6)
        assert result.status == RunStatus.FAILED

    def test_failed_with_zero_rows_processed(self) -> None:
        """Engine took the failed path before any row was counted (e.g. an
        exception during source iteration).  rows_processed=0 with status=FAILED
        is legitimate for the "exception bounded the run" case — see the
        ``_emit_failed_ceremony`` default-result path in
        engine/orchestrator/core.py.
        """
        result = self._build(status=RunStatus.FAILED, rows_processed=0, rows_succeeded=0)
        assert result.status == RunStatus.FAILED

    def test_empty_clean_run(self) -> None:
        """Source produced no rows, no failures."""
        result = self._build(status=RunStatus.EMPTY, rows_processed=0, rows_succeeded=0)
        assert result.status == RunStatus.EMPTY

    # -- Non-terminal / signal-bounded statuses bypass the biconditional ----

    def test_running_status_is_unconstrained(self) -> None:
        """RUNNING is a non-terminal, mid-flight status — partial counters legal."""
        result = self._build(status=RunStatus.RUNNING, rows_processed=5, rows_succeeded=3, rows_failed=1)
        assert result.status == RunStatus.RUNNING

    def test_interrupted_status_is_unconstrained(self) -> None:
        """INTERRUPTED is signal-bounded (SIGINT/SIGTERM) — any partial shape allowed."""
        result = self._build(status=RunStatus.INTERRUPTED, rows_processed=5, rows_succeeded=2, rows_failed=1)
        assert result.status == RunStatus.INTERRUPTED

    # -- Illegal shapes (must crash at construction) ------------------------

    def test_completed_with_zero_rows_processed_and_positive_succeeded_legal(self) -> None:
        """Resume / coalesce-continuation shape: rows_processed=0 (no new
        source rows ingested) but rows_succeeded>0 (restored coalesce
        flushed a successful token).  COMPLETED is the right label —
        operationally the run produced output.
        """
        result = self._build(status=RunStatus.COMPLETED, rows_processed=0, rows_succeeded=1)
        assert result.status == RunStatus.COMPLETED

    def test_completed_rejects_zero_rows_succeeded(self) -> None:
        """COMPLETED requires a clean terminal indicator —
        rows_succeeded > 0 OR rows_quarantined > 0.
        """
        with pytest.raises(ValueError, match=r"COMPLETED.*requires a clean terminal indicator"):
            self._build(status=RunStatus.COMPLETED, rows_processed=5, rows_succeeded=0)

    def test_completed_rejects_failures(self) -> None:
        """COMPLETED forbids uncaught rows_failed > 0 (use COMPLETED_WITH_FAILURES)."""
        with pytest.raises(ValueError, match=r"COMPLETED.*requires no uncaught failures"):
            self._build(status=RunStatus.COMPLETED, rows_processed=10, rows_succeeded=7, rows_failed=3)

    def test_completed_rejects_quarantine(self) -> None:
        """Quarantine is a clean terminal outcome but still warrants the
        COMPLETED_WITH_FAILURES label per CLAUDE.md Tier-3 manifesto — the
        operator should see "failures" on a run that quarantined any row.
        """
        with pytest.raises(ValueError, match=r"COMPLETED.*requires no quarantined rows"):
            self._build(
                status=RunStatus.COMPLETED,
                rows_processed=10,
                rows_succeeded=7,
                rows_failed=3,
                rows_quarantined=3,
            )

    def test_completed_with_failures_rejects_zero_succeeded(self) -> None:
        """COMPLETED_WITH_FAILURES requires a clean terminal indicator —
        rows_succeeded > 0 OR rows_quarantined > 0. With both at zero and
        only uncaught failures, the run is FAILED, not
        COMPLETED_WITH_FAILURES.
        """
        with pytest.raises(ValueError, match=r"COMPLETED_WITH_FAILURES.*requires a clean terminal indicator"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=6,
                rows_succeeded=0,
                rows_failed=6,
            )

    def test_completed_with_failures_rejects_no_failures(self) -> None:
        """COMPLETED_WITH_FAILURES requires at least one failure indicator."""
        with pytest.raises(ValueError, match=r"COMPLETED_WITH_FAILURES.*requires.*failure"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
            )

    def test_failed_tolerates_partial_successes_for_exception_bounded_runs(self) -> None:
        """FAILED has two origins: the predicate decided rows_succeeded==0,
        or an exception bounded the run with partial successes already
        counted (orchestrator ``_RunFailedWithPartialResultError`` path).
        The biconditional tolerates any rows_succeeded shape under FAILED;
        the predicate picks COMPLETED_WITH_FAILURES on the success path
        when rows_succeeded > 0 alongside failures, so this relaxation
        does NOT bypass the operator-discriminating taxonomy.
        """
        result = self._build(status=RunStatus.FAILED, rows_processed=10, rows_succeeded=5, rows_failed=5)
        assert result.status == RunStatus.FAILED

    def test_empty_rejects_nonzero_processed(self) -> None:
        """EMPTY requires rows_processed == 0."""
        with pytest.raises(ValueError, match=r"EMPTY.*rows_processed == 0"):
            self._build(status=RunStatus.EMPTY, rows_processed=5, rows_succeeded=0)

    def test_empty_rejects_nonzero_succeeded(self) -> None:
        # Post-quarantine-promotion (run-status policy revision): the EMPTY
        # guard now rejects any clean terminal indicator — rows_succeeded > 0
        # OR rows_quarantined > 0 both contradict EMPTY semantics.
        with pytest.raises(ValueError, match=r"EMPTY.*requires no clean terminal indicator"):
            self._build(status=RunStatus.EMPTY, rows_processed=0, rows_succeeded=1)

    def test_empty_rejects_failures(self) -> None:
        """EMPTY forbids any failure indicator (use FAILED if there were failures)."""
        with pytest.raises(ValueError, match=r"EMPTY.*requires no failures"):
            self._build(status=RunStatus.EMPTY, rows_processed=0, rows_succeeded=0, rows_failed=1)


class TestRunStatusRowsRoutedSplitPredicate:
    """elspeth-5069612f3c / ADR-019 — predicate behavior for split routed counters.

    The routed counters are reporting subsets. ``rows_succeeded`` and
    ``rows_failed`` remain the lifecycle counters that drive the status
    predicate; rows_routed_* explain which of those terminal rows used MOVE or
    DIVERT paths.

    REPLACES the older test_runstatus_rows_routed_only_classifies_as_failed
    pattern at lines 289-295 of this file, which asserted FAILED for the
    structurally ambiguous rows_routed counter (now removed).
    """

    def _build(
        self,
        *,
        status: RunStatus,
        rows_processed: int = 0,
        rows_succeeded: int = 0,
        rows_failed: int = 0,
        rows_routed_success: int = 0,
        rows_routed_failure: int = 0,
        rows_quarantined: int = 0,
        rows_coalesce_failed: int = 0,
    ) -> RunResult:
        return RunResult(
            run_id="rsp-1",
            status=status,
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed_success=rows_routed_success,
            rows_routed_failure=rows_routed_failure,
            rows_quarantined=rows_quarantined,
            rows_coalesce_failed=rows_coalesce_failed,
        )

    def test_gate_routed_only_classifies_as_completed(self) -> None:
        """User reproducer shape: csv -> gate -> sink_a/sink_b, every row
        intentionally gate-routed (MOVE). ``rows_routed_success`` reports that
        all successful rows used a gate route.
        """
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=8,
            rows_failed=0,
            rows_routed_success=8,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED
        # The biconditional invariant accepts routed success as a success subset.
        result = self._build(
            status=RunStatus.COMPLETED,
            rows_processed=8,
            rows_succeeded=8,
            rows_routed_success=8,
        )
        assert result.status == RunStatus.COMPLETED

    def test_on_error_routed_only_classifies_as_failed(self) -> None:
        """S1A reproducer shape: every row triggers a transform exception, all
        routed via on_error to a quarantine/error sink (DIVERT).
        ``rows_routed_failure`` reports that all failed rows used on_error.
        """
        derived = derive_terminal_run_status(
            rows_processed=2,
            rows_succeeded=0,
            rows_failed=2,
            rows_routed_success=0,
            rows_routed_failure=2,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.FAILED

    def test_mixed_gate_and_on_error_classifies_as_completed_with_failures(self) -> None:
        """Mixed shape: some rows gate-routed (success) AND some rows
        on_error-routed (failure). Predicate must report
        COMPLETED_WITH_FAILURES — at least one success indicator AND at least
        one failure indicator.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
            rows_routed_success=7,
            rows_routed_failure=3,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES

    def test_empty_pipeline_still_classifies_as_empty(self) -> None:
        """Regression: empty source (rows_processed == 0, no failure
        indicator) still classifies as EMPTY after the split.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.EMPTY

    def test_resume_continuation_still_classifies_as_completed(self) -> None:
        """Regression: resume / coalesce-continuation shape (rows_processed == 0
        AND rows_succeeded > 0) still classifies as COMPLETED after the split.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=3,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED

    def test_resume_continuation_with_success_and_failure_indicators_classifies_as_completed_with_failures(self) -> None:
        """Regression for resume-after-coalesce shapes: rows_processed can be 0
        while continuation bookkeeping reports both a success indicator and a
        failure indicator. derive_terminal_run_status() and the L0
        _check_status_invariant must agree on COMPLETED_WITH_FAILURES without
        requiring rows_processed > 0.
        """
        derived = derive_terminal_run_status(
            rows_processed=0,
            rows_succeeded=3,
            rows_failed=1,
            rows_routed_success=0,
            rows_routed_failure=1,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=0,
            rows_succeeded=3,
            rows_failed=1,
            rows_routed_failure=1,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    # ------------------------------------------------------------------
    # Additional positive-shape coverage (elspeth-5069612f3c review pass)
    # The six canonical shapes above cover the user-facing reproducer
    # scenarios and the resume-after-coalesce zero-processed mixed-indicator
    # regression. These three additional shapes pin mixed-counter cases
    # the canonical set leaves under-tested. Without them the predicate
    # could regress on operationally-common shapes (success-path-with-
    # on-error-failures, gate-routed-with-hard-failures, etc.) without
    # any test catching it.
    # ------------------------------------------------------------------

    def test_succeeded_mixed_with_on_error_routing_classifies_as_completed_with_failures(self) -> None:
        """Mixed-success shape: some rows reached on_success success-path sinks
        (rows_succeeded > 0) while others triggered transform exceptions and
        were on_error-routed (rows_routed_failure > 0). Predicate must report
        COMPLETED_WITH_FAILURES — both indicators are present.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
            rows_routed_success=0,
            rows_routed_failure=3,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        # Verify the L0 invariant accepts this shape without raising.
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
            rows_routed_failure=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_gate_routed_mixed_with_hard_failures_classifies_as_completed_with_failures(self) -> None:
        """Mixed-routing shape: some rows gate-routed via MOVE
        (rows_routed_success > 0) while others reached the canonical FAILED
        terminal via transform exceptions that were NOT on_error-rerouted
        (rows_failed > 0). Both indicators present; predicate is
        COMPLETED_WITH_FAILURES.
        """
        derived = derive_terminal_run_status(
            rows_processed=10,
            rows_succeeded=6,
            rows_failed=4,
            rows_routed_success=6,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        # Verify the L0 invariant accepts this shape without raising.
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=10,
            rows_succeeded=6,
            rows_failed=4,
            rows_routed_success=6,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_canonical_failed_via_rows_failed_only_classifies_as_failed(self) -> None:
        """Canonical FAILED shape: every row reached FAILURE/UNROUTED via an
        unhandled transform exception (no on_error reroute, no gate routing).
        rows_failed > 0 is the sole failure indicator; predicate is FAILED.

        This pins the legacy FAILED path that pre-existed the rows_routed
        split — without this test, a regression that only checked the new
        rows_routed_failure indicator could silently bypass rows_failed.
        """
        derived = derive_terminal_run_status(
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.FAILED

    # ------------------------------------------------------------------
    # Predicate matrix coverage for old failure counters crossed with the
    # new routed counters. These are cheap but important: rows_quarantined
    # and rows_coalesce_failed are pre-existing failure indicators, and the
    # rows_routed_success / rows_routed_failure split must compose with
    # them exactly like rows_failed does.
    # ------------------------------------------------------------------

    def test_gate_routed_with_quarantined_rows_classifies_as_completed_with_failures(self) -> None:
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=5,
            rows_failed=3,
            rows_routed_success=5,
            rows_routed_failure=0,
            rows_quarantined=3,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=8,
            rows_succeeded=5,
            rows_failed=3,
            rows_routed_success=5,
            rows_quarantined=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_gate_routed_with_coalesce_failures_classifies_as_completed_with_failures(self) -> None:
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=5,
            rows_failed=0,
            rows_routed_success=5,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_coalesce_failed=3,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=8,
            rows_succeeded=5,
            rows_routed_success=5,
            rows_coalesce_failed=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_on_error_routed_with_quarantine_and_coalesce_failures_classifies_as_completed_with_failures(self) -> None:
        """Post-quarantine-promotion: quarantine is a clean terminal outcome
        per CLAUDE.md Tier-3 manifesto. With ``rows_quarantined=2`` the
        pipeline made a clean determination on two rows, satisfying the
        ``terminal_clean_indicator``. The remaining 4 uncaught
        ``rows_failed`` (6 - 2 quarantined = 4) and 2 ``rows_coalesce_failed``
        still constitute a ``failure_indicator``. Both indicators present →
        COMPLETED_WITH_FAILURES, not FAILED.
        """
        derived = derive_terminal_run_status(
            rows_processed=8,
            rows_succeeded=0,
            rows_failed=6,
            rows_routed_success=0,
            rows_routed_failure=4,
            rows_quarantined=2,
            rows_coalesce_failed=2,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=8,
            rows_failed=6,
            rows_routed_failure=4,
            rows_quarantined=2,
            rows_coalesce_failed=2,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    # ------------------------------------------------------------------
    # Quarantine-promotion regression coverage (run-status policy
    # revision: quarantine is a clean terminal outcome per CLAUDE.md
    # Tier-3 manifesto, not a failure). Three canonical shapes:
    #   1) all rows quarantined cleanly -> COMPLETED_WITH_FAILURES
    #      (was FAILED; bug the policy revision fixed).
    #   2) mixed quarantine + uncaught failures, zero succeeded ->
    #      FAILED still — quarantine doesn't offset uncaught failure
    #      when no row reached an actual success path.
    #   3) succeeded + quarantine + uncaught failures unchanged from
    #      pre-revision behaviour -> COMPLETED_WITH_FAILURES.
    # ------------------------------------------------------------------

    def test_all_rows_quarantined_classifies_as_completed_with_failures(self) -> None:
        """Canonical bug: a pipeline that cleanly quarantines every row was
        classified FAILED. Per CLAUDE.md Tier-3 data manifesto quarantine is
        a deliberate clean terminal outcome ("row 42 was quarantined because
        field X was NULL" is legitimate audit evidence, not a framework
        failure). The pipeline made a clean determination on every row, so
        the verdict is COMPLETED_WITH_FAILURES — not FAILED.
        """
        derived = derive_terminal_run_status(
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=5,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_quarantined=5,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_partial_quarantine_with_uncaught_failures_zero_succeeded_classifies_as_completed_with_failures(self) -> None:
        """Mixed quarantine + uncaught failures with zero succeeded rows.
        ``rows_quarantined=3`` satisfies the clean terminal indicator, and
        the remaining ``rows_failed - rows_quarantined = 2`` is an uncaught
        failure indicator. Both indicators present →
        COMPLETED_WITH_FAILURES. The pipeline made a clean determination on
        the 3 quarantined rows, AND had 2 uncaught failures — the operator
        sees both signals via the "with failures" terminal label.

        (Contrast with the all-quarantined case where ``failure_indicator``
        is False and the policy lifts the verdict from FAILED to
        COMPLETED_WITH_FAILURES via the
        ``rows_quarantined > 0`` arm of the predicate.)
        """
        derived = derive_terminal_run_status(
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=3,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=5,
            rows_succeeded=0,
            rows_failed=5,
            rows_quarantined=3,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    def test_succeeded_with_partial_quarantine_classifies_as_completed_with_failures(self) -> None:
        """Pre-revision behaviour preserved: some rows succeeded, some
        quarantined, no uncaught failures still classifies as
        COMPLETED_WITH_FAILURES (the quarantine itself is the failure-like
        indicator that lifts the verdict above COMPLETED).
        """
        derived = derive_terminal_run_status(
            rows_processed=3,
            rows_succeeded=2,
            rows_failed=1,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=1,
            rows_coalesce_failed=0,
        )
        assert derived == RunStatus.COMPLETED_WITH_FAILURES
        result = self._build(
            status=RunStatus.COMPLETED_WITH_FAILURES,
            rows_processed=3,
            rows_succeeded=2,
            rows_failed=1,
            rows_quarantined=1,
        )
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES

    @given(
        rows_processed=st.integers(min_value=0, max_value=20),
        rows_succeeded=st.integers(min_value=0, max_value=20),
        rows_failed=st.integers(min_value=0, max_value=20),
        rows_routed_success=st.integers(min_value=0, max_value=20),
        rows_routed_failure=st.integers(min_value=0, max_value=20),
        rows_quarantined=st.integers(min_value=0, max_value=20),
        rows_coalesce_failed=st.integers(min_value=0, max_value=20),
    )
    def test_derived_status_round_trips_l0_invariant(
        self,
        rows_processed: int,
        rows_succeeded: int,
        rows_failed: int,
        rows_routed_success: int,
        rows_routed_failure: int,
        rows_quarantined: int,
        rows_coalesce_failed: int,
    ) -> None:
        """Biconditional property: any counter tuple classified by
        derive_terminal_run_status() must be accepted by RunResult's L0
        status invariant when used with the derived status.

        This is the cheapest guard against the mirror-drift class that
        produced elspeth-71520f5e30: the predicate function and the
        dataclass invariant must agree for arbitrary non-negative counter
        tuples, including the new routed counters crossed with legacy
        failure counters.
        """
        counters = {
            "rows_processed": rows_processed,
            "rows_succeeded": rows_succeeded,
            "rows_failed": rows_failed,
            "rows_routed_success": rows_routed_success,
            "rows_routed_failure": rows_routed_failure,
            "rows_quarantined": rows_quarantined,
            "rows_coalesce_failed": rows_coalesce_failed,
        }
        assume(rows_routed_success <= rows_succeeded)
        assume(rows_routed_failure <= rows_failed)
        assume(rows_quarantined <= rows_failed)
        derived = derive_terminal_run_status(**counters)
        result = self._build(status=derived, **counters)
        assert result.status == derived

    # ------------------------------------------------------------------
    # Negative invariant coverage (elspeth-5069612f3c review pass)
    # The updated _check_status_invariant has seven raise-paths. Without
    # negative tests, a future regression that relaxes the invariant
    # (admits a shape it should reject) passes every positive test
    # silently. These six negative tests pin the most consequential
    # raise-paths so loosened-invariant regressions are caught.
    # ------------------------------------------------------------------

    def test_completed_without_success_indicator_raises(self) -> None:
        """COMPLETED requires a clean terminal indicator (rows_succeeded > 0
        OR rows_quarantined > 0). With all clean counters at zero AND no
        failure indicator, the run should classify as EMPTY
        (rows_processed == 0) or FAILED (rows_processed > 0) — NOT
        COMPLETED. The invariant must reject this construction.
        """
        with pytest.raises(ValueError, match="status=COMPLETED requires a clean terminal indicator"):
            self._build(
                status=RunStatus.COMPLETED,
                rows_processed=5,
                rows_succeeded=0,
                rows_routed_success=0,
            )

    def test_completed_with_failure_indicator_raises(self) -> None:
        """COMPLETED requires NO uncaught failure indicator. If any uncaught
        failure counter is non-zero, the status is COMPLETED_WITH_FAILURES,
        not COMPLETED.
        """
        with pytest.raises(ValueError, match="status=COMPLETED requires no uncaught failures"):
            self._build(
                status=RunStatus.COMPLETED,
                rows_processed=5,
                rows_succeeded=4,
                rows_failed=1,  # Uncaught failure must trigger COMPLETED_WITH_FAILURES, not COMPLETED.
            )

    def test_completed_with_failures_without_clean_terminal_indicator_raises(self) -> None:
        """COMPLETED_WITH_FAILURES requires a clean terminal indicator
        (rows_succeeded > 0 OR rows_quarantined > 0). With only uncaught
        failures present and no clean terminal, the status must be FAILED,
        not COMPLETED_WITH_FAILURES.
        """
        with pytest.raises(ValueError, match="COMPLETED_WITH_FAILURES requires a clean terminal indicator"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=3,
                rows_failed=3,
            )

    def test_completed_with_failures_without_failure_indicator_raises(self) -> None:
        """COMPLETED_WITH_FAILURES requires at least one failure-like
        indicator: uncaught failure (rows_failed - rows_quarantined > 0),
        rows_coalesce_failed > 0, OR rows_quarantined > 0 (quarantine is a
        clean terminal outcome but still qualifies as a failure-like
        indicator that warrants the operator-visible failures label). With
        only succeeded rows and no quarantine the status must be COMPLETED.
        """
        with pytest.raises(ValueError, match="COMPLETED_WITH_FAILURES requires at least one failure-like indicator"):
            self._build(
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=3,
                rows_succeeded=3,
            )

    def test_empty_with_rows_processed_raises(self) -> None:
        """EMPTY requires rows_processed == 0 (no input rows reached the
        engine). Any non-zero rows_processed contradicts EMPTY semantics.
        """
        with pytest.raises(ValueError, match="status=EMPTY requires rows_processed == 0"):
            self._build(
                status=RunStatus.EMPTY,
                rows_processed=1,  # Contradicts EMPTY.
            )

    def test_empty_with_success_indicator_raises(self) -> None:
        """EMPTY requires no clean terminal indicator. A run with
        rows_succeeded > 0 OR rows_quarantined > 0 is not EMPTY by
        definition.
        """
        with pytest.raises(ValueError, match="status=EMPTY requires no clean terminal indicator"):
            self._build(
                status=RunStatus.EMPTY,
                rows_processed=0,
                rows_succeeded=1,
                rows_routed_success=1,  # Clean terminal indicator contradicts EMPTY.
            )
