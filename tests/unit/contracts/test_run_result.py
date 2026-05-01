"""Tests for RunResult — pipeline execution outcome contract.

Validates __post_init__ guards: empty run_id rejection, require_int on all
numeric fields (negative rejection, bool rejection, float rejection),
and freeze_fields on routed_destinations.
"""

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.run_result import RunResult
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
                rows_routed=0,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed",
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
            "rows_routed": 0,
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
            "rows_routed",
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
            "rows_routed": 0,
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
            "rows_routed",
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
            "rows_routed": 0,
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
            rows_succeeded=0,
            rows_failed=100,
            rows_routed=5,
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
    | ``COMPLETED``           | rows_processed>0, rows_succeeded>0,          |
    |                         | rows_failed=0, rows_quarantined=0,           |
    |                         | rows_coalesce_failed=0                       |
    +-------------------------+----------------------------------------------+
    | ``COMPLETED_WITH_FAIL`` | rows_processed>0, rows_succeeded>0,          |
    | ``URES``                | at least one of (rows_failed/rows_quarantined|
    |                         | /rows_coalesce_failed) > 0                   |
    +-------------------------+----------------------------------------------+
    | ``FAILED``              | rows_succeeded == 0 (rows_processed any)     |
    +-------------------------+----------------------------------------------+
    | ``EMPTY``               | rows_processed==0, rows_succeeded==0,        |
    |                         | rows_failed=0, rows_quarantined=0,           |
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
            "rows_routed": 0,
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

    def test_failed_s1a_reproducer(self) -> None:
        """S1A reproducer (notes/composer-llm-eval-2026-05-01.md):
        rows_processed=6, rows_succeeded=0, rows_routed=6, rows_failed=0.

        Every row took the on_error DIVERT to a user-named quarantine sink.
        rows_routed is structurally ambiguous (DIVERT vs MOVE) so the
        predicate excludes it; rows_succeeded == 0 maps to FAILED.
        """
        result = self._build(status=RunStatus.FAILED, rows_processed=6, rows_succeeded=0, rows_routed=6)
        assert result.status == RunStatus.FAILED

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
        """COMPLETED requires rows_succeeded > 0."""
        with pytest.raises(ValueError, match=r"COMPLETED.*rows_succeeded > 0"):
            self._build(status=RunStatus.COMPLETED, rows_processed=5, rows_succeeded=0)

    def test_completed_rejects_failures(self) -> None:
        """COMPLETED forbids rows_failed > 0 (use COMPLETED_WITH_FAILURES)."""
        with pytest.raises(ValueError, match=r"COMPLETED.*requires no failures"):
            self._build(status=RunStatus.COMPLETED, rows_processed=10, rows_succeeded=7, rows_failed=3)

    def test_completed_rejects_quarantine(self) -> None:
        with pytest.raises(ValueError, match=r"COMPLETED.*requires no failures"):
            self._build(status=RunStatus.COMPLETED, rows_processed=10, rows_succeeded=7, rows_quarantined=3)

    def test_completed_with_failures_rejects_zero_succeeded(self) -> None:
        """COMPLETED_WITH_FAILURES requires rows_succeeded > 0 (else it's FAILED)."""
        with pytest.raises(ValueError, match=r"COMPLETED_WITH_FAILURES.*rows_succeeded > 0"):
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
        with pytest.raises(ValueError, match=r"EMPTY.*rows_succeeded == 0"):
            self._build(status=RunStatus.EMPTY, rows_processed=0, rows_succeeded=1)

    def test_empty_rejects_failures(self) -> None:
        """EMPTY forbids any failure indicator (use FAILED if there were failures)."""
        with pytest.raises(ValueError, match=r"EMPTY.*requires no failures"):
            self._build(status=RunStatus.EMPTY, rows_processed=0, rows_succeeded=0, rows_failed=1)
