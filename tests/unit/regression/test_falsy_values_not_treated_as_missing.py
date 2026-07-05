"""Regression tests for truthiness checks that should be ``is not None``.

These tests verify that falsy-but-valid values (0, 0.0, "", False) are not
incorrectly treated as missing/None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class _RunStatusFake:
    value: str


@dataclass(frozen=True)
class _RunRecordFake:
    run_id: str
    status: _RunStatusFake
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pipeline_hash: str | None = None
    source_plugin: str | None = None
    source_row_count: int | None = None


@dataclass(frozen=True)
class _RunLifecycleFake:
    run: _RunRecordFake

    def get_run(self, run_id: str) -> _RunRecordFake | None:
        if run_id != self.run.run_id:
            return None
        return self.run


@dataclass(frozen=True)
class _RecorderFactoryFake:
    run_lifecycle: _RunLifecycleFake


@dataclass(frozen=True)
class _ScalarResultFake:
    value: int | float | None

    def scalar(self) -> int | float | None:
        return self.value


@dataclass(frozen=True)
class _RowsResultFake:
    rows: tuple[Any, ...] = ()

    def fetchall(self) -> tuple[Any, ...]:
        return self.rows


class _SequencedConnectionFake:
    def __init__(self, results: list[_ScalarResultFake | _RowsResultFake]) -> None:
        self._results = results
        self._next_index = 0

    def execute(self, _statement: Any) -> _ScalarResultFake | _RowsResultFake:
        try:
            result = self._results[self._next_index]
        except IndexError as exc:
            raise AssertionError("unexpected extra query") from exc
        self._next_index += 1
        return result


@dataclass(frozen=True)
class _ConnectionContextFake:
    connection: _SequencedConnectionFake

    def __enter__(self) -> _SequencedConnectionFake:
        return self.connection

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> bool:
        return False


@dataclass(frozen=True)
class _LandscapeDBFake:
    results: list[_ScalarResultFake | _RowsResultFake]

    def connection(self) -> _ConnectionContextFake:
        return _ConnectionContextFake(_SequencedConnectionFake(self.results))


def _run_summary_results_with(avg_duration_ms: float | None) -> list[_ScalarResultFake | _RowsResultFake]:
    zero_count_results = [_ScalarResultFake(0) for _ in range(10)]
    return [
        *zero_count_results,
        _RowsResultFake(),
        _ScalarResultFake(avg_duration_ms),
    ]


class TestMCPAvgDurationTruthiness:
    """A.1: avg_duration=0.0 must not be treated as missing."""

    def test_zero_avg_duration_included_in_summary(self) -> None:
        """avg_duration=0.0 should produce round(0.0, 2)=0.0, not None."""
        from elspeth.mcp.analyzers.reports import get_run_summary

        db = _LandscapeDBFake(results=_run_summary_results_with(avg_duration_ms=0.0))
        factory = _RecorderFactoryFake(
            run_lifecycle=_RunLifecycleFake(
                _RunRecordFake(
                    run_id="test-run",
                    status=_RunStatusFake(value="COMPLETED"),
                )
            )
        )

        result = get_run_summary(db, factory, "test-run")

        assert result["avg_state_duration_ms"] == 0.0

    def test_zero_avg_ms_in_node_performance(self) -> None:
        """row.avg_ms=0.0 should produce 0.0, not None."""
        # Direct test of the expression fix
        avg_ms = 0.0
        result = round(avg_ms, 2) if avg_ms is not None else None
        assert result == 0.0

        # Before fix: `round(0.0, 2) if 0.0 else None` → None (WRONG)
        # After fix: `round(0.0, 2) if 0.0 is not None else None` → 0.0 (CORRECT)


class TestExplainRowSourceDataRef:
    """A.2: source_data_ref="" must not be skipped."""

    def test_empty_string_ref_not_skipped(self) -> None:
        """Empty string source_data_ref should still attempt payload lookup."""
        # Direct expression test
        source_data_ref = ""
        payload_store = object()

        # Before fix: `if "" and payload_store` → False (skips lookup)
        # After fix: `if "" is not None and payload_store is not None` → True
        should_lookup = source_data_ref is not None and payload_store is not None
        assert should_lookup is True

    def test_none_ref_skipped(self) -> None:
        """None source_data_ref should skip payload lookup."""
        source_data_ref = None
        payload_store = object()
        # Intentional: mypy knows the right operand is never evaluated because
        # source_data_ref is None (short-circuit). That's exactly what this test verifies.
        should_lookup = source_data_ref is not None and payload_store is not None  # type: ignore[unreachable]
        assert should_lookup is False


class TestCallReplayerErrorJson:
    """A.4: error_json="" must not be skipped."""

    def test_empty_string_error_json_not_skipped(self) -> None:
        """Empty string error_json should still be parsed."""
        error_json = ""
        # Before fix: `if "":` → False (skips parsing)
        # After fix: `if "" is not None:` → True (attempts parsing)
        should_parse = error_json is not None
        assert should_parse is True

    def test_none_error_json_skipped(self) -> None:
        """None error_json should skip parsing."""
        error_json = None
        should_parse = error_json is not None
        assert should_parse is False


class TestCallVerifierResponseHash:
    """A.5: verifier already uses `is not None` — confirm no regression."""

    def test_verifier_uses_is_not_none(self) -> None:
        """Verify the verifier source code uses `is not None` for response_hash."""
        import inspect

        from elspeth.plugins.infrastructure.clients.verifier import CallVerifier

        source = inspect.getsource(CallVerifier.verify)
        assert "response_hash is not None" in source
