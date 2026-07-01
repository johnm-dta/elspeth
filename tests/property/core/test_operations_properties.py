# tests/property/core/test_operations_properties.py
"""Property-based tests for operation lifecycle management (track_operation).

The track_operation context manager has 4 exit paths:
1. Normal exit → status="completed"
2. Exception → status="failed"
3. BaseException (KeyboardInterrupt, SystemExit) → status="failed"
4. DB failure in finally → original exception propagates (or DB error if no original)

Properties tested:
- Context restoration: ctx.operation_id always restored regardless of exit path
- Status correctness: Each exit path produces the correct status
- Exception transparency: Original exceptions always propagate unchanged
- Audit integrity: DB failures on successful operations raise (not swallowed)
- Duration non-negativity: duration_ms is always >= 0
- Nesting: Multiple nested operations restore context correctly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.audit import Operation
from elspeth.contracts.plugin_context import PluginContext
from elspeth.core.landscape import ExecutionRepository
from elspeth.core.operations import track_operation

OperationType = Literal["source_load", "sink_write"]

# =============================================================================
# Minimal Mocks for Testing
# =============================================================================


@dataclass
class FakePluginContext:
    """Minimal PluginContext for property tests.

    Only the operation_id field matters for track_operation.
    """

    run_id: str = "test-run"
    config: dict[str, Any] = field(default_factory=dict)
    operation_id: str | None = None


@dataclass
class RecordedCompletion:
    """Captures what was passed to complete_operation."""

    operation_id: str
    status: str
    output_data: dict[str, Any] | None
    error: str | None
    duration_ms: float


class FakeRecorder:
    """Minimal ExecutionRepository that records operations in memory."""

    def __init__(self, *, fail_on_complete: bool = False) -> None:
        self._op_counter = 0
        self.completions: list[RecordedCompletion] = []
        self._fail_on_complete = fail_on_complete

    def begin_operation(
        self,
        run_id: str,
        node_id: str,
        operation_type: OperationType,
        input_data: dict[str, Any] | None = None,
    ) -> Operation:
        self._op_counter += 1
        return Operation(
            operation_id=f"op_{self._op_counter:04d}",
            run_id=run_id,
            node_id=node_id,
            operation_type=operation_type,
            started_at=datetime.now(UTC),
            status="open",
        )

    def complete_operation(
        self,
        operation_id: str,
        status: str,
        output_data: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        if self._fail_on_complete:
            raise RuntimeError("DB write failed")
        self.completions.append(
            RecordedCompletion(
                operation_id=operation_id,
                status=status,
                output_data=output_data,
                error=error,
                duration_ms=duration_ms,
            )
        )


def _as_execution_repo(recorder: FakeRecorder) -> ExecutionRepository:
    """Cast FakeRecorder to ExecutionRepository for track_operation calls."""
    return cast(ExecutionRepository, recorder)


def _as_ctx(ctx: FakePluginContext) -> PluginContext:
    """Cast FakePluginContext to PluginContext for track_operation calls."""
    return cast(PluginContext, ctx)


# =============================================================================
# Strategies
# =============================================================================

# Exception types that are subclasses of Exception (not BaseException)
standard_exceptions = st.sampled_from(
    [
        ValueError("test value error"),
        TypeError("test type error"),
        RuntimeError("test runtime error"),
        OSError("test io error"),
        KeyError("test key error"),
        AttributeError("test attribute error"),
        OSError("test os error"),
        ConnectionError("test connection error"),
    ]
)

operation_types: st.SearchStrategy[OperationType] = st.sampled_from(["source_load", "sink_write"])

# Previous operation IDs (None or a string)
previous_op_ids = st.one_of(st.none(), st.text(min_size=5, max_size=20, alphabet="abcdef0123456789"))


# =============================================================================
# Context Restoration Properties
# =============================================================================


class TestContextRestorationProperties:
    """ctx.operation_id must be restored on ALL exit paths."""

    @given(prev_op_id=previous_op_ids)
    @settings(max_examples=100)
    def test_normal_exit_restores_context(self, prev_op_id: str | None) -> None:
        """Property: Normal exit restores previous operation_id."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id=prev_op_id)

        with track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
            # During operation, ctx.operation_id is set to new operation
            assert ctx.operation_id is not None
            assert ctx.operation_id != prev_op_id

        # After exit, ctx.operation_id is restored
        assert ctx.operation_id == prev_op_id

    @given(prev_op_id=previous_op_ids, exc=standard_exceptions)
    @settings(max_examples=100)
    def test_exception_exit_restores_context(self, prev_op_id: str | None, exc: Exception) -> None:
        """Property: Exception exit restores previous operation_id."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id=prev_op_id)

        with pytest.raises(type(exc)), track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
            raise exc

        assert ctx.operation_id == prev_op_id

    @given(prev_op_id=previous_op_ids)
    @settings(max_examples=50)
    def test_keyboard_interrupt_restores_context(self, prev_op_id: str | None) -> None:
        """Property: KeyboardInterrupt (BaseException) restores context."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id=prev_op_id)

        with (
            pytest.raises(KeyboardInterrupt),
            track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)),
        ):
            raise KeyboardInterrupt()

        assert ctx.operation_id == prev_op_id

    @given(prev_op_id=previous_op_ids)
    @settings(max_examples=50)
    def test_db_failure_restores_context(self, prev_op_id: str | None) -> None:
        """Property: DB failure during complete_operation still restores context."""
        recorder = FakeRecorder(fail_on_complete=True)
        ctx = FakePluginContext(operation_id=prev_op_id)

        # DB failure on successful operation → DB error propagates
        with (
            pytest.raises(RuntimeError, match="DB write failed"),
            track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)),
        ):
            pass  # Operation "succeeds"

        assert ctx.operation_id == prev_op_id


# =============================================================================
# Status Correctness Properties
# =============================================================================


class TestStatusCorrectnessProperties:
    """Each exit path must produce the correct operation status."""

    @given(op_type=operation_types)
    @settings(max_examples=50)
    def test_normal_exit_status_completed(self, op_type: OperationType) -> None:
        """Property: Normal exit records status='completed'."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with track_operation(_as_execution_repo(recorder), "run-1", "node-1", op_type, _as_ctx(ctx)):
            pass

        assert len(recorder.completions) == 1
        assert recorder.completions[0].status == "completed"
        assert recorder.completions[0].error is None

    @given(exc=standard_exceptions)
    @settings(max_examples=100)
    def test_exception_exit_status_failed(self, exc: Exception) -> None:
        """Property: Any Exception produces status='failed'."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with pytest.raises(type(exc)), track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
            raise exc

        assert len(recorder.completions) == 1
        assert recorder.completions[0].status == "failed"
        assert recorder.completions[0].error == str(exc)

    def test_keyboard_interrupt_status_failed(self) -> None:
        """Property: KeyboardInterrupt produces status='failed'."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with (
            pytest.raises(KeyboardInterrupt),
            track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)),
        ):
            raise KeyboardInterrupt()

        assert len(recorder.completions) == 1
        assert recorder.completions[0].status == "failed"

    def test_system_exit_status_failed(self) -> None:
        """Property: SystemExit produces status='failed'."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with pytest.raises(SystemExit), track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
            raise SystemExit(1)

        assert len(recorder.completions) == 1
        assert recorder.completions[0].status == "failed"


# =============================================================================
# Exception Rendering Properties
# =============================================================================


class TestExceptionRenderingProperties:
    """Operation audit errors must honor exception sanitization."""

    def test_exception_str_redaction_is_preserved(self) -> None:
        """Property: a redacting __str__ controls the persisted audit error."""

        class RedactingException(Exception):
            def __str__(self) -> str:
                return "provider failure (credentials redacted)"

        recorder = FakeRecorder()
        ctx = FakePluginContext()
        raw_secret = "sk-live-secret-value"
        exc = RedactingException(f"provider rejected credential {raw_secret}")

        with pytest.raises(RedactingException), track_operation(
            _as_execution_repo(recorder),
            "run-1",
            "node-1",
            "source_load",
            _as_ctx(ctx),
        ):
            raise exc

        assert len(recorder.completions) == 1
        assert recorder.completions[0].error == "provider failure (credentials redacted)"
        assert raw_secret not in (recorder.completions[0].error or "")

    def test_broken_exception_str_falls_back_to_class_name(self) -> None:
        """Property: audit completion survives an exception with a broken __str__."""

        class BrokenStrException(Exception):
            def __str__(self) -> str:
                raise RuntimeError("rendering leaked secret")

        recorder = FakeRecorder()
        ctx = FakePluginContext()
        exc = BrokenStrException("raw credential should not be rendered")

        with pytest.raises(BrokenStrException), track_operation(
            _as_execution_repo(recorder),
            "run-1",
            "node-1",
            "source_load",
            _as_ctx(ctx),
        ):
            raise exc

        assert len(recorder.completions) == 1
        assert recorder.completions[0].error == "BrokenStrException"
        assert "raw credential" not in (recorder.completions[0].error or "")
        assert "rendering leaked secret" not in (recorder.completions[0].error or "")

    def test_empty_exception_str_falls_back_to_class_name(self) -> None:
        """Property: operation audit errors are never stored as empty strings."""

        class EmptyStrException(Exception):
            def __str__(self) -> str:
                return ""

        recorder = FakeRecorder()
        ctx = FakePluginContext()
        exc = EmptyStrException()

        with pytest.raises(EmptyStrException), track_operation(
            _as_execution_repo(recorder),
            "run-1",
            "node-1",
            "sink_write",
            _as_ctx(ctx),
        ):
            raise exc

        assert len(recorder.completions) == 1
        assert recorder.completions[0].error == "EmptyStrException"


# =============================================================================
# Exception Transparency Properties
# =============================================================================


class TestExceptionTransparencyProperties:
    """Original exceptions must always propagate unchanged."""

    @given(exc=standard_exceptions)
    @settings(max_examples=100)
    def test_original_exception_propagates(self, exc: Exception) -> None:
        """Property: The raised exception is the SAME object as the original."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        caught = None
        try:
            with track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
                raise exc
        except type(exc) as e:
            caught = e

        assert caught is exc  # Same object, not a copy or wrapper

    @given(exc=standard_exceptions)
    @settings(max_examples=50)
    def test_db_failure_does_not_replace_original_exception(self, exc: Exception) -> None:
        """Property: If both operation AND DB fail, original exception propagates.

        The DB error is logged but does not replace the original exception.
        """
        recorder = FakeRecorder(fail_on_complete=True)
        ctx = FakePluginContext()

        caught = None
        try:
            with track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
                raise exc
        except type(exc) as e:
            caught = e
        except RuntimeError:
            pytest.fail("DB error should NOT replace original exception")

        assert caught is exc

    def test_db_failure_on_success_raises_db_error(self) -> None:
        """Property: If operation succeeds but DB fails, DB error propagates.

        Audit integrity: a successful operation with missing audit record
        violates Tier-1 trust. The run MUST fail.
        """
        recorder = FakeRecorder(fail_on_complete=True)
        ctx = FakePluginContext()

        with (
            pytest.raises(RuntimeError, match="DB write failed"),
            track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)),
        ):
            pass  # Operation "succeeds"


# =============================================================================
# Duration Properties
# =============================================================================


class TestDurationProperties:
    """Duration calculations must always be non-negative."""

    @given(op_type=operation_types)
    @settings(max_examples=50)
    def test_duration_non_negative_on_success(self, op_type: OperationType) -> None:
        """Property: duration_ms >= 0 on normal exit."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with track_operation(_as_execution_repo(recorder), "run-1", "node-1", op_type, _as_ctx(ctx)):
            pass

        assert recorder.completions[0].duration_ms >= 0.0

    @given(exc=standard_exceptions)
    @settings(max_examples=50)
    def test_duration_non_negative_on_failure(self, exc: Exception) -> None:
        """Property: duration_ms >= 0 even on exception."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with pytest.raises(type(exc)), track_operation(_as_execution_repo(recorder), "run-1", "node-1", "source_load", _as_ctx(ctx)):
            raise exc

        assert recorder.completions[0].duration_ms >= 0.0


# =============================================================================
# Output Data Properties
# =============================================================================


class TestOutputDataProperties:
    """OperationHandle output_data must be recorded correctly."""

    @given(
        data=st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.text(max_size=50),
                max_size=5,
            ),
        )
    )
    @settings(max_examples=100)
    def test_output_data_recorded(self, data: dict[str, Any] | None) -> None:
        """Property: output_data set on handle is passed to complete_operation."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with track_operation(_as_execution_repo(recorder), "run-1", "node-1", "sink_write", _as_ctx(ctx)) as handle:
            handle.output_data = data

        assert recorder.completions[0].output_data == data

    def test_handle_exposes_operation(self) -> None:
        """Property: OperationHandle.operation has correct metadata."""
        recorder = FakeRecorder()
        ctx = FakePluginContext()

        with track_operation(
            _as_execution_repo(recorder),
            "run-1",
            "node-1",
            "source_load",
            _as_ctx(ctx),
            input_data={"source": "test"},
        ) as handle:
            assert handle.operation.run_id == "run-1"
            assert handle.operation.node_id == "node-1"
            assert handle.operation.operation_type == "source_load"
            assert handle.operation.status == "open"


# =============================================================================
# Nesting Properties
# =============================================================================


class TestNestingProperties:
    """Nested track_operation calls must restore context correctly."""

    @given(
        outer_prev=previous_op_ids,
    )
    @settings(max_examples=50)
    def test_nested_operations_restore_stack(self, outer_prev: str | None) -> None:
        """Property: Nested operations restore context like a stack."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id=outer_prev)

        with track_operation(_as_execution_repo(recorder), "run-1", "source", "source_load", _as_ctx(ctx)):
            outer_op_id = ctx.operation_id
            assert outer_op_id is not None

            with track_operation(_as_execution_repo(recorder), "run-1", "sink", "sink_write", _as_ctx(ctx)):
                inner_op_id = ctx.operation_id
                assert inner_op_id is not None
                assert inner_op_id != outer_op_id

            # After inner exits, restored to outer's operation_id
            assert ctx.operation_id == outer_op_id

        # After outer exits, restored to original
        assert ctx.operation_id == outer_prev

    def test_nested_failure_in_inner_restores_outer(self) -> None:
        """Property: Exception in inner operation restores outer's context."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id=None)

        with track_operation(_as_execution_repo(recorder), "run-1", "source", "source_load", _as_ctx(ctx)):
            outer_op_id = ctx.operation_id

            with pytest.raises(ValueError), track_operation(_as_execution_repo(recorder), "run-1", "sink", "sink_write", _as_ctx(ctx)):
                raise ValueError("inner failed")

            # Inner failure restores outer's operation_id
            assert ctx.operation_id == outer_op_id

        # Outer completes normally, restores None
        assert ctx.operation_id is None

    @given(depth=st.integers(min_value=1, max_value=6))
    @settings(max_examples=30)
    def test_arbitrary_nesting_depth_restores(self, depth: int) -> None:
        """Property: N levels of nesting all restore correctly."""
        recorder = FakeRecorder()
        ctx = FakePluginContext(operation_id="original")

        op_ids: list[str | None] = ["original"]

        # Build nested context managers
        cms = []
        for i in range(depth):
            cm = track_operation(_as_execution_repo(recorder), "run-1", f"node-{i}", "source_load", _as_ctx(ctx))
            cm.__enter__()
            op_ids.append(ctx.operation_id)
            cms.append(cm)

        # Unwind in reverse
        for i, cm in enumerate(reversed(cms)):
            cm.__exit__(None, None, None)
            expected = op_ids[depth - 1 - i]
            assert ctx.operation_id == expected

        assert ctx.operation_id == "original"
