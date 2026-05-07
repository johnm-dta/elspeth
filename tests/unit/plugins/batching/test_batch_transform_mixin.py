# tests/plugins/batching/test_batch_transform_mixin.py
"""Tests for BatchTransformMixin - row-level pipelining infrastructure.

These tests verify:
1. Token validation - mixin requires ctx.token to be set
2. Token identity preservation - output token matches input token
3. Stale token detection - catches synchronization issues
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.batch_adapter import SharedBatchAdapter
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import BatchTransformMixin
from elspeth.plugins.infrastructure.batching.ports import CollectorOutputPort, OutputPort
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


def _wait_for(condition: Callable[[], bool], *, timeout: float = 3.0, poll: float = 0.05, desc: str = "condition") -> None:
    """Poll until condition is true or timeout expires."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out after {timeout}s waiting for {desc}")
        time.sleep(poll)


def _make_factory() -> RecorderFactory:
    """Create an in-memory RecorderFactory for testing."""
    return make_factory()


def make_token(row_id: str, token_id: str | None = None, row_data: dict[str, Any] | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row(row_data or {}),
    )


class SimpleBatchTransform(BaseTransform, BatchTransformMixin):
    """Minimal batch transform for testing BatchTransformMixin behavior."""

    name = "simple_batch_transform"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._batch_initialized = False

    def connect_output(self, output: OutputPort, max_pending: int = 10) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: TransformContext) -> None:  # type: ignore[override]
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(make_pipeline_row(row), ctx, self._process_row)

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        # Simple passthrough - just add a marker
        output = row.to_dict()
        output["processed"] = True
        return TransformResult.success(make_pipeline_row(output), success_reason={"action": "test"})

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        raise NotImplementedError("Use accept() for row-level pipelining")

    def close(self) -> None:
        if self._batch_initialized:
            self.shutdown_batch_processing()


class Tier1FailingBatchTransform(SimpleBatchTransform):
    """Batch transform whose worker raises a Tier 1 audit exception."""

    name = "tier1_failing_batch_transform"

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        raise AuditIntegrityError("simulated audit integrity failure")


class TestBatchTransformMixinTokenValidation:
    """Tests for token validation in BatchTransformMixin."""

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort) -> Generator[SimpleBatchTransform, None, None]:
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    def test_accept_raises_value_error_when_token_is_none(self, transform: SimpleBatchTransform) -> None:
        """accept() raises ValueError when ctx.token is None.

        This is a contract violation - the engine must set ctx.token before
        calling batch transforms. The error message is user-facing and must
        be clear about what went wrong.
        """
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=None,
            token=None,  # Explicitly None - contract violation
        )

        with pytest.raises(ValueError, match=r"BatchTransformMixin requires ctx\.token to be set"):
            transform.accept({"data": "test"}, ctx)

    def test_accept_succeeds_when_token_is_set(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """accept() succeeds when ctx.token is properly set."""
        token = make_token("row-1", row_data={"data": "test"})
        ctx = make_context(
            landscape=_make_factory(),
            token=token,
            state_id="test-state-1",  # Required for batch processing
        )

        transform.accept({"data": "test"}, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        _output_token, result, _state_id = collector.results[0]
        assert isinstance(result, TransformResult)
        assert result.status == "success"


class TestBatchTransformMixinTokenIdentity:
    """Tests for token identity preservation through batch processing."""

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort) -> Generator[SimpleBatchTransform, None, None]:
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    def test_output_token_is_same_object_as_input_token(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """Output token must be the exact same object as input token.

        Token identity preservation is critical for:
        1. FIFO ordering - RowReorderBuffer uses token_id for ordering
        2. Audit attribution - the token tracks row lineage through the DAG
        """
        input_token = make_token("row-42", row_data={"value": 100})
        ctx = make_context(landscape=_make_factory(), token=input_token, state_id="test-state-1")

        transform.accept({"value": 100}, ctx)
        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 1
        output_token, _result, _state_id = collector.results[0]

        # Critical: must be the SAME object, not just equal values
        assert output_token is input_token
        assert output_token.row_id == "row-42"
        assert output_token.token_id == "token-row-42"

    def test_multiple_rows_preserve_their_respective_tokens(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """Each row preserves its own token identity through processing."""
        tokens = [make_token(f"row-{i}") for i in range(3)]

        for i, token in enumerate(tokens):
            ctx = make_context(landscape=_make_factory(), token=token, state_id=f"state-{i}")
            transform.accept({"index": i}, ctx)

        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 3

        # Results are in FIFO order, so token[i] should match result[i]
        for i, (output_token, result, _state_id) in enumerate(collector.results):
            assert output_token is tokens[i], f"Token {i} identity not preserved"
            assert isinstance(result, TransformResult)
            assert result.status == "success"


class TestStaleTokenDetection:
    """Tests for detecting stale token issues - the synchronization problem.

    The systems thinking review identified a critical risk: ctx.token is
    derivative state that must be kept in sync with the authoritative token.
    If the engine updates the token parameter but forgets to update ctx.token,
    the next transform sees stale data - an audit integrity violation.

    These tests verify that the system correctly handles token synchronization.
    """

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort) -> Generator[SimpleBatchTransform, None, None]:
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    def test_reusing_context_with_different_token_updates_correctly(
        self, transform: SimpleBatchTransform, collector: CollectorOutputPort
    ) -> None:
        """When ctx is reused, updating ctx.token must use the new token.

        This simulates the engine pattern where a single PluginContext is
        reused across multiple rows, with ctx.token updated per-row.
        """
        # Create a single context (engine pattern)
        ctx = make_context(landscape=_make_factory())

        # Process row 1 with token 1
        token1 = make_token("row-1")
        ctx.token = token1
        ctx.state_id = "state-1"
        transform.accept({"row": 1}, ctx)

        # Process row 2 with token 2 - MUST update ctx.token
        token2 = make_token("row-2")
        ctx.token = token2  # Critical: engine must do this
        ctx.state_id = "state-2"
        transform.accept({"row": 2}, ctx)

        transform.flush_batch_processing(timeout=10.0)

        assert len(collector.results) == 2

        # Verify each row got its correct token (not stale)
        output_token1, _result1, _state1 = collector.results[0]
        output_token2, _result2, _state2 = collector.results[1]

        assert output_token1 is token1, "Row 1 got wrong token (stale token bug)"
        assert output_token2 is token2, "Row 2 got wrong token (stale token bug)"

    def test_stale_token_not_reused_across_rows(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """Verifies that forgetting to update ctx.token would cause visible issues.

        This test demonstrates the CORRECT pattern. If someone forgets to
        update ctx.token, they would see the SAME token for multiple rows,
        which is detectable in tests.
        """
        ctx = make_context(landscape=_make_factory())

        # Process 3 rows, each with a unique token
        tokens = []
        for i in range(3):
            token = make_token(f"row-{i}")
            tokens.append(token)
            ctx.token = token  # Must update per-row
            ctx.state_id = f"state-{i}"
            transform.accept({"index": i}, ctx)

        transform.flush_batch_processing(timeout=10.0)

        # Verify all tokens are DISTINCT (no stale reuse)
        output_tokens = [t for t, _r, _s in collector.results]

        # Each output should be a different token object
        assert len(output_tokens) == 3
        assert output_tokens[0] is not output_tokens[1], "Token reused (stale)"
        assert output_tokens[1] is not output_tokens[2], "Token reused (stale)"
        assert output_tokens[0] is not output_tokens[2], "Token reused (stale)"

        # Each should match its expected input token
        for i, output_token in enumerate(output_tokens):
            assert output_token is tokens[i], f"Row {i} has wrong token"

    def test_context_token_update_is_visible_to_accept(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """Updating ctx.token before accept() uses the updated value.

        This verifies the synchronization contract: the executor sets
        ctx.token, then calls accept(), and accept() sees the updated value.
        """
        ctx = make_context(landscape=_make_factory())

        # Initial token
        initial_token = make_token("initial")
        ctx.token = initial_token
        ctx.state_id = "state-initial"

        # Update to a different token before accept
        updated_token = make_token("updated")
        ctx.token = updated_token
        ctx.state_id = "state-updated"

        transform.accept({"data": "test"}, ctx)
        transform.flush_batch_processing(timeout=10.0)

        output_token, _result, _state_id = collector.results[0]

        # Should see the UPDATED token, not the initial one
        assert output_token is updated_token
        assert output_token is not initial_token
        assert output_token.row_id == "updated"


class BlockingBatchTransform(BaseTransform, BatchTransformMixin):
    """Batch transform that blocks until explicitly released.

    Used for testing eviction scenarios where we need the worker to be
    blocked while we test the eviction logic.
    """

    name = "blocking_batch_transform"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._batch_initialized = False
        self._block_event = threading.Event()
        self._processing_started = threading.Event()

    def connect_output(self, output: OutputPort, max_pending: int = 10) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: TransformContext) -> None:  # type: ignore[override]
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(make_pipeline_row(row), ctx, self._process_row)

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        # Signal that processing has started
        self._processing_started.set()
        # Block until released
        self._block_event.wait()
        output = row.to_dict()
        output["processed"] = True
        return TransformResult.success(make_pipeline_row(output), success_reason={"action": "test"})

    def release_processing(self) -> None:
        """Release all blocked workers."""
        self._block_event.set()

    def wait_for_processing_started(self, timeout: float = 5.0) -> bool:
        """Wait until at least one worker has started processing."""
        return self._processing_started.wait(timeout=timeout)

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        raise NotImplementedError("Use accept() for row-level pipelining")

    def close(self) -> None:
        self._block_event.set()  # Release any blocked workers
        if self._batch_initialized:
            self.shutdown_batch_processing()


class TestBatchTransformMixinEviction:
    """Tests for eviction of timed-out submissions.

    When a waiter times out in the executor, the corresponding buffer entry
    must be evicted to prevent FIFO blocking of retry attempts.
    """

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    @pytest.fixture
    def blocking_transform(self, collector: CollectorOutputPort) -> Generator[BlockingBatchTransform, None, None]:
        """Transform that blocks until explicitly released."""
        transform = BlockingBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    @pytest.fixture
    def transform(self, collector: CollectorOutputPort) -> Generator[SimpleBatchTransform, None, None]:
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    def test_evict_submission_removes_entry_from_buffer(
        self, blocking_transform: BlockingBatchTransform, collector: CollectorOutputPort
    ) -> None:
        """evict_submission() removes the buffer entry for a timed-out submission.

        This is the core retry scenario:
        1. First attempt is submitted and tracked
        2. First attempt times out (waiter.wait() raises TimeoutError)
        3. Executor calls evict_submission() to remove buffer entry
        4. Retry can proceed without FIFO blocking
        """
        token = make_token("row-1")
        ctx = make_context(landscape=_make_factory(), token=token, state_id="state-attempt-1")

        # Submit the row (will block in worker)
        blocking_transform.accept({"value": 1}, ctx)

        # Wait for worker to start (ensures submission is in buffer)
        blocking_transform.wait_for_processing_started()

        # Before eviction, buffer should have 1 pending entry
        assert blocking_transform.batch_pending_count == 1

        # Evict the submission (simulating timeout scenario)
        assert ctx.state_id is not None
        evicted = blocking_transform.evict_submission(token.token_id, ctx.state_id)
        assert evicted is True

        # After eviction, buffer should be empty
        assert blocking_transform.batch_pending_count == 0

        # Release the blocked worker (cleanup)
        blocking_transform.release_processing()

    def test_evict_submission_returns_false_for_unknown_entry(self, transform: SimpleBatchTransform) -> None:
        """evict_submission() returns False for entries that don't exist."""
        # Try to evict an entry that was never submitted
        evicted = transform.evict_submission("nonexistent-token", "nonexistent-state")
        assert evicted is False

    def test_evict_submission_allows_retry_to_proceed(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """After eviction, retry submission can complete and be released.

        This test verifies the full retry scenario:
        1. Original submission (seq 0)
        2. Evict original (worker still running but we don't wait)
        3. Retry submission (seq 1)
        4. Retry completes and can be released (not blocked by seq 0)
        """
        token = make_token("row-1")

        # Original attempt
        ctx1 = make_context(landscape=_make_factory(), token=token, state_id="state-attempt-1")
        transform.accept({"attempt": 1}, ctx1)

        # Evict original (simulating timeout)
        assert ctx1.state_id is not None
        transform.evict_submission(token.token_id, ctx1.state_id)

        # Retry attempt with new state_id
        ctx2 = make_context(landscape=_make_factory(), token=token, state_id="state-attempt-2")
        transform.accept({"attempt": 2}, ctx2)

        # Flush and verify retry result is released
        transform.flush_batch_processing(timeout=10.0)

        # Should have result from retry (attempt 2)
        # Note: original (attempt 1) was evicted, so its result is lost
        assert len(collector.results) >= 1
        # The retry should complete successfully
        _, result, state_id = collector.results[-1]
        assert isinstance(result, TransformResult)
        assert result.status == "success"
        assert state_id == "state-attempt-2"


class SlowBatchTransform(BaseTransform, BatchTransformMixin):
    """Batch transform with configurable processing delay.

    Used for testing shutdown ordering where we need workers to be
    actively processing when shutdown is called.
    """

    name = "slow_batch_transform"

    def __init__(self, delay: float = 0.2) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._batch_initialized = False
        self._delay = delay
        self._processing_started = threading.Event()

    def connect_output(self, output: OutputPort, max_pending: int = 10) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: TransformContext) -> None:  # type: ignore[override]
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(make_pipeline_row(row), ctx, self._process_row)

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        self._processing_started.set()
        import time

        time.sleep(self._delay)
        output = row.to_dict()
        output["processed"] = True
        return TransformResult.success(make_pipeline_row(output), success_reason={"action": "test"})

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        raise NotImplementedError("Use accept() for row-level pipelining")

    def close(self) -> None:
        if self._batch_initialized:
            self.shutdown_batch_processing()


class TestShutdownDrainsInFlightRows:
    """Regression tests for P1-2026-02-14: shutdown can silently drop in-flight rows.

    The bug: shutdown_batch_processing() set _batch_shutdown before workers finished,
    causing the release loop to exit before completed results could be emitted.

    Fix: Release loop exits only on ShutdownError from buffer, not on _batch_shutdown.
    Shutdown waits for workers first, then shuts down buffer.
    """

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    def test_shutdown_emits_all_in_flight_rows(self, collector: CollectorOutputPort) -> None:
        """shutdown_batch_processing must not drop rows that workers are processing.

        Submits rows, then immediately calls shutdown. All rows must appear in
        the collector output -- none may be silently dropped.
        """
        transform = SlowBatchTransform(delay=0.1)
        transform.connect_output(collector, max_pending=5)

        num_rows = 3
        for i in range(num_rows):
            token = make_token(f"row-{i}")
            ctx = make_context(landscape=_make_factory(), token=token, state_id=f"state-{i}")
            transform.accept({"idx": i}, ctx)

        # Shutdown while workers are still processing
        transform.shutdown_batch_processing(timeout=10.0)

        # ALL rows must have been emitted -- no silent drops
        assert len(collector.results) == num_rows, (
            f"Expected {num_rows} results but got {len(collector.results)}. "
            f"Shutdown dropped {num_rows - len(collector.results)} in-flight rows."
        )

        # Verify all rows completed successfully
        for i, (_token_out, result, state_id) in enumerate(collector.results):
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert state_id == f"state-{i}"

    def test_shutdown_without_pending_rows_is_clean(self, collector: CollectorOutputPort) -> None:
        """Shutdown with no pending rows should not hang or error."""
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)

        # No rows submitted -- shutdown should be instant
        transform.shutdown_batch_processing(timeout=5.0)

        assert len(collector.results) == 0

    def test_shutdown_after_flush_emits_all_rows(self, collector: CollectorOutputPort) -> None:
        """Flush then shutdown should emit all rows exactly once."""
        transform = SlowBatchTransform(delay=0.05)
        transform.connect_output(collector, max_pending=5)

        num_rows = 4
        for i in range(num_rows):
            token = make_token(f"row-{i}")
            ctx = make_context(landscape=_make_factory(), token=token, state_id=f"state-{i}")
            transform.accept({"idx": i}, ctx)

        transform.flush_batch_processing(timeout=10.0)
        transform.shutdown_batch_processing(timeout=5.0)

        assert len(collector.results) == num_rows


class FailingOutputPort:
    """Output port that raises on first emit, then succeeds.

    Used for testing release loop error handling with stale token detection.
    """

    def __init__(self) -> None:
        self.results: list[tuple[Any, Any, Any]] = []
        self._fail_count = 0
        self._should_fail_at: set[int] = set()
        self._emit_count = 0

    def fail_at(self, *indices: int) -> None:
        """Configure which emit calls should fail (0-indexed)."""
        self._should_fail_at = set(indices)

    def emit(self, token: Any, result: Any, state_id: Any) -> None:
        current = self._emit_count
        self._emit_count += 1
        if current in self._should_fail_at:
            raise RuntimeError(f"Simulated output port failure at emit #{current}")
        self.results.append((token, result, state_id))


class TestReleaseLoopStaleTokenDetection:
    """Regression tests for P1-2026-02-14: release loop stale token/state_id.

    The bug: exception handler in _release_loop used token/state_id from
    a previous iteration when the exception occurred before entry.result unpack.

    Fix: Reset token/state_id to None at each loop start. If exception occurs
    with token=None, it's a pre-unpack internal error -- re-raise immediately.
    """

    def test_post_unpack_emit_failure_emits_exception_result(self) -> None:
        """When emit() fails AFTER unpacking entry.result, the exception handler
        should use the current token/state_id (not stale ones)."""
        port = FailingOutputPort()
        port.fail_at(0)  # First emit fails

        transform = SimpleBatchTransform()
        # Manually initialize with our special port
        transform.init_batch_processing(
            max_pending=5,
            output=port,
            name="stale-token-test",
            max_workers=5,
        )
        transform._batch_initialized = True

        try:
            token = make_token("row-0")
            ctx = make_context(landscape=_make_factory(), token=token, state_id="state-0")
            transform.accept({"data": "test"}, ctx)

            # Wait for release loop to process the failure and emit ExceptionResult
            _wait_for(lambda: len(port.results) >= 1, timeout=3.0, desc="ExceptionResult emit")

            # The second emit (ExceptionResult fallback) should have succeeded.
            # The port's results list should have the ExceptionResult with the
            # CORRECT token (row-0), not a stale one.
            assert len(port.results) >= 1
            emitted_token, emitted_result, emitted_state = port.results[0]

            # Import to check type
            from elspeth.contracts import ExceptionResult as ER

            assert isinstance(emitted_result, ER), f"Expected ExceptionResult from fallback emit, got {type(emitted_result).__name__}"
            # The token should be the CURRENT row's token, not stale
            assert emitted_token is token
            assert emitted_state == "state-0"
        finally:
            transform.shutdown_batch_processing(timeout=5.0)

    def test_multiple_rows_no_stale_token_crossover(self) -> None:
        """Processing multiple rows where emit fails on one should not
        cause the wrong token to be used in the error handler."""
        port = FailingOutputPort()
        port.fail_at(1)  # Second emit fails (row-1)

        transform = SimpleBatchTransform()
        transform.init_batch_processing(
            max_pending=5,
            output=port,
            name="crossover-test",
            max_workers=5,
        )
        transform._batch_initialized = True

        try:
            tokens = []
            for i in range(3):
                token = make_token(f"row-{i}")
                tokens.append(token)
                ctx = make_context(landscape=_make_factory(), token=token, state_id=f"state-{i}")
                transform.accept({"idx": i}, ctx)

            # Wait for all 3 rows to be processed and emitted
            _wait_for(lambda: len(port.results) >= 3, timeout=5.0, desc="all 3 row results")

            # Row 0 should have emitted successfully
            # Row 1 should have failed then emitted ExceptionResult
            # Row 2 should have emitted successfully
            # Total port.results should be at least 3 (row-0 success, row-1 ExceptionResult, row-2 success)
            assert len(port.results) >= 3

            # Verify row-0 succeeded
            t0, r0, s0 = port.results[0]
            assert t0 is tokens[0]
            assert isinstance(r0, TransformResult)
            assert s0 == "state-0"

            # Verify row-1's error used the correct token (not stale from row-0)
            t1, r1, s1 = port.results[1]
            from elspeth.contracts import ExceptionResult as ER

            assert isinstance(r1, ER), f"Expected ExceptionResult for row-1, got {type(r1).__name__}"
            assert t1 is tokens[1], "Error handler used stale token from previous row"
            assert s1 == "state-1", "Error handler used stale state_id from previous row"
        finally:
            transform.shutdown_batch_processing(timeout=5.0)


class AlwaysFailingOutputPort:
    """Output port where every emit() raises — simulates a completely broken port."""

    def emit(self, token: Any, result: Any, state_id: Any) -> None:
        raise RuntimeError("Port is completely broken")


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestReleaseLoopCrashesOnBrokenPort:
    """Regression tests for elspeth-dc2fff46fe: release loop must not silently
    continue when the output port is completely broken.

    Previously, a broken output port caused a ``critical`` log and the loop
    continued, silently losing the token's result. The waiter would hang until
    timeout with no indication of what went wrong.

    Fix: raise FrameworkBugError to crash the release thread, making the
    failure visible via waiter timeouts and thread-death detection.
    """

    def test_broken_port_kills_release_thread(self) -> None:
        """When both original emit and ExceptionResult emit fail, the release
        thread must crash (FrameworkBugError) rather than silently continuing."""
        port = AlwaysFailingOutputPort()
        transform = SimpleBatchTransform()
        transform.init_batch_processing(
            max_pending=5,
            output=port,
            name="broken-port-test",
            max_workers=5,
        )
        transform._batch_initialized = True

        token = make_token("row-0")
        ctx = make_context(landscape=_make_factory(), token=token, state_id="state-0")
        transform.accept({"data": "test"}, ctx)

        # Wait for release thread to crash from FrameworkBugError
        transform._batch_release_thread.join(timeout=3.0)

        # Release thread should have died (FrameworkBugError), not silently continued
        assert not transform._batch_release_thread.is_alive(), (
            "Release thread is still alive after port failure — "
            "it should have crashed with FrameworkBugError instead of silently continuing"
        )

        # Cleanup: shutdown won't raise because the thread is already dead
        transform._batch_executor.shutdown(wait=True)
        transform._batch_buffer.shutdown()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestShutdownRaisesOnThreadTimeout:
    """Regression test for elspeth-da9918e43a: shutdown_batch_processing must
    raise when the release thread fails to stop, not just warn.

    Previously, a warning was logged and the pipeline proceeded as if shutdown
    succeeded — potentially reporting success with undrained results.

    Fix: raise FrameworkBugError to prevent false success reporting.
    """

    def test_shutdown_completes_when_release_thread_already_crashed(self) -> None:
        """When the release thread has already crashed (e.g., broken port),
        shutdown_batch_processing completes without raising FrameworkBugError.

        FrameworkBugError only fires when the thread is alive but didn't stop.
        A crashed thread is already dead — join returns immediately.
        """
        port = AlwaysFailingOutputPort()
        transform = SimpleBatchTransform()
        transform.init_batch_processing(
            max_pending=5,
            output=port,
            name="shutdown-after-crash",
            max_workers=5,
        )
        transform._batch_initialized = True

        token = make_token("row-0")
        ctx = make_context(landscape=_make_factory(), token=token, state_id="state-0")
        transform.accept({"data": "test"}, ctx)

        # Wait for release thread to crash from broken port
        transform._batch_release_thread.join(timeout=3.0)
        assert not transform._batch_release_thread.is_alive()

        # Shutdown should complete without raising — thread is already dead
        transform.shutdown_batch_processing(timeout=5.0)


class TestBatchTransformMixinShutdownGuard:
    """Tests for accept_row() rejecting rows after shutdown."""

    def test_accept_row_raises_after_shutdown_signal(self) -> None:
        """accept_row() checks _batch_shutdown before touching the buffer.

        The mixin must guard at accept_row() level, not rely on the buffer's
        own shutdown check. This matters because _batch_shutdown is set before
        the buffer is shut down (step 1 vs step 3 of shutdown_batch_processing).
        The guard prevents new rows entering during drain.
        """
        from elspeth.plugins.infrastructure.batching.row_reorder_buffer import ShutdownError

        collector = CollectorOutputPort()
        transform = SimpleBatchTransform()
        transform.connect_output(collector, max_pending=5)

        # Set the shutdown signal directly (without full shutdown_batch_processing)
        # to isolate the mixin's own guard from the buffer's shutdown
        transform._batch_shutdown.set()

        token = make_token("row-post-shutdown")
        ctx = make_context(
            landscape=_make_factory(),
            token=token,
            state_id="state-post-shutdown",
        )

        with pytest.raises(ShutdownError, match="shut down"):
            transform.accept({"data": "should-fail"}, ctx)

        # Clean up: do full shutdown so threads stop
        transform.shutdown_batch_processing()


class TestFlushTimeoutRaisesTimeoutError:
    """Tests for flush_batch_processing() timeout path.

    When workers are stalled and the flush deadline expires,
    flush_batch_processing must raise TimeoutError rather than
    hanging indefinitely.
    """

    @pytest.fixture
    def collector(self) -> CollectorOutputPort:
        return CollectorOutputPort()

    @pytest.fixture
    def blocking_transform(self, collector: CollectorOutputPort) -> Generator[BlockingBatchTransform, None, None]:
        transform = BlockingBatchTransform()
        transform.connect_output(collector, max_pending=5)
        yield transform
        transform.close()

    def test_flush_raises_timeout_when_workers_stalled(
        self, blocking_transform: BlockingBatchTransform, collector: CollectorOutputPort
    ) -> None:
        """flush_batch_processing raises TimeoutError when rows remain pending past deadline."""
        token = make_token("row-stalled")
        ctx = make_context(landscape=_make_factory(), token=token, state_id="state-stalled")

        blocking_transform.accept({"data": "stuck"}, ctx)
        blocking_transform.wait_for_processing_started()

        assert blocking_transform.batch_pending_count > 0

        with pytest.raises(TimeoutError, match="rows still pending"):
            blocking_transform.flush_batch_processing(timeout=0.2)

    def test_flush_timeout_reports_pending_count(self, blocking_transform: BlockingBatchTransform, collector: CollectorOutputPort) -> None:
        """TimeoutError message includes the number of rows still pending."""
        tokens = [make_token(f"row-{i}") for i in range(3)]
        for i, token in enumerate(tokens):
            ctx = make_context(landscape=_make_factory(), token=token, state_id=f"state-{i}")
            blocking_transform.accept({"idx": i}, ctx)

        _wait_for(
            lambda: blocking_transform.batch_pending_count >= 1,
            timeout=3.0,
            desc="at least one row pending",
        )

        with pytest.raises(TimeoutError, match=r"\d+ rows still pending"):
            blocking_transform.flush_batch_processing(timeout=0.2)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
class TestBatchTransformMixinWorkerFailurePropagation:
    """Tests for worker-thread failure delivery to the orchestrator."""

    def test_tier1_worker_exception_reaches_waiter_immediately(self) -> None:
        """Tier 1 worker exceptions must travel through the waiter path.

        If the worker thread re-raises directly, the waiter never receives a
        result and times out instead of surfacing the original audit failure.
        """
        adapter = SharedBatchAdapter()
        transform = Tier1FailingBatchTransform()
        transform.connect_output(adapter, max_pending=5)

        token = make_token("row-tier1")
        ctx = make_context(
            landscape=_make_factory(),
            token=token,
            state_id="state-tier1",
        )
        waiter = adapter.register(token.token_id, "state-tier1")

        try:
            transform.accept({"data": "test"}, ctx)

            with pytest.raises(AuditIntegrityError, match="simulated audit integrity failure"):
                waiter.wait(timeout=0.2)
        finally:
            if ctx.state_id is not None:
                transform.evict_submission(token.token_id, ctx.state_id)
            transform.close()


class TestBatchTransformMixinSubmitRollback:
    """Tests for submit-time rollback when the worker pool rejects a row."""

    def test_submit_failure_returns_shutdown_error_without_stranding_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A submit-time shutdown race must not leave a pending ticket behind."""
        adapter = SharedBatchAdapter()
        transform = SimpleBatchTransform()
        transform.connect_output(adapter, max_pending=5)

        token = make_token("row-submit-race")
        state_id = "state-submit-race"
        ctx = make_context(
            landscape=_make_factory(),
            token=token,
            state_id=state_id,
        )
        waiter = adapter.register(token.token_id, state_id)

        def fail_submit(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(transform._batch_executor, "submit", fail_submit)

        try:
            transform.accept({"data": "test"}, ctx)

            result = waiter.wait(timeout=1.0)

            assert result.status == "error"
            assert result.reason == {
                "reason": "shutdown_requested",
                "error": "thread pool shut down during submission",
            }
            assert transform.batch_pending_count == 0
            assert (token.token_id, state_id) not in transform._batch_submissions
            assert (token.token_id, state_id) not in adapter._entries
        finally:
            transform.close()
