# tests/plugins/batching/test_batch_transform_mixin.py
"""Tests for BatchTransformMixin - row-level pipelining infrastructure.

These tests verify:
1. Token validation - mixin requires ctx.token to be set
2. Token identity preservation - output token matches input token
3. Stale token detection - catches synchronization issues
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.batching import BatchTransformMixin
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.context import PluginContext


def make_token(row_id: str, token_id: str | None = None, row_data: dict[str, Any] | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=row_data or {},
    )


class SimpleBatchTransform(BaseTransform, BatchTransformMixin):
    """Minimal batch transform for testing BatchTransformMixin behavior."""

    name = "simple_batch_transform"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})
        self._batch_initialized = False

    def connect_output(self, output: CollectorOutputPort, max_pending: int = 10) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> None:
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(row, ctx, self._process_row)

    def _process_row(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        # Simple passthrough - just add a marker
        output = dict(row)
        output["processed"] = True
        return TransformResult.success(output, success_reason={"action": "test"})

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        raise NotImplementedError("Use accept() for row-level pipelining")

    def close(self) -> None:
        if self._batch_initialized:
            self.shutdown_batch_processing()


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
            token=None,  # Explicitly None - contract violation
        )

        with pytest.raises(ValueError, match=r"BatchTransformMixin requires ctx\.token to be set"):
            transform.accept({"data": "test"}, ctx)

    def test_accept_succeeds_when_token_is_set(self, transform: SimpleBatchTransform, collector: CollectorOutputPort) -> None:
        """accept() succeeds when ctx.token is properly set."""
        token = make_token("row-1", row_data={"data": "test"})
        ctx = PluginContext(
            run_id="test-run",
            config={},
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
        ctx = PluginContext(run_id="test-run", config={}, token=input_token, state_id="test-state-1")

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
            ctx = PluginContext(run_id="test-run", config={}, token=token, state_id=f"state-{i}")
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
        ctx = PluginContext(run_id="test-run", config={})

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
        ctx = PluginContext(run_id="test-run", config={})

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
        ctx = PluginContext(run_id="test-run", config={})

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
        super().__init__({"schema": {"fields": "dynamic"}})
        self._batch_initialized = False
        self._block_event = threading.Event()
        self._processing_started = threading.Event()

    def connect_output(self, output: CollectorOutputPort, max_pending: int = 10) -> None:
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_initialized = True

    def accept(self, row: dict[str, Any], ctx: PluginContext) -> None:
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(row, ctx, self._process_row)

    def _process_row(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        # Signal that processing has started
        self._processing_started.set()
        # Block until released
        self._block_event.wait()
        output = dict(row)
        output["processed"] = True
        return TransformResult.success(output, success_reason={"action": "test"})

    def release_processing(self) -> None:
        """Release all blocked workers."""
        self._block_event.set()

    def wait_for_processing_started(self, timeout: float = 5.0) -> bool:
        """Wait until at least one worker has started processing."""
        return self._processing_started.wait(timeout=timeout)

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
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
        ctx = PluginContext(run_id="test-run", config={}, token=token, state_id="state-attempt-1")

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
        ctx1 = PluginContext(run_id="test-run", config={}, token=token, state_id="state-attempt-1")
        transform.accept({"attempt": 1}, ctx1)

        # Evict original (simulating timeout)
        assert ctx1.state_id is not None
        transform.evict_submission(token.token_id, ctx1.state_id)

        # Retry attempt with new state_id
        ctx2 = PluginContext(run_id="test-run", config={}, token=token, state_id="state-attempt-2")
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
