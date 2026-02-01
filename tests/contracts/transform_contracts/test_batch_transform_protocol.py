# tests/contracts/transform_contracts/test_batch_transform_protocol.py
"""Contract tests for Batch Transform plugins using BatchTransformMixin.

These tests verify that batch transform implementations honor the BatchTransformMixin
contract. They test interface guarantees, not implementation details.

Contract guarantees verified:
1. connect_output() MUST be called before accept()
2. accept() MUST return immediately (non-blocking except backpressure)
3. Results MUST arrive via OutputPort
4. Results MUST arrive in FIFO (submission) order
5. close() MUST be idempotent
6. Lifecycle hooks on_start/on_complete MUST not raise

Usage:
    Create a subclass with fixtures providing:
    - batch_transform: The batch transform plugin instance (not started)
    - valid_input: A dict that should process successfully
    - mock_ctx_factory: Factory that creates PluginContext with token set
    - mock_output_port: Captures emitted results

    class TestMyBatchTransformContract(BatchTransformContractTestBase):
        @pytest.fixture
        def batch_transform(self):
            return MyBatchTransform({"config": "value"})

        @pytest.fixture
        def valid_input(self):
            return {"field": "value"}
"""

from __future__ import annotations

import contextlib
import threading
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism, PluginSchema, TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.plugins.batching import OutputPort
from elspeth.plugins.batching.mixin import BatchTransformMixin
from elspeth.plugins.context import PluginContext

if TYPE_CHECKING:
    pass


class CollectingOutputPort(OutputPort):
    """OutputPort that collects emitted results for verification."""

    def __init__(self) -> None:
        self.results: list[tuple[TokenInfo, TransformResult, str | None]] = []
        self.emit_count = 0
        self._lock = threading.Lock()
        self._emit_event = threading.Event()

    def emit(self, token: TokenInfo, result: TransformResult, state_id: str | None) -> None:
        """Collect the emitted result."""
        with self._lock:
            self.results.append((token, result, state_id))
            self.emit_count += 1
            self._emit_event.set()

    def wait_for_results(self, count: int, timeout: float = 5.0) -> bool:
        """Wait for at least `count` results to arrive."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if len(self.results) >= count:
                    return True
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            self._emit_event.wait(timeout=min(0.1, remaining))
            self._emit_event.clear()

    def get_results(self) -> list[tuple[TokenInfo, TransformResult, str | None]]:
        """Get collected results (thread-safe copy)."""
        with self._lock:
            return list(self.results)


class BatchTransformContractTestBase(ABC):
    """Abstract base class for batch transform contract verification.

    Subclasses must provide fixtures for:
    - batch_transform: The batch transform plugin instance (not yet connected)
    - valid_input: A row dict that should process successfully
    - mock_ctx_factory: Factory that creates PluginContext with unique token
    """

    @pytest.fixture
    @abstractmethod
    def batch_transform(self) -> BatchTransformMixin:
        """Provide a configured batch transform instance (not yet connected/started).

        The transform should NOT have connect_output() called yet.
        """
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row that should process successfully."""
        raise NotImplementedError

    @pytest.fixture
    def output_port(self) -> CollectingOutputPort:
        """Provide a collecting output port for result verification."""
        return CollectingOutputPort()

    @pytest.fixture
    def mock_ctx_factory(self, valid_input: dict[str, Any]) -> Any:
        """Factory that creates PluginContext with unique token for each call."""
        counter = 0

        def _make_ctx() -> PluginContext:
            nonlocal counter
            counter += 1
            ctx = Mock(spec=PluginContext)
            ctx.run_id = "test-run-001"
            ctx.state_id = f"state-{counter:03d}"
            ctx.node_id = "test-batch-transform"
            ctx.landscape = Mock()
            ctx.landscape.record_call = Mock()
            ctx.landscape.allocate_call_index = Mock(return_value=0)
            ctx.token = TokenInfo(
                token_id=f"token-{counter:03d}",
                row_id=f"row-{counter:03d}",
                row_data=valid_input.copy(),
            )
            return ctx

        return _make_ctx

    @pytest.fixture
    def started_transform(
        self,
        batch_transform: BatchTransformMixin,
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> BatchTransformMixin:
        """Provide a fully initialized and started batch transform."""
        # Connect output port
        batch_transform.connect_output(output_port)

        # Start lifecycle
        ctx = mock_ctx_factory()
        batch_transform.on_start(ctx)

        yield batch_transform

        # Cleanup
        batch_transform.close()

    # =========================================================================
    # BatchTransformMixin Detection
    # =========================================================================

    def test_transform_uses_batch_mixin(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST use BatchTransformMixin."""
        assert isinstance(batch_transform, BatchTransformMixin), (
            f"Transform {type(batch_transform).__name__} does not use BatchTransformMixin"
        )

    # =========================================================================
    # Protocol Attribute Contracts (from TransformProtocol)
    # =========================================================================

    def test_transform_has_name(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have a 'name' attribute."""
        assert hasattr(batch_transform, "name")
        assert isinstance(batch_transform.name, str)
        assert len(batch_transform.name) > 0

    def test_transform_has_input_schema(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have an 'input_schema' attribute that is a PluginSchema subclass."""
        assert isinstance(batch_transform.input_schema, type)
        assert issubclass(batch_transform.input_schema, PluginSchema)

    def test_transform_has_output_schema(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have an 'output_schema' attribute that is a PluginSchema subclass."""
        assert isinstance(batch_transform.output_schema, type)
        assert issubclass(batch_transform.output_schema, PluginSchema)

    def test_transform_has_determinism(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have a 'determinism' attribute."""
        assert hasattr(batch_transform, "determinism")
        assert isinstance(batch_transform.determinism, Determinism)

    def test_transform_has_plugin_version(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have a 'plugin_version' attribute."""
        assert hasattr(batch_transform, "plugin_version")
        assert isinstance(batch_transform.plugin_version, str)

    def test_transform_has_batch_awareness_flag(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have 'is_batch_aware' attribute."""
        assert hasattr(batch_transform, "is_batch_aware")
        assert isinstance(batch_transform.is_batch_aware, bool)

    def test_transform_has_creates_tokens_flag(self, batch_transform: BatchTransformMixin) -> None:
        """Contract: Transform MUST have 'creates_tokens' attribute."""
        assert hasattr(batch_transform, "creates_tokens")
        assert isinstance(batch_transform.creates_tokens, bool)

    # =========================================================================
    # connect_output() Contracts
    # =========================================================================

    def test_connect_output_required_before_accept(
        self,
        batch_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: accept() MUST fail if connect_output() not called."""
        ctx = mock_ctx_factory()
        batch_transform.on_start(ctx)

        try:
            with pytest.raises((RuntimeError, AttributeError, ValueError)):
                batch_transform.accept(valid_input, ctx)
        finally:
            # Cleanup attempt (may fail, that's ok)
            with contextlib.suppress(Exception):
                batch_transform.close()

    def test_connect_output_cannot_be_called_twice(
        self,
        batch_transform: BatchTransformMixin,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: connect_output() MUST fail if called twice."""
        batch_transform.connect_output(output_port)

        try:
            with pytest.raises((RuntimeError, ValueError)):
                batch_transform.connect_output(CollectingOutputPort())
        finally:
            batch_transform.close()

    # =========================================================================
    # accept() Contracts
    # =========================================================================

    def test_accept_returns_none(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: accept() MUST return None (results via OutputPort)."""
        ctx = mock_ctx_factory()
        result = started_transform.accept(valid_input, ctx)
        assert result is None, f"accept() should return None, got {type(result)}"

    def test_accept_requires_token_in_context(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
    ) -> None:
        """Contract: accept() MUST fail if ctx.token is None."""
        ctx = Mock(spec=PluginContext)
        ctx.run_id = "test-run"
        ctx.state_id = "state-001"
        ctx.token = None  # No token!

        with pytest.raises(ValueError, match="token"):
            started_transform.accept(valid_input, ctx)

    # =========================================================================
    # Result Delivery Contracts
    # =========================================================================

    def test_results_arrive_via_output_port(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Results MUST eventually arrive through OutputPort."""
        ctx = mock_ctx_factory()
        started_transform.accept(valid_input, ctx)

        # Wait for result
        arrived = output_port.wait_for_results(1, timeout=10.0)
        assert arrived, "Result did not arrive via OutputPort within timeout"

        results = output_port.get_results()
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    def test_result_is_transform_result(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST be a TransformResult."""
        ctx = mock_ctx_factory()
        started_transform.accept(valid_input, ctx)

        output_port.wait_for_results(1, timeout=10.0)
        results = output_port.get_results()

        _token, result, _state_id = results[0]
        assert isinstance(result, TransformResult), f"Emitted result is {type(result).__name__}, expected TransformResult"

    def test_result_includes_correct_token(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST include the submitted token."""
        ctx = mock_ctx_factory()
        submitted_token = ctx.token
        started_transform.accept(valid_input, ctx)

        output_port.wait_for_results(1, timeout=10.0)
        results = output_port.get_results()

        returned_token, _, _ = results[0]
        assert returned_token.token_id == submitted_token.token_id, (
            f"Token mismatch: submitted {submitted_token.token_id}, got {returned_token.token_id}"
        )

    def test_result_includes_correct_state_id(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST include the correct state_id."""
        ctx = mock_ctx_factory()
        submitted_state_id = ctx.state_id
        started_transform.accept(valid_input, ctx)

        output_port.wait_for_results(1, timeout=10.0)
        results = output_port.get_results()

        _, _, returned_state_id = results[0]
        assert returned_state_id == submitted_state_id, f"state_id mismatch: submitted {submitted_state_id}, got {returned_state_id}"

    # =========================================================================
    # FIFO Ordering Contract
    # =========================================================================

    def test_results_arrive_in_fifo_order(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Results MUST arrive in submission (FIFO) order."""
        # Submit multiple rows
        submitted_tokens: list[str] = []
        for _ in range(5):
            ctx = mock_ctx_factory()
            submitted_tokens.append(ctx.token.token_id)
            started_transform.accept(valid_input.copy(), ctx)

        # Wait for all results
        arrived = output_port.wait_for_results(5, timeout=30.0)
        assert arrived, f"Not all results arrived, got {len(output_port.get_results())}/5"

        # Verify FIFO order
        results = output_port.get_results()
        received_tokens = [token.token_id for token, _, _ in results]

        assert received_tokens == submitted_tokens, f"FIFO order violated!\nSubmitted: {submitted_tokens}\nReceived:  {received_tokens}"

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_close_is_idempotent(
        self,
        batch_transform: BatchTransformMixin,
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: close() MUST be safe to call multiple times."""
        batch_transform.connect_output(output_port)
        batch_transform.on_start(mock_ctx_factory())

        # close() should not raise on first call
        batch_transform.close()

        # close() should not raise on subsequent calls (idempotent)
        batch_transform.close()
        batch_transform.close()

    def test_on_start_does_not_raise(
        self,
        batch_transform: BatchTransformMixin,
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: on_start() lifecycle hook MUST not raise."""
        batch_transform.connect_output(output_port)
        ctx = mock_ctx_factory()

        try:
            batch_transform.on_start(ctx)
        finally:
            batch_transform.close()

    def test_on_complete_does_not_raise(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: on_complete() lifecycle hook MUST not raise."""
        # Process something first
        ctx = mock_ctx_factory()
        started_transform.accept(valid_input, ctx)
        output_port.wait_for_results(1, timeout=10.0)

        # on_complete should not raise
        started_transform.on_complete(ctx)


class BatchTransformFIFOStressTestBase(BatchTransformContractTestBase):
    """Extended base with stress tests for FIFO ordering under load.

    Use this for transforms where FIFO ordering is critical and should
    be verified under concurrent processing conditions.
    """

    def test_fifo_order_under_concurrent_load(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Property: FIFO order MUST be preserved under concurrent processing."""
        # Submit many rows rapidly
        submitted_tokens: list[str] = []
        for _ in range(20):
            ctx = mock_ctx_factory()
            submitted_tokens.append(ctx.token.token_id)
            started_transform.accept(valid_input.copy(), ctx)

        # Wait for all results
        arrived = output_port.wait_for_results(20, timeout=60.0)
        assert arrived, f"Not all results arrived, got {len(output_port.get_results())}/20"

        # Verify FIFO order
        results = output_port.get_results()
        received_tokens = [token.token_id for token, _, _ in results]

        assert received_tokens == submitted_tokens, (
            f"FIFO order violated under load!\n"
            f"First mismatch at index {next(i for i, (s, r) in enumerate(zip(submitted_tokens, received_tokens, strict=False)) if s != r)}"
        )
