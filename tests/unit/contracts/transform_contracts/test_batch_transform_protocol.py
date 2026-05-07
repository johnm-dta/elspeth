# tests/unit/contracts/transform_contracts/test_batch_transform_protocol.py
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

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism, PluginSchema, TransformProtocol, TransformResult
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.engine.batch_adapter import ExceptionResult
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import OutputPort
from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.testing import make_contract, make_pipeline_row, make_row


class _BatchContractInputSchema(PluginSchema):
    """Input schema for the concrete contract-test batch exemplar."""

    value: int


class _BatchContractOutputSchema(PluginSchema):
    """Output schema for the concrete contract-test batch exemplar."""

    value: int
    processed: bool


class _BatchContractExemplarTransform(BaseTransform, BatchTransformMixin):
    """Minimal concrete transform that exercises real BatchTransformMixin behavior."""

    name = "batch_contract_exemplar"
    input_schema: type[PluginSchema] = _BatchContractInputSchema
    output_schema: type[PluginSchema] = _BatchContractOutputSchema
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:test-batch-contract"
    declared_output_fields = frozenset({"processed"})
    passes_through_input = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._batch_connected = False
        self._batch_closed = False
        self._output_schema_config = SchemaConfig.from_dict(
            {
                "mode": "observed",
                "guaranteed_fields": ["processed"],
            }
        )

    def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
        """Wire the exemplar into the production batch processing machinery."""
        if self._batch_connected:
            raise RuntimeError("connect_output() called more than once")
        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
        )
        self._batch_connected = True

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        """Accept one row and process it through BatchTransformMixin."""
        if not self._batch_connected:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(row, ctx, self._process_row)

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Return a successful enriched row for the submitted input."""
        output = row.to_dict()
        output["processed"] = True
        return TransformResult.success(
            PipelineRow(output, row.contract),
            success_reason={"action": "batch_contract_exemplar"},
        )

    def close(self) -> None:
        """Idempotently release batch resources."""
        if not self._batch_connected or self._batch_closed:
            return
        self.shutdown_batch_processing()
        self._batch_closed = True


def _assert_frozenset_of_str(value: object, *, attr_name: str) -> frozenset[str]:
    """Assert a protocol field is a frozenset[str] and return it narrowed."""
    assert isinstance(value, frozenset), f"{attr_name} is {type(value).__name__}, expected frozenset"
    assert all(isinstance(field, str) for field in value), f"{attr_name} must contain only str values"
    return value


def _submit_and_wait_for_single_result(
    started_transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
    output_port: CollectingOutputPort,
) -> tuple[TokenInfo, TransformResult, str | None]:
    """Submit valid input and return the emitted TransformResult."""
    pipeline_row = make_pipeline_row(valid_input)
    accept_result = started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]
    assert accept_result is None, f"accept() should return None, got {type(accept_result)}"

    arrived = output_port.wait_for_results(1, timeout=10.0)
    assert arrived, "Result did not arrive via OutputPort within timeout"
    results = output_port.get_results()
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    token, result, state_id = results[0]
    assert isinstance(result, TransformResult), f"Emitted result is {type(result).__name__}, expected TransformResult"
    return token, result, state_id


class CollectingOutputPort(OutputPort):
    """OutputPort that collects emitted results for verification."""

    def __init__(self) -> None:
        self.results: list[tuple[TokenInfo, TransformResult | ExceptionResult, str | None]] = []
        self.emit_count = 0
        self._lock = threading.Lock()
        self._emit_event = threading.Event()

    def emit(self, token: TokenInfo, result: TransformResult | ExceptionResult, state_id: str | None) -> None:
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

    def get_results(self) -> list[tuple[TokenInfo, TransformResult | ExceptionResult, str | None]]:
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
    def batch_transform(self) -> TransformProtocol:
        """Provide a configured batch transform instance (not yet connected/started).

        The transform should NOT have connect_output() called yet.
        Must implement TransformProtocol and use BatchTransformMixin.
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
            contract = make_contract(mode="FLEXIBLE")
            ctx.token = TokenInfo(
                token_id=f"token-{counter:03d}",
                row_id=f"row-{counter:03d}",
                row_data=make_row(valid_input.copy(), contract=contract),
            )
            return ctx

        return _make_ctx

    @pytest.fixture
    def started_transform(
        self,
        batch_transform: TransformProtocol,
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> Generator[TransformProtocol, None, None]:
        """Provide a fully initialized and started batch transform."""
        # Connect output port (BatchTransformMixin method)
        batch_transform.connect_output(output_port)  # type: ignore[attr-defined]

        # Start lifecycle
        ctx = mock_ctx_factory()
        batch_transform.on_start(ctx)

        yield batch_transform

        # Cleanup
        batch_transform.close()

    # =========================================================================
    # BatchTransformMixin Detection
    # =========================================================================

    def test_transform_uses_batch_mixin(self, batch_transform: TransformProtocol) -> None:
        """Contract: Transform MUST use BatchTransformMixin."""
        assert isinstance(batch_transform, BatchTransformMixin), (
            f"Transform {type(batch_transform).__name__} does not use BatchTransformMixin"
        )

    # =========================================================================
    # Protocol Attribute Contracts (from TransformProtocol)
    # =========================================================================

    def test_batch_transform_satisfies_runtime_protocol(self, batch_transform: TransformProtocol) -> None:
        """Contract: batch transform MUST satisfy the runtime engine protocol."""
        assert isinstance(batch_transform, TransformProtocol)

    def test_batch_transform_engine_identity_surface_is_coherent(self, batch_transform: TransformProtocol) -> None:
        """Contract: engine-facing identity and routing metadata MUST be well formed."""
        assert isinstance(batch_transform.name, str)
        assert len(batch_transform.name) > 0
        assert isinstance(batch_transform.input_schema, type)
        assert issubclass(batch_transform.input_schema, PluginSchema)
        assert isinstance(batch_transform.output_schema, type)
        assert issubclass(batch_transform.output_schema, PluginSchema)
        assert isinstance(batch_transform.determinism, Determinism)
        assert isinstance(batch_transform.plugin_version, str)
        assert isinstance(batch_transform.config, dict)
        assert batch_transform.node_id is None or isinstance(batch_transform.node_id, str)
        assert batch_transform.source_file_hash is None or isinstance(batch_transform.source_file_hash, str)
        assert isinstance(batch_transform._on_start_called, bool)
        assert batch_transform.on_error is None or isinstance(batch_transform.on_error, str)
        assert batch_transform.on_success is None or isinstance(batch_transform.on_success, str)

    def test_batch_transform_behavior_flags_are_boolean(self, batch_transform: TransformProtocol) -> None:
        """Contract: engine dispatch and governance flags MUST be explicit booleans."""
        assert isinstance(batch_transform.is_batch_aware, bool)
        assert isinstance(batch_transform.supports_row_mode_when_batch_aware, bool)
        assert isinstance(batch_transform.creates_tokens, bool)
        assert isinstance(batch_transform.passes_through_input, bool)
        assert isinstance(batch_transform.can_drop_rows, bool)

    def test_batch_declaration_surfaces_are_normalized_and_effective(
        self,
        batch_transform: TransformProtocol,
    ) -> None:
        """Contract: declared fields are normalized and included in the static contract."""
        _assert_frozenset_of_str(batch_transform.declared_input_fields, attr_name="declared_input_fields")
        static_contract = _assert_frozenset_of_str(
            batch_transform.effective_static_contract(),
            attr_name="effective_static_contract()",
        )
        declared_output_fields = _assert_frozenset_of_str(
            batch_transform.declared_output_fields,
            attr_name="declared_output_fields",
        )
        assert declared_output_fields <= static_contract, (
            "declared_output_fields must be included in effective_static_contract(); "
            f"missing={sorted(declared_output_fields - static_contract)!r}"
        )

    # =========================================================================
    # connect_output() Contracts
    # =========================================================================

    def test_connect_output_required_before_accept(
        self,
        batch_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: accept() MUST fail if connect_output() not called."""
        ctx = mock_ctx_factory()
        batch_transform.on_start(ctx)

        try:
            with pytest.raises(RuntimeError, match=r"connect_output\(\).*before accept"):
                pipeline_row = make_pipeline_row(valid_input)
                batch_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]
        finally:
            batch_transform.close()

    def test_connect_output_cannot_be_called_twice(
        self,
        batch_transform: TransformProtocol,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: connect_output() MUST fail if called twice."""
        batch_transform.connect_output(output_port)  # type: ignore[attr-defined]

        try:
            with pytest.raises(RuntimeError, match=r"connect_output\(\).*called|already"):
                batch_transform.connect_output(CollectingOutputPort())  # type: ignore[attr-defined]
        finally:
            batch_transform.close()

    # =========================================================================
    # accept() Contracts
    # =========================================================================

    def test_accept_returns_none(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: accept() MUST return None (results via OutputPort)."""
        ctx = mock_ctx_factory()
        pipeline_row = make_pipeline_row(valid_input)
        result = started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]
        assert result is None, f"accept() should return None, got {type(result)}"

    def test_accept_requires_token_in_context(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
    ) -> None:
        """Contract: accept() MUST fail if ctx.token is None."""
        ctx = Mock(spec=PluginContext)
        ctx.run_id = "test-run"
        ctx.state_id = "state-001"
        ctx.token = None  # No token!

        with pytest.raises(ValueError, match="token"):
            pipeline_row = make_pipeline_row(valid_input)
            started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

    # =========================================================================
    # Result Delivery Contracts
    # =========================================================================

    def test_results_arrive_via_output_port(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Results MUST eventually arrive through OutputPort."""
        ctx = mock_ctx_factory()
        pipeline_row = make_pipeline_row(valid_input)
        started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

        # Wait for result
        arrived = output_port.wait_for_results(1, timeout=10.0)
        assert arrived, "Result did not arrive via OutputPort within timeout"

        results = output_port.get_results()
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"

    def test_result_is_transform_result(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST be a TransformResult."""
        ctx = mock_ctx_factory()
        _submit_and_wait_for_single_result(started_transform, valid_input, ctx, output_port)

    def test_valid_input_emits_success_result(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: valid_input fixture MUST exercise the success path."""
        ctx = mock_ctx_factory()
        _token, result, _state_id = _submit_and_wait_for_single_result(started_transform, valid_input, ctx, output_port)
        assert result.status == "success", (
            f"valid_input fixture must emit a successful TransformResult; got status={result.status!r}, reason={result.reason!r}"
        )

    def test_success_result_has_output_data(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Success results MUST have output data (row or rows)."""
        ctx = mock_ctx_factory()
        _token, result, _state_id = _submit_and_wait_for_single_result(started_transform, valid_input, ctx, output_port)
        assert result.status == "success", (
            f"valid_input fixture must emit a successful TransformResult; got status={result.status!r}, reason={result.reason!r}"
        )
        assert result.has_output_data, (
            "Success TransformResult has no output data. Use TransformResult.success(row) or TransformResult.success_multi(rows)."
        )

    def test_success_single_row_is_pipeline_row(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Success single-row output MUST be a PipelineRow."""
        ctx = mock_ctx_factory()
        _token, result, _state_id = _submit_and_wait_for_single_result(started_transform, valid_input, ctx, output_port)
        if result.status == "success" and result.row is not None:
            assert isinstance(result.row, PipelineRow), f"TransformResult.row is {type(result.row).__name__}, expected PipelineRow"

    def test_success_multi_row_is_tuple_of_pipeline_rows(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Success multi-row output MUST be a tuple of PipelineRows."""
        ctx = mock_ctx_factory()
        _token, result, _state_id = _submit_and_wait_for_single_result(started_transform, valid_input, ctx, output_port)
        if result.status == "success" and result.rows is not None:
            assert isinstance(result.rows, tuple), f"TransformResult.rows is {type(result.rows).__name__}, expected tuple"
            for i, row in enumerate(result.rows):
                assert isinstance(row, PipelineRow), f"TransformResult.rows[{i}] is {type(row).__name__}, expected PipelineRow"

    def test_result_includes_correct_token(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST include the submitted token."""
        ctx = mock_ctx_factory()
        submitted_token = ctx.token
        pipeline_row = make_pipeline_row(valid_input)
        started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

        output_port.wait_for_results(1, timeout=10.0)
        results = output_port.get_results()

        returned_token, _, _ = results[0]
        assert returned_token.token_id == submitted_token.token_id, (
            f"Token mismatch: submitted {submitted_token.token_id}, got {returned_token.token_id}"
        )

    def test_result_includes_correct_state_id(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: Emitted result MUST include the correct state_id."""
        ctx = mock_ctx_factory()
        submitted_state_id = ctx.state_id
        pipeline_row = make_pipeline_row(valid_input)
        started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

        output_port.wait_for_results(1, timeout=10.0)
        results = output_port.get_results()

        _, _, returned_state_id = results[0]
        assert returned_state_id == submitted_state_id, f"state_id mismatch: submitted {submitted_state_id}, got {returned_state_id}"

    # =========================================================================
    # FIFO Ordering Contract
    # =========================================================================

    def test_results_arrive_in_fifo_order(
        self,
        started_transform: TransformProtocol,
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
            pipeline_row = make_pipeline_row(valid_input.copy())
            started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

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
        batch_transform: TransformProtocol,
        valid_input: dict[str, Any],
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: close() MUST drain accepted work and stay idempotent."""
        batch_transform.connect_output(output_port)  # type: ignore[attr-defined]
        ctx = mock_ctx_factory()
        batch_transform.on_start(ctx)
        submitted_token = ctx.token
        submitted_state_id = ctx.state_id
        pipeline_row = make_pipeline_row(valid_input)
        batch_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

        batch_transform.close()

        results = output_port.get_results()
        assert len(results) == 1, "close() must drain accepted work before shutdown returns"
        returned_token, result, returned_state_id = results[0]
        assert returned_token.token_id == submitted_token.token_id
        assert returned_state_id == submitted_state_id
        assert isinstance(result, TransformResult)
        assert result.status == "success"

        batch_transform.close()
        batch_transform.close()
        assert output_port.get_results() == results, "repeated close() calls must not emit duplicate results"

    def test_on_start_records_lifecycle_state(
        self,
        batch_transform: TransformProtocol,
        output_port: CollectingOutputPort,
        mock_ctx_factory: Any,
    ) -> None:
        """Contract: on_start() MUST record lifecycle state before processing."""
        batch_transform.connect_output(output_port)  # type: ignore[attr-defined]
        ctx = mock_ctx_factory()

        try:
            batch_transform.on_start(ctx)
            assert batch_transform._on_start_called is True
        finally:
            batch_transform.close()

    def test_on_complete_runs_after_emitted_result(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: on_complete() MUST run after accepted work has emitted."""
        # Process something first
        ctx = mock_ctx_factory()
        pipeline_row = make_pipeline_row(valid_input)
        started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]
        arrived = output_port.wait_for_results(1, timeout=10.0)
        assert arrived, "accepted work must emit before on_complete() runs"
        assert len(output_port.get_results()) == 1

        started_transform.on_complete(ctx)


class BatchTransformFIFOStressTestBase(BatchTransformContractTestBase):
    """Extended base with stress tests for FIFO ordering under load.

    Use this for transforms where FIFO ordering is critical and should
    be verified under concurrent processing conditions.
    """

    def test_fifo_order_under_concurrent_load(
        self,
        started_transform: TransformProtocol,
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
            pipeline_row = make_pipeline_row(valid_input.copy())
            started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

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


class TestBatchTransformContractBaseExemplar(BatchTransformContractTestBase):
    """Concrete exemplar so this contract file is directly executable."""

    @pytest.fixture
    def batch_transform(self) -> TransformProtocol:
        """Return the canonical simple batch transform used to prove base collection."""
        return _BatchContractExemplarTransform({})

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"value": 1}
