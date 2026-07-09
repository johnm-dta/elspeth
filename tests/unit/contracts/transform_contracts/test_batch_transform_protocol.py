# tests/unit/contracts/transform_contracts/test_batch_transform_protocol.py
"""Contract tests for Batch Transform plugins using BatchTransformMixin.

These tests verify that batch transform implementations honor the BatchTransformMixin
contract. They test interface guarantees, not implementation details.

Contract guarantees verified:
1. connect_output() MUST be called before accept()
2. accept() MUST return immediately (non-blocking except backpressure)
3. Results MUST arrive via OutputPort
4. close() MUST be idempotent
5. Lifecycle hooks on_start/on_complete MUST not raise

FIFO ordering (BatchTransformMixin's reorder buffer guarantee) is NOT verified
in :class:`BatchTransformContractTestBase`. The FIFO contract belongs to the
mixin's RowReorderBuffer, not to any individual transform — and verifying it
against a synchronous, CPU-trivial exemplar produces a non-falsifiable test:
with no real concurrency the buffer never has out-of-order completions to
reorder, so a hypothetical LIFO-buggy implementation would still pass. FIFO
is verified once, in :class:`BatchTransformFIFOStressTestBase`, with reverse
submission-order latency injection that forces workers to complete in reverse
order, exercising the buffer's actual reordering behaviour.

Subclasses that need FIFO coverage must inherit
:class:`BatchTransformFIFOStressTestBase` and provide a transform whose
``_process_row`` exposes a real concurrency surface (e.g., the
``_BatchContractExemplarTransform`` here, or a transform under load with
real I/O). Subclasses with synchronous mocks (e.g., the Azure transforms
under unittest.mock-patched httpx clients) get no useful FIFO coverage and
should NOT inherit the stress base — they would only restore the
non-falsifiability defect this file documents.

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
    """Minimal concrete transform that exercises real BatchTransformMixin behavior.

    Optional reverse-latency injection
    ----------------------------------
    Set ``config["reverse_latency_unit_seconds"]`` to a positive float to make
    ``_process_row`` sleep for ``(N - submission_index - 1) * unit`` seconds,
    where ``N`` is ``config["reverse_latency_total"]`` and ``submission_index``
    is a monotonic counter incremented on each ``accept_row`` call.

    With a thread pool wide enough to admit all submissions concurrently, this
    forces worker threads to complete in *reverse* submission order: the first
    submitted row finishes last. The production ``RowReorderBuffer`` must then
    restore FIFO output order — a non-FIFO release implementation would emit
    in worker-completion order (LIFO of submission), which the FIFO stress test
    detects.

    By default the latency knob is disabled and ``_process_row`` is purely
    synchronous, so all non-FIFO contract tests run at full speed.
    """

    name = "batch_contract_exemplar"
    determinism = Determinism.DETERMINISTIC
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
        # Reverse-latency injection (off by default). See class docstring.
        self._reverse_latency_unit = float(config.get("reverse_latency_unit_seconds", 0.0))
        self._reverse_latency_total = int(config.get("reverse_latency_total", 0))
        self._submission_counter = 0
        self._submission_counter_lock = threading.Lock()

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
        """Accept one row and process it through BatchTransformMixin.

        Captures a per-submission index *before* delegating to ``accept_row``
        so that ``_process_row`` (executed asynchronously in a worker thread)
        can compute the reverse-latency sleep deterministically based on
        submission order rather than worker-thread arrival order.
        """
        if not self._batch_connected:
            raise RuntimeError("connect_output() must be called before accept()")
        with self._submission_counter_lock:
            submission_index = self._submission_counter
            self._submission_counter += 1

        def _process_with_latency(row: PipelineRow, ctx: TransformContext) -> TransformResult:
            return self._process_row(row, ctx, submission_index)

        self.accept_row(row, ctx, _process_with_latency)

    def _process_row(
        self,
        row: PipelineRow,
        ctx: TransformContext,
        submission_index: int = 0,
    ) -> TransformResult:
        """Return a successful enriched row for the submitted input.

        If reverse-latency injection is enabled (see class docstring), sleeps
        for ``(total - submission_index - 1) * unit`` seconds before returning,
        forcing workers with smaller submission indices to complete *later*
        than workers with larger indices.
        """
        if self._reverse_latency_unit > 0.0 and self._reverse_latency_total > 0:
            sleep_seconds = max(
                0.0,
                (self._reverse_latency_total - submission_index - 1) * self._reverse_latency_unit,
            )
            if sleep_seconds > 0.0:
                time.sleep(sleep_seconds)

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


# Mocked-test timeout: short enough to surface hangs quickly, long enough to
# absorb thread-pool scheduling jitter on busy CI runners. The original 10s
# value masked hangs in the batch reorder/release loop. Real network paths
# (e.g. an actual httpx client) need their own bespoke timeouts; the contract
# tests here run against synchronous in-process exemplars and mocked HTTP
# clients that should never need more than a few hundred milliseconds.
_MOCKED_RESULT_TIMEOUT_SECONDS = 3.0


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

    arrived = output_port.wait_for_results(1, timeout=_MOCKED_RESULT_TIMEOUT_SECONDS)
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


class _BatchContractLandscape:
    """Minimal audit writer used by batch contract contexts."""

    def __init__(self) -> None:
        self.allocated_state_ids: list[str] = []
        self.recorded_calls: list[dict[str, Any]] = []

    def allocate_call_index(self, state_id: str) -> int:
        self.allocated_state_ids.append(state_id)
        return len(self.allocated_state_ids) - 1

    def record_call(self, **kwargs: Any) -> dict[str, Any]:
        self.recorded_calls.append(kwargs)
        return kwargs


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
            contract = make_contract(mode="FLEXIBLE")
            return PluginContext(
                run_id="test-run-001",
                config={},
                landscape=_BatchContractLandscape(),
                state_id=f"state-{counter:03d}",
                node_id="test-batch-transform",
                token=TokenInfo(
                    token_id=f"token-{counter:03d}",
                    row_id=f"row-{counter:03d}",
                    row_data=make_row(valid_input.copy(), contract=contract),
                ),
            )

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
        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=_BatchContractLandscape(),
            state_id="state-001",
            token=None,
        )

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
        arrived = output_port.wait_for_results(1, timeout=_MOCKED_RESULT_TIMEOUT_SECONDS)
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

        output_port.wait_for_results(1, timeout=_MOCKED_RESULT_TIMEOUT_SECONDS)
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

        output_port.wait_for_results(1, timeout=_MOCKED_RESULT_TIMEOUT_SECONDS)
        results = output_port.get_results()

        _, _, returned_state_id = results[0]
        assert returned_state_id == submitted_state_id, f"state_id mismatch: submitted {submitted_state_id}, got {returned_state_id}"

    # =========================================================================
    # FIFO Ordering Contract
    # =========================================================================
    #
    # FIFO output order is a guarantee of BatchTransformMixin's RowReorderBuffer,
    # not of any individual transform. Verifying it here against subclass
    # transforms produces a non-falsifiable test:
    #
    #   - With a synchronous, CPU-trivial ``_process_row``, worker threads
    #     complete in close-to-submission order anyway. Even an outright LIFO
    #     release implementation would pass at small n because the workers
    #     never have meaningfully out-of-order completions to reorder.
    #   - Mocked external clients (e.g. ``unittest.mock``-patched httpx) are
    #     equally synchronous and produce the same false-pass shape.
    #
    # FIFO is verified once, in :class:`BatchTransformFIFOStressTestBase`,
    # against ``_BatchContractExemplarTransform`` configured with reverse
    # submission-order latency injection. That setup forces workers to
    # complete in *reverse* order, so a buggy LIFO release implementation
    # produces output exactly opposite to submission order — making the
    # test cleanly falsifiable.

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
        """Contract: on_complete() MUST run after accepted work has emitted.

        Asserts ``_on_complete_called`` as a falsifiable post-condition,
        mirroring the ``_on_start_called`` pattern in
        ``test_on_start_records_lifecycle_state``. A subclass override that
        forgets ``super().on_complete(ctx)`` leaves the flag False and breaks
        this test, eliminating a class of latent silent failures (a stub
        override would no-op while satisfying a "doesn't raise" oracle).
        """
        # Process something first
        ctx = mock_ctx_factory()
        pipeline_row = make_pipeline_row(valid_input)
        started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]
        arrived = output_port.wait_for_results(1, timeout=_MOCKED_RESULT_TIMEOUT_SECONDS)
        assert arrived, "accepted work must emit before on_complete() runs"
        assert len(output_port.get_results()) == 1

        assert started_transform._on_complete_called is False, "fixture invariant: on_complete() must not yet have been called"
        started_transform.on_complete(ctx)
        assert started_transform._on_complete_called is True, (
            "on_complete() override must call super().on_complete(ctx) to record lifecycle invocation"
        )


# =============================================================================
# FIFO Stress Base — falsifiable FIFO contract verification
# =============================================================================
#
# Subclasses must provide a ``batch_transform`` whose ``_process_row`` exposes
# a real concurrency surface. The reverse-latency-injecting exemplar in this
# file is the canonical implementation; transforms with mocked synchronous
# clients (e.g. Azure transforms under unittest.mock-patched httpx) cannot
# meaningfully exercise this base and should NOT inherit it.
#
# Falsifiability argument (verified manually on each modification of the
# stress test or the underlying exemplar — see commit message for proof):
#
#   With reverse submission-order latency injection, worker threads complete
#   in *reverse* submission order. The production RowReorderBuffer must then
#   restore FIFO order before emitting to the OutputPort. If the buffer's
#   release loop is replaced with a LIFO (or worker-completion-order) release
#   strategy, this test sees ``received_tokens == reversed(submitted_tokens)``
#   and fails with a clear, deterministic diff. A naive synchronous LIFO
#   exemplar would produce the same failing diff. Therefore the test is
#   falsifiable: a buggy non-FIFO release implementation cannot pass.
#
# The fixture parameters below are tuned so total wall time on the success
# path is bounded (``_FIFO_STRESS_N * _FIFO_STRESS_LATENCY_UNIT`` ≈ 400ms at
# n=8, unit=50ms) while leaving enough scheduling headroom that the reverse
# completion order is reliably reproduced on busy CI runners.

# Number of rows submitted by the FIFO stress test. Chosen large enough that
# accidental serial dispatch (where workers happen to run in submission order
# regardless of latency) is statistically negligible — n=5 was demonstrably
# too small to distinguish FIFO from LIFO under realistic exemplar latency.
_FIFO_STRESS_N = 8

# Per-step latency unit. The k-th submitted row sleeps
# ``(_FIFO_STRESS_N - k - 1) * _FIFO_STRESS_LATENCY_UNIT`` seconds, so row 0
# sleeps longest and row N-1 sleeps zero. With a worker pool wide enough to
# admit all N submissions concurrently, workers complete in reverse
# submission order, exercising the reorder buffer's actual sort-on-release.
_FIFO_STRESS_LATENCY_UNIT_SECONDS = 0.05

# Total wait budget for all results. Computed as the worst-case wall time
# (longest single sleep: row 0 = (N-1) * unit) plus generous scheduling
# slack. Far tighter than the original 60s, which masked hangs and made the
# test unhelpful as a regression detector. Failures should surface within
# a couple of seconds, not a minute.
_FIFO_STRESS_TIMEOUT_SECONDS = 5.0


class BatchTransformFIFOStressTestBase(BatchTransformContractTestBase):
    """Extended base providing falsifiable FIFO ordering verification.

    Subclasses MUST provide a ``batch_transform`` fixture that returns an
    instance of ``_BatchContractExemplarTransform`` (or another transform
    with equivalent reverse-latency injection support). The stress test
    below configures the transform to inject reverse submission-order
    latency, forcing workers to complete in reverse order so the
    RowReorderBuffer's FIFO release behaviour is actually exercised.

    This base is intentionally narrow — it does NOT generalise to arbitrary
    batch transforms. The FIFO contract belongs to BatchTransformMixin
    (specifically RowReorderBuffer); verifying it once against the exemplar
    is sufficient. Plugin-level batch transforms whose ``_process_row``
    cannot expose real concurrency (synchronous mocks, deterministic CPU
    work) should remain on ``BatchTransformContractTestBase``.
    """

    @pytest.fixture
    def batch_transform(self) -> TransformProtocol:
        """Provide an exemplar configured for reverse-latency FIFO stress."""
        return _BatchContractExemplarTransform(
            {
                "reverse_latency_unit_seconds": _FIFO_STRESS_LATENCY_UNIT_SECONDS,
                "reverse_latency_total": _FIFO_STRESS_N,
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"value": 1}

    def test_fifo_order_under_reverse_latency_load(
        self,
        started_transform: TransformProtocol,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
    ) -> None:
        """Contract: FIFO release MUST hold when workers complete in reverse order.

        Submission injects reverse-order latency: row k sleeps
        ``(N - k - 1) * unit``. Workers therefore complete in reverse
        submission order. The production ``RowReorderBuffer`` must restore
        FIFO before emitting; a LIFO/worker-completion-order release
        implementation produces ``reversed(submitted_tokens)`` and fails.
        """
        submitted_tokens: list[str] = []
        for _ in range(_FIFO_STRESS_N):
            ctx = mock_ctx_factory()
            submitted_tokens.append(ctx.token.token_id)
            pipeline_row = make_pipeline_row(valid_input.copy())
            started_transform.accept(pipeline_row, ctx)  # type: ignore[attr-defined]

        arrived = output_port.wait_for_results(
            _FIFO_STRESS_N,
            timeout=_FIFO_STRESS_TIMEOUT_SECONDS,
        )
        assert arrived, (
            f"Not all results arrived within {_FIFO_STRESS_TIMEOUT_SECONDS}s, got {len(output_port.get_results())}/{_FIFO_STRESS_N}"
        )

        results = output_port.get_results()
        received_tokens = [token.token_id for token, _, _ in results]

        assert received_tokens == submitted_tokens, (
            f"FIFO order violated under reverse-latency load!\n"
            f"Submitted: {submitted_tokens}\n"
            f"Received:  {received_tokens}\n"
            f"(If this fails with received == reversed(submitted), the "
            f"reorder buffer is releasing in worker-completion order — a "
            f"FIFO contract violation, not a flaky test.)"
        )


class TestBatchTransformContractBaseExemplar(BatchTransformFIFOStressTestBase):
    """Concrete exemplar so this contract file is directly executable.

    Inherits from ``BatchTransformFIFOStressTestBase`` (and through it,
    ``BatchTransformContractTestBase``) so the contract file demonstrates
    both the per-row contract surface AND the falsifiable FIFO guarantee
    end-to-end against the production ``BatchTransformMixin`` machinery.
    """

    # batch_transform / valid_input fixtures inherited from the stress base.
