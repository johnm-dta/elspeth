# tests/unit/contracts/transform_contracts/test_transform_protocol.py
"""Contract tests for Transform plugins.

These tests verify that transform implementations honor the TransformProtocol contract.
They test interface guarantees, not implementation details.

Contract guarantees verified:
1. process() MUST return TransformResult
2. Success results MUST have output data (row or rows)
3. Error results MUST have reason dict
4. close() MUST be idempotent
5. Lifecycle hooks on_start/on_complete MUST not raise

Usage:
    Create a subclass with fixtures providing:
    - transform: The transform plugin instance
    - valid_input: A dict that should process successfully
    - ctx: A PluginContext for the test

    class TestMyTransformContract(TransformContractTestBase):
        @pytest.fixture
        def transform(self):
            return MyTransform({"config": "value"})

        @pytest.fixture
        def valid_input(self):
            return {"field": "value"}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from elspeth.contracts import Determinism, PluginSchema, TransformProtocol, TransformResult
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.transforms.passthrough import PassThrough
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

if TYPE_CHECKING:
    from elspeth.contracts.plugin_context import PluginContext


def _is_batch_transform(transform: TransformProtocol) -> bool:
    """Return True when transform uses BatchTransformMixin and lacks process()."""
    from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin

    return isinstance(transform, BatchTransformMixin)


def _skip_if_batch_transform(transform: TransformProtocol) -> None:
    """Skip process()-oriented contracts for batch-only transforms."""
    if _is_batch_transform(transform):
        pytest.skip("Transform uses BatchTransformMixin - process() not supported, use accept()")


def _process_valid_input(
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> TransformResult:
    """Run process() on the valid-input fixture and require a TransformResult."""
    pipeline_row = make_pipeline_row(valid_input)
    result = transform.process(pipeline_row, ctx)
    assert isinstance(result, TransformResult), f"process() returned {type(result).__name__}, expected TransformResult"
    return result


def _process_successful_valid_input(
    transform: TransformProtocol,
    valid_input: dict[str, Any],
    ctx: PluginContext,
) -> TransformResult:
    """Run process() on valid input and require the fixture to be genuinely valid."""
    result = _process_valid_input(transform, valid_input, ctx)
    assert result.status == "success", (
        f"valid_input fixture must produce a successful TransformResult; got status={result.status!r}, reason={result.reason!r}"
    )
    return result


def _assert_frozenset_of_str(value: object, *, attr_name: str) -> frozenset[str]:
    """Assert a protocol field is a frozenset[str] and return it narrowed."""
    assert isinstance(value, frozenset), f"{attr_name} is {type(value).__name__}, expected frozenset"
    assert all(isinstance(field, str) for field in value), f"{attr_name} must contain only str values"
    return value


def _transform_boundary_snapshot(transform: TransformProtocol) -> dict[str, Any]:
    return {
        "can_drop_rows": transform.can_drop_rows,
        "config": dict(transform.config),
        "creates_tokens": transform.creates_tokens,
        "declared_input_fields": transform.declared_input_fields,
        "declared_output_fields": transform.declared_output_fields,
        "determinism": transform.determinism,
        "effective_static_contract": transform.effective_static_contract(),
        "input_schema": transform.input_schema,
        "is_batch_aware": transform.is_batch_aware,
        "lifecycle_started": transform._on_start_called,
        "name": transform.name,
        "node_id": transform.node_id,
        "on_error": transform.on_error,
        "on_success": transform.on_success,
        "output_schema": transform.output_schema,
        "passes_through_input": transform.passes_through_input,
        "plugin_version": transform.plugin_version,
        "source_file_hash": transform.source_file_hash,
    }


def _transform_result_snapshot(result: TransformResult) -> dict[str, Any]:
    return {
        "reason": result.reason,
        "retryable": result.retryable,
        "row": result.row.to_dict() if result.row is not None else None,
        "rows": tuple(row.to_dict() for row in result.rows) if result.rows is not None else None,
        "status": result.status,
        "success_reason": result.success_reason,
    }


class TransformContractTestBase(ABC):
    """Abstract base class for transform contract verification.

    Subclasses must provide fixtures for:
    - transform: The transform plugin instance to test
    - valid_input: A row dict that should process successfully
    - ctx: A PluginContext for the test
    """

    @pytest.fixture
    @abstractmethod
    def transform(self) -> TransformProtocol:
        """Provide a configured transform instance."""
        raise NotImplementedError

    @pytest.fixture
    @abstractmethod
    def valid_input(self) -> dict[str, Any]:
        """Provide a valid input row that should process successfully."""
        raise NotImplementedError

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Provide a PluginContext for testing."""
        return make_context(run_id="test-run-001", node_id="test-transform")

    # =========================================================================
    # Protocol Attribute Contracts
    # =========================================================================

    def test_transform_satisfies_runtime_protocol(self, transform: TransformProtocol) -> None:
        """Contract: Transform MUST satisfy the runtime engine protocol."""
        assert isinstance(transform, TransformProtocol)

    def test_transform_engine_identity_surface_is_coherent(self, transform: TransformProtocol) -> None:
        """Contract: engine-facing identity and routing metadata MUST be well formed."""
        assert isinstance(transform.name, str)
        assert len(transform.name) > 0
        assert isinstance(transform.input_schema, type)
        assert issubclass(transform.input_schema, PluginSchema)
        assert isinstance(transform.output_schema, type)
        assert issubclass(transform.output_schema, PluginSchema)
        assert isinstance(transform.determinism, Determinism)
        assert isinstance(transform.plugin_version, str)
        assert isinstance(transform.config, dict)
        assert transform.node_id is None or isinstance(transform.node_id, str)
        assert transform.source_file_hash is None or isinstance(transform.source_file_hash, str)
        assert isinstance(transform._on_start_called, bool)
        assert transform.on_error is None or isinstance(transform.on_error, str)
        assert transform.on_success is None or isinstance(transform.on_success, str)

    def test_transform_behavior_flags_are_boolean(self, transform: TransformProtocol) -> None:
        """Contract: engine dispatch and governance flags MUST be explicit booleans."""
        assert isinstance(transform.is_batch_aware, bool)
        assert isinstance(transform.supports_row_mode_when_batch_aware, bool)
        assert isinstance(transform.creates_tokens, bool)
        assert isinstance(transform.passes_through_input, bool)
        assert isinstance(transform.can_drop_rows, bool)

    def test_declaration_surfaces_are_normalized_and_effective(self, transform: TransformProtocol) -> None:
        """Contract: declared fields are normalized and included in the static contract."""
        _assert_frozenset_of_str(transform.declared_input_fields, attr_name="declared_input_fields")
        static_contract = _assert_frozenset_of_str(
            transform.effective_static_contract(),
            attr_name="effective_static_contract()",
        )
        declared_output_fields = _assert_frozenset_of_str(
            transform.declared_output_fields,
            attr_name="declared_output_fields",
        )
        assert declared_output_fields <= static_contract, (
            "declared_output_fields must be included in effective_static_contract(); "
            f"missing={sorted(declared_output_fields - static_contract)!r}"
        )

    # =========================================================================
    # process() Method Contracts
    # =========================================================================

    def test_process_returns_transform_result(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: process() MUST return TransformResult."""
        _skip_if_batch_transform(transform)
        _process_valid_input(transform, valid_input, ctx)

    def test_valid_input_returns_success_status(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: valid_input fixture MUST exercise the success path."""
        _skip_if_batch_transform(transform)
        _process_successful_valid_input(transform, valid_input, ctx)

    def test_success_result_has_output_data(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Success results MUST have output data (row or rows)."""
        _skip_if_batch_transform(transform)
        result = _process_successful_valid_input(transform, valid_input, ctx)
        assert result.has_output_data, (
            "Success TransformResult has no output data. Use TransformResult.success(row) or TransformResult.success_multi(rows)."
        )

    def test_success_single_row_is_pipeline_row(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Success single-row output MUST be a PipelineRow."""
        _skip_if_batch_transform(transform)
        result = _process_successful_valid_input(transform, valid_input, ctx)
        if result.row is not None:
            assert isinstance(result.row, PipelineRow), f"TransformResult.row is {type(result.row).__name__}, expected PipelineRow"

    def test_success_multi_row_is_tuple_of_pipeline_rows(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Success multi-row output MUST be a tuple of PipelineRows."""
        _skip_if_batch_transform(transform)
        result = _process_successful_valid_input(transform, valid_input, ctx)
        if result.rows is not None:
            assert isinstance(result.rows, tuple), f"TransformResult.rows is {type(result.rows).__name__}, expected tuple"
            for i, row in enumerate(result.rows):
                assert isinstance(row, PipelineRow), f"TransformResult.rows[{i}] is {type(row).__name__}, expected PipelineRow"

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_close_is_idempotent(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: repeated close() preserves transform metadata and output snapshots."""
        result = None
        result_snapshot = None
        if not _is_batch_transform(transform):
            result = _process_successful_valid_input(transform, valid_input, ctx)
            result_snapshot = _transform_result_snapshot(result)
        boundary_snapshot = _transform_boundary_snapshot(transform)

        assert transform.close() is None
        assert transform.close() is None
        assert transform.close() is None

        assert _transform_boundary_snapshot(transform) == boundary_snapshot
        if result_snapshot is not None:
            assert result is not None
            assert _transform_result_snapshot(result) == result_snapshot

    def test_on_start_does_not_raise(
        self,
        transform: TransformProtocol,
        ctx: PluginContext,
    ) -> None:
        """Contract: on_start() returns None and sets the lifecycle guard."""
        boundary_snapshot = _transform_boundary_snapshot(transform)
        boundary_snapshot["lifecycle_started"] = True

        assert transform.on_start(ctx) is None

        assert transform._on_start_called is True
        assert _transform_boundary_snapshot(transform) == boundary_snapshot

    def test_on_complete_does_not_raise(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: on_complete() preserves transform metadata and output snapshots."""
        result = None
        result_snapshot = None
        if not _is_batch_transform(transform):
            result = _process_successful_valid_input(transform, valid_input, ctx)
            result_snapshot = _transform_result_snapshot(result)
        boundary_snapshot = _transform_boundary_snapshot(transform)

        assert transform.on_complete(ctx) is None

        assert _transform_boundary_snapshot(transform) == boundary_snapshot
        if result_snapshot is not None:
            assert result is not None
            assert _transform_result_snapshot(result) == result_snapshot


class TransformContractPropertyTestBase(TransformContractTestBase):
    """Extended base with property-based contract verification.

    Adds Hypothesis property tests for stronger contract guarantees.
    """

    @given(extra_field=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,19}", fullmatch=True))
    @settings(
        max_examples=50,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.differing_executors,
        ],
    )
    def test_process_handles_extra_fields_gracefully(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
        extra_field: str,
    ) -> None:
        """Property: Transform MUST return TransformResult even with extra fields.

        Extra fields are ignored per PluginSchema (extra="ignore").
        """
        _skip_if_batch_transform(transform)
        input_with_extra = {**valid_input, extra_field: "extra_value"}
        pipeline_row = make_pipeline_row(input_with_extra)
        result = transform.process(pipeline_row, ctx)
        assert isinstance(result, TransformResult)

    def test_deterministic_transform_produces_same_output(
        self,
        transform: TransformProtocol,
        valid_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Property: DETERMINISTIC transforms MUST produce same output for same input."""
        if transform.determinism == Determinism.DETERMINISTIC:
            pipeline_row = make_pipeline_row(valid_input)
            result1 = transform.process(pipeline_row, ctx)
            result2 = transform.process(pipeline_row, ctx)

            assert result1.status == "success"
            assert result2.status == "success"
            assert result1.row is not None
            assert result2.row is not None
            assert result1.row.to_dict() == result2.row.to_dict(), "Deterministic transform produced different outputs"


class TransformErrorContractTestBase(TransformContractTestBase):
    """Base class for testing transform error handling contracts.

    Subclasses should provide an error_input fixture that triggers an error.
    """

    @pytest.fixture
    @abstractmethod
    def error_input(self) -> dict[str, Any]:
        """Provide an input that should cause the transform to return an error."""
        raise NotImplementedError

    def test_error_input_returns_error_status(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: error_input fixture MUST produce an error result."""
        _skip_if_batch_transform(transform)
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error", f"error_input MUST produce error, got status={result.status}"

    def test_error_result_has_reason(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Error results MUST have a reason dict."""
        _skip_if_batch_transform(transform)
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.reason is not None, "Error TransformResult has None reason"
        assert isinstance(result.reason, dict), f"TransformResult.reason is {type(result.reason).__name__}, expected dict"

    def test_error_result_has_no_output_data(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Error results should NOT have output data."""
        _skip_if_batch_transform(transform)
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert result.row is None, "Error result should not have row"
        assert result.rows is None, "Error result should not have rows"

    def test_error_result_has_retryable_flag(
        self,
        transform: TransformProtocol,
        error_input: dict[str, Any],
        ctx: PluginContext,
    ) -> None:
        """Contract: Error results MUST have a retryable flag."""
        _skip_if_batch_transform(transform)
        pipeline_row = make_pipeline_row(error_input)
        result = transform.process(pipeline_row, ctx)
        assert result.status == "error"
        assert isinstance(result.retryable, bool)


class TestTransformContractBaseExemplar(TransformContractPropertyTestBase):
    """Concrete exemplar so this contract file is directly executable."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return the canonical simple transform used to prove base collection."""
        return PassThrough({"schema": {"mode": "observed"}})

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"id": 1, "name": "contract-exemplar", "value": 42}
