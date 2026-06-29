"""Tests for operation outcomes and results.

Tests for:
- TransformResult success/error factories
- TransformResult status is Literal (not enum) - can compare to string directly
- TransformResult has audit fields
- GateResult creation and audit fields
- RowResult creation with TokenInfo
- RowResult.error uses FailureInfo (not dict)
- FailureInfo creation and factory methods
- ArtifactDescriptor required fields (content_hash, size_bytes)
- ArtifactDescriptor uses artifact_type (not kind)
- ArtifactDescriptor factory methods

NOTE: AcceptResult tests deleted in aggregation structural cleanup.
Aggregation is now engine-controlled via batch-aware transforms.
"""

from types import MappingProxyType
from typing import Any

import pytest

from elspeth.contracts import RoutingAction, TokenInfo, TransformErrorReason
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import ConfigGateReason, MaxRetriesExceeded, OrchestrationInvariantError
from elspeth.contracts.results import (
    ArtifactDescriptor,
    FailureInfo,
    GateResult,
    RowResult,
    TransformResult,
)
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.url import SanitizedDatabaseUrl, SanitizedWebhookUrl
from elspeth.testing import make_pipeline_row


def _make_observed_contract() -> SchemaContract:
    """Create an OBSERVED schema contract for tests."""
    return SchemaContract(mode="OBSERVED", fields=())


def _wrap_dict_as_pipeline_row(data: dict[str, Any]) -> PipelineRow:
    """Wrap dict as PipelineRow with OBSERVED contract for tests."""
    return PipelineRow(data, _make_observed_contract())


class TestTransformResultMultiRow:
    """Tests for multi-row output support in TransformResult."""

    def test_transform_result_multi_row_success(self) -> None:
        """TransformResult.success_multi returns multiple rows."""
        contract = _make_observed_contract()
        raw_rows: list[dict[str, Any]] = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
        pipeline_rows = [PipelineRow(r, contract) for r in raw_rows]
        result = TransformResult.success_multi(pipeline_rows, success_reason={"action": "test"})

        assert result.status == "success"
        assert result.row is None  # Single row field is None
        assert result.rows is not None
        assert len(result.rows) == 2

    def test_transform_result_success_single_sets_rows_none(self) -> None:
        """TransformResult.success() sets rows to None for single-row output."""
        result = TransformResult.success(make_pipeline_row({"id": 1}), success_reason={"action": "test"})

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == {"id": 1}
        assert result.rows is None

    def test_transform_result_is_multi_row(self) -> None:
        """is_multi_row property distinguishes single vs multi output."""
        contract = _make_observed_contract()
        single = TransformResult.success(make_pipeline_row({"id": 1}), success_reason={"action": "test"})
        multi = TransformResult.success_multi(
            [PipelineRow({"id": 1}, contract), PipelineRow({"id": 2}, contract)],
            success_reason={"action": "test"},
        )

        assert single.is_multi_row is False
        assert multi.is_multi_row is True

    def test_transform_result_success_multi_rejects_empty_list(self) -> None:
        """success_multi raises ValueError for empty list."""
        with pytest.raises(ValueError, match="at least one row"):
            TransformResult.success_multi([], success_reason={"action": "test"})

    def test_transform_result_error_has_rows_none(self) -> None:
        """TransformResult.error() sets rows to None."""
        result = TransformResult.error({"reason": "test_error"})

        assert result.status == "error"
        assert result.row is None
        assert result.rows is None

    def test_success_multi_rejects_mixed_contracts(self) -> None:
        """success_multi raises PluginContractViolation when rows have different contracts."""
        from elspeth.contracts.errors import PluginContractViolation
        from elspeth.testing import make_contract

        contract_a = make_contract(fields={"value": int})
        contract_b = make_contract(fields={"other": str})
        rows = [
            PipelineRow({"value": 1}, contract_a),
            PipelineRow({"other": "x"}, contract_b),
        ]
        with pytest.raises(PluginContractViolation, match="inconsistent contracts"):
            TransformResult.success_multi(rows, success_reason={"action": "split"})

    def test_success_multi_accepts_same_contract(self) -> None:
        """success_multi accepts rows that all share the same contract instance."""
        contract = _make_observed_contract()
        rows = [
            PipelineRow({"value": 1}, contract),
            PipelineRow({"value": 2}, contract),
        ]
        result = TransformResult.success_multi(rows, success_reason={"action": "split"})
        assert result.is_multi_row

    def test_transform_result_has_output_data(self) -> None:
        """has_output_data property checks if ANY output exists."""
        single = TransformResult.success(make_pipeline_row({"id": 1}), success_reason={"action": "test"})
        multi = TransformResult.success_multi([make_pipeline_row({"id": 1})], success_reason={"action": "test"})
        error = TransformResult.error({"reason": "test_error"})

        assert single.has_output_data is True
        assert multi.has_output_data is True
        assert error.has_output_data is False


class TestTransformResultContextAfter:
    """Tests for context_after field in TransformResult.

    P3-2026-02-02: Pool metadata (ordering, stats) should flow through
    TransformResult to the audit trail via context_after.
    """

    def test_context_after_defaults_to_none(self) -> None:
        """context_after should default to None when not provided."""
        result = TransformResult.success(make_pipeline_row({"x": 1}), success_reason={"action": "test"})
        assert result.context_after is None

    def test_context_after_can_be_provided_to_success(self) -> None:
        """Success factory should accept context_after for audit metadata."""
        from tests.fixtures.factories import make_pool_execution_context

        pool_context = make_pool_execution_context()
        result = TransformResult.success(
            make_pipeline_row({"x": 1}),
            success_reason={"action": "enriched"},
            context_after=pool_context,
        )
        assert result.context_after is pool_context

    def test_context_after_can_be_provided_to_error(self) -> None:
        """Error factory should accept context_after for partial execution metadata."""
        from tests.fixtures.factories import make_pool_execution_context

        pool_context = make_pool_execution_context()
        result = TransformResult.error(
            {"reason": "retry_timeout"},
            context_after=pool_context,
        )
        assert result.context_after is pool_context

    def test_context_after_not_in_repr(self) -> None:
        """context_after should have repr=False for cleaner output."""
        from tests.fixtures.factories import make_pool_execution_context

        result = TransformResult.success(
            make_pipeline_row({"x": 1}),
            success_reason={"action": "test"},
            context_after=make_pool_execution_context(),
        )
        repr_str = repr(result)
        assert "context_after" not in repr_str


class TestTransformResult:
    """Tests for TransformResult."""

    def test_success_factory(self) -> None:
        """Success factory creates result with status='success' and row data."""
        row = make_pipeline_row({"field": "value", "count": 42})
        result = TransformResult.success(row, success_reason={"action": "processed"})

        assert result.status == "success"
        assert result.row is row
        assert result.reason is None
        assert result.retryable is False
        assert result.success_reason == {"action": "processed"}

    def test_success_factory_requires_success_reason(self) -> None:
        """Success factory raises ValueError without success_reason.

        This validates the invariant: success results MUST have success_reason.
        Missing success_reason is a plugin bug.
        """
        with pytest.raises(ValueError, match="MUST provide success_reason"):
            # Using the dataclass directly to bypass factory's keyword-only arg
            TransformResult(status="success", row=make_pipeline_row({"x": 1}), reason=None)

    @pytest.mark.parametrize("success_reason", [{}, {"action": 123}])
    def test_success_factory_requires_string_action(self, success_reason: Any) -> None:
        """Success metadata must include a runtime-valid action at construction."""
        with pytest.raises(ValueError, match=r"MUST include success_reason\['action'\]"):
            TransformResult.success(make_pipeline_row({"x": 1}), success_reason=success_reason)

    def test_error_factory(self) -> None:
        """Error factory creates result with status='error' and reason."""
        reason: TransformErrorReason = {"reason": "validation_failed", "field": "count"}
        result = TransformResult.error(reason)

        assert result.status == "error"
        assert result.row is None
        assert result.reason == reason
        assert result.retryable is False

    def test_error_factory_with_retryable(self) -> None:
        """Error factory accepts retryable flag."""
        reason: TransformErrorReason = {"reason": "retry_timeout", "error": "timeout"}
        result = TransformResult.error(reason, retryable=True)

        assert result.status == "error"
        assert result.retryable is True

    def test_audit_fields_default_to_none(self) -> None:
        """Audit fields default to None, set by executor."""
        result = TransformResult.success(make_pipeline_row({"x": 1}), success_reason={"action": "test"})

        assert result.input_hash is None
        assert result.output_hash is None
        assert result.duration_ms is None

    def test_audit_fields_can_be_set(self) -> None:
        """Audit fields can be set after creation."""
        result = TransformResult.success(make_pipeline_row({"x": 1}), success_reason={"action": "test"})
        result.input_hash = "abc123"
        result.output_hash = "def456"
        result.duration_ms = 12.5

        assert result.input_hash == "abc123"
        assert result.output_hash == "def456"
        assert result.duration_ms == 12.5

    def test_audit_fields_not_in_repr(self) -> None:
        """Audit fields have repr=False for cleaner output."""
        result = TransformResult.success(make_pipeline_row({"x": 1}), success_reason={"action": "test"})
        result.input_hash = "abc123"

        # audit fields should not appear in repr
        repr_str = repr(result)
        assert "input_hash" not in repr_str
        assert "output_hash" not in repr_str
        assert "duration_ms" not in repr_str


class TestTransformResultErrorInvariants:
    """Regression tests for P1: TransformResult error-path invariant enforcement.

    Bug: P1-2026-02-14-transformresult-does-not-enforce-error-path-invariants

    Error results MUST satisfy symmetric invariants:
    - status="error" requires reason is not None
    - status="error" forbids row/rows (no output data)
    - status="error" forbids success_reason
    """

    def test_error_without_reason_raises(self) -> None:
        """status='error' with reason=None raises ValueError."""
        with pytest.raises(ValueError, match="MUST provide reason"):
            TransformResult(status="error", row=None, reason=None)

    def test_error_without_reason_key_raises(self) -> None:
        """status='error' payloads must include the required 'reason' key."""
        reason_without_key: Any = {}
        with pytest.raises(ValueError, match=r"MUST include reason\['reason'\]"):
            TransformResult.error(reason_without_key)

    def test_error_with_row_raises(self) -> None:
        """status='error' with row set raises ValueError."""
        with pytest.raises(ValueError, match="MUST NOT include output data"):
            TransformResult(
                status="error",
                row=make_pipeline_row({"x": 1}),
                reason={"reason": "test_error"},
            )

    def test_error_with_rows_raises(self) -> None:
        """status='error' with rows set raises ValueError."""
        with pytest.raises(ValueError, match="MUST NOT include output data"):
            TransformResult(
                status="error",
                row=None,
                reason={"reason": "test_error"},
                rows=(make_pipeline_row({"x": 1}),),
            )

    def test_error_with_success_reason_raises(self) -> None:
        """status='error' with success_reason set raises ValueError."""
        with pytest.raises(ValueError, match="MUST NOT include success_reason"):
            TransformResult(
                status="error",
                row=None,
                reason={"reason": "test_error"},
                success_reason={"action": "processed"},
            )

    def test_error_factory_passes_invariants(self) -> None:
        """TransformResult.error() factory produces valid error results."""
        result = TransformResult.error({"reason": "test_error"})
        assert result.status == "error"
        assert result.reason is not None
        assert result.row is None
        assert result.rows is None
        assert result.success_reason is None

    def test_error_factory_with_retryable_passes_invariants(self) -> None:
        """TransformResult.error() with retryable=True passes invariants."""
        result = TransformResult.error({"reason": "api_error"}, retryable=True)
        assert result.status == "error"
        assert result.retryable is True
        assert result.reason is not None

    def test_error_factory_with_context_after_passes_invariants(self) -> None:
        """TransformResult.error() with context_after passes invariants."""
        from tests.fixtures.factories import make_pool_execution_context

        result = TransformResult.error(
            {"reason": "retry_timeout"},
            context_after=make_pool_execution_context(),
        )
        assert result.status == "error"
        assert result.context_after is not None


class TestGateResult:
    """Tests for GateResult."""

    def test_creation(self) -> None:
        """GateResult stores row and routing action."""
        row = {"value": 100}
        action = RoutingAction.route("high", reason=ConfigGateReason(condition="value > threshold", result="high"))
        result = GateResult(row=row, action=action)

        assert result.row == row
        assert result.action == action
        assert result.action.destinations == ("high",)

    def test_audit_fields_default_to_none(self) -> None:
        """Audit fields default to None."""
        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )

        assert result.input_hash is None
        assert result.output_hash is None
        assert result.duration_ms is None

    def test_audit_fields_can_be_set(self) -> None:
        """Audit fields can be set by executor."""
        result = GateResult(
            row={"x": 1},
            action=RoutingAction.continue_(),
        )
        result.input_hash = "hash1"
        result.output_hash = "hash2"
        result.duration_ms = 5.0

        assert result.input_hash == "hash1"
        assert result.output_hash == "hash2"
        assert result.duration_ms == 5.0


class TestAcceptResultDeleted:
    """Guard against AcceptResult reintroduction.

    AcceptResult was removed as part of the aggregation structural cleanup.
    These tests exist per the no-legacy-code policy: if someone accidentally
    re-adds AcceptResult, these tests will fail and surface the violation.
    """

    def test_accept_result_deleted_from_contracts(self) -> None:
        """AcceptResult must not exist in contracts.results."""
        with pytest.raises(ImportError):
            from elspeth.contracts.results import AcceptResult  # type: ignore[attr-defined] # noqa: F401

    def test_accept_result_not_exported(self) -> None:
        """AcceptResult should NOT be exported from elspeth.contracts."""
        import elspeth.contracts as contracts

        with pytest.raises(AttributeError):
            _ = contracts.AcceptResult  # type: ignore[attr-defined]


class TestRowResult:
    """Tests for RowResult."""

    def test_creation(self) -> None:
        """COMPLETED RowResult stores token, data, outcome, and required sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1, "processed": True}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="processed",
        )

        assert result.token == token
        assert result.final_data.to_dict() == {"x": 1, "processed": True}
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW
        assert result.sink_name == "processed"
        assert result.scheduler_pending_sink is False

    def test_scheduler_pending_sink_requires_bool(self) -> None:
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        with pytest.raises(OrchestrationInvariantError, match="scheduler_pending_sink must be bool"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="processed",
                scheduler_pending_sink=1,  # type: ignore[arg-type]
            )

    def test_completed_without_sink_name_raises(self) -> None:
        """COMPLETED outcome requires sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        with pytest.raises(OrchestrationInvariantError, match="DEFAULT_FLOW"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_routed_with_sink_name(self) -> None:
        """ROUTED outcome includes sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            sink_name="flagged",
        )

        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.GATE_ROUTED
        assert result.sink_name == "flagged"

    def test_routed_without_sink_name_raises(self) -> None:
        """ROUTED outcome requires sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        with pytest.raises(OrchestrationInvariantError, match="GATE_ROUTED"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
            )

    def test_coalesced_without_sink_name_raises(self) -> None:
        """COALESCED outcome requires sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        with pytest.raises(OrchestrationInvariantError, match="COALESCED"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.COALESCED,
            )

    def test_coalesced_with_sink_name(self) -> None:
        """COALESCED outcome accepts sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
            sink_name="output",
        )

        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.COALESCED
        assert result.sink_name == "output"

    def test_is_frozen(self) -> None:
        """RowResult is frozen — prevents post-construction mutation of sink_name/outcome."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        with pytest.raises(AttributeError):
            result.sink_name = "tampered"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            result.outcome = TerminalOutcome.FAILURE  # type: ignore[misc]


class TestRowResultTwoAxis:
    """ADR-019 Phase 1: RowResult carries (outcome, path) at the producer site."""

    def test_completed_row_result(self) -> None:
        token = TokenInfo(row_id="row_001", token_id="tok_001", row_data=_wrap_dict_as_pipeline_row({"k": "v"}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"k": "v"}),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW

    def test_routed_on_error_requires_error_field(self) -> None:
        token = TokenInfo(row_id="row_001", token_id="tok_001", row_data=_wrap_dict_as_pipeline_row({"k": "v"}))
        with pytest.raises(OrchestrationInvariantError, match="ON_ERROR_ROUTED"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"k": "v"}),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
                error=None,
            )

    def test_buffered_row_result(self) -> None:
        token = TokenInfo(row_id="row_001", token_id="tok_001", row_data=_wrap_dict_as_pipeline_row({"k": "v"}))
        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"k": "v"}),
            outcome=None,
            path=TerminalPath.BUFFERED,
        )
        assert result.outcome is None
        assert result.path == TerminalPath.BUFFERED

    def test_illegal_completed_pair_rejected_before_recording(self) -> None:
        token = TokenInfo(row_id="row_001", token_id="tok_001", row_data=_wrap_dict_as_pipeline_row({"k": "v"}))
        with pytest.raises(OrchestrationInvariantError, match="legal"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"k": "v"}),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.UNROUTED,
                sink_name="primary",
            )


class TestArtifactDescriptor:
    """Tests for ArtifactDescriptor."""

    def test_required_fields(self) -> None:
        """ArtifactDescriptor requires artifact_type, path_or_uri, content_hash, size_bytes."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///path/to/output.csv",
            content_hash="abc123",
            size_bytes=1024,
        )

        assert descriptor.artifact_type == "file"
        assert descriptor.path_or_uri == "file:///path/to/output.csv"
        assert descriptor.content_hash == "abc123"
        assert descriptor.size_bytes == 1024

    def test_uses_artifact_type_not_kind(self) -> None:
        """Field is named artifact_type, not kind - matches DB schema."""
        descriptor = ArtifactDescriptor(
            artifact_type="database",
            path_or_uri="db://table@url",
            content_hash="def456",
            size_bytes=500,
        )

        assert descriptor.artifact_type == "database"

        with pytest.raises(AttributeError):
            _ = descriptor.kind  # type: ignore[attr-defined]

    def test_content_hash_is_required(self) -> None:
        """content_hash is required (not optional) - audit integrity."""
        # This would fail at runtime with a TypeError if content_hash were omitted
        # We verify by constructing with all required fields
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="abc123",
            size_bytes=100,
        )
        assert descriptor.content_hash == "abc123"

    def test_size_bytes_is_required(self) -> None:
        """size_bytes is required (not optional) - verification."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="abc123",
            size_bytes=256,
        )
        assert descriptor.size_bytes == 256

    def test_metadata_is_optional(self) -> None:
        """metadata defaults to None."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="abc123",
            size_bytes=100,
        )
        assert descriptor.metadata is None

    def test_metadata_can_be_set(self) -> None:
        """metadata can be set with type-specific info."""
        descriptor = ArtifactDescriptor(
            artifact_type="database",
            path_or_uri="db://table@url",
            content_hash="abc123",
            size_bytes=100,
            metadata=MappingProxyType({"table": "results", "row_count": 50}),
        )
        assert descriptor.metadata == {"table": "results", "row_count": 50}

    def test_is_frozen(self) -> None:
        """ArtifactDescriptor is frozen (immutable)."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="abc123",
            size_bytes=100,
        )

        with pytest.raises(AttributeError):
            descriptor.content_hash = "new_hash"  # type: ignore[misc]

    def test_metadata_is_deeply_immutable(self) -> None:
        """metadata dict must be truly immutable — not just shallow-frozen.

        Bug: frozen=True prevents descriptor.metadata = new_dict, but does NOT
        prevent descriptor.metadata["key"] = "value". The fix is to convert
        metadata to MappingProxyType in __post_init__.
        """
        original = {"table": "results", "row_count": 50}
        descriptor = ArtifactDescriptor(
            artifact_type="database",
            path_or_uri="db://table@url",
            content_hash="abc123",
            size_bytes=100,
            metadata=MappingProxyType(original),
        )

        # Mutating the metadata dict must raise TypeError
        with pytest.raises(TypeError):
            descriptor.metadata["injected"] = "evil"  # type: ignore[index]

        # Mutating the original dict must NOT affect the descriptor (defensive copy)
        original["injected"] = "evil"
        assert descriptor.metadata is not None
        assert "injected" not in descriptor.metadata

    def test_missing_required_fields_raises_type_error(self) -> None:
        """ArtifactDescriptor requires all fields - TypeError on missing."""
        with pytest.raises(TypeError):
            ArtifactDescriptor(  # type: ignore[call-arg]
                artifact_type="file",
                path_or_uri="file:///test",
            )

        with pytest.raises(TypeError):
            ArtifactDescriptor(  # type: ignore[call-arg]
                artifact_type="file",
                path_or_uri="file:///test",
                content_hash="abc123",
            )


class TestArtifactDescriptorAuditInvariants:
    """Direct construction must enforce the same audit invariants as factories."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"artifact_type": "s3"},
            {"artifact_type": ""},
            {"path_or_uri": ""},
            {"content_hash": ""},
            {"content_hash": "not-hex"},
            {"content_hash": object()},
            {"size_bytes": -1},
            {"size_bytes": 1.5},
            {"size_bytes": True},
            {"metadata": [("table", "results")]},
            {"metadata": {1: "not-string"}},
        ],
    )
    def test_rejects_malformed_audit_fields(self, overrides: dict[str, object]) -> None:
        kwargs: dict[str, object] = {
            "artifact_type": "file",
            "path_or_uri": "file:///tmp/output.csv",
            "content_hash": "abc123",
            "size_bytes": 100,
        }
        kwargs.update(overrides)

        with pytest.raises((TypeError, ValueError)):
            ArtifactDescriptor(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "path_or_uri",
        [
            "webhook://https://user:" + "redacted" + "@example.com/hook",
            "webhook://https://api.example.com/hook?token=redacted",
            "webhook://https://hooks.slack.com/services/T00000000/B00000000/opaque_path_segment_value",
            "db://results@postgresql://user:" + "redacted" + "@db/app",
            "file:///tmp/output.csv?api_key=redacted",
        ],
    )
    def test_rejects_credential_bearing_path_or_uri(self, path_or_uri: str) -> None:
        with pytest.raises(ValueError):
            ArtifactDescriptor(
                artifact_type="file",
                path_or_uri=path_or_uri,
                content_hash="abc123",
                size_bytes=100,
            )


class TestArtifactDescriptorFactories:
    """Tests for ArtifactDescriptor factory methods."""

    def test_for_file(self) -> None:
        """for_file creates file artifact with file:// URI scheme."""
        descriptor = ArtifactDescriptor.for_file(
            path="/output/results.csv",
            content_hash="abc123",
            size_bytes=2048,
        )

        assert descriptor.artifact_type == "file"
        assert descriptor.path_or_uri == "file:///output/results.csv"
        assert descriptor.content_hash == "abc123"
        assert descriptor.size_bytes == 2048
        assert descriptor.metadata is None

    def test_for_file_encodes_uri_delimiters_in_literal_path(self) -> None:
        """Literal filename delimiters must not become artifact URI query params."""
        descriptor = ArtifactDescriptor.for_file(
            path="/output/results?token=literal#fragment.csv",
            content_hash="abc123",
            size_bytes=2048,
        )

        assert descriptor.path_or_uri == "file:///output/results%3Ftoken%3Dliteral%23fragment.csv"

    def test_for_database(self) -> None:
        """for_database creates database artifact with db:// URI scheme."""
        # Use SanitizedDatabaseUrl - URL without password so no fingerprint key needed
        sanitized_url = SanitizedDatabaseUrl.from_raw_url("postgresql://localhost/mydb", fail_if_no_key=False)
        descriptor = ArtifactDescriptor.for_database(
            url=sanitized_url,
            table="results",
            content_hash="def456",
            payload_size=1024,
            row_count=100,
        )

        assert descriptor.artifact_type == "database"
        assert descriptor.path_or_uri == "db://results@postgresql://localhost/mydb"
        assert descriptor.content_hash == "def456"
        assert descriptor.size_bytes == 1024
        # metadata includes table and row_count (no fingerprint since no password)
        assert descriptor.metadata is not None
        assert descriptor.metadata["table"] == "results"
        assert descriptor.metadata["row_count"] == 100

    def test_for_webhook(self) -> None:
        """for_webhook creates webhook artifact with webhook:// URI scheme."""
        # Use SanitizedWebhookUrl - URL without tokens so no fingerprint key needed
        sanitized_url = SanitizedWebhookUrl.from_raw_url("https://api.example.com/webhook", fail_if_no_key=False)
        descriptor = ArtifactDescriptor.for_webhook(
            url=sanitized_url,
            content_hash="abc789",
            request_size=512,
            response_code=200,
        )

        assert descriptor.artifact_type == "webhook"
        assert descriptor.path_or_uri == "webhook://https://api.example.com/webhook"
        assert descriptor.content_hash == "abc789"
        assert descriptor.size_bytes == 512
        assert descriptor.metadata == {"response_code": 200}

    def test_for_webhook_uses_redacted_slack_path_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """for_webhook stores the known-pattern sanitized URL."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")
        sanitized_url = SanitizedWebhookUrl.from_raw_url("https://hooks.slack.com/services/T00000000/B00000000/opaque_path_segment_value")

        descriptor = ArtifactDescriptor.for_webhook(
            url=sanitized_url,
            content_hash="abc789",
            request_size=512,
            response_code=200,
        )

        assert descriptor.path_or_uri == "webhook://https://hooks.slack.com/services/T00000000/B00000000/REDACTED"
        assert "opaque_path_segment_value" not in descriptor.path_or_uri
        assert descriptor.metadata is not None
        assert "url_fingerprint" in descriptor.metadata

    def test_for_webhook_with_error_response(self) -> None:
        """for_webhook captures error response codes."""
        sanitized_url = SanitizedWebhookUrl.from_raw_url("https://api.example.com/webhook", fail_if_no_key=False)
        descriptor = ArtifactDescriptor.for_webhook(
            url=sanitized_url,
            content_hash="abc",
            request_size=256,
            response_code=500,
        )

        assert descriptor.metadata == {"response_code": 500}


class TestArtifactDescriptorTypeSafety:
    """Tests for strict type enforcement in ArtifactDescriptor factories.

    Bug: P2-2026-01-31-artifactdescriptor-duck-typed-urls

    These tests verify that only properly sanitized URL types are accepted,
    not duck-typed objects that happen to have the same attributes.
    """

    def test_for_database_rejects_duck_typed_objects(self) -> None:
        """for_database rejects objects that look like SanitizedDatabaseUrl but aren't.

        This prevents security bypass where unsanitized URLs could reach audit trail.
        """
        from dataclasses import dataclass

        @dataclass
        class FakeSanitizedUrl:
            """Duck-typed fake that has same attributes but isn't the real type."""

            sanitized_url: str
            fingerprint: str | None

        fake_url = FakeSanitizedUrl(
            sanitized_url="postgresql://host/db?credential=redacted",  # Contains secret!
            fingerprint=None,
        )

        with pytest.raises(TypeError, match="must be a SanitizedDatabaseUrl instance"):
            ArtifactDescriptor.for_database(
                url=fake_url,  # type: ignore[arg-type]
                table="test",
                content_hash="abc",
                payload_size=100,
                row_count=10,
            )

    def test_for_webhook_rejects_duck_typed_objects(self) -> None:
        """for_webhook rejects objects that look like SanitizedWebhookUrl but aren't.

        This prevents security bypass where unsanitized URLs could reach audit trail.
        """
        from dataclasses import dataclass

        @dataclass
        class FakeSanitizedUrl:
            """Duck-typed fake that has same attributes but isn't the real type."""

            sanitized_url: str
            fingerprint: str | None

        fake_url = FakeSanitizedUrl(
            sanitized_url="https://api.example.com?token=sk-secret-key",  # Contains secret!
            fingerprint=None,
        )

        with pytest.raises(TypeError, match="must be a SanitizedWebhookUrl instance"):
            ArtifactDescriptor.for_webhook(
                url=fake_url,  # type: ignore[arg-type]
                content_hash="abc",
                request_size=100,
                response_code=200,
            )

    def test_for_database_rejects_plain_string(self) -> None:
        """for_database rejects plain string URLs."""
        with pytest.raises(TypeError, match="must be a SanitizedDatabaseUrl instance"):
            ArtifactDescriptor.for_database(
                url="postgresql://host/db",  # type: ignore[arg-type]
                table="test",
                content_hash="abc",
                payload_size=100,
                row_count=10,
            )

    def test_for_webhook_rejects_plain_string(self) -> None:
        """for_webhook rejects plain string URLs."""
        with pytest.raises(TypeError, match="must be a SanitizedWebhookUrl instance"):
            ArtifactDescriptor.for_webhook(
                url="https://api.example.com?token=secret",  # type: ignore[arg-type]
                content_hash="abc",
                request_size=100,
                response_code=200,
            )


class TestArtifactDescriptorTypes:
    """Tests for artifact_type values."""

    def test_file_type(self) -> None:
        """File artifact type."""
        descriptor = ArtifactDescriptor.for_file(
            path="/test.csv",
            content_hash="a",
            size_bytes=1,
        )
        assert descriptor.artifact_type == "file"

    def test_database_type(self) -> None:
        """Database artifact type."""
        sanitized_url = SanitizedDatabaseUrl.from_raw_url("sqlite:///:memory:", fail_if_no_key=False)
        descriptor = ArtifactDescriptor.for_database(
            url=sanitized_url,
            table="t",
            content_hash="a",
            payload_size=1,
            row_count=1,
        )
        assert descriptor.artifact_type == "database"

    def test_webhook_type(self) -> None:
        """Webhook artifact type."""
        sanitized_url = SanitizedWebhookUrl.from_raw_url("http://localhost", fail_if_no_key=False)
        descriptor = ArtifactDescriptor.for_webhook(
            url=sanitized_url,
            content_hash="a",
            request_size=1,
            response_code=200,
        )
        assert descriptor.artifact_type == "webhook"


class TestFailureInfo:
    """Tests for FailureInfo dataclass."""

    def test_creation_with_required_fields_only(self) -> None:
        """FailureInfo can be created with only required fields."""
        info = FailureInfo(
            exception_type="ValueError",
            message="Invalid value provided",
        )

        assert info.exception_type == "ValueError"
        assert info.message == "Invalid value provided"
        assert info.attempts is None
        assert info.last_error is None

    def test_creation_with_all_fields(self) -> None:
        """FailureInfo can be created with all fields."""
        info = FailureInfo(
            exception_type="MaxRetriesExceeded",
            message="Max retries (3) exceeded: Connection refused",
            attempts=3,
            last_error="Connection refused",
        )

        assert info.exception_type == "MaxRetriesExceeded"
        assert info.message == "Max retries (3) exceeded: Connection refused"
        assert info.attempts == 3
        assert info.last_error == "Connection refused"

    def test_from_max_retries_exceeded_factory(self) -> None:
        """from_max_retries_exceeded creates FailureInfo from exception."""
        original_error = ConnectionError("Connection refused")
        exc = MaxRetriesExceeded(attempts=3, last_error=original_error)

        info = FailureInfo.from_max_retries_exceeded(exc)

        assert info.exception_type == "MaxRetriesExceeded"
        assert info.message == str(exc)
        assert info.attempts == 3
        assert info.last_error == "Connection refused"

    def test_is_frozen(self) -> None:
        """FailureInfo is frozen — failure evidence must be immutable."""
        info = FailureInfo(
            exception_type="TestError",
            message="Test message",
        )

        with pytest.raises(AttributeError):
            info.attempts = 5  # type: ignore[misc]


class TestRowResultWithFailureInfo:
    """Tests for RowResult.error field using FailureInfo."""

    def test_failed_outcome_with_failure_info(self) -> None:
        """FAILED outcome includes FailureInfo error details."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        error = FailureInfo(
            exception_type="MaxRetriesExceeded",
            message="Max retries (3) exceeded",
            attempts=3,
            last_error="Connection refused",
        )

        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            error=error,
        )

        assert result.outcome == TerminalOutcome.FAILURE
        assert result.path == TerminalPath.UNROUTED
        assert result.error is not None
        assert result.error.exception_type == "MaxRetriesExceeded"
        assert result.error.attempts == 3

    def test_error_field_type_is_failure_info(self) -> None:
        """RowResult.error field is typed as FailureInfo | None."""
        from dataclasses import fields

        row_result_fields = {f.name: f for f in fields(RowResult)}
        error_field = row_result_fields["error"]

        # The type annotation should be FailureInfo | None
        # We verify by checking that FailureInfo is in the string representation
        type_str = str(error_field.type)
        assert "FailureInfo" in type_str or error_field.type is FailureInfo

    def test_routed_on_error_with_failure_info(self) -> None:
        """ROUTED_ON_ERROR carries sink_name and typed originating failure evidence."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        error = FailureInfo(
            exception_type="ValueError",
            message="bad row",
        )

        result = RowResult(
            token=token,
            final_data=_wrap_dict_as_pipeline_row({"x": 1}),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
            sink_name="error_sink",
            error=error,
        )

        assert result.outcome == TerminalOutcome.FAILURE
        assert result.path == TerminalPath.ON_ERROR_ROUTED
        assert result.sink_name == "error_sink"
        assert result.error is error

    def test_routed_on_error_without_sink_name_raises(self) -> None:
        """ROUTED_ON_ERROR must identify the failure sink for audit lineage."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))
        error = FailureInfo(exception_type="ValueError", message="bad row")

        with pytest.raises(OrchestrationInvariantError, match="ON_ERROR_ROUTED"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                error=error,
            )

    def test_routed_on_error_without_error_raises(self) -> None:
        """ROUTED_ON_ERROR must carry the originating transform failure."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))

        with pytest.raises(OrchestrationInvariantError, match=r"ON_ERROR_ROUTED.*error \(FailureInfo\)"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
            )

    def test_routed_on_error_with_non_failure_info_error_raises(self) -> None:
        """ROUTED_ON_ERROR rejects untyped error evidence at runtime."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data=_wrap_dict_as_pipeline_row({"x": 1}))

        with pytest.raises(OrchestrationInvariantError, match=r"ON_ERROR_ROUTED.*FailureInfo instance"):
            RowResult(
                token=token,
                final_data=_wrap_dict_as_pipeline_row({"x": 1}),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
                error=object(),  # type: ignore[arg-type]
            )


class TestExceptionResult:
    """Tests for ExceptionResult dataclass."""

    def test_exception_result_in_contracts(self) -> None:
        """ExceptionResult should be importable from contracts."""
        from elspeth.contracts import ExceptionResult

        err = ExceptionResult(
            exception=ValueError("test"),
            traceback="Traceback...",
        )
        assert err.exception.args == ("test",)
        assert err.traceback == "Traceback..."

    def test_exception_result_stores_base_exception(self) -> None:
        """ExceptionResult can store any BaseException subclass."""
        from elspeth.contracts import ExceptionResult

        # Regular exception
        err1 = ExceptionResult(exception=ValueError("value error"), traceback="tb1")
        assert isinstance(err1.exception, ValueError)

        # BaseException subclass (KeyboardInterrupt, SystemExit)
        err2 = ExceptionResult(exception=KeyboardInterrupt(), traceback="tb2")
        assert isinstance(err2.exception, KeyboardInterrupt)

    def test_exception_result_in_results_module(self) -> None:
        """ExceptionResult should also be importable from contracts.results."""
        from elspeth.contracts.results import ExceptionResult

        err = ExceptionResult(
            exception=RuntimeError("test runtime"),
            traceback="Traceback (most recent call last):\n...",
        )
        assert isinstance(err.exception, RuntimeError)
        assert "most recent call last" in err.traceback


class TestArtifactDescriptorDeepFreeze:
    """ArtifactDescriptor.metadata must be deeply frozen."""

    def test_nested_metadata_is_frozen(self) -> None:
        """Nested dicts in metadata must become MappingProxyType."""
        from types import MappingProxyType

        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/tmp/test.csv",
            content_hash="abc123",
            size_bytes=100,
            metadata=MappingProxyType({"nested": {"inner_key": "inner_value"}}),
        )
        assert isinstance(descriptor.metadata, MappingProxyType)
        # The nested dict must also be frozen
        assert isinstance(descriptor.metadata["nested"], MappingProxyType)

    def test_nested_list_in_metadata_is_frozen(self) -> None:
        """Nested lists in metadata must become tuples."""
        from types import MappingProxyType

        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="/tmp/test.csv",
            content_hash="abc123",
            size_bytes=100,
            metadata=MappingProxyType({"tags": ["a", "b"]}),
        )
        assert isinstance(descriptor.metadata, MappingProxyType)
        assert isinstance(descriptor.metadata["tags"], tuple)


# ---------------------------------------------------------------------------
# Regression: SourceRow exception type (elspeth-a286241cfb)
# ---------------------------------------------------------------------------


class TestSourceRowExceptionType:
    """SourceRow construction and to_pipeline_row exception types."""

    def test_no_contract_raises_at_construction(self) -> None:
        """Valid SourceRow without contract raises ValueError at construction.

        Bug fix: elspeth-a27e71979f — the invariant is now enforced in
        __post_init__ instead of deferring to to_pipeline_row(). This
        supersedes elspeth-a286241cfb which moved the error to FrameworkBugError.
        """
        from elspeth.contracts.results import SourceRow

        with pytest.raises(ValueError, match=r"[Vv]alid.*contract"):
            SourceRow(row={"id": 1}, is_quarantined=False, contract=None, source_row_index=0)

    def test_quarantined_row_still_raises_value_error(self) -> None:
        """Quarantined rows raise ValueError — this is a state violation, not a bug."""
        from elspeth.contracts.results import SourceRow

        row = SourceRow.quarantined(row={"id": 1}, error="bad data", destination="errors", source_row_index=0)
        with pytest.raises(ValueError, match="quarantined"):
            row.to_pipeline_row()
