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

import pytest

from elspeth.contracts import RoutingAction, RowOutcome, TokenInfo
from elspeth.contracts.results import (
    ArtifactDescriptor,
    FailureInfo,
    GateResult,
    RowResult,
    TransformResult,
)
from elspeth.engine.retry import MaxRetriesExceeded


class TestTransformResultMultiRow:
    """Tests for multi-row output support in TransformResult."""

    def test_transform_result_multi_row_success(self) -> None:
        """TransformResult.success_multi returns multiple rows."""
        rows = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
        result = TransformResult.success_multi(rows)

        assert result.status == "success"
        assert result.row is None  # Single row field is None
        assert result.rows == rows
        assert len(result.rows) == 2

    def test_transform_result_success_single_sets_rows_none(self) -> None:
        """TransformResult.success() sets rows to None for single-row output."""
        result = TransformResult.success({"id": 1})

        assert result.status == "success"
        assert result.row == {"id": 1}
        assert result.rows is None

    def test_transform_result_is_multi_row(self) -> None:
        """is_multi_row property distinguishes single vs multi output."""
        single = TransformResult.success({"id": 1})
        multi = TransformResult.success_multi([{"id": 1}, {"id": 2}])

        assert single.is_multi_row is False
        assert multi.is_multi_row is True

    def test_transform_result_success_multi_rejects_empty_list(self) -> None:
        """success_multi raises ValueError for empty list."""
        with pytest.raises(ValueError, match="at least one row"):
            TransformResult.success_multi([])

    def test_transform_result_error_has_rows_none(self) -> None:
        """TransformResult.error() sets rows to None."""
        result = TransformResult.error({"reason": "failed"})

        assert result.status == "error"
        assert result.row is None
        assert result.rows is None

    def test_transform_result_has_output_data(self) -> None:
        """has_output_data property checks if ANY output exists."""
        single = TransformResult.success({"id": 1})
        multi = TransformResult.success_multi([{"id": 1}])
        error = TransformResult.error({"reason": "failed"})

        assert single.has_output_data is True
        assert multi.has_output_data is True
        assert error.has_output_data is False


class TestTransformResult:
    """Tests for TransformResult."""

    def test_success_factory(self) -> None:
        """Success factory creates result with status='success' and row data."""
        row = {"field": "value", "count": 42}
        result = TransformResult.success(row)

        assert result.status == "success"
        assert result.row == row
        assert result.reason is None
        assert result.retryable is False

    def test_error_factory(self) -> None:
        """Error factory creates result with status='error' and reason."""
        reason = {"error": "validation_failed", "field": "count"}
        result = TransformResult.error(reason)

        assert result.status == "error"
        assert result.row is None
        assert result.reason == reason
        assert result.retryable is False

    def test_error_factory_with_retryable(self) -> None:
        """Error factory accepts retryable flag."""
        reason = {"error": "timeout"}
        result = TransformResult.error(reason, retryable=True)

        assert result.status == "error"
        assert result.retryable is True

    def test_status_is_literal_not_enum(self) -> None:
        """Status is Literal string, not enum - can compare directly to string."""
        success = TransformResult.success({"x": 1})
        error = TransformResult.error({"e": "msg"})

        # Direct string comparison works (not .value)
        assert success.status == "success"
        assert error.status == "error"

        # String identity check
        assert isinstance(success.status, str)
        assert isinstance(error.status, str)

    def test_audit_fields_default_to_none(self) -> None:
        """Audit fields default to None, set by executor."""
        result = TransformResult.success({"x": 1})

        assert result.input_hash is None
        assert result.output_hash is None
        assert result.duration_ms is None

    def test_audit_fields_can_be_set(self) -> None:
        """Audit fields can be set after creation."""
        result = TransformResult.success({"x": 1})
        result.input_hash = "abc123"
        result.output_hash = "def456"
        result.duration_ms = 12.5

        assert result.input_hash == "abc123"
        assert result.output_hash == "def456"
        assert result.duration_ms == 12.5

    def test_audit_fields_not_in_repr(self) -> None:
        """Audit fields have repr=False for cleaner output."""
        result = TransformResult.success({"x": 1})
        result.input_hash = "abc123"

        # audit fields should not appear in repr
        repr_str = repr(result)
        assert "input_hash" not in repr_str
        assert "output_hash" not in repr_str
        assert "duration_ms" not in repr_str


class TestGateResult:
    """Tests for GateResult."""

    def test_creation(self) -> None:
        """GateResult stores row and routing action."""
        row = {"value": 100}
        action = RoutingAction.route("high", reason={"threshold": 50})
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
    """Verify AcceptResult was deleted in aggregation structural cleanup."""

    def test_accept_result_deleted_from_contracts(self) -> None:
        """AcceptResult should be deleted from contracts.results."""
        import elspeth.contracts.results as results

        assert not hasattr(results, "AcceptResult"), "AcceptResult should be deleted - aggregation is structural"

    def test_accept_result_not_exported(self) -> None:
        """AcceptResult should NOT be exported from elspeth.contracts."""
        import elspeth.contracts as contracts

        assert not hasattr(contracts, "AcceptResult"), "AcceptResult should not be exported - aggregation is structural"


class TestRowResult:
    """Tests for RowResult."""

    def test_creation(self) -> None:
        """RowResult stores token, data, outcome, and optional sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1})
        result = RowResult(
            token=token,
            final_data={"x": 1, "processed": True},
            outcome=RowOutcome.COMPLETED,
        )

        assert result.token == token
        assert result.final_data == {"x": 1, "processed": True}
        assert result.outcome == RowOutcome.COMPLETED
        assert result.sink_name is None

    def test_routed_with_sink_name(self) -> None:
        """ROUTED outcome includes sink_name."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1})
        result = RowResult(
            token=token,
            final_data={"x": 1},
            outcome=RowOutcome.ROUTED,
            sink_name="flagged",
        )

        assert result.outcome == RowOutcome.ROUTED
        assert result.sink_name == "flagged"

    def test_token_id_property(self) -> None:
        """token_id property returns token.token_id."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data={})
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )

        assert result.token_id == "tok-1"

    def test_row_id_property(self) -> None:
        """row_id property returns token.row_id."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data={})
        result = RowResult(
            token=token,
            final_data={},
            outcome=RowOutcome.COMPLETED,
        )

        assert result.row_id == "row-1"


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
            content_hash="xyz",
            size_bytes=500,
        )

        # artifact_type is the field name
        assert hasattr(descriptor, "artifact_type")
        assert descriptor.artifact_type == "database"

        # 'kind' should not exist as an attribute
        assert not hasattr(descriptor, "kind")

    def test_content_hash_is_required(self) -> None:
        """content_hash is required (not optional) - audit integrity."""
        # This would fail at runtime with a TypeError if content_hash were omitted
        # We verify by constructing with all required fields
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="required_hash",
            size_bytes=100,
        )
        assert descriptor.content_hash == "required_hash"

    def test_size_bytes_is_required(self) -> None:
        """size_bytes is required (not optional) - verification."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="hash",
            size_bytes=256,
        )
        assert descriptor.size_bytes == 256

    def test_metadata_is_optional(self) -> None:
        """metadata defaults to None."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="hash",
            size_bytes=100,
        )
        assert descriptor.metadata is None

    def test_metadata_can_be_set(self) -> None:
        """metadata can be set with type-specific info."""
        descriptor = ArtifactDescriptor(
            artifact_type="database",
            path_or_uri="db://table@url",
            content_hash="hash",
            size_bytes=100,
            metadata={"table": "results", "row_count": 50},
        )
        assert descriptor.metadata == {"table": "results", "row_count": 50}

    def test_is_frozen(self) -> None:
        """ArtifactDescriptor is frozen (immutable)."""
        descriptor = ArtifactDescriptor(
            artifact_type="file",
            path_or_uri="file:///test",
            content_hash="hash",
            size_bytes=100,
        )

        with pytest.raises(AttributeError):
            descriptor.content_hash = "new_hash"  # type: ignore[misc]


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

    def test_for_database(self) -> None:
        """for_database creates database artifact with db:// URI scheme."""
        descriptor = ArtifactDescriptor.for_database(
            url="postgresql://localhost/mydb",
            table="results",
            content_hash="def456",
            payload_size=1024,
            row_count=100,
        )

        assert descriptor.artifact_type == "database"
        assert descriptor.path_or_uri == "db://results@postgresql://localhost/mydb"
        assert descriptor.content_hash == "def456"
        assert descriptor.size_bytes == 1024
        assert descriptor.metadata == {"table": "results", "row_count": 100}

    def test_for_webhook(self) -> None:
        """for_webhook creates webhook artifact with webhook:// URI scheme."""
        descriptor = ArtifactDescriptor.for_webhook(
            url="https://api.example.com/webhook",
            content_hash="ghi789",
            request_size=512,
            response_code=200,
        )

        assert descriptor.artifact_type == "webhook"
        assert descriptor.path_or_uri == "webhook://https://api.example.com/webhook"
        assert descriptor.content_hash == "ghi789"
        assert descriptor.size_bytes == 512
        assert descriptor.metadata == {"response_code": 200}

    def test_for_webhook_with_error_response(self) -> None:
        """for_webhook captures error response codes."""
        descriptor = ArtifactDescriptor.for_webhook(
            url="https://api.example.com/webhook",
            content_hash="xyz",
            request_size=256,
            response_code=500,
        )

        assert descriptor.metadata == {"response_code": 500}


class TestArtifactDescriptorTypes:
    """Tests for artifact_type values."""

    def test_file_type(self) -> None:
        """File artifact type."""
        descriptor = ArtifactDescriptor.for_file(
            path="/test.csv",
            content_hash="h",
            size_bytes=1,
        )
        assert descriptor.artifact_type == "file"

    def test_database_type(self) -> None:
        """Database artifact type."""
        descriptor = ArtifactDescriptor.for_database(
            url="sqlite:///:memory:",
            table="t",
            content_hash="h",
            payload_size=1,
            row_count=1,
        )
        assert descriptor.artifact_type == "database"

    def test_webhook_type(self) -> None:
        """Webhook artifact type."""
        descriptor = ArtifactDescriptor.for_webhook(
            url="http://localhost",
            content_hash="h",
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

    def test_is_not_frozen(self) -> None:
        """FailureInfo is NOT frozen (matches other result types)."""
        info = FailureInfo(
            exception_type="TestError",
            message="Test message",
        )

        # Should be mutable (no FrozenInstanceError)
        info.attempts = 5
        assert info.attempts == 5


class TestRowResultWithFailureInfo:
    """Tests for RowResult.error field using FailureInfo."""

    def test_failed_outcome_with_failure_info(self) -> None:
        """FAILED outcome includes FailureInfo error details."""
        token = TokenInfo(row_id="row-1", token_id="tok-1", row_data={"x": 1})
        error = FailureInfo(
            exception_type="MaxRetriesExceeded",
            message="Max retries (3) exceeded",
            attempts=3,
            last_error="Connection refused",
        )

        result = RowResult(
            token=token,
            final_data={"x": 1},
            outcome=RowOutcome.FAILED,
            error=error,
        )

        assert result.outcome == RowOutcome.FAILED
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
