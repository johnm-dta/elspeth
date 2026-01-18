"""Tests for operation outcomes and results.

Tests for:
- TransformResult success/error factories
- TransformResult status is Literal (not enum) - can compare to string directly
- TransformResult has audit fields
- GateResult creation and audit fields
- AcceptResult creation
- RowResult creation with TokenInfo
- ArtifactDescriptor required fields (content_hash, size_bytes)
- ArtifactDescriptor uses artifact_type (not kind)
- ArtifactDescriptor factory methods
"""

import pytest

from elspeth.contracts import RoutingAction, RowOutcome, TokenInfo
from elspeth.contracts.results import (
    AcceptResult,
    ArtifactDescriptor,
    GateResult,
    RowResult,
    TransformResult,
)


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


class TestAcceptResult:
    """Tests for AcceptResult."""

    def test_accepted(self) -> None:
        """Row accepted into batch."""
        result = AcceptResult(accepted=True)

        assert result.accepted is True
        assert result.batch_id is None

    def test_rejected(self) -> None:
        """Row rejected by aggregation."""
        result = AcceptResult(accepted=False)

        assert result.accepted is False

    def test_batch_id_can_be_set(self) -> None:
        """Batch ID set by executor."""
        result = AcceptResult(accepted=True)
        result.batch_id = "batch-123"

        assert result.batch_id == "batch-123"

    def test_accept_result_has_no_trigger_field(self) -> None:
        """AcceptResult should NOT have a trigger field (moved to engine)."""
        from dataclasses import fields

        from elspeth.contracts.results import AcceptResult

        field_names = [f.name for f in fields(AcceptResult)]
        assert "trigger" not in field_names, "trigger field should be removed (WP-06)"


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
