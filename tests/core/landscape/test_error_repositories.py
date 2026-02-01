"""Tests for error-related repositories.

Tests ValidationErrorRepository, TransformErrorRepository, and
TokenOutcomeRepository which handle error/outcome audit records.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from elspeth.contracts.audit import (
    TokenOutcome,
    TransformErrorRecord,
    ValidationErrorRecord,
)
from elspeth.contracts.enums import RowOutcome
from elspeth.core.landscape.repositories import (
    TokenOutcomeRepository,
    TransformErrorRepository,
    ValidationErrorRepository,
)


class TestValidationErrorRepository:
    """Tests for ValidationErrorRepository."""

    def test_load_validation_error(self) -> None:
        """Load returns ValidationErrorRecord with all fields mapped."""
        created_at = datetime.now(UTC)
        row = MagicMock(
            error_id="verr_abc123",
            run_id="run_1",
            node_id="source_1",
            row_hash="hash_abc",
            error='{"code": "INVALID"}',
            schema_mode="strict",
            destination="quarantine",
            created_at=created_at,
            row_data_json='{"field": "value"}',
        )

        repo = ValidationErrorRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, ValidationErrorRecord)
        assert result.error_id == "verr_abc123"
        assert result.run_id == "run_1"
        assert result.node_id == "source_1"
        assert result.row_hash == "hash_abc"
        assert result.error == '{"code": "INVALID"}'
        assert result.schema_mode == "strict"
        assert result.destination == "quarantine"
        assert result.created_at == created_at
        assert result.row_data_json == '{"field": "value"}'

    def test_load_validation_error_with_null_node_id(self) -> None:
        """Load handles NULL node_id (pre-node validation errors)."""
        row = MagicMock(
            error_id="verr_def456",
            run_id="run_2",
            node_id=None,
            row_hash="hash_def",
            error="Missing required field",
            schema_mode="dynamic",
            destination="quarantine",
            created_at=datetime.now(UTC),
            row_data_json=None,
        )

        repo = ValidationErrorRepository(MagicMock())
        result = repo.load(row)

        assert result.node_id is None
        assert result.row_data_json is None


class TestTransformErrorRepository:
    """Tests for TransformErrorRepository."""

    def test_load_transform_error(self) -> None:
        """Load returns TransformErrorRecord with all fields mapped."""
        created_at = datetime.now(UTC)
        row = MagicMock(
            error_id="terr_abc123",
            run_id="run_1",
            token_id="token_1",
            transform_id="transform_classify",
            row_hash="hash_xyz",
            destination="error_sink",
            created_at=created_at,
            row_data_json='{"input": "data"}',
            error_details_json='{"message": "failed", "code": "E001"}',
        )

        repo = TransformErrorRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, TransformErrorRecord)
        assert result.error_id == "terr_abc123"
        assert result.run_id == "run_1"
        assert result.token_id == "token_1"
        assert result.transform_id == "transform_classify"
        assert result.row_hash == "hash_xyz"
        assert result.destination == "error_sink"
        assert result.created_at == created_at
        assert result.row_data_json == '{"input": "data"}'
        assert result.error_details_json == '{"message": "failed", "code": "E001"}'

    def test_load_transform_error_with_null_optionals(self) -> None:
        """Load handles NULL optional fields."""
        row = MagicMock(
            error_id="terr_def456",
            run_id="run_2",
            token_id="token_2",
            transform_id="transform_validate",
            row_hash="hash_abc",
            destination="quarantine",
            created_at=datetime.now(UTC),
            row_data_json=None,
            error_details_json=None,
        )

        repo = TransformErrorRepository(MagicMock())
        result = repo.load(row)

        assert result.row_data_json is None
        assert result.error_details_json is None


class TestTokenOutcomeRepository:
    """Tests for TokenOutcomeRepository."""

    def test_load_token_outcome_completed(self) -> None:
        """Load returns TokenOutcome with RowOutcome enum conversion."""
        recorded_at = datetime.now(UTC)
        row = MagicMock(
            outcome_id="out_123",
            run_id="run_1",
            token_id="token_1",
            outcome="completed",  # String from database
            is_terminal=True,
            recorded_at=recorded_at,
            sink_name="output",
            batch_id=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            error_hash=None,
            context_json=None,
        )

        repo = TokenOutcomeRepository(MagicMock())
        result = repo.load(row)

        assert isinstance(result, TokenOutcome)
        assert result.outcome_id == "out_123"
        assert result.run_id == "run_1"
        assert result.token_id == "token_1"
        # Critical: outcome must be converted to RowOutcome enum
        assert result.outcome == RowOutcome.COMPLETED
        assert isinstance(result.outcome, RowOutcome)
        assert result.is_terminal is True
        assert result.recorded_at == recorded_at
        assert result.sink_name == "output"

    def test_load_token_outcome_forked(self) -> None:
        """Load handles FORKED outcome with fork_group_id."""
        row = MagicMock(
            outcome_id="out_456",
            run_id="run_1",
            token_id="token_2",
            outcome="forked",
            is_terminal=True,
            recorded_at=datetime.now(UTC),
            sink_name=None,
            batch_id=None,
            fork_group_id="fork_group_abc",
            join_group_id=None,
            expand_group_id=None,
            error_hash=None,
            context_json=None,
        )

        repo = TokenOutcomeRepository(MagicMock())
        result = repo.load(row)

        assert result.outcome == RowOutcome.FORKED
        assert result.fork_group_id == "fork_group_abc"

    def test_load_token_outcome_buffered_non_terminal(self) -> None:
        """Load handles BUFFERED outcome which is non-terminal."""
        row = MagicMock(
            outcome_id="out_789",
            run_id="run_1",
            token_id="token_3",
            outcome="buffered",
            is_terminal=False,  # BUFFERED is non-terminal
            recorded_at=datetime.now(UTC),
            sink_name=None,
            batch_id="batch_abc",
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            error_hash=None,
            context_json=None,
        )

        repo = TokenOutcomeRepository(MagicMock())
        result = repo.load(row)

        assert result.outcome == RowOutcome.BUFFERED
        assert result.is_terminal is False
        assert result.batch_id == "batch_abc"

    def test_load_token_outcome_failed_with_error_hash(self) -> None:
        """Load handles FAILED outcome with error_hash."""
        row = MagicMock(
            outcome_id="out_fail",
            run_id="run_1",
            token_id="token_4",
            outcome="failed",
            is_terminal=True,
            recorded_at=datetime.now(UTC),
            sink_name=None,
            batch_id=None,
            fork_group_id=None,
            join_group_id=None,
            expand_group_id=None,
            error_hash="error_hash_xyz",
            context_json='{"reason": "processing_error"}',
        )

        repo = TokenOutcomeRepository(MagicMock())
        result = repo.load(row)

        assert result.outcome == RowOutcome.FAILED
        assert result.error_hash == "error_hash_xyz"
        assert result.context_json == '{"reason": "processing_error"}'

    def test_load_token_outcome_all_outcome_types(self) -> None:
        """All RowOutcome enum values are correctly converted."""
        for outcome_value in RowOutcome:
            row = MagicMock(
                outcome_id=f"out_{outcome_value.value}",
                run_id="run_1",
                token_id=f"token_{outcome_value.value}",
                outcome=outcome_value.value,  # String from database
                is_terminal=outcome_value.is_terminal,
                recorded_at=datetime.now(UTC),
                sink_name=None,
                batch_id=None,
                fork_group_id=None,
                join_group_id=None,
                expand_group_id=None,
                error_hash=None,
                context_json=None,
            )

            repo = TokenOutcomeRepository(MagicMock())
            result = repo.load(row)

            assert result.outcome == outcome_value
            assert isinstance(result.outcome, RowOutcome)
