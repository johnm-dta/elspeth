"""Mutation gap tests for core/landscape/lineage.py.

Tests targeting specific mutation survivors:
- Line 56-58: transform_errors defaults to empty list (not None)
- Line 59: outcome defaults to None
"""

from datetime import UTC, datetime

import pytest

from elspeth.contracts import RowLineage, Token
from elspeth.core.landscape.lineage import LineageResult


class TestLineageResultDefaults:
    """Tests for LineageResult dataclass field defaults.

    Targets lines 56-59: default values for optional fields.
    """

    @pytest.fixture
    def minimal_token(self) -> Token:
        """Create minimal Token for testing."""
        return Token(
            token_id="test-token",
            row_id="test-row",
            created_at=datetime.now(UTC),
        )

    @pytest.fixture
    def minimal_row_lineage(self) -> RowLineage:
        """Create minimal RowLineage for testing."""
        return RowLineage(
            row_id="test-row",
            run_id="test-run",
            source_node_id="source-node",
            row_index=0,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
            source_data={"field": "value"},
            payload_available=True,
        )

    def test_transform_errors_defaults_to_empty_list(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """Line 56: transform_errors must default to empty list, not None.

        Mutant might change default_factory=list to None or remove the default.
        """
        result = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            # NOT providing transform_errors - should default
        )

        # Must be empty list, not None
        assert result.transform_errors == [], f"transform_errors should default to [], got {result.transform_errors!r}"
        assert result.transform_errors is not None, "transform_errors should default to empty list, not None"
        assert isinstance(result.transform_errors, list), f"transform_errors should be list, got {type(result.transform_errors)}"

    def test_transform_errors_default_is_independent_per_instance(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """Line 56: Each instance should get its own empty list.

        Ensures default_factory=list creates new list per instance,
        not a shared mutable default.
        """
        result1 = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        result2 = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        # Should be different list instances (not shared)
        assert result1.transform_errors is not result2.transform_errors, "Each LineageResult should have its own transform_errors list"

    def test_outcome_defaults_to_none(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """Line 59: outcome must default to None.

        Mutant might change None to a different default value.
        """
        result = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            # NOT providing outcome - should default to None
        )

        assert result.outcome is None, f"outcome should default to None, got {result.outcome!r}"

    def test_validation_errors_defaults_to_empty_list(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """Line 53: validation_errors must default to empty list.

        Complements transform_errors test - same pattern.
        """
        result = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            # NOT providing validation_errors - should default
        )

        assert result.validation_errors == [], f"validation_errors should default to [], got {result.validation_errors!r}"
        assert result.validation_errors is not None

    def test_validation_errors_default_is_independent_per_instance(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """Line 53: Each instance should get its own empty list for validation_errors.

        Ensures default_factory=list creates new list per instance,
        not a shared mutable default. Mirrors transform_errors test.
        """
        result1 = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        result2 = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
        )

        # Should be different list instances (not shared)
        assert result1.validation_errors is not result2.validation_errors, "Each LineageResult should have its own validation_errors list"
        # Mutation on one should not affect the other
        # Note: Intentionally appending wrong type to test list isolation, not type correctness
        result1.validation_errors.append("test")  # type: ignore[arg-type]
        assert result2.validation_errors == [], "Shared mutable default detected"


class TestLineageResultFieldTypes:
    """Tests ensuring field types are correct."""

    @pytest.fixture
    def minimal_token(self) -> Token:
        """Create minimal Token for testing."""
        return Token(
            token_id="test-token",
            row_id="test-row",
            created_at=datetime.now(UTC),
        )

    @pytest.fixture
    def minimal_row_lineage(self) -> RowLineage:
        """Create minimal RowLineage for testing."""
        return RowLineage(
            row_id="test-row",
            run_id="test-run",
            source_node_id="source-node",
            row_index=0,
            source_data_hash="abc123",
            created_at=datetime.now(UTC),
            source_data={"field": "value"},
            payload_available=True,
        )

    def test_transform_errors_accepts_list_of_records(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """transform_errors should accept a list of TransformErrorRecord."""
        from elspeth.contracts import TransformErrorRecord

        error = TransformErrorRecord(
            error_id="err-001",
            run_id="test-run",
            token_id="test-token",
            transform_id="transform-001",
            row_hash="abc123",
            destination="error_sink",
            created_at=datetime.now(UTC),
        )

        result = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            transform_errors=[error],
        )

        assert len(result.transform_errors) == 1
        assert result.transform_errors[0].error_id == "err-001"

    def test_outcome_accepts_token_outcome(self, minimal_token: Token, minimal_row_lineage: RowLineage) -> None:
        """outcome should accept TokenOutcome dataclass."""
        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="outcome-001",
            run_id="test-run",
            token_id="test-token",
            outcome=RowOutcome.COMPLETED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
        )

        result = LineageResult(
            token=minimal_token,
            source_row=minimal_row_lineage,
            node_states=[],
            routing_events=[],
            calls=[],
            parent_tokens=[],
            outcome=outcome,
        )

        assert result.outcome is not None
        assert result.outcome.outcome == RowOutcome.COMPLETED
