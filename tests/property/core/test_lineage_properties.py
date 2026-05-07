# tests/property/core/test_lineage_properties.py
"""Property-based tests for lineage query validation.

These tests verify the input validation invariants of explain():

Argument Validation Properties:
- Must provide either token_id or row_id (not neither)
- Can provide both token_id and row_id (token_id takes precedence)
- sink parameter is optional

Error Condition Properties:
- ValueError when neither token_id nor row_id provided
- ValueError when multiple terminal tokens and no sink specified
- ValueError when multiple tokens at same sink (pipeline config issue)

Return Type Properties:
- Returns LineageResult or None
- None when token/row not found
- None when no terminal tokens exist

LineageResult Structure Properties:
- All required fields are populated when result returned
- node_states sorted by step_index
- parent_tokens list populated for fork/coalesce tokens
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.lineage import LineageResult, explain
from tests.strategies.ids import id_strings, sink_names

# =============================================================================
# explain() Input Validation Property Tests
# =============================================================================


class TestExplainInputValidationProperties:
    """Property tests for explain() argument validation."""

    @given(run_id=id_strings)
    @settings(max_examples=50)
    def test_neither_token_nor_row_raises(self, run_id: str) -> None:
        """Property: Must provide either token_id or row_id.

        Calling explain() without any row/token identifier is a
        programming error that should fail fast.
        """
        query = MagicMock()
        data_flow = MagicMock()

        with pytest.raises(ValueError, match="Must provide either token_id or row_id"):
            explain(query, data_flow, run_id)

    @given(run_id=id_strings)
    @settings(max_examples=50)
    def test_none_token_and_none_row_raises(self, run_id: str) -> None:
        """Property: Explicit None values also raise ValueError."""
        query = MagicMock()
        data_flow = MagicMock()

        with pytest.raises(ValueError, match="Must provide either token_id or row_id"):
            explain(query, data_flow, run_id, token_id=None, row_id=None)

    @given(run_id=id_strings, token_id=id_strings)
    @settings(max_examples=50)
    def test_token_id_alone_is_valid(self, run_id: str, token_id: str) -> None:
        """Property: Providing only token_id is valid input."""
        query = MagicMock()
        data_flow = MagicMock()
        query.get_token.return_value = None  # Token not found

        # Should not raise - returns None for not found
        result = explain(query, data_flow, run_id, token_id=token_id)

        assert result is None
        query.get_token.assert_called_once_with(token_id)

    @given(run_id=id_strings, row_id=id_strings)
    @settings(max_examples=50)
    def test_row_id_alone_is_valid(self, run_id: str, row_id: str) -> None:
        """Property: Providing only row_id is valid input."""
        query = MagicMock()
        data_flow = MagicMock()
        data_flow.get_token_outcomes_for_row.return_value = []  # No outcomes

        # Should not raise - returns None for not found
        result = explain(query, data_flow, run_id, row_id=row_id)

        assert result is None

    @given(run_id=id_strings, token_id=id_strings, row_id=id_strings)
    @settings(max_examples=50)
    def test_both_token_and_row_is_valid(self, run_id: str, token_id: str, row_id: str) -> None:
        """Property: Providing both token_id and row_id is valid.

        When both are provided, token_id takes precedence (more specific).
        """
        query = MagicMock()
        data_flow = MagicMock()
        query.get_token.return_value = None  # Token not found

        # Should use token_id path
        result = explain(query, data_flow, run_id, token_id=token_id, row_id=row_id)

        assert result is None
        # Should have queried by token_id, not row_id
        query.get_token.assert_called_once_with(token_id)
        data_flow.get_token_outcomes_for_row.assert_not_called()


# =============================================================================
# explain() Row Resolution Property Tests
# =============================================================================


class TestExplainRowResolutionProperties:
    """Property tests for row_id -> token_id resolution logic."""

    @given(run_id=id_strings, row_id=id_strings)
    @settings(max_examples=50)
    def test_no_outcomes_returns_none(self, run_id: str, row_id: str) -> None:
        """Property: No outcomes for row_id returns None."""
        query = MagicMock()
        data_flow = MagicMock()
        data_flow.get_token_outcomes_for_row.return_value = []

        result = explain(query, data_flow, run_id, row_id=row_id)

        assert result is None

    @given(run_id=id_strings, row_id=id_strings)
    @settings(max_examples=50)
    def test_no_terminal_outcomes_returns_none(self, run_id: str, row_id: str) -> None:
        """Property: Non-terminal outcomes only returns None.

        If all tokens are still processing (e.g., BUFFERED), there's
        no complete lineage to return yet.
        """
        query = MagicMock()
        data_flow = MagicMock()

        # Create mock non-terminal outcome
        non_terminal = MagicMock()
        non_terminal.completed = False

        data_flow.get_token_outcomes_for_row.return_value = [non_terminal]

        result = explain(query, data_flow, run_id, row_id=row_id)

        assert result is None

    @given(run_id=id_strings, row_id=id_strings, sink=sink_names)
    @settings(max_examples=50)
    def test_sink_filter_no_match_returns_none(self, run_id: str, row_id: str, sink: str) -> None:
        """Property: Specified sink with no matching tokens returns None."""
        query = MagicMock()
        data_flow = MagicMock()

        # Terminal outcome at different sink
        terminal = MagicMock()
        terminal.completed = True
        terminal.sink_name = "other_sink"

        data_flow.get_token_outcomes_for_row.return_value = [terminal]

        result = explain(query, data_flow, run_id, row_id=row_id, sink=sink)

        assert result is None


# =============================================================================
# explain() Ambiguity Detection Property Tests
# =============================================================================


class TestExplainAmbiguityProperties:
    """Property tests for ambiguous row resolution detection."""

    @given(run_id=id_strings, row_id=id_strings)
    @settings(max_examples=50)
    def test_multiple_terminals_no_sink_raises(self, run_id: str, row_id: str) -> None:
        """Property: Multiple terminal tokens without sink raises ValueError.

        This is a DAG scenario - fork paths created multiple terminal
        tokens. User must specify which sink to query.
        """
        query = MagicMock()
        data_flow = MagicMock()

        # Two terminal outcomes at different sinks
        terminal1 = MagicMock()
        terminal1.completed = True
        terminal1.sink_name = "sink_a"

        terminal2 = MagicMock()
        terminal2.completed = True
        terminal2.sink_name = "sink_b"

        data_flow.get_token_outcomes_for_row.return_value = [terminal1, terminal2]

        with pytest.raises(ValueError, match="terminal tokens"):
            explain(query, data_flow, run_id, row_id=row_id)

    @given(run_id=id_strings, row_id=id_strings, sink=sink_names)
    @settings(max_examples=50)
    def test_multiple_tokens_same_sink_raises(self, run_id: str, row_id: str, sink: str) -> None:
        """Property: Multiple tokens at same sink raises ValueError.

        This indicates a pipeline configuration issue - fork paths
        should not converge to the same sink without coalescing.
        """
        query = MagicMock()
        data_flow = MagicMock()

        # Two terminal outcomes at SAME sink
        terminal1 = MagicMock()
        terminal1.completed = True
        terminal1.sink_name = sink
        terminal1.token_id = "token_1"

        terminal2 = MagicMock()
        terminal2.completed = True
        terminal2.sink_name = sink
        terminal2.token_id = "token_2"

        data_flow.get_token_outcomes_for_row.return_value = [terminal1, terminal2]

        with pytest.raises(ValueError, match="tokens at sink"):
            explain(query, data_flow, run_id, row_id=row_id, sink=sink)


# =============================================================================
# LineageResult Structure Property Tests
# =============================================================================


class TestLineageResultStructureProperties:
    """Property tests for LineageResult dataclass structure."""

    def test_required_fields_present(self) -> None:
        """Property: LineageResult has all required fields defined."""
        required_fields = {
            "token",
            "source_row",
            "node_states",
            "routing_events",
            "calls",
            "parent_tokens",
        }

        field_names = {f.name for f in fields(LineageResult)}

        assert required_fields.issubset(field_names)

    def test_optional_fields_have_defaults(self) -> None:
        """Property: Optional fields have default values."""
        optional_with_defaults = {
            "validation_errors": tuple,
            "transform_errors": tuple,
            "outcome": type(None),
        }

        for field_obj in fields(LineageResult):
            if field_obj.name in optional_with_defaults:
                # Field should have a default or default_factory
                assert field_obj.default is not None or field_obj.default_factory is not None

    def test_error_tuples_default_to_empty(self) -> None:
        """Property: Error tuples default to empty (not None)."""
        # Create minimal valid LineageResult with matching row and run ownership.
        token = MagicMock()
        token.row_id = "row-1"
        token.run_id = "run-1"
        source_row = MagicMock()
        source_row.row_id = "row-1"
        source_row.run_id = "run-1"
        result = LineageResult(
            token=token,
            source_row=source_row,
            node_states=(),
            routing_events=(),
            calls=(),
            parent_tokens=(),
        )

        assert result.validation_errors == ()
        assert result.transform_errors == ()
        assert result.outcome is None


# =============================================================================
# explain() Tier 1 Trust Property Tests
# =============================================================================


class TestExplainTierOneTrustProperties:
    """Property tests for Tier 1 audit integrity checks in explain()."""

    @given(run_id=id_strings, token_id=id_strings)
    @settings(max_examples=30)
    def test_missing_parent_token_crashes(self, run_id: str, token_id: str) -> None:
        """Property: Missing parent token raises ValueError (Tier 1 integrity).

        If token_parents references a non-existent parent, that's audit
        database corruption. We crash rather than silently skipping.
        """
        query = MagicMock()
        data_flow = MagicMock()

        # Token exists with a fork group ID (required for parent consistency)
        token = MagicMock()
        token.token_id = token_id
        token.row_id = "row_123"
        token.run_id = run_id
        token.fork_group_id = "some-fork-group"
        token.join_group_id = None
        token.expand_group_id = None
        query.get_token.return_value = token

        # Source row exists
        source_row = MagicMock()
        source_row.row_id = "row_123"
        source_row.run_id = run_id
        query.explain_row.return_value = source_row

        # Node states exist
        query.get_node_states_for_token.return_value = []

        # Routing events and calls for states
        query.get_routing_events_for_states.return_value = []
        query.get_calls_for_states.return_value = []

        # Parent reference exists but parent doesn't
        parent_ref = MagicMock()
        parent_ref.parent_token_id = "missing_parent_id"
        query.get_token_parents.return_value = [parent_ref]
        query.get_token.side_effect = lambda tid: token if tid == token_id else None

        with pytest.raises(AuditIntegrityError, match="Audit integrity violation"):
            explain(query, data_flow, run_id, token_id=token_id)


# =============================================================================
# explain() Return Value Property Tests
# =============================================================================


class TestExplainReturnValueProperties:
    """Property tests for explain() return value invariants."""

    @given(run_id=id_strings, token_id=id_strings)
    @settings(max_examples=30)
    def test_token_not_found_returns_none(self, run_id: str, token_id: str) -> None:
        """Property: Non-existent token returns None (not exception)."""
        query = MagicMock()
        data_flow = MagicMock()
        query.get_token.return_value = None

        result = explain(query, data_flow, run_id, token_id=token_id)

        assert result is None

    @given(run_id=id_strings, token_id=id_strings)
    @settings(max_examples=30)
    def test_source_row_not_found_raises_audit_integrity(self, run_id: str, token_id: str) -> None:
        """Property: Token exists but source row missing is Tier 1 corruption — crash."""
        query = MagicMock()
        data_flow = MagicMock()

        token = MagicMock()
        token.row_id = "row_123"
        token.token_id = token_id
        query.get_token.return_value = token
        query.explain_row.return_value = None

        with pytest.raises(AuditIntegrityError, match="does not exist in rows table"):
            explain(query, data_flow, run_id, token_id=token_id)
