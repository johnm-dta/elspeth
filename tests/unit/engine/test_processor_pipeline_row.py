# tests/unit/engine/test_processor_pipeline_row.py
"""Tests for RowProcessor with PipelineRow support (Task 6)."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.engine.processor import DAGTraversalContext
from elspeth.engine.spans import SpanFactory
from elspeth.testing import make_field, make_row, make_source_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory, make_landscape_db


def _make_contract() -> SchemaContract:
    """Create a minimal schema contract for testing."""
    return SchemaContract(
        fields=(
            make_field(
                "amount",
                python_type=int,
                original_name="'Amount'",
                required=True,
                source="declared",
            ),
        ),
        mode="OBSERVED",
        locked=True,
    )


def _make_mock_factory() -> MagicMock:
    """Create a mock RecorderFactory with execution and data_flow sub-mocks."""
    factory = MagicMock()
    factory.data_flow.create_row.return_value = Mock(row_id="row_001")
    factory.data_flow.create_token.return_value = Mock(token_id="token_001")
    return factory


def _make_mock_span_factory() -> SpanFactory:
    """Create a real SpanFactory with no tracer — all spans are no-ops."""
    return SpanFactory()


def _empty_traversal(source_node_id: str = "source_001") -> DAGTraversalContext:
    source_node = NodeID(source_node_id)
    return DAGTraversalContext(
        node_step_map={source_node: 0},
        node_to_plugin={},
        first_transform_node_id=None,
        node_to_next={source_node: None},
        coalesce_node_map={},
    )


class TestRowProcessorPipelineRow:
    """Tests for RowProcessor.process_row() with SourceRow."""

    def test_constructor_allows_missing_source_plugin_when_traversal_omits_source(self) -> None:
        """Traversal metadata excludes source nodes, so source_plugin may remain None."""
        from elspeth.engine.processor import RowProcessor

        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            source_plugin=None,
            traversal=_empty_traversal(),
        )

        assert processor._source_plugin is None

    def test_process_row_accepts_source_row(self) -> None:
        """process_row should accept SourceRow and pass it to create_initial_token."""
        from elspeth.engine.processor import RowProcessor

        contract = _make_contract()
        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            source_plugin=None,
            traversal=_empty_traversal(),
        )

        source_row = make_source_row({"amount": 100}, contract=contract)
        landscape_db = make_landscape_db()
        landscape_factory = make_factory(landscape_db)
        ctx = make_context(run_id="run_001", landscape=landscape_factory)

        # No transforms - token should be created and completed immediately
        processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        # Should have created a row and token via data_flow repository
        factory.data_flow.create_row.assert_called_once()
        factory.data_flow.create_token.assert_called_once()

    def test_process_row_creates_pipeline_row(self) -> None:
        """process_row should create token with PipelineRow containing contract."""
        from elspeth.engine.processor import RowProcessor

        contract = _make_contract()
        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            traversal=_empty_traversal(),
        )

        source_row = make_source_row({"amount": 100}, contract=contract)
        landscape_db = make_landscape_db()
        landscape_factory = make_factory(landscape_db)
        ctx = make_context(run_id="run_001", landscape=landscape_factory)

        results = processor.process_row(
            row_index=0,
            source_row=source_row,
            transforms=[],
            ctx=ctx,
        )

        # Result should have token with PipelineRow
        assert len(results) >= 1
        result = results[0]
        assert isinstance(result.token.row_data, PipelineRow)
        assert result.token.row_data["amount"] == 100
        assert result.token.row_data.contract is contract

    def test_process_row_requires_contract_on_source_row(self) -> None:
        """process_row should raise if SourceRow has no contract.

        The error propagates from TokenManager.create_initial_token(),
        which enforces this requirement.
        """
        from elspeth.engine.processor import RowProcessor

        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()

        RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            traversal=_empty_traversal(),
        )

        # Since elspeth-a27e71979f, SourceRow.__post_init__ rejects contract=None
        # at construction time, so the engine's guard is now unreachable via
        # normal construction. Verify the earlier guard fires instead.
        from elspeth.contracts import SourceRow

        with pytest.raises(TypeError, match="contract"):
            SourceRow.valid({"amount": 100})


class TestRowProcessorExistingRow:
    """Tests for RowProcessor.process_existing_row() with PipelineRow."""

    def test_process_existing_row_accepts_pipeline_row(self) -> None:
        """process_existing_row should accept PipelineRow for resume scenarios."""
        from elspeth.engine.processor import RowProcessor

        contract = _make_contract()
        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            traversal=_empty_traversal(),
        )

        # PipelineRow for resume (row already exists in database)
        row_data = make_row({"amount": 100}, contract=contract)
        landscape_db = make_landscape_db()
        landscape_factory = make_factory(landscape_db)
        ctx = make_context(run_id="run_001", landscape=landscape_factory)

        results = processor.process_existing_row(
            row_id="existing_row_001",
            row_data=row_data,
            transforms=[],
            ctx=ctx,
        )

        # Should create token for existing row (NOT create_row)
        factory.data_flow.create_token.assert_called_once()
        factory.data_flow.create_row.assert_not_called()

        # Result should have token with PipelineRow
        assert len(results) >= 1
        result = results[0]
        assert isinstance(result.token.row_data, PipelineRow)
        assert result.token.row_data.contract is contract

    def test_process_existing_row_does_not_run_source_boundary_checks(self) -> None:
        """Resume path reuses original source provenance and skips source boundary VAL."""
        from elspeth.engine.processor import RowProcessor

        contract = _make_contract()
        factory = _make_mock_factory()
        span_factory = _make_mock_span_factory()
        source_plugin = type("ResumeSourcePlugin", (), {})()
        source_plugin.name = "resume-source"
        source_plugin.node_id = "source_001"
        source_plugin.declared_guaranteed_fields = frozenset({"amount"})

        processor = RowProcessor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=span_factory,
            run_id="run_001",
            source_node_id=NodeID("source_001"),
            source_on_success="default",
            source_plugin=source_plugin,
            traversal=_empty_traversal(),
        )

        row_data = make_row({"amount": 100}, contract=contract)
        landscape_db = make_landscape_db()
        landscape_factory = make_factory(landscape_db)
        ctx = make_context(run_id="run_001", landscape=landscape_factory)

        with patch("elspeth.engine.processor.run_boundary_checks") as boundary_check:
            processor.process_existing_row(
                row_id="existing_row_001",
                row_data=row_data,
                transforms=[],
                ctx=ctx,
            )

        boundary_check.assert_not_called()
