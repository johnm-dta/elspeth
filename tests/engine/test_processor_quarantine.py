# tests/engine/test_processor_quarantine.py
"""Quarantine integration tests for RowProcessor.

Tests the full quarantine flow including:
- Pipeline continuation after quarantine
- Audit trail recording for quarantined rows
"""

from typing import Any

from elspeth.contracts import NodeType
from elspeth.contracts.types import NodeID
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import (
    RowOutcome,
    TransformResult,
)
from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema


class TestQuarantineIntegration:
    """Integration tests for full quarantine flow."""

    def test_pipeline_continues_after_quarantine(self) -> None:
        """Pipeline should continue processing after quarantining a row.

        Processes 5 rows with mixed outcomes:
        - 3 positive values -> COMPLETED (validated)
        - 2 negative values -> QUARANTINED (rejected by validator)

        Verifies:
        - All 5 rows are processed
        - Correct outcomes assigned to each
        - Completed rows have "validated" flag added
        - Quarantined rows have original data (not modified)
        """
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="validator",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class ValidatingTransform(BaseTransform):
            """Validator that quarantines negative values (on_error='discard')."""

            name = "validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Intentionally quarantine errors

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                if row["value"] < 0:
                    return TransformResult.error(
                        {
                            "reason": "validation_failed",
                            "error": "negative values not allowed",
                            "value": row["value"],
                        }
                    )
                return TransformResult.success({**row, "validated": True}, success_reason={"action": "validate"})

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        # Process 5 rows: [10, -5, 20, -1, 30]
        test_values = [10, -5, 20, -1, 30]
        all_results: list[Any] = []

        for i, value in enumerate(test_values):
            results = processor.process_row(
                row_index=i,
                row_data={"value": value},
                transforms=[ValidatingTransform(transform.node_id)],
                ctx=ctx,
            )
            all_results.extend(results)

        # Verify 5 results total (one per row)
        assert len(all_results) == 5

        # Verify outcomes
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        quarantined = [r for r in all_results if r.outcome == RowOutcome.QUARANTINED]

        assert len(completed) == 3  # Positive values
        assert len(quarantined) == 2  # Negative values

        # Verify completed rows have "validated" flag
        for result in completed:
            assert result.final_data["validated"] is True
            assert result.final_data["value"] > 0

        # Verify quarantined rows have original data (not modified)
        for result in quarantined:
            assert "validated" not in result.final_data
            assert result.final_data["value"] < 0

    def test_quarantine_records_audit_trail(self) -> None:
        """Quarantined rows should be recorded in audit trail.

        Verifies that when a row is quarantined:
        - The outcome is QUARANTINED
        - A node_state was recorded with status="failed"
        - The node_state record exists in the database
        """
        from elspeth.contracts import NodeStateFailed
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        transform = recorder.register_node(
            run_id=run.run_id,
            plugin_name="strict_validator",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class StrictValidator(BaseTransform):
            """Validator that rejects rows with missing 'required_field'."""

            name = "strict_validator"
            input_schema = _TestSchema
            output_schema = _TestSchema
            _on_error = "discard"  # Quarantine invalid rows

            def __init__(self, node_id: str) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.node_id = node_id

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                # row.get() is allowed here - this is row data (their data, Tier 2)
                if "required_field" not in row:
                    return TransformResult.error(
                        {
                            "reason": "missing_field",
                            "error": "missing required_field",
                        }
                    )
                return TransformResult.success({**row, "validated": True}, success_reason={"action": "validate"})

        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
        )

        # Process an invalid row (missing required_field)
        results = processor.process_row(
            row_index=0,
            row_data={"other_field": "some_value"},
            transforms=[StrictValidator(transform.node_id)],
            ctx=ctx,
        )

        # Single result
        assert len(results) == 1
        result = results[0]

        # Verify outcome is QUARANTINED
        assert result.outcome == RowOutcome.QUARANTINED

        # Verify original data is preserved
        assert result.final_data == {"other_field": "some_value"}

        # Query the node_states table to confirm the record exists
        states = recorder.get_node_states_for_token(result.token.token_id)

        # Should have exactly 1 node_state (for the transform)
        assert len(states) == 1

        state = states[0]
        assert isinstance(state, NodeStateFailed)
        assert state.status.value == "failed"
        assert state.node_id == transform.node_id
        assert state.token_id == result.token.token_id

        # Verify the error was recorded
        assert state.error_json is not None
        import json

        error_data = json.loads(state.error_json)
        assert error_data["reason"] == "missing_field"
        assert error_data["error"] == "missing required_field"
