"""Tests for TransformSuccessReason audit trail integration.

Verifies that success_reason flows from transform through executor
to Landscape audit trail correctly.
"""

from __future__ import annotations

import json

import pytest

from elspeth.contracts import TransformSuccessReason
from elspeth.contracts.audit import NodeStateCompleted
from elspeth.contracts.enums import NodeStateStatus, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTransformSuccessReasonAudit:
    """Tests for success_reason in audit trail."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory database."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    def test_success_reason_stored_in_node_state(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason is stored in node_states table."""
        # Setup run
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run.run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="field_tracking_transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"amount": 100},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create and complete node state with success_reason
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run.run_id,
            step_index=0,
            input_data={"amount": 100},
        )

        success_reason: TransformSuccessReason = {
            "action": "processed",
            "fields_added": ["processed", "amount_usd"],
        }

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"amount": 100, "processed": True, "amount_usd": 100.0},
            duration_ms=5.0,
            success_reason=success_reason,
        )

        # Verify
        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is not None
        parsed = json.loads(completed.success_reason_json)
        assert parsed["action"] == "processed"
        assert parsed["fields_added"] == ["processed", "amount_usd"]

    def test_success_reason_none_when_not_provided(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason_json is NULL when transform doesn't provide it."""
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run.run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run.run_id,
            step_index=0,
            input_data={"x": 1},
        )

        # Complete WITHOUT success_reason
        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=1.0,
        )

        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is None

    def test_validation_warnings_captured(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """validation_warnings flow through to audit trail."""
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run.run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="data_quality_transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"amount": 950},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run.run_id,
            step_index=0,
            input_data={"amount": 950},
        )

        success_reason: TransformSuccessReason = {
            "action": "validated",
            "validation_warnings": ["amount near threshold (950 of 1000 limit)"],
        }

        completed = recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"amount": 950},
            duration_ms=2.0,
            success_reason=success_reason,
        )

        assert isinstance(completed, NodeStateCompleted)
        assert completed.success_reason_json is not None
        parsed = json.loads(completed.success_reason_json)
        assert parsed["action"] == "validated"
        assert len(parsed["validation_warnings"]) == 1
        assert "950" in parsed["validation_warnings"][0]

    def test_success_reason_round_trips_through_repository(
        self,
        recorder: LandscapeRecorder,
    ) -> None:
        """success_reason survives write → read via repository."""
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            node_id="transform_1",
            run_id=run.run_id,
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id="transform_1",
            run_id=run.run_id,
            step_index=0,
            input_data={"x": 1},
        )

        success_reason: TransformSuccessReason = {
            "action": "enriched",
            "fields_added": ["enrichment_score"],
            "metadata": {"source": "external_api"},
        }

        # Write
        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1, "enrichment_score": 0.95},
            duration_ms=10.0,
            success_reason=success_reason,
        )

        # Read back via get_node_state (uses repository)
        loaded = recorder.get_node_state(state.state_id)
        assert isinstance(loaded, NodeStateCompleted)
        assert loaded.success_reason_json is not None
        parsed = json.loads(loaded.success_reason_json)
        assert parsed["action"] == "enriched"
        assert parsed["fields_added"] == ["enrichment_score"]
        assert parsed["metadata"]["source"] == "external_api"
