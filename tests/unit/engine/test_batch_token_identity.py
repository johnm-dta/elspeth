# tests/unit/engine/test_batch_token_identity.py
"""Token identity tests for batch aggregation.

These tests verify that aggregation correctly creates NEW output tokens
instead of reusing input tokens. This catches bug elspeth-rapid-nd3.

The bug was that `output_mode='single'` reused the triggering token's ID
as the output, breaking audit lineage. These tests verify:
1. ALL batch members have CONSUMED_IN_BATCH outcome (not just count)
2. The output token has a DIFFERENT token_id from all inputs

These are identity-based regression tests, not count-based tests. They verify
WHICH SPECIFIC tokens have WHICH outcomes, preventing bugs where the wrong
tokens get wrong outcomes but counts still match.
"""

from typing import Any

from elspeth.contracts import Determinism, NodeType, SourceRow
from elspeth.contracts.enums import RowOutcome
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, TriggerConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.engine.processor import DAGTraversalContext, RowProcessor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from elspeth.testing import make_field, make_pipeline_row
from tests.unit.engine.conftest import DYNAMIC_SCHEMA, _TestSchema

# ---------------------------------------------------------------------------
# Audit assertion helpers (inlined from tests/helpers/audit_assertions.py)
# ---------------------------------------------------------------------------


def _assert_all_batch_members_consumed(
    recorder: LandscapeRecorder,
    run_id: str,
    batch_id: str,
) -> None:
    """Assert ALL tokens in a batch have CONSUMED_IN_BATCH outcome."""
    with recorder._db.connection() as conn:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import batch_members_table, token_outcomes_table

        members = conn.execute(
            select(batch_members_table.c.token_id, batch_members_table.c.ordinal)
            .where(batch_members_table.c.batch_id == batch_id)
            .order_by(batch_members_table.c.ordinal)
        ).fetchall()

        assert len(members) > 0, f"Batch {batch_id} has no members"

        for member in members:
            token_id = member.token_id
            ordinal = member.ordinal

            outcome_row = conn.execute(select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.token_id == token_id)).fetchone()

            assert outcome_row is not None, f"Batch member {token_id} (ordinal {ordinal}) has no outcome recorded"

            actual = RowOutcome(outcome_row.outcome)
            assert actual == RowOutcome.CONSUMED_IN_BATCH, (
                f"Batch member {token_id} (ordinal {ordinal}) has outcome {actual}, expected CONSUMED_IN_BATCH."
            )


def _assert_output_token_distinct_from_inputs(
    output_token_id: str,
    input_token_ids: list[str],
) -> None:
    """Assert output token has a DIFFERENT token_id from all inputs."""
    assert output_token_id not in input_token_ids, (
        f"Output token {output_token_id} reuses an input token_id! "
        f"Token-producing operations must create NEW tokens for audit lineage. "
        f"Input token_ids: {input_token_ids}"
    )


def _single_node_traversal(node_id: NodeID, plugin: BaseTransform) -> DAGTraversalContext:
    return DAGTraversalContext(
        node_step_map={node_id: 1},
        node_to_plugin={node_id: plugin},
        first_transform_node_id=node_id,
        node_to_next={node_id: None},
        coalesce_node_map={},
    )


class SumTransform(BaseTransform):
    """Sums values in a batch, outputs single aggregated row."""

    name = "summer"
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    creates_tokens = True  # Transform mode: creates new tokens
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0"

    def __init__(self, node_id: str) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.node_id = node_id
        self.on_success = "output"

    def process(self, rows: list[dict[str, Any]] | PipelineRow, ctx: PluginContext) -> TransformResult:
        if isinstance(rows, list):
            total = sum(r.get("value", 0) for r in rows)
            output_row = {"total": total}
            # Create contract for the output row
            fields = tuple(make_field(key) for key in output_row)
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            return TransformResult.success(PipelineRow(output_row, contract), success_reason={"action": "sum"})
        return TransformResult.success(rows, success_reason={"action": "passthrough"})


class TestBatchTokenIdentity:
    """Tests that batch aggregation creates distinct output tokens."""

    def test_all_batch_members_consumed_in_batch(self) -> None:
        """ALL tokens in a batch must have CONSUMED_IN_BATCH outcome.

        This is the core regression test for elspeth-rapid-nd3.
        The bug was that the triggering token (last in batch) got
        COMPLETED instead of CONSUMED_IN_BATCH.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                input="default",
                trigger=TriggerConfig(count=3),  # Batch of 3
                output_mode="transform",
            ),
        }

        transform = SumTransform(agg_node.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            source_on_success="default",
            traversal=_single_node_traversal(NodeID(agg_node.node_id), transform),
            aggregation_settings=aggregation_settings,
        )
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows to trigger batch flush
        all_results = []
        input_token_ids = []
        for i in range(3):
            pipeline_row = make_pipeline_row({"value": (i + 1) * 10})  # 10, 20, 30
            source_row = SourceRow.valid(pipeline_row.to_dict(), contract=pipeline_row.contract)
            results = processor.process_row(
                row_index=i,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)
            # Collect token IDs from CONSUMED_IN_BATCH results
            for r in results:
                if r.outcome == RowOutcome.CONSUMED_IN_BATCH:
                    input_token_ids.append(r.token.token_id)

        # Get the batch_id from the recorder
        from sqlalchemy import select

        from elspeth.core.landscape.schema import batches_table

        with recorder._db.connection() as conn:
            batch = conn.execute(select(batches_table).where(batches_table.c.run_id == run.run_id)).fetchone()
            assert batch is not None, "Batch record should exist"
            batch_id = batch.batch_id

        # CRITICAL ASSERTION: All batch members must be CONSUMED_IN_BATCH
        _assert_all_batch_members_consumed(recorder, run.run_id, batch_id)

        # Get output token
        completed = [r for r in all_results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed) == 1, f"Expected 1 COMPLETED, got {len(completed)}"
        output_token_id = completed[0].token.token_id

        # CRITICAL ASSERTION: Output token must be DISTINCT from inputs
        _assert_output_token_distinct_from_inputs(output_token_id, input_token_ids)

        # Verify aggregation result
        final_data = completed[0].final_data
        assert final_data.to_dict()["total"] == 60  # 10 + 20 + 30

    def test_triggering_token_not_reused(self) -> None:
        """The token that triggers the flush must NOT be reused as output.

        This specifically tests the bug scenario: the 2nd row triggers
        the batch flush. Its token_id must NOT appear in the output.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                input="default",
                trigger=TriggerConfig(count=2),  # Batch of 2
                output_mode="transform",
            ),
        }

        transform = SumTransform(agg_node.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            source_on_success="default",
            traversal=_single_node_traversal(NodeID(agg_node.node_id), transform),
            aggregation_settings=aggregation_settings,
        )
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process row 0 - buffered, returns CONSUMED_IN_BATCH
        pipeline_row_0 = make_pipeline_row({"value": 10})
        source_row_0 = SourceRow.valid(pipeline_row_0.to_dict(), contract=pipeline_row_0.contract)
        results_0 = processor.process_row(
            row_index=0,
            source_row=source_row_0,
            transforms=[transform],
            ctx=ctx,
        )
        assert len(results_0) == 1
        assert results_0[0].outcome == RowOutcome.CONSUMED_IN_BATCH
        first_token_id = results_0[0].token.token_id

        # Process row 1 - triggers flush
        pipeline_row_1 = make_pipeline_row({"value": 20})
        source_row_1 = SourceRow.valid(pipeline_row_1.to_dict(), contract=pipeline_row_1.contract)
        results_1 = processor.process_row(
            row_index=1,
            source_row=source_row_1,
            transforms=[transform],
            ctx=ctx,
        )

        # Should have: CONSUMED_IN_BATCH (triggering token) + COMPLETED (output)
        consumed = [r for r in results_1 if r.outcome == RowOutcome.CONSUMED_IN_BATCH]
        completed = [r for r in results_1 if r.outcome == RowOutcome.COMPLETED]

        assert len(consumed) == 1, "Triggering token must be CONSUMED_IN_BATCH"
        assert len(completed) == 1, "Must have exactly 1 COMPLETED output"

        triggering_token_id = consumed[0].token.token_id
        output_token_id = completed[0].token.token_id

        # THE BUG: In the old code, output_token_id == triggering_token_id
        # THE FIX: They must be different
        assert output_token_id != triggering_token_id, (
            f"Output token {output_token_id} should NOT equal triggering token! "
            f"This is the elspeth-rapid-nd3 bug: token reuse breaks audit lineage."
        )
        assert output_token_id != first_token_id, "Output token should not equal first buffered token either"

        # Verify output data is correct
        final_data = completed[0].final_data
        assert final_data.to_dict()["total"] == 30  # 10 + 20

    def test_batch_members_correctly_recorded(self) -> None:
        """Batch members table should contain all input tokens.

        This verifies that the batch membership is correctly recorded
        in the audit trail, which is essential for explaining lineage.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        source_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="summer",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        aggregation_settings = {
            NodeID(agg_node.node_id): AggregationSettings(
                name="batch_sum",
                plugin="summer",
                input="default",
                trigger=TriggerConfig(count=3),  # Batch of 3
                output_mode="transform",
            ),
        }

        transform = SumTransform(agg_node.node_id)
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source_node.node_id),
            source_on_success="default",
            traversal=_single_node_traversal(NodeID(agg_node.node_id), transform),
            aggregation_settings=aggregation_settings,
        )
        ctx = PluginContext(run_id=run.run_id, config={})

        # Process 3 rows to trigger batch flush
        all_results = []
        input_token_ids = []
        for i in range(3):
            pipeline_row = make_pipeline_row({"value": (i + 1) * 10})
            source_row = SourceRow.valid(pipeline_row.to_dict(), contract=pipeline_row.contract)
            results = processor.process_row(
                row_index=i,
                source_row=source_row,
                transforms=[transform],
                ctx=ctx,
            )
            all_results.extend(results)
            for r in results:
                if r.outcome == RowOutcome.CONSUMED_IN_BATCH:
                    input_token_ids.append(r.token.token_id)

        # Verify we got all 3 consumed tokens
        assert len(input_token_ids) == 3, f"Expected 3 consumed tokens, got {len(input_token_ids)}"

        # Verify batch_members table records all input tokens
        from sqlalchemy import select

        from elspeth.core.landscape.schema import batch_members_table, batches_table

        with recorder._db.connection() as conn:
            # Find the batch
            batch = conn.execute(select(batches_table).where(batches_table.c.run_id == run.run_id)).fetchone()
            assert batch is not None, "Batch should exist in audit trail"

            # Check batch_members contains all input tokens
            members = conn.execute(
                select(batch_members_table.c.token_id).where(batch_members_table.c.batch_id == batch.batch_id)
            ).fetchall()
            member_token_ids = {m.token_id for m in members}

            assert member_token_ids == set(input_token_ids), (
                f"batch_members should contain all input tokens.\nExpected: {set(input_token_ids)}\nGot: {member_token_ids}"
            )
