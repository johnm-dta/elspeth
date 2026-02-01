# tests/core/landscape/test_token_outcome_constraints.py
"""Tests for token outcome constraint enforcement.

Critical audit integrity tests: Verifies that the partial UNIQUE index on
token_outcomes_table prevents recording multiple terminal outcomes for the
same token, which would corrupt the audit trail.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.enums import NodeType, RowOutcome
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import token_outcomes_table

DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestTokenOutcomeConstraints:
    """Tests for terminal outcome uniqueness constraint (audit integrity)."""

    def test_double_terminal_outcome_raises_integrity_error(self) -> None:
        """Recording two terminal outcomes for same token must fail.

        This is CRITICAL for audit integrity - a token can only have ONE
        terminal state. The database enforces this via partial unique index:
        UNIQUE(token_id) WHERE is_terminal=1

        If this constraint isn't enforced, the audit trail becomes ambiguous:
        "Did token X complete to sink A or sink B?" - both recorded as terminal.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup: Create run, source node, row, and token
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # First terminal outcome: COMPLETED
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output_sink",
        )

        # Second terminal outcome for SAME token must raise IntegrityError
        with pytest.raises(IntegrityError):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.ROUTED,
                sink_name="alternate_sink",
            )

    def test_non_terminal_then_terminal_is_allowed(self) -> None:
        """Non-terminal (BUFFERED) followed by terminal (CONSUMED_IN_BATCH) is valid.

        The aggregation pattern: token enters aggregation (BUFFERED), then
        batch flushes (CONSUMED_IN_BATCH). Both are recorded, but only
        CONSUMED_IN_BATCH is terminal.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        # Need an aggregation node for batch FK
        agg_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create batch to satisfy FK constraint
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node.node_id,
        )

        # Non-terminal: BUFFERED (is_terminal=False)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id=batch.batch_id,
        )

        # Terminal: CONSUMED_IN_BATCH (is_terminal=True) - should succeed
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
            batch_id=batch.batch_id,
        )

        # Verify both outcomes were recorded via direct query
        with db.connection() as conn:
            query = select(token_outcomes_table).where(token_outcomes_table.c.token_id == token.token_id)
            results = conn.execute(query).fetchall()

        assert len(results) == 2
        outcomes = {row.outcome for row in results}
        assert outcomes == {RowOutcome.BUFFERED.value, RowOutcome.CONSUMED_IN_BATCH.value}

    def test_multiple_non_terminal_outcomes_allowed(self) -> None:
        """Multiple non-terminal outcomes for same token are allowed.

        Edge case: A token could theoretically have multiple BUFFERED outcomes
        if it passes through multiple aggregations (though this is rare).
        The constraint only prevents multiple TERMINAL outcomes.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        # Need aggregation nodes for batch FK
        agg_node1 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg_1",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        agg_node2 = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_agg_2",
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Create batches to satisfy FK constraint
        batch1 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node1.node_id,
        )
        batch2 = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg_node2.node_id,
        )

        # Multiple non-terminal outcomes (both BUFFERED in different batches)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id=batch1.batch_id,
        )

        # Second BUFFERED should also succeed (not terminal)
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id=batch2.batch_id,
        )

        # Verify both outcomes were recorded via direct query
        with db.connection() as conn:
            query = select(token_outcomes_table).where(token_outcomes_table.c.token_id == token.token_id)
            results = conn.execute(query).fetchall()

        assert len(results) == 2


class TestTokenOutcomeCanonicalJson:
    """Tests for canonical JSON enforcement in token outcome context.

    Bug: P2-2026-01-31-token-outcome-context-non-canonical

    The context parameter in record_token_outcome() was using json.dumps()
    instead of canonical_json(), allowing NaN/Infinity to slip into the
    audit trail. This violates CLAUDE.md: "NaN and Infinity are strictly
    rejected, not silently converted."
    """

    def test_context_with_nan_raises_value_error(self) -> None:
        """Context containing NaN must be rejected, not silently serialized.

        NaN in audit data corrupts the trail - it can't be deterministically
        hashed and may fail to deserialize in other systems.
        """
        import math

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Context with NaN must raise ValueError (canonical_json rejects NaN)
        with pytest.raises(ValueError, match=r"[Nn]a[Nn]|non-finite"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="output_sink",
                context={"score": math.nan},
            )

    def test_context_with_infinity_raises_value_error(self) -> None:
        """Context containing Infinity must be rejected."""
        import math

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Context with Infinity must raise ValueError
        with pytest.raises(ValueError, match=r"[Ii]nf|non-finite"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="output_sink",
                context={"ratio": math.inf},
            )

    def test_context_with_valid_data_succeeds(self) -> None:
        """Context with normal JSON-serializable data should succeed."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Setup
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"value": 42},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Valid context should work fine
        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output_sink",
            context={"score": 0.95, "label": "approved", "count": 42},
        )

        # Verify it was recorded
        assert outcome_id is not None
        assert outcome_id.startswith("out_")
