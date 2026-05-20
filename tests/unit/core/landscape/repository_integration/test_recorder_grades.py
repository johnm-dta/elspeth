"""Tests for RecorderFactory reproducibility grade computation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import CallStatus, CallType, Determinism, NodeStateStatus, NodeType, RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import calls_table, node_states_table, rows_table, tokens_table

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _insert_purged_call(
    db: LandscapeDB,
    run_id: str,
    node_id: str,
    *,
    state_id: str = "st-purge",
    call_id: str = "call-purge",
) -> None:
    """Insert the minimal row/token/node_state/call chain for a purged response.

    Creates evidence that a nondeterministic node made a call whose response payload
    has been purged (response_hash set, response_ref NULL). This is the condition
    update_grade_after_purge checks before downgrading REPLAY_REPRODUCIBLE.

    The node itself must already be registered via data_flow.register_node() before
    calling this helper so that the FK constraints on node_states are satisfied.

    Args:
        db: LandscapeDB instance
        run_id: Run the call belongs to
        node_id: Already-registered nondeterministic node ID
        state_id: state_id to use for node_states row (must be unique per test)
        call_id: call_id to use for calls row (must be unique per test)
    """
    from elspeth.core.canonical import stable_hash

    now = datetime.now(UTC)
    row_id = f"row-{node_id}"
    token_id = f"tok-{node_id}"

    with db.connection() as conn:
        conn.execute(
            rows_table.insert().values(
                row_id=row_id,
                run_id=run_id,
                source_node_id=node_id,
                row_index=0,
                source_data_hash=stable_hash({"row": row_id}),
                created_at=now,
            )
        )
        conn.execute(
            tokens_table.insert().values(
                token_id=token_id,
                row_id=row_id,
                run_id=run_id,
                created_at=now,
            )
        )
        conn.execute(
            node_states_table.insert().values(
                state_id=state_id,
                token_id=token_id,
                run_id=run_id,
                node_id=node_id,
                step_index=0,
                attempt=0,
                status=NodeStateStatus.COMPLETED.value,
                input_hash="in_hash",
                output_hash="out_hash",
                started_at=now,
            )
        )
        conn.execute(
            calls_table.insert().values(
                call_id=call_id,
                state_id=state_id,
                operation_id=None,
                call_index=0,
                call_type=CallType.HTTP.value,
                status=CallStatus.SUCCESS.value,
                request_hash="req_hash",
                response_hash="resp_hash",  # Proof the payload once existed
                response_ref=None,  # NULL = payload has been purged
                created_at=now,
            )
        )


class TestReproducibilityGradeComputation:
    """Tests for reproducibility grade computation based on node determinism values."""

    def test_pure_pipeline_gets_full_reproducible(self) -> None:
        """Pipeline with only deterministic/seeded nodes gets FULL_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # All deterministic nodes
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="seeded_sampler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.SEEDED,  # seeded counts as reproducible
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        grade = factory.run_lifecycle.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_external_calls_gets_replay_reproducible(self) -> None:
        """Pipeline with nondeterministic nodes gets REPLAY_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Mix of deterministic and nondeterministic nodes
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="llm_classifier",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,  # LLM call
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        grade = factory.run_lifecycle.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_finalize_run_sets_grade(self) -> None:
        """finalize_run() computes grade and completes the run."""
        from elspeth.contracts import Determinism, RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register deterministic nodes
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        completed_run = factory.run_lifecycle.finalize_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed_run.status == RunStatus.COMPLETED
        assert completed_run.completed_at is not None
        assert completed_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_grade_degrades_after_purge(self) -> None:
        """REPLAY_REPRODUCIBLE degrades to ATTRIBUTABLE_ONLY after purge.

        Degradation requires evidence that replay-critical payloads were purged:
        a call from a nondeterministic node with response_hash set (payload existed)
        but response_ref NULL (payload has been deleted).
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize with REPLAY_REPRODUCIBLE grade
        completed_run = factory.run_lifecycle.finalize_run(run.run_id, status=RunStatus.COMPLETED)
        assert completed_run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

        # Simulate a purged response: insert a call with response_hash set but
        # response_ref NULL — this is the evidence update_grade_after_purge looks for.
        _insert_purged_call(db, run.run_id, node_id=node.node_id)

        # Simulate purge - grade should degrade because replay-critical payload is gone
        update_grade_after_purge(db, run.run_id)

        # Check grade was degraded
        updated_run = factory.run_lifecycle.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

    def test_full_reproducible_unchanged_after_purge(self) -> None:
        """FULL_REPRODUCIBLE remains unchanged after purge (payloads not needed for replay)."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Deterministic pipeline
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize with FULL_REPRODUCIBLE grade
        completed_run = factory.run_lifecycle.finalize_run(run.run_id, status=RunStatus.COMPLETED)
        assert completed_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

        # Simulate purge - grade should NOT degrade
        update_grade_after_purge(db, run.run_id)

        # Check grade unchanged
        updated_run = factory.run_lifecycle.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_compute_grade_empty_pipeline(self) -> None:
        """Empty pipeline (no nodes) gets FULL_REPRODUCIBLE."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # No nodes registered
        grade = factory.run_lifecycle.compute_reproducibility_grade(run.run_id)

        # Empty pipeline is trivially reproducible
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_update_grade_after_purge_nonexistent_run_raises(self) -> None:
        """update_grade_after_purge() crashes on nonexistent run — caller bug or corruption."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.reproducibility import update_grade_after_purge

        db = LandscapeDB.in_memory()

        with pytest.raises(AuditIntegrityError, match="does not exist"):
            update_grade_after_purge(db, "nonexistent_run_id")

    def test_attributable_only_unchanged_after_purge(self) -> None:
        """ATTRIBUTABLE_ONLY remains unchanged after purge (already at lowest grade).

        First purge degrades REPLAY_REPRODUCIBLE → ATTRIBUTABLE_ONLY (requires evidence
        of purged replay-critical payloads). Second purge is a no-op.
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        node = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize with REPLAY_REPRODUCIBLE grade
        factory.run_lifecycle.finalize_run(run.run_id, status=RunStatus.COMPLETED)

        # Insert a purged call record — evidence that a replay-critical payload was deleted
        _insert_purged_call(db, run.run_id, node_id=node.node_id)

        # First purge: degrades REPLAY_REPRODUCIBLE → ATTRIBUTABLE_ONLY
        update_grade_after_purge(db, run.run_id)

        # Verify it's ATTRIBUTABLE_ONLY
        run_after_first_purge = factory.run_lifecycle.get_run(run.run_id)
        assert run_after_first_purge is not None
        assert run_after_first_purge.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

        # Second purge: no-op, already at lowest grade
        update_grade_after_purge(db, run.run_id)

        run_after_second_purge = factory.run_lifecycle.get_run(run.run_id)
        assert run_after_second_purge is not None
        assert run_after_second_purge.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY

    def test_default_determinism_counts_as_deterministic(self) -> None:
        """Nodes registered without explicit determinism default to DETERMINISTIC."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        # Register nodes WITHOUT specifying determinism - should default to DETERMINISTIC
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            # determinism not specified - should default to DETERMINISTIC
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            # determinism not specified - should default to DETERMINISTIC
        )

        grade = factory.run_lifecycle.compute_reproducibility_grade(run.run_id)

        # Since defaults are DETERMINISTIC, should get FULL_REPRODUCIBLE
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE
