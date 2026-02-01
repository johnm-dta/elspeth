"""Tests for LandscapeRecorder reproducibility grade computation."""

from __future__ import annotations

from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.schema import SchemaConfig

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


class TestReproducibilityGradeComputation:
    """Tests for reproducibility grade computation based on node determinism values."""

    def test_pure_pipeline_gets_full_reproducible(self) -> None:
        """Pipeline with only deterministic/seeded nodes gets FULL_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # All deterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="seeded_sampler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.SEEDED,  # seeded counts as reproducible
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_external_calls_gets_replay_reproducible(self) -> None:
        """Pipeline with nondeterministic nodes gets REPLAY_REPRODUCIBLE."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Mix of deterministic and nondeterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_classifier",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,  # LLM call
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        assert grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE

    def test_finalize_run_sets_grade(self) -> None:
        """finalize_run() computes grade and completes the run."""
        from elspeth.contracts import Determinism, RunStatus
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register deterministic nodes
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        completed_run = recorder.finalize_run(run.run_id, status=RunStatus.COMPLETED)

        assert completed_run.status == RunStatus.COMPLETED
        assert completed_run.completed_at is not None
        assert completed_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE.value

    def test_grade_degrades_after_purge(self) -> None:
        """REPLAY_REPRODUCIBLE degrades to ATTRIBUTABLE_ONLY after purge."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize with REPLAY_REPRODUCIBLE grade
        completed_run = recorder.finalize_run(run.run_id, status=RunStatus.COMPLETED)
        assert completed_run.reproducibility_grade == ReproducibilityGrade.REPLAY_REPRODUCIBLE.value

        # Simulate purge - grade should degrade
        update_grade_after_purge(db, run.run_id)

        # Check grade was degraded
        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value

    def test_full_reproducible_unchanged_after_purge(self) -> None:
        """FULL_REPRODUCIBLE remains unchanged after purge (payloads not needed for replay)."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Deterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize with FULL_REPRODUCIBLE grade
        completed_run = recorder.finalize_run(run.run_id, status=RunStatus.COMPLETED)
        assert completed_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE.value

        # Simulate purge - grade should NOT degrade
        update_grade_after_purge(db, run.run_id)

        # Check grade unchanged
        updated_run = recorder.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.reproducibility_grade == ReproducibilityGrade.FULL_REPRODUCIBLE.value

    def test_compute_grade_empty_pipeline(self) -> None:
        """Empty pipeline (no nodes) gets FULL_REPRODUCIBLE."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # No nodes registered
        grade = recorder.compute_reproducibility_grade(run.run_id)

        # Empty pipeline is trivially reproducible
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE

    def test_update_grade_after_purge_nonexistent_run(self) -> None:
        """update_grade_after_purge() silently handles nonexistent run."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.reproducibility import update_grade_after_purge

        db = LandscapeDB.in_memory()

        # Should not raise - silently returns for nonexistent run
        update_grade_after_purge(db, "nonexistent_run_id")

    def test_attributable_only_unchanged_after_purge(self) -> None:
        """ATTRIBUTABLE_ONLY remains unchanged after purge (already at lowest grade)."""
        from elspeth.contracts import Determinism
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import (
            ReproducibilityGrade,
            update_grade_after_purge,
        )

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Nondeterministic pipeline
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.EXTERNAL_CALL,
            schema_config=DYNAMIC_SCHEMA,
        )

        # Finalize and degrade to ATTRIBUTABLE_ONLY
        recorder.finalize_run(run.run_id, status=RunStatus.COMPLETED)
        update_grade_after_purge(db, run.run_id)

        # Verify it's ATTRIBUTABLE_ONLY
        run_after_first_purge = recorder.get_run(run.run_id)
        assert run_after_first_purge is not None
        assert run_after_first_purge.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value

        # Call purge again - should remain ATTRIBUTABLE_ONLY
        update_grade_after_purge(db, run.run_id)

        run_after_second_purge = recorder.get_run(run.run_id)
        assert run_after_second_purge is not None
        assert run_after_second_purge.reproducibility_grade == ReproducibilityGrade.ATTRIBUTABLE_ONLY.value

    def test_default_determinism_counts_as_deterministic(self) -> None:
        """Nodes registered without explicit determinism default to DETERMINISTIC."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.recorder import LandscapeRecorder
        from elspeth.core.landscape.reproducibility import ReproducibilityGrade

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register nodes WITHOUT specifying determinism - should default to DETERMINISTIC
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            # determinism not specified - should default to DETERMINISTIC
        )
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
            # determinism not specified - should default to DETERMINISTIC
        )

        grade = recorder.compute_reproducibility_grade(run.run_id)

        # Since defaults are DETERMINISTIC, should get FULL_REPRODUCIBLE
        assert grade == ReproducibilityGrade.FULL_REPRODUCIBLE
