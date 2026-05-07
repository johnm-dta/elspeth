# tests/integration/pipeline/orchestrator/test_orchestrator_checkpointing.py
"""Tests for Orchestrator checkpointing functionality.

Migrated from tests/engine/test_orchestrator_checkpointing.py.
Uses v2 fixtures with function-scoped databases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.fixtures.base_classes import (
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.landscape import make_factory
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, FailingSink, ListSource, PassTransform

if TYPE_CHECKING:
    from elspeth.core.landscape import LandscapeDB


class TestOrchestratorCheckpointing:
    """Tests for checkpoint integration in Orchestrator."""

    def test_orchestrator_accepts_checkpoint_manager(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with CheckpointManager."""
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.engine.orchestrator import Orchestrator

        checkpoint_mgr = CheckpointManager(landscape_db)
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
        )
        assert orchestrator._checkpoint_manager is checkpoint_mgr

    def test_orchestrator_accepts_checkpoint_config(self, landscape_db: LandscapeDB) -> None:
        """Orchestrator can be initialized with RuntimeCheckpointConfig."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator

        settings = CheckpointSettings(frequency="every_n", checkpoint_interval=10)
        config = RuntimeCheckpointConfig.from_settings(settings)
        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_config=config,
        )
        assert orchestrator._checkpoint_config == config

    def test_maybe_checkpoint_creates_on_every_row(self, landscape_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint creates checkpoint when frequency=every_row."""

        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassTransform(on_success="default")
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 3

        assert len(checkpoint_calls) == 3, f"Expected 3 checkpoint calls (one per row), got {len(checkpoint_calls)}"

    def test_maybe_checkpoint_respects_interval(self, landscape_db: LandscapeDB, payload_store) -> None:
        """_maybe_checkpoint only creates checkpoint every N rows."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_n", checkpoint_interval=3)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        source = ListSource([{"value": i} for i in range(7)])
        transform = PassTransform(on_success="default")
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 7

        assert len(checkpoint_calls) == 2, f"Expected 2 checkpoint calls (at rows 3 and 6), got {len(checkpoint_calls)}"

    def test_checkpoint_deleted_on_successful_completion(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are deleted when run completes successfully."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = PassTransform(on_success="default")
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        remaining_checkpoints = checkpoint_mgr.get_checkpoints(result.run_id)
        assert len(remaining_checkpoints) == 0

    def test_checkpoint_preserved_on_failure(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Checkpoints are preserved when run fails for recovery."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings, GateSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        gate_config = GateSettings(
            name="split",
            input="transform_out",
            condition="row['value'] % 2 == 1",
            routes={"true": "good", "false": "bad"},
        )

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassTransform(name="passthrough", on_error="discard")
        good_sink = CollectSink(name="good_sink")
        bad_sink = FailingSink(name="bad_sink", error_message="Bad sink failure")

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"good": as_sink(good_sink), "bad": as_sink(bad_sink)},
            gates=[gate_config],
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="Bad sink failure"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        factory = make_factory(landscape_db)
        runs = factory.run_lifecycle.list_runs()
        assert len(runs) >= 1
        run_id = sorted(runs, key=lambda r: r.started_at, reverse=True)[0].run_id

        remaining_checkpoints = checkpoint_mgr.get_checkpoints(run_id)

        # Rows are processed sequentially: row 1 (odd→good) succeeds and creates
        # a checkpoint before row 2 (even→bad) crashes at FailingSink.
        assert len(good_sink.results) >= 1, "At least one odd-valued row should reach good_sink before bad_sink crashes"
        assert len(remaining_checkpoints) == len(good_sink.results), (
            f"Expected {len(good_sink.results)} checkpoints for written rows, got {len(remaining_checkpoints)}"
        )

    def test_checkpoint_disabled_skips_checkpoint_creation(self, landscape_db: LandscapeDB, payload_store) -> None:
        """No checkpoints created when checkpointing is disabled."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.checkpoint import CheckpointManager
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        checkpoint_mgr = CheckpointManager(landscape_db)
        settings = CheckpointSettings(enabled=False)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        checkpoint_calls: list[dict[str, Any]] = []
        original_create = checkpoint_mgr.create_checkpoint

        def tracking_create(*args: Any, **kwargs: Any) -> Any:
            checkpoint_calls.append({"args": args, "kwargs": kwargs})
            return original_create(*args, **kwargs)

        checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

        source = ListSource([{"value": 1}, {"value": 2}])
        transform = PassTransform(on_success="default")
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"

        assert len(checkpoint_calls) == 0, f"Expected 0 checkpoint calls when disabled, got {len(checkpoint_calls)}"

    def test_no_checkpoint_manager_skips_checkpointing(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Orchestrator works fine without checkpoint manager."""
        from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
        from elspeth.core.config import CheckpointSettings
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(settings)

        source = ListSource([{"value": 1}])
        transform = PassTransform(on_success="default")
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db=landscape_db,
            checkpoint_config=checkpoint_config,
        )
        result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        assert result.status == "completed"
        assert result.rows_processed == 1
