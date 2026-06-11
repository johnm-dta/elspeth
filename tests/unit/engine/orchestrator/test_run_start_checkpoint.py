# tests/unit/engine/orchestrator/test_run_start_checkpoint.py
"""Sequence-0 run-start checkpoint (F1 design D4).

Every checkpointing-enabled run writes a baseline checkpoint row (sequence 0,
no barrier scalars) BEFORE source iteration. This makes resume topology
validation unconditional: a run with no checkpoint rows at all genuinely
predates run-start checkpointing or ran with checkpointing disabled
(can_resume's missing-baseline refusal, F1 Task 3.2).

Post-sink checkpoints PRE-increment the sequence counter (0 -> 1 on first
fire), so the baseline never collides; the manager's duplicate-sequence
guard is the backstop. The resume path rebases onto the persisted sequence
and must NOT rewrite sequence 0.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.core.canonical import compute_full_topology_hash
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import checkpoints_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, FailingSink, ListSource, PassTransform
from tests.fixtures.stores import MockPayloadStore


def _checkpoint_config(*, enabled: bool = True) -> RuntimeCheckpointConfig:
    return RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=enabled, frequency="every_row"))


def _select_checkpoints(db: LandscapeDB, run_id: str) -> list[Any]:
    with db.engine.connect() as conn:
        return list(
            conn.execute(
                select(checkpoints_table).where(checkpoints_table.c.run_id == run_id).order_by(checkpoints_table.c.sequence_number)
            ).fetchall()
        )


class TestRunStartCheckpoint:
    def test_run_writes_sequence_zero_checkpoint_at_start(self) -> None:
        """A checkpointing-enabled run persists a sequence-0 topology baseline.

        The sink crashes on the first write, so NO post-sink checkpoint ever
        fires — the surviving row is exactly the run-start baseline: sequence
        0, NULL barrier scalars, full-topology hash of the executed graph.
        """
        db = LandscapeDB.in_memory()
        try:
            payload_store = MockPayloadStore()
            checkpoint_mgr = CheckpointManager(db)

            source = ListSource([{"value": 1}])
            sink = FailingSink(error_message="boom at first write")
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[],
                sinks={"default": as_sink(sink)},
            )
            graph = build_production_graph(config)

            orchestrator = Orchestrator(
                db=db,
                checkpoint_manager=checkpoint_mgr,
                checkpoint_config=_checkpoint_config(),
            )
            with pytest.raises(RuntimeError, match="boom at first write"):
                orchestrator.run(config, graph=graph, payload_store=payload_store, run_id="run-start-cp")

            rows = _select_checkpoints(db, "run-start-cp")
            assert len(rows) == 1, f"Expected exactly the run-start baseline, got {len(rows)} rows"
            assert rows[0].sequence_number == 0
            assert rows[0].barrier_scalars_json is None
            assert rows[0].upstream_topology_hash == compute_full_topology_hash(graph)
        finally:
            db.close()

    def test_successful_run_writes_baseline_first_then_deletes_it(self) -> None:
        """On success the baseline is the FIRST checkpoint written and is
        deleted with the rest (deletion is run-scoped, not sequence-scoped)."""
        db = LandscapeDB.in_memory()
        try:
            payload_store = MockPayloadStore()
            checkpoint_mgr = CheckpointManager(db)

            checkpoint_calls: list[dict[str, Any]] = []
            original_create = checkpoint_mgr.create_checkpoint

            def tracking_create(*args: Any, **kwargs: Any) -> Any:
                checkpoint_calls.append(kwargs)
                return original_create(*args, **kwargs)

            checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

            source = ListSource([{"value": 1}])
            transform = PassTransform(on_success="default")
            sink = CollectSink()
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
            )
            graph = build_production_graph(config)

            orchestrator = Orchestrator(
                db=db,
                checkpoint_manager=checkpoint_mgr,
                checkpoint_config=_checkpoint_config(),
            )
            result = orchestrator.run(config, graph=graph, payload_store=payload_store)

            assert result.status == "completed"
            # Run-start baseline + one every_row post-sink checkpoint.
            assert len(checkpoint_calls) == 2
            assert checkpoint_calls[0]["sequence_number"] == 0
            assert checkpoint_calls[0]["barrier_scalars"] is None
            assert checkpoint_calls[1]["sequence_number"] == 1
            # Success deletes ALL checkpoints for the run, baseline included.
            assert _select_checkpoints(db, result.run_id) == []
        finally:
            db.close()

    def test_disabled_checkpointing_writes_no_run_start_checkpoint(self) -> None:
        """Checkpointing disabled => zero checkpoint writes, baseline included."""
        db = LandscapeDB.in_memory()
        try:
            payload_store = MockPayloadStore()
            checkpoint_mgr = CheckpointManager(db)

            checkpoint_calls: list[dict[str, Any]] = []
            original_create = checkpoint_mgr.create_checkpoint

            def tracking_create(*args: Any, **kwargs: Any) -> Any:
                checkpoint_calls.append(kwargs)
                return original_create(*args, **kwargs)

            checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

            source = ListSource([{"value": 1}])
            sink = CollectSink()
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[],
                sinks={"default": as_sink(sink)},
            )
            graph = build_production_graph(config)

            orchestrator = Orchestrator(
                db=db,
                checkpoint_manager=checkpoint_mgr,
                checkpoint_config=_checkpoint_config(enabled=False),
            )
            result = orchestrator.run(config, graph=graph, payload_store=payload_store)

            assert result.status == "completed"
            assert checkpoint_calls == []
        finally:
            db.close()

    def test_resume_does_not_rewrite_sequence_zero(self) -> None:
        """Crash after start, resume: exactly one sequence-0 row survives.

        The resume path rebases onto the persisted sequence
        (rebase_sequence) instead of re-running checkpoint_run_start; the
        original baseline row (same checkpoint_id) is the only sequence-0
        row before AND after the resume attempt.
        """
        db = LandscapeDB.in_memory()
        try:
            payload_store = MockPayloadStore()
            checkpoint_mgr = CheckpointManager(db)

            source = ListSource([{"value": 1}])
            sink = FailingSink(error_message="still broken")
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[],
                sinks={"default": as_sink(sink)},
            )
            graph = build_production_graph(config)

            orchestrator = Orchestrator(
                db=db,
                checkpoint_manager=checkpoint_mgr,
                checkpoint_config=_checkpoint_config(),
            )
            with pytest.raises(RuntimeError, match="still broken"):
                orchestrator.run(config, graph=graph, payload_store=payload_store, run_id="run-resume-cp")

            baseline_rows = _select_checkpoints(db, "run-resume-cp")
            assert [r.sequence_number for r in baseline_rows] == [0]
            baseline_id = baseline_rows[0].checkpoint_id

            recovery_mgr = RecoveryManager(db, checkpoint_mgr)
            check = recovery_mgr.can_resume("run-resume-cp", graph)
            assert check.can_resume, f"Expected resumable run, got refusal: {check.reason}"
            resume_point = recovery_mgr.get_resume_point("run-resume-cp", graph)
            assert resume_point is not None
            assert resume_point.checkpoint.sequence_number == 0

            # Track checkpoint writes made by the resume attempt only.
            resume_calls: list[dict[str, Any]] = []
            original_create = checkpoint_mgr.create_checkpoint

            def tracking_create(*args: Any, **kwargs: Any) -> Any:
                resume_calls.append(kwargs)
                return original_create(*args, **kwargs)

            checkpoint_mgr.create_checkpoint = tracking_create  # type: ignore[method-assign]

            # The replay outcome itself is NOT this test's contract (the sink
            # still fails, and mid-flight token replay is Task 4.x territory) —
            # what Task 3.3 pins is the resume ENTRY: rebase onto the persisted
            # sequence, never a second run-start write.
            resume_error: Exception | None = None
            try:
                orchestrator.resume(
                    resume_point=resume_point,
                    config=config,
                    graph=graph,
                    payload_store=payload_store,
                )
            except Exception as exc:  # any replay failure shape is acceptable here
                resume_error = exc

            assert all(call["sequence_number"] != 0 for call in resume_calls), (
                f"Resume must never rewrite the sequence-0 baseline; saw writes at {[c['sequence_number'] for c in resume_calls]}"
            )
            if resume_error is None:
                # A successful resume deletes the run's checkpoints wholesale.
                assert _select_checkpoints(db, "run-resume-cp") == []
            else:
                rows_after = _select_checkpoints(db, "run-resume-cp")
                sequence_zero_rows = [r for r in rows_after if r.sequence_number == 0]
                assert len(sequence_zero_rows) == 1, f"Expected exactly one sequence-0 row, got {len(sequence_zero_rows)}"
                assert sequence_zero_rows[0].checkpoint_id == baseline_id, "Resume must not rewrite the run-start baseline"
        finally:
            db.close()
