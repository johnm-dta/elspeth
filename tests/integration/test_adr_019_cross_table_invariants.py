"""ADR-019 Phase 4 cross-table invariant coverage.

These tests exercise structural consistency between ``token_outcomes``,
``node_states``, ``artifacts``, ``token_parents``, and ``batches``. The invalid
states are constructed directly because a well-formed pipeline should not be
able to produce them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.contracts import ExecutionError, NodeType, RunStatus
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenRef
from elspeth.contracts.enums import BatchStatus, NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.factory import RecorderFactory

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})
_ERROR_HASH = "abcd1234" * 8


def _build_base_run(factory: RecorderFactory) -> tuple[str, str, str]:
    """Create a run, source node, row, and token."""
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=source.node_id,
        row_index=0,
        data={"x": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    return run.run_id, source.node_id, token.token_id


def _register_sink_node(factory: RecorderFactory, run_id: str, *, name: str = "failsink") -> str:
    node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name=name,
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    return node.node_id


def _record_completed_sink_state_with_artifact(
    factory: RecorderFactory,
    *,
    run_id: str,
    token_id: str,
    sink_node_id: str,
    step_index: int = 0,
) -> tuple[str, str]:
    state = factory.execution.begin_node_state(
        token_id=token_id,
        node_id=sink_node_id,
        run_id=run_id,
        step_index=step_index,
        input_data={},
    )
    factory.execution.complete_node_state(
        state_id=state.state_id,
        status=NodeStateStatus.COMPLETED,
        output_data={"written": True},
        duration_ms=1.0,
    )
    artifact = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state.state_id,
        sink_node_id=sink_node_id,
        artifact_type="test",
        path=f"memory://failsink/{token_id}/{step_index}",
        content_hash="deadbeef" * 8,
        size_bytes=0,
    )
    return state.state_id, artifact.artifact_id


class TestI1cFailsinkPaired:
    """I1c: failsink fallback needs exact node-state and artifact witnesses."""

    def test_failsink_pair_present_passes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id)
        _state_id, artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sink_node_id,
        )

        outcome_id = landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            sink_node_id=sink_node_id,
            artifact_id=artifact_id,
            error_hash=_ERROR_HASH,
        )

        assert outcome_id.startswith("out_")

    def test_failsink_node_state_missing_crashes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)

        with pytest.raises(AuditIntegrityError, match=r"I1c.*failsink"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id="missing-failsink-node",
                artifact_id="missing-artifact",
                error_hash=_ERROR_HASH,
            )

    def test_failsink_artifact_missing_crashes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id)
        state = landscape_factory.execution.begin_node_state(
            token_id=token_id,
            node_id=sink_node_id,
            run_id=run_id,
            step_index=0,
            input_data={},
        )
        landscape_factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"written": True},
            duration_ms=1.0,
        )

        with pytest.raises(AuditIntegrityError, match=r"I1c.*artifact"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=sink_node_id,
                artifact_id="missing-artifact",
                error_hash=_ERROR_HASH,
            )

    def test_failsink_wrong_sink_node_crashes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        failsink_node_id = _register_sink_node(landscape_factory, run_id)
        sibling_sink_node_id = _register_sink_node(landscape_factory, run_id, name="other_failsink")
        _state_id, artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=failsink_node_id,
        )

        with pytest.raises(AuditIntegrityError, match=r"I1c.*node"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=sibling_sink_node_id,
                artifact_id=artifact_id,
                error_hash=_ERROR_HASH,
            )

    def test_failsink_wrong_artifact_crashes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        failsink_node_id = _register_sink_node(landscape_factory, run_id)
        sibling_sink_node_id = _register_sink_node(landscape_factory, run_id, name="other_failsink")
        _state_id, _correct_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=failsink_node_id,
        )
        _other_state_id, wrong_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sibling_sink_node_id,
            step_index=1,
        )

        with pytest.raises(AuditIntegrityError, match=r"I1c.*artifact"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                sink_node_id=failsink_node_id,
                artifact_id=wrong_artifact_id,
                error_hash=_ERROR_HASH,
            )

    def test_failsink_same_sink_artifact_from_different_token_passes(
        self,
        landscape_factory: RecorderFactory,
    ) -> None:
        run_id, source_node_id, token_id = _build_base_run(landscape_factory)
        failsink_node_id = _register_sink_node(landscape_factory, run_id)
        _state_id, _correct_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=failsink_node_id,
        )
        other_row = landscape_factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=1,
            data={"x": 2},
            source_row_index=1,
            ingest_sequence=1,
        )
        other_token = landscape_factory.data_flow.create_token(row_id=other_row.row_id)
        _other_state_id, wrong_artifact_id = _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=other_token.token_id,
            sink_node_id=failsink_node_id,
            step_index=1,
        )

        outcome_id = landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
            sink_name="failsink",
            sink_node_id=failsink_node_id,
            artifact_id=wrong_artifact_id,
            error_hash=_ERROR_HASH,
        )

        assert outcome_id.startswith("out_")


class TestI3DiscardNoFailsink:
    """I3: discard means the row was not absorbed by a completed failsink."""

    def test_discard_with_failed_sink_state_passes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        primary_sink_node_id = _register_sink_node(landscape_factory, run_id, name="primary_sink")
        state = landscape_factory.execution.begin_node_state(
            token_id=token_id,
            node_id=primary_sink_node_id,
            run_id=run_id,
            step_index=0,
            input_data={},
        )
        landscape_factory.execution.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.FAILED,
            error=ExecutionError(
                exception="discarded",
                exception_type="DiscardedForTest",
                phase="sink_write",
            ),
            duration_ms=1.0,
        )

        outcome_id = landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
            sink_name=DISCARD_SINK_NAME,
            error_hash=_ERROR_HASH,
        )

        assert outcome_id.startswith("out_")

    def test_discard_with_completed_sink_state_crashes(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        sink_node_id = _register_sink_node(landscape_factory, run_id)
        _record_completed_sink_state_with_artifact(
            landscape_factory,
            run_id=run_id,
            token_id=token_id,
            sink_node_id=sink_node_id,
        )

        with pytest.raises(AuditIntegrityError, match=r"I3.*discard"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash=_ERROR_HASH,
            )

    def test_discard_wrong_sink_name_rejected_by_scalar_guard(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)

        with pytest.raises(ValueError, match="sink_name"):
            landscape_factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_id, run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name="other_sink_name",
                error_hash=_ERROR_HASH,
            )


class TestI1aForkParentDeferred:
    """I1a: transient fork or expand parents need child outcome witnesses."""

    def _build_fork_parent_orphan(self, landscape_factory: RecorderFactory) -> tuple[str, str]:
        run_id, _source_node_id, token_id = _build_base_run(landscape_factory)
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.FORK_PARENT,
            fork_group_id="fg_orphan_001",
        )
        return run_id, token_id

    def test_orphan_parent_detected_by_repository_helper(self, landscape_factory: RecorderFactory) -> None:
        run_id, token_id = self._build_fork_parent_orphan(landscape_factory)

        orphans = landscape_factory.data_flow.find_orphaned_transient_parents(run_id)

        assert len(orphans) == 1
        assert orphans[0].token_id == token_id
        assert orphans[0].path == TerminalPath.FORK_PARENT.value

    def test_fork_parent_with_child_not_flagged(self, landscape_factory: RecorderFactory) -> None:
        run_id, _source_node_id, parent_token_id = _build_base_run(landscape_factory)
        parent_row_id, _owner_run_id = landscape_factory.data_flow._resolve_token_ownership(parent_token_id)
        children, _fork_group_id = landscape_factory.data_flow.fork_token(
            parent_ref=TokenRef(token_id=parent_token_id, run_id=run_id),
            row_id=parent_row_id,
            branches=["branch_a", "branch_b"],
        )
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=children[0].token_id, run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )

        orphans = landscape_factory.data_flow.find_orphaned_transient_parents(run_id)

        assert orphans == []

    def test_fork_parent_orphan_crashes_sweep(self, landscape_factory: RecorderFactory) -> None:
        run_id, _token_id = self._build_fork_parent_orphan(landscape_factory)

        with pytest.raises(AuditIntegrityError, match="I1a"):
            landscape_factory.data_flow.sweep_deferred_invariants_or_crash(run_id)


class TestI1bBatchConsumedDeferred:
    """I1b: batch-consumed tokens need completed batch witnesses."""

    def _build_batch_consumed_orphan(self, landscape_factory: RecorderFactory) -> tuple[str, str, str]:
        run_id, source_node_id, token_id = _build_base_run(landscape_factory)
        batch_id = "batch_orphan_001"
        landscape_factory.execution.create_batch(
            run_id=run_id,
            aggregation_node_id=source_node_id,
            batch_id=batch_id,
        )
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )
        return run_id, token_id, batch_id

    def test_orphan_batch_detected_by_repository_helper(self, landscape_factory: RecorderFactory) -> None:
        run_id, _token_id, batch_id = self._build_batch_consumed_orphan(landscape_factory)

        orphan_batch_ids = landscape_factory.data_flow.find_orphaned_batch_consumptions(run_id)

        assert orphan_batch_ids == [batch_id]

    def test_batch_consumed_with_completed_batch_not_flagged(self, landscape_factory: RecorderFactory) -> None:
        run_id, source_node_id, token_id = _build_base_run(landscape_factory)
        batch_id = "batch_complete_001"
        landscape_factory.execution.create_batch(
            run_id=run_id,
            aggregation_node_id=source_node_id,
            batch_id=batch_id,
        )
        landscape_factory.execution.complete_batch(
            batch_id=batch_id,
            status=BatchStatus.COMPLETED,
        )
        landscape_factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token_id, run_id=run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch_id,
        )

        orphan_batch_ids = landscape_factory.data_flow.find_orphaned_batch_consumptions(run_id)

        assert batch_id not in orphan_batch_ids

    def test_batch_consumed_orphan_crashes_sweep(self, landscape_factory: RecorderFactory) -> None:
        run_id, _token_id, _batch_id = self._build_batch_consumed_orphan(landscape_factory)

        with pytest.raises(AuditIntegrityError, match="I1b"):
            landscape_factory.data_flow.sweep_deferred_invariants_or_crash(run_id)


def test_valid_fork_coalesce_run_does_not_false_positive_after_sink_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid fork/coalesce run must pass the post-sink sweep."""
    from elspeth.core.config import CoalesceSettings, ElspethSettings, GateSettings
    from elspeth.core.landscape.data_flow_repository import DataFlowRepository
    from elspeth.core.landscape.database import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from tests.fixtures.base_classes import as_sink, as_source
    from tests.fixtures.pipeline import build_fork_pipeline

    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    sweep_calls: list[str] = []

    original_sweep = DataFlowRepository.sweep_deferred_invariants_or_crash

    def _spy_sweep(self: DataFlowRepository, run_id: str) -> None:
        sweep_calls.append(run_id)
        original_sweep(self, run_id)

    monkeypatch.setattr(DataFlowRepository, "sweep_deferred_invariants_or_crash", _spy_sweep)

    gate = GateSettings(
        name="fork_gate",
        input="primary_out",
        condition="True",
        routes={"true": "fork", "false": "fork"},
        fork_to=["path_a", "path_b"],
    )
    coalesce = CoalesceSettings(
        name="merge_results",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
        on_success="default",
    )
    source, _transforms, sinks, graph = build_fork_pipeline(
        [{"id": 1, "value": 7}],
        gate=gate,
        branch_transforms={"path_a": [], "path_b": []},
        coalesce_settings=[coalesce],
    )
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={name: as_sink(sink) for name, sink in sinks.items()},
        coalesce_settings=[coalesce],
        gates=[gate],
    )
    settings = ElspethSettings(
        sources={"primary": {"plugin": "list_source", "on_success": "primary_out", "options": {}}},
        sinks={name: {"plugin": "collect", "on_write_failure": "discard", "options": {}} for name in sinks},
        gates=[gate],
        coalesce=[coalesce],
    )

    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=payload_store)

    assert result.status == RunStatus.COMPLETED
    assert sweep_calls == [result.run_id]
