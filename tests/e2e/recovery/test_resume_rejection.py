"""E2E tests for resume rejection edge cases.

Tests verify that resume operations are correctly rejected when
inappropriate: completed runs, non-existent runs, and FAILED runs with
nothing recoverable. (Real crash/concurrent resume coverage lives in
test_concurrent_resume.py.)

Durability-unification (F1) survival note: every assertion here reads the
public ``can_resume`` boolean plus runs-table-derived reasons — never
checkpoint-blob internals or checkpoint-specific reason wording. The one
checkpoint-coupled surface is the RecoveryManager constructor itself, which
is confined to the single :func:`_recovery_manager` seam below; an F1
signature change edits that one helper and nothing else.

Uses file-based SQLite and real payload stores. No mocks except
external services.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elspeth.contracts import (
    Determinism,
    NodeType,
    RunStatus,
)
from elspeth.contracts.contract_records import ContractAuditRecord
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    run_sources_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, ListSource

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _recovery_manager(db: LandscapeDB) -> RecoveryManager:
    """SINGLE construction seam for the recovery entry point.

    The RecoveryManager constructor is currently checkpoint-coupled (it
    takes a CheckpointManager). When the durability unification (F1)
    reshapes that signature, this helper is the only construction site in
    this file that needs editing — every test below asserts purely on the
    public ``can_resume`` result and status-derived reasons.
    """
    return RecoveryManager(db, CheckpointManager(db))


def _create_test_schema_contract() -> tuple[str, str]:
    """Create a minimal schema contract for test runs.

    Returns:
        Tuple of (schema_contract_json, schema_contract_hash)
    """
    field_contracts = (
        FieldContract(
            normalized_name="id",
            original_name="id",
            python_type=int,
            required=True,
            source="declared",
        ),
        FieldContract(
            normalized_name="value",
            original_name="value",
            python_type=int,
            required=True,
            source="declared",
        ),
    )
    contract = SchemaContract(fields=field_contracts, mode="FIXED", locked=True)
    audit_record = ContractAuditRecord.from_contract(contract)
    return audit_record.to_json(), contract.version_hash()


class TestResumeRejection:
    """Tests for resume rejection scenarios."""

    def test_second_resume_of_completed_run_rejected(self, tmp_path: Path) -> None:
        """Run pipeline, let it complete, try to resume.

        Verify can_resume() returns False for completed runs.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        source_data = [{"id": i, "value": i * 10} for i in range(3)]
        source = ListSource(source_data)
        sink = CollectSink("default")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"default": as_sink(sink)},
        )

        # Use from_plugin_instances() for production-path fidelity
        _source_for_graph, _transforms, _sinks, graph = build_linear_pipeline(source_data)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert result.status == RunStatus.COMPLETED

        # Try to resume the completed run
        recovery_mgr = _recovery_manager(db)

        check = recovery_mgr.can_resume(result.run_id, graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "completed" in check.reason.lower()

        db.close()

    def test_resume_of_non_existent_run_rejected(self, tmp_path: Path) -> None:
        """Try to resume a run_id that doesn't exist.

        Verify can_resume() returns False with appropriate reason.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")

        recovery_mgr = _recovery_manager(db)

        # Build a minimal graph for the can_resume call
        # NOTE: Manual ExecutionGraph construction is acceptable here
        # because we are testing resume rejection, not pipeline execution.
        # No checkpoint or pipeline data exists for this fake run_id.
        graph = ExecutionGraph()
        schema_config: dict[str, Any] = {"schema": {"mode": "observed"}}
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test",
            config=schema_config,
        )

        check = recovery_mgr.can_resume("non-existent-run-id-12345", graph)
        assert check.can_resume is False
        assert check.reason is not None
        assert "not found" in check.reason.lower()

        db.close()

    def test_resume_of_failed_run_with_nothing_recoverable_rejected(self, tmp_path: Path) -> None:
        """A FAILED run with nothing recoverable is rejected for resume.

        The crafted run has row/token audit rows but NO recoverable resume
        state: zero ``token_work_items`` journal rows and no checkpoint.
        Whatever layer the resume gate reads — the checkpoint store today,
        the scheduler journal after the durability unification (F1) — a run
        with nothing to recover must be rejected. The assertion is therefore
        only the public ``can_resume`` boolean plus a non-empty reason; the
        gate's wording is deliberately NOT pinned (today it is
        checkpoint-flavoured; post-F1 it becomes journal-flavoured).
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")

        run_id = "failed-no-checkpoint-run"
        now = datetime.now(UTC)
        contract_json, contract_hash = _create_test_schema_contract()

        # Create a failed run with rows but NO checkpoint.
        # ADR-025 §3 Decision 5 (G6): schema contracts live exclusively in
        # ``run_sources`` — the legacy ``runs.schema_contract_json`` /
        # ``schema_contract_hash`` columns no longer exist, so the contract
        # is recorded on the per-source ``run_sources`` row instead.
        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="source",
                    source_name="source",
                    plugin_name="test_source",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json='{"properties": {"id": {"type": "integer"}, "value": {"type": "integer"}}, "required": ["id", "value"]}',
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create a row so the run has data
            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="source",
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="hash-0",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    run_id=run_id,
                    created_at=now,
                )
            )
            conn.commit()

        # Nothing recoverable was written: no checkpoint, and no
        # token_work_items journal rows (the durable resume truth under F1).

        recovery_mgr = _recovery_manager(db)

        # Build a minimal graph for the can_resume call
        graph = ExecutionGraph()
        schema_config_dict: dict[str, Any] = {"schema": {"mode": "observed"}}
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config=schema_config_dict,
        )

        check = recovery_mgr.can_resume(run_id, graph)
        assert check.can_resume is False
        # The durable claim is "a run with nothing recoverable is rejected",
        # not any particular reason wording (checkpoint-flavoured today,
        # journal-flavoured after F1) — so only non-emptiness is pinned.
        assert check.reason is not None
        assert check.reason != ""

        db.close()
