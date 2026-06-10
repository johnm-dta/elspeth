"""Integration tests for Tier-1 resume guardrails in Orchestrator.

Covers bead scug.3:
- missing payload_store -> ValueError
- missing checkpoint manager -> ValueError
- missing schema contract -> OrchestrationInvariantError
- graph edges present but DB edge map empty -> ValueError
- positive control: valid preconditions resume successfully
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from elspeth.contracts import Checkpoint, PluginSchema, ResumedRow, ResumePoint, RunStatus
from elspeth.contracts.errors import AuditIntegrityError, EmptyResumeStateError, OrchestrationInvariantError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource


class _ResumeSourceSchema(PluginSchema):
    """Typed schema used to populate run.source_schema_json for resume tests."""

    id: int
    value: str


def _make_schema_contract() -> SchemaContract:
    """Create a minimal fixed contract for resume-path PipelineRow wrapping."""
    return SchemaContract(
        mode="FIXED",
        fields=(
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
                python_type=str,
                required=True,
                source="declared",
            ),
        ),
        locked=True,
    )


def _make_resume_point(run_id: str) -> ResumePoint:
    """Create a synthetic ResumePoint targeting a run in the test database."""
    checkpoint = Checkpoint(
        checkpoint_id="cp-test",
        run_id=run_id,
        sequence_number=0,
        created_at=datetime.now(UTC),
        upstream_topology_hash="topology-hash",
        format_version=Checkpoint.CURRENT_FORMAT_VERSION,
    )
    return ResumePoint(
        checkpoint=checkpoint,
        sequence_number=checkpoint.sequence_number,
        aggregation_state=None,
    )


def _build_pipeline() -> tuple[PipelineConfig, Any]:
    """Build a minimal source->sink pipeline and production graph."""
    source = ListSource([{"id": 1, "value": "alpha"}], on_success="default")
    sink = CollectSink("default")
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(sink)},
    )
    return config, build_production_graph(config)


def _create_failed_run(
    factory: RecorderFactory,
    *,
    include_contract: bool,
) -> str:
    """Create a FAILED run with source schema, optionally with a per-source contract record.

    ADR-025 §3 Decision 5 (G6): schema contracts now live exclusively in
    ``run_sources``. ``include_contract=True`` writes a ``run_sources`` row so
    resume reconstruction can find a per-source contract; ``False`` leaves
    ``run_sources`` empty so resume refuses with ``EmptyResumeStateError``.
    """
    from elspeth.contracts import NodeType
    from elspeth.contracts.contract_records import ContractAuditRecord
    from elspeth.contracts.enums import Determinism
    from elspeth.core.landscape.schema import nodes_table, run_sources_table

    run = factory.run_lifecycle.begin_run(
        config={"test": "resume-guardrails"},
        canonical_version="v1",
        status=RunStatus.FAILED,
        source_schema_json=json.dumps(_ResumeSourceSchema.model_json_schema()),
    )
    if include_contract:
        contract = _make_schema_contract()
        audit_record = ContractAuditRecord.from_contract(contract)
        now = datetime.now(UTC)
        with factory.run_lifecycle._db.connection() as conn:
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run.run_id,
                    plugin_name="list_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="resume-guardrails",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run.run_id,
                    source_node_id="source-node",
                    source_name="primary",
                    plugin_name="list_source",
                    lifecycle_state="loaded",
                    config_hash="resume-guardrails",
                    schema_json=json.dumps(_ResumeSourceSchema.model_json_schema()),
                    schema_contract_json=audit_record.to_json(),
                    schema_contract_hash=contract.version_hash(),
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )
    return run.run_id


class TestResumeGuardrails:
    """Regression coverage for Tier-1 resume precondition failures."""

    def test_resume_requires_payload_store(self, landscape_db: LandscapeDB) -> None:
        """Resume must hard-fail immediately when payload_store is missing."""
        orchestrator = Orchestrator(landscape_db)
        config, graph = _build_pipeline()

        with pytest.raises(OrchestrationInvariantError, match="payload_store is required for resume"):
            orchestrator.resume(
                resume_point=_make_resume_point("run-missing-payload-store"),
                config=config,
                graph=graph,
                payload_store=None,  # type: ignore[arg-type]
            )

    def test_resume_requires_checkpoint_manager(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must hard-fail when Orchestrator has no CheckpointManager."""
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=True)
        orchestrator = Orchestrator(resume_test_env["db"])
        config, graph = _build_pipeline()

        with pytest.raises(OrchestrationInvariantError, match="CheckpointManager is required for resume"):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

    def test_resume_fails_with_empty_resume_state_when_schema_contract_is_missing(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must raise typed EmptyResumeStateError when contract is absent.

        Per P1 elspeth-81dad89e1c, a NULL ``schema_contract_json`` in an
        otherwise-structurally-valid runs row reflects the documented
        "on_start failure before contract written" scenario named in
        the ``EmptyResumeStateError`` docstring — NOT generic Tier-1
        corruption. The legacy pre-RC6 path at orchestrator/core.py
        previously raised ``OrchestrationInvariantError`` (Tier-1
        traceback) for this case; the typed raise gives the CLI a
        clean operator-interpretable exit path.

        ``EmptyResumeStateError`` is a subclass of
        ``OrchestrationInvariantError`` so any existing
        ``except OrchestrationInvariantError`` catch still matches.
        """
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=False)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        with (
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data",
                return_value=[],
            ) as mock_get_unprocessed,
            pytest.raises(EmptyResumeStateError) as exc_info,
        ):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        mock_get_unprocessed.assert_not_called()
        assert exc_info.value.run_id == run_id
        # ADR-025 §3 Decision 5 (G6): the singleton ``schema_contract_json``
        # column was deleted. The empty-resume-state precondition now keys on
        # the presence of ``run_sources`` records — the same observable
        # condition the legacy NULL-singleton check used to describe.
        assert "no run_sources records were written" in str(exc_info.value).lower()
        # Subclass relationship is load-bearing — see the typed-error
        # subclass commentary on the companion test below.
        assert isinstance(exc_info.value, OrchestrationInvariantError)

    def test_resume_fails_when_graph_has_edges_but_db_edge_map_is_empty(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must fail if graph edges exist but original run edge data is missing."""
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=True)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        # ADR-025 §3 Decision 5 (G6): resume now reconstructs unprocessed
        # rows via ``get_unprocessed_row_data_by_source``. Patching the
        # legacy ``get_unprocessed_row_data`` would silently no-op.
        with (
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data_by_source",
                return_value=(
                    ResumedRow(
                        row_id="row-1",
                        row_index=0,
                        source_node_id=NodeID("source-node"),
                        row_data={"id": 1, "value": "alpha"},
                    ),
                ),
            ),
            pytest.raises(AuditIntegrityError, match="has no edges registered") as exc_info,
        ):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        assert run_id in str(exc_info.value)
        assert "cannot build edge map" in str(exc_info.value).lower()

    def test_resume_fails_when_runtime_val_manifest_has_drifted(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must fail closed when the current contract registry differs from the original run."""
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=True)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        with (
            patch(
                "elspeth.engine.orchestrator.resume.build_runtime_val_manifest",
                return_value={
                    "declaration_contracts": [{"name": "drifted"}],
                    "expected_contract_sites": {"drifted": ["boundary_check"]},
                    "tier_1_errors": [],
                },
            ),
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data",
                return_value=[],
            ) as mock_get_unprocessed,
            pytest.raises(OrchestrationInvariantError, match="runtime VAL manifest") as exc_info,
        ):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        mock_get_unprocessed.assert_not_called()
        assert run_id in str(exc_info.value)
        assert "contract registry" in str(exc_info.value).lower()

    def test_resume_refuses_with_typed_error_when_no_sources_recorded(self, resume_test_env: dict[str, Any]) -> None:
        """ADR-025 §3 Decision 5 (G6): empty resume state is refused with a typed exception.

        A run that failed before any row was committed AND wrote no
        ``run_sources`` records has nothing to key a contract under, so
        ``_reconstruct_resume_state`` refuses upstream with
        :class:`EmptyResumeStateError`. The CLI can then present a clean
        "this run is not resumable, start a fresh run" outcome rather
        than a Tier-1 invariant traceback.

        Fixture shape: ``_create_failed_run(include_contract=False)``
        writes the ``runs`` row but does NOT call ``record_run_source``
        — the exact precondition that triggers the strict refuse.

        The genuine "early-exit on empty rows" success path is
        exercised in
        ``tests/unit/engine/orchestrator/test_resume_failure.py::test_resume_treats_empty_coalesce_checkpoint_as_all_rows_processed``,
        which constructs ``ResumeState`` directly with a non-empty
        contract map (the RC6 shape: run_sources records present,
        unprocessed_rows empty).
        """
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=False)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        with (
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data_by_source",
                return_value=(),
            ),
            pytest.raises(EmptyResumeStateError) as exc_info,
        ):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        # EmptyResumeStateError carries the run_id directly so the
        # upstream caller (CLI) can present an operator-facing message
        # without parsing the exception text.
        assert exc_info.value.run_id == run_id
        # The exception message mentions the empty-work shape so an
        # operator reading stderr understands why resume refused.
        assert "no rows were committed" in str(exc_info.value).lower()
        # Subclass relationship is load-bearing: every existing
        # ``except OrchestrationInvariantError`` catch must still
        # match this exception so callers without explicit
        # EmptyResumeStateError handling do not silently miss it.
        assert isinstance(exc_info.value, OrchestrationInvariantError)
