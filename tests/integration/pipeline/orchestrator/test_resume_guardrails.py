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

from elspeth.contracts import Checkpoint, PluginSchema, ResumePoint, RunStatus
from elspeth.contracts.errors import AuditIntegrityError, OrchestrationInvariantError
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
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


def _make_resume_point(run_id: str, *, node_id: str = "source") -> ResumePoint:
    """Create a synthetic ResumePoint targeting a run in the test database."""
    checkpoint = Checkpoint(
        checkpoint_id="cp-test",
        run_id=run_id,
        token_id="tok-test",
        node_id=node_id,
        sequence_number=0,
        created_at=datetime.now(UTC),
        upstream_topology_hash="topology-hash",
        checkpoint_node_config_hash="config-hash",
        format_version=Checkpoint.CURRENT_FORMAT_VERSION,
    )
    return ResumePoint(
        checkpoint=checkpoint,
        token_id=checkpoint.token_id,
        node_id=checkpoint.node_id,
        sequence_number=checkpoint.sequence_number,
        aggregation_state=None,
    )


def _build_pipeline() -> tuple[PipelineConfig, Any]:
    """Build a minimal source->sink pipeline and production graph."""
    source = ListSource([{"id": 1, "value": "alpha"}], on_success="default")
    sink = CollectSink("default")
    config = PipelineConfig(
        source=as_source(source),
        transforms=[],
        sinks={"default": as_sink(sink)},
    )
    return config, build_production_graph(config)


def _create_failed_run(
    factory: RecorderFactory,
    *,
    include_contract: bool,
) -> str:
    """Create a FAILED run with source schema, optionally with schema contract."""
    run = factory.run_lifecycle.begin_run(
        config={"test": "resume-guardrails"},
        canonical_version="v1",
        status=RunStatus.FAILED,
        source_schema_json=json.dumps(_ResumeSourceSchema.model_json_schema()),
        schema_contract=_make_schema_contract() if include_contract else None,
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

    def test_resume_fails_when_schema_contract_is_missing(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must not infer/fallback when schema contract is absent in audit trail."""
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
            pytest.raises(OrchestrationInvariantError, match="schema contract is missing from audit trail") as exc_info,
        ):
            orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        mock_get_unprocessed.assert_not_called()
        assert run_id in str(exc_info.value)
        assert "cannot proceed safely without the schema contract" in str(exc_info.value).lower()

    def test_resume_fails_when_graph_has_edges_but_db_edge_map_is_empty(self, resume_test_env: dict[str, Any]) -> None:
        """Resume must fail if graph edges exist but original run edge data is missing."""
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=True)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        with (
            patch(
                "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data",
                return_value=[("row-1", 0, {"id": 1, "value": "alpha"})],
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

    def test_resume_positive_control_succeeds_with_valid_preconditions(self, resume_test_env: dict[str, Any]) -> None:
        """Valid setup still resumes successfully (early exit when no rows remain)."""
        run_id = _create_failed_run(resume_test_env["factory"], include_contract=True)
        orchestrator = Orchestrator(
            resume_test_env["db"],
            checkpoint_manager=resume_test_env["checkpoint_manager"],
        )
        config, graph = _build_pipeline()

        with patch(
            "elspeth.core.checkpoint.recovery.RecoveryManager.get_unprocessed_row_data",
            return_value=[],
        ):
            result = orchestrator.resume(
                resume_point=_make_resume_point(run_id),
                config=config,
                graph=graph,
                payload_store=resume_test_env["payload_store"],
            )

        # Phase 2.2 (elspeth-0de989c56d): the resume's early-exit branch
        # now derives the truthful terminal status from the audit DB
        # ``token_outcomes`` instead of writing ``COMPLETED``
        # unconditionally.  This run was set up via ``_create_failed_run``
        # with ``status=FAILED`` and zero token_outcomes — the truthful
        # state is therefore ``EMPTY`` (no row outcomes) rather than the
        # pre-Phase-2.2 unconditional ``COMPLETED``.  Both the local
        # ``RunResult`` and the persisted Landscape ``Run`` reflect the
        # same predicate output.
        assert result.status == RunStatus.EMPTY
        assert result.rows_processed == 0
        run = resume_test_env["factory"].run_lifecycle.get_run(run_id)
        assert run is not None
        assert run.status == RunStatus.EMPTY
