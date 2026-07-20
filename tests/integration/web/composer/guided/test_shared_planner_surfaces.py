"""Shared-planner identity checks for the production guided-full adapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from elspeth.contracts.hashing import stable_hash
from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer import pipeline_planner
from elspeth.web.composer import service as service_module
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.availability import ComposerAvailability
from elspeth.web.composer.capability_skill import build_planner_capability_manifest
from elspeth.web.composer.guided.prompts import load_step_planner_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.pipeline_planner import (
    PipelinePlanResult,
    PlannerOriginatingMessage,
    planner_tool_definitions,
)
from elspeth.web.composer.pipeline_proposal import (
    PipelineProposal,
    PlannerSurface,
    PresentBase,
    composition_content_hash,
)
from elspeth.web.composer.prompts import build_system_prompt
from elspeth.web.composer.protocol import ComposerService
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema
from elspeth.web.sessions.protocol import GuidedOperationFence


def test_guided_full_is_an_explicit_composer_service_surface() -> None:
    assert "plan_guided_full_pipeline" in ComposerService.__dict__


def test_guided_full_controller_owns_no_topology_constructor() -> None:
    source = Path("src/elspeth/web/sessions/routes/composer/guided_plan.py").read_text(encoding="utf-8")
    assert "plan_guided_full_pipeline" in source
    assert "plan_pipeline(" not in source
    assert "solve_chain" not in source


def test_guided_full_runtime_calls_the_shared_module_planner_exactly_once(
    composer_test_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert service_module.plan_pipeline is pipeline_planner.plan_pipeline
    assert ComposerServiceImpl.plan_guided_full_pipeline.__globals__["plan_pipeline"] is pipeline_planner.plan_pipeline
    assert ComposerServiceImpl.plan_guided_pipeline.__globals__["plan_pipeline"] is pipeline_planner.plan_pipeline
    assert ComposerServiceImpl._plan_and_stage_empty_pipeline.__globals__["plan_pipeline"] is pipeline_planner.plan_pipeline

    app = composer_test_client.app
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(
            available=True,
            model="test/shared-planner",
            provider="test",
        ),
    )
    service = ComposerServiceImpl(
        app.state.catalog_service,
        app.state.settings,
        sessions_service=app.state.session_service,
        session_engine=app.state.session_engine,
        plugin_snapshot_factory=app.state.plugin_snapshot_factory,
        operator_profile_registry=app.state.operator_profile_registry,
    )
    user = UserIdentity(user_id="alice", username="alice")
    snapshot = app.state.plugin_snapshot_factory(user)
    policy_catalog = PolicyCatalogView(
        app.state.catalog_service,
        snapshot,
        app.state.operator_profile_registry,
    )
    session_id = uuid4()
    checkpoint_id = uuid4()
    message_id = uuid4()
    state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    base = PresentBase(
        state_id=checkpoint_id,
        composition_content_hash=composition_content_hash(state),
    )
    captured: list[dict[str, object]] = []

    async def fake_plan_pipeline(**kwargs):
        captured.append(kwargs)
        proposal = PipelineProposal.create(
            pipeline={
                "sources": {},
                "nodes": [],
                "edges": [],
                "outputs": [],
            },
            base=kwargs["base"],
            reviewed_facts=kwargs["reviewed_facts"],
            surface=kwargs["surface"],
            repair_count=0,
            skill_hash=stable_hash(kwargs["rendered_skill"]),
            covered_deferred_intent_ids=(),
            supersedes_draft_hash=None,
        )
        return PipelinePlanResult(
            proposal=proposal,
            tool_call_id="shared-planner-runtime-proof",
            custody_result="not_required",
            model_identifier="test/shared-planner",
            model_version="v1",
            provider="test",
        )

    monkeypatch.setattr(service_module, "plan_pipeline", fake_plan_pipeline)
    result, catalog_ids = asyncio.run(
        service.plan_guided_full_pipeline(
            intent="Build a complete pipeline.",
            current_state=state,
            originating_message=PlannerOriginatingMessage(
                session_id=str(session_id),
                message_id=str(message_id),
                content="Build a complete pipeline.",
                user_id="alice",
            ),
            base=base,
            policy_catalog=policy_catalog,
            plugin_snapshot=snapshot,
            recorder=BufferingRecorder(),
            operation_fence=GuidedOperationFence(
                session_id=session_id,
                operation_id="00000000-0000-4000-8000-000000000041",
                lease_token="runtime-identity-proof",
                attempt=1,
            ),
        )
    )

    assert result.proposal.surface is PlannerSurface.GUIDED_FULL
    assert len(captured) == 1
    call = captured[0]
    assert call["surface"] is PlannerSurface.GUIDED_FULL
    assert call["profile"] == "ordinary"
    assert call["reviewed_facts"] == {}
    assert call["reviewed_planner_context"] == {}
    assert call["eligible_deferred_intent_ids"] == ()
    assert call["claim_evaluator"] is None
    assert call["supersedes_draft_hash"] is None
    assert call["rendered_skill"] == build_system_prompt(str(app.state.settings.data_dir))
    assert call["candidate_finalizer"](result.proposal.pipeline) is result.proposal.pipeline
    assert set(catalog_ids) == {"source", "transform", "sink"}


def test_all_planner_surfaces_share_canonical_core_schema_and_tool_identity() -> None:
    surfaces = (
        (PlannerSurface.FREEFORM, "ordinary", build_system_prompt(None)),
        (PlannerSurface.GUIDED_FULL, "ordinary", build_system_prompt(None)),
        (
            PlannerSurface.GUIDED_STAGED,
            "ordinary",
            load_step_planner_skill(GuidedStep.STEP_3_TRANSFORMS),
        ),
        (
            PlannerSurface.TUTORIAL_PROFILE,
            "tutorial",
            load_step_planner_skill(GuidedStep.STEP_3_TRANSFORMS),
        ),
    )
    manifests = [
        build_planner_capability_manifest(
            surface=surface,
            profile=profile,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Build a complete pipeline."},
            ],
            tools=planner_tool_definitions(),
            canonical_schema=canonical_set_pipeline_schema(),
        )
        for surface, profile, prompt in surfaces
    ]

    assert {manifest.surface for manifest in manifests} == set(PlannerSurface)
    assert len({manifest.planner_implementation_id for manifest in manifests}) == 1
    assert len({manifest.capability_core_hash for manifest in manifests}) == 1
    assert len({manifest.canonical_schema_hash for manifest in manifests}) == 1
    assert len({manifest.effective_tool_hash for manifest in manifests}) == 1
