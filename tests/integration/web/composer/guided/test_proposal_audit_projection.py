"""Security and integrity contracts for guided proposal projection."""

from __future__ import annotations

import inspect
from dataclasses import replace
from uuid import UUID

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.guided.planning import (
    build_guided_proposal_projection,
    guided_private_reviewed_facts,
    guided_redacted_current_state_context,
    guided_redacted_planner_context,
    verify_guided_proposal_projection,
)
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.state_machine import (
    DeferredStageIntent,
    GuidedSession,
    OptionValueConstraint,
    StableSubject,
)
from elspeth.web.composer.pipeline_planner import plan_pipeline, prepare_pipeline_plan
from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface, PresentBase
from elspeth.web.composer.state import CompositionState, PipelineMetadata

SOURCE_ID = "00000000-0000-4000-8000-000000000101"
OUTPUT_ID = "00000000-0000-4000-8000-000000000102"
PROPOSAL_ID = UUID("00000000-0000-4000-8000-000000000103")
CHECKPOINT_ID = UUID("00000000-0000-4000-8000-000000000104")
CANARIES = (
    "RAW-INLINE-CONTENT-CANARY",
    "CREDENTIAL-CANARY",
    "RESOLVED-SECRET-CANARY",
    "RAW-VALIDATION-CANARY",
    "RAW-PROVIDER-ERROR-CANARY",
)
DEFERRED_VALUE_CANARY = "PRIVATE-OPTION-VALUE-CANARY"
DEFERRED_PATH_CANARY = "private_credential_path_canary"


def _guided() -> GuidedSession:
    return replace(
        GuidedSession.initial(),
        reviewed_sources={
            SOURCE_ID: SourceResolved(
                name="primary",
                plugin="csv",
                options={
                    "inline_blob": {"content": CANARIES[0]},
                    "credentials": {"secret_ref": CANARIES[1], "resolved": CANARIES[2]},
                    "schema": {"mode": "observed"},
                },
                observed_columns=("name", "score"),
                sample_rows=({"name": CANARIES[3], "score": 42},),
                on_validation_failure="discard",
            )
        },
        reviewed_outputs={
            OUTPUT_ID: SinkOutputResolved(
                name="cleaned",
                plugin="json",
                options={"path": CANARIES[4], "schema": {"mode": "observed"}},
                required_fields=("name",),
                schema_mode="observed",
                on_write_failure="discard",
            )
        },
        source_order=(SOURCE_ID,),
        output_order=(OUTPUT_ID,),
        step=GuidedStep.STEP_3_TRANSFORMS,
    )


def _proposal(guided: GuidedSession) -> PipelineProposal:
    return PipelineProposal.create(
        pipeline={
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "rows",
                    "options": {"credentials": {"secret_ref": CANARIES[1]}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [
                {
                    "id": "clean",
                    "node_type": "transform",
                    "plugin": "normalize",
                    "input": "rows",
                    "on_success": "cleaned",
                    "on_error": "discard",
                    "options": {"rules": [{"column": "name", "operation": "strip"}]},
                }
            ],
            "edges": [],
            "outputs": [
                {
                    "name": "cleaned",
                    "plugin": "json",
                    "options": {"path": CANARIES[4]},
                    "on_write_failure": "discard",
                }
            ],
        },
        base=PresentBase(state_id=CHECKPOINT_ID, composition_content_hash="a" * 64),
        reviewed_facts=guided_private_reviewed_facts(guided),
        surface=PlannerSurface.GUIDED_STAGED,
        repair_count=0,
        skill_hash=stable_hash("guided planner skill"),
        covered_deferred_intent_ids=(),
        supersedes_draft_hash=None,
    )


def test_planner_context_is_redacted_but_private_anchor_keeps_exact_reviewed_facts() -> None:
    guided = _guided()

    private = guided_private_reviewed_facts(guided)
    public = guided_redacted_planner_context(guided)

    private_text = repr(private)
    public_text = repr(public)
    assert all(canary in private_text for canary in CANARIES[:3])
    assert all(canary not in public_text for canary in CANARIES)
    assert public == {
        "schema": "guided.reviewed-planner-context.v1",
        "sources": [
            {
                "stable_id": SOURCE_ID,
                "plugin": "csv",
                "observed_columns": ["name", "score"],
                "option_keys": ["credentials", "inline_blob", "schema"],
                "on_validation_failure": "discard",
            }
        ],
        "outputs": [
            {
                "stable_id": OUTPUT_ID,
                "plugin": "json",
                "required_fields": ["name"],
                "schema_mode": "observed",
                "option_keys": ["path", "schema"],
                "on_write_failure": "discard",
            }
        ],
        "deferred_intents": [],
    }


def test_option_value_constraint_exposes_only_closed_structural_semantics_to_provider() -> None:
    guided = replace(
        _guided(),
        deferred_intents=(
            DeferredStageIntent.create(
                intent_id="00000000-0000-4000-8000-000000000105",
                receiving_stage="output",
                target_stage="topology",
                catalog_kind="source",
                catalog_name="csv",
                redacted_summary="Apply a private option constraint.",
                originating_message_id="00000000-0000-4000-8000-000000000106",
                message_content_hash=stable_hash("private option instruction"),
                constraints=(
                    OptionValueConstraint(
                        kind="option_value",
                        subject=StableSubject(kind="stable", component_kind="source", stable_id=SOURCE_ID),
                        option_path=(DEFERRED_PATH_CANARY, "value"),
                        operator="equals",
                        value=DEFERRED_VALUE_CANARY,
                    ),
                ),
            ),
        ),
    )
    state = CompositionState(
        sources={},
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
        guided_session=guided,
    )

    private_constraint = guided.deferred_intents[0].constraints[0].to_dict()
    provider_reviewed = guided_redacted_planner_context(guided)
    provider_current = guided_redacted_current_state_context(state)

    assert private_constraint["option_path"] == [DEFERRED_PATH_CANARY, "value"]
    assert private_constraint["value"] == DEFERRED_VALUE_CANARY
    for provider_context in (provider_reviewed, provider_current):
        rendered = repr(provider_context)
        assert DEFERRED_PATH_CANARY not in rendered
        assert DEFERRED_VALUE_CANARY not in rendered
    assert provider_reviewed["deferred_intents"][0]["constraints"] == [
        {
            "kind": "option_value",
            "subject": {"kind": "stable", "component_kind": "source", "stable_id": SOURCE_ID},
            "operator": "equals",
            "value_type": "string",
            "value_present": True,
        }
    ]


def test_projection_is_closed_redacted_and_reverified_against_private_authority() -> None:
    guided = _guided()
    proposal = _proposal(guided)

    payload = build_guided_proposal_projection(
        proposal_id=PROPOSAL_ID,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids={
            "source": frozenset({"csv"}),
            "transform": frozenset({"normalize"}),
            "sink": frozenset({"json"}),
        },
    )

    rendered = repr(payload)
    assert all(canary not in rendered for canary in CANARIES)
    assert payload["proposal_id"] == str(PROPOSAL_ID)
    assert payload["draft_hash"] == proposal.draft_hash
    assert payload["graph"]["sources"][0]["stable_id"] == SOURCE_ID
    assert payload["outputs"][0]["stable_id"] == OUTPUT_ID
    verify_guided_proposal_projection(
        payload=payload,
        proposal_id=PROPOSAL_ID,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids={
            "source": frozenset({"csv"}),
            "transform": frozenset({"normalize"}),
            "sink": frozenset({"json"}),
        },
    )

    payload["nodes"][0]["plugin"]["id"] = "different"
    with pytest.raises(AuditIntegrityError, match="projection"):
        verify_guided_proposal_projection(
            payload=payload,
            proposal_id=PROPOSAL_ID,
            proposal=proposal,
            guided=guided,
            catalog_plugin_ids={
                "source": frozenset({"csv"}),
                "transform": frozenset({"normalize"}),
                "sink": frozenset({"json"}),
            },
        )


def test_projection_rejects_plugins_outside_the_same_catalog_snapshot() -> None:
    guided = _guided()
    with pytest.raises(AuditIntegrityError, match="catalog"):
        build_guided_proposal_projection(
            proposal_id=PROPOSAL_ID,
            proposal=_proposal(guided),
            guided=guided,
            catalog_plugin_ids={
                "source": frozenset({"csv"}),
                "transform": frozenset(),
                "sink": frozenset({"json"}),
            },
        )


@pytest.mark.parametrize("planner", (plan_pipeline, prepare_pipeline_plan))
def test_planner_requires_private_and_provider_safe_facts_plus_lineage(planner: object) -> None:
    signature = inspect.signature(planner)
    for name in (
        "reviewed_facts",
        "reviewed_planner_context",
        "covered_deferred_intent_ids",
        "supersedes_draft_hash",
    ):
        assert signature.parameters[name].default is inspect.Parameter.empty
