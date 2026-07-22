"""Security and integrity contracts for guided proposal projection."""

from __future__ import annotations

import inspect
from dataclasses import replace
from itertools import permutations, product
from typing import get_type_hints
from uuid import UUID

import pytest

import elspeth.web.composer.guided.planning as guided_planning
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.guided.planning import (
    build_guided_proposal_projection,
    guided_private_reviewed_facts,
    guided_redacted_current_state_context,
    guided_redacted_planner_context,
    verified_remaining_deferred_intents,
    verify_guided_proposal_projection,
)
from elspeth.web.composer.guided.protocol import GuidedStep, ProposePipelinePayload, proposal_structural_label
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SourceResolved
from elspeth.web.composer.guided.stage_subjects import EdgeRouteConstraint, OptionValueConstraint, PluginSubject, StableSubject
from elspeth.web.composer.guided.state_machine import DeferredStageIntent, GuidedSession
from elspeth.web.composer.pipeline_planner import plan_pipeline, prepare_pipeline_plan
from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface, PresentBase
from elspeth.web.composer.state import CompositionState, PipelineMetadata

SOURCE_ID = "00000000-0000-4000-8000-000000000101"
OUTPUT_ID = "00000000-0000-4000-8000-000000000102"
PROPOSAL_ID = UUID("00000000-0000-4000-8000-000000000103")
CHECKPOINT_ID = UUID("00000000-0000-4000-8000-000000000104")
MIXED_INTENT_ID = "00000000-0000-4000-8000-000000000107"
MIXED_MESSAGE_ID = "00000000-0000-4000-8000-000000000108"
GATE_ID = "00000000-0000-4000-8000-000000000109"
PASSTHROUGH_SUBJECT_ID = "00000000-0000-4000-8000-000000000110"
SECOND_GATE_ID = "00000000-0000-4000-8000-000000000111"
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


def _mixed_gate_guided() -> GuidedSession:
    return replace(
        _guided(),
        deferred_intents=(
            DeferredStageIntent.create(
                intent_id=MIXED_INTENT_ID,
                receiving_stage="output",
                target_stage="wire_review",
                catalog_kind=None,
                catalog_name=None,
                redacted_summary="Retain direct and fork gate routes.",
                originating_message_id=MIXED_MESSAGE_ID,
                message_content_hash=stable_hash("mixed gate instruction"),
                constraints=(
                    EdgeRouteConstraint(
                        kind="edge_route",
                        from_subject=StableSubject(kind="stable", component_kind="node", stable_id=GATE_ID),
                        edge_type="route_false",
                        to_subject=StableSubject(kind="stable", component_kind="output", stable_id=OUTPUT_ID),
                        present=True,
                    ),
                    EdgeRouteConstraint(
                        kind="edge_route",
                        from_subject=StableSubject(kind="stable", component_kind="node", stable_id=GATE_ID),
                        edge_type="fork",
                        to_subject=PluginSubject(
                            kind="plugin",
                            subject_id=PASSTHROUGH_SUBJECT_ID,
                            plugin_kind="transform",
                            plugin_name="passthrough",
                        ),
                        present=True,
                    ),
                ),
            ),
        ),
    )


def _mixed_gate_proposal(guided: GuidedSession, routes: dict[str, str]) -> PipelineProposal:
    return PipelineProposal.create(
        pipeline={
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "gate-input",
                    "options": {"schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [
                {
                    "id": GATE_ID,
                    "node_type": "gate",
                    "plugin": None,
                    "input": "gate-input",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "condition": "row['accepted']",
                    "routes": routes,
                    "fork_to": ["accepted"],
                },
                {
                    "id": "copy",
                    "node_type": "transform",
                    "plugin": "passthrough",
                    "input": "accepted",
                    "on_success": "cleaned",
                    "on_error": "discard",
                    "options": {},
                },
            ],
            "edges": [],
            "outputs": [
                {
                    "name": "cleaned",
                    "plugin": "json",
                    "options": {"schema": {"mode": "observed"}},
                    "on_write_failure": "discard",
                }
            ],
        },
        base=PresentBase(state_id=CHECKPOINT_ID, composition_content_hash="a" * 64),
        reviewed_facts=guided_private_reviewed_facts(guided),
        surface=PlannerSurface.GUIDED_STAGED,
        repair_count=0,
        skill_hash=stable_hash("guided planner skill"),
        covered_deferred_intent_ids=(MIXED_INTENT_ID,),
        supersedes_draft_hash=None,
    )


def _two_gate_proposal(
    guided: GuidedSession,
    *,
    first_routes: dict[str, str],
    second_routes: dict[str, str],
) -> PipelineProposal:
    return PipelineProposal.create(
        pipeline={
            "sources": {
                "primary": {
                    "plugin": "csv",
                    "on_success": "first-gate-input",
                    "options": {"schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [
                {
                    "id": GATE_ID,
                    "node_type": "gate",
                    "plugin": None,
                    "input": "first-gate-input",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "condition": "row['first']",
                    "routes": first_routes,
                    "fork_to": [],
                },
                {
                    "id": SECOND_GATE_ID,
                    "node_type": "gate",
                    "plugin": None,
                    "input": "second-gate-input",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "condition": "row['second']",
                    "routes": second_routes,
                    "fork_to": [],
                },
                {
                    "id": "copy",
                    "node_type": "transform",
                    "plugin": "passthrough",
                    "input": "accepted",
                    "on_success": "cleaned",
                    "on_error": "discard",
                    "options": {},
                },
            ],
            "edges": [],
            "outputs": [
                {
                    "name": "cleaned",
                    "plugin": "json",
                    "options": {"schema": {"mode": "observed"}},
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


def _build_with_fixed_projection_ids(
    monkeypatch: pytest.MonkeyPatch,
    *,
    proposal: PipelineProposal,
    guided: GuidedSession,
    catalog: dict[str, frozenset[str]],
    allocated_id_count: int,
) -> ProposePipelinePayload:
    allocated_ids = iter(UUID(f"00000000-0000-4000-8000-{index:012d}") for index in range(200, 200 + allocated_id_count))
    monkeypatch.setattr(guided_planning, "uuid4", lambda: next(allocated_ids))
    return build_guided_proposal_projection(
        proposal_id=PROPOSAL_ID,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids=catalog,
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
                "name": "primary",
                "plugin": "csv",
                "observed_columns": ["name", "score"],
                "option_keys": ["credentials", "inline_blob", "schema"],
                "on_validation_failure": "discard",
            }
        ],
        "outputs": [
            {
                "stable_id": OUTPUT_ID,
                "name": "cleaned",
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


def test_mixed_gate_projection_is_canonical_and_exact_for_every_route_insertion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guided = _mixed_gate_guided()
    catalog = {
        "source": frozenset({"csv"}),
        "transform": frozenset({"passthrough"}),
        "sink": frozenset({"json"}),
    }
    route_entries = (
        ("alpha", "accepted"),
        ("false", "cleaned"),
        ("beta", "fork"),
        ("true", "fork"),
    )
    payloads = []
    draft_hashes = set()
    for route_permutation in permutations(route_entries):
        proposal = _mixed_gate_proposal(guided, dict(route_permutation))
        payload = _build_with_fixed_projection_ids(
            monkeypatch,
            proposal=proposal,
            guided=guided,
            catalog=catalog,
            allocated_id_count=10,
        )
        verify_guided_proposal_projection(
            payload=payload,
            proposal_id=PROPOSAL_ID,
            proposal=proposal,
            guided=guided,
            catalog_plugin_ids=catalog,
        )
        assert verified_remaining_deferred_intents(guided=guided, proposal=proposal) == ()
        draft_hashes.add(proposal.draft_hash)
        payloads.append(payload)

    assert len(draft_hashes) == 1
    assert all(payload == payloads[0] for payload in payloads)
    payload = payloads[0]
    route_aliases = [proposal_structural_label("route", index) for index in range(4)]
    gate = next(node for node in payload["nodes"] if node["stable_id"] and node["node_type"] == "gate")
    assert gate["behavior"] == {
        "kind": "gate",
        "route_aliases": route_aliases,
        "fork_branches": [
            {
                "routes": route_aliases[2:],
                "branch": proposal_structural_label("branch", 0),
            }
        ],
    }
    gate_flows = [edge["flow"] for edge in payload["graph"]["edges"] if edge["from_endpoint"].get("stable_id") == gate["stable_id"]]
    assert gate_flows == [
        {"kind": "gate_route", "route": route_aliases[0], "branch": None},
        {"kind": "gate_route", "route": route_aliases[1], "branch": None},
        {
            "kind": "gate_fork",
            "routes": route_aliases[2:],
            "branch": proposal_structural_label("branch", 0),
        },
    ]


def test_repeated_route_labels_are_gate_local_and_canonical_for_every_insertion_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guided = _guided()
    catalog = {
        "source": frozenset({"csv"}),
        "transform": frozenset({"passthrough"}),
        "sink": frozenset({"json"}),
    }
    first_entries = (("true", "second-gate-input"), ("false", "cleaned"))
    second_entries = (("true", "accepted"), ("false", "cleaned"))
    payloads = []
    draft_hashes = set()
    route_orders = product(permutations(first_entries), permutations(second_entries))
    for first_routes, second_routes in route_orders:
        proposal = _two_gate_proposal(
            guided,
            first_routes=dict(first_routes),
            second_routes=dict(second_routes),
        )
        payload = _build_with_fixed_projection_ids(
            monkeypatch,
            proposal=proposal,
            guided=guided,
            catalog=catalog,
            allocated_id_count=12,
        )
        verify_guided_proposal_projection(
            payload=payload,
            proposal_id=PROPOSAL_ID,
            proposal=proposal,
            guided=guided,
            catalog_plugin_ids=catalog,
        )
        draft_hashes.add(proposal.draft_hash)
        payloads.append(payload)

    assert len(draft_hashes) == 1
    assert all(payload == payloads[0] for payload in payloads)
    gates = [node for node in payloads[0]["nodes"] if node["node_type"] == "gate"]
    route_aliases = [proposal_structural_label("route", index) for index in range(4)]
    assert [gate["behavior"]["route_aliases"] for gate in gates] == [route_aliases[:2], route_aliases[2:]]
    assert len({alias for gate in gates for alias in gate["behavior"]["route_aliases"]}) == 4
    assert [
        edge["flow"]
        for gate in gates
        for edge in payloads[0]["graph"]["edges"]
        if edge["from_endpoint"].get("stable_id") == gate["stable_id"]
    ] == [
        {"kind": "gate_route", "route": route_aliases[0], "branch": None},
        {"kind": "gate_route", "route": route_aliases[1], "branch": None},
        {"kind": "gate_route", "route": route_aliases[2], "branch": None},
        {"kind": "gate_route", "route": route_aliases[3], "branch": None},
    ]


def _ab_coalesce_guided() -> GuidedSession:
    return replace(
        GuidedSession.initial(),
        reviewed_sources={
            SOURCE_ID: SourceResolved(
                name="source",
                plugin="csv",
                options={"path": "blob:00000000-0000-0000-0000-000000000001", "schema": {"mode": "observed"}},
                observed_columns=("color_name", "hex"),
                sample_rows=(),
                on_validation_failure="discard",
            )
        },
        reviewed_outputs={
            OUTPUT_ID: SinkOutputResolved(
                name="colour_ab_out",
                plugin="json",
                options={"path": "out.json", "schema": {"mode": "observed"}},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            )
        },
        source_order=(SOURCE_ID,),
        output_order=(OUTPUT_ID,),
        step=GuidedStep.STEP_3_TRANSFORMS,
    )


def _ab_coalesce_proposal(guided: GuidedSession) -> PipelineProposal:
    """A runnable fork -> llm x2 -> coalesce -> field_mapper A/B (session 30acb16e shape)."""
    return PipelineProposal.create(
        pipeline={
            "sources": {
                "source": {
                    "plugin": "csv",
                    "on_success": "csv_rows",
                    "options": {"path": "blob:00000000-0000-0000-0000-000000000001", "schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [
                {
                    "id": "fork_gate",
                    "node_type": "gate",
                    "plugin": None,
                    "input": "csv_rows",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "condition": "len(row['color_name']) > 0",
                    "routes": {"true": "fork", "false": "fork"},
                    "fork_to": ["a_rows", "b_rows"],
                },
                {
                    "id": "llm_variant_a",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "a_rows",
                    "on_success": "a_out",
                    "on_error": "discard",
                    "options": {"provider": "openrouter"},
                },
                {
                    "id": "llm_variant_b",
                    "node_type": "transform",
                    "plugin": "llm",
                    "input": "b_rows",
                    "on_success": "b_out",
                    "on_error": "discard",
                    "options": {"provider": "openrouter"},
                },
                {
                    "id": "reconcile",
                    "node_type": "coalesce",
                    "plugin": None,
                    "input": "a_out",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "branches": {"a_rows": "a_out", "b_rows": "b_out"},
                    "policy": "require_all",
                    "merge": "union",
                },
                {
                    "id": "cleanup",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "reconcile",
                    "on_success": "colour_ab_out",
                    "on_error": "discard",
                    "options": {},
                },
            ],
            "edges": [],
            "outputs": [
                {
                    "name": "colour_ab_out",
                    "plugin": "json",
                    "options": {"path": "out.json"},
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


def test_fork_coalesce_ab_projection_routes_every_branch_output_into_the_coalesce() -> None:
    """A fork/coalesce A/B plan must project without an orphaned branch output.

    Regression for guided A/B session 503cac64: ``canonical_connection_consumers``
    keys consumers off ``node.input`` only, so the coalesce's second branch output
    (``b_out``, referenced only via ``branches``) had "no canonical consumer" and
    ``_build_projection`` raised AuditIntegrityError before the proposal could show.
    """
    guided = _ab_coalesce_guided()
    proposal = _ab_coalesce_proposal(guided)
    catalog = {
        "source": frozenset({"csv"}),
        "transform": frozenset({"llm", "field_mapper"}),
        "sink": frozenset({"json"}),
    }

    payload = build_guided_proposal_projection(
        proposal_id=PROPOSAL_ID,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids=catalog,
    )

    verify_guided_proposal_projection(
        payload=payload,
        proposal_id=PROPOSAL_ID,
        proposal=proposal,
        guided=guided,
        catalog_plugin_ids=catalog,
    )
    node_id_by_role = {node["behavior"]["kind"]: node["stable_id"] for node in payload["nodes"]}
    coalesce_id = node_id_by_role["coalesce"]
    # Both LLM branch producers route their success output into the coalesce.
    coalesce_incoming = [edge for edge in payload["graph"]["edges"] if edge["to_endpoint"].get("stable_id") == coalesce_id]
    assert len(coalesce_incoming) == 2
    assert all(edge["flow"]["kind"] == "node_success" for edge in coalesce_incoming)
    # The coalesce republishes its merged rows to the downstream field_mapper.
    coalesce_outgoing = [edge for edge in payload["graph"]["edges"] if edge["from_endpoint"].get("stable_id") == coalesce_id]
    assert [edge["flow"]["kind"] for edge in coalesce_outgoing] == ["coalesce_success"]


def _ab_coalesce_proposal_ordered(
    guided: GuidedSession,
    *,
    branch_order: tuple[int, int],
    node_order: tuple[int, int],
) -> PipelineProposal:
    """The fork/coalesce A/B with the branch dict and LLM node order permuted."""
    branches = [("a_rows", "a_out"), ("b_rows", "b_out")]
    llm_nodes = [
        {
            "id": "llm_variant_a",
            "node_type": "transform",
            "plugin": "llm",
            "input": "a_rows",
            "on_success": "a_out",
            "on_error": "discard",
            "options": {"provider": "openrouter"},
        },
        {
            "id": "llm_variant_b",
            "node_type": "transform",
            "plugin": "llm",
            "input": "b_rows",
            "on_success": "b_out",
            "on_error": "discard",
            "options": {"provider": "openrouter"},
        },
    ]
    ordered_branches = {branches[i][0]: branches[i][1] for i in branch_order}
    ordered_llm = [llm_nodes[i] for i in node_order]
    return PipelineProposal.create(
        pipeline={
            "sources": {
                "source": {
                    "plugin": "csv",
                    "on_success": "csv_rows",
                    "options": {"path": "blob:00000000-0000-0000-0000-000000000001", "schema": {"mode": "observed"}},
                    "on_validation_failure": "discard",
                }
            },
            "nodes": [
                {
                    "id": "fork_gate",
                    "node_type": "gate",
                    "plugin": None,
                    "input": "csv_rows",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "condition": "len(row['color_name']) > 0",
                    "routes": {"true": "fork", "false": "fork"},
                    "fork_to": ["a_rows", "b_rows"],
                },
                *ordered_llm,
                {
                    "id": "reconcile",
                    "node_type": "coalesce",
                    "plugin": None,
                    "input": "a_out",
                    "on_success": None,
                    "on_error": None,
                    "options": {},
                    "branches": ordered_branches,
                    "policy": "require_all",
                    "merge": "union",
                },
                {
                    "id": "cleanup",
                    "node_type": "transform",
                    "plugin": "field_mapper",
                    "input": "reconcile",
                    "on_success": "colour_ab_out",
                    "on_error": "discard",
                    "options": {},
                },
            ],
            "edges": [],
            "outputs": [{"name": "colour_ab_out", "plugin": "json", "options": {"path": "out.json"}, "on_write_failure": "discard"}],
        },
        base=PresentBase(state_id=CHECKPOINT_ID, composition_content_hash="a" * 64),
        reviewed_facts=guided_private_reviewed_facts(guided),
        surface=PlannerSurface.GUIDED_STAGED,
        repair_count=0,
        skill_hash=stable_hash("guided planner skill"),
        covered_deferred_intent_ids=(),
        supersedes_draft_hash=None,
    )


def test_fork_coalesce_projection_is_invariant_to_branch_and_node_ordering() -> None:
    """The coalesce projection must validate+verify for every authored order.

    Regression for live guided session 63f0b04a (AuditIntegrityError at
    validate_payload:1011, "coalesce branch aliases do not match its incoming
    flows"): the planner authors the coalesce ``branches`` dict and its
    branch-producer nodes in a nondeterministic order, and the validator requires
    ``behavior.branch_aliases`` to equal the incoming-flow branch order. Deriving
    the behavior aliases from the coalesce's own incoming edges makes them agree
    by construction, so every permutation must project AND re-verify.
    """
    guided = _ab_coalesce_guided()
    catalog = {
        "source": frozenset({"csv"}),
        "transform": frozenset({"llm", "field_mapper"}),
        "sink": frozenset({"json"}),
    }
    for branch_order in permutations((0, 1)):
        for node_order in permutations((0, 1)):
            proposal = _ab_coalesce_proposal_ordered(guided, branch_order=branch_order, node_order=node_order)
            payload = build_guided_proposal_projection(
                proposal_id=PROPOSAL_ID,
                proposal=proposal,
                guided=guided,
                catalog_plugin_ids=catalog,
            )
            verify_guided_proposal_projection(
                payload=payload,
                proposal_id=PROPOSAL_ID,
                proposal=proposal,
                guided=guided,
                catalog_plugin_ids=catalog,
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


def test_planner_requires_private_provider_safe_and_model_claim_authority() -> None:
    model_signature = inspect.signature(plan_pipeline)
    for name in (
        "reviewed_facts",
        "reviewed_planner_context",
        "eligible_deferred_intent_ids",
        "claim_evaluator",
        "supersedes_draft_hash",
    ):
        assert model_signature.parameters[name].default is inspect.Parameter.empty

    server_signature = inspect.signature(prepare_pipeline_plan)
    for name in ("reviewed_facts", "reviewed_planner_context", "supersedes_draft_hash"):
        assert server_signature.parameters[name].default is inspect.Parameter.empty
    assert "covered_deferred_intent_ids" not in server_signature.parameters

    verifier_signature = inspect.signature(verified_remaining_deferred_intents)
    assert tuple(verifier_signature.parameters) == ("guided", "proposal")
    assert get_type_hints(verified_remaining_deferred_intents)["return"] == tuple[DeferredStageIntent, ...]
