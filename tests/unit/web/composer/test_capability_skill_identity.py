"""Capability-core identity, coverage, and planner-manifest contracts."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.capability_skill import (
    CANONICAL_CAPABILITY_FIELDS,
    CAPABILITY_CORE_NODE_GUIDANCE,
    PlannerCapabilityManifest,
    build_planner_capability_manifest,
    canonical_capability_fields,
    documented_capability_fields,
    load_pipeline_capability_core,
    validate_capability_field_contract,
)
from elspeth.web.composer.guided.prompts import load_step_chat_skill, load_step_planner_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.pipeline_planner import PLANNER_DISCOVERY_TOOL_NAMES, planner_tool_definitions
from elspeth.web.composer.pipeline_proposal import PlannerSurface
from elspeth.web.composer.prompts import PIPELINE_COMPOSER_SKILL_HASH, build_system_prompt
from elspeth.web.composer.skills import load_skill
from elspeth.web.composer.state import COMPOSER_NODE_TYPES
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema


def _messages(rendered_skill: str, *, sensitive_user_text: str = "build it") -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": rendered_skill},
        {"role": "user", "content": sensitive_user_text},
    ]


def _manifest(
    rendered_skill: str,
    *,
    surface: PlannerSurface,
    profile: str,
    sensitive_user_text: str = "build it",
) -> PlannerCapabilityManifest:
    return build_planner_capability_manifest(
        surface=surface,
        profile=profile,
        messages=_messages(rendered_skill, sensitive_user_text=sensitive_user_text),
        tools=planner_tool_definitions(),
        canonical_schema=canonical_set_pipeline_schema(),
    )


def test_freeform_and_every_guided_planner_prepend_identical_core_bytes() -> None:
    core = load_pipeline_capability_core()
    surfaces = [build_system_prompt(None), *(load_step_planner_skill(step) for step in GuidedStep)]

    for rendered in surfaces:
        assert rendered.startswith(core)
        assert rendered.count(core) == 1


def test_guided_chat_prompts_are_interaction_only_and_advertise_no_planner_terminal() -> None:
    core = load_pipeline_capability_core()

    for step in GuidedStep:
        rendered = load_step_chat_skill(step)
        assert not rendered.startswith(core)
        assert core not in rendered
        assert "emit_pipeline_proposal" not in rendered


def test_guided_chat_prompts_name_only_tools_in_their_actual_palette() -> None:
    prompts = {step: load_step_chat_skill(step) for step in GuidedStep}

    for prompt in prompts.values():
        assert "list_sources" not in prompt
        assert "list_transforms" not in prompt
        assert "list_models" not in prompt

    assert "list_sinks" not in prompts[GuidedStep.STEP_1_SOURCE]
    assert "get_plugin_schema" not in prompts[GuidedStep.STEP_1_SOURCE]
    assert "list_sinks" in prompts[GuidedStep.STEP_2_SINK]
    assert "get_plugin_schema" in prompts[GuidedStep.STEP_2_SINK]
    assert "confirm_wiring" not in prompts[GuidedStep.STEP_4_WIRE]


def test_capability_facts_have_one_document_owner() -> None:
    core = load_skill("pipeline_capabilities")
    interaction = load_skill("pipeline_composer")
    anchors = (
        "[capability:discovery-order]",
        "[capability:topology]",
        "[capability:canonical-fields]",
        "[capability:field-contracts]",
        "[capability:structured-output-repair]",
        "[capability:plugin-assistance]",
    )

    for anchor in anchors:
        assert core.count(anchor) == 1
        assert anchor not in interaction

    assert "## Discovery And Credentials" not in interaction
    assert "### Multi-source Pipelines" not in interaction
    assert "### Field Wiring" not in interaction
    assert "### LLM Nodes" not in interaction
    assert "For `batch_stats`" not in interaction
    assert "For `batch_stats`" not in core
    assert "get_plugin_assistance" in core


def test_static_planner_guidance_contains_no_deployment_plugin_facts() -> None:
    rendered_prompts = [build_system_prompt(None), *(load_step_planner_skill(step) for step in GuidedStep)]
    forbidden_facts = (
        "web_scrape",
        "field_mapper",
        "azure_prompt_shield",
        '"provider": "openrouter"',
        "OPENROUTER_API_KEY",
        "collision_policy",
        "auto_increment",
        "url_field",
        "response_field",
        "llm_response",
        "CSV source means CSV sinks",
        "Author free-text generated sources as JSON",
        "Headered mode (no `columns`)",
        "`text` source treats every non-blank line",
        "Azure-blob",
        "Dataverse",
        "dataverse",
        "null-source placeholder",
        "csv_source_blob_header_mismatch",
    )

    for rendered in rendered_prompts:
        assert all(fact not in rendered for fact in forbidden_facts)

    core = load_pipeline_capability_core()
    assert "get_plugin_assistance" in core
    assert "policy-visible" in core
    assert "prompt-injection" in core
    assert "cleanup" in core
    assert "An option key such as" not in core
    assert "Single-query LLM output remains one response column" not in core


def test_freeform_skill_hash_covers_the_exact_composed_static_prompt() -> None:
    rendered = build_system_prompt(None)

    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == PIPELINE_COMPOSER_SKILL_HASH


def test_guided_skills_have_no_capability_reduction_disclaimers() -> None:
    forbidden = (
        "guided chains cannot include",
        "chain cannot express",
        "must be added as a gate after the guided build",
        "switch to freeform for",
        "supports only a single linear spine",
        "cannot build multiple sources",
        "cannot build multiple outputs",
        "aggregation is unavailable",
        "row expansion is unavailable",
        "structured llm fields are unavailable",
    )

    for step in GuidedStep:
        lowered = load_step_planner_skill(step).lower()
        assert all(phrase not in lowered for phrase in forbidden)


def test_capability_coverage_is_exactly_derived_from_canonical_authorities() -> None:
    actual_fields = canonical_capability_fields(canonical_set_pipeline_schema())

    assert actual_fields == CANONICAL_CAPABILITY_FIELDS
    assert documented_capability_fields(load_pipeline_capability_core()) == actual_fields
    assert set(CAPABILITY_CORE_NODE_GUIDANCE) == set(COMPOSER_NODE_TYPES)
    core = load_pipeline_capability_core()
    assert all(core.count(anchor) == 1 for anchor in CAPABILITY_CORE_NODE_GUIDANCE.values())


def test_capability_field_extraction_detects_new_structural_field() -> None:
    schema = canonical_set_pipeline_schema()
    schema["properties"]["future_topology"] = {"type": "object"}

    assert canonical_capability_fields(schema) != CANONICAL_CAPABILITY_FIELDS


def test_manifest_rejects_schema_and_terminal_updated_without_documented_field() -> None:
    schema = canonical_set_pipeline_schema()
    schema["properties"]["future_topology"] = {"type": "object"}
    tools = planner_tool_definitions()
    tools[-1]["function"]["parameters"]["properties"]["pipeline"] = schema

    with pytest.raises(AuditIntegrityError, match="documented capability fields drifted"):
        build_planner_capability_manifest(
            surface=PlannerSurface.FREEFORM,
            profile="ordinary",
            messages=_messages(build_system_prompt(None)),
            tools=tools,
            canonical_schema=schema,
        )


@pytest.mark.parametrize("mutation", ("missing_start", "missing_family", "malformed_fields"))
def test_documented_field_inventory_fails_closed_on_drift(mutation: str) -> None:
    core = load_pipeline_capability_core()
    if mutation == "missing_start":
        changed = core.replace("<!-- canonical-field-inventory:start -->", "", 1)
    elif mutation == "missing_family":
        changed = core.replace("| trigger | `count`, `timeout_seconds`, `condition` |\n", "", 1)
    else:
        changed = core.replace("| metadata | `name`, `description` |", "| metadata | name, description |", 1)

    with pytest.raises(AuditIntegrityError):
        if mutation == "missing_family":
            validate_capability_field_contract(canonical_set_pipeline_schema(), changed)
        else:
            documented_capability_fields(changed)


def test_manifest_uses_exact_ordered_advertised_tool_definitions() -> None:
    tools = planner_tool_definitions()
    manifest = _manifest(build_system_prompt(None), surface=PlannerSurface.FREEFORM, profile="ordinary")

    assert [tool["function"]["name"] for tool in tools] == [*PLANNER_DISCOVERY_TOOL_NAMES, "emit_pipeline_proposal"]
    assert manifest.effective_tool_hash == stable_hash(tools)
    assert manifest.canonical_schema_hash == stable_hash(canonical_set_pipeline_schema())


@pytest.mark.parametrize(
    ("surface", "profile", "rendered_skill"),
    (
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
    ),
)
def test_every_runtime_surface_builds_the_same_capability_manifest_core(
    surface: PlannerSurface,
    profile: str,
    rendered_skill: str,
) -> None:
    manifest = _manifest(rendered_skill, surface=surface, profile=profile)

    assert manifest.surface is surface
    assert manifest.profile == profile
    assert manifest.capability_core_hash == hashlib.sha256(load_pipeline_capability_core().encode("utf-8")).hexdigest()
    assert manifest.canonical_schema_hash == stable_hash(canonical_set_pipeline_schema())
    assert manifest.effective_tool_hash == stable_hash(planner_tool_definitions())


@pytest.mark.parametrize("mutation", ("missing_core", "duplicate_core", "reordered_tools", "mutated_schema"))
def test_manifest_fails_closed_on_capability_identity_drift(mutation: str) -> None:
    core = load_pipeline_capability_core()
    messages = _messages(build_system_prompt(None))
    tools = planner_tool_definitions()
    if mutation == "missing_core":
        messages[0]["content"] = "interaction only"
    elif mutation == "duplicate_core":
        messages[0]["content"] = f"{core}{core}"
    elif mutation == "reordered_tools":
        tools[0], tools[1] = tools[1], tools[0]
    else:
        tools[-1]["function"]["parameters"]["properties"]["pipeline"]["properties"].pop("edges")

    with pytest.raises(AuditIntegrityError):
        build_planner_capability_manifest(
            surface=PlannerSurface.FREEFORM,
            profile="ordinary",
            messages=messages,
            tools=tools,
            canonical_schema=canonical_set_pipeline_schema(),
        )


def test_manifest_is_hash_only_and_never_copies_private_prompt_values(tmp_path: Path) -> None:
    private_value = "sk-private-provider-value-never-public"
    deployment = tmp_path / "skills"
    deployment.mkdir()
    (deployment / "pipeline_composer.md").write_text(f"Private deployment instruction: {private_value}\n")

    manifest = _manifest(
        build_system_prompt(str(tmp_path)),
        surface=PlannerSurface.FREEFORM,
        profile="ordinary",
        sensitive_user_text=private_value,
    )
    rendered = repr(asdict(manifest))

    assert private_value not in rendered
    assert set(asdict(manifest)) == {
        "surface",
        "profile",
        "planner_implementation_id",
        "capability_core_hash",
        "canonical_schema_hash",
        "effective_tool_hash",
        "rendered_prompt_hash",
    }
