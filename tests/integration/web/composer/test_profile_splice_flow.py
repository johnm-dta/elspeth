"""Combined profile lowering, splice, review, audit, and YAML contract."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import pytest

from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import begin_dispatch, finish_success
from elspeth.web.composer.proposals import build_tool_proposal_summary
from elspeth.web.composer.redaction import redact_tool_call_arguments
from elspeth.web.composer.redaction_telemetry import NoopRedactionTelemetry
from elspeth.web.composer.state import CompositionState, EdgeSpec, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.composer.tools import execute_tool
from elspeth.web.composer.yaml_generator import generate_public_yaml
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import (
    INTERPRETATION_REQUIREMENTS_KEY,
    PROMPT_SHIELD_USER_TERM,
    WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
    pipeline_decision_artifact_hash,
)
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig

_SHIELD = PluginId("transform", "aws_bedrock_prompt_shield")
_PRIVATE_BINDING = "privatespliceguardrailmarker"


@dataclass(frozen=True)
class _AllSecretsInventory:
    def has_server_ref(self, name: str) -> bool:
        del name
        return True

    def has_user_ref(self, principal: str, name: str) -> bool:
        del principal, name
        return True

    def has_ref(self, principal: str, name: str) -> bool:
        del principal, name
        return True

    def server_generation(self, name: str) -> str:
        del name
        return "generation"

    def user_generation(self, principal: str, name: str) -> str:
        del principal, name
        return "generation"


def _policy_context() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    settings = WebSettings.model_validate(
        {
            "composer_max_composition_turns": 4,
            "composer_max_discovery_turns": 4,
            "composer_timeout_seconds": 60,
            "composer_rate_limit_per_minute": 20,
            "shareable_link_signing_key": b"0123456789abcdef0123456789abcdef",
            "plugin_allowlist": (str(_SHIELD),),
            "bedrock_guardrail_profiles": (
                {
                    "alias": "prompt-default",
                    "plugin": _SHIELD.name,
                    "guardrail_identifier": _PRIVATE_BINDING,
                    "guardrail_version": "91",
                    "region": "ap-southeast-1",
                },
            ),
            "llm_profiles": {
                "llm-default": {
                    "provider": "bedrock",
                    "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
                    "region_name": "ap-southeast-1",
                }
            },
        }
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
    catalog = create_catalog_service()
    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=catalog,
        profiles=profiles,
        principal_scope="web:splice-test",
        secret_inventory=_AllSecretsInventory(),
        generation_key=b"profile-splice-flow-generation-key",
    )
    return PolicyCatalogView(catalog, snapshot, profiles), snapshot


def _resolved_requirement(
    *,
    requirement_id: str,
    kind: InterpretationKind,
    user_term: str,
    draft: str,
    accepted_hash: str,
    prompt_hash: bool = False,
) -> dict[str, object]:
    return {
        "id": requirement_id,
        "kind": kind.value,
        "user_term": user_term,
        "status": "resolved",
        "draft": draft,
        "event_id": f"event:{requirement_id}",
        "accepted_value": "approved",
        "accepted_artifact_hash": None if prompt_hash else accepted_hash,
        "resolved_prompt_template_hash": accepted_hash if prompt_hash else None,
    }


def _reviewed_state() -> CompositionState:
    scrape_options: dict[str, object] = {
        "url_field": "url",
        "content_field": "page_text",
        "fingerprint_field": "page_fingerprint",
        "format": "text",
        "http": {
            "abuse_contact": "profile-splice@foundryside.dev",
            "scraping_reason": "profile splice integration contract",
        },
        "schema": {"mode": "observed"},
    }
    llm_options: dict[str, object] = {
        "profile": "llm-default",
        "prompt_template": "Summarise {{ row['page_text'] }}",
        "required_input_fields": ["page_text"],
        "schema": {"mode": "observed", "fields": None},
    }
    scrape = NodeSpec(
        id="scrape",
        node_type="transform",
        plugin="web_scrape",
        input="scrape_in",
        on_success="llm_in",
        on_error="discard",
        options=scrape_options,
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    llm = NodeSpec(
        id="llm",
        node_type="transform",
        plugin="llm",
        input="llm_in",
        on_success="output",
        on_error="discard",
        options=llm_options,
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    http_review = _resolved_requirement(
        requirement_id="http-identity:scrape",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM,
        draft="Approve the web scrape HTTP identity.",
        accepted_hash=pipeline_decision_artifact_hash(scrape, (scrape, llm), user_term=WEB_SCRAPE_HTTP_IDENTITY_USER_TERM),
    )
    shield_review = _resolved_requirement(
        requirement_id="prompt-shield:llm",
        kind=InterpretationKind.PIPELINE_DECISION,
        user_term=PROMPT_SHIELD_USER_TERM,
        draft="Insert a prompt shield.",
        accepted_hash=pipeline_decision_artifact_hash(llm, (scrape, llm), user_term=PROMPT_SHIELD_USER_TERM),
    )
    prompt_review = _resolved_requirement(
        requirement_id="prompt-template:llm",
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
        user_term="llm_prompt_template:llm",
        draft="Summarise the page text.",
        accepted_hash=stable_hash(llm_options["prompt_template"]),
        prompt_hash=True,
    )
    scrape = replace(scrape, options={**scrape_options, INTERPRETATION_REQUIREMENTS_KEY: [http_review]})
    llm = replace(llm, options={**llm_options, INTERPRETATION_REQUIREMENTS_KEY: [shield_review, prompt_review]})
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="scrape_in",
            options={"path": "rows.csv", "schema": {"mode": "fixed", "fields": ["url: str"]}},
            on_validation_failure="discard",
        ),
        nodes=(scrape, llm),
        edges=(
            EdgeSpec(id="source-scrape", from_node="source", to_node="scrape", edge_type="on_success", label=None),
            EdgeSpec(id="scrape-llm", from_node="scrape", to_node="llm", edge_type="on_success", label=None),
            EdgeSpec(id="llm-output", from_node="llm", to_node="output", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="output",
                plugin="json",
                options={
                    "path": "out.jsonl",
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="profile splice"),
        version=8,
    )


def test_profile_splice_flow_is_valid_review_aware_and_private_binding_safe(caplog: pytest.LogCaptureFixture) -> None:
    view, snapshot = _policy_context()
    state = _reviewed_state()
    arguments = {
        "predecessor_id": "scrape",
        "successor_id": "llm",
        "node": {
            "id": "prompt_shield",
            "plugin": _SHIELD.name,
            "options": {
                "profile": "prompt-default",
                "fields": ["page_text"],
                "schema": {"mode": "observed", "fields": None},
            },
            "on_error": "discard",
        },
    }

    result = execute_tool("splice_transform", arguments, state, view, plugin_snapshot=snapshot)

    assert result.success, result.data
    assert result.updated_state.version == state.version + 1
    assert [node.id for node in result.updated_state.nodes] == ["scrape", "prompt_shield", "llm"]
    assert [edge.id for edge in result.updated_state.edges] == [
        "source-scrape",
        "scrape-llm",
        "scrape-llm__splice__prompt_shield",
        "llm-output",
    ]
    shield = result.updated_state.nodes[1]
    assert shield.options["profile"] == "prompt-default"
    assert "guardrail_identifier" not in shield.options
    llm_requirements = result.updated_state.nodes[2].options[INTERPRETATION_REQUIREMENTS_KEY]
    assert [row["kind"] for row in llm_requirements] == [InterpretationKind.LLM_PROMPT_TEMPLATE.value]
    assert llm_requirements[0]["status"] == "resolved"
    scrape_requirements = result.updated_state.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY]
    assert scrape_requirements[0]["status"] == "resolved"
    assert result.validation.is_valid, result.validation.errors
    shield_contract = next(contract for contract in result.validation.edge_contracts if contract.from_id == "prompt_shield")
    assert "page_text" in shield_contract.producer_guarantees
    assert shield_contract.consumer_requires == ("page_text",)
    assert shield_contract.satisfied

    preview = execute_tool(
        "preview_pipeline",
        {},
        result.updated_state,
        view,
        plugin_snapshot=snapshot,
    )
    assert preview.success, preview.data
    assert preview.validation.is_valid, preview.validation.errors

    redacted = redact_tool_call_arguments(
        "splice_transform",
        arguments,
        telemetry=NoopRedactionTelemetry(),
    )
    proposal = build_tool_proposal_summary(tool_name="splice_transform", arguments=arguments, redacted_arguments=redacted)
    audit = finish_success(
        begin_dispatch("call-splice", "splice_transform", arguments, version_before=state.version, actor="composer"),
        result_payload=result.to_dict(),
        version_after=result.updated_state.version,
    )
    validation = view.validate_composition_state(result.updated_state)
    assert validation.executable_state.nodes[1].options["guardrail_identifier"] == _PRIVATE_BINDING
    surfaces = {
        "dispatcher": json.dumps(result.to_dict(), sort_keys=True),
        "tool_message": json.dumps(result.to_dict(), sort_keys=True),
        "proposal": json.dumps(deep_thaw(proposal.arguments_redacted_json), sort_keys=True),
        "audit": json.dumps(audit.to_dict(), sort_keys=True),
        "yaml": generate_public_yaml(result.updated_state),
        "logs": caplog.text,
    }
    for surface, rendered in surfaces.items():
        assert _PRIVATE_BINDING not in rendered, surface
