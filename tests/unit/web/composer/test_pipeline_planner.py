"""Shared pipeline planner contract tests.

Every scripted model response uses the LiteLLM response shape and therefore
crosses the production response parser.  Tests never inject a proposal or a
candidate state as a provider result.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
import structlog
from litellm.exceptions import APIError as LiteLLMAPIError
from sqlalchemy import func, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.capability_skill import load_pipeline_capability_core
from elspeth.web.composer.guided.deferred_intents import DeferredIntentClaimError
from elspeth.web.composer.guided.prompts import load_step_planner_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.pipeline_planner import (
    PLANNER_DISCOVERY_TOOL_NAMES,
    PipelinePlannerError,
    PlannerBudgetPolicy,
    PlannerCustodyConfig,
    PlannerDeclined,
    PlannerModelConfig,
    PlannerOriginatingMessage,
    PlannerRequestLifecycle,
    _allowlisted_candidate_feedback,
    _parse_response_tool_calls,
    plan_pipeline,
    planner_tool_definitions,
)
from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
from elspeth.web.composer.planner_authoring_aids import build_planner_authoring_aids
from elspeth.web.composer.prompts import build_system_prompt
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationEntry, ValidationSummary
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, composition_proposals_table
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@dataclass
class _Function:
    name: str
    arguments: object


@dataclass
class _ToolCall:
    id: str
    function: _Function | None


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: Mapping[str, object]
    model: str | None = "provider/planner-v1"
    id: str = "request-1"


class _ScriptedCompletion:
    def __init__(self, *responses: _Response | BaseException) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> _Response:
        self.requests.append(deepcopy(kwargs))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _planner_usage(*, cost: object = 0.01) -> dict[str, object]:
    return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost}


def _response(*calls: tuple[str, object], cost: object = 0.01) -> _Response:
    tool_calls = [
        _ToolCall(id=f"call-{index}", function=_Function(name=name, arguments=json.dumps(arguments)))
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    return _Response(
        choices=[_Choice(message=_Message(content=None, tool_calls=tool_calls))],
        usage=_planner_usage(cost=cost),
    )


def _response_with_usage(
    *calls: tuple[str, object],
    cost: object = 0.01,
    completion_tokens: object = 5,
) -> _Response:
    response = _response(*calls, cost=cost)
    response.usage = {
        "prompt_tokens": 10,
        "completion_tokens": completion_tokens,
        "total_tokens": 10 + completion_tokens if isinstance(completion_tokens, int) else None,
        "cost": cost,
    }
    return response


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _pipeline(data_dir: Path) -> dict[str, Any]:
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(data_dir / "blobs" / "input.csv"), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [],
        "edges": [],
        "outputs": [
            {
                "sink_name": "rows",
                "plugin": "json",
                "options": {
                    "path": str(data_dir / "outputs" / "result.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
    }


def _inline_pipeline(data_dir: Path, *, output_name: str = "rows") -> dict[str, Any]:
    pipeline = _pipeline(data_dir)
    pipeline["source"] = {
        "plugin": "csv",
        "on_success": "rows",
        "options": {"schema": {"mode": "observed"}},
        "on_validation_failure": "discard",
        "inline_blob": {
            "filename": "input.csv",
            "mime_type": "text/csv",
            "content": "name,score\nada,42\n",
        },
    }
    pipeline["outputs"][0]["sink_name"] = output_name
    return pipeline


def _invalid_pipeline(data_dir: Path) -> dict[str, Any]:
    pipeline = _pipeline(data_dir)
    pipeline["outputs"][0]["sink_name"] = "not_rows"
    return pipeline


def _pipeline_with_short_form_llm_review(data_dir: Path) -> dict[str, Any]:
    """A valid csv -> llm -> json plan whose LLM node carries the skill's short form.

    The composer skill instructs the planner to stage a ``pipeline_decision``
    review as ``{kind, user_term, draft}`` (no server-owned ``id`` / ``status``).
    """
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"path": str(data_dir / "blobs" / "input.csv"), "schema": {"mode": "observed"}},
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "summarise",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "summarised",
                "on_error": "discard",
                "options": {
                    "schema": {"mode": "observed"},
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet-4.6",
                    "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                    "prompt_template": "Summarise {{ text }}",
                    "interpretation_requirements": [
                        {
                            "kind": "pipeline_decision",
                            "user_term": "prompt_injection_shield_recommendation",
                            "draft": "Recommend inserting a prompt-injection shield before this LLM.",
                        }
                    ],
                },
            }
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "summarised",
                "plugin": "json",
                "options": {
                    "path": str(data_dir / "outputs" / "result.jsonl"),
                    "schema": {"mode": "observed"},
                    "format": "jsonl",
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
    }


def _budget(**overrides: object) -> PlannerBudgetPolicy:
    values: dict[str, object] = {
        "max_total_provider_calls": 4,
        "max_request_bytes": 1_000_000,
        "max_completion_tokens": 800,
        "max_cumulative_provider_cost": Decimal("1.00"),
    }
    values.update(overrides)
    return PlannerBudgetPolicy(**values)  # type: ignore[arg-type]


def _model(completion: _ScriptedCompletion, **overrides: object) -> PlannerModelConfig:
    values: dict[str, object] = {
        "completion": completion,
        "model_identifier": "anthropic/claude-planner",
        "provider": "test-provider",
        "temperature": 0.0,
        "seed": 7,
        "timeout_seconds": 5.0,
        "max_composition_turns": 4,
        "max_discovery_turns": 3,
        "max_tool_calls_per_turn": 3,
        "max_api_attempts": 1,
        "api_retry_base_seconds": 0.0,
    }
    values.update(overrides)
    return PlannerModelConfig(**values)  # type: ignore[arg-type]


def _origin() -> PlannerOriginatingMessage:
    return PlannerOriginatingMessage(
        session_id=str(uuid4()),
        message_id=str(uuid4()),
        content="Build the requested pipeline.",
        user_id="planner-user",
    )


def _custody(tmp_path: Path) -> PlannerCustodyConfig:
    return PlannerCustodyConfig(
        data_dir=str(tmp_path),
        session_engine=None,
        max_storage_per_session=1_000_000,
        secret_service=None,
        runtime_preflight=None,
    )


def _lifecycle(events: list[str] | None = None) -> PlannerRequestLifecycle:
    observed = events if events is not None else []

    async def before_start() -> None:
        observed.append("before")

    @asynccontextmanager
    async def request_scope() -> AsyncIterator[None]:
        observed.append("scope-enter")
        try:
            yield
        finally:
            observed.append("scope-exit")

    async def on_settled(outcome: str) -> None:
        observed.append(f"settled:{outcome}")

    return PlannerRequestLifecycle(
        before_start=before_start,
        request_scope=request_scope,
        on_settled=on_settled,
        progress=None,
    )


async def _plan(
    *,
    tmp_path: Path,
    tool_context: ToolContext,
    completion: _ScriptedCompletion,
    recorder: BufferingRecorder | None = None,
    budget: PlannerBudgetPolicy | None = None,
    repair_budget: int = 1,
    lifecycle: PlannerRequestLifecycle | None = None,
    model_overrides: Mapping[str, object] | None = None,
    originating_message: PlannerOriginatingMessage | None = None,
    custody_config: PlannerCustodyConfig | None = None,
    current_state: CompositionState | None = None,
    intent: str = "Build the requested pipeline.",
    surface: PlannerSurface = PlannerSurface.FREEFORM,
    profile: str | None = None,
    eligible_deferred_intent_ids: tuple[str, ...] = (),
    claim_evaluator: Any = None,
    rendered_skill: str | None = None,
    supersedes_draft_hash: str | None = None,
) -> Any:
    # Candidate validation needs the real plugin contracts.  ``tool_context``
    # remains in the test signature so the standard composer fixture proves
    # the API accepts the same context types, but its deliberately skeletal
    # MagicMock catalog is insufficient for a complete pipeline.
    del tool_context
    full_catalog = create_catalog_service()
    plugin_snapshot = PluginAvailabilitySnapshot.for_trained_operator(full_catalog)
    policy_catalog = PolicyCatalogView.for_trained_operator(full_catalog, plugin_snapshot)
    return await plan_pipeline(
        intent=intent,
        current_state=current_state or _empty_state(),
        provider_current_state=(current_state or _empty_state()).to_dict(),
        reviewed_facts={"request": "Build the requested pipeline."},
        reviewed_planner_context={"request": "Build the requested pipeline."},
        eligible_deferred_intent_ids=eligible_deferred_intent_ids,
        claim_evaluator=claim_evaluator,
        supersedes_draft_hash=supersedes_draft_hash,
        surface=surface,
        profile=profile or ("tutorial" if surface is PlannerSurface.TUTORIAL_PROFILE else "ordinary"),
        policy_catalog=policy_catalog,
        plugin_snapshot=plugin_snapshot,
        originating_message=originating_message or _origin(),
        base=AbsentBase(),
        model_config=_model(completion, **dict(model_overrides or {})),
        rendered_skill=rendered_skill or f"{load_pipeline_capability_core()}\n\nYou are the bounded ELSPETH pipeline planner.",
        repair_budget=repair_budget,
        budget_policy=budget or _budget(),
        custody_config=custody_config or _custody(tmp_path),
        lifecycle=lifecycle or _lifecycle(),
        recorder=recorder or BufferingRecorder(),
        candidate_finalizer=lambda candidate: candidate,
    )


def test_planner_palette_is_pinned_read_only_and_terminal_schema_is_exact() -> None:
    expected_discovery = {
        "diff_pipeline",
        "explain_validation_error",
        "get_audit_info",
        "get_expression_grammar",
        "get_pipeline_state",
        "get_plugin_assistance",
        "get_plugin_schema",
        "list_models",
        "list_recipes",
        "list_sinks",
        "list_sources",
        "list_transforms",
        "preview_pipeline",
        "get_blob_content",
        "get_blob_metadata",
        "inspect_source",
        "list_blobs",
        "list_composer_blobs",
        "list_secret_refs",
        "validate_secret_ref",
    }
    assert set(PLANNER_DISCOVERY_TOOL_NAMES) == expected_discovery

    tools = planner_tool_definitions()
    assert {tool["function"]["name"] for tool in tools[:-1]} == expected_discovery
    terminal = tools[-1]["function"]
    assert terminal["name"] == "emit_pipeline_proposal"
    assert terminal["parameters"] == {
        "type": "object",
        "properties": {
            "pipeline": canonical_set_pipeline_schema(),
            "claimed_deferred_intent_ids": {
                "type": "array",
                "items": {"type": "string", "format": "uuid"},
                "uniqueItems": True,
            },
        },
        "required": ["pipeline"],
        "additionalProperties": False,
    }
    serialized = canonical_json(terminal)
    assert "rationale" not in serialized
    assert '"base"' not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_claims",
    [
        [str(uuid4())],
        ["00000000-0000-4000-8000-000000000311", "00000000-0000-4000-8000-000000000311"],
    ],
)
@pytest.mark.parametrize("surface", (PlannerSurface.FREEFORM, PlannerSurface.GUIDED_FULL))
async def test_ineligible_and_duplicate_deferred_claims_use_bounded_terminal_repair(
    tmp_path: Path,
    tool_context: ToolContext,
    invalid_claims: list[str],
    surface: PlannerSurface,
) -> None:
    completion = _ScriptedCompletion(
        _response(
            (
                "emit_pipeline_proposal",
                {"pipeline": _pipeline(tmp_path), "claimed_deferred_intent_ids": invalid_claims},
            )
        ),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    result = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, surface=surface)

    assert result.proposal.repair_count == 1
    feedback = completion.requests[1]["messages"][-1]
    assert feedback["role"] == "tool"
    assert "deferred_intent_claim" in feedback["content"]


@pytest.mark.asyncio
async def test_guided_claims_are_verified_from_candidate_and_unproven_claims_repair(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    intent_id = "00000000-0000-4000-8000-000000000312"
    completion = _ScriptedCompletion(
        _response(
            (
                "emit_pipeline_proposal",
                {"pipeline": _pipeline(tmp_path), "claimed_deferred_intent_ids": [intent_id]},
            )
        ),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    evaluations = 0

    def reject_unproven(_candidate: CompositionState, _claims: tuple[str, ...]) -> tuple[str, ...]:
        nonlocal evaluations
        evaluations += 1
        raise DeferredIntentClaimError("unproven")

    result = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        surface=PlannerSurface.GUIDED_STAGED,
        eligible_deferred_intent_ids=(intent_id,),
        claim_evaluator=reject_unproven,
    )

    assert evaluations == 1
    assert result.proposal.covered_deferred_intent_ids == ()
    assert result.proposal.repair_count == 1
    assert "deferred_intent_claim" in completion.requests[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_eligible_deferred_claims_require_a_mechanical_evaluator(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    with pytest.raises(ValueError, match="claim_evaluator"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=_ScriptedCompletion(),
            surface=PlannerSurface.GUIDED_STAGED,
            eligible_deferred_intent_ids=("00000000-0000-4000-8000-000000000313",),
            claim_evaluator=None,
        )


@pytest.mark.asyncio
async def test_happy_path_returns_proposal_and_audits_exact_marked_wire_payload(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    pipeline = _pipeline(tmp_path)
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": pipeline})))
    recorder = BufferingRecorder()
    lifecycle_events: list[str] = []
    policy = _budget()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        budget=policy,
        lifecycle=_lifecycle(lifecycle_events),
    )

    assert deep_thaw(proposal.proposal.pipeline) == pipeline
    assert proposal.proposal.base == AbsentBase()
    assert proposal.proposal.repair_count == 0
    assert lifecycle_events == ["before", "scope-enter", "scope-exit", "settled:complete"]
    assert len(completion.requests) == 1
    sent = completion.requests[0]
    assert sent["max_tokens"] == policy.max_completion_tokens
    assert sent["messages"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["tools"][-1]["cache_control"] == {"type": "ephemeral"}

    (audit,) = recorder.llm_calls
    assert audit.messages_hash == stable_hash(sent["messages"])
    assert audit.tools_spec_hash == stable_hash(sent["tools"])
    assert audit.max_completion_tokens_requested == policy.max_completion_tokens
    assert audit.planner_policy_hash == policy.audit_hash
    assert audit.planner_call_ordinal == 1


@pytest.mark.asyncio
async def test_authored_short_form_node_review_is_canonicalized_into_the_sealed_proposal(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """The durable proposal — not only the transient candidate — carries the full form.

    Regression for the guided first-run tutorial 500: the planner authors the
    skill's short-form ``{kind, user_term, draft}`` review; canonicalisation must
    reach ``safe_pipeline`` so the sealed proposal a later accept/commit re-reads
    is already valid, not a latent re-crash.
    """
    pipeline = _pipeline_with_short_form_llm_review(tmp_path)
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": pipeline})))

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    sealed = deep_thaw(proposal.proposal.pipeline)
    requirements = sealed["nodes"][0]["options"]["interpretation_requirements"]
    shield = next(item for item in requirements if item["user_term"] == "prompt_injection_shield_recommendation")
    assert shield["id"] == "prompt_injection_shield_recommendation:summarise"
    assert shield["status"] == "pending"


@pytest.mark.asyncio
async def test_unguarded_candidate_error_becomes_typed_planner_failure(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KeyError/TypeError/ValueError escaping the candidate builder is not a raw 500.

    Backstop for any residual unguarded lookup: it must surface as the planner's
    typed failure idiom naming the offending key, not propagate as a bare
    exception the route reports as "The operation failed."
    """
    import elspeth.web.composer.pipeline_planner as planner_module

    def _raise_unguarded(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyError("status")

    monkeypatch.setattr(planner_module, "build_set_pipeline_candidate", _raise_unguarded)
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    with pytest.raises(PipelinePlannerError) as caught:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert caught.value.code == "CANDIDATE_CONSTRUCTION_ERROR"
    assert "KeyError" in str(caught.value)
    assert "status" in str(caught.value)


@pytest.mark.asyncio
async def test_real_planner_call_builds_manifest_from_exact_audited_inputs(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    captured: list[Any] = []
    real_builder = planner_module.build_planner_capability_manifest

    def capture_manifest(**kwargs: Any) -> Any:
        manifest = real_builder(**kwargs)
        captured.append(manifest)
        return manifest

    monkeypatch.setattr(planner_module, "build_planner_capability_manifest", capture_manifest)
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
    recorder = BufferingRecorder()

    await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert len(captured) == 1
    assert captured[0].rendered_prompt_hash == recorder.llm_calls[0].messages_hash
    assert captured[0].effective_tool_hash == recorder.llm_calls[0].tools_spec_hash


@pytest.mark.asyncio
async def test_real_planner_surface_paths_share_exact_capabilities_tools_and_audit(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    captured: list[Any] = []
    real_builder = planner_module.build_planner_capability_manifest

    def capture_manifest(**kwargs: Any) -> Any:
        manifest = real_builder(**kwargs)
        captured.append(manifest)
        return manifest

    monkeypatch.setattr(planner_module, "build_planner_capability_manifest", capture_manifest)
    step_3_planner = load_step_planner_skill(GuidedStep.STEP_3_TRANSFORMS)
    scenarios = (
        (PlannerSurface.FREEFORM, "ordinary", build_system_prompt(None)),
        (PlannerSurface.GUIDED_STAGED, "ordinary", step_3_planner),
        (PlannerSurface.TUTORIAL_PROFILE, "tutorial", step_3_planner),
    )
    requests: list[dict[str, Any]] = []
    audits: list[Any] = []

    for surface, profile, rendered_skill in scenarios:
        completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
        recorder = BufferingRecorder()
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            recorder=recorder,
            surface=surface,
            profile=profile,
            rendered_skill=rendered_skill,
        )
        requests.append(completion.requests[0])
        audits.append(recorder.llm_calls[0])

    assert [(manifest.surface, manifest.profile) for manifest in captured] == [
        (surface, profile) for surface, profile, _rendered_skill in scenarios
    ]
    assert len({manifest.planner_implementation_id for manifest in captured}) == 1
    assert len({manifest.capability_core_hash for manifest in captured}) == 1
    assert len({manifest.canonical_schema_hash for manifest in captured}) == 1
    assert len({manifest.effective_tool_hash for manifest in captured}) == 1
    assert requests[0]["messages"][0]["content"] == build_system_prompt(None)
    assert requests[1]["messages"][0]["content"] == step_3_planner
    assert requests[2]["messages"][0]["content"] == step_3_planner
    assert requests[1]["tools"] == requests[2]["tools"] == requests[0]["tools"]
    for manifest, request, audit_call in zip(captured, requests, audits, strict=True):
        assert manifest.rendered_prompt_hash == stable_hash(request["messages"])
        assert manifest.effective_tool_hash == stable_hash(request["tools"])
        assert audit_call.messages_hash == manifest.rendered_prompt_hash
        assert audit_call.tools_spec_hash == manifest.effective_tool_hash


@pytest.mark.asyncio
async def test_provider_side_call_input_mutation_is_detected_as_audit_integrity_failure(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    class _MutatingCompletion(_ScriptedCompletion):
        async def __call__(self, **kwargs: Any) -> _Response:
            kwargs["tools"][-1]["function"]["parameters"]["properties"]["pipeline"]["properties"].pop("edges")
            return await super().__call__(**kwargs)

    completion = _MutatingCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    with pytest.raises(AuditIntegrityError, match="planner call inputs changed"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_outcome", "expected_status"),
    (
        (_response(("emit_pipeline_proposal", {"pipeline": {}})), ComposerLLMCallStatus.SUCCESS),
        (RuntimeError("provider unavailable"), ComposerLLMCallStatus.API_ERROR),
        (asyncio.CancelledError(), ComposerLLMCallStatus.CANCELLED),
    ),
)
async def test_provider_input_mutation_is_audited_once_before_integrity_failure(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
    provider_outcome: _Response | BaseException,
    expected_status: ComposerLLMCallStatus,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    class _MutatingCompletion(_ScriptedCompletion):
        async def __call__(self, **kwargs: Any) -> _Response:
            kwargs["messages"][0]["content"] += "\nprovider-side mutation"
            return await super().__call__(**kwargs)

    recorder = BufferingRecorder()
    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=_ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)}))),
        recorder=recorder,
    )
    (unrelated_prior_call,) = recorder.llm_calls
    captured_manifests: list[Any] = []
    real_builder = planner_module.build_planner_capability_manifest

    def capture_manifest(**kwargs: Any) -> Any:
        manifest = real_builder(**kwargs)
        captured_manifests.append(manifest)
        return manifest

    monkeypatch.setattr(planner_module, "build_planner_capability_manifest", capture_manifest)

    with pytest.raises(AuditIntegrityError, match="planner call inputs changed") as caught:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=_MutatingCompletion(provider_outcome),
            recorder=recorder,
        )

    assert len(captured_manifests) == 1
    assert len(recorder.llm_calls) == 2
    (manifest,) = captured_manifests
    audit_call = recorder.llm_calls[-1]
    assert caught.value.llm_calls == (audit_call,)  # type: ignore[attr-defined]
    assert unrelated_prior_call not in caught.value.llm_calls  # type: ignore[attr-defined]
    assert audit_call.status is expected_status
    assert audit_call.messages_hash != manifest.rendered_prompt_hash
    assert audit_call.tools_spec_hash == manifest.effective_tool_hash


@pytest.mark.asyncio
async def test_discovery_round_uses_real_read_only_tool_then_terminal(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    # Tool results land before the budget-pressure notice that fires at two
    # remaining discovery turns.
    assert completion.requests[1]["messages"][-2]["role"] == "tool"
    assert completion.requests[1]["messages"][-1]["role"] == "user"
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].tool_name == "list_sources"
    assert [call.planner_call_ordinal for call in recorder.llm_calls] == [1, 2]


@pytest.mark.asyncio
async def test_parallel_discovery_results_remain_correlated_by_tool_call_id(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    original = planner_module.execute_discovery_tool_with_context
    rendezvous = threading.Barrier(2)

    def synchronized_discovery(*args: Any, **kwargs: Any) -> Any:
        tool_name = args[0]
        rendezvous.wait(timeout=2)
        result = original(*args, **kwargs)
        return replace(result, data={"marker": tool_name})

    monkeypatch.setattr(planner_module, "execute_discovery_tool_with_context", synchronized_discovery)
    completion = _ScriptedCompletion(
        _response(("list_sources", {}), ("list_sinks", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    tool_messages = [message for message in completion.requests[1]["messages"] if message["role"] == "tool"]
    assert [(message["tool_call_id"], json.loads(message["content"])["data"]["marker"]) for message in tool_messages] == [
        ("call-1", "list_sources"),
        ("call-2", "list_sinks"),
    ]
    assert {call.tool_call_id: call.tool_name for call in recorder.invocations} == {
        "call-1": "list_sources",
        "call-2": "list_sinks",
    }


@pytest.mark.asyncio
async def test_parallel_discovery_failure_closes_every_audit_before_return(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    original = planner_module.execute_discovery_tool_with_context
    rendezvous = threading.Barrier(2)
    release_sibling = threading.Event()
    sibling_worker_finished = threading.Event()

    def controlled_discovery(*args: Any, **kwargs: Any) -> Any:
        tool_name = args[0]
        rendezvous.wait(timeout=2)
        if tool_name == "list_sources":
            raise RuntimeError("deterministic-primary-discovery-failure")
        release_sibling.wait(timeout=5)
        try:
            return original(*args, **kwargs)
        finally:
            sibling_worker_finished.set()

    monkeypatch.setattr(planner_module, "execute_discovery_tool_with_context", controlled_discovery)
    completion = _ScriptedCompletion(_response(("list_sources", {}), ("list_sinks", {})))
    recorder = BufferingRecorder()
    events: list[str] = []

    try:
        with pytest.raises(RuntimeError, match="deterministic-primary-discovery-failure"):
            await _plan(
                tmp_path=tmp_path,
                tool_context=tool_context,
                completion=completion,
                recorder=recorder,
                lifecycle=_lifecycle(events),
            )
        assert len(recorder.invocations) == 2
        invocations_by_tool = {invocation.tool_name: invocation for invocation in recorder.invocations}
        assert invocations_by_tool["list_sources"].status.value == "plugin_crash"
        assert invocations_by_tool["list_sources"].error_class == "RuntimeError"
        assert invocations_by_tool["list_sinks"].status.value == "cancelled"
        assert invocations_by_tool["list_sinks"].error_class == "CancelledError"
        assert invocations_by_tool["list_sinks"].error_message == "sibling_failure"
        closed_snapshot = tuple(call.to_dict() for call in recorder.invocations)
        assert events[-1] == "settled:failed"
    finally:
        release_sibling.set()

    for _attempt in range(200):
        if sibling_worker_finished.is_set():
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0)
    assert sibling_worker_finished.is_set()
    assert tuple(call.to_dict() for call in recorder.invocations) == closed_snapshot


@pytest.mark.asyncio
async def test_parallel_discovery_cancellation_closes_every_audit_before_return(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    original = planner_module.execute_discovery_tool_with_context
    rendezvous = threading.Barrier(2)
    both_workers_entered = threading.Event()
    release_workers = threading.Event()
    finished_count = 0
    finished_lock = threading.Lock()

    def controlled_discovery(*args: Any, **kwargs: Any) -> Any:
        nonlocal finished_count
        rendezvous.wait(timeout=2)
        both_workers_entered.set()
        release_workers.wait(timeout=5)
        try:
            return original(*args, **kwargs)
        finally:
            with finished_lock:
                finished_count += 1

    monkeypatch.setattr(planner_module, "execute_discovery_tool_with_context", controlled_discovery)
    recorder = BufferingRecorder()
    events: list[str] = []
    task = asyncio.create_task(
        _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=_ScriptedCompletion(_response(("list_sources", {}), ("list_sinks", {}))),
            recorder=recorder,
            lifecycle=_lifecycle(events),
        )
    )
    try:
        await asyncio.wait_for(asyncio.to_thread(both_workers_entered.wait, 2), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(recorder.invocations) == 2
        assert {invocation.status.value for invocation in recorder.invocations} == {"cancelled"}
        assert {invocation.error_class for invocation in recorder.invocations} == {"CancelledError"}
        assert {invocation.error_message for invocation in recorder.invocations} == {"coordinator_cancelled"}
        closed_snapshot = tuple(call.to_dict() for call in recorder.invocations)
        assert events[-1] == "settled:cancelled"
    finally:
        release_workers.set()

    for _attempt in range(200):
        with finished_lock:
            if finished_count == 2:
                break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0)
    with finished_lock:
        assert finished_count == 2
    assert tuple(call.to_dict() for call in recorder.invocations) == closed_snapshot


@pytest.mark.asyncio
async def test_exact_intent_appears_in_the_sole_user_role_message(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    exact_intent = "Read the audited input and write the canonical report."
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        intent=exact_intent,
    )

    user_messages = [message for message in completion.requests[0]["messages"] if message["role"] == "user"]
    assert len(user_messages) == 1
    assert json.loads(user_messages[0]["content"])["intent"] == exact_intent


@pytest.mark.asyncio
async def test_live_authoring_aids_ride_in_the_reviewed_context_user_message(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """Deployment plugin exemplars enter through the live user-message channel.

    The static system prompt must stay free of deployment plugin facts (the
    ``no_deployment_plugin_facts`` gate), so the worked exemplars render at
    prompt-build from the policy-visible catalog into the same reviewed-context
    message that carries the intent — present on every turn, not only after a
    failure.
    """
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    request = completion.requests[0]
    user_messages = [message for message in request["messages"] if message["role"] == "user"]
    assert len(user_messages) == 1
    payload = json.loads(user_messages[0]["content"])
    full_catalog = create_catalog_service()
    plugin_snapshot = PluginAvailabilitySnapshot.for_trained_operator(full_catalog)
    expected = build_planner_authoring_aids(PolicyCatalogView.for_trained_operator(full_catalog, plugin_snapshot))
    assert payload["authoring_aids"] == json.loads(canonical_json(expected))
    # The aids PAYLOAD stays out of the system message: skill-hash identity is
    # pinned to the static pack, and the capability core's
    # no-deployment-inventory claim about system text must remain true. (The
    # core may NAME the authoring_aids channel as a pointer — what must never
    # appear in system text is the rendered payload itself.)
    system_messages = [message for message in request["messages"] if message["role"] == "system"]
    assert all("set_pipeline_exemplar" not in message["content"] for message in system_messages)
    assert all("composer_hints" not in message["content"] for message in system_messages)
    assert all('"authoring_aids"' not in message["content"] for message in system_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "profile"),
    [
        (PlannerSurface.GUIDED_STAGED, "ordinary"),
        (PlannerSurface.TUTORIAL_PROFILE, "tutorial"),
    ],
)
async def test_authoring_aids_reach_guided_staged_and_tutorial_surfaces(
    tmp_path: Path,
    tool_context: ToolContext,
    surface: PlannerSurface,
    profile: str,
) -> None:
    """The live aids ride in the planner user message on EVERY surface.

    The guided-staged and tutorial planners enter through the same
    ``_plan_pipeline_inner`` message assembly as freeform (their surface is a
    ``plan_pipeline`` argument, not a different code path), so the digest and
    worked exemplars must appear in their sole reviewed-context user message
    under the guided step skill exactly as they do under the freeform skill —
    and the payload must stay out of the (hash-pinned) step system text.
    """
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        surface=surface,
        profile=profile,
        rendered_skill=load_step_planner_skill(GuidedStep.STEP_3_TRANSFORMS),
    )

    request = completion.requests[0]
    user_messages = [message for message in request["messages"] if message["role"] == "user"]
    assert len(user_messages) == 1
    payload = json.loads(user_messages[0]["content"])
    full_catalog = create_catalog_service()
    plugin_snapshot = PluginAvailabilitySnapshot.for_trained_operator(full_catalog)
    expected = build_planner_authoring_aids(PolicyCatalogView.for_trained_operator(full_catalog, plugin_snapshot))
    assert payload["authoring_aids"] == json.loads(canonical_json(expected))
    assert "discovery_digest" in payload["authoring_aids"]
    system_messages = [message for message in request["messages"] if message["role"] == "system"]
    assert all("set_pipeline_exemplar" not in message["content"] for message in system_messages)
    assert all('"authoring_aids"' not in message["content"] for message in system_messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("temperature", "seed", "expected"),
    [
        (None, None, {}),
        (0.25, 1234, {"temperature": 0.25, "seed": 1234}),
    ],
)
async def test_temperature_and_seed_are_omitted_or_passed_exactly(
    tmp_path: Path,
    tool_context: ToolContext,
    temperature: float | None,
    seed: int | None,
    expected: dict[str, object],
) -> None:
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        model_overrides={"temperature": temperature, "seed": seed},
    )

    actual = {key: completion.requests[0][key] for key in ("temperature", "seed") if key in completion.requests[0]}
    assert actual == expected


@pytest.mark.asyncio
async def test_non_anthropic_requests_have_no_cache_markers(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        model_overrides={"model_identifier": "openai/gpt-5"},
    )

    for request in completion.requests:
        assert all("cache_control" not in message for message in request["messages"])
        assert all("cache_control" not in tool for tool in request["tools"])


@pytest.mark.asyncio
async def test_anthropic_cache_markers_stay_stable_across_discovery_rounds(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    marked_system = [request["messages"][0] for request in completion.requests]
    marked_tools = [request["tools"] for request in completion.requests]
    assert all(message["cache_control"] == {"type": "ephemeral"} for message in marked_system)
    assert marked_system[0] == marked_system[1] == marked_system[2]
    assert marked_tools[0] == marked_tools[1] == marked_tools[2]
    assert all(tools[-1]["cache_control"] == {"type": "ephemeral"} for tools in marked_tools)


@pytest.mark.asyncio
async def test_missing_source_candidate_fails_closed_before_full_candidate_is_accepted(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    missing_source = _pipeline(tmp_path)
    del missing_source["source"]
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": missing_source})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert proposal.proposal.repair_count == 1
    feedback = json.loads(completion.requests[1]["messages"][-1]["content"])
    assert feedback["success"] is False
    assert feedback["validation"]["is_valid"] is False
    # The pre-application rejection carries the closed code itself; the
    # unchanged empty state's errors are gated out of planner feedback
    # (tutorial op 1152d7e3: they were red herrings on every OTHER semantic
    # rejection, steering repairs toward re-authoring source/sinks).
    assert [error["component"] for error in feedback["validation"]["errors"]] == ["rejected_mutation"]
    assert feedback["validation"]["errors"][0]["error_code"] == "no_source_configured"
    assert all(error["error_class"] == "ValidationError" for error in feedback["validation"]["errors"])


@pytest.mark.asyncio
async def test_nodeless_revision_candidate_gets_one_coded_nudge_then_valve(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A guided revision that nets zero transforms is nudged once, not shipped.

    Tutorial op 1152d7e3 (2026-07-22): after blind repairs the planner
    "converged" by dropping every node — a bare passthrough whose metadata
    still claimed to scrape/summarize/clean. On a revision turn (a rejected
    draft is superseded, so the operator explicitly asked for changes) a
    candidate with zero transform/aggregation nodes must draw ONE coded
    rejection; re-emitting the same nodeless pipeline is the escape valve
    confirming a deliberate pass-through (9137456ad omit-valve pattern).
    """
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        surface=PlannerSurface.GUIDED_STAGED,
        supersedes_draft_hash=stable_hash("superseded-draft"),
    )

    assert proposal.proposal.repair_count == 1
    feedback = json.loads(completion.requests[1]["messages"][-1]["content"])
    assert feedback["success"] is False
    codes = [error["error_code"] for error in feedback["validation"]["errors"]]
    assert codes == ["proposal_missing_requested_transforms"]
    assert feedback["validation"]["errors"][0]["explanation"]
    assert feedback["validation"]["errors"][0]["suggested_fix"]


@pytest.mark.asyncio
async def test_transformful_revision_candidate_passes_without_nudge(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline_with_short_form_llm_review(tmp_path)})),
    )

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        surface=PlannerSurface.GUIDED_STAGED,
        supersedes_draft_hash=stable_hash("superseded-draft"),
    )

    assert proposal.proposal.repair_count == 0


@pytest.mark.asyncio
async def test_nodeless_initial_guided_plan_is_not_nudged(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    # The initial step_3 auto-proposal (no superseded draft — the operator has
    # not asked for a revision) may legitimately be a plain pass-through; the
    # guard must not tax every simple walk with a nudge cycle.
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        surface=PlannerSurface.GUIDED_STAGED,
        supersedes_draft_hash=None,
    )

    assert proposal.proposal.repair_count == 0


def test_allowlisted_candidate_feedback_enriches_node_shape_codes() -> None:
    """A rejected candidate's bare closed codes carry their fix guidance.

    The repair feedback strips raw validation messages (they can quote plugin
    names, option values, or row content), so a bare ``unknown_node_type`` gave
    the planner no way to learn that forking is a gate, not a node_type — it
    re-emitted ``node_type='fork'`` until its budget exhausted. Each closed code
    now carries the static catalogue ``explanation``/``suggested_fix``, while the
    raw message stays withheld and ``error_code`` (the signal
    ``_feedback_error_codes`` reads) is preserved. A code with no catalogue entry
    stays bare.
    """
    summary = ValidationSummary(
        is_valid=False,
        errors=(
            ValidationEntry(
                component="node:fork_ab",
                message="unknown node_type 'fork' RAW_MESSAGE_CANARY quoting plugin/rows",
                severity="error",
                error_code="unknown_node_type",
            ),
            ValidationEntry(
                component="node:reconcile",
                message="RAW_MESSAGE_CANARY",
                severity="error",
                error_code="coalesce_on_success_must_be_sink",
            ),
            ValidationEntry(
                component="pipeline",
                message="RAW_MESSAGE_CANARY",
                severity="error",
                error_code=None,
            ),
        ),
    )

    feedback = _allowlisted_candidate_feedback(cast(Any, SimpleNamespace(validation=summary)))

    assert feedback["success"] is False
    assert feedback["validation"]["is_valid"] is False
    entries = feedback["validation"]["errors"]

    fork_entry = next(e for e in entries if e["error_code"] == "unknown_node_type")
    assert "no 'fork' node_type" in fork_entry["explanation"]
    assert "GATE" in fork_entry["suggested_fix"] and "fork_to" in fork_entry["suggested_fix"]

    coalesce_entry = next(e for e in entries if e["error_code"] == "coalesce_on_success_must_be_sink")
    assert "sink" in coalesce_entry["suggested_fix"].lower()

    # A code with no catalogue entry falls back to the bare structured shape.
    bare_entry = next(e for e in entries if e["error_code"] == "validation_error")
    assert set(bare_entry) == {"component", "severity", "error_code", "error_class"}

    # Raw messages must never ride the redaction-safe feedback, and error_class
    # / error_code stay intact for downstream consumers.
    assert "RAW_MESSAGE_CANARY" not in json.dumps(feedback)
    assert all(e["error_class"] == "ValidationError" for e in entries)

    # The feedback teaches the expansion move: live planners called
    # explain_validation_error with junk like {"error_text": "ValidationError"}
    # because nothing told them the exact code string is the key. One static
    # line, no topology hints (mid-repair suggestions have derailed runs).
    assert feedback["guidance"] == ("To expand any code, call explain_validation_error with the exact code string.")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["set_pipeline", "hallucinated_tool"])
async def test_mutation_or_unknown_tool_is_rejected_without_dispatch_or_retry(
    tmp_path: Path,
    tool_context: ToolContext,
    tool_name: str,
) -> None:
    completion = _ScriptedCompletion(_response((tool_name, {})))
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError, match="read-only discovery"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert recorder.invocations == ()
    assert len(completion.requests) == 1


@pytest.mark.asyncio
async def test_invalid_candidate_gets_allowlisted_feedback_then_repairs(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    raw_canary = "RAW_VALIDATION_EXCEPTION_CANARY"
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path, output_name=raw_canary)})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        repair_budget=1,
    )

    assert proposal.proposal.repair_count == 1
    feedback = completion.requests[1]["messages"][-1]
    assert feedback["role"] == "tool"
    # "guidance" is a static usage line (how to expand a code via
    # explain_validation_error) — never per-request data, so it does not
    # widen the redaction boundary this allowlist protects.
    assert set(json.loads(feedback["content"])) == {"success", "validation", "guidance"}
    feedback_payload = json.loads(feedback["content"])
    assert set(feedback_payload["validation"]) == {"is_valid", "errors"}
    # Closed codes may additionally carry the STATIC catalogue enrichment
    # (explanation/suggested_fix, always paired) and schema-contract entries
    # the structured "contract" facts (component ids + schema field names,
    # never row content) — nothing else may ride.
    for item in feedback_payload["validation"]["errors"]:
        assert {"component", "severity", "error_code", "error_class"} <= set(item), item
        assert set(item) <= {"component", "severity", "error_code", "error_class", "explanation", "suggested_fix", "contract"}, item
        assert ("explanation" in item) == ("suggested_fix" in item), item
    assert raw_canary not in feedback["content"]
    assert raw_canary not in canonical_json([call.to_dict() for call in recorder.llm_calls])


@pytest.mark.asyncio
async def test_safe_candidate_argument_error_gets_closed_feedback_then_repairs_without_custody(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context()
    raw_canary = "RAW_INVALID_FILENAME_CONTENT_CANARY"
    invalid = _inline_pipeline(tmp_path)
    invalid["source"]["inline_blob"]["filename"] = ""
    invalid["source"]["inline_blob"]["content"] = raw_canary
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": invalid})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        repair_budget=1,
        originating_message=origin,
        custody_config=PlannerCustodyConfig(
            data_dir=str(tmp_path),
            session_engine=engine,
            max_storage_per_session=1_000_000,
            secret_service=None,
            runtime_preflight=None,
        ),
    )

    assert proposal.proposal.repair_count == 1
    feedback = json.loads(completion.requests[1]["messages"][-1]["content"])
    assert set(feedback) == {"success", "validation"}
    assert set(feedback["validation"]) == {"is_valid", "errors"}
    assert feedback["validation"]["is_valid"] is False
    assert feedback["validation"]["errors"] == [
        {
            "component": "filename",
            "severity": "high",
            "error_code": "argument_error",
            "error_class": "ToolArgumentError",
        }
    ]
    assert raw_canary not in canonical_json(feedback)
    with engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0
        assert conn.execute(select(func.count()).select_from(composition_proposals_table)).scalar_one() == 0
    assert tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file()) == ()


@pytest.mark.asyncio
async def test_safe_candidate_argument_error_exhaustion_fails_without_custody(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context()
    invalid = _inline_pipeline(tmp_path)
    invalid["source"]["inline_blob"]["filename"] = ""
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": invalid})))

    with pytest.raises(PipelinePlannerError, match="repair budget exhausted"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            repair_budget=0,
            originating_message=origin,
            custody_config=PlannerCustodyConfig(
                data_dir=str(tmp_path),
                session_engine=engine,
                max_storage_per_session=1_000_000,
                secret_service=None,
                runtime_preflight=None,
            ),
        )

    with engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0
        assert conn.execute(select(func.count()).select_from(composition_proposals_table)).scalar_one() == 0
    assert tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file()) == ()


@pytest.mark.asyncio
async def test_exact_request_bytes_and_post_call_cost_caps_fail_closed(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    never_called = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
    with pytest.raises(PipelinePlannerError, match="request byte budget"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=never_called,
            budget=_budget(max_request_bytes=1),
        )
    assert never_called.requests == []

    costly = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)}), cost=0.11))
    recorder = BufferingRecorder()
    with pytest.raises(PipelinePlannerError, match="cost continuation cap"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=costly,
            recorder=recorder,
            budget=_budget(max_cumulative_provider_cost=Decimal("0.10")),
        )
    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].provider_cost == 0.11


@pytest.mark.asyncio
async def test_missing_provider_cost_is_audited_before_fail_closed(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)}), cost="bad"))
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError, match="provider cost metadata"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].provider_cost is None


def test_budget_policy_is_frozen_slotted_and_rejects_non_decimal_cost() -> None:
    policy = _budget()
    with pytest.raises(TypeError):
        vars(policy)
    with pytest.raises((TypeError, ValueError)):
        _budget(max_cumulative_provider_cost=1.0)


@pytest.mark.asyncio
async def test_pydantic_invalid_terminal_draft_gets_bounded_schema_repair(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    malformed = _pipeline(tmp_path)
    malformed["source"]["plugin"] = 123
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": malformed})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert proposal.proposal.repair_count == 1
    feedback = json.loads(completion.requests[1]["messages"][-1]["content"])
    assert feedback == {
        "success": False,
        "validation": {
            "is_valid": False,
            "errors": [
                {
                    "component": "pipeline",
                    "severity": "high",
                    "error_code": "canonical_schema",
                    "error_class": "SchemaValidationError",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_reported_completion_token_overage_is_audited_then_rejected(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response_with_usage(
            ("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)}),
            completion_tokens=801,
        )
    )
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError, match="completion token limit"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert recorder.llm_calls[0].completion_tokens == 801


@pytest.mark.asyncio
async def test_missing_completion_token_metadata_is_audited_then_rejected(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response_with_usage(
            ("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)}),
            completion_tokens=None,
        )
    )
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError) as caught:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert caught.value.code == "MALFORMED_RESPONSE"
    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].status is ComposerLLMCallStatus.MALFORMED_RESPONSE
    assert recorder.llm_calls[0].completion_tokens is None


@pytest.mark.parametrize(
    "raw_arguments",
    [
        pytest.param("[" * 2_000 + "0" + "]" * 2_000, id="depth"),
        pytest.param('{"value":"' + "x" * 1_048_577 + '"}', id="bytes"),
    ],
)
def test_planner_rejects_over_budget_tool_json_as_malformed_response(raw_arguments: str) -> None:
    response = _Response(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="deep",
                            function=_Function("list_sources", raw_arguments),
                        )
                    ],
                )
            )
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01},
    )

    with pytest.raises(PipelinePlannerError) as caught:
        _parse_response_tool_calls(response, max_tool_calls=3)

    assert caught.value.code == "MALFORMED_RESPONSE"


def test_planner_rejects_excessive_tool_call_container_before_argument_parsing() -> None:
    calls = [_ToolCall(id=f"call-{index}", function=_Function("list_sources", "[" * 2_000 + "0" + "]" * 2_000)) for index in range(4)]
    response = _Response(
        choices=[_Choice(message=_Message(content=None, tool_calls=calls))],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.01},
    )

    with pytest.raises(PipelinePlannerError, match="tool call") as caught:
        _parse_response_tool_calls(response, max_tool_calls=3)

    assert caught.value.code == "MALFORMED_RESPONSE"


@pytest.mark.asyncio
async def test_exhausted_provider_error_is_wrapped_class_only_after_audit(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    raw_canary = "RAW_PROVIDER_SDK_EXCEPTION_CANARY"
    failure = LiteLLMAPIError(
        status_code=503,
        message=raw_canary,
        llm_provider="test-provider",
        model="anthropic/claude-planner",
    )
    completion = _ScriptedCompletion(failure)
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError) as caught:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert raw_canary not in str(caught.value)
    assert raw_canary not in canonical_json([call.to_dict() for call in recorder.llm_calls])
    assert recorder.llm_calls[0].error_class == "APIError"


@pytest.mark.asyncio
async def test_discovery_crash_is_audited_from_preopened_envelope(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_canary = "RAW_DISCOVERY_CRASH_CANARY"

    def crash(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(raw_canary)

    monkeypatch.setattr("elspeth.web.composer.pipeline_planner.execute_discovery_tool_with_context", crash)
    recorder = BufferingRecorder()
    completion = _ScriptedCompletion(_response(("list_sources", {})))

    with pytest.raises(RuntimeError, match=raw_canary):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].status.value == "plugin_crash"
    assert recorder.invocations[0].error_message == "RuntimeError"
    assert raw_canary not in canonical_json(recorder.invocations[0].to_dict())


@pytest.mark.asyncio
async def test_discovery_state_change_is_classified_inside_audit_envelope(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    original = planner_module.execute_discovery_tool_with_context

    def return_changed_state(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        return replace(result, updated_state=replace(result.updated_state, version=result.updated_state.version + 1))

    monkeypatch.setattr(planner_module, "execute_discovery_tool_with_context", return_changed_state)
    recorder = BufferingRecorder()
    completion = _ScriptedCompletion(_response(("list_sources", {})))

    with pytest.raises(AuditIntegrityError, match="read-only planner discovery changed composition state"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].status.value == "plugin_crash"
    assert recorder.invocations[0].error_class == "AuditIntegrityError"
    assert recorder.invocations[0].error_message == "AuditIntegrityError"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_phase", ["policy", "candidate", "discovery"])
@pytest.mark.parametrize("termination", ["deadline", "cancellation"])
async def test_sync_planner_phases_run_off_loop_and_terminate_responsively(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
    blocked_phase: str,
    termination: str,
) -> None:
    import elspeth.web.composer.pipeline_planner as planner_module

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered = asyncio.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    worker_threads: list[int] = []

    def block_then_call(delegate: Any, *args: Any, **kwargs: Any) -> Any:
        worker_threads.append(threading.get_ident())
        loop.call_soon_threadsafe(entered.set)
        release.wait(timeout=3.0)
        try:
            return delegate(*args, **kwargs)
        finally:
            worker_finished.set()

    if blocked_phase == "policy":
        original_policy = PolicyCatalogView.validate_composition_state

        def blocking_policy(self: PolicyCatalogView, *args: Any, **kwargs: Any) -> Any:
            return block_then_call(original_policy, self, *args, **kwargs)

        monkeypatch.setattr(PolicyCatalogView, "validate_composition_state", blocking_policy)
        completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
    elif blocked_phase == "candidate":
        original_candidate = planner_module.build_set_pipeline_candidate

        def blocking_candidate(*args: Any, **kwargs: Any) -> Any:
            return block_then_call(original_candidate, *args, **kwargs)

        monkeypatch.setattr(planner_module, "build_set_pipeline_candidate", blocking_candidate)
        completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
    else:
        original_discovery = planner_module.execute_discovery_tool_with_context

        def blocking_discovery(*args: Any, **kwargs: Any) -> Any:
            return block_then_call(original_discovery, *args, **kwargs)

        monkeypatch.setattr(planner_module, "execute_discovery_tool_with_context", blocking_discovery)
        completion = _ScriptedCompletion(_response(("list_sources", {})))

    unobserved: list[Mapping[str, Any]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))
    plan_task = asyncio.create_task(
        _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            model_overrides={"timeout_seconds": 0.2 if termination == "deadline" else 5.0},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1.5)
        await asyncio.sleep(0)
        assert not worker_finished.is_set()
        if termination == "deadline":
            with pytest.raises(PipelinePlannerError, match="wall-clock"):
                await asyncio.wait_for(plan_task, timeout=2.0)
        else:
            plan_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(plan_task, timeout=2.0)
        assert not worker_finished.is_set()
    finally:
        release.set()
        if not plan_task.done():
            plan_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await plan_task
        for _attempt in range(200):
            if worker_finished.is_set():
                break
            await asyncio.sleep(0.01)
        loop.set_exception_handler(prior_handler)

    assert worker_finished.is_set()
    assert unobserved == []
    assert worker_threads
    assert all(thread_id != loop_thread for thread_id in worker_threads)


@pytest.mark.asyncio
async def test_each_transient_api_retry_consumes_and_audits_a_wire_attempt(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    raw_canary = "RAW_PROVIDER_ERROR_CANARY"
    transient = LiteLLMAPIError(
        status_code=503,
        message=raw_canary,
        llm_provider="test-provider",
        model="anthropic/claude-planner",
    )
    completion = _ScriptedCompletion(
        transient,
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        model_overrides={"max_api_attempts": 2},
    )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    assert [request["max_tokens"] for request in completion.requests] == [800, 800]
    assert [request["num_retries"] for request in completion.requests] == [0, 0]
    assert [request["max_retries"] for request in completion.requests] == [0, 0]
    assert [call.planner_call_ordinal for call in recorder.llm_calls] == [1, 2]
    assert [call.status.value for call in recorder.llm_calls] == ["api_error", "success"]
    assert raw_canary not in canonical_json([call.to_dict() for call in recorder.llm_calls])


@pytest.mark.asyncio
async def test_total_provider_call_bound_is_independent_of_logical_turns(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError, match="provider call budget"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            recorder=recorder,
            budget=_budget(max_total_provider_calls=1),
        )

    assert len(completion.requests) == 1
    assert [call.planner_call_ordinal for call in recorder.llm_calls] == [1]


@pytest.mark.asyncio
async def test_repeated_discovery_call_hits_explicit_cycle_guard_before_redispatch(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(_response(("list_sources", {})), _response(("list_sources", {})))
    recorder = BufferingRecorder()

    with pytest.raises(PipelinePlannerError, match="repetition/cycle guard"):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert len(recorder.llm_calls) == 2
    assert len(recorder.invocations) == 1


async def _session_context(*, content: str = "Use this CSV: name,score\nada,42\n") -> tuple[Any, PlannerOriginatingMessage]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.pipeline-planner-custody"),
    )
    session = await service.create_session("planner-user", "planner custody", "local")
    message = await service.add_message(
        session.id,
        "user",
        content,
        writer_principal="route_user_message",
    )
    return engine, PlannerOriginatingMessage(
        session_id=str(session.id),
        message_id=str(message.id),
        content=content,
        user_id="planner-user",
    )


@pytest.mark.asyncio
async def test_invalid_inline_draft_exhaustion_leaves_zero_pre_custody_residue(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context()
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path, output_name="not_rows")}))
    )

    state = _empty_state()
    before_state = deepcopy(state.to_dict())
    with pytest.raises(PipelinePlannerError, match="repair budget exhausted"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            repair_budget=0,
            originating_message=origin,
            custody_config=PlannerCustodyConfig(
                data_dir=str(tmp_path),
                session_engine=engine,
                max_storage_per_session=1_000_000,
                secret_service=None,
                runtime_preflight=None,
            ),
            current_state=state,
        )

    with engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0
        assert conn.execute(select(func.count()).select_from(composition_proposals_table)).scalar_one() == 0
    assert tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file()) == ()
    assert state.to_dict() == before_state


@pytest.mark.asyncio
async def test_cancellation_during_custody_settles_then_reraises_without_proposal(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, origin = await _session_context()
    entered = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    async def controlled_finalize(*_args: Any, **_kwargs: Any) -> object:
        entered.set()
        await release.wait()
        settled.set()
        return object()

    monkeypatch.setattr("elspeth.web.composer.pipeline_planner.finalize_pipeline_custody", controlled_finalize)
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path)})))
    lifecycle_events: list[str] = []
    task = asyncio.create_task(
        _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            originating_message=origin,
            custody_config=PlannerCustodyConfig(
                data_dir=str(tmp_path),
                session_engine=engine,
                max_storage_per_session=1_000_000,
                secret_service=None,
                runtime_preflight=None,
            ),
            lifecycle=_lifecycle(lifecycle_events),
        )
    )
    await entered.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert settled.is_set()
    assert lifecycle_events[-1] == "settled:cancelled"
    with engine.begin() as conn:
        assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0


@pytest.mark.asyncio
async def test_real_inline_custody_returns_only_blob_id_and_ready_row(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context()
    raw_content = "name,score\nada,42\n"
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path)})))
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        originating_message=origin,
        custody_config=PlannerCustodyConfig(
            data_dir=str(tmp_path),
            session_engine=engine,
            max_storage_per_session=1_000_000,
            secret_service=None,
            runtime_preflight=None,
        ),
    )

    public = proposal.proposal.to_dict()
    assert "inline_blob" not in canonical_json(public)
    assert raw_content not in canonical_json(public)
    blob_id = public["pipeline"]["source"]["blob_id"]
    with engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).mappings().one()
        assert conn.execute(select(func.count()).select_from(composition_proposals_table)).scalar_one() == 0
    assert row["status"] == "ready"
    assert Path(row["storage_path"]).read_text(encoding="utf-8") == raw_content
    assert raw_content not in canonical_json([call.to_dict() for call in recorder.llm_calls])
    assert raw_content not in canonical_json([invocation.to_dict() for invocation in recorder.invocations])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_returned", "expected_model_version"),
    [
        ("provider/resolved-planner-2026-07-18", "provider/resolved-planner-2026-07-18"),
        (None, "anthropic/claude-planner"),
    ],
    ids=["provider-returned-alias", "requested-model-fallback"],
)
async def test_inline_custody_provenance_uses_terminal_audited_model_returned_or_requested_fallback(
    tmp_path: Path,
    tool_context: ToolContext,
    model_returned: str | None,
    expected_model_version: str,
) -> None:
    engine, origin = await _session_context(content="Generate a fresh CSV for this pipeline.")
    response = _response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path)}))
    response.model = model_returned
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=_ScriptedCompletion(response),
        recorder=recorder,
        originating_message=origin,
        custody_config=PlannerCustodyConfig(
            data_dir=str(tmp_path),
            session_engine=engine,
            max_storage_per_session=1_000_000,
            secret_service=None,
            runtime_preflight=None,
        ),
    )

    blob_id = proposal.proposal.to_dict()["pipeline"]["source"]["blob_id"]
    with engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).mappings().one()

    assert recorder.llm_calls[-1].model_returned == model_returned
    assert row["creating_model_identifier"] == "anthropic/claude-planner"
    assert row["creating_model_version"] == expected_model_version


@pytest.mark.asyncio
async def test_route_lifecycle_rate_rejection_happens_before_provider_attempt(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))
    events: list[str] = []

    async def reject_rate() -> None:
        events.append("rate-rejected")
        raise PipelinePlannerError("planner route rate rejected (429)", code="RATE_LIMITED")

    @asynccontextmanager
    async def scope() -> AsyncIterator[None]:
        events.append("unexpected-scope")
        yield

    async def settled(outcome: str) -> None:
        events.append(f"settled:{outcome}")

    lifecycle = PlannerRequestLifecycle(
        before_start=reject_rate,
        request_scope=scope,
        on_settled=settled,
        progress=None,
    )

    with pytest.raises(PipelinePlannerError, match="429"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            lifecycle=lifecycle,
        )
    assert completion.requests == []
    assert events == ["rate-rejected", "settled:failed"]


@pytest.mark.asyncio
async def test_absolute_deadline_audits_slow_provider_timeout_and_settles(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    class SlowCompletion(_ScriptedCompletion):
        async def __call__(self, **kwargs: Any) -> _Response:
            self.requests.append(deepcopy(kwargs))
            await asyncio.sleep(60)
            raise AssertionError("unreachable")

    completion = SlowCompletion()
    recorder = BufferingRecorder()
    events: list[str] = []
    with pytest.raises(PipelinePlannerError, match="wall-clock"):
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            recorder=recorder,
            # The absolute deadline now includes the worker-offloaded initial
            # policy validation. Leave enough time to reach the deliberately
            # hanging provider so this test remains specifically about its
            # audited timeout path under xdist load.
            model_overrides={"timeout_seconds": 1.0},
            lifecycle=_lifecycle(events),
        )
    assert recorder.llm_calls[0].status.value == "timeout"
    assert events[-1] == "settled:failed"


@pytest.mark.asyncio
async def test_disconnect_cancellation_during_provider_call_audits_and_settles(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    entered = asyncio.Event()

    class HangingCompletion(_ScriptedCompletion):
        async def __call__(self, **kwargs: Any) -> _Response:
            self.requests.append(deepcopy(kwargs))
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    completion = HangingCompletion()
    recorder = BufferingRecorder()
    events: list[str] = []
    task = asyncio.create_task(
        _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            recorder=recorder,
            lifecycle=_lifecycle(events),
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert recorder.llm_calls[0].status.value == "cancelled"
    assert events[-1] == "settled:cancelled"


@pytest.mark.asyncio
async def test_settlement_failure_does_not_replace_primary_provider_failure(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    raw_provider_canary = "RAW_PRIMARY_PROVIDER_FAILURE_CANARY"
    provider_failure = LiteLLMAPIError(
        status_code=503,
        message=raw_provider_canary,
        llm_provider="test-provider",
        model="anthropic/claude-planner",
    )
    completion = _ScriptedCompletion(provider_failure)

    class SettlementFailure(RuntimeError):
        pass

    async def fail_settlement(_outcome: str) -> None:
        raise SettlementFailure("secondary settlement failure")

    lifecycle = replace(_lifecycle(), on_settled=fail_settlement)

    with pytest.raises(PipelinePlannerError) as caught:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            lifecycle=lifecycle,
        )

    assert caught.value.code == "PROVIDER_ERROR"
    assert raw_provider_canary not in str(caught.value)
    assert any("SettlementFailure" in note for note in getattr(caught.value, "__notes__", ()))


@pytest.mark.asyncio
async def test_secondary_cancellation_observes_failing_settlement_and_preserves_cancelled_error(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    provider_entered = asyncio.Event()
    settlement_entered = asyncio.Event()
    settlement_release = asyncio.Event()

    class HangingCompletion(_ScriptedCompletion):
        async def __call__(self, **kwargs: Any) -> _Response:
            self.requests.append(deepcopy(kwargs))
            provider_entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    class SettlementFailure(RuntimeError):
        pass

    async def fail_settlement(_outcome: str) -> None:
        settlement_entered.set()
        await settlement_release.wait()
        raise SettlementFailure("secondary settlement failure")

    loop = asyncio.get_running_loop()
    unobserved: list[Mapping[str, Any]] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))
    task = asyncio.create_task(
        _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=HangingCompletion(),
            lifecycle=replace(_lifecycle(), on_settled=fail_settlement),
        )
    )
    try:
        await provider_entered.wait()
        task.cancel()
        await settlement_entered.wait()
        task.cancel()
        settlement_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(prior_handler)

    assert unobserved == []


@pytest.mark.asyncio
async def test_settlement_failure_after_success_fails_the_request(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})))

    class SettlementFailure(RuntimeError):
        pass

    settlement_failure = SettlementFailure("required bookkeeping failed")

    async def fail_settlement(_outcome: str) -> None:
        raise settlement_failure

    with pytest.raises(SettlementFailure) as caught:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            lifecycle=replace(_lifecycle(), on_settled=fail_settlement),
        )

    assert caught.value is settlement_failure


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        _Response(choices=[], usage=_planner_usage()),
        _Response(choices=[_Choice(message=None)], usage=_planner_usage()),  # type: ignore[arg-type]
        # NOTE: the no-tool-call shape (content=None, tool_calls=None) left
        # this matrix when the loop gained the bounded prose nudge — see
        # test_prose_reply_gets_bounded_nudge_then_converges and
        # test_prose_replies_exhaust_nudge_budget_then_terminate_malformed.
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=None)]))],
            usage=_planner_usage(),
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("", "{}"))]))],
            usage=_planner_usage(),
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", 3))]))],
            usage=_planner_usage(),
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", "{"))]))],
            usage=_planner_usage(),
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", "[]"))]))],
            usage=_planner_usage(),
        ),
        _response(("emit_pipeline_proposal", {"pipeline": {}}), ("emit_pipeline_proposal", {"pipeline": {}})),
        _response(("emit_pipeline_proposal", {"pipeline": {}}), ("list_sources", {})),
    ],
    ids=[
        "choices",
        "message",
        "function",
        "name",
        "arguments-type",
        "arguments-json",
        "arguments-object",
        "multiple-terminal",
        "terminal-sibling",
    ],
)
async def test_malformed_provider_tool_call_matrix_fails_without_dispatch(
    tmp_path: Path,
    tool_context: ToolContext,
    response: _Response,
) -> None:
    completion = _ScriptedCompletion(response)
    recorder = BufferingRecorder()
    with pytest.raises(PipelinePlannerError):
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)
    assert len(recorder.llm_calls) == 1
    audit = recorder.llm_calls[0]
    sent = completion.requests[0]
    assert audit.status.value == "malformed_response"
    assert audit.messages_hash == stable_hash(sent["messages"])
    assert audit.tools_spec_hash == stable_hash(sent["tools"])
    assert audit.prompt_tokens == response.usage.get("prompt_tokens")
    assert audit.completion_tokens == response.usage.get("completion_tokens")
    assert audit.total_tokens == response.usage.get("total_tokens")
    assert audit.provider_cost == response.usage["cost"]
    assert audit.max_completion_tokens_requested == 800
    assert audit.planner_policy_hash == _budget().audit_hash
    assert audit.planner_call_ordinal == 1
    assert audit.error_class == "PipelinePlannerError"
    assert audit.error_message == "MALFORMED_RESPONSE"
    assert recorder.invocations == ()


@pytest.mark.asyncio
async def test_terminal_cannot_replace_server_base_and_proposal_is_created_once(
    tmp_path: Path,
    tool_context: ToolContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = {"pipeline": _pipeline(tmp_path), "base": {"kind": "present", "state_id": str(uuid4())}}
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", attempted)),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    calls = 0
    original = PipelineProposal.create.__func__

    def counted_create(cls: type[PipelineProposal], **kwargs: Any) -> PipelineProposal:
        nonlocal calls
        calls += 1
        return original(cls, **kwargs)

    monkeypatch.setattr(PipelineProposal, "create", classmethod(counted_create))
    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)
    assert proposal.proposal.base == AbsentBase()
    assert calls == 1


@pytest.mark.asyncio
async def test_blob_content_discovery_audit_projection_never_retains_content(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context()
    custody = PlannerCustodyConfig(
        data_dir=str(tmp_path),
        session_engine=engine,
        max_storage_per_session=1_000_000,
        secret_service=None,
        runtime_preflight=None,
    )
    first = _ScriptedCompletion(_response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path)})))
    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=first,
        originating_message=origin,
        custody_config=custody,
    )
    blob_id = proposal.proposal.to_dict()["pipeline"]["source"]["blob_id"]
    second = _ScriptedCompletion(
        _response(("get_blob_content", {"blob_id": blob_id})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()
    await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=second,
        recorder=recorder,
        originating_message=origin,
        custody_config=custody,
    )
    invocation = recorder.invocations[0]
    assert invocation.tool_name == "get_blob_content"
    assert "name,score" not in (invocation.result_canonical or "")
    assert set(json.loads(invocation.result_canonical or "{}")) == {"success", "validation", "version"}


def _text_response(text: str, *, cost: object = 0.01) -> _Response:
    return _Response(
        choices=[_Choice(message=_Message(content=text, tool_calls=None))],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost},
    )


def test_escape_hatch_model_must_be_none_or_nonempty() -> None:
    with pytest.raises(ValueError, match="escape_hatch_model"):
        _model(_ScriptedCompletion(), escape_hatch_model="   ")


@pytest.mark.parametrize(
    "overrides",
    [
        {"escape_hatch_model": "openrouter/advisor-under-test"},
        {"escape_hatch_provider": "openrouter"},
    ],
)
def test_escape_hatch_model_and_provider_must_be_configured_together(overrides: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="escape_hatch_model and escape_hatch_provider"):
        _model(_ScriptedCompletion(), **overrides)


@pytest.mark.asyncio
async def test_discovery_pressure_notice_injected_at_two_turns_remaining(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )

    await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    # max_discovery_turns defaults to 3: after the first discovery turn two
    # remain, which is exactly when the budget-pressure steering must land.
    first_turn = completion.requests[0]["messages"]
    second_turn = completion.requests[1]["messages"]
    assert not any("discovery turns remain" in str(message.get("content")) for message in first_turn)
    pressure = [
        message for message in second_turn if message["role"] == "user" and "only 2 discovery turns remain" in str(message.get("content"))
    ]
    assert len(pressure) == 1


@pytest.mark.asyncio
async def test_escape_hatch_overtime_turn_runs_advisor_with_terminal_tool_only(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        model_overrides={
            "max_discovery_turns": 1,
            "escape_hatch_model": "openrouter/advisor-under-test",
            "escape_hatch_provider": "openrouter",
        },
    )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 3
    hatch_request = completion.requests[2]
    assert hatch_request["model"] == "openrouter/advisor-under-test"
    assert [tool["function"]["name"] for tool in hatch_request["tools"]] == ["emit_pipeline_proposal"]
    notices = [
        message for message in hatch_request["messages"] if message["role"] == "user" and "escape hatch" in str(message.get("content"))
    ]
    assert len(notices) == 1
    # The over-budget discovery attempt is dropped: no dangling assistant
    # tool_calls message without its tool results.
    for message in hatch_request["messages"]:
        if message["role"] == "assistant" and message.get("tool_calls"):
            call_ids = {call["id"] for call in message["tool_calls"]}
            answered = {reply["tool_call_id"] for reply in hatch_request["messages"] if reply["role"] == "tool"}
            assert call_ids <= answered
    # Truthful audit attribution: the overtime call records the advisor model.
    assert [call.model_requested for call in recorder.llm_calls] == [
        "anthropic/claude-planner",
        "anthropic/claude-planner",
        "openrouter/advisor-under-test",
    ]
    assert proposal.model_identifier == "openrouter/advisor-under-test"
    assert proposal.provider == "openrouter"
    # The second discovery batch was never dispatched.
    assert [invocation.tool_name for invocation in recorder.invocations] == ["list_sources"]


@pytest.mark.asyncio
async def test_escape_hatch_provider_identity_reaches_inline_custody(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = await _session_context(content="Generate a fresh CSV after discovery.")
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _inline_pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        originating_message=origin,
        custody_config=PlannerCustodyConfig(
            data_dir=str(tmp_path),
            session_engine=engine,
            max_storage_per_session=1_000_000,
            secret_service=None,
            runtime_preflight=None,
        ),
        model_overrides={
            "max_discovery_turns": 1,
            "escape_hatch_model": "openrouter/advisor-under-test",
            "escape_hatch_provider": "openrouter",
        },
    )

    blob_id = proposal.proposal.to_dict()["pipeline"]["source"]["blob_id"]
    with engine.begin() as conn:
        row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id)).mappings().one()

    assert recorder.llm_calls[-1].model_requested == "openrouter/advisor-under-test"
    assert proposal.model_identifier == "openrouter/advisor-under-test"
    assert proposal.provider == "openrouter"
    assert row["creating_model_identifier"] == "openrouter/advisor-under-test"
    assert row["creating_provider"] == "openrouter"


@pytest.mark.asyncio
async def test_escape_hatch_text_reply_is_honest_decline(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
        _text_response("I cannot build this pipeline: the request needs a streaming join no available plugin provides."),
    )

    with pytest.raises(PlannerDeclined) as excinfo:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            model_overrides={
                "max_discovery_turns": 1,
                "escape_hatch_model": "openrouter/advisor-under-test",
                "escape_hatch_provider": "openrouter",
            },
        )

    assert excinfo.value.code == "DECLINED"
    assert "streaming join" in excinfo.value.decline_text
    assert isinstance(excinfo.value, PipelinePlannerError)


@pytest.mark.asyncio
async def test_escape_hatch_non_terminal_reply_reraises_original_exhaustion(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
        _response(("list_models", {})),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            model_overrides={
                "max_discovery_turns": 1,
                "escape_hatch_model": "openrouter/advisor-under-test",
                "escape_hatch_provider": "openrouter",
            },
        )

    assert excinfo.value.code == "DISCOVERY_EXHAUSTED"
    assert not isinstance(excinfo.value, PlannerDeclined)


@pytest.mark.asyncio
async def test_escape_hatch_fires_on_repair_exhaustion(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        repair_budget=1,
        model_overrides={
            "escape_hatch_model": "openrouter/advisor-under-test",
            "escape_hatch_provider": "openrouter",
        },
    )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert completion.requests[2]["model"] == "openrouter/advisor-under-test"
    assert [tool["function"]["name"] for tool in completion.requests[2]["tools"]] == ["emit_pipeline_proposal"]


@pytest.mark.asyncio
async def test_no_hatch_configured_preserves_discovery_exhaustion(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sinks", {})),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            model_overrides={"max_discovery_turns": 1},
        )

    assert excinfo.value.code == "DISCOVERY_EXHAUSTED"
    assert len(completion.requests) == 2


@pytest.mark.asyncio
async def test_escape_hatch_fires_on_discovery_cycle(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A cycling planner is stuck — the cycle guard engages the hatch, not a 502."""
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
        model_overrides={
            "escape_hatch_model": "openrouter/advisor-under-test",
            "escape_hatch_provider": "openrouter",
        },
    )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert completion.requests[2]["model"] == "openrouter/advisor-under-test"
    assert [tool["function"]["name"] for tool in completion.requests[2]["tools"]] == ["emit_pipeline_proposal"]
    # The repeated discovery batch is never dispatched.
    assert [invocation.tool_name for invocation in recorder.invocations] == ["list_sources"]


@pytest.mark.asyncio
async def test_discovery_cycle_without_hatch_still_raises(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("list_sources", {})),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert excinfo.value.code == "DISCOVERY_CYCLE"
    assert len(completion.requests) == 2


@pytest.mark.asyncio
async def test_discovery_reread_after_candidate_rejection_is_not_a_cycle(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """Re-reading discovery after a validation rejection is repair, not cycling.

    The capability core's discovery-order step 3 explicitly blesses
    ``get_plugin_schema`` (and ``explain_validation_error``) "when repairing
    against a validation rejection", but the repetition guard's window spanned
    the whole request: any re-read of a key seen before the rejection tripped
    DISCOVERY_CYCLE. Live guided sessions bad64533-08a1 and b3acc846-7b89
    (2026-07-22) died exactly this way — discovery, candidate, repair rounds,
    then a legitimate state re-read fired the guard and the opus hatch could
    not save them. The window is scoped per repair round: a candidate
    rejection resets it.
    """
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        # Same discovery key as turn 1 — legal now, a rejection intervened.
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(
        tmp_path=tmp_path,
        tool_context=tool_context,
        completion=completion,
        recorder=recorder,
    )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    # Both reads dispatched — the post-rejection re-read was served, not guarded.
    assert [inv.tool_name for inv in recorder.invocations if inv.tool_name == "list_sources"] == ["list_sources", "list_sources"]


@pytest.mark.asyncio
async def test_discovery_repetition_within_one_repair_round_still_trips(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """The per-round window still catches a genuinely stuck planner.

    After a rejection opens a fresh round, the first re-read is served but a
    second identical read in the SAME round is cycling by definition and must
    trip DISCOVERY_CYCLE exactly as before.
    """
    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("list_sources", {})),
        # Repeat within the same repair round — no rejection in between.
        _response(("list_sources", {})),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert excinfo.value.code == "DISCOVERY_CYCLE"
    assert len(completion.requests) == 4


def _truncated_response(*, completion_tokens: int, cost: object = 0.01) -> _Response:
    """A response cut off at the output token limit: no tool calls, partial text."""
    return _Response(
        choices=[_Choice(message=_Message(content='{"pipeline": {"source": {"plugin": "csv", "opti', tool_calls=None))],
        usage={"prompt_tokens": 10, "completion_tokens": completion_tokens, "total_tokens": 10 + completion_tokens, "cost": cost},
    )


@pytest.mark.asyncio
async def test_truncated_response_gets_compactness_repair(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A response cut off at the completion-token cap is repairable, not fatal."""
    completion = _ScriptedCompletion(
        _truncated_response(completion_tokens=800),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    notices = [
        message
        for message in completion.requests[1]["messages"]
        if message["role"] == "user" and "cut off at the output token limit" in str(message.get("content"))
    ]
    assert len(notices) == 1
    # The truncated call is audited with its own discriminant.
    assert recorder.llm_calls[0].error_message == "RESPONSE_TRUNCATED"


@pytest.mark.asyncio
async def test_truncated_responses_exhaust_repair_budget_without_hatch(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    completion = _ScriptedCompletion(
        _truncated_response(completion_tokens=800),
        _truncated_response(completion_tokens=800),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, repair_budget=1)

    assert excinfo.value.code == "REPAIR_EXHAUSTED"
    assert len(completion.requests) == 2


@pytest.mark.asyncio
async def test_prose_reply_gets_bounded_nudge_then_converges(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A prose reply mid-plan is nudged back to tool calling, not fatal.

    Tutorial session a2513c3c (2026-07-22): four clean discovery rounds, then
    the model thought aloud in prose — no tool call — and the loop died
    terminal MALFORMED_RESPONSE with zero repair consumed. A single prose
    reply is ordinary LLM behaviour; like truncation (4115fac13) it gets a
    bounded retry with a static notice before the terminal code.
    """
    completion = _ScriptedCompletion(
        _text_response("I think a csv source feeding one passthrough is right; let me lay that out."),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    notices = [
        message
        for message in completion.requests[1]["messages"]
        if message["role"] == "user" and "called no tool" in str(message.get("content"))
    ]
    assert len(notices) == 1
    # The prose reply is discarded from the conversation (parallel to the
    # truncation repair) and audited with its own discriminant.
    assert recorder.llm_calls[0].error_message == "PROSE_REPLY"


@pytest.mark.asyncio
async def test_prose_replies_exhaust_nudge_budget_then_terminate_malformed(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """The nudge budget is its own bound: prose past it is the old terminal.

    The nudge budget is separate from the repair budget (repair_budget=1
    here would otherwise die one round earlier) and the exhausted case keeps
    the existing terminal MALFORMED_RESPONSE disposition.
    """
    completion = _ScriptedCompletion(
        _text_response("Thinking aloud, round one."),
        _text_response("Thinking aloud, round two."),
        _text_response("Thinking aloud, round three."),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, repair_budget=1)

    assert excinfo.value.code == "MALFORMED_RESPONSE"
    assert len(completion.requests) == 3


@pytest.mark.asyncio
async def test_malformed_tool_call_arguments_stay_fatal(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """Genuinely malformed output (non-prose) keeps the immediate terminal.

    The nudge covers only the no-tool-call class; a tool call whose
    arguments are not strict JSON is provider/model breakage, not thinking
    aloud, and dies MALFORMED_RESPONSE on the spot exactly as before.
    """
    bad_call = _ToolCall(id="call-1", function=_Function(name="list_sources", arguments="{not json"))
    completion = _ScriptedCompletion(
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[bad_call]))],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
        )
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion)

    assert excinfo.value.code == "MALFORMED_RESPONSE"
    assert len(completion.requests) == 1


@pytest.mark.asyncio
async def test_discovery_argument_error_is_recoverable_not_fatal(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A discovery tool call with a bad enum arg (the model guessing
    plugin_type='node') must feed back a failure tool result the planner can
    repair from, NOT crash the whole request as a non-planner 500 (live:
    session bf109c43 — get_plugin_schema plugin_type='node' → ToolArgumentError
    → HTTP 500, no disposition)."""
    completion = _ScriptedCompletion(
        _response(("get_plugin_schema", {"plugin_type": "node", "name": "coalesce"})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    # The bad-arg call fed back a failure tool message the model saw next turn.
    tool_messages = [m for m in completion.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["success"] is False
    # The invocation is still audited as an argument error.
    assert recorder.invocations[0].status.value == "arg_error"


@pytest.mark.asyncio
async def test_discovery_argument_error_alongside_valid_call_both_feed_back(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A bad-enum call in a parallel batch must not abort its siblings."""
    completion = _ScriptedCompletion(
        _response(("get_plugin_schema", {"plugin_type": "node", "name": "coalesce"}), ("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    recorder = BufferingRecorder()

    proposal = await _plan(tmp_path=tmp_path, tool_context=tool_context, completion=completion, recorder=recorder)

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    tool_messages = [m for m in completion.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert {inv.tool_name for inv in recorder.invocations} == {"get_plugin_schema", "list_sources"}


@pytest.mark.asyncio
async def test_repair_exhaustion_records_last_rejection_codes(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """A candidate rejected to exhaustion must surface WHY on the error, so a
    live 502 disposition names the blocking validation codes instead of
    needing a temp DIAG (permanent forensics — the-DB-is-the-log)."""
    # _invalid_pipeline mints a sink_name mismatch → a closed validation code.
    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
    )

    with pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            repair_budget=1,
            model_overrides={"escape_hatch_model": None},
        )

    assert excinfo.value.code == "REPAIR_EXHAUSTED"
    assert excinfo.value.detail_codes, "exhaustion must carry the last rejection's codes"
    # Codes are the closed, leak-safe discriminant — no messages/paths.
    assert all(isinstance(c, str) for c in excinfo.value.detail_codes)


@pytest.mark.asyncio
async def test_planner_attempt_trail_names_reject_repair_accept(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """Every attempt emits a trail event; success emits a summary too.

    The terminal disposition only carries the LAST failure's codes, so a run
    whose final attempt failed at a non-candidate layer reported
    rejection_codes=[] and the whole repair history was invisible (guided
    session 5a5082e6, op 408acf3a). The per-attempt trail names each round —
    and the success-path summary closes the churn-observability gap
    (assessment item 5) at the same time.
    """
    from structlog.testing import capture_logs

    completion = _ScriptedCompletion(
        _response(("list_sources", {})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _pipeline(tmp_path)})),
    )
    origin = _origin()

    with capture_logs() as logs:
        proposal = await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            originating_message=origin,
        )

    assert deep_thaw(proposal.proposal.pipeline) == _pipeline(tmp_path)
    attempts = [entry for entry in logs if entry["event"] == "composer.planner_attempt"]
    assert [entry["attempt"] for entry in attempts] == [1, 2, 3]
    assert all(entry["session_id"] == origin.session_id for entry in attempts)
    assert all(entry["surface"] == "freeform" for entry in attempts)

    discovery, rejected, accepted = attempts
    assert (discovery["phase"], discovery["outcome"]) == ("discovery", "discovery_executed")
    assert discovery["tool_calls"] == 1
    assert (rejected["phase"], rejected["outcome"]) == ("candidate", "candidate_rejected")
    assert rejected["rejection_codes"], "the rejected attempt must name its codes"
    assert rejected["led_to"] == "repair"
    assert (accepted["phase"], accepted["outcome"]) == ("repair", "accepted")
    assert accepted["led_to"] == "done"

    summaries = [entry for entry in logs if entry["event"] == "composer.planner_summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["final_outcome"] == "accepted"
    assert summary["total_attempts"] == 3
    assert summary["phase_counts"] == {"discovery": 1, "candidate": 1, "repair": 1}
    assert summary["rejection_history"] == [
        {"attempt": 2, "outcome": "candidate_rejected", "codes": rejected["rejection_codes"]},
    ]
    assert summary["session_id"] == origin.session_id


@pytest.mark.asyncio
async def test_planner_summary_on_exhaustion_carries_the_full_code_history(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    """REPAIR_EXHAUSTED with a shape-level last failure is no longer blind:
    the summary names every earlier round's rejection codes."""
    from structlog.testing import capture_logs

    completion = _ScriptedCompletion(
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
        _response(("emit_pipeline_proposal", {"pipeline": _invalid_pipeline(tmp_path)})),
    )

    with capture_logs() as logs, pytest.raises(PipelinePlannerError) as excinfo:
        await _plan(
            tmp_path=tmp_path,
            tool_context=tool_context,
            completion=completion,
            repair_budget=1,
            model_overrides={"escape_hatch_model": None},
        )

    assert excinfo.value.code == "REPAIR_EXHAUSTED"
    attempts = [entry for entry in logs if entry["event"] == "composer.planner_attempt"]
    assert [entry["outcome"] for entry in attempts] == ["candidate_rejected", "candidate_rejected"]
    assert attempts[0]["led_to"] == "repair"
    assert attempts[1]["led_to"] == "terminal"
    assert attempts[1]["planner_code"] == "REPAIR_EXHAUSTED"

    summaries = [entry for entry in logs if entry["event"] == "composer.planner_summary"]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["final_outcome"] == "REPAIR_EXHAUSTED"
    assert summary["total_attempts"] == 2
    # BOTH rounds' codes survive — the blindspot this trail exists to close.
    assert [entry["attempt"] for entry in summary["rejection_history"]] == [1, 2]
    assert all(entry["codes"] for entry in summary["rejection_history"])
