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
from typing import Any
from uuid import uuid4

import pytest
from litellm.exceptions import APIError as LiteLLMAPIError
from sqlalchemy import func, insert, select
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.pipeline_planner import (
    PLANNER_DISCOVERY_TOOL_NAMES,
    PipelinePlannerError,
    PlannerBudgetPolicy,
    PlannerCustodyConfig,
    PlannerModelConfig,
    PlannerOriginatingMessage,
    PlannerRequestLifecycle,
    plan_pipeline,
    planner_tool_definitions,
)
from elspeth.web.composer.pipeline_proposal import AbsentBase, PipelineProposal, PlannerSurface
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, composition_proposals_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


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


def _response(*calls: tuple[str, object], cost: object = 0.01) -> _Response:
    tool_calls = [
        _ToolCall(id=f"call-{index}", function=_Function(name=name, arguments=json.dumps(arguments)))
        for index, (name, arguments) in enumerate(calls, start=1)
    ]
    return _Response(
        choices=[_Choice(message=_Message(content=None, tool_calls=tool_calls))],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": cost},
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
        intent="Build the requested pipeline.",
        current_state=current_state or _empty_state(),
        reviewed_facts={"request": "Build the requested pipeline."},
        surface=PlannerSurface.FREEFORM,
        policy_catalog=policy_catalog,
        plugin_snapshot=plugin_snapshot,
        originating_message=originating_message or _origin(),
        base=AbsentBase(),
        model_config=_model(completion, **dict(model_overrides or {})),
        rendered_skill="You are the bounded ELSPETH pipeline planner.",
        repair_budget=repair_budget,
        budget_policy=budget or _budget(),
        custody_config=custody_config or _custody(tmp_path),
        lifecycle=lifecycle or _lifecycle(),
        recorder=recorder or BufferingRecorder(),
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
        "properties": {"pipeline": canonical_set_pipeline_schema()},
        "required": ["pipeline"],
        "additionalProperties": False,
    }
    serialized = canonical_json(terminal)
    assert "rationale" not in serialized
    assert '"base"' not in serialized


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

    assert deep_thaw(proposal.pipeline) == pipeline
    assert proposal.base == AbsentBase()
    assert proposal.repair_count == 0
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

    assert deep_thaw(proposal.pipeline) == _pipeline(tmp_path)
    assert len(completion.requests) == 2
    assert completion.requests[1]["messages"][-1]["role"] == "tool"
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].tool_name == "list_sources"
    assert [call.planner_call_ordinal for call in recorder.llm_calls] == [1, 2]


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

    assert proposal.repair_count == 1
    feedback = completion.requests[1]["messages"][-1]
    assert feedback["role"] == "tool"
    assert set(json.loads(feedback["content"])) == {"success", "validation"}
    feedback_payload = json.loads(feedback["content"])
    assert set(feedback_payload["validation"]) == {"is_valid", "errors"}
    assert all(set(item) <= {"component", "severity", "error_code", "error_class"} for item in feedback_payload["validation"]["errors"])
    assert raw_canary not in feedback["content"]
    assert raw_canary not in canonical_json([call.to_dict() for call in recorder.llm_calls])


@pytest.mark.asyncio
async def test_safe_candidate_argument_error_gets_closed_feedback_then_repairs_without_custody(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = _session_context()
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

    assert proposal.repair_count == 1
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
    engine, origin = _session_context()
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

    assert proposal.repair_count == 1
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

    assert deep_thaw(proposal.pipeline) == _pipeline(tmp_path)
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


def _session_context(*, content: str = "Use this CSV: name,score\nada,42\n") -> tuple[Any, PlannerOriginatingMessage]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    message_id = str(uuid4())
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="planner-user",
                auth_provider_type="local",
                title="planner custody",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=message_id,
                session_id=session_id,
                role="user",
                content=content,
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return engine, PlannerOriginatingMessage(
        session_id=session_id,
        message_id=message_id,
        content=content,
        user_id="planner-user",
    )


@pytest.mark.asyncio
async def test_invalid_inline_draft_exhaustion_leaves_zero_pre_custody_residue(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = _session_context()
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
    engine, origin = _session_context()
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
    engine, origin = _session_context()
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

    public = proposal.to_dict()
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
    engine, origin = _session_context(content="Generate a fresh CSV for this pipeline.")
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

    blob_id = proposal.to_dict()["pipeline"]["source"]["blob_id"]
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
        _Response(choices=[], usage={"cost": 0.01}),
        _Response(choices=[_Choice(message=None)], usage={"cost": 0.01}),  # type: ignore[arg-type]
        _Response(choices=[_Choice(message=_Message(content=None, tool_calls=None))], usage={"cost": 0.01}),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=None)]))],
            usage={"cost": 0.01},
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("", "{}"))]))],
            usage={"cost": 0.01},
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", 3))]))],
            usage={"cost": 0.01},
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", "{"))]))],
            usage={"cost": 0.01},
        ),
        _Response(
            choices=[_Choice(message=_Message(content=None, tool_calls=[_ToolCall(id="x", function=_Function("list_sources", "[]"))]))],
            usage={"cost": 0.01},
        ),
        _response(("emit_pipeline_proposal", {"pipeline": {}}), ("emit_pipeline_proposal", {"pipeline": {}})),
        _response(("emit_pipeline_proposal", {"pipeline": {}}), ("list_sources", {})),
    ],
    ids=[
        "choices",
        "message",
        "calls",
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
    assert proposal.base == AbsentBase()
    assert calls == 1


@pytest.mark.asyncio
async def test_blob_content_discovery_audit_projection_never_retains_content(
    tmp_path: Path,
    tool_context: ToolContext,
) -> None:
    engine, origin = _session_context()
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
    blob_id = proposal.to_dict()["pipeline"]["source"]["blob_id"]
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
