"""Tests for the request_advisor_hint MCP tool — the frontier-model
escape hatch for the composer LLM.

Covers:
- Advisor is mandatory: the tool is always exposed by _get_litellm_tools().
- CLI MCP allowlist (_COMPOSER_TOOL_NAMES) excludes it by design — the
  advisor is web-composer-only because the CLI MCP server's allowlist
  is built from _DISCOVERY_TOOLS / _MUTATION_TOOLS, neither of which
  the advisor enters.
- Compose-loop happy path: advisor call returns guidance; outer
  ComposerToolInvocation captures the prompt and reply via canonical
  hashing; inner ComposerLLMCall records model + tokens + latency.
- Budget exhaustion returns BUDGET_EXHAUSTED status, not a crash, and
  the §7.7 anti-anchor tracker is NOT counted (budget exhaustion is
  policy, not LLM repetition).
- Advisor LLM failure: structured ADVISOR_ERROR tool result, anti-anchor
  counts the failure, inner ComposerLLMCall records the failure status.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from elspeth.contracts.composer_audit import ComposerToolInvocation
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.prompts import SYSTEM_PROMPT
from elspeth.web.composer.protocol import ComposerConvergenceError
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import get_tool_definitions
from elspeth.web.config import WebSettings
from tests.unit.web.composer._helpers import _stub_advisor_end_gate_clean  # noqa: F401  (autouse end-gate CLEAN stub)

# --- Test scaffolding (mirrors test_compose_loop_anti_anchor.py) ---


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[_FakeToolCall] | None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]
    model: str = "anthropic/claude-sonnet-4-6"
    usage: _FakeUsage | None = None
    id: str | None = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = _FakeUsage()


def _result_canonical(invocation: ComposerToolInvocation) -> str:
    assert invocation.result_canonical is not None
    return invocation.result_canonical


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _mock_catalog() -> MagicMock:
    catalog = MagicMock(spec=CatalogService)
    catalog.list_sources.return_value = [
        PluginSummary(name="csv", description="CSV", plugin_type="source", config_fields=[]),
    ]
    catalog.list_transforms.return_value = []
    catalog.list_sinks.return_value = []
    catalog.get_schema.return_value = PluginSchemaInfo(
        name="csv",
        plugin_type="source",
        description="CSV source",
        json_schema={"type": "object", "properties": {}},
        knob_schema={"fields": []},
    )
    return catalog


def _make_settings(
    *,
    budget: int = 4,
    composer_timeout_seconds: float = 85.0,
    advisor_timeout_seconds: float = 60.0,
    data_dir: Path | str = Path("/data"),
) -> WebSettings:
    return WebSettings(
        data_dir=Path(data_dir),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=composer_timeout_seconds,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=budget,
        composer_advisor_timeout_seconds=advisor_timeout_seconds,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _make_advisor_tool_call(
    call_id: str,
    *,
    problem: str = "stuck on llm config",
    trigger: str = "proactive_security_safety",
) -> _FakeLLMResponse:
    args = {
        "trigger": trigger,
        "problem_summary": problem,
        "recent_errors": ["error A", "error A"],
        "attempted_actions": ["set_pipeline with options={}", "checked relevant schema"],
    }
    return _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id=call_id,
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(args),
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _make_text_only_response(content: str) -> _FakeLLMResponse:
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=None))])


def _make_advisor_response(text: str = "Try setting `provider: azure` and supplying `deployment`.") -> _FakeLLMResponse:
    return _FakeLLMResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=text, tool_calls=None))],
        model="anthropic/claude-sonnet-4-6",
        usage=_FakeUsage(prompt_tokens=120, completion_tokens=45, total_tokens=165),
    )


@pytest.fixture(autouse=True)
def _composer_available_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    def _available(self: ComposerServiceImpl) -> ComposerAvailability:
        return ComposerAvailability(available=True, model=self._model, provider="test")

    monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


# --- 1. Advisor is mandatory — the tool is always exposed ---


def test_advisor_tool_exposed() -> None:
    """Advisor is mandatory, so _get_litellm_tools() always includes
    request_advisor_hint in the LiteLLM function format.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    tools = service._get_litellm_tools()
    names = {t["function"]["name"] for t in tools}
    assert "request_advisor_hint" in names
    advisor = next(t for t in tools if t["function"]["name"] == "request_advisor_hint")
    assert advisor["function"]["description"], "advisor tool needs a description"
    assert "ADVICE" in advisor["function"]["description"], "description must steer the LLM that the reply is advice not config"


def test_advisor_tool_schema_requires_trigger_and_mentions_proactive_criteria() -> None:
    """The tool contract must make the call criteria mechanical enough for
    the LLM to declare why it is escalating.

    The skill allows proactive calls for safety/security and red-listed
    plugins; the tool description must not contradict that by describing
    only the reactive validation-loop path.
    """
    advisor = next(defn for defn in get_tool_definitions() if defn["name"] == "request_advisor_hint")
    parameters = advisor["parameters"]
    assert "trigger" in parameters["required"]
    assert parameters["additionalProperties"] is False

    trigger_schema = parameters["properties"]["trigger"]
    assert trigger_schema["enum"] == [
        "proactive_security_safety",
        "proactive_red_listed_plugin",
    ]

    description = advisor["description"]
    assert "security" in description
    assert "red-listed" in description
    assert "before `set_pipeline`" in description


def test_advisor_argument_validation_rejects_unknown_keys() -> None:
    """Advisor args are LLM-supplied; the runtime boundary must reject extras."""
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    raw_extra_context = "RAW_EXTRA_CONTEXT: raw traceback and source excerpt"

    payload = service._validate_advisor_arguments(
        {
            "trigger": "proactive_security_safety",
            "problem_summary": "stuck",
            "recent_errors": ["e1"],
            "attempted_actions": ["a1"],
            "full_context": raw_extra_context,
        }
    )

    assert payload is not None
    assert payload["status"] == "ARG_ERROR"
    assert payload["error_class"] == "ValueError"
    assert "unknown argument" in payload["error"]
    assert "full_context" not in payload["error"]
    assert raw_extra_context not in payload["error"]


def test_reactive_trigger_is_retired() -> None:
    """The reactive validation-loop trigger is no longer an LLM-supplied
    trigger; ``request_advisor_hint`` with ``trigger="reactive_validation_loop"``
    must be rejected as an unknown trigger.
    """
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    payload = service._validate_advisor_arguments(
        {
            "trigger": "reactive_validation_loop",
            "problem_summary": "stuck",
            "recent_errors": ["e1", "e2"],
            "attempted_actions": ["a1", "a2"],
        }
    )
    assert payload is not None  # ARG_ERROR
    assert payload["status"] == "ARG_ERROR"
    assert "must be one of" in payload["error"]
    assert "reactive_validation_loop" not in payload["error"]


def test_proactive_triggers_still_valid() -> None:
    """The two proactive triggers must still validate cleanly."""
    service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
    for trig in ("proactive_security_safety", "proactive_red_listed_plugin"):
        payload = service._validate_advisor_arguments(
            {
                "trigger": trig,
                "problem_summary": "p",
                "recent_errors": [],
                "attempted_actions": [],
            }
        )
        assert payload is None  # valid


# --- 2. CLI MCP allowlist excludes the advisor by design ---


def test_advisor_tool_excluded_from_cli_mcp_allowlist() -> None:
    """The CLI MCP server's _COMPOSER_TOOL_NAMES allowlist is built from
    _DISCOVERY_TOOLS and _MUTATION_TOOLS dispatch tables. The advisor is
    intercepted in the compose loop, NOT registered in either dispatch
    table, so it is automatically excluded from the CLI MCP — this is
    architecturally enforced "web-composer only" with no extra logic.
    """
    from elspeth.composer_mcp.server import _COMPOSER_TOOL_NAMES

    assert "request_advisor_hint" not in _COMPOSER_TOOL_NAMES, "advisor leaked into CLI MCP allowlist — should be web-only"

    # Sanity: it IS in the runtime tool definitions (from which the
    # allowlist filters), so the exclusion is via not-in-dispatch-table,
    # not by a separate exemption rule.
    runtime_names = {d["name"] for d in get_tool_definitions()}
    assert "request_advisor_hint" in runtime_names


# --- 3. Compose-loop happy path (advisor returns guidance) ---


@pytest.mark.asyncio
async def test_advisor_call_records_outer_invocation_and_inner_llm_call() -> None:
    """End-to-end happy path: the LLM calls request_advisor_hint, the
    interception forwards to the advisor model, and the audit captures
    BOTH records:
      - outer ComposerToolInvocation: arguments_canonical (the prompt
        forwarded to advisor) and result_canonical (the guidance + meta).
      - inner ComposerLLMCall: model_requested, tokens, latency, status.

    No new audit dataclass needed — the existing primitives cover the
    whole shape because the inner call lands on recorder.record_llm_call
    via build_llm_call_record fired from _call_advisor_with_audit's
    finally block.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()

    turns = [
        _make_advisor_tool_call("call_advisor_1"),
        _make_text_only_response("OK, I see — I'll apply the suggested change."),
    ]

    advisor_response = _make_advisor_response()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        mock_acompletion.return_value = advisor_response
        result = await service.compose("help me", [], state)

    # _litellm_acompletion was called once (the advisor call). _call_llm
    # is the primary-composer path (mocked separately above).
    assert mock_acompletion.call_count == 1
    advisor_call_kwargs = mock_acompletion.call_args.kwargs
    assert advisor_call_kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert advisor_call_kwargs["max_tokens"] == 1500  # default completion cap

    # Outer ComposerToolInvocation should be present in the recorder.
    invocations = [inv for inv in result.tool_invocations if inv.tool_name == "request_advisor_hint"]
    assert len(invocations) == 1, "outer invocation record missing"
    inv = invocations[0]
    assert inv.status.name == "SUCCESS"
    # arguments_canonical contains the LLM's stuck-message
    assert "stuck on llm config" in inv.arguments_canonical
    # result_canonical contains the advisor's guidance + metadata
    result_canonical = _result_canonical(inv)
    assert "Try setting" in result_canonical
    assert "anthropic/claude-sonnet-4-6" in result_canonical
    assert "budget_remaining" in result_canonical

    # Inner ComposerLLMCall record is in result.llm_calls
    advisor_llm_calls = [call for call in result.llm_calls if call.model_requested == "anthropic/claude-sonnet-4-6"]
    assert len(advisor_llm_calls) == 1, "inner advisor LLM call audit record missing"
    llm_call = advisor_llm_calls[0]
    assert llm_call.status.name == "SUCCESS"
    assert llm_call.prompt_tokens == 120
    assert llm_call.completion_tokens == 45


@pytest.mark.asyncio
async def test_advisor_only_turn_does_not_consume_discovery_budget() -> None:
    """Advisor-only turns have their own budget and must not spend discovery turns.

    With composer_max_discovery_turns=1, the old classifier charged the
    successful advisor call as discovery and raised before the composer LLM
    could read the guidance on the next turn.
    """
    catalog = _mock_catalog()
    settings = WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=1,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=3,
        composer_advisor_timeout_seconds=60.0,
        shareable_link_signing_key=b"\x00" * 32,
    )
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_make_advisor_response(),
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = [
            _make_advisor_tool_call("call_advisor_1"),
            _make_text_only_response("I read the guidance and can continue."),
        ]
        result = await service.compose("help me", [], state)

    assert result.message == "I read the guidance and can continue."
    assert mock_llm.call_count == 2
    assert mock_acompletion.call_count == 1
    invocations = [inv for inv in result.tool_invocations if inv.tool_name == "request_advisor_hint"]
    assert len(invocations) == 1


@pytest.mark.asyncio
async def test_advisor_call_includes_core_and_deployment_skill_context(tmp_path: Path) -> None:
    """The advisor LLM engagement must carry the same composer skill stack
    as normal composer/diagnostics LLM calls, including deployment overlays.
    """
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "pipeline_composer.md").write_text(
        "# Deployment rules\n\nUse DEPLOYMENT_PROVIDER_MAPPING_SENTINEL.\n",
        encoding="utf-8",
    )
    catalog = _mock_catalog()
    service = ComposerServiceImpl(
        catalog=catalog,
        settings=_make_settings(data_dir=tmp_path),
    )
    recorder = BufferingRecorder()
    args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck",
        "recent_errors": ["validator rejected provider"],
        "attempted_actions": ["set_pipeline once"],
    }

    with patch(
        "elspeth.web.composer.service._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_make_advisor_response(),
    ) as mock_acompletion:
        await service._call_advisor_with_audit(args, recorder=recorder)

    messages = mock_acompletion.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    assert SYSTEM_PROMPT in system_content
    assert "DEPLOYMENT_PROVIDER_MAPPING_SENTINEL" in system_content
    assert "You are advising another LLM" in system_content


@pytest.mark.asyncio
async def test_advisor_omits_seed_when_advisor_model_does_not_support_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The advisor model has its own provider surface; direct Anthropic
    defaults must share the same seed-support gate as the primary composer.
    """
    import litellm

    monkeypatch.setattr(
        litellm,
        "get_supported_openai_params",
        lambda model: ["temperature", "max_tokens"],
    )
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    recorder = BufferingRecorder()
    args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck",
        "recent_errors": ["validator rejected provider", "validator rejected provider"],
        "attempted_actions": ["set_pipeline once", "checked schema"],
    }

    with patch(
        "elspeth.web.composer.service._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_make_advisor_response(),
    ) as mock_acompletion:
        await service._call_advisor_with_audit(args, recorder=recorder)

    kwargs = mock_acompletion.call_args.kwargs
    assert "seed" not in kwargs
    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].seed is None


# --- 4. Budget exhaustion ---


@pytest.mark.asyncio
async def test_budget_exhaustion_returns_structured_error() -> None:
    """The 4th advisor call after a budget of 3 must return a
    BUDGET_EXHAUSTED structured result without making any LiteLLM call,
    and the audit envelope must still close cleanly (SUCCESS dispatch,
    BUDGET_EXHAUSTED in result_payload).
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()

    # Drive 4 advisor calls: 1, 2, 3 succeed; 4 hits budget.
    turns = [
        _make_advisor_tool_call("call_1"),
        _make_advisor_tool_call("call_2"),
        _make_advisor_tool_call("call_3"),
        _make_advisor_tool_call("call_4"),
        _make_text_only_response("done"),
    ]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        mock_acompletion.return_value = _make_advisor_response()
        result = await service.compose("help me", [], state)

    # Only 3 outbound advisor calls — 4th is short-circuited by budget.
    assert mock_acompletion.call_count == 3, f"expected 3 outbound advisor calls (budget=3), got {mock_acompletion.call_count}"

    # 4 outer ComposerToolInvocation records (all dispatch envelopes close).
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 4

    # The 4th invocation's result_canonical must encode BUDGET_EXHAUSTED.
    exhausted_result = _result_canonical(invs[3])
    assert "BUDGET_EXHAUSTED" in exhausted_result
    assert "budget_remaining" in exhausted_result

    # The 1st-3rd invocations must encode SUCCESS status in result_payload.
    for inv in invs[:3]:
        result_canonical = _result_canonical(inv)
        assert "SUCCESS" in result_canonical
        assert "Try setting" in result_canonical


@pytest.mark.asyncio
async def test_exhausted_advisor_turn_charges_discovery_budget() -> None:
    """BUDGET_EXHAUSTED advisor turns must still be bounded by loop budget.

    Successful advisor guidance is budgeted separately so the primary model
    can read it. A policy refusal is different: no guidance was produced, and
    repeated calls must converge via the generic discovery-turn guard rather
    than spinning until the wall-clock compose timeout.
    """
    catalog = _mock_catalog()
    settings = WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=1,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=0,
        composer_advisor_timeout_seconds=60.0,
        shareable_link_signing_key=b"\x00" * 32,
    )
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
        pytest.raises(ComposerConvergenceError) as exc_info,
    ):
        mock_llm.side_effect = [
            _make_advisor_tool_call("call_budget_exhausted"),
            _make_text_only_response("would incorrectly continue"),
        ]
        await service.compose("help me", [], state)

    assert exc_info.value.budget_exhausted == "discovery"
    assert mock_llm.call_count == 1
    assert mock_acompletion.call_count == 0
    invocations = [inv for inv in exc_info.value.tool_invocations if inv.tool_name == "request_advisor_hint"]
    assert len(invocations) == 1
    assert "BUDGET_EXHAUSTED" in _result_canonical(invocations[0])


# --- 5. Disabled-but-LLM-tries (defense-in-depth) ---


@pytest.mark.asyncio
async def test_advisor_call_failure_records_inner_status_and_outer_error() -> None:
    """When the advisor LiteLLM call raises (timeout, API error, malformed
    response), the interception must:
      - Convert the failure into a structured ADVISOR_ERROR tool result so
        the composer LLM can see the failure mode rather than silently stall.
      - Fire the inner ComposerLLMCall audit record with a non-SUCCESS status
        (the finally block in _call_advisor_with_audit handles this).
      - Increment the §7.7 anti-anchor failure tracker so repeated identical
        failed advisor calls trigger the structural hint.

    This test underwrites the safety claim in the tier-model allowlist
    entry for `web/composer/service.py:R4:ComposerServiceImpl:_compose_loop`
    (the broad-except in the advisor interception). Without this test, that
    entry's safety: field is an unverified claim.
    """
    from litellm.exceptions import APIError as LiteLLMAPIError

    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()

    turns = [
        _make_advisor_tool_call("call_failing"),
        _make_text_only_response("OK, I'll try something else."),
    ]

    # LiteLLMAPIError is a recognised provider error class — _call_advisor_with_audit
    # catches it, records an API_ERROR-status ComposerLLMCall, and re-raises.
    api_error = LiteLLMAPIError(status_code=500, message="upstream server error", llm_provider="anthropic", model="claude-sonnet-4-6")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=api_error,
        ),
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help me", [], state)

    # Outer ComposerToolInvocation: SUCCESS dispatch (the envelope itself
    # closed cleanly), with ADVISOR_ERROR in result_payload.
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert invs[0].status.name == "SUCCESS", (
        "outer dispatch envelope must close as SUCCESS even when the inner LLM call fails — "
        "the failure mode is captured in result_payload, not status"
    )
    result_canonical = _result_canonical(invs[0])
    assert "ADVISOR_ERROR" in result_canonical
    assert "APIError" in result_canonical, (
        "result must carry the underlying error class so the composer LLM can distinguish "
        "transient API failures from auth/budget/disabled cases"
    )

    # Inner ComposerLLMCall: API_ERROR-status record fired by the
    # _call_advisor_with_audit finally block.
    advisor_llm_calls = [c for c in result.llm_calls if c.model_requested == "anthropic/claude-sonnet-4-6"]
    assert len(advisor_llm_calls) == 1
    assert advisor_llm_calls[0].status.name == "API_ERROR", (
        f"inner LLM call status was {advisor_llm_calls[0].status.name}, expected API_ERROR; "
        "the finally block in _call_advisor_with_audit may not be classifying LiteLLMAPIError correctly"
    )


@pytest.mark.asyncio
async def test_three_failed_advisor_calls_trigger_anti_anchor_hint() -> None:
    """Three identical failed advisor calls in a row must populate the
    §7.7 anti-anchor tracker so the structural hint fires on turn 4.

    This proves the allowlist's claim: "Anti-anchor failure tracker is
    incremented (so repeated identical failed advisor calls trigger the
    §7.7 hint)" — by observing the hint actually appearing in the post-loop
    LLM messages, not by inspecting tracker internals.
    """
    from litellm.exceptions import APIError as LiteLLMAPIError

    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=10))
    state = _empty_state()

    # Three identical advisor calls, all fail. Then turn 4 final text.
    turns = [
        _make_advisor_tool_call("call_1", problem="stuck on llm config"),
        _make_advisor_tool_call("call_2", problem="stuck on llm config"),
        _make_advisor_tool_call("call_3", problem="stuck on llm config"),
        _make_text_only_response("Giving up."),
    ]
    api_error = LiteLLMAPIError(status_code=500, message="x", llm_provider="anthropic", model="claude-sonnet-4-6")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=api_error,
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("help me", [], state)

    # The 4th LLM call is the post-anchor LLM call. It must contain the
    # [ELSPETH-SYSTEM-HINT] message because three identical failed advisor
    # calls hit the tracker's threshold.
    assert mock_llm.call_count == 4
    fourth_call_messages = mock_llm.call_args_list[3].args[0]
    hint_messages = [
        m
        for m in fourth_call_messages
        if isinstance(m, dict) and m.get("role") == "user" and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))
    ]
    assert len(hint_messages) == 1, (
        f"three identical failed advisor calls did not trigger the anti-anchor hint; expected 1 hint message, got {len(hint_messages)}"
    )
    assert "request_advisor_hint" in hint_messages[0]["content"], "hint should name the anchored tool"


@pytest.mark.asyncio
async def test_successful_advisor_call_resets_anti_anchor_failures() -> None:
    """A successful advisor response is real progress and must clear the
    prior advisor failure chain before later identical failures are counted.
    """
    from litellm.exceptions import APIError as LiteLLMAPIError

    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=10))
    state = _empty_state()

    turns = [
        _make_advisor_tool_call("call_1", problem="stuck on llm config"),
        _make_advisor_tool_call("call_2", problem="stuck on llm config"),
        _make_advisor_tool_call("call_success", problem="stuck on llm config"),
        _make_advisor_tool_call("call_3", problem="stuck on llm config"),
        _make_text_only_response("done"),
    ]
    api_errors = [
        LiteLLMAPIError(
            status_code=500,
            message="x",
            llm_provider="anthropic",
            model="claude-sonnet-4-6",
        )
        for _ in range(3)
    ]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=[
                api_errors[0],
                api_errors[1],
                _make_advisor_response("Use the provider/deployment fields from the schema."),
                api_errors[2],
            ],
        ),
    ):
        mock_llm.side_effect = turns
        await service.compose("help me", [], state)

    assert mock_llm.call_count == 5
    fifth_call_messages = mock_llm.call_args_list[4].args[0]
    hint_messages = [
        m
        for m in fifth_call_messages
        if isinstance(m, dict) and m.get("role") == "user" and "[ELSPETH-SYSTEM-HINT]" in str(m.get("content", ""))
    ]
    assert hint_messages == [], "a successful advisor call did not clear stale advisor failures before a later identical failure"


@pytest.mark.asyncio
async def test_advisor_call_is_bounded_by_remaining_compose_deadline() -> None:
    """When the compose deadline is tighter than the advisor timeout, the
    advisor helper receives the remaining compose budget and its TimeoutError
    terminates the compose request instead of being returned as ADVISOR_ERROR.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(
        catalog=catalog,
        settings=_make_settings(
            budget=10,
            composer_timeout_seconds=5.0,
            advisor_timeout_seconds=60.0,
        ),
    )
    state = _empty_state()

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch.object(service, "_call_advisor_with_audit", new_callable=AsyncMock) as mock_advisor,
        pytest.raises(ComposerConvergenceError) as exc_info,
    ):
        mock_llm.side_effect = [
            _make_advisor_tool_call("call_deadline"),
            _make_text_only_response("would incorrectly continue"),
        ]
        mock_advisor.side_effect = TimeoutError()
        await service.compose("help me", [], state)

    assert exc_info.value.budget_exhausted == "timeout"
    assert mock_llm.call_count == 1
    advisor_await = mock_advisor.await_args
    assert advisor_await is not None
    advisor_timeout = advisor_await.kwargs["timeout"]
    assert 0 < advisor_timeout <= 5.0
    assert len(exc_info.value.tool_invocations) == 1
    assert "COMPOSE_TIMEOUT" in _result_canonical(exc_info.value.tool_invocations[0])


@pytest.mark.asyncio
async def test_missing_advisor_trigger_rejects_without_outbound_call() -> None:
    """A call without the call-criteria trigger is locally invalid and
    must not spend an advisor call.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()

    args = {
        "problem_summary": "stuck",
        "recent_errors": ["error A", "error A"],
        "attempted_actions": ["set_pipeline once", "checked schema"],
    }
    missing_trigger_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="missing_trigger",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(args),
                            ),
                        )
                    ],
                )
            )
        ]
    )

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=_make_advisor_response(),
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = [missing_trigger_response, _make_text_only_response("done")]
        result = await service.compose("help", [], state)

    assert mock_acompletion.call_count == 0
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert invs[0].status.name == "ARG_ERROR"
    assert "trigger" in _result_canonical(invs[0])


# --- Post-review fixes (P2 findings) ---


@pytest.mark.asyncio
async def test_f5_advisor_unclassified_exception_still_records_llm_call() -> None:
    """F5: If LiteLLM (or its codec, httpx, json, anything) raises an
    exception class NOT in the typed except clauses of
    _call_advisor_with_audit, the inner ComposerLLMCall record MUST still
    land. Otherwise the broad-except in the compose-loop interception
    has an audit gap for exactly the failure mode that justifies its
    tier-model allowlist entry.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()
    turns = [_make_advisor_tool_call("call_unclass"), _make_text_only_response("done")]

    # ValueError is not in the typed except clauses (Timeout/Cancelled/Auth/
    # BadRequest/APIError/MalformedResponse). Without a catch-all, status
    # stays None and the finally block skips record_llm_call.
    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=ValueError("unexpected codec failure"),
        ),
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help me", [], state)

    advisor_llm_calls = [c for c in result.llm_calls if c.model_requested == "anthropic/claude-sonnet-4-6"]
    assert len(advisor_llm_calls) == 1, (
        "ComposerLLMCall record missing for unclassified exception — the audit-trail-preserves-everything claim is broken"
    )
    assert advisor_llm_calls[0].status.name != "SUCCESS"
    assert advisor_llm_calls[0].error_class == "ValueError"


@pytest.mark.asyncio
async def test_f4_advisor_empty_content_classified_as_malformed() -> None:
    """F4: When the advisor response has a choice but message.content is
    None or empty (content filter, malformed provider output, tool-call-
    only response), this MUST classify as MALFORMED_RESPONSE rather than
    SUCCESS-with-empty-guidance. Empty success consumes budget and tells
    the composer LLM "you got advice" when no information was produced.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()
    turns = [_make_advisor_tool_call("call_empty"), _make_text_only_response("ok")]
    empty_response = _FakeLLMResponse(
        choices=[_FakeChoice(message=_FakeMessage(content=None, tool_calls=None))],
    )

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            return_value=empty_response,
        ),
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help", [], state)

    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert "ADVISOR_ERROR" in _result_canonical(invs[0]), "empty advisor content was treated as success — should be ADVISOR_ERROR"

    advisor_calls = [c for c in result.llm_calls if c.model_requested == "anthropic/claude-sonnet-4-6"]
    assert len(advisor_calls) == 1
    assert advisor_calls[0].status.name == "MALFORMED_RESPONSE"


@pytest.mark.asyncio
async def test_f2_failed_advisor_call_consumes_budget() -> None:
    """F2: When budget is 1 and the first advisor call fails (any LLM-level
    failure), the second call MUST hit BUDGET_EXHAUSTED. Otherwise
    composer_advisor_max_calls_per_compose only limits successful calls,
    and a failing-provider scenario can rack up unbounded outbound spend
    until anti-anchor or discovery-turn limits eventually fire.
    """
    from litellm.exceptions import APIError as LiteLLMAPIError

    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=1))
    state = _empty_state()
    turns = [
        _make_advisor_tool_call("call_1_fails"),
        _make_advisor_tool_call("call_2_should_be_budget_exhausted"),
        _make_text_only_response("done"),
    ]
    api_error = LiteLLMAPIError(status_code=500, message="x", llm_provider="anthropic", model="claude-sonnet-4-6")

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=api_error,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help", [], state)

    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 2
    assert "ADVISOR_ERROR" in _result_canonical(invs[0])
    assert "BUDGET_EXHAUSTED" in _result_canonical(invs[1]), (
        "second advisor call after a failed one was not budget-blocked — the failed first call did not consume budget (F2)"
    )
    assert mock_acompletion.call_count == 1, (
        f"expected 1 outbound advisor call (budget=1, failure counts), got {mock_acompletion.call_count}"
    )


@pytest.mark.asyncio
async def test_f3a_advisor_rejects_non_list_recent_errors() -> None:
    """F3a: recent_errors must be list[str]. _TOOL_REQUIRED_PATHS only
    checks key presence, so a non-list value (LLM bug, prompt injection)
    slips through schema validation. Without local type-check, the
    iterator silently iterates a string character-by-character, sending
    a corrupt prompt at full provider cost.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    state = _empty_state()

    bad_args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck",
        "recent_errors": "single error string not list",  # WRONG TYPE
        "attempted_actions": ["x"],
    }
    bad_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="bad",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(bad_args),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turns = [bad_response, _make_text_only_response("done")]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help", [], state)

    assert mock_acompletion.call_count == 0, "advisor sent malformed-typed argument to provider — should reject locally"
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert "ARG_ERROR" in _result_canonical(invs[0])
    assert invs[0].status.name == "ARG_ERROR", (
        "advisor argument rejection must use the audit ARG_ERROR status, not SUCCESS with ARG_ERROR buried inside result_canonical"
    )


@pytest.mark.asyncio
async def test_f3b_advisor_rejects_oversized_prompt() -> None:
    """F3b: When the combined user prompt would exceed
    composer_advisor_max_prompt_tokens * 4 chars (rough char-to-token
    estimate), reject as ARG_ERROR locally rather than pay unbounded
    outbound cost. Otherwise the cap setting is dead config.
    """
    catalog = _mock_catalog()
    settings = WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=3,
        composer_advisor_max_prompt_tokens=1000,  # → ~4000 char cap
        shareable_link_signing_key=b"\x00" * 32,
    )
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    huge_string = "X" * 50_000
    big_args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck",
        "recent_errors": [huge_string],
        "attempted_actions": ["x"],
    }
    huge_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="big",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(big_args),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turns = [huge_response, _make_text_only_response("done")]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        result = await service.compose("help", [], state)

    assert mock_acompletion.call_count == 0, "oversized prompt was sent to provider — local cap not enforced"
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert "ARG_ERROR" in _result_canonical(invs[0])
    assert invs[0].status.name == "ARG_ERROR", (
        "advisor oversized prompt rejection must use the audit ARG_ERROR status, not SUCCESS with ARG_ERROR buried inside result_canonical"
    )


@pytest.mark.asyncio
async def test_f3c_advisor_prompt_size_counts_formatting_overhead() -> None:
    """F3c: Empty list items still become bullets/newlines in the actual
    advisor prompt. The cap must count that formatted overhead before any
    outbound LiteLLM call.
    """
    catalog = _mock_catalog()
    settings = WebSettings(
        data_dir=Path("/data"),
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        composer_advisor_max_calls_per_compose=3,
        composer_advisor_max_prompt_tokens=20,  # -> ~80 char variable-prompt cap
        shareable_link_signing_key=b"\x00" * 32,
    )
    service = ComposerServiceImpl(catalog=catalog, settings=settings)
    state = _empty_state()

    overhead_args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "x",
        "recent_errors": [""] * 60,
        "attempted_actions": [""] * 60,
    }
    overhead_response = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id="overhead",
                            function=_FakeFunction(
                                name="request_advisor_hint",
                                arguments=json.dumps(overhead_args),
                            ),
                        )
                    ],
                )
            )
        ]
    )
    turns = [overhead_response, _make_text_only_response("done")]

    with (
        patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm,
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
        ) as mock_acompletion,
    ):
        mock_llm.side_effect = turns
        mock_acompletion.return_value = _make_advisor_response()
        result = await service.compose("help", [], state)

    assert mock_acompletion.call_count == 0, "advisor sent a formatted prompt whose bullet/newline overhead exceeded the local cap"
    invs = [i for i in result.tool_invocations if i.tool_name == "request_advisor_hint"]
    assert len(invs) == 1
    assert "ARG_ERROR" in _result_canonical(invs[0])
    assert invs[0].status.name == "ARG_ERROR"


@pytest.mark.asyncio
async def test_f1_skill_text_always_includes_advisor() -> None:
    """F1: advisor is mandatory, so the system prompt fed to the LLM always
    mentions request_advisor_hint so the LLM knows when to use it.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings())
    state = _empty_state()
    turns = [_make_text_only_response("ok")]

    with patch.object(service, "_call_llm", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = turns
        await service.compose("hello", [], state)

    sent_messages = mock_llm.call_args_list[0].args[0]
    system_msg = next(m["content"] for m in sent_messages if m["role"] == "system")
    assert "request_advisor_hint" in system_msg, "system prompt does not mention the advisor tool when it IS enabled"


@pytest.mark.asyncio
async def test_advisor_cancelled_error_carries_buffered_llm_calls() -> None:
    """A cancellation during the advisor LiteLLM call must attach the
    in-memory ComposerLLMCall buffer to the raised CancelledError so the
    session route's cancellation handler can persist the audit row.
    """
    catalog = _mock_catalog()
    service = ComposerServiceImpl(catalog=catalog, settings=_make_settings(budget=3))
    recorder = BufferingRecorder()
    args = {
        "trigger": "proactive_security_safety",
        "problem_summary": "stuck",
        "recent_errors": ["error A"],
        "attempted_actions": ["set_pipeline once"],
    }

    with (
        patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError(),
        ),
        pytest.raises(asyncio.CancelledError) as exc_info,
    ):
        await service._call_advisor_with_audit(args, recorder=recorder)

    assert len(recorder.llm_calls) == 1
    assert recorder.llm_calls[0].status.name == "CANCELLED"
    exc_calls = exc_info.value.__dict__.get("llm_calls")
    assert exc_calls == recorder.llm_calls, (
        "advisor cancellation recorded the inner LLM audit row in memory but did not "
        "attach it to the CancelledError for route-level persistence"
    )
