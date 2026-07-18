"""p1 Task 2 — sink driver resolves free text into a SinkResolved."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus, ComposerLLMCallStatus
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.catalog.protocol import PluginKind
from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.chat_solver import Step2SinkChatOutcome, maybe_resolve_step_2_sink_chat
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions._guided_step_chat import (
    Step2SinkChatResult,
    resolve_step_2_sink_chat_with_auto_drop,
)


@dataclass(frozen=True, slots=True)
class _FakeCatalogService:
    sinks: tuple[PluginSummary, ...] = ()

    def list_sources(self) -> list[PluginSummary]:
        return []

    def list_transforms(self) -> list[PluginSummary]:
        return []

    def list_sinks(self) -> list[PluginSummary]:
        return list(self.sinks)

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        raise AssertionError(f"unexpected schema lookup for {plugin_type}:{name}")

    def post_call_hints(
        self,
        *,
        plugin_type: PluginKind,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        return ()


def _fake_resolve_sink_response(args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_sink",
                                arguments=json.dumps(args),
                            )
                        )
                    ],
                )
            )
        ]
    )


_JSON_SINK_ARGS = {
    "resolution": "sink",
    "outputs": [
        {
            "plugin": "json",
            "options": {"path": "out.jsonl", "schema": {"mode": "observed"}},
            "required_fields": [],
            "schema_mode": "observed",
        }
    ],
    "assistant_message": "I set the output to a JSON Lines file.",
}


@pytest.mark.asyncio
async def test_sink_driver_resolves_json_output() -> None:
    async def _return_json_sink_response(**_kwargs: object) -> SimpleNamespace:
        return _fake_resolve_sink_response(_JSON_SINK_ARGS)

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_return_json_sink_response,
    ):
        outcome = await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="write the results to a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    sink = outcome.sink
    assert sink is not None
    assert isinstance(sink, SinkResolved)
    assert len(sink.outputs) == 1
    assert sink.outputs[0].plugin == "json"
    assert sink.outputs[0].options["path"] == "out.jsonl"
    assert outcome.assistant_message == "I set the output to a JSON Lines file."


@pytest.mark.asyncio
async def test_sink_driver_captures_prose_reply_on_decline() -> None:
    """No resolve_sink call: the outcome carries the model's own prose reply.

    Captured directly rather than discarded — the caller (the guided-chat
    route) uses this to answer without a second, tool-less call (C-1 fix).
    """
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A sink writes rows out.", tool_calls=None))])

    async def _return_prose_reply(**_kwargs: object) -> SimpleNamespace:
        return prose

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_return_prose_reply,
    ):
        outcome = await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert outcome.sink is None
    assert outcome.assistant_message == "A sink writes rows out."


@pytest.mark.asyncio
async def test_sink_driver_returns_both_none_on_hallucinated_tool_call() -> None:
    """A non-discovery, non-resolve_sink tool call: both fields None — prose is
    NOT trusted here even if present, since the response shape is more
    suspicious than a clean prose decline (unchanged from before the salvage)."""
    hallucinated = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="I'll check something first.",
                    tool_calls=[SimpleNamespace(function=SimpleNamespace(name="delete_everything", arguments="{}"))],
                )
            )
        ]
    )

    async def _return_hallucinated_tool_call(**_kwargs: object) -> SimpleNamespace:
        return hallucinated

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_return_hallucinated_tool_call,
    ):
        outcome = await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert outcome.sink is None
    assert outcome.assistant_message is None


@pytest.mark.asyncio
async def test_sink_driver_rejects_scaffold_leak_in_declined_prose() -> None:
    """A scaffold leak in the DECLINED-PROSE branch, not the tool argument.

    Same register guard (``_require_prose_assistant_message``), a different
    call site: this is the new salvage path added alongside the C-1 fix, so
    it needs its own coverage that a leak here raises loudly too, exactly
    like a leak in ``resolve_sink``'s own ``assistant_message`` argument.
    """
    from elspeth.web.composer.guided.chat_solver import AssistantScaffoldLeakError

    scaffold_reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='Let me check. <tool_call>{"name": "list_sinks"}</tool_call> json fits.',
                    tool_calls=None,
                )
            )
        ]
    )

    async def _return_scaffold_reply(**_kwargs: object) -> SimpleNamespace:
        return scaffold_reply

    with (
        patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=_return_scaffold_reply,
        ),
        pytest.raises(AssistantScaffoldLeakError, match="user-facing prose"),
    ):
        await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
        )


@pytest.mark.asyncio
async def test_sink_driver_revise_threads_current_sink() -> None:
    current = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="main",
                on_write_failure="discard",
                plugin="json",
                options={"path": "old.jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_resolve_sink_response(_JSON_SINK_ARGS)

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_capture,
    ):
        await maybe_resolve_step_2_sink_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="rename the file to out.jsonl",
            current_sink=current,
            temperature=None,
            seed=None,
        )
    system_prompt = captured["messages"][0]["content"]
    assert '"plugin": "json"' in system_prompt
    assert '"schema_mode": "observed"' in system_prompt
    assert '"option_count": 1' in system_prompt
    assert "old.jsonl" not in system_prompt


@pytest.mark.asyncio
async def test_sink_wrapper_carries_declined_prose_as_prose_chat() -> None:
    """The step-2 single-LLM-call salvage seam: when the driver declines the
    tool and replies in prose, the wrapper must surface that reply on
    ``prose_chat`` (a SUCCESS turn), NOT ``fallback_chat``. The route consumes
    ``fallback_chat or prose_chat`` so this is what makes the step-2 answer a
    single call instead of a second, tool-less one — mirrors the step-1
    single-call guarantee. Without this coverage a refactor that stopped
    populating ``prose_chat`` (it defaults to None for construction-site
    compatibility) would silently revert step 2 to double-call, all tests
    green. Regression for the fp-review step-2-salvage-untested finding.
    """
    declined = Step2SinkChatOutcome(sink=None, assistant_message="A sink writes your rows out to a file or table.")

    async def _decline_sink_chat(**_kwargs: object) -> Step2SinkChatOutcome:
        return declined

    with patch(
        "elspeth.web.sessions._guided_step_chat.maybe_resolve_step_2_sink_chat",
        new=_decline_sink_chat,
    ):
        result = await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert isinstance(result, Step2SinkChatResult)
    assert result.sink_resolution is None
    assert result.fallback_chat is None
    assert result.prose_chat is not None
    assert result.prose_chat.assistant_message == "A sink writes your rows out to a file or table."
    assert result.prose_chat.status == ComposerChatTurnStatus.SUCCESS


@pytest.mark.asyncio
async def test_sink_wrapper_threads_progress_sink_to_driver() -> None:
    """The route's ``progress`` kwarg must reach the real discovery loop, not
    get dropped at the auto-drop wrapper seam. Patches the wrapper's actual
    dependency (``maybe_resolve_step_2_sink_chat``) rather than the sink's own
    call signature so a future rename of the ``progress`` parameter on either
    side would fail this test instead of silently decoupling them.
    """
    captured_kwargs: dict[str, object] = {}

    async def _capture_and_resolve(**kwargs: object) -> Step2SinkChatOutcome:
        captured_kwargs.update(kwargs)
        return Step2SinkChatOutcome(sink=None, assistant_message="ok")

    async def _progress_sink(_event: object) -> None:
        return None

    with patch(
        "elspeth.web.sessions._guided_step_chat.maybe_resolve_step_2_sink_chat",
        new=_capture_and_resolve,
    ):
        await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a sink?",
            current_sink=None,
            temperature=None,
            seed=None,
            progress=_progress_sink,
        )
    assert captured_kwargs["progress"] is _progress_sink


@pytest.mark.asyncio
async def test_sink_wrapper_absorbs_transient_into_synthetic_unavailable() -> None:
    async def _raise_provider_timeout(**_kwargs: object) -> Step2SinkChatOutcome:
        raise TimeoutError("provider timeout")

    with patch(
        "elspeth.web.sessions._guided_step_chat.maybe_resolve_step_2_sink_chat",
        new=_raise_provider_timeout,
    ):
        result = await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="write a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
        )
    assert isinstance(result, Step2SinkChatResult)
    assert result.sink_resolution is None
    assert result.fallback_chat is not None
    assert result.fallback_chat.error_class == "TimeoutError"
    assert result.fallback_chat.status == ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE


@pytest.mark.asyncio
async def test_sink_wrapper_classifies_strict_snapshot_failure_as_malformed_and_auto_drops() -> None:
    malformed_args = {
        **_JSON_SINK_ARGS,
        "outputs": [
            {
                **_JSON_SINK_ARGS["outputs"][0],
                "options": {"bad": float("nan")},
            }
        ],
    }

    async def _return_malformed_sink_response(**_kwargs: object) -> SimpleNamespace:
        return _fake_resolve_sink_response(malformed_args)

    recorder = BufferingRecorder()
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_return_malformed_sink_response,
    ):
        result = await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="write a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
            recorder=recorder,
        )

    assert result.sink_resolution is None
    assert result.fallback_chat is not None
    assert result.fallback_chat.error_class == "ValueError"
    assert result.fallback_chat.status == ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE
    assert recorder.llm_calls[-1].status is ComposerLLMCallStatus.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_sink_wrapper_absorbs_malformed_discovery_args_into_synthetic_unavailable() -> None:
    """A malformed discovery call (non-object arguments) raises
    ``ChainSolverResponseShapeError`` deep in the sink discovery loop; the
    wrapper must absorb it into the synthetic-unavailable fallback — exactly
    like ``solve_chain``'s auto-drop path — not let it escape as a 500.

    Regression for the sink/chain asymmetry: ``solve_chain`` lists the class in
    its transient set but the sink twin did not, so a model that emitted an
    *allowed* discovery tool with garbage arguments crashed the request.
    """
    full_catalog = _FakeCatalogService(
        sinks=(PluginSummary(name="json", description="JSON Lines sink", plugin_type="sink", config_fields=[]),)
    )
    plugin_snapshot = PluginAvailabilitySnapshot.for_trained_operator(full_catalog)
    catalog = PolicyCatalogView.for_trained_operator(full_catalog, plugin_snapshot)
    state = CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)
    # ``list_sinks`` is an allowed discovery tool, but its arguments decode to a
    # non-object, so the production ``_execute_discovery_call`` raises
    # ``ChainSolverResponseShapeError`` when the loop dispatches it.
    malformed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="list_sinks", arguments="[1, 2, 3]"),
                        )
                    ],
                )
            )
        ]
    )

    async def _return_malformed_discovery_call(**_kwargs: object) -> SimpleNamespace:
        return malformed

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_return_malformed_discovery_call,
    ):
        result = await resolve_step_2_sink_chat_with_auto_drop(
            site="test",
            session_id="s1",
            user_id="u1",
            model="anthropic/claude-sonnet-4.6",
            user_message="write a jsonl file",
            current_sink=None,
            temperature=None,
            seed=None,
            state=state,
            catalog=catalog,
            plugin_snapshot=plugin_snapshot,
        )
    assert isinstance(result, Step2SinkChatResult)
    assert result.sink_resolution is None
    assert result.fallback_chat is not None
    assert result.fallback_chat.error_class == "ChainSolverResponseShapeError"
    assert result.fallback_chat.status == ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE
