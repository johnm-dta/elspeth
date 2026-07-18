"""Unit tests for the per-step chat solver (Phase A — advisory text only).

The solver builds a step-scoped system prompt + user message + temp/seed
kwargs, invokes ``_litellm_acompletion``, and returns the assistant message
content as a plain string.

Phase B (separate slice) introduces the per-step tool palette + Tier-3 args
validation; tests for that surface live in test_step_tool_scope.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from elspeth.web.composer.guided import chat_solver
from elspeth.web.composer.guided.chat_solver import (
    AssistantScaffoldLeakError,
    Step1SourceChatResolution,
    _build_step_1_source_dynamic_block,
    _build_step_2_sink_tool_prompt,
    _parse_step_1_source_tool_arguments,
    _parse_step_2_sink_tool_arguments,
    build_step_chat_context_block,
    solve_step_chat,
)
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved


@dataclass
class _FakeMessage:
    content: str | None
    tool_calls: list[Any] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeLLMResponse:
    choices: list[_FakeChoice]


def _ok_response(text: str) -> _FakeLLMResponse:
    return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=text))])


def test_step_1_source_chat_resolution_deep_freezes_container_fields() -> None:
    resolution = Step1SourceChatResolution(
        assistant_message="Created a CSV source.",
        plugin="csv",
        filename="rows.csv",
        mime_type="text/csv",
        content="name\nalice\n",
        options={"schema": {"fields": ["name"]}},
        observed_columns=("name",),
        sample_rows=({"name": "alice"},),
        on_validation_failure="discard",
    )

    with pytest.raises(TypeError):
        resolution.options["delimiter"] = ","  # type: ignore[index]
    with pytest.raises(TypeError):
        resolution.options["schema"]["fields"] = ["other"]  # type: ignore[index,call-overload]
    with pytest.raises(TypeError):
        resolution.sample_rows[0]["name"] = "bob"  # type: ignore[index]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", list(GuidedStep))
async def test_solver_sends_step_scoped_system_prompt(monkeypatch: pytest.MonkeyPatch, step: GuidedStep) -> None:
    """The solver's system prompt must be the per-step skill, NOT the full skill.

    This is the entire point of Phase A — verify scoping is mechanical, not
    a comment.  We capture the kwargs the solver sends to _litellm_acompletion
    and assert the system prompt matches load_step_chat_skill(step).
    """
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeLLMResponse:
        captured.update(kwargs)
        return _ok_response("here's some advice")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    reply = await solve_step_chat(
        model="test/model",
        step=step,
        user_message="hi",
        temperature=None,
        seed=None,
        timeout_seconds=30.0,
    )

    assert reply == "here's some advice"
    messages = captured["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    # messages[1] is the no-tools addendum (solve_step_chat never attaches
    # tools) — a fixed second system message, not step-scoped.
    assert messages[1]["role"] == "system"
    assert messages[2] == {"role": "user", "content": "hi"}

    from elspeth.web.composer.guided.prompts import load_step_chat_skill

    assert messages[0]["content"] == load_step_chat_skill(step), (
        f"system prompt for {step.value} did not match load_step_chat_skill output — per-step scoping is broken"
    )


@pytest.mark.asyncio
async def test_empty_user_message_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty user message is a route-handler-validation gap; we crash loudly."""
    with pytest.raises(InvariantError, match="user_message is empty"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="",
            temperature=None,
            seed=None,
            timeout_seconds=30.0,
        )


@pytest.mark.asyncio
async def test_missing_response_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns None for message.content is defective; we crash loudly."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=None))])

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="hello",
            temperature=None,
            seed=None,
            timeout_seconds=30.0,
        )


@pytest.mark.asyncio
async def test_whitespace_only_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An LLM that returns only whitespace is also defective — same crash path."""

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _ok_response("   \n  \t  \n")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(InvariantError, match="missing message content"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_2_SINK,
            user_message="hello",
            temperature=None,
            seed=None,
            timeout_seconds=30.0,
        )


def test_build_step_chat_context_block_names_artifacts_llm_safely() -> None:
    """The advisory context block carries plugin names / schema modes / field
    lists via the SAME LLM-safe serializers the revision prompts use — never
    raw options, blob paths, or secret-bearing values."""
    current_source = SourceResolved(
        name="source",
        plugin="csv",
        options={
            "schema": {"mode": "observed", "guaranteed_fields": ["url"]},
            "blob_ref": {"id": "blob-1", "storage_path": "/srv/elspeth/blobs/private.csv"},
            "raw_option_should_not_leave": "sk-secret",
        },
        observed_columns=("url",),
        sample_rows=({"url": "https://example.test/a"},),
        on_validation_failure="discard",
    )
    current_sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="main",
                plugin="json",
                options={"path": "results.jsonl", "token": "sk-sink-secret"},
                required_fields=("url", "score"),
                schema_mode="observed",
                on_write_failure="discard",
            ),
        )
    )

    block = build_step_chat_context_block(
        step=GuidedStep.STEP_2_SINK,
        current_source=current_source,
        current_sink=current_sink,
        state=None,
    )

    assert "step_2_sink" in block
    assert '"plugin": "csv"' in block
    assert '"plugin": "json"' in block
    assert '"guaranteed_fields": ["url"]' in block
    # LLM-safe: raw option values, blob paths, and secrets never egress.
    assert "sk-secret" not in block
    assert "sk-sink-secret" not in block
    assert "/srv/elspeth/blobs" not in block
    assert "results.jsonl" not in block


def test_build_step_chat_context_block_is_honest_when_nothing_is_built() -> None:
    block = build_step_chat_context_block(
        step=GuidedStep.STEP_1_SOURCE,
        current_source=None,
        current_sink=None,
        state=None,
    )
    assert "Applied source: none yet." in block
    assert "Applied output: none yet." in block


@pytest.mark.asyncio
async def test_solve_step_chat_threads_context_block_as_third_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The context block rides in messages[2] — the per-step skill stays the
    byte-stable, cache-markable head (same split as the step-1 resolve path);
    the no-tools addendum is the fixed messages[1]."""
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> _FakeLLMResponse:
        captured.update(kwargs)
        return _ok_response("here's what you're seeing")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    reply = await solve_step_chat(
        model="test/model",
        step=GuidedStep.STEP_2_SINK,
        user_message="explain this",
        temperature=None,
        seed=None,
        timeout_seconds=30.0,
        context_block="## Current build\n\nApplied source: none yet.\n",
    )

    assert reply == "here's what you're seeing"
    messages = captured["messages"]
    assert len(messages) == 4
    assert messages[1]["role"] == "system"
    assert messages[2]["role"] == "system"
    assert messages[2]["content"].startswith("## Current build")
    assert messages[3] == {"role": "user", "content": "explain this"}


@pytest.mark.asyncio
async def test_solve_step_chat_rejects_tool_scaffolding_in_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    """The advisory reply persists into chat_history verbatim — same register
    guard as the resolve-path assistant_message args.

    Observed live 2026-07-03 (live guided, step_1): the model answered the
    advisory path with a full pseudo <tool_call>/<tool_response> transcript
    as literal content, which rendered raw in the user-facing bubble. The
    dedicated subclass lets the auto-drop wrapper absorb it (synthetic
    unavailable, Send retryable) while bare ValueError still propagates as a
    caller bug.
    """

    async def fake_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        return _ok_response('Let me look. <tool_call>{"name": "list_sources"}</tool_call> ...prose after.')

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(AssistantScaffoldLeakError, match="user-facing prose"):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="read my csv",
            temperature=None,
            seed=None,
            timeout_seconds=30.0,
        )


def _source_tool_args(**overrides: Any) -> str:
    """A valid resolve_source argument blob (json-encoded), overridable per test."""
    args: dict[str, Any] = {
        "resolution": "source",
        "plugin": "json",
        "filename": "rows.json",
        "mime_type": "application/json",
        "content": '[{"url": "https://example.test/a"}]',
        "options": {"schema": {"mode": "observed"}},
        "observed_columns": ["url"],
        "sample_rows": [{"url": "https://example.test/a"}],
        "assistant_message": "Created the source.",
    }
    args.update(overrides)
    return json.dumps(args)


def test_parse_defaults_on_validation_failure_to_discard_when_omitted() -> None:
    """The optional knob is absent -> default to 'discard' so a passive walk never stalls."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(), plugin_hint="json")
    assert resolution.on_validation_failure == "discard"


def test_parse_preserves_explicit_on_validation_failure() -> None:
    """A composer-chosen value (non-default sentinel) survives the parse verbatim."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure="quarantine_sink"), plugin_hint="json")
    assert resolution.on_validation_failure == "quarantine_sink"


def test_parse_empty_on_validation_failure_defaults_to_discard() -> None:
    """An empty string is treated as 'not set' and defaults to 'discard'."""
    resolution = _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure=""), plugin_hint="json")
    assert resolution.on_validation_failure == "discard"


def test_parse_non_string_on_validation_failure_raises() -> None:
    """When the model sends a non-string value, reject at the Tier-3 boundary."""
    with pytest.raises(ValueError, match="on_validation_failure must be a string"):
        _parse_step_1_source_tool_arguments(_source_tool_args(on_validation_failure=123), plugin_hint="json")


def test_parse_step_2_sink_rejects_non_object_arguments() -> None:
    """Malformed LLM resolve_sink arguments are rejected at the Tier-3 parse boundary."""
    with pytest.raises(ValueError, match="must decode to an object"):
        _parse_step_2_sink_tool_arguments('["not", "an", "object"]')


def test_step_2_sink_tool_schema_and_parser_are_exactly_singular() -> None:
    parameters = chat_solver._STEP_2_SINK_TOOL["function"]["parameters"]
    assert parameters["required"] == ["resolution", "output", "assistant_message"]
    assert "outputs" not in parameters["properties"]
    assert parameters["properties"]["output"]["type"] == "object"

    sink, message = _parse_step_2_sink_tool_arguments(
        json.dumps(
            {
                "resolution": "sink",
                "output": {
                    "name": "accepted",
                    "plugin": "json",
                    "options": {"path": "accepted.jsonl"},
                    "required_fields": ["id"],
                    "schema_mode": "fixed",
                    "on_write_failure": "discard",
                },
                "assistant_message": "Configured the output.",
            }
        )
    )

    assert message == "Configured the output."
    assert [output.name for output in sink.outputs] == ["accepted"]
    assert [output.on_write_failure for output in sink.outputs] == ["discard"]


def test_parse_step_2_sink_rejects_legacy_plural_outputs_field() -> None:
    with pytest.raises(ValueError, match="must contain exactly"):
        _parse_step_2_sink_tool_arguments(
            json.dumps(
                {
                    "resolution": "sink",
                    "outputs": [],
                    "assistant_message": "Configured outputs.",
                }
            )
        )


@pytest.mark.parametrize("failure_case", ["non_finite", "aggregate", "depth", "surrogate"])
def test_parse_step_2_sink_translates_strict_snapshot_failures_to_malformed(failure_case: str) -> None:
    if failure_case == "non_finite":
        bad_options: dict[str, Any] = {"bad": float("nan")}
    elif failure_case == "aggregate":
        bad_options = {f"text_{index}": "x" * 65_000 for index in range(17)}
    elif failure_case == "surrogate":
        bad_options = {"bad": "\ud800"}
    else:
        bad_options = {}
        cursor = bad_options
        for _ in range(65):
            child: dict[str, Any] = {}
            cursor["child"] = child
            cursor = child
    arguments = json.dumps(
        {
            "resolution": "sink",
            "output": {
                "name": "results",
                "plugin": "json",
                "options": bad_options,
                "required_fields": [],
                "schema_mode": "observed",
                "on_write_failure": "discard",
            },
            "assistant_message": "Configured output.",
        }
    )

    with pytest.raises(ValueError, match=r"resolve_sink.*malformed"):
        _parse_step_2_sink_tool_arguments(arguments)


@pytest.mark.parametrize(
    "field_name",
    ["options", "sample_rows", "sample_rows_count", "observed_columns", "surrogate"],
)
def test_parse_step_1_source_translates_strict_snapshot_failures_to_malformed(field_name: str) -> None:
    payload = json.loads(_source_tool_args())
    if field_name == "sample_rows":
        payload[field_name] = [{"bad": float("inf")}]
    elif field_name == "sample_rows_count":
        payload["sample_rows"] = [{}] * 10_001
    elif field_name == "observed_columns":
        payload[field_name] = [f"column-{index}-{'x' * 64_980}" for index in range(17)]
    elif field_name == "surrogate":
        payload["options"] = {"bad": "\ud800"}
    else:
        payload[field_name] = {"bad": float("inf")}

    with pytest.raises(ValueError, match=r"resolve_source.*malformed"):
        _parse_step_1_source_tool_arguments(json.dumps(payload), plugin_hint="json")


def test_parse_step_1_source_rejects_deep_snapshot_before_route_side_effects() -> None:
    deep: dict[str, object] = {}
    cursor = deep
    for _ in range(65):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child

    with pytest.raises(ValueError, match=r"resolve_source.*malformed"):
        _parse_step_1_source_tool_arguments(_source_tool_args(options=deep), plugin_hint="json")


def test_parse_rejects_tool_scaffolding_in_assistant_message() -> None:
    """A model that leaks its agentic scratchpad into assistant_message is rejected.

    Observed live 2026-07-03: a 2.8KB pseudo tool-call transcript
    (``<tool_call>{"name": "list_sources"}...``) persisted verbatim into a
    tutorial chat history and rendered as the learner-facing reply. The
    register violation must route to MALFORMED_RESPONSE (retryable advisory),
    never into chat_history.
    """
    scratchpad = 'Let me check.\n\n<tool_call>{"name": "list_sources"}</tool_call>\n<tool_response>[...]</tool_response>\nDone.'
    with pytest.raises(ValueError, match="user-facing prose"):
        _parse_step_1_source_tool_arguments(_source_tool_args(assistant_message=scratchpad), plugin_hint="json")


def test_parse_rejects_tool_scaffolding_case_insensitively() -> None:
    with pytest.raises(ValueError, match="user-facing prose"):
        _parse_step_1_source_tool_arguments(_source_tool_args(assistant_message="<TOOL_CALL>{}</TOOL_CALL>"), plugin_hint="json")


def test_step_1_source_dynamic_block_guides_aws_s3_endpoint_url_policy() -> None:
    prompt = _build_step_1_source_dynamic_block(plugin_hint="csv")

    expected_guidance = ("`aws_s3`", "`endpoint_url`", "CLI/batch-only", "web authors must never set it")
    assert [text for text in expected_guidance if text not in prompt] == []


def test_step_1_revision_prompt_uses_llm_safe_source_context() -> None:
    current_source = SourceResolved(
        name="source",
        plugin="csv",
        options={
            "schema": {"mode": "observed", "guaranteed_fields": ["email", "profile_url"]},
            "blob_ref": {"id": "blob-private-source-id", "storage_path": "/srv/elspeth/blobs/private.csv"},
            "raw_option_key_should_not_leave": "sk-option-secret",
        },
        observed_columns=("email", "profile_url", "note"),
        sample_rows=(
            {
                "email": "person@example.test",
                "profile_url": "https://example.test/private?token=secret",
                "note": "customer asked for refunds",
            },
        ),
        on_validation_failure="quarantine",
    )

    prompt = _build_step_1_source_dynamic_block(plugin_hint="csv", current_source=current_source)

    assert "person@example.test" not in prompt
    assert "https://example.test/private" not in prompt
    assert "customer asked for refunds" not in prompt
    assert "blob-private-source-id" not in prompt
    assert "/srv/elspeth/blobs/private.csv" not in prompt
    assert "raw_option_key_should_not_leave" not in prompt
    assert "sk-option-secret" not in prompt
    assert '"plugin": "csv"' in prompt
    assert '"mode": "observed"' in prompt
    assert '"guaranteed_fields": ["email", "profile_url"]' in prompt
    assert "<sample:email-like>" in prompt
    assert "<sample:url>" in prompt
    assert "<sample:string:" in prompt


def test_step_2_revision_prompt_uses_llm_safe_sink_context() -> None:
    current_sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="main",
                plugin="azure_blob",
                options={
                    "path": "/srv/elspeth/exports/private-output.jsonl",
                    "sas_token": "sv=private-token",
                    "raw_sink_option_key_should_not_leave": {"secret_ref": "PROD_BLOB_SECRET"},
                },
                required_fields=("email_hash", "profile_url"),
                schema_mode="fixed",
                on_write_failure="discard",
            ),
        )
    )

    prompt = _build_step_2_sink_tool_prompt(current_sink=current_sink)

    assert "/srv/elspeth/exports/private-output.jsonl" not in prompt
    assert "sv=private-token" not in prompt
    assert "raw_sink_option_key_should_not_leave" not in prompt
    assert "PROD_BLOB_SECRET" not in prompt
    assert '"plugin": "azure_blob"' in prompt
    assert '"schema_mode": "fixed"' in prompt
    assert '"required_fields": ["email_hash", "profile_url"]' in prompt
    assert '"option_count": 3' in prompt


@pytest.mark.asyncio
async def test_solve_step_chat_timeout_seconds_bounds_the_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guided chat LLM call is server-side bounded (elspeth-fb4464cdf0).

    A hung provider call must raise TimeoutError once ``timeout_seconds``
    elapses — the same freeform-compose bound (asyncio.wait_for on
    ``composer_timeout_seconds``).
    """
    import asyncio

    async def hung_acompletion(**_kwargs: Any) -> _FakeLLMResponse:
        await asyncio.sleep(60)
        raise AssertionError("unreachable — the wait_for bound must fire first")

    monkeypatch.setattr(chat_solver, "_litellm_acompletion", hung_acompletion)

    with pytest.raises(TimeoutError):
        await solve_step_chat(
            model="test/model",
            step=GuidedStep.STEP_1_SOURCE,
            user_message="hello",
            temperature=None,
            seed=None,
            timeout_seconds=0.01,
        )


@pytest.mark.asyncio
async def test_bounded_acompletion_rejects_absent_or_invalid_timeout() -> None:
    import inspect

    assert inspect.signature(solve_step_chat).parameters["timeout_seconds"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="finite positive"):
        await chat_solver._bounded_acompletion({}, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite positive"):
        await chat_solver._bounded_acompletion({}, 0)
