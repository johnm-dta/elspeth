"""p1 Task 1 — source driver accepts current applied source for in-place revise."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
from elspeth.web.composer.guided.resolved import SourceResolved


def _fake_resolve_source_response(args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="resolve_source",
                                arguments=json.dumps(args),
                            )
                        )
                    ],
                )
            )
        ]
    )


@pytest.mark.asyncio
async def test_source_driver_retries_inline_json_control_advice_into_tool_call() -> None:
    """The source chat must not tell users to pick a nonexistent inline-JSON
    wizard control when it has the resolve_source tool available.
    """
    calls: list[dict] = []
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Choose the inline JSON source option and paste the rows there.",
                        tool_calls=None,
                    )
                )
            ]
        ),
        _fake_resolve_source_response(
            {
                "resolution": "source",
                "plugin": "json",
                "filename": "rows.json",
                "mime_type": "application/json",
                "content": '[{"line": "alpha"}]',
                "options": {"schema": {"mode": "observed", "guaranteed_fields": ["line"]}},
                "observed_columns": ["line"],
                "sample_rows": [{"line": "alpha"}],
                "assistant_message": "Created the JSON rows as the source.",
            }
        ),
    ]

    async def _first_advises_missing_control_then_resolves(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_first_advises_missing_control_then_resolves,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message='Create a simple inline JSON source with [{"line": "alpha"}].',
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )

    assert len(calls) == 2
    assert outcome.prose_reply is None
    assert outcome.resolution is not None
    assert outcome.resolution.plugin == "json"
    assert outcome.resolution.observed_columns == ("line",)


@pytest.mark.asyncio
async def test_source_driver_includes_current_source_in_prompt() -> None:
    current = SourceResolved(
        name="source",
        on_validation_failure="discard",
        plugin="json",
        options={"schema": {"mode": "observed"}, "blob_ref": "abc"},
        observed_columns=("url",),
        sample_rows=({"url": "https://example.test/a"},),
    )
    captured: dict = {}

    async def _capture(**kwargs):
        captured.update(kwargs)
        return _fake_resolve_source_response(
            {
                "resolution": "source",
                "plugin": "json",
                "filename": "urls.json",
                "mime_type": "application/json",
                "content": '[{"url": "https://example.test/a"}]',
                "options": {"schema": {"mode": "observed"}},
                "observed_columns": ["url"],
                "sample_rows": [{"url": "https://example.test/a"}],
                "assistant_message": "Updated the URL list.",
            }
        )

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_capture,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="add a second url",
            plugin_hint="json",
            current_source=current,
            temperature=None,
            seed=None,
        )

    result = outcome.resolution
    assert result is not None
    assert result.plugin == "json"
    # The system prompt is SPLIT: messages[0] is the stable skill head (the
    # markable cache prefix), and the dynamic block — including the current-source
    # revision JSON — rides in messages[1]. Relocation, not a regression: the
    # current source is still threaded, still redacted.
    dynamic_block = captured["messages"][1]["content"]
    # The current applied source MUST be threaded so "add" resolves relative to it,
    # without leaking literal sample values into the prompt.
    assert '"plugin": "json"' in dynamic_block
    assert '"observed_columns": ["url"]' in dynamic_block
    assert '"url": "<sample:url>"' in dynamic_block
    assert "https://example.test/a" not in dynamic_block


@pytest.mark.asyncio
async def test_source_driver_strips_echoed_server_owned_keys() -> None:
    """A re-resolve must not carry server-owned keys back into the resolution options.

    On a SECOND Send the current_source is the already-committed source, whose
    options carry a system-injected ``blob_ref`` (``handle_step_1_source`` looks
    the blob up by storage_path and stamps its UUID). That source is threaded
    into the resolver prompt, so the model parrots the ``blob_ref`` back into
    ``resolve_source``'s options. ``blob_ref`` is re-derived authoritatively at
    commit, so ``set_source`` REJECTS any caller-supplied ``blob_ref`` — which
    turned the second Send into a 400 "Step 1 source commit failed". The parser
    must drop ``blob_ref`` at this Tier-3 boundary so the re-commit succeeds.
    """
    # A committed blob-backed source carries BOTH server-owned keys: blob_ref
    # (stamped by handle_step_1_source) and source_authoring (stamped by
    # set_source_from_blob for LLM-authored / dynamic sources).
    current = SourceResolved(
        name="source",
        on_validation_failure="discard",
        plugin="json",
        options={
            "schema": {"mode": "observed"},
            "blob_ref": "abc",
            "source_authoring": {"creation_modality": "verbatim", "content_hash": "0" * 64},
            "path": "/x",
        },
        observed_columns=("url",),
        sample_rows=({"url": "https://example.test/a"},),
    )

    async def _echo_server_owned_keys(**kwargs):
        return _fake_resolve_source_response(
            {
                "resolution": "source",
                "plugin": "json",
                "filename": "urls.json",
                "mime_type": "application/json",
                "content": '[{"url": "https://example.test/a"}]',
                # The model parrots the server-owned keys it saw in current_source,
                # alongside legitimate keys (schema) and a PENDING invented-source
                # review requirement, which must SURVIVE the narrow strip.
                "options": {
                    "schema": {"mode": "observed"},
                    "blob_ref": "abc",
                    "source_authoring": {"creation_modality": "verbatim", "content_hash": "0" * 64},
                    "interpretation_requirements": [{"kind": "invented_source", "status": "pending"}],
                },
                "observed_columns": ["url"],
                "sample_rows": [{"url": "https://example.test/a"}],
                "assistant_message": "Updated the URL list.",
            }
        )

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_echo_server_owned_keys,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="add a second url",
            plugin_hint="json",
            current_source=current,
            temperature=None,
            seed=None,
        )

    result = outcome.resolution
    assert result is not None
    # Drop-direction: both keys are server-owned; neither may survive an LLM source
    # resolution (set_source rejects caller-supplied blob_ref / source_authoring,
    # which turned the next Send into a 400 "Step 1 source commit failed").
    assert "blob_ref" not in result.options
    assert "source_authoring" not in result.options
    # Keep-direction: the strip is deliberately NARROW — legitimate keys and a
    # pending invented-source review requirement must be preserved (a regression
    # that wiped all options, or widened the blocklist, would be caught here).
    # (options are deep-frozen to mappingproxy/tuples — assert presence/readability,
    # not deep dict/list equality.)
    assert result.options["schema"]["mode"] == "observed"
    assert "interpretation_requirements" in result.options
    assert result.options["interpretation_requirements"][0]["kind"] == "invented_source"


def test_resolver_forbidden_keys_stay_in_lockstep_with_commit_side_class() -> None:
    """The resolver strip set MUST equal the commit-side class of server-owned source
    keys (``_WEB_ONLY_SOURCE_KEYS``) that ``set_source`` rejects. If a new server-owned
    key gains a reject guard but is not added to the resolver strip, a re-Send echo of
    that key silently re-triggers the original commit-failure mode. This guard fails
    loudly on that drift."""
    from elspeth.web.composer.guided.chat_solver import _RESOLVER_FORBIDDEN_SOURCE_OPTION_KEYS
    from elspeth.web.composer.tools._common import _WEB_ONLY_SOURCE_KEYS

    assert _RESOLVER_FORBIDDEN_SOURCE_OPTION_KEYS == _WEB_ONLY_SOURCE_KEYS


@pytest.mark.asyncio
async def test_source_driver_captures_prose_reply_on_decline() -> None:
    """No resolve_source call: the outcome carries the model's own prose reply.

    Captured directly rather than discarded — the caller (the guided-chat
    route) uses this to answer without a second, tool-less call (C-1 fix).
    """
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Here is some advice.", tool_calls=None))])

    async def _decline_with_prose(**kwargs):
        return prose

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_decline_with_prose,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a source?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert outcome.resolution is None
    assert outcome.prose_reply == "Here is some advice."


@pytest.mark.asyncio
async def test_source_driver_rejects_scaffold_leak_in_declined_prose() -> None:
    """A scaffold leak in the DECLINED-PROSE branch, not the tool argument.

    Same register guard (``_require_prose_assistant_message``), a different
    call site: this is the new salvage path added alongside the C-1 fix, so
    it needs its own coverage that a leak here raises loudly too, exactly
    like a leak in ``resolve_source``'s own ``assistant_message`` argument.
    """
    from elspeth.web.composer.guided.chat_solver import AssistantScaffoldLeakError

    scaffold_reply = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='Let me check. <tool_call>{"name": "list_sources"}</tool_call> csv fits.',
                    tool_calls=None,
                )
            )
        ]
    )

    async def _decline_with_scaffold_leak(**kwargs):
        return scaffold_reply

    with (
        patch(
            "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
            new=_decline_with_scaffold_leak,
        ),
        pytest.raises(AssistantScaffoldLeakError, match="user-facing prose"),
    ):
        await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a source?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )


@pytest.mark.asyncio
async def test_source_driver_declines_prose_beside_hallucinated_tool_call() -> None:
    """Prose that arrives ALONGSIDE a hallucinated (non-resolve_source) tool
    call must NOT be salvaged: the narration describes an action that never
    ran. This mirrors the step-2 sink salvage's ``not tool_calls`` gate — the
    outcome is both-None so the route falls back to the tool-less advisory
    call (grounded by the no-tools addendum) rather than showing the user a
    dangling "Let me look up the sources…" reply. Regression for the fp-review
    step-1/step-2 salvage-asymmetry finding.
    """
    hallucinated = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Let me look up the available source types for you...",
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(name="list_sources", arguments="{}")),
                    ],
                )
            )
        ]
    )

    async def _decline_with_hallucinated_tool_call(**kwargs):
        return hallucinated

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_decline_with_hallucinated_tool_call,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what sources can I use?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert outcome.resolution is None
    assert outcome.prose_reply is None


@pytest.mark.asyncio
async def test_source_driver_returns_both_none_on_empty_response() -> None:
    """No tool call AND no content: both fields None — the genuinely defective
    case, unchanged from before the salvage — the caller falls back to the
    advisory chat path (which raises InvariantError on this)."""
    empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None))])

    async def _empty_response(**kwargs):
        return empty

    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=_empty_response,
    ):
        outcome = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a source?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert outcome.resolution is None
    assert outcome.prose_reply is None
