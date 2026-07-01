"""p1 Task 1 — source driver accepts current applied source for in-place revise."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

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
async def test_source_driver_includes_current_source_in_prompt() -> None:
    current = SourceResolved(
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
        new=AsyncMock(side_effect=_capture),
    ):
        result = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="add a second url",
            plugin_hint="json",
            current_source=current,
            temperature=None,
            seed=None,
        )

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
        new=AsyncMock(side_effect=_echo_server_owned_keys),
    ):
        result = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="add a second url",
            plugin_hint="json",
            current_source=current,
            temperature=None,
            seed=None,
        )

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
async def test_source_driver_returns_none_on_prose() -> None:
    prose = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Here is some advice.", tool_calls=None))])
    with patch(
        "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
        new=AsyncMock(return_value=prose),
    ):
        result = await maybe_resolve_step_1_source_chat(
            model="anthropic/claude-sonnet-4.6",
            user_message="what is a source?",
            plugin_hint="json",
            current_source=None,
            temperature=None,
            seed=None,
        )
    assert result is None
