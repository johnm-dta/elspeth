"""Unit tests for Anthropic prompt-cache marker wiring (elspeth-4e79436719 §Phase 3).

The transform helpers in ``llm_response_parsing.py``:

- ``supports_anthropic_prompt_cache_markers`` — provider-detection
  predicate. Returns True for Anthropic-family routes that honor
  ``cache_control`` markers; False for OpenAI/Azure/Gemini and
  unrecognized strings.
- ``apply_anthropic_cache_markers`` — applies
  ``cache_control: {"type": "ephemeral"}`` to the first system message
  and the trailing tool, returning new lists without mutating the
  inputs. Composer message construction keeps dynamic state in a later
  user-role data message so the first marker covers only the stable prompt
  prefix.

These tests target the helpers directly so a future change to either
detection or marker placement fails this file with a focused error,
not a downstream integration miss.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.core.canonical import stable_hash
from elspeth.web.composer.llm_response_parsing import (
    apply_anthropic_cache_markers,
    supports_anthropic_prompt_cache_markers,
)
from tests.unit.web.composer._helpers import _stub_advisor_end_gate_clean  # noqa: F401  (autouse end-gate CLEAN stub)


class TestSupportsAnthropicPromptCacheMarkers:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-3-5-haiku-20241022",
            "openrouter/anthropic/claude-sonnet-4",
            "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            "vertex_ai/claude-3-5-sonnet@20241022",
            "claude-3-5-sonnet-20241022",
            "claude-opus-4-5",
        ],
    )
    def test_anthropic_family_models_supported(self, model: str) -> None:
        assert supports_anthropic_prompt_cache_markers(model) is True, (
            f"Anthropic-family model {model!r} must opt into cache_control markers."
        )

    @pytest.mark.parametrize(
        "model",
        [
            "openai/gpt-4.1",
            "openrouter/openai/gpt-5.5",
            "azure/gpt-4-turbo",
            "gemini/gemini-2.5-pro",
            "vertex_ai/gemini-2.5-pro",
            "ollama/llama3",
        ],
    )
    def test_non_anthropic_models_not_supported(self, model: str) -> None:
        assert supports_anthropic_prompt_cache_markers(model) is False, (
            f"Non-Anthropic model {model!r} must NOT carry cache_control markers."
        )

    def test_none_input_returns_false(self) -> None:
        assert supports_anthropic_prompt_cache_markers(None) is False

    def test_non_string_input_returns_false(self) -> None:
        assert supports_anthropic_prompt_cache_markers(42) is False  # type: ignore[arg-type]


class TestApplyAnthropicCacheMarkers:
    def test_system_message_receives_cache_control(self) -> None:
        messages = [
            {"role": "system", "content": "You are a pipeline composer."},
            {"role": "user", "content": "Build me a CSV pipeline."},
        ]
        new_messages, _ = apply_anthropic_cache_markers(messages, None)
        assert new_messages[0]["cache_control"] == {"type": "ephemeral"}
        assert new_messages[1] == messages[1], "User message must be untouched"

    def test_inputs_are_not_mutated(self) -> None:
        messages = [{"role": "system", "content": "..."}]
        tools: list[dict[str, Any]] = [
            {"type": "function", "function": {"name": "set_source"}},
            {"type": "function", "function": {"name": "set_pipeline"}},
        ]
        apply_anthropic_cache_markers(messages, tools)
        assert "cache_control" not in messages[0]
        assert "cache_control" not in tools[-1]

    def test_only_first_system_message_marked(self) -> None:
        """If two system messages exist, only the first gets the marker.

        Composer uses this shape intentionally: the stable skill prompt
        is the first system message, and dynamic current-state JSON is a
        later system message that must not become part of the stable
        prompt-cache breakpoint.
        """
        messages = [
            {"role": "system", "content": "Skill prompt."},
            {"role": "system", "content": "Current pipeline state and available plugins:\n{}"},
        ]
        new_messages, _ = apply_anthropic_cache_markers(messages, None)
        assert "cache_control" in new_messages[0]
        assert "cache_control" not in new_messages[1]

    def test_no_system_message_means_no_marker_added(self) -> None:
        """If there is no system message, leave messages alone."""
        messages = [{"role": "user", "content": "hi"}]
        new_messages, _ = apply_anthropic_cache_markers(messages, None)
        assert new_messages == messages

    def test_tools_last_entry_receives_cache_control(self) -> None:
        tools: list[dict[str, Any]] = [
            {"type": "function", "function": {"name": "set_source"}},
            {"type": "function", "function": {"name": "set_pipeline"}},
            {"type": "function", "function": {"name": "preview_pipeline"}},
        ]
        _, new_tools = apply_anthropic_cache_markers([], tools)
        assert new_tools is not None
        assert new_tools[0] == tools[0]
        assert new_tools[1] == tools[1]
        assert new_tools[-1]["cache_control"] == {"type": "ephemeral"}
        # The function dict inside the tool stays unchanged.
        assert new_tools[-1]["function"] == tools[-1]["function"]

    def test_empty_tools_list_returns_none(self) -> None:
        _, new_tools = apply_anthropic_cache_markers([], [])
        assert new_tools is None

    def test_none_tools_returns_none(self) -> None:
        _, new_tools = apply_anthropic_cache_markers([], None)
        assert new_tools is None

    def test_returns_new_lists_not_aliases(self) -> None:
        messages = [{"role": "system", "content": "..."}]
        tools: list[dict[str, Any]] = [{"type": "function", "function": {"name": "x"}}]
        new_messages, new_tools = apply_anthropic_cache_markers(messages, tools)
        assert new_messages is not messages
        assert new_tools is not tools

    def test_inputs_are_not_mutated_at_any_depth(self) -> None:
        """Strengthened immutability check (post-review concern #1).

        The transform uses shallow merges (``{**entry, "cache_control": ...}``)
        which create fresh outer dicts but share nested references between
        input and output. This test pins both halves of that contract:

        1. The original messages/tools dicts must NOT gain ``cache_control``
           (the helper does not mutate).
        2. The nested ``function`` dict on the marked tool must remain
           identity-shared with the original — this confirms the helper
           did NOT defensively deep-copy. If a future change starts
           deep-copying, the test fails and the change must be justified
           (deep-copy has a non-trivial cost on the hot path).
        """
        original_function = {"name": "set_pipeline", "parameters": {"type": "object"}}
        messages = [{"role": "system", "content": "anchor"}]
        tools: list[dict[str, Any]] = [
            {"type": "function", "function": {"name": "set_source"}},
            {"type": "function", "function": original_function},
        ]
        _, new_tools = apply_anthropic_cache_markers(messages, tools)
        # Original entries unchanged.
        assert "cache_control" not in messages[0]
        assert "cache_control" not in tools[-1]
        # Identity preservation: marker output shares nested function dict
        # with the input (the contract documented on apply_anthropic_cache_markers).
        assert new_tools is not None
        assert new_tools[-1]["function"] is original_function, (
            "The marker transform must not deep-copy the inner function dict. "
            "If this is intentional, document the cost at the call site and "
            "update the helper docstring."
        )


class TestToolListOrderIsCacheKeyContract:
    """Lock down the order contract that ``apply_anthropic_cache_markers``
    depends on (post-review concern #2).

    Anthropic prompt caching uses ``cache_control`` markers as cache
    breakpoints — content up to and including the marker becomes the cache
    key. The transform places the marker on the LAST tool, which means the
    SET of cached tools is "all of them" — provided ``get_tool_definitions()``
    returns the same order on every call within a session.

    A future maintainer who alphabetizes the tool list (or reorders for
    readability) silently changes the cache key for every follow-up turn,
    yielding 100% cache misses without any visible breakage. These tests
    fire loudly when that happens.
    """

    def test_tool_definitions_are_order_stable_across_calls(self) -> None:
        from elspeth.web.composer.tools import get_tool_definitions

        first = [d["name"] for d in get_tool_definitions()]
        second = [d["name"] for d in get_tool_definitions()]
        assert first == second, (
            "get_tool_definitions() must return tools in the same order on every call. "
            "The Anthropic cache_control marker is placed on the LAST tool — a different "
            "trailing tool between calls invalidates the prompt cache."
        )

    def test_litellm_tools_preserves_definition_order(self) -> None:
        """``_get_litellm_tools`` is a list comprehension over ``get_tool_definitions``;
        order must be preserved (modulo the advisor-toggle filter, which is
        allowed to drop ``request_advisor_hint`` when disabled but must never
        reorder the remaining tools).

        The cache-key invariant is "the relative order of tools that DO appear
        in ``_get_litellm_tools`` matches the relative order in
        ``get_tool_definitions``." Set inequality (one filtered out) is fine;
        order swap (cache-invalidating reorder) is not.
        """
        from elspeth.web.composer.service import ComposerServiceImpl
        from elspeth.web.composer.tools import get_tool_definitions
        from tests.unit.web.composer._helpers import _make_settings, _mock_catalog

        catalog = _mock_catalog()
        settings = _make_settings()
        service = ComposerServiceImpl.for_trained_operator(catalog=catalog, settings=settings)

        defn_names = [d["name"] for d in get_tool_definitions()]
        tool_names = [t["function"]["name"] for t in service._get_litellm_tools()]

        # Subsequence-order invariant: every tool emitted is in the definition
        # list, and the indices form a strictly increasing sequence (i.e., no
        # reordering relative to definitions).
        defn_index = {name: i for i, name in enumerate(defn_names)}
        emitted_positions = [defn_index[name] for name in tool_names]
        assert emitted_positions == sorted(emitted_positions), (
            f"_get_litellm_tools reordered tools relative to get_tool_definitions; "
            f"emitted positions={emitted_positions}, names={tool_names}"
        )
        # Every emitted name must come from definitions (no tool fabricated
        # by the filter).
        assert set(tool_names).issubset(set(defn_names))

    def test_trailing_tool_name_is_locked(self) -> None:
        """Lock the trailing tool's NAME so a reorder of ``get_tool_definitions()``
        breaks this test rather than silently invalidating Anthropic's prompt cache.

        If you intentionally need to change the trailing tool, update both this
        test AND the call-site comment in ``apply_anthropic_cache_markers`` —
        and consider the cache-miss cost on the next deploy.
        """
        from elspeth.web.composer.tools import get_tool_definitions

        trailing = get_tool_definitions()[-1]["name"]
        assert trailing == "wire_secret_ref", (
            f"Trailing tool changed from 'wire_secret_ref' to {trailing!r}. "
            "Anthropic cache_control markers go on the trailing tool — reordering "
            "invalidates the prompt cache for every follow-up turn until the new "
            "trailing-tool prefix warms up. If intentional, update this test."
        )


class TestCacheMarkersWiredAtCallSite:
    """Integration tests for the call-site wiring inside ``_call_llm_with_audit``.

    Verifies the transform fires once per call for Anthropic models,
    does NOT fire for OpenAI models, and that the messages_hash on the
    audit record reflects what was actually sent to LiteLLM.
    """

    @pytest.mark.asyncio
    async def test_anthropic_model_emits_cache_control_to_litellm(self) -> None:
        from collections.abc import Mapping
        from dataclasses import dataclass
        from unittest.mock import patch

        from elspeth.web.composer.service import (
            ComposerAvailability,
            ComposerServiceImpl,
        )
        from tests.unit.web.composer._helpers import (
            FakeChoice,
            _empty_state,
            _make_llm_response,
            _make_settings,
            _mock_catalog,
        )

        @dataclass
        class _Resp:
            choices: list[FakeChoice]
            usage: Mapping[str, Any]
            model: str = "anthropic/claude-sonnet-4.5"
            id: str = "msg_test"

        catalog = _mock_catalog()
        settings = _make_settings(composer_model="anthropic/claude-sonnet-4.5")
        service = ComposerServiceImpl.for_trained_operator(catalog=catalog, settings=settings)
        # Bypass availability check (no real Anthropic API key needed).
        service._availability = ComposerAvailability(available=True, model=service._model, provider="test")
        state = _empty_state()
        captured: dict[str, Any] = {}

        text_response = _make_llm_response(content="Done.")
        anthropic_response = _Resp(choices=text_response.choices, usage={"prompt_tokens": 10, "completion_tokens": 2})

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return anthropic_response

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new=fake_acompletion,
        ):
            result = await service.compose("Build a CSV pipeline.", [], state)

        # The stable system prompt MUST carry cache_control after the transform.
        sent_messages = captured["messages"]
        system_messages = [m for m in sent_messages if m.get("role") == "system"]
        assert len(system_messages) == 1
        stable_system_msg = system_messages[0]
        dynamic_context_msg = sent_messages[1]
        assert stable_system_msg["cache_control"] == {"type": "ephemeral"}
        assert "Current pipeline state" not in stable_system_msg["content"]
        assert dynamic_context_msg["role"] == "user"
        assert dynamic_context_msg["content"].startswith("Current pipeline state and available plugins")
        assert "UNTRUSTED DATA" in dynamic_context_msg["content"]
        assert "cache_control" not in dynamic_context_msg

        # The trailing tool MUST carry cache_control after the transform.
        sent_tools = captured["tools"]
        assert sent_tools[-1]["cache_control"] == {"type": "ephemeral"}
        # Other tools are NOT marked (Anthropic caches up to and including the marker).
        for non_trailing in sent_tools[:-1]:
            assert "cache_control" not in non_trailing

        transmitted_names = tuple(tool["function"]["name"] for tool in sent_tools)
        call = result.llm_calls[0]
        assert "splice_transform" in transmitted_names
        assert call.declared_tool_names == transmitted_names
        assert call.tools_spec_hash == stable_hash(sent_tools)

    @pytest.mark.asyncio
    async def test_openai_model_does_not_emit_cache_control(self) -> None:
        """OpenAI/OpenRouter use automatic prefix caching; markers would be
        ignored on the wire and would needlessly grow the request body and
        ``messages_hash`` digest. Verify the transform is skipped.
        """
        from collections.abc import Mapping
        from dataclasses import dataclass
        from unittest.mock import patch

        from elspeth.web.composer.service import (
            ComposerAvailability,
            ComposerServiceImpl,
        )
        from tests.unit.web.composer._helpers import (
            FakeChoice,
            _empty_state,
            _make_llm_response,
            _make_settings,
            _mock_catalog,
        )

        @dataclass
        class _Resp:
            choices: list[FakeChoice]
            usage: Mapping[str, Any]
            model: str = "openrouter/openai/gpt-5.5"
            id: str = "chatcmpl_test"

        catalog = _mock_catalog()
        # Default _make_settings model is gpt-5.5 (OpenAI-shape).
        settings = _make_settings()
        service = ComposerServiceImpl.for_trained_operator(catalog=catalog, settings=settings)
        service._availability = ComposerAvailability(available=True, model=service._model, provider="test")
        state = _empty_state()
        captured: dict[str, Any] = {}

        text_response = _make_llm_response(content="Done.")
        oai_response = _Resp(choices=text_response.choices, usage={"prompt_tokens": 10, "completion_tokens": 2})

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return oai_response

        with patch(
            "elspeth.web.composer.service._litellm_acompletion",
            new=fake_acompletion,
        ):
            await service.compose("Build a CSV pipeline.", [], state)

        sent_messages = captured["messages"]
        system_messages = [m for m in sent_messages if m.get("role") == "system"]
        assert len(system_messages) == 1
        for system_msg in system_messages:
            assert "cache_control" not in system_msg
        assert sent_messages[1]["role"] == "user"
        assert sent_messages[1]["content"].startswith("Current pipeline state and available plugins")
        assert "UNTRUSTED DATA" in sent_messages[1]["content"]
        assert "cache_control" not in sent_messages[1]

        sent_tools = captured["tools"]
        for tool in sent_tools:
            assert "cache_control" not in tool
