# tests/unit/plugins/llm/test_provider_protocol.py
"""Tests for LLMProvider protocol and DTOs."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.transforms.llm.provider import (
    FinishReason,
    LLMProvider,
    LLMQueryResult,
    UnrecognizedFinishReason,
    parse_finish_reason,
)


class TestLLMQueryResult:
    """Tests for the LLMQueryResult frozen dataclass."""

    def test_is_frozen(self) -> None:
        result = LLMQueryResult(
            content="hello",
            usage=TokenUsage.known(10, 5),
            model="gpt-4o",
        )
        with pytest.raises(FrozenInstanceError):
            result.content = "modified"  # type: ignore[misc]

    def test_fields(self) -> None:
        usage = TokenUsage.known(10, 5)
        result = LLMQueryResult(
            content="hello",
            usage=usage,
            model="gpt-4o",
            finish_reason=FinishReason.STOP,
        )
        assert result.content == "hello"
        assert result.usage is usage
        assert result.model == "gpt-4o"
        assert result.finish_reason == FinishReason.STOP
        # raw_response is NOT on LLMQueryResult
        assert not hasattr(result, "raw_response")

    def test_post_init_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError, match="content must be non-empty"):
            LLMQueryResult(
                content="",
                usage=TokenUsage.unknown(),
                model="gpt-4o",
            )

    def test_post_init_rejects_whitespace_content(self) -> None:
        with pytest.raises(ValueError, match="content must be non-empty"):
            LLMQueryResult(
                content="   ",
                usage=TokenUsage.unknown(),
                model="gpt-4o",
            )

    def test_post_init_rejects_empty_model(self) -> None:
        with pytest.raises(ValueError, match="model must be non-empty"):
            LLMQueryResult(
                content="hello",
                usage=TokenUsage.unknown(),
                model="",
            )

    def test_post_init_rejects_whitespace_model(self) -> None:
        with pytest.raises(ValueError, match="model must be non-empty"):
            LLMQueryResult(
                content="hello",
                usage=TokenUsage.unknown(),
                model="   ",
            )

    def test_finish_reason_defaults_to_none(self) -> None:
        result = LLMQueryResult(
            content="hello",
            usage=TokenUsage.unknown(),
            model="gpt-4o",
        )
        assert result.finish_reason is None

    def test_post_init_rejects_wrong_usage_type(self) -> None:
        """usage must be a TokenUsage instance, not a dict or other type.

        Bug: elspeth-42cb31ce6f. Without runtime validation, a caller
        passing a dict or None for usage succeeds at construction and
        explodes later when the transform accesses .prompt_tokens.
        """
        with pytest.raises(TypeError, match="usage"):
            LLMQueryResult(
                content="hello",
                usage={"prompt_tokens": 10, "completion_tokens": 5},  # type: ignore[arg-type]
                model="gpt-4o",
            )

    def test_post_init_rejects_none_usage(self) -> None:
        """usage=None must be rejected — TokenUsage.unknown() exists for that."""
        with pytest.raises(TypeError, match="usage"):
            LLMQueryResult(
                content="hello",
                usage=None,  # type: ignore[arg-type]
                model="gpt-4o",
            )

    def test_post_init_rejects_wrong_finish_reason_type(self) -> None:
        """finish_reason must be ParsedFinishReason, not a raw string.

        Bug: elspeth-42cb31ce6f. A raw string like "stop" bypasses the
        FinishReason enum and UnrecognizedFinishReason sentinel.
        """
        with pytest.raises(TypeError, match="finish_reason"):
            LLMQueryResult(
                content="hello",
                usage=TokenUsage.unknown(),
                model="gpt-4o",
                finish_reason="stop",  # type: ignore[arg-type]  # deliberate: tests rejection of raw string (must use FinishReason.STOP)
            )


class TestFinishReason:
    """Tests for FinishReason StrEnum."""

    def test_enum_values(self) -> None:
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.CONTENT_FILTER.value == "content_filter"
        assert FinishReason.TOOL_CALLS.value == "tool_calls"

    def test_from_string(self) -> None:
        assert FinishReason("stop") is FinishReason.STOP
        assert FinishReason("length") is FinishReason.LENGTH


class TestParseFinishReason:
    """Tests for parse_finish_reason helper."""

    def test_none_returns_none(self) -> None:
        assert parse_finish_reason(None) is None

    def test_valid_stop(self) -> None:
        assert parse_finish_reason("stop") is FinishReason.STOP

    def test_valid_length(self) -> None:
        assert parse_finish_reason("length") is FinishReason.LENGTH

    def test_valid_content_filter(self) -> None:
        assert parse_finish_reason("content_filter") is FinishReason.CONTENT_FILTER

    def test_valid_tool_calls(self) -> None:
        assert parse_finish_reason("tool_calls") is FinishReason.TOOL_CALLS

    def test_unknown_returns_unrecognized_sentinel(self) -> None:
        result = parse_finish_reason("end_turn")
        assert isinstance(result, UnrecognizedFinishReason)
        assert result.raw == "end_turn"

    def test_empty_string_returns_unrecognized_sentinel(self) -> None:
        result = parse_finish_reason("")
        assert isinstance(result, UnrecognizedFinishReason)
        assert result.raw == ""


class TestLLMProviderProtocol:
    """Tests for the LLMProvider protocol."""

    def test_is_runtime_checkable(self) -> None:
        # Verify @runtime_checkable allows isinstance checks.
        # Without the decorator, isinstance() would raise TypeError.
        assert not isinstance(object(), LLMProvider)

    def test_mock_provider_satisfies_protocol(self) -> None:
        class MockProvider:
            def execute_query(
                self,
                messages: list[dict[str, str]],
                *,
                model: str,
                temperature: float,
                max_tokens: int | None,
                state_id: str,
                token_id: str,
            ) -> LLMQueryResult:
                return LLMQueryResult(
                    content="test",
                    usage=TokenUsage.unknown(),
                    model=model,
                )

            def close(self) -> None:
                pass

        provider = MockProvider()
        assert isinstance(provider, LLMProvider)
