# tests/testing/chaosllm/test_response_generator.py
"""Tests for ChaosLLM response generator."""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from elspeth.testing.chaosllm.config import (
    PresetResponseConfig,
    RandomResponseConfig,
    ResponseConfig,
    TemplateResponseConfig,
)
from elspeth.testing.chaosllm.response_generator import (
    ENGLISH_VOCABULARY,
    LOREM_VOCABULARY,
    OpenAIResponse,
    PresetBank,
    ResponseGenerator,
)


class FixedRandom(random.Random):
    """A Random instance that returns fixed values for testing."""

    def __init__(self, value: float = 0.5, int_value: int = 50) -> None:
        super().__init__()
        self._fixed_value = value
        self._fixed_int = int_value

    def random(self) -> float:
        return self._fixed_value

    def randint(self, a: int, b: int) -> int:
        # Clamp to valid range
        return max(a, min(b, self._fixed_int))

    def choice(self, seq: Any) -> Any:
        return seq[0]  # Always return first element


class TestOpenAIResponse:
    """Tests for OpenAIResponse dataclass."""

    def test_basic_response(self) -> None:
        """OpenAIResponse has correct fields."""
        response = OpenAIResponse(
            id="fake-12345",
            object="chat.completion",
            created=1706644800,
            model="gpt-4",
            content="Test response content",
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason="stop",
        )
        assert response.id == "fake-12345"
        assert response.model == "gpt-4"
        assert response.content == "Test response content"

    def test_total_tokens(self) -> None:
        """total_tokens property sums prompt and completion."""
        response = OpenAIResponse(
            id="fake-12345",
            object="chat.completion",
            created=1706644800,
            model="gpt-4",
            content="Test",
            prompt_tokens=10,
            completion_tokens=25,
            finish_reason="stop",
        )
        assert response.total_tokens == 35

    def test_to_dict_format(self) -> None:
        """to_dict returns OpenAI-compatible format."""
        response = OpenAIResponse(
            id="fake-12345",
            object="chat.completion",
            created=1706644800,
            model="gpt-4",
            content="Generated response here",
            prompt_tokens=10,
            completion_tokens=25,
            finish_reason="stop",
        )

        result = response.to_dict()

        assert result == {
            "id": "fake-12345",
            "object": "chat.completion",
            "created": 1706644800,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Generated response here",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 25,
                "total_tokens": 35,
            },
        }

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict result can be serialized to JSON."""
        response = OpenAIResponse(
            id="fake-abc123",
            object="chat.completion",
            created=1706644800,
            model="gpt-4-turbo",
            content="Hello, world!",
            prompt_tokens=5,
            completion_tokens=3,
            finish_reason="stop",
        )

        # Should not raise
        json_str = json.dumps(response.to_dict())
        assert "fake-abc123" in json_str
        assert "gpt-4-turbo" in json_str


class TestPresetBank:
    """Tests for PresetBank class."""

    def test_random_selection(self) -> None:
        """Random selection picks from responses."""
        bank = PresetBank(["a", "b", "c"], "random")
        responses = {bank.next() for _ in range(50)}
        # Should see some variation
        assert len(responses) >= 2

    def test_sequential_selection(self) -> None:
        """Sequential selection cycles through responses."""
        bank = PresetBank(["first", "second", "third"], "sequential")

        assert bank.next() == "first"
        assert bank.next() == "second"
        assert bank.next() == "third"
        # Should cycle back
        assert bank.next() == "first"
        assert bank.next() == "second"

    def test_reset_sequential(self) -> None:
        """Reset restarts sequential from beginning."""
        bank = PresetBank(["a", "b", "c"], "sequential")

        bank.next()  # a
        bank.next()  # b
        bank.reset()
        assert bank.next() == "a"

    def test_deterministic_with_seeded_random(self) -> None:
        """PresetBank with seeded random is deterministic."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        bank1 = PresetBank(["a", "b", "c", "d", "e"], "random", rng=rng1)
        bank2 = PresetBank(["a", "b", "c", "d", "e"], "random", rng=rng2)

        for _ in range(20):
            assert bank1.next() == bank2.next()

    def test_empty_responses_raises(self) -> None:
        """PresetBank requires at least one response."""
        with pytest.raises(ValueError, match="at least one response"):
            PresetBank([], "random")

    def test_from_jsonl(self, tmp_path: Path) -> None:
        """Load PresetBank from JSONL file."""
        jsonl_file = tmp_path / "responses.jsonl"
        jsonl_file.write_text('{"content": "Response 1"}\n{"content": "Response 2"}\n{"content": "Response 3"}\n')

        bank = PresetBank.from_jsonl(jsonl_file, "sequential")
        assert bank.next() == "Response 1"
        assert bank.next() == "Response 2"
        assert bank.next() == "Response 3"

    def test_from_jsonl_skips_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines in JSONL are skipped."""
        jsonl_file = tmp_path / "responses.jsonl"
        jsonl_file.write_text('{"content": "First"}\n\n{"content": "Second"}\n   \n{"content": "Third"}\n')

        bank = PresetBank.from_jsonl(jsonl_file, "sequential")
        assert bank.next() == "First"
        assert bank.next() == "Second"
        assert bank.next() == "Third"

    def test_from_jsonl_file_not_found(self, tmp_path: Path) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Preset file not found"):
            PresetBank.from_jsonl(tmp_path / "nonexistent.jsonl", "random")

    def test_from_jsonl_invalid_json(self, tmp_path: Path) -> None:
        """Invalid JSON line raises ValueError."""
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text('{"content": "ok"}\nnot valid json\n')

        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            PresetBank.from_jsonl(jsonl_file, "random")

    def test_from_jsonl_not_object(self, tmp_path: Path) -> None:
        """Non-object JSON raises ValueError."""
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text('["array", "not", "object"]\n')

        with pytest.raises(ValueError, match="must be a JSON object"):
            PresetBank.from_jsonl(jsonl_file, "random")

    def test_from_jsonl_missing_content(self, tmp_path: Path) -> None:
        """Missing content field raises ValueError."""
        jsonl_file = tmp_path / "bad.jsonl"
        jsonl_file.write_text('{"other_field": "value"}\n')

        with pytest.raises(ValueError, match="missing required 'content' field"):
            PresetBank.from_jsonl(jsonl_file, "random")

    def test_from_jsonl_empty_file(self, tmp_path: Path) -> None:
        """Empty file raises ValueError."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        with pytest.raises(ValueError, match="contains no valid responses"):
            PresetBank.from_jsonl(jsonl_file, "random")


class TestResponseGeneratorBasic:
    """Basic tests for ResponseGenerator."""

    def test_default_config(self) -> None:
        """ResponseGenerator works with default config."""
        config = ResponseConfig()
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        response = generator.generate(request)

        assert response.id.startswith("fake-")
        assert response.object == "chat.completion"
        assert response.model == "gpt-4"
        assert response.finish_reason == "stop"
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0

    def test_deterministic_with_fixed_deps(self) -> None:
        """Generator is deterministic with fixed dependencies."""
        config = ResponseConfig(mode="random")

        # Fixed time and UUID
        fixed_time = 1706644800.0
        fixed_uuid = "deadbeef12345678"

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        gen1 = ResponseGenerator(
            config,
            time_func=lambda: fixed_time,
            rng=rng1,
            uuid_func=lambda: fixed_uuid,
        )
        gen2 = ResponseGenerator(
            config,
            time_func=lambda: fixed_time,
            rng=rng2,
            uuid_func=lambda: fixed_uuid,
        )

        request = {"model": "gpt-4", "messages": [{"role": "user", "content": "Test"}]}

        r1 = gen1.generate(request)
        r2 = gen2.generate(request)

        assert r1.id == r2.id
        assert r1.content == r2.content
        assert r1.created == r2.created


class TestRandomMode:
    """Tests for random response generation mode."""

    def test_random_mode_generates_text(self) -> None:
        """Random mode generates text content."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert len(response.content) > 0
        # Should end with period
        assert response.content.endswith(".")

    def test_word_count_in_range(self) -> None:
        """Random text word count is within configured range."""
        config = ResponseConfig(
            mode="random",
            random=RandomResponseConfig(min_words=5, max_words=10),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        for _ in range(20):
            response = generator.generate(request)
            # Remove period and count words
            words = response.content.rstrip(".").split()
            assert 5 <= len(words) <= 10

    def test_english_vocabulary(self) -> None:
        """English vocabulary uses common English words."""
        config = ResponseConfig(
            mode="random",
            random=RandomResponseConfig(vocabulary="english", min_words=20, max_words=20),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        # All words (lowercased, without punctuation) should be in vocabulary
        words = response.content.rstrip(".").lower().split()
        for word in words:
            assert word in ENGLISH_VOCABULARY, f"'{word}' not in ENGLISH_VOCABULARY"

    def test_lorem_vocabulary(self) -> None:
        """Lorem vocabulary uses Lorem Ipsum words."""
        config = ResponseConfig(
            mode="random",
            random=RandomResponseConfig(vocabulary="lorem", min_words=20, max_words=20),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        # All words (lowercased, without punctuation) should be in vocabulary
        words = response.content.rstrip(".").lower().split()
        for word in words:
            assert word in LOREM_VOCABULARY, f"'{word}' not in LOREM_VOCABULARY"

    def test_first_word_capitalized(self) -> None:
        """First word of random text is capitalized."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        for _ in range(10):
            response = generator.generate(request)
            first_char = response.content[0]
            assert first_char.isupper(), f"First char '{first_char}' should be uppercase"


class TestTemplateMode:
    """Tests for template response generation mode."""

    def test_static_template(self) -> None:
        """Static template returns literal content."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body='{"result": "success"}'),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.content == '{"result": "success"}'

    def test_template_with_request_context(self) -> None:
        """Template can access request context."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="Model: {{ model }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4-turbo", "messages": []}
        response = generator.generate(request)

        assert response.content == "Model: gpt-4-turbo"

    def test_template_with_messages(self) -> None:
        """Template can access messages list."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="Got {{ messages | length }} messages"),
        )
        generator = ResponseGenerator(config)

        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        response = generator.generate(request)

        assert response.content == "Got 3 messages"

    def test_random_choice_helper(self) -> None:
        """Template random_choice helper works."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ random_choice('yes', 'no', 'maybe') }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        results = {generator.generate(request).content for _ in range(50)}

        # Should see some variation
        assert len(results) >= 2
        assert results <= {"yes", "no", "maybe"}

    def test_random_float_helper(self) -> None:
        """Template random_float helper generates floats in range."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ random_float(0.0, 1.0) }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        for _ in range(20):
            response = generator.generate(request)
            value = float(response.content)
            assert 0.0 <= value <= 1.0

    def test_random_int_helper(self) -> None:
        """Template random_int helper generates integers in range."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ random_int(1, 10) }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        for _ in range(20):
            response = generator.generate(request)
            value = int(response.content)
            assert 1 <= value <= 10

    def test_random_words_helper(self) -> None:
        """Template random_words helper generates words."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ random_words(5, 'english') }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        words = response.content.split()
        assert len(words) == 5

    def test_timestamp_helper(self) -> None:
        """Template timestamp helper returns current time."""
        fixed_time = 1706644800.0
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ timestamp() }}"),
        )
        generator = ResponseGenerator(config, time_func=lambda: fixed_time)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.content == "1706644800"

    def test_template_override(self) -> None:
        """Template override replaces configured template."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="Original template"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request, template_override="Overridden: {{ model }}")

        assert response.content == "Overridden: gpt-4"

    def test_template_undefined_variable_raises(self) -> None:
        """Template with undefined variable raises error."""
        import jinja2

        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="{{ undefined_var }}"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        with pytest.raises(jinja2.UndefinedError):
            generator.generate(request)


class TestEchoMode:
    """Tests for echo response generation mode."""

    def test_echo_last_user_message(self) -> None:
        """Echo mode returns last user message."""
        config = ResponseConfig(mode="echo")
        generator = ResponseGenerator(config)

        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        }
        response = generator.generate(request)

        assert response.content == "Echo: Second question"

    def test_echo_no_messages(self) -> None:
        """Echo with no messages returns placeholder."""
        config = ResponseConfig(mode="echo")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.content == "Echo: (no messages provided)"

    def test_echo_no_user_messages(self) -> None:
        """Echo with no user messages falls back to last message."""
        config = ResponseConfig(mode="echo")
        generator = ResponseGenerator(config)

        request = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "assistant", "content": "Ready to help"},
            ],
        }
        response = generator.generate(request)

        # Should use last message of any role
        assert response.content == "Echo: Ready to help"

    def test_echo_truncates_long_messages(self) -> None:
        """Echo truncates messages over 200 chars."""
        config = ResponseConfig(mode="echo")
        generator = ResponseGenerator(config)

        long_content = "a" * 300
        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": long_content}],
        }
        response = generator.generate(request)

        # Should be truncated to 200 chars + "..."
        assert len(response.content) == len("Echo: ") + 200 + len("...")
        assert response.content.endswith("...")


class TestPresetMode:
    """Tests for preset response generation mode."""

    def test_preset_mode(self, tmp_path: Path) -> None:
        """Preset mode returns responses from file."""
        jsonl_file = tmp_path / "responses.jsonl"
        jsonl_file.write_text('{"content": "Preset response 1"}\n{"content": "Preset response 2"}\n')

        config = ResponseConfig(
            mode="preset",
            preset=PresetResponseConfig(
                file=str(jsonl_file),
                selection="sequential",
            ),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        r1 = generator.generate(request)
        assert r1.content == "Preset response 1"

        r2 = generator.generate(request)
        assert r2.content == "Preset response 2"

    def test_preset_random_selection(self, tmp_path: Path) -> None:
        """Preset random selection picks varied responses."""
        jsonl_file = tmp_path / "responses.jsonl"
        jsonl_file.write_text('{"content": "A"}\n{"content": "B"}\n{"content": "C"}\n{"content": "D"}\n{"content": "E"}\n')

        config = ResponseConfig(
            mode="preset",
            preset=PresetResponseConfig(
                file=str(jsonl_file),
                selection="random",
            ),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        results = {generator.generate(request).content for _ in range(50)}

        # Should see some variation
        assert len(results) >= 2

    def test_preset_file_not_found(self, tmp_path: Path) -> None:
        """Preset with missing file raises error on first generate."""
        config = ResponseConfig(
            mode="preset",
            preset=PresetResponseConfig(
                file=str(tmp_path / "nonexistent.jsonl"),
                selection="random",
            ),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        with pytest.raises(FileNotFoundError):
            generator.generate(request)

    def test_preset_reset(self, tmp_path: Path) -> None:
        """Reset restarts preset sequential selection."""
        jsonl_file = tmp_path / "responses.jsonl"
        jsonl_file.write_text('{"content": "First"}\n{"content": "Second"}\n')

        config = ResponseConfig(
            mode="preset",
            preset=PresetResponseConfig(
                file=str(jsonl_file),
                selection="sequential",
            ),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        r1 = generator.generate(request)
        assert r1.content == "First"

        r2 = generator.generate(request)
        assert r2.content == "Second"

        generator.reset()

        r3 = generator.generate(request)
        assert r3.content == "First"


class TestModeOverride:
    """Tests for per-request mode override."""

    def test_mode_override_to_random(self) -> None:
        """Mode override switches from template to random."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="Fixed template"),
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}

        # Without override: template mode
        r1 = generator.generate(request)
        assert r1.content == "Fixed template"

        # With override: random mode
        r2 = generator.generate(request, mode_override="random")
        assert r2.content != "Fixed template"

    def test_mode_override_to_echo(self) -> None:
        """Mode override switches to echo mode."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello world"}],
        }

        response = generator.generate(request, mode_override="echo")
        assert response.content == "Echo: Hello world"


class TestTokenEstimation:
    """Tests for token count estimation."""

    def test_prompt_token_estimation(self) -> None:
        """Prompt tokens are estimated from messages."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        # Short prompt
        short_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        r1 = generator.generate(short_request)

        # Long prompt
        long_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "A" * 1000}],
        }
        r2 = generator.generate(long_request)

        # Long prompt should have more tokens
        assert r2.prompt_tokens > r1.prompt_tokens

    def test_completion_token_estimation(self) -> None:
        """Completion tokens are estimated from content."""
        # Generate short response
        config_short = ResponseConfig(
            mode="random",
            random=RandomResponseConfig(min_words=5, max_words=5),
        )
        gen_short = ResponseGenerator(config_short)

        # Generate long response
        config_long = ResponseConfig(
            mode="random",
            random=RandomResponseConfig(min_words=100, max_words=100),
        )
        gen_long = ResponseGenerator(config_long)

        request = {"model": "gpt-4", "messages": []}

        r_short = gen_short.generate(request)
        r_long = gen_long.generate(request)

        # Long response should have more completion tokens
        assert r_long.completion_tokens > r_short.completion_tokens

    def test_minimum_token_count(self) -> None:
        """Token count is at least 1."""
        config = ResponseConfig(
            mode="template",
            template=TemplateResponseConfig(body="Hi"),  # Very short
        )
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.completion_tokens >= 1


class TestResponseMetadata:
    """Tests for response metadata fields."""

    def test_response_id_format(self) -> None:
        """Response ID starts with 'fake-'."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.id.startswith("fake-")

    def test_response_object_type(self) -> None:
        """Response object type is 'chat.completion'."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.object == "chat.completion"

    def test_response_created_timestamp(self) -> None:
        """Response created timestamp matches time function."""
        fixed_time = 1706644800.0
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config, time_func=lambda: fixed_time)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.created == 1706644800

    def test_response_model_from_request(self) -> None:
        """Response model matches request model."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4-turbo-preview", "messages": []}
        response = generator.generate(request)

        assert response.model == "gpt-4-turbo-preview"

    def test_response_model_default(self) -> None:
        """Response model defaults to 'gpt-4'."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"messages": []}  # No model specified
        response = generator.generate(request)

        assert response.model == "gpt-4"

    def test_finish_reason_is_stop(self) -> None:
        """Finish reason is always 'stop'."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config)

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.finish_reason == "stop"

    def test_custom_uuid_func(self) -> None:
        """Custom UUID function is used for response ID."""
        config = ResponseConfig(mode="random")
        generator = ResponseGenerator(config, uuid_func=lambda: "custom-uuid-12345")

        request = {"model": "gpt-4", "messages": []}
        response = generator.generate(request)

        assert response.id == "fake-custom-uuid-12345"


class TestVocabularyConstants:
    """Tests for vocabulary constants."""

    def test_english_vocabulary_has_words(self) -> None:
        """English vocabulary is non-empty."""
        assert len(ENGLISH_VOCABULARY) > 50

    def test_lorem_vocabulary_has_words(self) -> None:
        """Lorem vocabulary is non-empty."""
        assert len(LOREM_VOCABULARY) > 50

    def test_vocabularies_are_tuples(self) -> None:
        """Vocabularies are immutable tuples."""
        assert isinstance(ENGLISH_VOCABULARY, tuple)
        assert isinstance(LOREM_VOCABULARY, tuple)

    def test_english_vocabulary_lowercase(self) -> None:
        """English vocabulary words are lowercase."""
        for word in ENGLISH_VOCABULARY:
            assert word == word.lower(), f"'{word}' should be lowercase"

    def test_lorem_vocabulary_lowercase(self) -> None:
        """Lorem vocabulary words are lowercase."""
        for word in LOREM_VOCABULARY:
            assert word == word.lower(), f"'{word}' should be lowercase"
