"""Tests for TokenUsage frozen dataclass."""

import pytest

from elspeth.contracts.token_usage import TokenUsage


class TestTokenUsageFactories:
    """Tests for known(), unknown(), and from_dict() factories."""

    def test_known_factory(self) -> None:
        usage = TokenUsage.known(10, 20)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20

    def test_unknown_factory(self) -> None:
        usage = TokenUsage.unknown()
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None

    def test_partial_known_prompt_only(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=None)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens is None

    def test_partial_known_completion_only(self) -> None:
        usage = TokenUsage(prompt_tokens=None, completion_tokens=20)
        assert usage.prompt_tokens is None
        assert usage.completion_tokens == 20

    def test_default_is_unknown(self) -> None:
        usage = TokenUsage()
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None

    def test_known_rejects_negative_prompt_tokens(self) -> None:
        """Negative token counts are physically impossible."""
        import pytest

        with pytest.raises(ValueError, match="prompt_tokens must be >= 0"):
            TokenUsage.known(-1, 20)

    def test_known_rejects_negative_completion_tokens(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="completion_tokens must be >= 0"):
            TokenUsage.known(10, -5)

    def test_direct_construction_rejects_negative(self) -> None:
        """Direct construction also validates (not just factories)."""
        import pytest

        with pytest.raises(ValueError, match="prompt_tokens must be >= 0"):
            TokenUsage(prompt_tokens=-100, completion_tokens=None)

    def test_zero_tokens_accepted(self) -> None:
        """Zero is valid (cached responses may report 0 completion tokens)."""
        usage = TokenUsage.known(0, 0)
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


class TestRequireIntValidation:
    """require_int guards reject bool (and wrong types) on int fields."""

    def test_rejects_bool_prompt_tokens(self) -> None:
        import pytest

        with pytest.raises(TypeError, match="prompt_tokens must be int"):
            TokenUsage(prompt_tokens=True, completion_tokens=None)

    def test_rejects_bool_completion_tokens(self) -> None:
        import pytest

        with pytest.raises(TypeError, match="completion_tokens must be int"):
            TokenUsage(prompt_tokens=None, completion_tokens=False)


class TestTokenUsageProperties:
    """Tests for total_tokens and is_known derived properties."""

    def test_total_tokens_known(self) -> None:
        usage = TokenUsage.known(10, 20)
        assert usage.total_tokens == 30

    def test_total_tokens_unknown(self) -> None:
        usage = TokenUsage.unknown()
        assert usage.total_tokens is None

    def test_total_tokens_partial_prompt_only(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=None)
        assert usage.total_tokens is None

    def test_total_tokens_partial_completion_only(self) -> None:
        usage = TokenUsage(prompt_tokens=None, completion_tokens=20)
        assert usage.total_tokens is None

    def test_total_tokens_zero(self) -> None:
        usage = TokenUsage.known(0, 0)
        assert usage.total_tokens == 0

    def test_is_known_true(self) -> None:
        usage = TokenUsage.known(10, 20)
        assert usage.is_known is True

    def test_is_known_false_unknown(self) -> None:
        usage = TokenUsage.unknown()
        assert usage.is_known is False

    def test_is_known_false_partial(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=None)
        assert usage.is_known is False

    def test_is_known_true_with_zeros(self) -> None:
        usage = TokenUsage.known(0, 0)
        assert usage.is_known is True


class TestTokenUsageToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_full(self) -> None:
        usage = TokenUsage.known(10, 20)
        assert usage.to_dict() == {"prompt_tokens": 10, "completion_tokens": 20}

    def test_to_dict_empty(self) -> None:
        usage = TokenUsage.unknown()
        assert usage.to_dict() == {}

    def test_to_dict_partial_prompt_only(self) -> None:
        usage = TokenUsage(prompt_tokens=10, completion_tokens=None)
        assert usage.to_dict() == {"prompt_tokens": 10}

    def test_to_dict_partial_completion_only(self) -> None:
        usage = TokenUsage(prompt_tokens=None, completion_tokens=20)
        assert usage.to_dict() == {"completion_tokens": 20}

    def test_to_dict_zero_values(self) -> None:
        """Zero is a valid known value, not unknown."""
        usage = TokenUsage.known(0, 0)
        assert usage.to_dict() == {"prompt_tokens": 0, "completion_tokens": 0}


class TestTokenUsageFromDict:
    """Tests for from_dict() Tier 3 boundary reconstruction."""

    def test_from_dict_full(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 20})
        assert usage == TokenUsage.known(10, 20)

    def test_from_dict_empty(self) -> None:
        usage = TokenUsage.from_dict({})
        assert usage == TokenUsage.unknown()

    def test_from_dict_none_input(self) -> None:
        usage = TokenUsage.from_dict(None)
        assert usage == TokenUsage.unknown()

    def test_from_dict_non_dict_input(self) -> None:
        usage = TokenUsage.from_dict("not a dict")
        assert usage == TokenUsage.unknown()

    def test_from_dict_non_int_prompt_tokens(self) -> None:
        """Non-int values should be coerced to None."""
        usage = TokenUsage.from_dict({"prompt_tokens": "10", "completion_tokens": 20})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens == 20

    def test_from_dict_non_int_completion_tokens(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 3.5})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens is None

    def test_from_dict_null_values(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": None, "completion_tokens": None})
        assert usage == TokenUsage.unknown()

    def test_from_dict_total_only_preserved(self) -> None:
        """Provider-reported total_tokens without breakdown is preserved, not dropped."""
        usage = TokenUsage.from_dict({"total_tokens": 30})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.reported_total == 30
        assert usage.total_tokens == 30  # property falls back to reported_total

    def test_from_dict_extra_keys_ignored(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "extra": "ignored"})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.reported_total == 30
        assert usage.total_tokens == 30  # computed sum matches reported

    def test_from_dict_bool_coerced_to_none(self) -> None:
        """bool is subclass of int in Python, but True/False are not valid token counts."""
        usage = TokenUsage.from_dict({"prompt_tokens": True, "completion_tokens": False})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None

    def test_from_dict_bool_prompt_with_valid_completion(self) -> None:
        """Bool in one field shouldn't affect valid int in the other."""
        usage = TokenUsage.from_dict({"prompt_tokens": True, "completion_tokens": 20})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens == 20

    def test_from_dict_negative_prompt_coerced_to_none(self) -> None:
        """Negative ints are invalid at Tier 3 boundary — coerce, don't crash."""
        usage = TokenUsage.from_dict({"prompt_tokens": -5, "completion_tokens": 20})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens == 20

    def test_from_dict_negative_completion_coerced_to_none(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": -1})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens is None

    def test_from_dict_negative_total_coerced_to_none(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": -100})
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.reported_total is None

    def test_from_dict_all_negative_returns_unknown(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": -1, "completion_tokens": -2, "total_tokens": -3})
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.reported_total is None


class TestTokenUsageImmutability:
    """Tests for frozen dataclass invariants."""

    def test_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        usage = TokenUsage.known(10, 20)
        with pytest.raises(FrozenInstanceError):
            usage.prompt_tokens = 99  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TokenUsage.known(10, 20)
        b = TokenUsage.known(10, 20)
        assert a == b

    def test_inequality(self) -> None:
        a = TokenUsage.known(10, 20)
        b = TokenUsage.known(10, 30)
        assert a != b

    def test_hashable(self) -> None:
        usage = TokenUsage.known(10, 20)
        # Frozen dataclass with slots=True is hashable
        s = {usage}
        assert len(s) == 1

    def test_hash_consistency(self) -> None:
        a = TokenUsage.known(10, 20)
        b = TokenUsage.known(10, 20)
        assert hash(a) == hash(b)


class TestTokenUsageRoundTrip:
    """Tests for to_dict() → from_dict() round-trip."""

    def test_round_trip_known(self) -> None:
        original = TokenUsage.known(10, 20)
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_unknown(self) -> None:
        original = TokenUsage.unknown()
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_partial(self) -> None:
        original = TokenUsage(prompt_tokens=10, completion_tokens=None)
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored == original

    def test_json_round_trip_known(self) -> None:
        """Round-trip through JSON serialization preserves known token counts.

        This is the real persistence path: to_dict → json.dumps → json.loads → from_dict.
        """
        import json

        original = TokenUsage.known(150, 42)
        serialized = json.dumps(original.to_dict())
        deserialized = json.loads(serialized)
        restored = TokenUsage.from_dict(deserialized)
        assert restored == original

    def test_json_round_trip_unknown(self) -> None:
        """JSON round-trip for fully unknown usage (empty dict)."""
        import json

        original = TokenUsage.unknown()
        serialized = json.dumps(original.to_dict())
        deserialized = json.loads(serialized)
        restored = TokenUsage.from_dict(deserialized)
        assert restored == original

    def test_json_round_trip_partial(self) -> None:
        """JSON round-trip for partial usage (one field known, one unknown)."""
        import json

        original = TokenUsage(prompt_tokens=10, completion_tokens=None)
        serialized = json.dumps(original.to_dict())
        deserialized = json.loads(serialized)
        restored = TokenUsage.from_dict(deserialized)
        assert restored == original


class TestTokenUsageProviderCacheFields:
    """Cache-token capture across provider shapes (elspeth-4e79436719 Bug C).

    OpenAI / OpenRouter expose cache hits via the nested
    ``prompt_tokens_details.cached_tokens`` field. Anthropic exposes
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` as
    sibling fields on the usage object. Both shapes must land on the
    canonical ``TokenUsage`` fields so the audit row records what the
    provider actually reported — never fabricating zero when absent.
    """

    def test_openai_nested_cached_tokens_shape(self) -> None:
        usage = TokenUsage.from_dict(
            {
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "total_tokens": 1280,
                "prompt_tokens_details": {"cached_tokens": 1024},
            }
        )
        assert usage.prompt_tokens == 1200
        assert usage.cached_prompt_tokens == 1024
        # Anthropic-shape fields stay None when only OpenAI shape is present.
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_anthropic_sibling_cache_fields_shape(self) -> None:
        usage = TokenUsage.from_dict(
            {
                "prompt_tokens": 8200,
                "completion_tokens": 120,
                "cache_creation_input_tokens": 7000,
                "cache_read_input_tokens": 1100,
            }
        )
        assert usage.cache_creation_input_tokens == 7000
        assert usage.cache_read_input_tokens == 1100
        # OpenAI-shape field stays None when only Anthropic shape is present.
        assert usage.cached_prompt_tokens is None

    @pytest.mark.parametrize(
        ("field_name", "expected_payload"),
        [
            ("cached_prompt_tokens", {"cached_prompt_tokens": 1024}),
            ("cache_creation_input_tokens", {"cache_creation_input_tokens": 7000}),
            ("cache_read_input_tokens", {"cache_read_input_tokens": 1100}),
        ],
    )
    def test_cache_only_usage_has_data(self, field_name: str, expected_payload: dict[str, int]) -> None:
        usage = TokenUsage(**{field_name: next(iter(expected_payload.values()))})

        assert usage.has_data is True
        assert usage.to_dict() == expected_payload

    def test_canonical_top_level_cached_prompt_tokens_round_trip(self) -> None:
        """``to_dict`` emits ``cached_prompt_tokens`` at top level — round-trip preserves it."""
        original = TokenUsage(
            prompt_tokens=1200,
            completion_tokens=80,
            cached_prompt_tokens=1024,
        )
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored == original

    def test_anthropic_round_trip(self) -> None:
        original = TokenUsage(
            prompt_tokens=8200,
            completion_tokens=120,
            cache_creation_input_tokens=7000,
            cache_read_input_tokens=1100,
        )
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored == original

    def test_no_cache_fields_returns_none(self) -> None:
        usage = TokenUsage.from_dict({"prompt_tokens": 100, "completion_tokens": 20})
        assert usage.cached_prompt_tokens is None
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None


class TestTokenUsageReasoningTokens:
    """Provider-reported reasoning-token counters are audit data, not inferred data."""

    def test_reasoning_tokens_round_trip_when_provider_reports_top_level_counter(self) -> None:
        original = TokenUsage(prompt_tokens=100, completion_tokens=20, reasoning_tokens=12)

        payload = original.to_dict()
        restored = TokenUsage.from_dict(payload)

        assert payload["reasoning_tokens"] == 12
        assert restored.reasoning_tokens == 12

    def test_reasoning_tokens_extract_from_openai_completion_details_shape(self) -> None:
        usage = TokenUsage.from_dict(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 12},
            }
        )

        assert usage.reasoning_tokens == 12

    @pytest.mark.parametrize(
        "raw_value",
        [None, -1, True, 2.5, "12"],
    )
    def test_reasoning_tokens_invalid_external_values_become_unknown(self, raw_value: object) -> None:
        usage = TokenUsage.from_dict({"reasoning_tokens": raw_value})

        assert usage.reasoning_tokens is None

    def test_negative_cached_tokens_coerced_to_none(self) -> None:
        """Tier-3 boundary: negative values mean broken provider, not negative cache."""
        usage = TokenUsage.from_dict(
            {
                "prompt_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": -5},
                "cache_creation_input_tokens": -1,
                "cache_read_input_tokens": "not-an-int",
            }
        )
        assert usage.cached_prompt_tokens is None
        assert usage.cache_creation_input_tokens is None
        assert usage.cache_read_input_tokens is None

    def test_non_mapping_prompt_tokens_details_ignored(self) -> None:
        """If prompt_tokens_details is a string or list, treat as no cache info."""
        usage = TokenUsage.from_dict({"prompt_tokens": 100, "prompt_tokens_details": "garbage"})
        assert usage.cached_prompt_tokens is None

    def test_to_dict_omits_unset_cache_fields(self) -> None:
        usage = TokenUsage.known(100, 20)
        d = usage.to_dict()
        assert "cached_prompt_tokens" not in d
        assert "cache_creation_input_tokens" not in d
        assert "cache_read_input_tokens" not in d

    def test_negative_cached_prompt_tokens_rejected_by_post_init(self) -> None:
        """Direct construction (Tier 1) refuses negative values."""
        import pytest

        with pytest.raises(ValueError, match="cached_prompt_tokens must be >= 0"):
            TokenUsage(prompt_tokens=100, cached_prompt_tokens=-1)
