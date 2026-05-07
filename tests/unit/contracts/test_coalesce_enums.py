"""Tests for CoalescePolicy and MergeStrategy enums."""

from elspeth.contracts.coalesce_enums import CoalescePolicy, MergeStrategy


class TestCoalescePolicy:
    def test_values_match_config_literals(self) -> None:
        """Values must match the Literal strings in CoalesceSettings.policy."""
        assert CoalescePolicy.REQUIRE_ALL.value == "require_all"
        assert CoalescePolicy.QUORUM.value == "quorum"
        assert CoalescePolicy.BEST_EFFORT.value == "best_effort"
        assert CoalescePolicy.FIRST.value == "first"

    def test_round_trip_from_string(self) -> None:
        for member in CoalescePolicy:
            assert CoalescePolicy(member.value) is member

    def test_invalid_value_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            CoalescePolicy("nonexistent")


class TestMergeStrategy:
    def test_values_match_config_literals(self) -> None:
        assert MergeStrategy.UNION.value == "union"
        assert MergeStrategy.NESTED.value == "nested"
        assert MergeStrategy.SELECT.value == "select"

    def test_invalid_value_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            MergeStrategy("nonexistent")
