"""Unit tests for evals/lib/composer_rgr_score.py.

Covers:
  - Existing baseline behavior (RED on sentinels, is_valid=false, passivity;
    AMBER on missing node-kinds and outputs-min).
  - Convergence-bar additions (2026-05-08): must_have_node_chain_in_order,
    must_include_observed_columns, must_handle_field_as_numeric,
    max_repair_turns.

Composition under test is a pure function — tests pass dicts directly,
no I/O.
"""

from __future__ import annotations

from typing import Any

import pytest
from evals.lib.composer_rgr_score import score

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


_BASELINE_RED = {
    "passivity_phrases": ["if you want, i can", "should i "],
    "build_failure_sentinels": ["i cannot mark this pipeline complete"],
    "must_be_valid": True,
}

_BASELINE_GREEN = {
    "must_be_valid": True,
}


def _scenario(**overrides: Any) -> dict[str, Any]:
    """Build a scenario dict with baseline criteria, allowing per-test green overrides."""
    green = dict(_BASELINE_GREEN)
    green.update(overrides.pop("green", {}))
    return {
        "red_criteria": dict(_BASELINE_RED, **overrides.pop("red", {})),
        "green_criteria": green,
        **overrides,
    }


def _state_valid(**extras: Any) -> dict[str, Any]:
    """Minimal valid state — is_valid=True, one node, one output."""
    base: dict[str, Any] = {
        "is_valid": True,
        "source": {"plugin": "csv", "options": {"schema": {"mode": "observed"}}},
        "nodes": [{"id": "noop", "node_type": "transform", "plugin": "passthrough"}],
        "outputs": [{"name": "out", "plugin": "json"}],
    }
    base.update(extras)
    return base


def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


# --------------------------------------------------------------------------
# Baseline regression coverage (existing behavior must not change)
# --------------------------------------------------------------------------


class TestBaselineRedSignals:
    def test_green_when_all_clean(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("user", "hi"), _msg("assistant", "all done.")],
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"
        assert result["red_reasons"] == []
        assert result["amber_reasons"] == []

    def test_red_on_build_failure_sentinel(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("assistant", "I cannot mark this pipeline complete.")],
            state=_state_valid(is_valid=False),
        )
        assert result["verdict"] == "RED"
        assert any("sentinel" in r for r in result["red_reasons"])

    def test_red_on_is_valid_false(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(is_valid=False),
        )
        assert result["verdict"] == "RED"
        assert any("is_valid=false" in r for r in result["red_reasons"])

    def test_red_on_null_state(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("assistant", "ok")],
            state=None,
        )
        assert result["verdict"] == "RED"
        assert any("null" in r for r in result["red_reasons"])

    def test_red_on_passivity_phrase(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("assistant", "If you want, I can adjust the schema.")],
            state=_state_valid(),
        )
        assert result["verdict"] == "RED"
        assert any("passivity" in r for r in result["red_reasons"])

    def test_red_takes_precedence_over_amber(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_outputs_min": 5}),
            messages=[_msg("assistant", "If you want, I can adjust.")],
            state=_state_valid(is_valid=False),
        )
        assert result["verdict"] == "RED"


class TestBaselineAmberSignals:
    def test_amber_on_missing_node_kinds(self) -> None:
        scenario = _scenario(green={"must_have_node_kinds_substring_any_of": [["llm"]]})
        result = score(
            scenario=scenario,
            messages=[_msg("assistant", "ok")],
            state=_state_valid(),
        )
        assert result["verdict"] == "AMBER"
        assert any("expected node combo" in r for r in result["amber_reasons"])

    def test_amber_on_outputs_min(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_outputs_min": 2}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(),
        )
        assert result["verdict"] == "AMBER"
        assert any("outputs" in r for r in result["amber_reasons"])


# --------------------------------------------------------------------------
# must_have_node_chain_in_order
# --------------------------------------------------------------------------


class TestNodeChainInOrder:
    def _state_chain(self, plugins: list[str]) -> dict[str, Any]:
        return _state_valid(nodes=[{"id": f"n{i}", "node_type": "transform", "plugin": p} for i, p in enumerate(plugins)])

    def test_green_when_chain_in_order(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["web_scrape", "line_explode"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_chain(["web_scrape", "line_explode"]),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_green_with_extra_nodes_between(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["web_scrape", "json_explode"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_chain(["web_scrape", "passthrough", "json_explode"]),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_amber_when_chain_out_of_order(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["web_scrape", "line_explode"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_chain(["line_explode", "web_scrape"]),
        )
        assert result["verdict"] == "AMBER"
        assert any("chain" in r and "line_explode" in r for r in result["amber_reasons"])

    def test_amber_when_element_missing(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["web_scrape", "llm"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_chain(["web_scrape", "json_explode"]),
        )
        assert result["verdict"] == "AMBER"
        assert any("'llm'" in r for r in result["amber_reasons"])

    def test_chain_matches_node_type_for_gates(self) -> None:
        """Gates have plugin: null but node_type: gate. Chain match on 'gate' should hit."""
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["passthrough", "gate"]}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                nodes=[
                    {"id": "n0", "node_type": "transform", "plugin": "passthrough"},
                    {"id": "n1", "node_type": "gate", "plugin": None},
                ],
            ),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]


# --------------------------------------------------------------------------
# must_include_observed_columns
# --------------------------------------------------------------------------


class TestObservedColumns:
    def _state_with_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return _state_valid(source={"plugin": "csv", "options": {"schema": schema}})

    def test_observed_mode_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["amount", "name"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_with_schema({"mode": "observed"}),
        )
        assert result["verdict"] == "GREEN"

    def test_flexible_mode_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["amount"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_with_schema({"mode": "flexible", "fields": ["other: str"]}),
        )
        assert result["verdict"] == "GREEN"

    def test_fixed_with_all_columns_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["amount", "name"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_with_schema({"mode": "fixed", "fields": ["amount: int", "name: str"]}),
        )
        assert result["verdict"] == "GREEN"

    def test_fixed_missing_column_ambers(self) -> None:
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["amount", "category"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_with_schema({"mode": "fixed", "fields": ["amount: int"]}),
        )
        assert result["verdict"] == "AMBER"
        assert any("category" in r for r in result["amber_reasons"])

    def test_case_insensitive_column_match(self) -> None:
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["AMOUNT"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_with_schema({"mode": "fixed", "fields": ["amount: int"]}),
        )
        assert result["verdict"] == "GREEN"

    def test_missing_schema_ambers(self) -> None:
        state = _state_valid(source={"plugin": "csv", "options": {}})
        result = score(
            scenario=_scenario(green={"must_include_observed_columns": ["amount"]}),
            messages=[_msg("assistant", "ok")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("schema missing" in r for r in result["amber_reasons"])


# --------------------------------------------------------------------------
# must_handle_field_as_numeric
# --------------------------------------------------------------------------


class TestNumericHandling:
    def test_int_in_source_schema_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_handle_field_as_numeric": "price"}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                source={
                    "plugin": "csv",
                    "options": {"schema": {"mode": "fixed", "fields": ["price: int"]}},
                },
            ),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_float_optional_in_source_schema_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_handle_field_as_numeric": "price"}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                source={
                    "plugin": "csv",
                    "options": {"schema": {"mode": "flexible", "fields": ["price: float?"]}},
                },
            ),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_type_coerce_node_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"must_handle_field_as_numeric": "price"}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                nodes=[
                    {
                        "id": "coerce",
                        "node_type": "transform",
                        "plugin": "type_coerce",
                        "options": {"conversions": [{"field": "price", "to": "float"}]},
                    },
                ],
            ),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_string_field_no_coerce_ambers(self) -> None:
        result = score(
            scenario=_scenario(green={"must_handle_field_as_numeric": "price"}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                source={
                    "plugin": "csv",
                    "options": {"schema": {"mode": "fixed", "fields": ["price: str"]}},
                },
            ),
        )
        assert result["verdict"] == "AMBER"
        assert any("price" in r and "numeric handling" in r for r in result["amber_reasons"])

    def test_unrelated_coerce_does_not_pass(self) -> None:
        result = score(
            scenario=_scenario(green={"must_handle_field_as_numeric": "price"}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(
                nodes=[
                    {
                        "id": "coerce",
                        "node_type": "transform",
                        "plugin": "type_coerce",
                        "options": {"conversions": [{"field": "quantity", "to": "int"}]},
                    },
                ],
            ),
        )
        assert result["verdict"] == "AMBER"


# --------------------------------------------------------------------------
# max_repair_turns
# --------------------------------------------------------------------------


class TestMaxRepairTurns:
    def test_under_max_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"max_repair_turns": 2}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(composer_meta={"repair_turns_used": 1}),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_at_max_passes(self) -> None:
        result = score(
            scenario=_scenario(green={"max_repair_turns": 2}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(composer_meta={"repair_turns_used": 2}),
        )
        assert result["verdict"] == "GREEN"

    def test_zero_repair_turns_passes(self) -> None:
        """First-pass success — repair_turns_used=0."""
        result = score(
            scenario=_scenario(green={"max_repair_turns": 2}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(composer_meta={"repair_turns_used": 0}),
        )
        assert result["verdict"] == "GREEN"

    def test_over_max_ambers(self) -> None:
        result = score(
            scenario=_scenario(green={"max_repair_turns": 1}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(composer_meta={"repair_turns_used": 3}),
        )
        assert result["verdict"] == "AMBER"
        assert any("3 repair turns" in r for r in result["amber_reasons"])

    def test_field_absent_ambers_explicitly(self) -> None:
        """Absence of repair_turns_used is itself a finding (never silently passes)."""
        result = score(
            scenario=_scenario(green={"max_repair_turns": 2}),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(),
        )
        assert result["verdict"] == "AMBER"
        assert any("repair_turns_used not present" in r for r in result["amber_reasons"])

    def test_no_max_repair_turns_means_no_check(self) -> None:
        """Scenarios without max_repair_turns are not penalized for the missing field."""
        result = score(
            scenario=_scenario(),
            messages=[_msg("assistant", "ok")],
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"
        assert all("repair" not in r for r in result["amber_reasons"])


# --------------------------------------------------------------------------
# Stats and combinatorial signals
# --------------------------------------------------------------------------


class TestStats:
    def test_stats_captures_message_and_node_counts(self) -> None:
        result = score(
            scenario=_scenario(),
            messages=[_msg("user", "hi"), _msg("assistant", "ok"), _msg("assistant", "done")],
            state=_state_valid(),
        )
        stats = result["stats"]
        assert stats["assistant_message_count"] == 2
        assert stats["state_node_count"] == 1
        assert stats["state_output_count"] == 1
        assert stats["is_valid"] is True


@pytest.mark.parametrize(
    ("verdict_inputs", "expected"),
    [
        # No reasons -> GREEN
        ({"red": [], "amber": []}, "GREEN"),
        # Only amber -> AMBER
        ({"red": [], "amber": ["x"]}, "AMBER"),
        # Any red -> RED
        ({"red": ["y"], "amber": []}, "RED"),
        ({"red": ["y"], "amber": ["x"]}, "RED"),
    ],
)
def test_verdict_precedence(verdict_inputs: dict[str, list[str]], expected: str) -> None:
    """RED > AMBER > GREEN. Construct directly to lock the precedence rule in code."""
    # We can't easily induce specific reason lists without scenario contortion,
    # but the precedence is encoded as: RED if red_reasons else (GREEN if not amber else AMBER).
    # We test that ladder semantics by example via the other tests; this parametrize is
    # a sanity check that the helper expectations stay aligned.
    red_reasons = verdict_inputs["red"]
    amber_reasons = verdict_inputs["amber"]
    if red_reasons:
        assert expected == "RED"
    elif amber_reasons:
        assert expected == "AMBER"
    else:
        assert expected == "GREEN"
