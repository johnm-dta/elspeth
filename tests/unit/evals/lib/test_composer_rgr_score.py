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
    """Minimal valid state — is_valid=True, one node, one output.

    Sources use the multi-source shape (ADR-025): a ``sources`` map keyed by
    name. The ``source=`` convenience kwarg sets the default "primary" named
    source so per-test fixtures stay terse.
    """
    source = extras.pop("source", {"plugin": "csv", "options": {"schema": {"mode": "observed"}}})
    base: dict[str, Any] = {
        "is_valid": True,
        "sources": {"primary": source},
        "nodes": [{"id": "noop", "node_type": "transform", "plugin": "passthrough"}],
        "outputs": [{"name": "out", "plugin": "json"}],
    }
    base.update(extras)
    return base


def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


def _assistant_tool_call(name: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-{name}",
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


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

    def test_node_kind_group_does_not_match_substring_inside_plugin(self) -> None:
        scenario = _scenario(green={"must_have_node_kinds_substring_any_of": [["gate"]]})
        result = score(
            scenario=scenario,
            messages=[_msg("assistant", "ok")],
            state=_state_valid(nodes=[{"id": "aggregate", "node_type": "transform", "plugin": "aggregate"}]),
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

    @pytest.mark.parametrize("tool_call_count", [9, 12])
    def test_amber_on_tool_call_budget_above_green_target_below_red_limit(self, tool_call_count: int) -> None:
        result = score(
            scenario=_scenario(
                red={"max_persisted_tool_calls": 12},
                green={"max_tool_calls_for_green": 8},
            ),
            messages=[_assistant_tool_call(f"tool_{i}") for i in range(tool_call_count)],
            state=_state_valid(),
        )

        assert result["verdict"] == "AMBER"
        assert any("tool calls" in reason for reason in result["amber_reasons"])


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

    def test_chain_does_not_match_substring_inside_plugin(self) -> None:
        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["gate"]}),
            messages=[_msg("assistant", "ok")],
            state=self._state_chain(["aggregate"]),
        )

        assert result["verdict"] == "AMBER"
        assert any("'gate'" in r for r in result["amber_reasons"])

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

    def test_chain_matches_source_transforms_and_sink(self) -> None:
        state = _state_valid(
            source={"plugin": "csv", "options": {"schema": {"mode": "observed"}}},
            nodes=[
                {"id": "n0", "node_type": "transform", "plugin": "t1"},
                {"id": "n1", "node_type": "transform", "plugin": "t2"},
            ],
            outputs=[{"name": "out", "plugin": "jsonl"}],
        )

        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["csv", "t1", "t2", "jsonl"]}),
            messages=[_msg("assistant", "ok")],
            state=state,
        )

        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_chain_fails_when_transform_missing_between_source_and_sink(self) -> None:
        state = _state_valid(
            source={"plugin": "csv", "options": {"schema": {"mode": "observed"}}},
            nodes=[{"id": "n0", "node_type": "transform", "plugin": "t1"}],
            outputs=[{"name": "out", "plugin": "jsonl"}],
        )

        result = score(
            scenario=_scenario(green={"must_have_node_chain_in_order": ["csv", "t1", "t2", "jsonl"]}),
            messages=[_msg("assistant", "ok")],
            state=state,
        )

        assert result["verdict"] == "AMBER"
        assert any("'t2'" in r for r in result["amber_reasons"])


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


# --------------------------------------------------------------------------
# Hint-uptake asserters (composer-jit-hints Phase 1)
# --------------------------------------------------------------------------


class TestSourceOptionKeyAsserters:
    """must_have_options_keys_for_source / must_not_have_options_keys_for_source."""

    def test_green_when_required_source_keys_present(self) -> None:
        state = _state_valid()
        state["sources"] = {"primary": {"plugin": "csv", "options": {"columns": ["a", "b"], "schema": {"mode": "observed"}}}}
        result = score(
            scenario=_scenario(green={"must_have_options_keys_for_source": ["columns"]}),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "GREEN"

    def test_amber_when_required_source_key_missing(self) -> None:
        state = _state_valid()  # source has no 'columns'
        result = score(
            scenario=_scenario(green={"must_have_options_keys_for_source": ["columns"]}),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("columns" in r for r in result["amber_reasons"])

    def test_amber_when_forbidden_source_key_present(self) -> None:
        state = _state_valid()
        state["sources"] = {"primary": {"plugin": "csv", "options": {"schema": {"fields": ["id: int"], "mode": "fixed"}}}}
        result = score(
            scenario=_scenario(green={"must_not_have_options_keys_for_source": ["schema.fields"]}),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("schema.fields" in r for r in result["amber_reasons"])


class TestOutputOptionAsserters:
    """must_have_options_value_for_output — by-name output (sink) value pinning."""

    def test_green_when_sink_write_mode_matches(self) -> None:
        state = _state_valid()
        state["outputs"] = [{"name": "customers", "plugin": "database", "options": {"write_mode": "upsert"}}]
        result = score(
            scenario=_scenario(
                green={
                    "must_have_options_value_for_output": [{"sink_name": "customers", "key": "write_mode", "allowed_values": ["upsert"]}]
                }
            ),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "GREEN"

    def test_amber_when_sink_write_mode_is_wrong(self) -> None:
        state = _state_valid()
        state["outputs"] = [{"name": "customers", "plugin": "database", "options": {"write_mode": "insert"}}]
        result = score(
            scenario=_scenario(
                green={
                    "must_have_options_value_for_output": [{"sink_name": "customers", "key": "write_mode", "allowed_values": ["upsert"]}]
                }
            ),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("write_mode" in r for r in result["amber_reasons"])

    def test_amber_when_sink_option_key_missing(self) -> None:
        state = _state_valid()
        state["outputs"] = [{"name": "customers", "plugin": "database", "options": {}}]
        result = score(
            scenario=_scenario(
                green={
                    "must_have_options_value_for_output": [{"sink_name": "customers", "key": "write_mode", "allowed_values": ["upsert"]}]
                }
            ),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("write_mode" in r for r in result["amber_reasons"])


class TestOutputPluginAsserters:
    """must_have_output_plugins — output sink plugin multiset pinning."""

    def test_green_when_required_output_plugins_present(self) -> None:
        state = _state_valid(
            outputs=[
                {"name": "approved", "plugin": "csv", "options": {}},
                {"name": "rejected", "plugin": "csv", "options": {}},
            ]
        )
        result = score(
            scenario=_scenario(green={"must_have_output_plugins": ["csv", "csv"]}),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_amber_when_json_outputs_replace_required_csv_sinks(self) -> None:
        state = _state_valid(
            outputs=[
                {"name": "approved", "plugin": "json", "options": {"format": "jsonl"}},
                {"name": "rejected", "plugin": "json", "options": {"format": "jsonl"}},
            ]
        )
        result = score(
            scenario=_scenario(green={"must_have_output_plugins": ["csv", "csv"]}),
            messages=[_msg("assistant", "done")],
            state=state,
        )
        assert result["verdict"] == "AMBER"
        assert any("output plugins" in r and "csv" in r and "json" in r for r in result["amber_reasons"])


# --------------------------------------------------------------------------
# Tool-sequence asserters (gov-pages-rate-cool scenario, 2026-05-23)
# --------------------------------------------------------------------------


import json as _json  # noqa: E402  -- after all baseline imports


def _tool_call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    """Build a LiteLLM-shape ToolCall dict for tests."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": _json.dumps(arguments)},
    }


def _assistant_with_calls(*calls: dict[str, Any]) -> dict[str, Any]:
    """Assistant message row carrying one or more tool_calls."""
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


def _tool_row(call_id: str, *, success: bool, content_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """role=tool message row carrying a ToolResult-shaped JSON content blob."""
    payload: dict[str, Any] = {"success": success}
    if content_extra:
        payload.update(content_extra)
    return {"role": "tool", "tool_call_id": call_id, "content": _json.dumps(payload)}


class TestMaxPersistedToolCalls:
    """red_criteria.max_persisted_tool_calls — trajectory length cap."""

    def test_green_when_under_cap(self) -> None:
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "list_sources")),
            _tool_row("c1", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(red={"max_persisted_tool_calls": 8}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN", result["red_reasons"]

    def test_green_at_exact_cap(self) -> None:
        # 3 calls, cap = 3 — boundary inclusive
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(
                _tool_call("c1", "list_sources"),
                _tool_call("c2", "list_transforms"),
                _tool_call("c3", "list_sinks"),
            ),
            _tool_row("c1", success=True),
            _tool_row("c2", success=True),
            _tool_row("c3", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(red={"max_persisted_tool_calls": 3}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"

    def test_red_when_over_cap(self) -> None:
        calls = [_tool_call(f"c{i}", "list_sources") for i in range(5)]
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(*calls),
            *[_tool_row(f"c{i}", success=True) for i in range(5)],
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(red={"max_persisted_tool_calls": 3}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "RED"
        assert any("persisted 5 tool calls" in r for r in result["red_reasons"])

    def test_cap_absent_means_no_check(self) -> None:
        calls = [_tool_call(f"c{i}", "list_sources") for i in range(20)]
        messages = [_assistant_with_calls(*calls), _msg("assistant", "done")]
        result = score(
            scenario=_scenario(),
            messages=messages,
            state=_state_valid(),
        )
        # No max_persisted_tool_calls in scenario -> baseline scoring,
        # GREEN despite 20 calls.
        assert result["verdict"] == "GREEN", result["red_reasons"]


class TestSetPipelineRejectionWithoutSuccess:
    """red_criteria.set_pipeline_rejection_without_success — convergence failure."""

    def test_red_when_only_rejections(self) -> None:
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "set_pipeline"), _tool_call("c2", "set_pipeline")),
            _tool_row("c1", success=False),
            _tool_row("c2", success=False),
            _msg("assistant", "I tried."),
        ]
        result = score(
            scenario=_scenario(red={"set_pipeline_rejection_without_success": True}),
            messages=messages,
            state=_state_valid(is_valid=False),
        )
        assert result["verdict"] == "RED"
        assert any("set_pipeline" in r and "0 successful" in r for r in result["red_reasons"])

    def test_green_when_eventual_success(self) -> None:
        # Two rejections, then one success — convergence achieved.
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            _tool_row("c1", success=False),
            _assistant_with_calls(_tool_call("c2", "set_pipeline")),
            _tool_row("c2", success=False),
            _assistant_with_calls(_tool_call("c3", "set_pipeline")),
            _tool_row("c3", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(red={"set_pipeline_rejection_without_success": True}),
            messages=messages,
            state=_state_valid(),
        )
        # Some rejections + one success => rule does not fire.
        assert result["verdict"] == "GREEN", result["red_reasons"]

    def test_green_when_no_attempts(self) -> None:
        # Passivity case — no set_pipeline at all. This rule is silent;
        # other rules (passivity_phrases / must_be_valid) own the verdict.
        messages = [_msg("user", "hi"), _msg("assistant", "Sure, what do you want?")]
        result = score(
            scenario=_scenario(red={"set_pipeline_rejection_without_success": True}),
            messages=messages,
            state=_state_valid(),
        )
        # No set_pipeline rejections -> rule does NOT fire. Other red signals
        # still apply via the baseline rules.
        assert result["verdict"] == "GREEN", result["red_reasons"]

    def test_rule_disabled_when_flag_absent(self) -> None:
        messages = [
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            _tool_row("c1", success=False),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(),  # no set_pipeline_rejection_without_success
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"

    def test_malformed_tool_row_does_not_count(self) -> None:
        # Tool row content not parseable as JSON dict -> result_unknown,
        # rule does not fire.
        messages = [
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            {"role": "tool", "tool_call_id": "c1", "content": "not json"},
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(red={"set_pipeline_rejection_without_success": True}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"


class TestDiscoverBeforeMutation:
    """green_criteria.must_discover_schema_before_first_mutation — discover-first."""

    def test_green_when_schema_precedes_mutation(self) -> None:
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "get_plugin_schema")),
            _tool_row("c1", success=True),
            _assistant_with_calls(_tool_call("c2", "set_pipeline")),
            _tool_row("c2", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(green={"must_discover_schema_before_first_mutation": True}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN", result["amber_reasons"]

    def test_amber_when_mutation_precedes_schema(self) -> None:
        # Schema-after-rejection pattern: model tried set_pipeline first,
        # got rejected, then looked up the schema. That doesn't satisfy
        # the discover-first contract.
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            _tool_row("c1", success=False),
            _assistant_with_calls(_tool_call("c2", "get_plugin_schema")),
            _tool_row("c2", success=True),
            _assistant_with_calls(_tool_call("c3", "set_pipeline")),
            _tool_row("c3", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(green={"must_discover_schema_before_first_mutation": True}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "AMBER"
        assert any("discovery" in r and "post-rejection" in r for r in result["amber_reasons"])

    def test_amber_when_no_schema_at_all(self) -> None:
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            _tool_row("c1", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(green={"must_discover_schema_before_first_mutation": True}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "AMBER"
        assert any("not preceded by any get_plugin_schema" in r for r in result["amber_reasons"])

    def test_vacuously_green_when_no_mutations(self) -> None:
        # Nothing mutated -> rule is vacuously satisfied. The empty
        # pipeline is caught by must_be_valid, not this rule.
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "list_sources")),
            _tool_row("c1", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(green={"must_discover_schema_before_first_mutation": True}),
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"

    def test_rule_disabled_when_flag_absent(self) -> None:
        messages = [
            _assistant_with_calls(_tool_call("c1", "set_pipeline")),
            _tool_row("c1", success=True),
            _msg("assistant", "done"),
        ]
        result = score(
            scenario=_scenario(),  # no flag
            messages=messages,
            state=_state_valid(),
        )
        assert result["verdict"] == "GREEN"

    def test_mutation_recognises_full_mutating_set(self) -> None:
        # Any of the mutating tool names triggers the check.
        for mutating in ("set_source", "upsert_node", "set_output", "set_source_from_blob", "apply_pipeline_recipe", "patch_node_options"):
            messages = [
                _assistant_with_calls(_tool_call("c1", mutating)),
                _tool_row("c1", success=True),
                _msg("assistant", "done"),
            ]
            result = score(
                scenario=_scenario(green={"must_discover_schema_before_first_mutation": True}),
                messages=messages,
                state=_state_valid(),
            )
            assert result["verdict"] == "AMBER", f"expected AMBER for first-call mutation {mutating}, got {result}"


class TestPersistedToolCallCountStat:
    """Stats: persisted_tool_call_count surfaced for diagnostic use."""

    def test_stat_counts_assistant_tool_calls(self) -> None:
        messages = [
            _msg("user", "hi"),
            _assistant_with_calls(_tool_call("c1", "list_sources"), _tool_call("c2", "list_sinks")),
            _tool_row("c1", success=True),
            _tool_row("c2", success=True),
            _assistant_with_calls(_tool_call("c3", "set_pipeline")),
            _tool_row("c3", success=True),
            _msg("assistant", "done"),
        ]
        result = score(scenario=_scenario(), messages=messages, state=_state_valid())
        assert result["stats"]["persisted_tool_call_count"] == 3

    def test_stat_zero_on_passivity(self) -> None:
        messages = [_msg("user", "hi"), _msg("assistant", "what do you want?")]
        result = score(scenario=_scenario(), messages=messages, state=_state_valid())
        assert result["stats"]["persisted_tool_call_count"] == 0
