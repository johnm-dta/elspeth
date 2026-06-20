"""Sanity tests for the composer-rgr convergence-suite scenario files.

These tests do NOT run the live RGR harness (no LLM cost). They:

  1. Confirm each scenario.json is well-formed and uses only known
     red/green criterion keys.
  2. Build a synthetic ideal state and verify each scenario scores GREEN
     against it. This catches scenario-internal contradictions early —
     e.g., a green criterion that no real state shape could satisfy.
  3. Confirm a "minimum-viable broken" state correctly RED/AMBERs each
     scenario, so the criteria are not vacuously satisfied.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from evals.lib.composer_rgr_score import score

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUITE = _REPO_ROOT / "evals" / "composer-rgr" / "scenarios" / "convergence-suite"
_GOV_PAGES_RATE_COOL = _REPO_ROOT / "evals" / "composer-rgr" / "scenarios" / "gov-pages-rate-cool" / "scenario.json"

_KNOWN_RED_KEYS = {
    "passivity_phrases",
    "passivity_phrases_note",
    "build_failure_sentinels",
    "build_failure_note",
    "must_be_valid",
    "must_be_valid_note",
}

_KNOWN_GREEN_KEYS = {
    "must_be_valid",
    "must_be_valid_note",
    "must_have_node_kinds_substring_any_of",
    "must_have_node_chain_in_order",
    "must_have_node_chain_in_order_note",
    "must_include_observed_columns",
    "must_include_observed_columns_note",
    "must_handle_field_as_numeric",
    "must_handle_field_as_numeric_note",
    "must_have_outputs_min",
    "must_have_outputs_min_note",
    "max_repair_turns",
    "max_repair_turns_note",
    "must_not_contain_passivity",
}


def _scenario_paths() -> list[Path]:
    return sorted(_SUITE.glob("*/scenario.json"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _gov_pages_messages_with_tool_calls(count: int) -> list[dict[str, Any]]:
    """Build a successful tool trajectory with an exact persisted-call count."""
    if count < 2:
        raise ValueError("gov-pages boundary fixtures need discovery plus mutation")
    tool_calls: list[dict[str, Any]] = [
        {"id": "call_schema", "type": "function", "function": {"name": "get_plugin_schema", "arguments": "{}"}},
        {"id": "call_set_pipeline", "type": "function", "function": {"name": "set_pipeline", "arguments": "{}"}},
    ]
    tool_calls.extend(
        {
            "id": f"call_state_{index}",
            "type": "function",
            "function": {"name": "get_pipeline_state", "arguments": "{}"},
        }
        for index in range(count - 2)
    )
    return [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "tool_call_id": "call_set_pipeline", "content": json.dumps({"success": True})},
        {"role": "assistant", "content": "Pipeline ready."},
    ]


def _gov_pages_valid_state() -> dict[str, Any]:
    """Small state satisfying the gov-pages structural green criteria."""
    return {
        "is_valid": True,
        "source": {"plugin": "csv", "options": {"schema": {"mode": "observed"}}},
        "nodes": [{"id": "rate_coolness", "node_type": "transform", "plugin": "llm"}],
        "outputs": [{"name": "rated_pages", "plugin": "json"}],
    }


# --------------------------------------------------------------------------
# 1. Well-formedness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_scenario_is_well_formed(scenario_path: Path) -> None:
    scenario = _load(scenario_path)
    assert isinstance(scenario.get("scenario_id"), str)
    assert isinstance(scenario.get("opening_prompt"), str)
    assert scenario["opening_prompt"], "opening_prompt is empty"
    assert isinstance(scenario.get("red_criteria"), dict)
    assert isinstance(scenario.get("green_criteria"), dict)


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_scenario_uses_known_red_keys(scenario_path: Path) -> None:
    red = _load(scenario_path)["red_criteria"]
    unknown = set(red) - _KNOWN_RED_KEYS
    assert not unknown, f"unknown red_criteria keys: {unknown}"


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_scenario_uses_known_green_keys(scenario_path: Path) -> None:
    green = _load(scenario_path)["green_criteria"]
    unknown = set(green) - _KNOWN_GREEN_KEYS
    assert not unknown, f"unknown green_criteria keys: {unknown}"


# --------------------------------------------------------------------------
# 2. Internal consistency — synthetic ideal state scores GREEN
# --------------------------------------------------------------------------


def _ideal_state_for(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build the smallest synthetic state that satisfies every green criterion.

    Mirrors what the runtime should produce on a converged session: source
    with appropriate schema mode, every node-chain plugin present in order,
    enough outputs, and the convergence-bar repair-turn meta field.
    """
    green = scenario["green_criteria"]

    # Source schema covers observed columns (or numeric handling demand).
    fields: list[str] = []
    numeric_field = green.get("must_handle_field_as_numeric")
    observed = green.get("must_include_observed_columns") or []

    schema_mode: str
    if observed:
        schema_mode = "fixed"
        for col in observed:
            type_str = "float" if numeric_field and col == numeric_field else "str"
            fields.append(f"{col}: {type_str}")
    elif numeric_field:
        schema_mode = "fixed"
        fields.append(f"{numeric_field}: float")
    else:
        schema_mode = "observed"

    schema: dict[str, Any] = {"mode": schema_mode}
    if fields:
        schema["fields"] = fields

    nodes: list[dict[str, Any]] = []
    chain = green.get("must_have_node_chain_in_order") or []
    for i, plugin in enumerate(chain):
        nodes.append({"id": f"n{i}", "node_type": plugin, "plugin": plugin})

    # Outputs to meet must_have_outputs_min.
    out_count = max(green.get("must_have_outputs_min", 1), 1)
    outputs = [{"name": f"out{i}", "plugin": "json"} for i in range(out_count)]

    state: dict[str, Any] = {
        "is_valid": True,
        "sources": {"primary": {"plugin": "csv", "options": {"schema": schema}}},
        "nodes": nodes,
        "outputs": outputs,
    }

    if isinstance(green.get("max_repair_turns"), int):
        state["composer_meta"] = {"repair_turns_used": 0}

    return state


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_ideal_state_scores_green(scenario_path: Path) -> None:
    """Each scenario's criteria must be jointly satisfiable by a real state shape.

    If this fails, the scenario has internally contradictory criteria —
    no real composer output could pass it.
    """
    scenario = _load(scenario_path)
    state = _ideal_state_for(scenario)
    result = score(scenario, [{"role": "assistant", "content": "ok"}], state)
    assert result["verdict"] == "GREEN", (
        f"{scenario_path.parent.name}: ideal synthetic state did not score GREEN. "
        f"red={result['red_reasons']} amber={result['amber_reasons']}"
    )


# --------------------------------------------------------------------------
# 3. Negative coverage — broken state actually triggers AMBER/RED
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_invalid_state_scores_red(scenario_path: Path) -> None:
    """A trivially invalid pipeline (is_valid=false, no nodes) must RED."""
    scenario = _load(scenario_path)
    state: dict[str, Any] = {"is_valid": False, "source": None, "nodes": [], "outputs": []}
    result = score(scenario, [{"role": "assistant", "content": "ok"}], state)
    assert result["verdict"] == "RED"


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_passivity_phrase_scores_red(scenario_path: Path) -> None:
    """A clarifying-question final message hits the passivity phrase guard."""
    scenario = _load(scenario_path)
    result = score(
        scenario,
        [{"role": "assistant", "content": "If you want, I can pick a default schema."}],
        _ideal_state_for(scenario),
    )
    assert result["verdict"] == "RED"
    assert any("passivity" in r for r in result["red_reasons"])


@pytest.mark.parametrize(
    ("persisted_tool_calls", "expected_verdict"),
    [
        (8, "GREEN"),
        (9, "AMBER"),
        (12, "AMBER"),
        (13, "RED"),
    ],
)
def test_gov_pages_rate_cool_tool_call_efficiency_boundaries(
    persisted_tool_calls: int,
    expected_verdict: str,
) -> None:
    """The gov-pages rate-cool 9-12 call window is intentionally AMBER.

    The scenario prose distinguishes the green target (8 or fewer tool
    calls), amber inefficiency (9-12), and red catastrophic thrash (>12).
    This test loads the real scenario so the executable criteria cannot
    drift from that prose.
    """
    scenario = _load(_GOV_PAGES_RATE_COOL)
    result = score(
        scenario,
        _gov_pages_messages_with_tool_calls(persisted_tool_calls),
        _gov_pages_valid_state(),
    )
    assert result["verdict"] == expected_verdict, (
        f"{persisted_tool_calls} calls scored {result['verdict']}; red={result['red_reasons']} amber={result['amber_reasons']}"
    )


@pytest.mark.parametrize("scenario_path", _scenario_paths(), ids=lambda p: p.parent.name)
def test_repair_turns_field_absent_ambers(scenario_path: Path) -> None:
    """An otherwise-ideal state without ``composer_meta.repair_turns_used`` ambers explicitly.

    The scoring contract treats absence as ambiguous (we cannot prove the
    convergence-budget criterion was met without the field) rather than green
    or red. This pins that behaviour against scenarios whose green_criteria
    declare ``max_repair_turns``.
    """
    scenario = _load(scenario_path)
    state = _ideal_state_for(scenario)
    state.pop("composer_meta", None)
    result = score(scenario, [{"role": "assistant", "content": "ok"}], state)
    if isinstance(scenario["green_criteria"].get("max_repair_turns"), int):
        assert result["verdict"] == "AMBER"
        assert any("repair_turns_used not present" in r for r in result["amber_reasons"])
    else:
        # Sanity: if a future scenario drops max_repair_turns, this still passes.
        pytest.skip("scenario does not declare max_repair_turns")
