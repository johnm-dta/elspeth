"""Tests for panel/RGR scenario criteria derivation."""

from __future__ import annotations

from typing import Any

from evals.lib.scenario_from_example import build_criteria_from_target


def test_build_criteria_pins_output_plugins_from_structural_target() -> None:
    target: dict[str, Any] = {
        "source": {"plugin": "csv", "shape_token": "csv", "columns": ["id", "approved"]},
        "transforms": [],
        "gates": [
            {
                "name": "approval_check",
                "condition": "row['approved'] == 'true'",
                "is_fork": False,
                "fork_paths": [],
                "routes": {"true": "approved", "false": "rejected"},
            }
        ],
        "aggregations": [],
        "coalesce_nodes": [],
        "sinks": [
            {"name": "approved", "plugin": "csv", "shape_token": "csv"},
            {"name": "rejected", "plugin": "csv", "shape_token": "csv"},
        ],
    }

    criteria = build_criteria_from_target(target)

    assert criteria["green_criteria"]["must_have_output_plugins"] == ["csv", "csv"]
