# tests/unit/core/test_config_name_uniqueness.py
"""Tests for validate_globally_unique_node_names() cross-type collision detection.

The validator (config.py:1270-1299) ensures all processing node names are unique
across transforms, gates, aggregations, coalesce nodes, and sinks. A collision
would create ambiguous audit entries and routing errors.

These tests exercise the Pydantic model validator at the ElspethSettings level.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.core.config import (
    AggregationSettings,
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
    TriggerConfig,
)


def _source(**kwargs) -> SourceSettings:
    defaults = {
        "plugin": "csv",
        "on_success": "src_out",
        "options": {
            "path": "test.csv",
            "on_validation_failure": "discard",
            "schema": {"mode": "observed"},
        },
    }
    defaults.update(kwargs)
    return SourceSettings(**defaults)


def _sink(**kwargs) -> SinkSettings:
    defaults = {
        "plugin": "json",
        "on_write_failure": "discard",
        "options": {"path": "output.json", "schema": {"mode": "observed"}},
    }
    defaults.update(kwargs)
    return SinkSettings(**defaults)


class TestCrossTypeNameCollisions:
    """Verify that name collisions across different node types are rejected."""

    def test_transform_name_equals_gate_name_rejected(self) -> None:
        """A transform and gate sharing the same name must be rejected."""
        with pytest.raises(ValidationError, match=r"both.*transform.*and.*gate|both.*gate.*and.*transform"):
            ElspethSettings(
                sources={"primary": _source()},
                sinks={"output": _sink()},
                transforms=[
                    TransformSettings(
                        name="shared_name",
                        plugin="passthrough",
                        input="src_out",
                        on_success="output",
                        on_error="discard",
                        options={"schema": {"mode": "observed"}},
                    ),
                ],
                gates=[
                    GateSettings(
                        name="shared_name",
                        input="src_out",
                        condition="True",
                        routes={"true": "output", "false": "output"},
                    ),
                ],
            )

    def test_transform_name_equals_aggregation_name_rejected(self) -> None:
        """A transform and aggregation sharing the same name must be rejected."""
        with pytest.raises(ValidationError, match=r"both.*transform.*and.*aggregation|both.*aggregation.*and.*transform"):
            ElspethSettings(
                sources={"primary": _source()},
                sinks={"output": _sink()},
                transforms=[
                    TransformSettings(
                        name="shared_name",
                        plugin="passthrough",
                        input="src_out",
                        on_success="output",
                        on_error="discard",
                        options={"schema": {"mode": "observed"}},
                    ),
                ],
                aggregations=[
                    AggregationSettings(
                        name="shared_name",
                        plugin="passthrough",
                        input="agg_input",
                        on_success="output",
                        on_error="discard",
                        trigger=TriggerConfig(count=1),
                        options={"schema": {"mode": "observed"}},
                    ),
                ],
            )

    def test_gate_name_equals_coalesce_name_rejected(self) -> None:
        """A gate and coalesce sharing the same name must be rejected."""
        with pytest.raises(ValidationError, match=r"both.*gate.*and.*coalesce|both.*coalesce.*and.*gate"):
            ElspethSettings(
                sources={"primary": _source()},
                sinks={"output": _sink()},
                gates=[
                    GateSettings(
                        name="shared_name",
                        input="src_out",
                        condition="True",
                        routes={"true": "output", "false": "output"},
                    ),
                ],
                coalesce=[
                    CoalesceSettings(
                        name="shared_name",
                        branches=["branch_a", "branch_b"],
                        on_success="output",
                    ),
                ],
            )

    def test_transform_name_equals_sink_name_rejected(self) -> None:
        """A transform and sink sharing the same name must be rejected."""
        with pytest.raises(ValidationError, match=r"both.*transform.*and.*sink|both.*sink.*and.*transform"):
            ElspethSettings(
                sources={"primary": _source()},
                sinks={
                    "output": _sink(),
                    "shared_name": _sink(options={"path": "shared.json", "schema": {"mode": "observed"}}),
                },
                transforms=[
                    TransformSettings(
                        name="shared_name",
                        plugin="passthrough",
                        input="src_out",
                        on_success="output",
                        on_error="discard",
                        options={"schema": {"mode": "observed"}},
                    ),
                ],
            )

    def test_all_unique_names_across_types_accepted(self) -> None:
        """Distinct names across all types should pass validation."""
        config = ElspethSettings(
            sources={"primary": _source()},
            sinks={
                "output": _sink(),
                "flagged": _sink(options={"path": "flagged.json", "schema": {"mode": "observed"}}),
            },
            transforms=[
                TransformSettings(
                    name="my_transform",
                    plugin="passthrough",
                    input="src_out",
                    on_success="gate_input",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                ),
            ],
            gates=[
                GateSettings(
                    name="my_gate",
                    input="gate_input",
                    condition="True",
                    routes={"true": "output", "false": "flagged"},
                ),
            ],
        )
        # Should not raise
        assert config.sources["primary"].plugin == "csv"
