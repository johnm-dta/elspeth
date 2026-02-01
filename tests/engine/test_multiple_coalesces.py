"""Integration test for multiple independent coalesce points.

BUG-LINEAGE-01 P1.7: Verify that multiple independent fork/coalesce pairs
work correctly when using the production path (ExecutionGraph.from_plugin_instances).
"""

from __future__ import annotations

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import BranchName, CoalesceName
from elspeth.core.config import (
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph


class TestMultipleCoalescePoints:
    """Test pipeline with multiple independent fork/coalesce pairs."""

    def test_two_independent_fork_coalesce_points(self, plugin_manager) -> None:
        """Pipeline with two independent fork gates and two coalesce points should work correctly.

        Topology:
        source -> forker1 (fork to path_a, path_b) -> merge1 ->
                  forker2 (fork to path_c, path_d) -> merge2 -> sink

        This tests that:
        1. branch_to_coalesce mapping works for multiple coalesce points
        2. coalesce_step_map has correct entries for both merge points
        3. Production path (from_plugin_instances) exercises correct mappings
        """
        settings = ElspethSettings(
            source=SourceSettings(
                plugin="null",
                options={"count": 2},
            ),
            transforms=[
                TransformSettings(
                    plugin="passthrough",
                    options={"schema": {"fields": "dynamic"}},
                ),
            ],
            default_sink="output",
            sinks={
                "output": SinkSettings(
                    plugin="json",
                    options={
                        "path": "/tmp/test_multiple_coalesces.json",
                        "schema": {"fields": "dynamic"},
                    },
                ),
            },
            gates=[
                GateSettings(
                    name="forker1",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
                GateSettings(
                    name="forker2",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_c", "path_d"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge1",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
                CoalesceSettings(
                    name="merge2",
                    branches=["path_c", "path_d"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Use production path - instantiate real plugins
        plugins = instantiate_plugins_from_config(settings)

        # Build graph using production path
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=list(settings.coalesce),
            default_sink=settings.default_sink,
        )

        # CRITICAL: Verify branch_to_coalesce mapping is correct for multiple coalesce points
        branch_to_coalesce = graph.get_branch_to_coalesce_map()

        # First fork's branches should map to first coalesce
        assert branch_to_coalesce[BranchName("path_a")] == CoalesceName("merge1"), "path_a should map to merge1 name"
        assert branch_to_coalesce[BranchName("path_b")] == CoalesceName("merge1"), "path_b should map to merge1 name"

        # Second fork's branches should map to second coalesce
        assert branch_to_coalesce[BranchName("path_c")] == CoalesceName("merge2"), "path_c should map to merge2 name"
        assert branch_to_coalesce[BranchName("path_d")] == CoalesceName("merge2"), "path_d should map to merge2 name"

        # Verify all 4 branches are mapped (no missing/extra)
        assert len(branch_to_coalesce) == 4, "Should have exactly 4 branches mapped"

        # SUCCESS! Multiple independent coalesce points work correctly with the production path
