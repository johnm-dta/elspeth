# tests/engine/test_orchestrator_fork_coalesce.py
"""Tests for Orchestrator fork and coalesce functionality.

All test plugins inherit from base classes (BaseTransform, BaseGate)
because the processor uses isinstance() for type-safe plugin detection.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import SourceRow
from elspeth.plugins.base import BaseTransform
from tests.conftest import (
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.engine.orchestrator_test_helpers import build_production_graph

if TYPE_CHECKING:
    from elspeth.contracts.results import TransformResult


# =============================================================================
# Reusable Test Fixtures for Coalesce Tests
# =============================================================================


class CoalesceTestSource(_TestSourceBase):
    """Reusable source for coalesce tests that yields configurable rows.

    Unlike the null source (0 rows), this source yields actual rows for tests
    that need to exercise row processing paths.
    """

    name = "coalesce_test_source"
    output_schema = _TestSchema

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self._rows = rows or [{"value": 1}]  # Default: 1 row

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._rows:
            yield SourceRow.valid(row)


class TestOrchestratorForkExecution:
    """Test orchestrator handles fork results correctly.

    NOTE: Full fork testing at orchestrator level is blocked by ExecutionGraph
    using DiGraph instead of MultiDiGraph (can't store multiple edges between
    same nodes). See WP-07 notes. Fork logic is tested at processor level in
    test_processor.py::TestRowProcessorWorkQueue.
    """

    def test_orchestrator_handles_list_results_from_processor(self, payload_store) -> None:
        """Orchestrator correctly iterates over list[RowResult] from processor.

        This tests the basic plumbing (list handling, counting) without forks.
        Fork-specific behavior is tested at processor level.
        """
        import hashlib

        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import TransformResult

        db = LandscapeDB.in_memory()

        class RowSchema(_TestSchema):
            value: int

        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = RowSchema

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield SourceRow.valid({"value": 1})
                yield SourceRow.valid({"value": 2})
                yield SourceRow.valid({"value": 3})

            def close(self) -> None:
                pass

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: Any, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "passthrough"})

        class CollectSink(_TestSinkBase):
            name = "collect_sink"

            def __init__(self) -> None:
                self.results: list[Any] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def on_complete(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                content = str(rows).encode()
                return ArtifactDescriptor(
                    artifact_type="file",
                    path_or_uri="memory://test",
                    content_hash=hashlib.sha256(content).hexdigest(),
                    size_bytes=len(content),
                )

            def close(self) -> None:
                pass

        source = ListSource()
        transform = PassthroughTransform()
        sink = CollectSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        graph = build_production_graph(config)

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert run_result.status == "completed"
        # 3 rows from source
        assert run_result.rows_processed == 3
        # All 3 reach COMPLETED (no forks)
        assert run_result.rows_succeeded == 3
        # All 3 written to sink
        assert len(sink.results) == 3


class TestCoalesceWiring:
    """Test that coalesce is wired into orchestrator."""

    def test_orchestrator_creates_coalesce_executor_when_config_present(
        self,
        plugin_manager,
        payload_store,
    ) -> None:
        """When settings.coalesce is non-empty, CoalesceExecutor should be created."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # Use null source - no file access
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_out.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Use real plugins from instantiate_plugins_from_config
        plugins = instantiate_plugins_from_config(settings)

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # NOTE: RowProcessor mock is SUSPICIOUS - tests implementation detail
        # (that coalesce_executor kwarg exists) rather than behavior.
        # TODO: Replace with behavior-based test in Phase 5.
        with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor:
            mock_processor.return_value.process_row.return_value = []
            mock_processor.return_value.token_manager = MagicMock()

            orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # RowProcessor should have been called with coalesce_executor
            call_kwargs = mock_processor.call_args.kwargs
            assert "coalesce_executor" in call_kwargs
            assert call_kwargs["coalesce_executor"] is not None
            assert "coalesce_node_ids" in call_kwargs
            assert call_kwargs["coalesce_node_ids"] is not None
            # Verify the coalesce_node_ids contains our registered coalesce
            assert "merge_results" in call_kwargs["coalesce_node_ids"]

    def test_orchestrator_handles_coalesced_outcome(self, plugin_manager, payload_store) -> None:
        """COALESCED outcome should route merged token to output sink."""
        from unittest.mock import MagicMock, patch

        from elspeth.contracts import RowOutcome, TokenInfo
        from elspeth.contracts.results import RowResult
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Use CoalesceTestSource that yields 1 row (needed to trigger processing)
        test_source = CoalesceTestSource(rows=[{"value": 1}])

        # Settings with coalesce (needed to enable coalesce path in orchestrator)
        # Note: Settings use null for instantiate_plugins_from_config to get graph
        # structure (gates, coalesce), but we use test_source in config for actual rows
        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_coalesced.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Get plugins for graph structure
        plugins = instantiate_plugins_from_config(settings)

        # Use test_source in config (yields rows), but real plugins for sinks
        config = PipelineConfig(
            source=as_source(test_source),
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=as_source(test_source),  # Use test source in graph too
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        orchestrator = Orchestrator(db=db)

        # =====================================================================
        # MOCK CASCADE EXPLANATION (Unit Test Boundary)
        # =====================================================================
        # This test verifies Orchestrator's handling of COALESCED outcomes in
        # isolation. The mock cascade exists because:
        #
        # 1. RowProcessor mock → Returns fake COALESCED result
        # 2. Fake token_id="merged_token_1" → Not in database (no FK reference)
        # 3. record_token_outcome mock → Avoids FK constraint violation
        # 4. SinkExecutor mock → Avoids FK errors when recording sink writes
        #
        # INTEGRATION COVERAGE: Full coalesce behavior (with real tokens) is
        # tested in:
        # - test_coalesce_integration.py::test_fork_coalesce_pipeline_produces_merged_output
        # - test_processor_coalesce.py::test_fork_then_coalesce_require_all
        # - test_integration.py::test_fork_coalesce_writes_merged_to_sink
        #
        # This test specifically verifies:
        # - Orchestrator correctly counts COALESCED outcomes (rows_coalesced)
        # - Orchestrator routes merged tokens to the sink
        # =====================================================================

        merged_token = TokenInfo(
            row_id="row_1",
            token_id="merged_token_1",  # Fake - not in DB
            row_data={"merged": True},
            branch_name=None,
        )
        coalesced_result = RowResult(
            token=merged_token,
            final_data={"merged": True},
            outcome=RowOutcome.COALESCED,
        )

        with (
            patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor_cls,
            patch("elspeth.engine.executors.SinkExecutor") as mock_sink_executor_cls,
            patch("elspeth.core.landscape.LandscapeRecorder.record_token_outcome") as mock_record_outcome,
        ):
            mock_processor = MagicMock()
            mock_processor.process_row.return_value = [coalesced_result]
            mock_processor.token_manager.create_initial_token.return_value = MagicMock(row_id="row_1", token_id="t1", row_data={"value": 1})
            mock_processor_cls.return_value = mock_processor

            mock_sink_executor = MagicMock()
            mock_sink_executor_cls.return_value = mock_sink_executor

            mock_record_outcome.return_value = "mock_outcome_id"

            result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # COALESCED should count toward rows_coalesced
            assert result.rows_coalesced == 1

            # Verify the merged token was added to pending_tokens and passed to sink
            # SinkExecutor.write should have been called with the merged token
            assert mock_sink_executor.write.called
            write_call = mock_sink_executor.write.call_args
            tokens_written = write_call.kwargs.get("tokens") or write_call.args[1]
            assert len(tokens_written) == 1
            assert tokens_written[0].token_id == "merged_token_1"

    def test_orchestrator_calls_flush_pending_at_end(self, plugin_manager, payload_store) -> None:
        """flush_pending should be called on coalesce executor at end of source."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # Use null source - no file access
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_flush.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Use real plugins from settings
        plugins = instantiate_plugins_from_config(settings)

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # NOTE: CoalesceExecutor mock is LEGITIMATE here - this test verifies
        # that Orchestrator calls flush_pending() at end of source processing,
        # not that CoalesceExecutor works (tested in test_coalesce_executor.py)
        with patch("elspeth.engine.coalesce_executor.CoalesceExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            mock_executor.flush_pending.return_value = []
            mock_executor_cls.return_value = mock_executor

            orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

    def test_orchestrator_flush_pending_routes_merged_tokens_to_sink(self, plugin_manager, payload_store) -> None:
        """Merged tokens from flush_pending should be routed to output sink."""
        from unittest.mock import MagicMock, patch

        from elspeth.contracts import TokenInfo
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.coalesce_executor import CoalesceOutcome
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # Use null source - no file access
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_routes.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="best_effort",  # best_effort merges whatever arrived
                    merge="union",
                    timeout_seconds=10.0,  # Required for best_effort
                ),
            ],
        )

        # Use real plugins from settings
        plugins = instantiate_plugins_from_config(settings)

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # =====================================================================
        # MOCK CASCADE EXPLANATION (Unit Test Boundary)
        # =====================================================================
        # This test verifies Orchestrator's handling of flush_pending results.
        # The mock cascade exists because:
        #
        # 1. CoalesceExecutor mock → Returns controlled flush_pending result
        #    (LEGITIMATE: testing Orchestrator's coordination, not CoalesceExecutor)
        # 2. Fake token_id="flushed_merged_token" → Not in database
        # 3. SinkExecutor mock → Avoids FK errors when recording sink writes
        # 4. record_token_outcome mock → Avoids FK constraint violation
        #
        # INTEGRATION COVERAGE: Full flush_pending behavior (with real tokens)
        # is tested in:
        # - test_coalesce_integration.py::test_fork_coalesce_pipeline_produces_merged_output
        # - test_processor_coalesce.py tests with require_all/best_effort policies
        #
        # This test specifically verifies:
        # - Orchestrator calls flush_pending at end of source processing
        # - flush_pending results are counted in rows_coalesced
        # - Merged tokens from flush are routed to sink
        # =====================================================================

        merged_token = TokenInfo(
            row_id="row_1",
            token_id="flushed_merged_token",  # Fake - not in DB
            row_data={"merged_at_flush": True},
            branch_name=None,
        )

        with (
            patch("elspeth.engine.coalesce_executor.CoalesceExecutor") as mock_executor_cls,
            patch("elspeth.engine.executors.SinkExecutor") as mock_sink_executor_cls,
            patch("elspeth.core.landscape.LandscapeRecorder.record_token_outcome") as mock_record_outcome,
        ):
            mock_executor = MagicMock()
            mock_executor.flush_pending.return_value = [
                CoalesceOutcome(
                    held=False,
                    merged_token=merged_token,
                    consumed_tokens=[],
                    coalesce_metadata={"policy": "best_effort"},
                    coalesce_name="merge_results",
                )
            ]
            mock_executor_cls.return_value = mock_executor

            mock_sink_executor = MagicMock()
            mock_sink_executor_cls.return_value = mock_sink_executor

            mock_record_outcome.return_value = "mock_outcome_id"

            result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

            # The merged token from flush should count toward rows_coalesced
            assert result.rows_coalesced == 1

            # The merged token should be written to the sink
            assert mock_sink_executor.write.called
            write_call = mock_sink_executor.write.call_args
            tokens_written = write_call.kwargs.get("tokens") or write_call.args[1]
            assert len(tokens_written) == 1
            assert tokens_written[0].token_id == "flushed_merged_token"

    def test_orchestrator_flush_pending_handles_failures(self, plugin_manager, payload_store) -> None:
        """Failed coalesce outcomes from flush_pending should not crash."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.coalesce_executor import CoalesceOutcome
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # Use null source - no file access
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_failures.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],
                    policy="require_all",  # Will fail if not all branches arrive
                    merge="union",
                ),
            ],
        )

        # Use real plugins from settings
        plugins = instantiate_plugins_from_config(settings)

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # NOTE: CoalesceExecutor mock is LEGITIMATE here - this test verifies
        # Orchestrator's handling of failed flush_pending results without crashing.
        with patch("elspeth.engine.coalesce_executor.CoalesceExecutor") as mock_executor_cls:
            mock_executor = MagicMock()
            # flush_pending returns a failure outcome (incomplete branches)
            mock_executor.flush_pending.return_value = [
                CoalesceOutcome(
                    held=False,
                    merged_token=None,  # No merged token on failure
                    failure_reason="incomplete_branches",
                    coalesce_metadata={
                        "policy": "require_all",
                        "expected_branches": ["path_a", "path_b"],
                        "branches_arrived": ["path_a"],
                    },
                    coalesce_name="merge_results",
                )
            ]
            mock_executor_cls.return_value = mock_executor

            # Should not raise - failures are recorded but don't crash
            result = orchestrator.run(config, graph=graph, settings=settings, payload_store=payload_store)

            # flush_pending should have been called
            mock_executor.flush_pending.assert_called_once()

            # No merged tokens means no rows_coalesced increment
            assert result.rows_coalesced == 0

    def test_orchestrator_computes_coalesce_step_map(self, plugin_manager, payload_store) -> None:
        """Orchestrator should compute step positions for each coalesce point."""

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
            TransformSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),  # Use null source - no file access
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test_stepmap.json", "schema": {"fields": "dynamic"}})},
            default_sink="output",
            transforms=[
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),  # Step 0
                TransformSettings(plugin="passthrough", options={"schema": {"fields": "dynamic"}}),  # Step 1
            ],
            gates=[
                GateSettings(
                    name="forker",  # Step 2
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["path_a", "path_b"],
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",  # Step 3
                    branches=["path_a", "path_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        # Use real plugins from settings
        plugins = instantiate_plugins_from_config(settings)

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db=db)

        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        # Test the step map computation directly instead of mocking RowProcessor
        # This verifies the calculation without needing to intercept constructor args
        step_map = orchestrator._compute_coalesce_step_map(graph, config, settings)

        # forker gate is at pipeline index 2 (after 2 transforms)
        # coalesce_step = len(transforms) + len(gates) + coalesce_index
        #               = 2 + 1 + 0 = 3
        assert "merge_results" in step_map
        assert step_map["merge_results"] == 3


class TestCoalesceStepMapCalculation:
    """Test that coalesce_step_map is computed from graph topology."""

    def test_coalesce_step_map_uses_graph_gate_index(
        self,
        plugin_manager,
    ) -> None:
        """coalesce_step_map places coalesce AFTER all transforms and gates.

        Given:
          - Pipeline with fork_gate at index 0 (first gate)
          - downstream_gate at index 1 (second gate)
          - Coalesce for fork_gate's branches

        The coalesce_step should be len(transforms) + len(gates) + coalesce_index,
        which places coalesce steps AFTER all gates. This avoids step index
        collisions when there are gates after the fork gate.

        Fork children skip directly to this coalesce step, bypassing all
        intermediate transforms and gates.
        """

        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.contracts.types import CoalesceName
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
                GateSettings(
                    name="downstream_gate",  # Gate AFTER fork
                    condition="False",
                    routes={"true": "output", "false": "continue"},
                ),
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json", "schema": {"fields": "dynamic"}})},
            coalesce=[
                CoalesceSettings(
                    name="merge_branches",
                    branches=["branch_a", "branch_b"],
                    policy="require_all",
                    merge="union",
                ),
            ],
            default_sink="output",
        )

        db = LandscapeDB.in_memory()
        orchestrator = Orchestrator(db)

        # Use real plugins from settings
        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            default_sink=settings.default_sink,
            coalesce_settings=settings.coalesce,
        )

        config = PipelineConfig(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            gates=settings.gates,
        )

        # Test the step map computation directly - no mocking needed
        step_map = orchestrator._compute_coalesce_step_map(graph, config, settings)

        # With Option B (execution matches graph topology), coalesce step uses
        # the gate index from the graph. The coalesce step is ONE AFTER the
        # fork gate, allowing merged tokens to continue downstream processing.
        #
        # fork_gate is at index 0 in config.gates, so:
        #   coalesce_step = gate_idx + 1 = 0 + 1 = 1
        #
        # This allows merged tokens to process through downstream_gate (step 1).
        assert CoalesceName("merge_branches") in step_map

        # coalesce_step = gate_idx + 1 where gate_idx is from graph.get_coalesce_gate_index()
        # fork_gate produces merge_branches branches, fork_gate is at gate index 0
        expected_step = 0 + 1  # gate_idx(0) + 1 = 1
        assert step_map[CoalesceName("merge_branches")] == expected_step
