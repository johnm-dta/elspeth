# Coalesce Integration Implementation Plan (Revised)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing CoalesceSettings config, CoalesceExecutor, and RowProcessor coalesce support into the DAG compiler and Orchestrator so fork/join pipelines actually work.

**Architecture:** The coalesce feature is 80% implemented but not integrated. CoalesceExecutor handles token merging, RowProcessor has coalesce infrastructure (`_WorkItem` already has `coalesce_at_step` and `coalesce_name` fields), config validates correctly. Missing: DAG compiler doesn't create coalesce nodes, Orchestrator doesn't wire the executor or handle COALESCED outcomes.

**Tech Stack:** Python 3.11+, Pydantic, pluggy, SQLAlchemy, pytest

**Reference Docs:**
- `docs/contracts/plugin-protocol.md:911-985` - Coalesce specification
- `docs/bugs/open/2026-01-19-coalesce-config-ignored.md` - Bug report with full analysis

**Key Verified Facts:**
- `NodeType.COALESCE` already exists at `src/elspeth/contracts/enums.py:86`
- `_WorkItem` already has `coalesce_at_step` and `coalesce_name` at `processor.py:46-47`
- `register_coalesce(settings, node_id)` signature confirmed at `coalesce_executor.py:98-110`
- `ExecutionGraph.__init__` pattern: initialize private maps in constructor (lines 48-60)

---

## Task 1: Add Coalesce Node Creation to DAG Compiler

**Files:**
- Modify: `src/elspeth/core/dag.py:48-60` (add to `__init__`)
- Modify: `src/elspeth/core/dag.py:245-435` (inside `from_config()`)
- Modify: `src/elspeth/core/dag.py:462+` (add accessor methods)
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
class TestCoalesceNodes:
    """Test coalesce node creation in DAG."""

    def test_from_config_creates_coalesce_node(self) -> None:
        """Coalesce config should create a coalesce node in the graph."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={
                "output": SinkSettings(plugin="csv", options={"path": "out.csv"}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
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

        graph = ExecutionGraph.from_config(settings)

        # Use proper accessor, not string matching
        coalesce_map = graph.get_coalesce_id_map()
        assert "merge_results" in coalesce_map

        # Verify node type
        node_id = coalesce_map["merge_results"]
        node_info = graph.get_node(node_id)
        assert node_info.node_type == "coalesce"
        assert node_info.plugin_name == "coalesce:merge_results"

    def test_from_config_coalesce_edges_from_fork_branches(self) -> None:
        """Coalesce node should have edges from fork gate (via branches)."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={
                "output": SinkSettings(plugin="csv", options={"path": "out.csv"}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
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

        graph = ExecutionGraph.from_config(settings)

        # Find coalesce node
        coalesce_id = graph.get_coalesce_id_map()["merge_results"]

        # Should have edges from fork gate to coalesce (one per branch)
        edges = graph.get_edges()
        incoming_edges = [e for e in edges if e.to_node == coalesce_id]
        assert len(incoming_edges) == 2
        assert {e.label for e in incoming_edges} == {"path_a", "path_b"}

    def test_partial_branch_coverage_branches_not_in_coalesce_route_to_sink(self) -> None:
        """Fork branches not in any coalesce should still route to output sink."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={
                "output": SinkSettings(plugin="csv", options={"path": "out.csv"}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
                    fork_to=["path_a", "path_b", "path_c"],  # 3 branches
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches=["path_a", "path_b"],  # Only 2 branches coalesce
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        graph = ExecutionGraph.from_config(settings)

        # path_c should route to output sink, not coalesce
        branch_to_coalesce = graph.get_branch_to_coalesce_map()
        assert "path_a" in branch_to_coalesce
        assert "path_b" in branch_to_coalesce
        assert "path_c" not in branch_to_coalesce  # Not in any coalesce

    def test_get_coalesce_id_map_returns_mapping(self) -> None:
        """get_coalesce_id_map should return coalesce_name -> node_id."""
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
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

        graph = ExecutionGraph.from_config(settings)
        coalesce_map = graph.get_coalesce_id_map()

        assert "merge_results" in coalesce_map
        assert coalesce_map["merge_results"].startswith("coalesce_")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_dag.py::TestCoalesceNodes -v`
Expected: FAIL with "get_coalesce_id_map" not found or KeyError

**Step 3: Write minimal implementation**

In `src/elspeth/core/dag.py`, add to `__init__` (after line 53):

```python
        self._coalesce_id_map: dict[str, str] = {}  # coalesce_name -> node_id
        self._branch_to_coalesce: dict[str, str] = {}  # branch_name -> coalesce_name
```

In `from_config()`, add after the config gate section (after line 419, before output sink edge):

```python
        # Build coalesce nodes (processed AFTER config gates)
        # Coalesce merges tokens from fork paths. It sits between fork gate and output sink.
        coalesce_ids: dict[str, str] = {}
        branch_to_coalesce: dict[str, str] = {}  # branch_name -> coalesce_name

        for coalesce_config in config.coalesce:
            cid = node_id("coalesce", coalesce_config.name)
            coalesce_ids[coalesce_config.name] = cid

            # Track which branches lead to this coalesce
            for branch in coalesce_config.branches:
                branch_to_coalesce[branch] = coalesce_config.name

            # Store config in node for audit trail
            coalesce_node_config = {
                "branches": list(coalesce_config.branches),
                "policy": coalesce_config.policy,
                "merge": coalesce_config.merge,
                "timeout_seconds": coalesce_config.timeout_seconds,
                "quorum_count": coalesce_config.quorum_count,
                "select_branch": coalesce_config.select_branch,
            }

            graph.add_node(
                cid,
                node_type="coalesce",
                plugin_name=f"coalesce:{coalesce_config.name}",
                config=coalesce_node_config,
            )

        # Store mappings using proper instance attributes (not setattr)
        graph._coalesce_id_map = coalesce_ids
        graph._branch_to_coalesce = branch_to_coalesce

        # Create edges from fork gates to coalesce nodes
        # Only branches that are in a coalesce's branches list route to that coalesce
        # Other branches continue to output_sink (handled by existing fork edge logic)
        for gate_config in config.gates:
            if gate_config.fork_to:
                gate_id = config_gate_ids[gate_config.name]
                for branch in gate_config.fork_to:
                    if branch in branch_to_coalesce:
                        coalesce_name = branch_to_coalesce[branch]
                        coalesce_id = coalesce_ids[coalesce_name]
                        # Edge from gate to coalesce (replaces edge to output_sink for this branch)
                        graph.add_edge(
                            gate_id,
                            coalesce_id,
                            label=branch,
                            mode=RoutingMode.COPY,
                        )
                    # Branches NOT in any coalesce keep their existing edge to output_sink
                    # (already created by the fork handling above)

        # Create edges from coalesce nodes to output sink
        for coalesce_name, cid in coalesce_ids.items():
            graph.add_edge(
                cid,
                output_sink_node,
                label="continue",
                mode=RoutingMode.MOVE,
            )
```

Add accessor methods after `get_aggregation_id_map()` (around line 470):

```python
    def get_coalesce_id_map(self) -> dict[str, str]:
        """Get explicit coalesce_name -> node_id mapping.

        Returns:
            Dict mapping coalesce name to its graph node ID.
        """
        return dict(self._coalesce_id_map)

    def get_branch_to_coalesce_map(self) -> dict[str, str]:
        """Get branch_name -> coalesce_name mapping.

        Used to determine which fork branches should route to coalesce points.
        Branches NOT in this map route directly to sinks.

        Returns:
            Dict mapping fork branch names to their coalesce point names.
        """
        return dict(self._branch_to_coalesce)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_dag.py::TestCoalesceNodes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "feat(dag): add coalesce node creation in from_config()

Coalesce nodes are now created when config.coalesce is non-empty.
Edges are created from fork gates to coalesce nodes based on matching
branch names. Branches not in any coalesce route to output sink.

Closes part of coalesce-config-ignored bug."
```

---

## ~~Task 2: Register NodeType.COALESCE~~ **DELETED**

**Reason:** Already exists at `src/elspeth/contracts/enums.py:86`

---

## Task 2: Wire CoalesceExecutor into Orchestrator (was Task 3)

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:559-570` (RowProcessor creation)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
class TestCoalesceWiring:
    """Test that coalesce is wired into orchestrator."""

    def test_orchestrator_creates_coalesce_executor_when_config_present(
        self,
        db: LandscapeDB,
    ) -> None:
        """When settings.coalesce is non-empty, CoalesceExecutor should be created."""
        from unittest.mock import patch, MagicMock
        from elspeth.core.config import (
            CoalesceSettings,
            DatasourceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
        )
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
            sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
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

        # Mock source/sink to avoid file access
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([])
        mock_source.plugin_version = "1.0.0"
        mock_source.determinism = "deterministic"

        mock_sink = MagicMock()
        mock_sink.name = "csv"
        mock_sink.plugin_version = "1.0.0"
        mock_sink.determinism = "deterministic"

        config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={"output": mock_sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        orchestrator = Orchestrator(db=db)

        # Patch RowProcessor to capture its args
        with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor:
            mock_processor.return_value.process_row.return_value = []
            mock_processor.return_value.token_manager = MagicMock()

            orchestrator.execute(config, settings=settings)

            # RowProcessor should have been called with coalesce_executor
            call_kwargs = mock_processor.call_args.kwargs
            assert "coalesce_executor" in call_kwargs
            assert call_kwargs["coalesce_executor"] is not None
            assert "coalesce_node_ids" in call_kwargs
            assert call_kwargs["coalesce_node_ids"] == {"merge_results": mock.ANY}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator.py::TestCoalesceWiring -v`
Expected: FAIL (coalesce_executor is None or missing)

**Step 3: Write minimal implementation**

In `src/elspeth/engine/orchestrator.py`, add import at top:

```python
from elspeth.engine.coalesce_executor import CoalesceExecutor
```

In `_execute_run()`, before RowProcessor creation (around line 555):

```python
        # Create coalesce executor if config has coalesce settings
        coalesce_executor: CoalesceExecutor | None = None
        coalesce_node_ids: dict[str, str] = {}
        branch_to_coalesce: dict[str, str] = {}

        if settings is not None and settings.coalesce:
            # Reuse the existing TokenManager from processor (created below)
            # For now, we create the executor without token_manager and set it after
            # Actually, looking at CoalesceExecutor, it needs TokenManager at init
            # So we create TokenManager first, then both executor and processor use it
            from elspeth.engine.tokens import TokenManager
            token_manager = TokenManager(recorder)

            coalesce_executor = CoalesceExecutor(
                recorder=recorder,
                span_factory=self._span_factory,
                token_manager=token_manager,
                run_id=run_id,
            )

            # Register each coalesce point
            coalesce_node_ids = graph.get_coalesce_id_map()
            branch_to_coalesce = graph.get_branch_to_coalesce_map()
            for coalesce_settings in settings.coalesce:
                node_id = coalesce_node_ids.get(coalesce_settings.name)
                if node_id:
                    coalesce_executor.register_coalesce(coalesce_settings, node_id)
```

Modify RowProcessor creation to include coalesce params:

```python
        processor = RowProcessor(
            recorder=recorder,
            span_factory=self._span_factory,
            run_id=run_id,
            source_node_id=source_id,
            edge_map=edge_map,
            route_resolution_map=route_resolution_map,
            config_gates=config.gates,
            config_gate_id_map=config_gate_id_map,
            aggregation_settings=config.aggregation_settings,
            retry_manager=retry_manager,
            coalesce_executor=coalesce_executor,
            coalesce_node_ids=coalesce_node_ids,
            branch_to_coalesce=branch_to_coalesce,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator.py::TestCoalesceWiring -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(orchestrator): wire CoalesceExecutor into RowProcessor

When settings.coalesce is non-empty, create CoalesceExecutor and pass
it to RowProcessor. Register each coalesce point with its settings.
Pass branch_to_coalesce mapping for fork→coalesce linkage."
```

---

## Task 3: Handle COALESCED Outcome in Orchestrator (was Task 4)

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:66-78` (RunResult dataclass)
- Modify: `src/elspeth/engine/orchestrator.py:637-675` (result handling loop)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_orchestrator_handles_coalesced_outcome(self, db: LandscapeDB) -> None:
    """COALESCED outcome should route merged token to output sink."""
    from unittest.mock import MagicMock, patch
    from elspeth.contracts import TokenInfo, RowOutcome
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
    from elspeth.engine.processor import RowResult

    mock_source = MagicMock()
    mock_source.name = "csv"
    mock_source.load.return_value = iter([
        MagicMock(is_quarantined=False, row={"value": 1})
    ])
    mock_source.plugin_version = "1.0.0"
    mock_source.determinism = "deterministic"

    mock_sink = MagicMock()
    mock_sink.name = "csv"
    mock_sink.plugin_version = "1.0.0"
    mock_sink.determinism = "deterministic"

    config = PipelineConfig(
        source=mock_source,
        transforms=[],
        sinks={"output": mock_sink},
        gates=[],
        coalesce_settings=[],
        aggregation_settings={},
        config={},
    )

    orchestrator = Orchestrator(db=db)

    # Mock RowProcessor to return COALESCED outcome
    merged_token = TokenInfo(
        row_id="row_1",
        token_id="merged_token_1",
        row_data={"merged": True},
        branch_name=None,
    )
    coalesced_result = RowResult(
        token=merged_token,
        final_data={"merged": True},
        outcome=RowOutcome.COALESCED,
    )

    with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor_cls:
        mock_processor = MagicMock()
        mock_processor.process_row.return_value = [coalesced_result]
        mock_processor.token_manager.create_initial_token.return_value = MagicMock(
            row_id="row_1", token_id="t1", row_data={"value": 1}
        )
        mock_processor_cls.return_value = mock_processor

        result = orchestrator.execute(config)

        # COALESCED should count toward rows_coalesced
        assert result.rows_coalesced == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_handles_coalesced_outcome -v`
Expected: FAIL (rows_coalesced doesn't exist or COALESCED not handled)

**Step 3: Write minimal implementation**

Modify `RunResult` dataclass (around line 66-78):

```python
@dataclass
class RunResult:
    """Result of a pipeline run."""

    run_id: str
    status: RunStatus
    rows_processed: int
    rows_succeeded: int
    rows_failed: int
    rows_routed: int
    rows_quarantined: int = 0
    rows_forked: int = 0
    rows_coalesced: int = 0  # ADD THIS
```

Add tracking variable in `_execute_run()` (around line 580):

```python
        rows_coalesced = 0
```

In the result handling loop (add after CONSUMED_IN_BATCH handling, around line 674):

```python
                        elif result.outcome == RowOutcome.COALESCED:
                            # Merged token from coalesce - route to output sink
                            # Note: Don't increment rows_succeeded - the fork children
                            # were already counted when they entered the fork.
                            # The merged token is the continuation, not a new success.
                            rows_coalesced += 1
                            pending_tokens[output_sink_name].append(result.token)

                            # Checkpoint with coalesce node ID
                            coalesce_name = getattr(result, 'coalesce_name', None)
                            if coalesce_name and coalesce_name in coalesce_node_ids:
                                checkpoint_node_id = coalesce_node_ids[coalesce_name]
                            else:
                                checkpoint_node_id = last_node_id
                            self._maybe_checkpoint(
                                run_id=run_id,
                                token_id=result.token.token_id,
                                node_id=checkpoint_node_id,
                            )
```

Update RunResult return at end of function:

```python
        return RunResult(
            run_id=run_id,
            status=RunStatus.RUNNING,
            rows_processed=rows_processed,
            rows_succeeded=rows_succeeded,
            rows_failed=rows_failed,
            rows_routed=rows_routed,
            rows_quarantined=rows_quarantined,
            rows_forked=rows_forked,
            rows_coalesced=rows_coalesced,  # ADD THIS
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_handles_coalesced_outcome -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(orchestrator): handle COALESCED outcome

Merged tokens from coalesce operations route to output sink.
Added rows_coalesced to RunResult for tracking.

Note: rows_coalesced tracks merges, not additional successes.
Fork children were already counted when they entered the fork."
```

---

## Task 4: Call flush_pending at End of Run (was Task 5)

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (after source loop, before sink writes)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_orchestrator_calls_flush_pending_at_end(self, db: LandscapeDB) -> None:
    """flush_pending should be called on coalesce executor at end of source."""
    from unittest.mock import MagicMock, patch
    from elspeth.core.config import (
        CoalesceSettings,
        DatasourceSettings,
        ElspethSettings,
        GateSettings,
        SinkSettings,
    )
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
        sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
        output_sink="output",
        gates=[
            GateSettings(
                name="forker",
                condition="True",
                routes={"true": "fork"},
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

    mock_source = MagicMock()
    mock_source.name = "csv"
    mock_source.load.return_value = iter([])  # Empty - immediate end
    mock_source.plugin_version = "1.0.0"
    mock_source.determinism = "deterministic"

    mock_sink = MagicMock()
    mock_sink.name = "csv"
    mock_sink.plugin_version = "1.0.0"
    mock_sink.determinism = "deterministic"

    config = PipelineConfig(
        source=mock_source,
        transforms=[],
        sinks={"output": mock_sink},
        gates=settings.gates,
        coalesce_settings=settings.coalesce,
        aggregation_settings={},
        config={},
    )

    orchestrator = Orchestrator(db=db)

    with patch("elspeth.engine.orchestrator.CoalesceExecutor") as mock_executor_cls:
        mock_executor = MagicMock()
        mock_executor.flush_pending.return_value = []
        mock_executor_cls.return_value = mock_executor

        orchestrator.execute(config, settings=settings)

        # flush_pending should have been called with step parameter
        mock_executor.flush_pending.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_calls_flush_pending_at_end -v`
Expected: FAIL (flush_pending not called)

**Step 3: Write minimal implementation**

In `src/elspeth/engine/orchestrator.py`, after the source processing loop but before sink writes (around line 676):

```python
            # Flush pending coalesce operations at end-of-source
            if coalesce_executor is not None:
                # Coalesce step position in pipeline
                # This is after all transforms and gates, at the coalesce position
                coalesce_step = len(config.transforms) + len(config.gates) + 1
                pending_outcomes = coalesce_executor.flush_pending(coalesce_step)

                # Handle any merged tokens from flush
                for outcome in pending_outcomes:
                    if outcome.merged_token is not None:
                        # Successful merge - route to output sink
                        rows_coalesced += 1
                        pending_tokens[output_sink_name].append(outcome.merged_token)
                    elif outcome.failure_reason:
                        # Coalesce failed (timeout, missing branches)
                        # consumed_tokens were already counted when they entered fork
                        # Record failure but don't double-count rows
                        # The individual tokens are already terminal (held in coalesce)
                        pass  # Failure is recorded in audit trail by executor
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_calls_flush_pending_at_end -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(orchestrator): call flush_pending at end of source

Ensures pending coalesce operations are resolved when source exhausts.
Merged tokens route to output sink. Failures are recorded in audit trail."
```

---

## Task 5: Link Fork Gates to Coalesce Points (was Task 6)

**Files:**
- Modify: `src/elspeth/engine/processor.py:85-133` (add params to `__init__`)
- Modify: `src/elspeth/engine/processor.py:573-593` (fork child handling)
- Test: `tests/engine/test_processor.py`

**Note:** `_WorkItem` already has `coalesce_at_step` and `coalesce_name` fields at lines 46-47.

**Step 1: Write the failing test**

Add to `tests/engine/test_processor.py`:

```python
class TestCoalesceLinkage:
    """Test fork -> coalesce linkage."""

    def test_processor_accepts_coalesce_mapping_params(
        self,
        recorder: LandscapeRecorder,
        run: Run,
    ) -> None:
        """RowProcessor should accept branch_to_coalesce and coalesce_step_map."""
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory

        # Should not raise - params are accepted
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id="source_1",
            branch_to_coalesce={"path_a": "merge_point"},
            coalesce_step_map={"merge_point": 3},
        )

        assert processor._branch_to_coalesce == {"path_a": "merge_point"}
        assert processor._coalesce_step_map == {"merge_point": 3}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_processor.py::TestCoalesceLinkage -v`
Expected: FAIL (unexpected keyword argument)

**Step 3: Write minimal implementation**

In `src/elspeth/engine/processor.py`, add to `__init__` parameters (after line 94):

```python
        branch_to_coalesce: dict[str, str] | None = None,
        coalesce_step_map: dict[str, int] | None = None,
```

Add to docstring:

```python
            branch_to_coalesce: Map of branch_name -> coalesce_name for fork linkage
            coalesce_step_map: Map of coalesce_name -> step position in pipeline
```

Store in `__init__` body (after line 122):

```python
        self._branch_to_coalesce = branch_to_coalesce or {}
        self._coalesce_step_map = coalesce_step_map or {}
```

In the fork handling code (around line 573-593), when creating child work items, update to use the mappings:

```python
                    for child_token in outcome.child_tokens:
                        # Look up coalesce info for this branch
                        branch_name = child_token.branch_name
                        child_coalesce_name: str | None = None
                        child_coalesce_step: int | None = None

                        if branch_name and branch_name in self._branch_to_coalesce:
                            child_coalesce_name = self._branch_to_coalesce[branch_name]
                            child_coalesce_step = self._coalesce_step_map.get(child_coalesce_name)

                        child_items.append(
                            _WorkItem(
                                token=child_token,
                                start_step=next_step,
                                coalesce_at_step=child_coalesce_step,
                                coalesce_name=child_coalesce_name,
                            )
                        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_processor.py::TestCoalesceLinkage -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(processor): link fork children to coalesce points

Fork children now receive coalesce_at_step and coalesce_name based on
branch_to_coalesce mapping. Enables coalesce executor to know when
tokens should merge."
```

---

## Task 6: Compute Coalesce Step Positions in Orchestrator (was part of Task 6)

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (compute coalesce_step_map)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

```python
def test_orchestrator_computes_coalesce_step_map(self, db: LandscapeDB) -> None:
    """Orchestrator should compute step positions for each coalesce point."""
    from unittest.mock import MagicMock, patch
    from elspeth.core.config import (
        CoalesceSettings,
        DatasourceSettings,
        ElspethSettings,
        GateSettings,
        SinkSettings,
        RowPluginSettings,
    )
    from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

    settings = ElspethSettings(
        datasource=DatasourceSettings(plugin="csv", options={"path": "test.csv"}),
        sinks={"output": SinkSettings(plugin="csv", options={"path": "out.csv"})},
        output_sink="output",
        row_plugins=[
            RowPluginSettings(plugin="passthrough"),  # Step 0
            RowPluginSettings(plugin="passthrough"),  # Step 1
        ],
        gates=[
            GateSettings(
                name="forker",  # Step 2
                condition="True",
                routes={"true": "fork"},
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

    mock_source = MagicMock()
    mock_source.name = "csv"
    mock_source.load.return_value = iter([])
    mock_source.plugin_version = "1.0.0"
    mock_source.determinism = "deterministic"

    mock_sink = MagicMock()
    mock_sink.name = "csv"
    mock_sink.plugin_version = "1.0.0"
    mock_sink.determinism = "deterministic"

    mock_transform = MagicMock()
    mock_transform.name = "passthrough"
    mock_transform.plugin_version = "1.0.0"
    mock_transform.determinism = "deterministic"
    mock_transform.is_batch_aware = False

    config = PipelineConfig(
        source=mock_source,
        transforms=[mock_transform, mock_transform],
        sinks={"output": mock_sink},
        gates=settings.gates,
        coalesce_settings=settings.coalesce,
        aggregation_settings={},
        config={},
    )

    orchestrator = Orchestrator(db=db)

    with patch("elspeth.engine.orchestrator.RowProcessor") as mock_processor_cls:
        mock_processor = MagicMock()
        mock_processor.process_row.return_value = []
        mock_processor_cls.return_value = mock_processor

        orchestrator.execute(config, settings=settings)

        # Check coalesce_step_map was passed
        call_kwargs = mock_processor_cls.call_args.kwargs
        assert "coalesce_step_map" in call_kwargs
        # 2 transforms + 1 gate = step 3 for coalesce
        assert call_kwargs["coalesce_step_map"]["merge_results"] == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_computes_coalesce_step_map -v`
Expected: FAIL (coalesce_step_map not in call or wrong value)

**Step 3: Write minimal implementation**

In orchestrator, before RowProcessor creation, add:

```python
        # Compute coalesce step positions
        # Coalesce step = after all transforms and gates
        coalesce_step_map: dict[str, int] = {}
        if settings is not None and settings.coalesce:
            base_step = len(config.transforms) + len(config.gates)
            for i, cs in enumerate(settings.coalesce):
                # Each coalesce gets its own step (in case of multiple)
                coalesce_step_map[cs.name] = base_step + i + 1
```

Pass to RowProcessor:

```python
            coalesce_step_map=coalesce_step_map,
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_computes_coalesce_step_map -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "feat(orchestrator): compute coalesce step positions

Step positions are computed as transforms + gates + coalesce index.
Passed to RowProcessor for fork→coalesce linkage."
```

---

## Task 7: Add coalesce_settings to PipelineConfig (was Task 7)

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (PipelineConfig dataclass)
- Test: Verify existing tests pass

**Step 1: Check current PipelineConfig**

```python
# If coalesce_settings is missing, add it
@dataclass
class PipelineConfig:
    source: SourceProtocol
    transforms: list[BaseTransform]
    sinks: dict[str, SinkProtocol]
    gates: list[GateSettings]
    coalesce_settings: list[CoalesceSettings]  # ADD IF MISSING
    aggregation_settings: dict[str, AggregationSettings]
    config: dict[str, Any]
```

**Step 2: Run all tests**

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: PASS

**Step 3: Commit (if changes made)**

```bash
git add src/elspeth/engine/orchestrator.py
git commit -m "feat(orchestrator): add coalesce_settings to PipelineConfig"
```

---

## Task 8: Integration Test - Full Fork/Join Pipeline (was Task 8)

**Files:**
- Create: `tests/engine/test_coalesce_integration.py`

**Step 1: Write comprehensive integration test**

```python
"""Integration tests for fork/coalesce pipelines."""

import pytest
from typing import Any

from elspeth.contracts import RowOutcome
from elspeth.core.config import (
    CoalesceSettings,
    DatasourceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sources.memory_source import MemorySource
from elspeth.plugins.sinks.memory_sink import MemorySink


class TestForkCoalescePipeline:
    """Test complete fork -> process -> coalesce -> sink flow."""

    @pytest.fixture
    def db(self) -> LandscapeDB:
        return LandscapeDB.in_memory()

    def test_fork_coalesce_pipeline_produces_merged_output(
        self,
        db: LandscapeDB,
    ) -> None:
        """Complete fork/join pipeline should produce merged output."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="memory",
                options={"data": [{"id": 1, "value": 100}]},
            ),
            sinks={
                "output": SinkSettings(plugin="memory", options={}),
            },
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
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

        source = MemorySource({"data": [{"id": 1, "value": 100}]})
        sink = MemorySink({})

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        orchestrator = Orchestrator(db=db)
        result = orchestrator.execute(config, settings=settings)

        # Should have processed rows
        assert result.rows_processed == 1
        assert result.rows_coalesced == 1

        # Sink should have received merged output
        assert len(sink.rows) >= 1
        merged = sink.rows[0]
        assert merged["id"] == 1
        assert merged["value"] == 100

    def test_partial_branch_coverage_non_coalesced_branches_reach_sink(
        self,
        db: LandscapeDB,
    ) -> None:
        """Branches not in coalesce should still reach output sink."""
        settings = ElspethSettings(
            datasource=DatasourceSettings(
                plugin="memory",
                options={"data": [{"id": 1}]},
            ),
            sinks={"output": SinkSettings(plugin="memory", options={})},
            output_sink="output",
            gates=[
                GateSettings(
                    name="forker",
                    condition="True",
                    routes={"true": "fork"},
                    fork_to=["path_a", "path_b", "path_c"],  # 3 branches
                ),
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_ab",
                    branches=["path_a", "path_b"],  # Only 2 coalesce
                    policy="require_all",
                    merge="union",
                ),
            ],
        )

        source = MemorySource({"data": [{"id": 1}]})
        sink = MemorySink({})

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"output": sink},
            gates=settings.gates,
            coalesce_settings=settings.coalesce,
            aggregation_settings={},
            config={},
        )

        orchestrator = Orchestrator(db=db)
        result = orchestrator.execute(config, settings=settings)

        # Should have:
        # - 1 merged token from path_a + path_b
        # - 1 direct token from path_c
        assert result.rows_processed == 1
        assert result.rows_coalesced == 1
        assert len(sink.rows) == 2  # Merged + path_c
```

**Step 2: Run integration test**

Run: `pytest tests/engine/test_coalesce_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/engine/test_coalesce_integration.py
git commit -m "test(integration): add fork/coalesce pipeline integration tests

Tests complete flow: source -> fork -> parallel paths -> coalesce -> sink
Verifies merged output and partial branch coverage."
```

---

## Task 9: Update Documentation (was Task 10)

**Files:**
- Move: `docs/bugs/open/2026-01-19-coalesce-config-ignored.md` → `docs/bugs/closed/`
- Modify: `docs/design/requirements.md`

**Step 1: Close bug report**

Add resolution section:

```markdown
## Resolution

- **Fixed in:** [commit hash]
- **Date:** 2026-01-20
- **Resolution:** Full coalesce integration implemented:
  - DAG compiler creates coalesce nodes (Task 1)
  - Orchestrator wires CoalesceExecutor (Task 2)
  - COALESCED outcome handled (Task 3)
  - flush_pending called at end-of-source (Task 4)
  - Fork children linked to coalesce points (Tasks 5-6)
  - Integration tests verify end-to-end flow (Task 8)
```

**Step 2: Commit**

```bash
git mv docs/bugs/open/2026-01-19-coalesce-config-ignored.md docs/bugs/closed/
git add docs/bugs/closed/2026-01-19-coalesce-config-ignored.md docs/design/requirements.md
git commit -m "docs: close coalesce-config-ignored bug, update requirements"
```

---

## Summary (Revised)

| Task | Est. Time | Description |
|------|-----------|-------------|
| **1** | 30 min | DAG compiler creates coalesce nodes + edges |
| ~~2~~ | — | ~~DELETED - NodeType.COALESCE already exists~~ |
| **2** | 30 min | Wire CoalesceExecutor into Orchestrator |
| **3** | 20 min | Handle COALESCED outcome + add to RunResult |
| **4** | 20 min | Call flush_pending at end of source |
| **5** | 30 min | Add branch_to_coalesce params to RowProcessor |
| **6** | 15 min | Compute coalesce_step_map in Orchestrator |
| **7** | 10 min | Add coalesce_settings to PipelineConfig |
| **8** | 30 min | Integration tests |
| **9** | 10 min | Documentation updates |

**Total:** ~3 hours

---

## Key Fixes from Review

1. **Removed Task 2** - NodeType.COALESCE already exists
2. **Fixed test patterns** - Use `node_info.node_type == "coalesce"` not string matching
3. **Follow encapsulation** - Initialize `_coalesce_id_map` in `__init__`
4. **Added partial branch test** - Branches not in coalesce still work
5. **Fixed counting** - rows_coalesced tracks merges, not additional successes
6. **Verified signatures** - register_coalesce(settings, node_id) confirmed
7. **Confirmed _WorkItem** - Already has coalesce fields, no changes needed
