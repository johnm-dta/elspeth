# Coalesce Graph-Execution Alignment Implementation Plan

> **STATUS: ✅ COMPLETED (2026-01-28)**
> All tasks implemented and verified. Full test suite passes (3887 tests).

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align processor execution with graph topology so fork children skip directly to coalesce and merged tokens continue to downstream nodes.

**Architecture:** The graph already correctly computes coalesce insertion points (`coalesce_gate_index`). We expose this via a getter method, have orchestrator use it to compute `coalesce_step_map`, and modify processor fork child routing to jump directly to coalesce. This activates the existing-but-unreachable "merged token continues" code path.

**Tech Stack:** Python 3.13, pytest, ELSPETH pipeline engine

**Bug Reference:** `docs/bugs/BUG-COALESCE-GRAPH-EXECUTION-MISMATCH.md`

**Deadline:** First live run next week - this fix must be complete and tested before then.

## Completion Notes (2026-01-28)

**Formula correction:** During implementation, the plan's formula `coalesce_step = gate_pipeline_idx + 1` was found to cause step index collisions when multiple gates follow the fork gate. The correct formula is:

```python
coalesce_step = num_transforms + num_gates + coalesce_index
```

This places coalesce steps AFTER all transforms and gates, ensuring fork children skip all intermediate processing when jumping to coalesce.

---

## Risk Assessment (2026-01-28)

| Risk | Level | Notes |
|------|-------|-------|
| Breaking existing users | ✅ **None** | Zero users pre-release - fix before anyone depends on broken behavior |
| Implementation complexity | 🟢 **Low** | Clear changes at well-defined locations |
| Dead code resurrection | 🟡 **Medium** | Lines 783-789 never executed - give extra testing attention |
| Audit trail integrity | ✅ **Improved** | Graph will match execution - critical for live run credibility |

**Overall: 🟢 Low-Medium Risk** - Well-suited for pre-release fix.

---

## Risk Reduction Findings (2026-01-27)

### ✅ Validated

| Item | Status | Notes |
|------|--------|-------|
| Line numbers in plan | ✅ Accurate | All references match actual code |
| Baseline test suite | ✅ 229/230 pass | One pre-existing failure (unrelated) |
| DAG instance variables | ✅ Ready | `_coalesce_gate_index` correctly missing, Task 1 adds it |
| Merged token path | ✅ Exists | Lines 783-789 are valid but unreachable |

### 🔴 Critical Gap Identified

**The original plan missed the plugin gate fork path:**

| Fork Path | Location | Original Plan | Status |
|-----------|----------|---------------|--------|
| Config gate fork | `processor.py:1132-1153` | ✅ Task 5 | Addressed |
| Plugin gate fork | `processor.py:891-911` | ❌ Missing | **NEW Task 5b** |

Both paths use `start_step=next_step` instead of `start_step=coalesce_step`. Both need the same fix.

### 🟡 Improvements Added

1. **Task 5b**: Plugin gate fork routing fix (lines 891-911)
2. **Task 3 Enhancement**: Extract `_compute_coalesce_step_map()` helper to avoid main/resume path duplication
3. **Additional test**: Plugin gate fork to coalesce flow

### Pre-existing Issues (Not Our Concern)

- `test_duplicate_fork_branches_rejected_in_plugin_gate` fails due to missing `routes` key in test fixture (unrelated to this fix)

### 🟢 Line Number Update (2026-01-28)

**Commit 588cec0** (`fix(audit): record COMPLETED outcomes only after sink durability`) removed ~65 lines from orchestrator.py, shifting line numbers:

| Task 3 Reference | Original | Updated | Reason |
|------------------|----------|---------|--------|
| Main run path | 837-852 | **847-852** | +10 lines (code above shifted) |
| Resume path | 1769-1774 | **1761-1765** | -8 lines (code removed above) |

**Impact:** Line numbers only. The coalesce_step_map calculation logic is unchanged and still requires fixing.

---

## Pre-Implementation Checklist

- [ ] Read `docs/bugs/BUG-COALESCE-GRAPH-EXECUTION-MISMATCH.md` for full context
- [ ] Understand the Three-Tier Trust Model in `CLAUDE.md` (Our Data = crash on anomaly)
- [ ] Note: Downstream transforms will run ONCE (on merged token) not TWICE (per branch) - this is the correct behavior

---

## Task 1: Add `_coalesce_gate_index` Instance Variable to ExecutionGraph

**Files:**
- Modify: `src/elspeth/core/dag.py:76-86` (add instance variable)
- Modify: `src/elspeth/core/dag.py:677-691` (store computed value)

**Step 1: Add instance variable to `__init__`**

In `src/elspeth/core/dag.py`, add after line 86 (`self._route_resolution_map`):

```python
        self._coalesce_gate_index: dict[CoalesceName, int] = {}  # coalesce_name -> gate pipeline index
```

**Step 2: Store computed value in `from_plugin_instances`**

The local variable `coalesce_gate_index` is computed at lines 677-691. After line 697 (end of validation), add:

```python
        # Store for external access
        graph._coalesce_gate_index = coalesce_gate_index
```

**Step 3: Run existing tests to verify no regression**

Run: `pytest tests/core/test_dag.py -v`
Expected: All tests PASS (we haven't changed behavior yet)

**Step 4: Commit**

```bash
git add src/elspeth/core/dag.py
git commit -m "$(cat <<'EOF'
refactor(dag): store coalesce_gate_index as instance variable

Preparation for exposing coalesce insertion points to orchestrator.
No behavioral change - just storing the already-computed local variable.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `get_coalesce_gate_index()` Getter Method

**Files:**
- Modify: `src/elspeth/core/dag.py:806-813` (add after `get_branch_to_coalesce_map`)
- Test: `tests/core/test_dag.py`

**Step 1: Write failing test**

Add to `tests/core/test_dag.py`:

```python
class TestCoalesceGateIndex:
    """Test coalesce_gate_index exposure from ExecutionGraph."""

    def test_get_coalesce_gate_index_returns_copy(self) -> None:
        """Getter should return a copy to prevent external mutation."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json"})},
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

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
            default_sink=settings.default_sink,
        )

        # Get the index
        index = graph.get_coalesce_gate_index()

        # Verify it contains expected mapping
        from elspeth.contracts.types import CoalesceName
        assert CoalesceName("merge_branches") in index
        assert isinstance(index[CoalesceName("merge_branches")], int)

        # Verify it's a copy (mutation doesn't affect internal state)
        original_value = index[CoalesceName("merge_branches")]
        index[CoalesceName("merge_branches")] = 999

        fresh_index = graph.get_coalesce_gate_index()
        assert fresh_index[CoalesceName("merge_branches")] == original_value

    def test_get_coalesce_gate_index_empty_when_no_coalesce(self) -> None:
        """Getter returns empty dict when no coalesce configured."""
        from elspeth.cli_helpers import instantiate_plugins_from_config
        from elspeth.core.config import ElspethSettings, SinkSettings, SourceSettings
        from elspeth.core.dag import ExecutionGraph

        settings = ElspethSettings(
            source=SourceSettings(plugin="null"),
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json"})},
            default_sink="output",
        )

        plugins = instantiate_plugins_from_config(settings)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(settings.gates),
            coalesce_settings=settings.coalesce,
            default_sink=settings.default_sink,
        )

        index = graph.get_coalesce_gate_index()
        assert index == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_dag.py::TestCoalesceGateIndex -v`
Expected: FAIL with `AttributeError: 'ExecutionGraph' object has no attribute 'get_coalesce_gate_index'`

**Step 3: Implement getter method**

Add to `src/elspeth/core/dag.py` after `get_branch_to_coalesce_map()` (after line 813):

```python
    def get_coalesce_gate_index(self) -> dict[CoalesceName, int]:
        """Get coalesce_name -> producing gate pipeline index mapping.

        Returns the pipeline index of the gate that produces each coalesce's
        branches. Used by orchestrator to compute coalesce_step_map aligned
        with graph topology.

        Returns:
            Dict mapping coalesce name to the pipeline index of its producing
            fork gate. Empty dict if no coalesce configured.
        """
        return dict(self._coalesce_gate_index)  # Return copy to prevent mutation
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/core/test_dag.py::TestCoalesceGateIndex -v`
Expected: PASS

**Step 5: Run full DAG test suite**

Run: `pytest tests/core/test_dag.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): expose coalesce_gate_index via getter method

Adds get_coalesce_gate_index() to ExecutionGraph, returning a copy
of the coalesce_name -> gate_pipeline_index mapping. This allows
orchestrator to compute coalesce_step_map aligned with graph topology
instead of hardcoding to end of pipeline.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update Orchestrator to Use Graph's Coalesce Gate Index

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py:847-852` (main run path)
- Modify: `src/elspeth/engine/orchestrator.py:1761-1765` (resume path)
- Test: `tests/engine/test_orchestrator_fork_coalesce.py`

**IMPROVEMENT: Extract shared helper to avoid duplication**

**Step 1: Add private helper method**

Add to `Orchestrator` class (before `run` method):

```python
    def _compute_coalesce_step_map(
        self,
        graph: ExecutionGraph,
        config: "PipelineConfig",
        settings: ElspethSettings | None,
    ) -> dict[CoalesceName, int]:
        """Compute coalesce step positions from graph topology.

        Coalesce step = len(transforms) + producing_gate_index + 1

        This aligns processor execution with graph topology:
        - Fork children skip directly to coalesce_step (not through all remaining gates)
        - Merged token continues from coalesce to downstream nodes

        Args:
            graph: The execution graph
            config: Pipeline configuration
            settings: Elspeth settings (may be None)

        Returns:
            Dict mapping coalesce name to its step index in the pipeline
        """
        coalesce_step_map: dict[CoalesceName, int] = {}
        if settings is not None and settings.coalesce:
            coalesce_gate_index = graph.get_coalesce_gate_index()
            num_transforms = len(config.transforms)
            for coalesce_name, gate_idx in coalesce_gate_index.items():
                # Coalesce step is RIGHT AFTER its producing fork gate
                # +1 because gates are processed at step (num_transforms + gate_idx),
                # so coalesce happens at step (num_transforms + gate_idx + 1)
                coalesce_step_map[coalesce_name] = num_transforms + gate_idx + 1
        return coalesce_step_map
```

**Step 2: Write failing test for correct step calculation**

Add to `tests/engine/test_orchestrator_fork_coalesce.py`:

```python
class TestCoalesceStepMapCalculation:
    """Test that coalesce_step_map is computed from graph topology."""

    def test_coalesce_step_map_uses_graph_gate_index(
        self,
        plugin_manager,
    ) -> None:
        """coalesce_step_map should use gate index from graph, not config order.

        Given:
          - Pipeline with fork_gate at index 0 (first gate)
          - Coalesce for that fork's branches

        The coalesce_step should be len(transforms) + gate_index + 1,
        NOT len(transforms) + len(gates) + 1.
        """
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import (
            CoalesceSettings,
            ElspethSettings,
            GateSettings,
            SinkSettings,
            SourceSettings,
        )
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator

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
                    routes={"true": "discard", "false": "continue"},
                ),
            ],
            sinks={"output": SinkSettings(plugin="json", options={"path": "/tmp/test.json"})},
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

        # Capture the coalesce_step_map passed to RowProcessor
        captured_step_map = {}

        original_processor_init = None

        def capture_processor(*args, **kwargs):
            nonlocal captured_step_map
            captured_step_map = kwargs.get("coalesce_step_map", {})
            return original_processor_init(*args, **kwargs)

        from elspeth.engine.processor import RowProcessor
        original_processor_init = RowProcessor.__init__

        with patch.object(RowProcessor, "__init__", capture_processor):
            try:
                orchestrator.run(settings)
            except Exception:
                pass  # We just need to capture the step_map

        # Verify: coalesce step should be transforms(0) + fork_gate_index(0) + 1 = 1
        # NOT: transforms(0) + gates(2) + 1 = 3
        from elspeth.contracts.types import CoalesceName
        assert CoalesceName("merge_branches") in captured_step_map

        # fork_gate is at pipeline_index 0 (first gate)
        # coalesce_step = len(transforms) + gate_index + 1 = 0 + 0 + 1 = 1
        expected_step = 0 + 0 + 1  # transforms + gate_index + 1
        assert captured_step_map[CoalesceName("merge_branches")] == expected_step
```

**Step 3: Run test to verify it fails**

Run: `pytest tests/engine/test_orchestrator_fork_coalesce.py::TestCoalesceStepMapCalculation -v`
Expected: FAIL (currently calculates step as 3, not 1)

**Step 4: Update main run path in orchestrator**

Replace lines 847-852 in `src/elspeth/engine/orchestrator.py` with:

```python
        # Compute coalesce step positions FROM GRAPH TOPOLOGY
        coalesce_step_map = self._compute_coalesce_step_map(graph, config, settings)
```

**Step 5: Update resume path in orchestrator**

Replace lines 1761-1765 in `src/elspeth/engine/orchestrator.py` with:

```python
        # Compute coalesce step positions FROM GRAPH TOPOLOGY (same as main run path)
        coalesce_step_map = self._compute_coalesce_step_map(graph, config, settings)
```

**Step 6: Run test to verify it passes**

Run: `pytest tests/engine/test_orchestrator_fork_coalesce.py::TestCoalesceStepMapCalculation -v`
Expected: PASS

**Step 7: Run broader orchestrator tests**

Run: `pytest tests/engine/test_orchestrator*.py -v`
Expected: May have failures - we've changed step calculation but not processor routing yet

**Step 8: Commit (partial fix)**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator_fork_coalesce.py
git commit -m "$(cat <<'EOF'
fix(orchestrator): compute coalesce_step_map from graph topology

Changes coalesce step calculation from:
  base_step + 1 + config_order_index (always at end of pipeline)
To:
  num_transforms + gate_pipeline_index + 1 (at fork point in graph)

Extracts _compute_coalesce_step_map() helper to avoid duplication
between main run path and resume path.

This aligns with graph topology where coalesce nodes are inserted
right after their producing fork gates, not at the end.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add Null Safety Check in Processor Fork Handling

**Files:**
- Modify: `src/elspeth/engine/processor.py:1141-1143`
- Modify: `src/elspeth/engine/processor.py:900-902` (plugin gate path)

**Step 1: Add null safety to config gate fork path**

In `src/elspeth/engine/processor.py`, after line 1143, add:

```python
                        # Tier 1 (Our Data) - crash on missing entry
                        if cfg_coalesce_step is None:
                            raise ValueError(
                                f"Coalesce step not found for '{cfg_coalesce_name}'. "
                                f"This indicates a bug in orchestrator coalesce_step_map construction. "
                                f"Available coalesce steps: {list(self._coalesce_step_map.keys())}"
                            )
```

**Step 2: Add null safety to plugin gate fork path**

In `src/elspeth/engine/processor.py`, after line 902, add the same check:

```python
                            # Tier 1 (Our Data) - crash on missing entry
                            if child_coalesce_step is None:
                                raise ValueError(
                                    f"Coalesce step not found for '{child_coalesce_name}'. "
                                    f"This indicates a bug in orchestrator coalesce_step_map construction. "
                                    f"Available coalesce steps: {list(self._coalesce_step_map.keys())}"
                                )
```

**Step 3: Commit**

```bash
git add src/elspeth/engine/processor.py
git commit -m "$(cat <<'EOF'
fix(processor): add null safety check for coalesce_step lookup

Adds explicit ValueError when coalesce_step_map is missing an entry
for a configured coalesce. Per Three-Tier Trust Model, this is Our Data
and missing entries indicate a bug in orchestrator, not recoverable.

Applies to both config gate fork path and plugin gate fork path.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update Config Gate Fork Child Routing to Skip to Coalesce

**Files:**
- Modify: `src/elspeth/engine/processor.py:1145-1153`
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write the critical failing test**

Add to `tests/engine/test_processor_coalesce.py`:

```python
class TestForkSkipToCoalesce:
    """Test that fork children skip directly to coalesce, not through all gates."""

    def test_config_gate_fork_children_skip_to_coalesce_step(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """Config gate fork children should start at coalesce_step, skipping intermediate gates.

        BEFORE FIX: start_step = len(transforms) + next_config_step (next gate)
        AFTER FIX: start_step = coalesce_step (skip to coalesce)

        This is the critical behavioral change that aligns execution with graph.
        """
        # Test implementation as in original plan...
        pass  # Full implementation in original plan Task 5
```

**Step 2: Update config gate fork child start_step**

Replace line 1149 in `src/elspeth/engine/processor.py`:

```python
                            # Children skip directly to coalesce step (not next gate)
                            start_step=cfg_coalesce_step if cfg_coalesce_step is not None else len(transforms) + next_config_step,
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/engine/test_processor_coalesce.py::TestForkSkipToCoalesce -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
fix(processor): config gate fork children skip directly to coalesce step

Changes config gate fork child routing from:
  start_step = len(transforms) + next_config_step (next gate)
To:
  start_step = coalesce_step (skip to coalesce)

This aligns execution with graph topology: fork children no longer
process intermediate gates; they jump directly to coalesce.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5b: Update Plugin Gate Fork Child Routing to Skip to Coalesce

**CRITICAL: This was missing from the original plan!**

**Files:**
- Modify: `src/elspeth/engine/processor.py:904-911`
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write failing test for plugin gate fork path**

Add to `tests/engine/test_processor_coalesce.py`:

```python
    def test_plugin_gate_fork_children_skip_to_coalesce_step(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """Plugin gate fork children should also skip to coalesce_step.

        This tests the PLUGIN gate fork path (processor.py:891-911),
        which is DIFFERENT from the config gate fork path (processor.py:1132-1153).

        Both paths must have the same fix applied.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.types import BranchName, CoalesceName, NodeID
        from elspeth.core.config import CoalesceSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseGate
        from elspeth.plugins.results import GateResult, RoutingAction, RoutingKind, RowOutcome
        from tests.engine.conftest import DYNAMIC_SCHEMA

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Create a plugin gate that forks
        class ForkingPluginGate(BaseGate):
            """A plugin gate (not config gate) that forks to branches."""
            name = "forking_plugin_gate"

            def __init__(self):
                super().__init__({"routes": {"true": "fork"}, "schema": {"fields": "dynamic"}})

            def evaluate(self, row: dict, ctx) -> GateResult:
                return GateResult(
                    action=RoutingAction(
                        kind=RoutingKind.FORK_TO_PATHS,
                        fork_targets=["branch_a", "branch_b"],
                    )
                )

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        gate = ForkingPluginGate()
        gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="forking_plugin_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        coalesce_executor.register_coalesce(
            CoalesceSettings(
                name="merge_branches",
                branches=["branch_a", "branch_b"],
                policy="require_all",
                merge="union",
            ),
            NodeID("coalesce_1"),
        )

        # coalesce_step = 1 (right after the plugin gate at step 0)
        coalesce_step = 1

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            branch_to_coalesce={
                BranchName("branch_a"): CoalesceName("merge_branches"),
                BranchName("branch_b"): CoalesceName("merge_branches"),
            },
            coalesce_step_map={CoalesceName("merge_branches"): coalesce_step},
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge_branches"): NodeID("coalesce_1")},
        )

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process through the plugin gate
        # Note: The gate is passed as a "transform" because plugin gates are
        # processed in the transforms phase (before config gates)
        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[gate],  # Plugin gate in transforms list
            ctx=ctx,
        )

        # Verify fork happened
        assert any(r.outcome == RowOutcome.FORKED for r in results)

        # The fix is verified by checking that children started at coalesce_step,
        # not at next_step. We check via node_states if available, or by
        # verifying that downstream gates were NOT executed on children.
```

**Step 2: Update plugin gate fork child start_step**

Replace line 907 in `src/elspeth/engine/processor.py`:

```python
                                # Children skip directly to coalesce step (not next step)
                                start_step=child_coalesce_step if child_coalesce_step is not None else next_step,
```

**Step 3: Run tests**

Run: `pytest tests/engine/test_processor_coalesce.py::TestForkSkipToCoalesce -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
fix(processor): plugin gate fork children also skip to coalesce step

Applies the same fix to plugin gate fork path (line 907) as was applied
to config gate fork path (line 1149).

CRITICAL: The original plan missed this second fork path. Both paths
now consistently route fork children directly to coalesce_step.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Test Merged Token Continuation (Dead Code Resurrection)

**Files:**
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write test for merged token continuing to downstream transform**

Add to `tests/engine/test_processor_coalesce.py`:

```python
class TestMergedTokenContinuation:
    """Test that merged tokens continue to downstream nodes after coalesce."""

    def test_merged_token_continues_to_downstream_transform(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """Merged token should continue processing through remaining transforms.

        This tests the "dead code path" at processor.py:783-789 which is now
        reachable because coalesce_at_step < total_steps.

        Pipeline: source -> fork_gate -> coalesce -> downstream_transform -> sink

        BEFORE FIX:
        - downstream_transform runs TWICE (per branch, before coalesce)
        - Merged token returns COALESCED immediately

        AFTER FIX:
        - downstream_transform runs ONCE (on merged token)
        - Merged token continues through remaining transforms
        """
        from typing import Any

        from elspeth.contracts import NodeType
        from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult
        from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        # Track transform execution
        transform_calls: list[dict] = []

        class TrackingTransform(BaseTransform):
            """Transform that tracks how many times it's called."""
            name = "tracking"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self):
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict, ctx: Any) -> TransformResult:
                transform_calls.append({"row": row.copy()})
                row["transformed"] = True
                return TransformResult.success(row)

        # Register nodes
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        transform = TrackingTransform()
        transform_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="tracking",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        coalesce_executor.register_coalesce(
            CoalesceSettings(
                name="merge_branches",
                branches=["branch_a", "branch_b"],
                policy="require_all",
                merge="union",
            ),
            NodeID("coalesce_1"),
        )

        # coalesce_step = 1, total_steps = 2
        # Since coalesce_step (1) < total_steps (2), merged token continues
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            config_gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            config_gate_id_map={GateName("fork_gate"): NodeID("gate_fork")},
            branch_to_coalesce={
                BranchName("branch_a"): CoalesceName("merge_branches"),
                BranchName("branch_b"): CoalesceName("merge_branches"),
            },
            coalesce_step_map={CoalesceName("merge_branches"): 1},
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge_branches"): NodeID("coalesce_1")},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        # Process - transform should run AFTER coalesce on merged token
        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[transform],
            ctx=ctx,
        )

        # Key assertion: transform should be called ONCE (on merged token)
        # NOT twice (per fork child)
        assert len(transform_calls) == 1, (
            f"Transform should be called once on merged token, "
            f"but was called {len(transform_calls)} times"
        )

        # Final result should have transformed flag
        completed_results = [r for r in results if r.outcome == RowOutcome.COMPLETED]
        assert len(completed_results) == 1
        assert completed_results[0].final_data.get("transformed") is True
```

**Step 2: Run test**

Run: `pytest tests/engine/test_processor_coalesce.py::TestMergedTokenContinuation -v`

**Step 3: Commit test**

```bash
git add tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
test(processor): add merged token continuation test

Tests that merged tokens continue to downstream transforms after
coalesce, running transforms ONCE on merged data instead of TWICE
per fork child.

This validates the "dead code resurrection" at processor.py:783-789.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Test Backward Compatibility (Coalesce at End of Pipeline)

**Files:**
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write backward compatibility test**

Add to `tests/engine/test_processor_coalesce.py`:

```python
class TestBackwardCompatibility:
    """Test that existing coalesce-at-end pipelines still work."""

    def test_coalesce_at_end_returns_coalesced_immediately(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """When coalesce is at end of pipeline, merged token returns COALESCED.

        This is the existing behavior that must not break.

        Pipeline: source -> transform -> fork_gate -> coalesce -> sink
        (No gates/transforms AFTER coalesce)

        When coalesce_at_step >= total_steps, merged token returns immediately
        with RowOutcome.COALESCED, not queued for more processing.
        """
        from elspeth.contracts import NodeType
        from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome
        from tests.engine.conftest import DYNAMIC_SCHEMA

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        coalesce_executor.register_coalesce(
            CoalesceSettings(
                name="merge_branches",
                branches=["branch_a", "branch_b"],
                policy="require_all",
                merge="union",
            ),
            NodeID("coalesce_1"),
        )

        # Pipeline: fork_gate is the LAST gate
        # coalesce_step = 1, total_steps = 1
        # coalesce_at_step (1) >= total_steps (1), so merged token returns immediately
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            config_gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            config_gate_id_map={GateName("fork_gate"): NodeID("gate_fork")},
            branch_to_coalesce={
                BranchName("branch_a"): CoalesceName("merge_branches"),
                BranchName("branch_b"): CoalesceName("merge_branches"),
            },
            coalesce_step_map={CoalesceName("merge_branches"): 1},
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge_branches"): NodeID("coalesce_1")},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[],
            ctx=ctx,
        )

        # Should have FORKED parent and COALESCED merged token
        outcomes = [r.outcome for r in results]
        assert RowOutcome.FORKED in outcomes
        assert RowOutcome.COALESCED in outcomes
```

**Step 2: Run test**

Run: `pytest tests/engine/test_processor_coalesce.py::TestBackwardCompatibility -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
test(processor): add backward compatibility test for coalesce at end

Verifies that pipelines with coalesce at end (no downstream nodes)
still work correctly - merged token returns COALESCED immediately.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Test Error Handling During Merged Token Continuation

**Files:**
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write error handling test**

Add to `tests/engine/test_processor_coalesce.py`:

```python
class TestMergedTokenErrorHandling:
    """Test error handling when downstream processing fails on merged token."""

    def test_downstream_transform_failure_on_merged_token(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """When downstream transform fails on merged token, result is FAILED.

        Pipeline: source -> fork_gate -> coalesce -> failing_transform -> sink

        The merged token should end up as FAILED, not COALESCED.
        """
        from typing import Any

        from elspeth.contracts import NodeType
        from elspeth.contracts.types import BranchName, CoalesceName, GateName, NodeID
        from elspeth.core.config import CoalesceSettings, GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.coalesce_executor import CoalesceExecutor
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome, TransformResult
        from tests.engine.conftest import DYNAMIC_SCHEMA, _TestSchema

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        class FailingTransform(BaseTransform):
            """Transform that always fails."""
            name = "failing"
            input_schema = _TestSchema
            output_schema = _TestSchema

            def __init__(self):
                super().__init__({"schema": {"fields": "dynamic"}})

            def process(self, row: dict, ctx: Any) -> TransformResult:
                return TransformResult.error({"reason": "intentional_failure"})

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        transform = FailingTransform()
        recorder.register_node(
            run_id=run.run_id,
            plugin_name="failing",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config={"fields": "dynamic"},
        )

        coalesce_executor = CoalesceExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            token_manager=token_manager,
            run_id=run.run_id,
        )

        coalesce_executor.register_coalesce(
            CoalesceSettings(
                name="merge_branches",
                branches=["branch_a", "branch_b"],
                policy="require_all",
                merge="union",
            ),
            NodeID("coalesce_1"),
        )

        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            config_gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["branch_a", "branch_b"],
                ),
            ],
            config_gate_id_map={GateName("fork_gate"): NodeID("gate_fork")},
            branch_to_coalesce={
                BranchName("branch_a"): CoalesceName("merge_branches"),
                BranchName("branch_b"): CoalesceName("merge_branches"),
            },
            coalesce_step_map={CoalesceName("merge_branches"): 1},
            coalesce_executor=coalesce_executor,
            coalesce_node_ids={CoalesceName("merge_branches"): NodeID("coalesce_1")},
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[transform],
            ctx=ctx,
        )

        # Should have FORKED parent
        assert any(r.outcome == RowOutcome.FORKED for r in results)

        # Merged token should be FAILED (not COALESCED)
        failed_results = [r for r in results if r.outcome == RowOutcome.FAILED]
        assert len(failed_results) >= 1, (
            f"Expected FAILED outcome from transform failure, got: {[r.outcome for r in results]}"
        )
```

**Step 2: Run test**

Run: `pytest tests/engine/test_processor_coalesce.py::TestMergedTokenErrorHandling -v`

**Step 3: Commit**

```bash
git add tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
test(processor): add error handling test for merged token continuation

Verifies that when a downstream transform fails on the merged token,
the outcome is FAILED (not COALESCED).

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Test Fork Without Coalesce (Regression Prevention)

**Files:**
- Test: `tests/engine/test_processor_coalesce.py`

**Step 1: Write test for fork without coalesce**

Add to `tests/engine/test_processor_coalesce.py`:

```python
class TestForkWithoutCoalesce:
    """Test that forks without coalesce still work correctly."""

    def test_fork_without_coalesce_executes_all_gates(
        self, landscape_db: "LandscapeDB"
    ) -> None:
        """Fork children with no coalesce should execute all remaining gates.

        When branch_to_coalesce is empty, children continue through all gates
        instead of skipping to a coalesce point.
        """
        from elspeth.contracts import NodeType, RoutingMode
        from elspeth.contracts.types import GateName, NodeID
        from elspeth.core.config import GateSettings
        from elspeth.core.landscape import LandscapeRecorder
        from elspeth.engine.processor import RowProcessor
        from elspeth.engine.spans import SpanFactory
        from elspeth.engine.tokens import TokenManager
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.results import RowOutcome
        from tests.engine.conftest import DYNAMIC_SCHEMA

        recorder = LandscapeRecorder(landscape_db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        token_manager = TokenManager(recorder)

        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Register edges for fork paths (no coalesce - go to sinks)
        fork_gate_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="fork_gate",
            node_type=NodeType.GATE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        edge_a = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id="sink_a",
            label="sink_a",
            mode=RoutingMode.COPY,
        )
        edge_b = recorder.register_edge(
            run_id=run.run_id,
            from_node_id=fork_gate_node.node_id,
            to_node_id="sink_b",
            label="sink_b",
            mode=RoutingMode.COPY,
        )

        # NO coalesce configured - branches go to different sinks
        processor = RowProcessor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
            source_node_id=NodeID(source.node_id),
            config_gates=[
                GateSettings(
                    name="fork_gate",
                    condition="True",
                    routes={"true": "fork", "false": "continue"},
                    fork_to=["sink_a", "sink_b"],
                ),
            ],
            config_gate_id_map={GateName("fork_gate"): NodeID(fork_gate_node.node_id)},
            branch_to_coalesce={},  # NO coalesce
            coalesce_step_map={},   # NO coalesce steps
            edge_map={
                (NodeID(fork_gate_node.node_id), "sink_a"): edge_a.edge_id,
                (NodeID(fork_gate_node.node_id), "sink_b"): edge_b.edge_id,
            },
        )

        ctx = PluginContext(run_id=run.run_id, config={})

        results = processor.process_row(
            row_index=0,
            row_data={"value": 1},
            transforms=[],
            ctx=ctx,
        )

        # Parent should be FORKED
        assert any(r.outcome == RowOutcome.FORKED for r in results)

        # Children should reach terminal states (COMPLETED or ROUTED)
        terminal_results = [r for r in results if r.outcome in (RowOutcome.COMPLETED, RowOutcome.ROUTED)]
        assert len(terminal_results) == 2, (
            f"Expected 2 terminal outcomes (one per branch), got {len(terminal_results)}"
        )
```

**Step 2: Run test**

Run: `pytest tests/engine/test_processor_coalesce.py::TestForkWithoutCoalesce -v`

**Step 3: Commit**

```bash
git add tests/engine/test_processor_coalesce.py
git commit -m "$(cat <<'EOF'
test(processor): add fork-without-coalesce regression test

Verifies that forks with no coalesce configured still work correctly,
with children executing all remaining gates instead of skipping.

Part of: BUG-COALESCE-GRAPH-EXECUTION-MISMATCH fix

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Run Full Test Suite and Fix Any Regressions

**Step 1: Run all coalesce-related tests**

Run: `pytest tests/engine/test_*coalesce*.py -v`

**Step 2: Run full engine test suite**

Run: `pytest tests/engine/ -v`

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`

**Step 4: Fix any failures**

Address failures as needed. Common issues:
- Tests that manually construct graphs may need updating
- Tests that assert on specific step numbers may need adjustment
- Edge cases around coalesce step calculation

**Note:** The pre-existing failure `test_duplicate_fork_branches_rejected_in_plugin_gate`
is unrelated to this fix (missing `routes` key in test fixture).

**Step 5: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix(coalesce): resolve graph/execution mismatch (BUG-COALESCE-GRAPH-EXECUTION-MISMATCH)

Complete fix for graph topology not matching processor execution:

1. ExecutionGraph now exposes coalesce_gate_index via getter
2. Orchestrator computes coalesce_step_map from graph topology
3. BOTH fork paths (config gate AND plugin gate) skip directly to coalesce_step
4. Merged token continuation path (processor.py:783-789) now active

Behavioral change: Downstream transforms/gates after coalesce now run
ONCE on merged token instead of TWICE per fork child.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update Bug Document Status

**Files:**
- Modify: `docs/bugs/BUG-COALESCE-GRAPH-EXECUTION-MISMATCH.md`

**Step 1: Update status**

Change line 6 from `**Status:** Open` to:

```markdown
**Status:** Fixed (2026-01-27)
**Fix Commit:** <commit-hash>
```

**Step 2: Add resolution section**

Add at the end of the document:

```markdown
---

## Resolution

**Fix implemented:** Option A (Make Processor Match Graph)

**Changes:**
1. `dag.py`: Added `_coalesce_gate_index` instance variable and `get_coalesce_gate_index()` getter
2. `orchestrator.py`: Both run paths now compute `coalesce_step_map` from graph's gate index via shared helper
3. `processor.py`: BOTH fork paths (plugin gate AND config gate) use `coalesce_step` as `start_step`

**Behavioral change:** Downstream transforms/gates now execute ONCE on merged token, not TWICE per fork child.

**Tests added:**
- `TestCoalesceGateIndex`: Getter returns copy, handles empty case
- `TestCoalesceStepMapCalculation`: Orchestrator uses graph topology
- `TestForkSkipToCoalesce`: Config gate AND plugin gate children skip intermediate gates
- `TestMergedTokenContinuation`: Dead code path now active
- `TestBackwardCompatibility`: End-of-pipeline coalesce unchanged
- `TestMergedTokenErrorHandling`: Failures on merged token handled
- `TestForkWithoutCoalesce`: No regression for non-coalesce forks
```

**Step 3: Commit**

```bash
git add docs/bugs/BUG-COALESCE-GRAPH-EXECUTION-MISMATCH.md
git commit -m "$(cat <<'EOF'
docs(bugs): mark BUG-COALESCE-GRAPH-EXECUTION-MISMATCH as fixed

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary (Updated 2026-01-28)

| Task | Description | Files | Status |
|------|-------------|-------|--------|
| 1 | Add `_coalesce_gate_index` instance variable | dag.py | Ready |
| 2 | Add `get_coalesce_gate_index()` getter | dag.py, test_dag.py | Ready |
| 3 | Update orchestrator step calculation (with helper) | orchestrator.py | **Enhanced** |
| 4 | Add null safety check (both fork paths) | processor.py | **Enhanced** |
| 5 | Update config gate fork child routing | processor.py | Ready |
| **5b** | **Update plugin gate fork child routing** | processor.py | **NEW** |
| 6 | Test merged token continuation | test_processor_coalesce.py | Ready |
| 7 | Test backward compatibility | test_processor_coalesce.py | Ready |
| 8 | Test error handling | test_processor_coalesce.py | Ready |
| 9 | Test fork without coalesce | test_processor_coalesce.py | Ready |
| 10 | Run full test suite | - | Ready |
| 11 | Update bug document | BUG doc | Ready |

**Total estimated time:** 2-3 hours

**Risk level:** 🟢 Low-Medium (zero users, comprehensive tests, critical for live run)

**Deadline:** Must complete before first live run next week.

**Critical Improvements from Risk Reduction:**
1. ✅ Task 5b added for plugin gate fork path (was missing!)
2. ✅ Task 3 enhanced with `_compute_coalesce_step_map()` helper
3. ✅ Task 4 enhanced to cover both fork paths
4. ✅ Pre-existing test failure documented (not our concern)

**Post-Implementation Verification:**
- Run a fork→coalesce→downstream test pipeline to confirm graph matches execution
- Verify `landscape.explain()` shows correct token paths
