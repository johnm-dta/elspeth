# Architecture Alignment Remediation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align implementation with architecture.md on four issues: output sink routing, edge label semantics, filter transform contract, and row_plugins execution.

**Architecture:** The implementation has drifted from the architecture specification in ways that break core invariants. This plan fixes each issue incrementally, maintaining test coverage throughout. Issues are ordered by dependency - Issue 1 is a prerequisite for Issue 4.

**Tech Stack:** Python, pytest, Pydantic, NetworkX

---

## Issue Summary

| Issue | Severity | Root Cause |
|-------|----------|------------|
| 1. Output sink hardcoded to "default" | Critical | `orchestrator.py:230` uses literal `"default"` instead of `graph.get_output_sink()` |
| 2. Edge label vs sink name mismatch | Critical | DAG labels edges with route labels, but lookup uses sink names |
| 3. Filter violates "no silent drops" | High | Architecture requires routing to discard sink; Filter returns `success(None)` |
| 4. row_plugins not executed | Medium | CLI hardcodes `transforms=[]` ignoring config |

---

## Task 1: Fix Output Sink Routing

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (lines 165-166, 230)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
class TestOrchestratorOutputSinkRouting:
    """Verify completed rows go to the configured output_sink, not hardcoded 'default'."""

    def test_completed_rows_go_to_output_sink(self) -> None:
        """Rows that complete the pipeline go to the output_sink from config."""
        from unittest.mock import MagicMock

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig

        db = LandscapeDB.in_memory()

        # Config with output_sink="results" (NOT "default")
        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "errors": SinkSettings(plugin="csv"),
            },
            output_sink="results",
        )
        graph = ExecutionGraph.from_config(settings)

        # Mock source that yields one row
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1, "value": "test"}])

        # Mock sinks - track what gets written
        mock_results_sink = MagicMock()
        mock_results_sink.name = "csv"
        mock_results_sink.write = MagicMock()

        mock_errors_sink = MagicMock()
        mock_errors_sink.name = "csv"
        mock_errors_sink.write = MagicMock()

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[],
            sinks={
                "results": mock_results_sink,
                "errors": mock_errors_sink,
            },
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph)

        # Row should go to "results" sink, not "default"
        assert result.rows_processed == 1
        # Results sink should have been written to
        assert mock_results_sink.write.called or mock_results_sink.finalize.called
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorOutputSinkRouting -v`
Expected: FAIL with `KeyError: 'default'`

**Step 3: Write minimal implementation**

Modify `src/elspeth/engine/orchestrator.py`:

After line 165 (after `transform_id_map = graph.get_transform_id_map()`), add:

```python
        output_sink_name = graph.get_output_sink()
```

Then at line 230, replace:
```python
                        pending_tokens["default"].append(result.token)
```

With:
```python
                        pending_tokens[output_sink_name].append(result.token)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorOutputSinkRouting -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
fix(orchestrator): use graph.get_output_sink() instead of hardcoded 'default'

Completed rows now route to the configured output_sink, not a
hardcoded 'default' key that may not exist.

Fixes architecture alignment issue #1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add Route Label to Edge Mapping

The architecture specifies that edges are labeled with **route labels** (like "emergency"), but gates return **sink names** (like "emergency_broadcast"). We need a mapping from sink name back to route label for edge lookup.

**Files:**
- Modify: `src/elspeth/core/dag.py`
- Test: `tests/core/test_dag.py`

**Step 1: Write the failing test**

Add to `tests/core/test_dag.py`:

```python
class TestExecutionGraphRouteMapping:
    """Test route label <-> sink name mapping for edge lookup."""

    def test_get_route_label_for_sink(self) -> None:
        """Get route label that leads to a sink from a gate."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="classifier",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)

        # Get the gate's node_id
        transform_map = graph.get_transform_id_map()
        gate_node_id = transform_map[0]

        # Given gate node and sink name, get the route label
        route_label = graph.get_route_label(gate_node_id, "flagged")

        assert route_label == "suspicious"

    def test_get_route_label_for_continue(self) -> None:
        """Continue routes return 'continue' as label."""
        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph

        config = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={"results": SinkSettings(plugin="csv")},
            row_plugins=[
                RowPluginSettings(
                    plugin="gate",
                    type="gate",
                    routes={"pass": "continue"},
                ),
            ],
            output_sink="results",
        )

        graph = ExecutionGraph.from_config(config)
        transform_map = graph.get_transform_id_map()
        gate_node_id = transform_map[0]

        # The edge to output sink uses "continue" label
        route_label = graph.get_route_label(gate_node_id, "results")
        assert route_label == "continue"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphRouteMapping -v`
Expected: FAIL with `'ExecutionGraph' object has no attribute 'get_route_label'`

**Step 3: Write minimal implementation**

Add to `src/elspeth/core/dag.py` in `ExecutionGraph.__init__()`:

```python
        self._route_label_map: dict[tuple[str, str], str] = {}  # (gate_node, sink_name) -> route_label
```

Update `from_config()` to populate the mapping when processing gate routes:

```python
            # Gate routes to sinks
            if is_gate and plugin_config.routes:
                for route_label, target in plugin_config.routes.items():
                    if target == "continue":
                        continue  # Not a sink route
                    if target not in sink_ids:
                        raise GraphValidationError(
                            f"Gate '{plugin_config.plugin}' routes '{route_label}' "
                            f"to unknown sink '{target}'. "
                            f"Available sinks: {list(sink_ids.keys())}"
                        )
                    target_node_id = sink_ids[target]
                    graph.add_edge(tid, target_node_id, label=route_label, mode="move")
                    # Store reverse mapping: (gate_node, sink_name) -> route_label
                    graph._route_label_map[(tid, target)] = route_label
```

Add the accessor method:

```python
    def get_route_label(self, from_node_id: str, sink_name: str) -> str:
        """Get the route label for an edge from a gate to a sink.

        Args:
            from_node_id: The gate node ID
            sink_name: The sink name (not node ID)

        Returns:
            The route label (e.g., "suspicious") or "continue" for default path
        """
        # Check explicit route mapping first
        if (from_node_id, sink_name) in self._route_label_map:
            return self._route_label_map[(from_node_id, sink_name)]

        # Default path uses "continue" label
        return "continue"
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/core/test_dag.py::TestExecutionGraphRouteMapping -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/dag.py tests/core/test_dag.py
git commit -m "$(cat <<'EOF'
feat(dag): add get_route_label() for sink name to route label mapping

Gates return sink names but edges are labeled with route labels.
This method provides the reverse mapping needed for edge lookup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Fix Edge Lookup to Use Route Labels

Now update the orchestrator to use route labels for edge lookup.

**Files:**
- Modify: `src/elspeth/engine/orchestrator.py` (lines 148-158)
- Test: `tests/engine/test_orchestrator.py`

**Step 1: Write the failing test**

Add to `tests/engine/test_orchestrator.py`:

```python
class TestOrchestratorGateRouting:
    """Test that gate routing works with route labels."""

    def test_gate_routes_to_named_sink(self) -> None:
        """Gate can route rows to a named sink using route labels."""
        from unittest.mock import MagicMock

        from elspeth.core.config import (
            DatasourceSettings,
            ElspethSettings,
            RowPluginSettings,
            SinkSettings,
        )
        from elspeth.core.dag import ExecutionGraph
        from elspeth.core.landscape import LandscapeDB
        from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
        from elspeth.plugins.results import GateResult, RoutingAction

        db = LandscapeDB.in_memory()

        settings = ElspethSettings(
            datasource=DatasourceSettings(plugin="csv"),
            sinks={
                "results": SinkSettings(plugin="csv"),
                "flagged": SinkSettings(plugin="csv"),
            },
            row_plugins=[
                RowPluginSettings(
                    plugin="test_gate",
                    type="gate",
                    routes={"suspicious": "flagged", "clean": "continue"},
                ),
            ],
            output_sink="results",
        )
        graph = ExecutionGraph.from_config(settings)

        # Mock source
        mock_source = MagicMock()
        mock_source.name = "csv"
        mock_source.load.return_value = iter([{"id": 1, "score": 0.2}])

        # Mock gate that routes to "flagged"
        mock_gate = MagicMock()
        mock_gate.name = "test_gate"
        mock_gate.evaluate.return_value = GateResult(
            row={"id": 1, "score": 0.2},
            action=RoutingAction.route_to_sink("flagged", reason={"score": "low"}),
        )

        # Mock sinks
        mock_results = MagicMock()
        mock_results.name = "csv"
        mock_flagged = MagicMock()
        mock_flagged.name = "csv"

        pipeline_config = PipelineConfig(
            source=mock_source,
            transforms=[mock_gate],
            sinks={"results": mock_results, "flagged": mock_flagged},
        )

        orchestrator = Orchestrator(db)
        result = orchestrator.run(pipeline_config, graph=graph)

        # Row should be routed, not completed
        assert result.rows_processed == 1
        # Flagged sink should receive the row
        assert mock_flagged.write.called or mock_flagged.finalize.called
```

**Step 2: Analyze the current edge_map construction**

The current code in `orchestrator.py` lines 148-158 builds:
```python
edge_map[(from_id, edge_data["label"])] = edge.edge_id
```

Where `label` is the route label (e.g., "suspicious"). But executors look up by sink name.

**Step 3: Update edge_map to be keyed by sink name**

Modify `orchestrator.py` at lines 148-158:

```python
        # Register edges from graph - key by (from_node, to_sink_name) for lookup
        edge_map: dict[tuple[str, str], str] = {}
        sink_id_to_name = {v: k for k, v in sink_id_map.items()}  # Reverse mapping

        for from_id, to_id, edge_data in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=from_id,
                to_node_id=to_id,
                label=edge_data["label"],
                mode=edge_data["mode"],
            )
            # Key by sink NAME (what gates return), not route label
            # to_id might be a sink or a transform - only map sinks
            if to_id in sink_id_to_name:
                sink_name = sink_id_to_name[to_id]
                edge_map[(from_id, sink_name)] = edge.edge_id
            else:
                # Non-sink edges (transform chains) - keep label-based key
                edge_map[(from_id, edge_data["label"])] = edge.edge_id
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py::TestOrchestratorGateRouting -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/elspeth/engine/orchestrator.py tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
fix(orchestrator): key edge_map by sink name for gate routing

Gates return sink names in RoutingAction.destinations, so the
edge_map must be keyed by sink name for successful lookup.

Fixes architecture alignment issue #2.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Create FilterGate Plugin

The architecture states "no silent drops" - every row must have an explicit terminal state. Per architecture.md:148, filtering is done via a Gate that routes filtered rows to a discard sink. The current Filter transform returns `success(None)` which violates this invariant.

**Files:**
- Create: `src/elspeth/plugins/gates/__init__.py` (add export)
- Create: `src/elspeth/plugins/gates/filter_gate.py`
- Create: `tests/plugins/gates/__init__.py`
- Create: `tests/plugins/gates/test_filter_gate.py`

**Step 1: Create test directory**

Run: `mkdir -p tests/plugins/gates && touch tests/plugins/gates/__init__.py`

**Step 2: Write the failing test**

Create `tests/plugins/gates/test_filter_gate.py`:

```python
"""Tests for FilterGate plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


@pytest.fixture
def ctx() -> "PluginContext":
    """Create minimal plugin context."""
    from elspeth.plugins.context import PluginContext

    return PluginContext(
        run_id="test-run",
        config={},
        landscape=MagicMock(),
    )


class TestFilterGate:
    """Test FilterGate routes instead of silent drops."""

    def test_passing_row_continues(self, ctx: "PluginContext") -> None:
        """Row that passes filter continues to next stage."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "score": 0.8}, ctx)

        assert result.action.kind == "continue"

    def test_failing_row_routes_to_discard(self, ctx: "PluginContext") -> None:
        """Row that fails filter routes to discard sink."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("filtered_out",)
        assert "filtered" in result.action.reason.get("result", "")

    def test_reason_includes_filter_details(self, ctx: "PluginContext") -> None:
        """Routing reason includes why the row was filtered."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "trash",
        })

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.reason["field"] == "score"
        assert result.action.reason["value"] == 0.3
        assert result.action.reason["condition"] == "greater_than"
        assert result.action.reason["threshold"] == 0.5

    def test_missing_field_routes_to_discard(self, ctx: "PluginContext") -> None:
        """Row with missing field routes to discard sink by default."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("filtered_out",)
        assert "missing" in result.action.reason.get("result", "")

    def test_missing_field_passes_when_allowed(self, ctx: "PluginContext") -> None:
        """Row with missing field continues if allow_missing=True."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
            "allow_missing": True,
        })

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "continue"
```

**Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/plugins/gates/test_filter_gate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.plugins.gates.filter_gate'`

**Step 4: Write minimal implementation**

Create `src/elspeth/plugins/gates/filter_gate.py`:

```python
"""FilterGate plugin.

Routes rows that fail filter conditions to a discard sink instead of
silently dropping them. This maintains the "no silent drops" invariant
per architecture.md:148 and subsystems/00-overview.md:281.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.schemas import PluginSchema


class FilterGateSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class FilterGate(BaseGate):
    """Filter rows by routing non-matching rows to a discard sink.

    Unlike a Filter transform, this gate never silently drops rows.
    Rows that fail the condition are explicitly routed to a discard sink,
    maintaining complete audit trail per architecture spec.

    Config options:
        field: Field to check (supports dot notation for nested fields)
        discard_sink: Sink name for rows that fail the filter (REQUIRED)
        allow_missing: If True, missing fields pass filter (default: False)

        Conditions (exactly one required):
        - equals: Field must equal this value
        - not_equals: Field must not equal this value
        - greater_than: Field must be > this value (numeric)
        - less_than: Field must be < this value (numeric)
        - contains: Field must contain this substring
        - matches: Field must match this regex pattern
        - in_list: Field must be one of these values (list)
    """

    name = "filter_gate"
    input_schema = FilterGateSchema
    output_schema = FilterGateSchema

    _CONDITION_KEYS = frozenset({
        "equals",
        "not_equals",
        "greater_than",
        "less_than",
        "contains",
        "matches",
        "in_list",
    })

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._discard_sink: str = config["discard_sink"]
        self._allow_missing: bool = config.get("allow_missing", False)

        # Find which condition is specified
        self._condition_key: str | None = None
        self._condition_value: Any = None
        for key in self._CONDITION_KEYS:
            if key in config:
                self._condition_key = key
                self._condition_value = config[key]
                break

        if self._condition_key is None:
            raise ValueError(
                f"FilterGate requires one condition: {self._CONDITION_KEYS}"
            )

        # Pre-compile regex if needed
        self._regex: re.Pattern[str] | None = None
        if self._condition_key == "matches":
            self._regex = re.compile(self._condition_value)

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate filter condition and route accordingly.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with continue (pass) or route_to_sink (fail)
        """
        field_value, found = self._get_nested(row, self._field)

        # Handle missing field
        if not found:
            if self._allow_missing:
                return GateResult(
                    row=copy.deepcopy(row),
                    action=RoutingAction.continue_(reason={
                        "field": self._field,
                        "result": "pass_missing",
                    }),
                )
            return GateResult(
                row=copy.deepcopy(row),
                action=RoutingAction.route_to_sink(
                    self._discard_sink,
                    reason={
                        "field": self._field,
                        "result": "filtered_missing_field",
                    },
                ),
            )

        # Evaluate condition
        passes = self._evaluate_condition(field_value)

        if passes:
            return GateResult(
                row=copy.deepcopy(row),
                action=RoutingAction.continue_(reason={
                    "field": self._field,
                    "value": field_value,
                    "condition": self._condition_key,
                    "threshold": self._condition_value,
                    "result": "pass",
                }),
            )

        # Row fails - route to discard sink with full reason
        return GateResult(
            row=copy.deepcopy(row),
            action=RoutingAction.route_to_sink(
                self._discard_sink,
                reason={
                    "field": self._field,
                    "value": field_value,
                    "condition": self._condition_key,
                    "threshold": self._condition_value,
                    "result": "filtered",
                },
            ),
        )

    def _evaluate_condition(self, value: Any) -> bool:
        """Evaluate the condition against a value."""
        if self._condition_key == "equals":
            return value == self._condition_value
        elif self._condition_key == "not_equals":
            return value != self._condition_value
        elif self._condition_key == "greater_than":
            return value > self._condition_value
        elif self._condition_key == "less_than":
            return value < self._condition_value
        elif self._condition_key == "contains":
            return self._condition_value in str(value)
        elif self._condition_key == "matches":
            return bool(self._regex and self._regex.search(str(value)))
        elif self._condition_key == "in_list":
            return value in self._condition_value
        return False

    def _get_nested(self, data: dict[str, Any], path: str) -> tuple[Any, bool]:
        """Get value from nested dict using dot notation.

        Args:
            data: Dict to traverse
            path: Dot-separated field path

        Returns:
            Tuple of (value, found). If not found, returns (None, False).
        """
        parts = path.split(".")
        current: Any = data
        for part in parts:
            # Row data is always a dict - this is a contract, not a defensive check
            if part not in current:
                return None, False
            current = current[part]
        return current, True
```

**Step 5: Update gates __init__.py**

Add to `src/elspeth/plugins/gates/__init__.py`:

```python
from elspeth.plugins.gates.filter_gate import FilterGate

__all__ = ["FilterGate", "FieldMatchGate", "ThresholdGate"]
```

**Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/plugins/gates/test_filter_gate.py -v`
Expected: PASS

**Step 7: Run mypy on new file**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/gates/filter_gate.py`
Expected: Success

**Step 8: Commit**

```bash
git add src/elspeth/plugins/gates/filter_gate.py src/elspeth/plugins/gates/__init__.py tests/plugins/gates/
git commit -m "$(cat <<'EOF'
feat(gates): add FilterGate for auditable row filtering

FilterGate routes non-matching rows to a discard sink instead of
silently dropping them. This maintains the "no silent drops" invariant
required by the architecture (architecture.md:148, subsystems:281).

Use case:
  - plugin: filter_gate
    type: gate
    options:
      field: score
      greater_than: 0.5
      discard_sink: low_scores
    routes:
      low_scores: discarded

Fixes architecture alignment issue #3.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add Deprecation Warning to Filter Transform

**Files:**
- Modify: `src/elspeth/plugins/transforms/filter.py`
- Test: `tests/plugins/transforms/test_filter_deprecation.py`

**Step 1: Write the test**

Create `tests/plugins/transforms/test_filter_deprecation.py`:

```python
"""Test Filter transform deprecation warning."""

import warnings

import pytest


class TestFilterDeprecation:
    """Verify Filter emits deprecation warning."""

    def test_filter_emits_deprecation_warning(self) -> None:
        """Filter.__init__ should emit DeprecationWarning."""
        from elspeth.plugins.transforms.filter import Filter

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Filter({"field": "x", "equals": 1})

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "FilterGate" in str(w[0].message)
```

**Step 2: Add deprecation warning**

Modify `src/elspeth/plugins/transforms/filter.py` to add warning in `__init__`:

```python
import warnings

# In __init__, at the very start:
        warnings.warn(
            "Filter transform is deprecated. Use FilterGate instead for "
            "auditable filtering that routes to a discard sink. "
            "See architecture.md:148 for the correct pattern.",
            DeprecationWarning,
            stacklevel=2,
        )
```

**Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/plugins/transforms/test_filter_deprecation.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/plugins/transforms/filter.py tests/plugins/transforms/test_filter_deprecation.py
git commit -m "$(cat <<'EOF'
deprecate(transforms): mark Filter as deprecated in favor of FilterGate

Filter transform violates "no silent drops" invariant. Users should
migrate to FilterGate which routes to an explicit discard sink.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire row_plugins to Execution

**Files:**
- Modify: `src/elspeth/cli.py` (lines 221-281)
- Test: `tests/cli/test_cli.py`

**Step 1: Write the failing test**

Add to `tests/cli/test_cli.py`:

```python
class TestRunCommandExecutesTransforms:
    """Verify row_plugins are actually executed."""

    def test_transforms_from_config_are_instantiated(self, tmp_path: Path) -> None:
        """Transforms in row_plugins are instantiated and passed to orchestrator."""
        from typer.testing import CliRunner

        from elspeth.cli import app

        runner = CliRunner()

        # Create input CSV
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,hello\n2,world\n")

        output_file = tmp_path / "output.csv"
        audit_db = tmp_path / "audit.db"

        # Config with a passthrough transform
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(f"""
datasource:
  plugin: csv
  options:
    path: "{input_file}"

sinks:
  results:
    plugin: csv
    options:
      path: "{output_file}"

row_plugins:
  - plugin: passthrough

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: "sqlite:///{audit_db}"
""")

        result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])

        assert result.exit_code == 0
        # Output should exist with data processed
        assert output_file.exists()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/cli/test_cli.py::TestRunCommandExecutesTransforms -v`
Expected: FAIL (transforms not actually executed, just `transforms=[]`)

**Step 3: Write implementation**

Modify `src/elspeth/cli.py`. After line 226 (after existing imports), add:

```python
    from elspeth.plugins.base import BaseGate, BaseTransform
    from elspeth.plugins.transforms.passthrough import PassThrough
    from elspeth.plugins.transforms.field_mapper import FieldMapper
    from elspeth.plugins.gates.threshold_gate import ThresholdGate
    from elspeth.plugins.gates.field_match_gate import FieldMatchGate
    from elspeth.plugins.gates.filter_gate import FilterGate
```

Then, after the sinks loop (after line 270, before "# Get database URL"), add:

```python
    # Plugin registries
    TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
        "passthrough": PassThrough,
        "field_mapper": FieldMapper,
    }
    GATE_PLUGINS: dict[str, type[BaseGate]] = {
        "threshold_gate": ThresholdGate,
        "field_match_gate": FieldMatchGate,
        "filter_gate": FilterGate,
    }

    # Instantiate transforms/gates from row_plugins
    transforms: list[BaseTransform | BaseGate] = []
    for plugin_config in config.row_plugins:
        plugin_name = plugin_config.plugin
        plugin_options = dict(plugin_config.options)

        if plugin_config.type == "gate":
            if plugin_name not in GATE_PLUGINS:
                raise typer.BadParameter(f"Unknown gate plugin: {plugin_name}")
            gate_class = GATE_PLUGINS[plugin_name]
            transforms.append(gate_class(plugin_options))
        else:
            if plugin_name not in TRANSFORM_PLUGINS:
                raise typer.BadParameter(f"Unknown transform plugin: {plugin_name}")
            transform_class = TRANSFORM_PLUGINS[plugin_name]
            transforms.append(transform_class(plugin_options))
```

Then update the PipelineConfig construction (around line 277-281):

```python
    # Build PipelineConfig
    pipeline_config = PipelineConfig(
        source=source,
        transforms=transforms,  # Now uses instantiated transforms
        sinks=sinks,
    )
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/cli/test_cli.py::TestRunCommandExecutesTransforms -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/elspeth/cli.py tests/cli/test_cli.py
git commit -m "$(cat <<'EOF'
feat(cli): wire row_plugins to pipeline execution

Transforms and gates from row_plugins are now instantiated and
passed to the orchestrator instead of being ignored with transforms=[].

Fixes architecture alignment issue #4.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final Verification

**Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: All tests pass

**Step 2: Run type checking**

Run: `.venv/bin/python -m mypy src/elspeth/core/dag.py src/elspeth/engine/orchestrator.py src/elspeth/cli.py src/elspeth/plugins/gates/filter_gate.py`
Expected: Success

**Step 3: Run linting**

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No errors (or only pre-existing ones)

**Step 4: Manual verification**

Create a test config that exercises all fixes:

```bash
cat > /tmp/test_all_fixes.yaml << 'EOF'
datasource:
  plugin: csv
  options:
    path: /tmp/input.csv

sinks:
  results:
    plugin: csv
    options:
      path: /tmp/results.csv
  discarded:
    plugin: csv
    options:
      path: /tmp/discarded.csv

row_plugins:
  - plugin: filter_gate
    type: gate
    options:
      field: score
      greater_than: 0.5
      discard_sink: discarded
    routes:
      discarded: discarded

output_sink: results

landscape:
  enabled: true
  backend: sqlite
  url: sqlite:////tmp/audit.db
EOF

echo "id,score" > /tmp/input.csv
echo "1,0.8" >> /tmp/input.csv
echo "2,0.3" >> /tmp/input.csv
echo "3,0.9" >> /tmp/input.csv
```

Run validation:
```bash
.venv/bin/python -m elspeth validate -s /tmp/test_all_fixes.yaml
```

Expected: Validation passes, shows graph info

---

## Summary

| Task | Description | Issue Fixed |
|------|-------------|-------------|
| 1 | Use `graph.get_output_sink()` | #1 - Output sink routing |
| 2 | Add route label mapping | #2 - Edge label semantics (prep) |
| 3 | Key edge_map by sink name | #2 - Edge label semantics (fix) |
| 4 | Create FilterGate | #3 - No silent drops |
| 5 | Deprecate Filter transform | #3 - No silent drops |
| 6 | Wire row_plugins to execution | #4 - Transform execution |
| 7 | Final verification | All issues |

**Estimated total:** ~350-450 lines changed across 7 files
