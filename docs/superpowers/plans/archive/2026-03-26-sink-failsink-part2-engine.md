# Sink Failsink Part 2: Engine & Wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the failsink infrastructure from Part 1 into the engine: SinkExecutor handles diversions, config validation enforces failsink rules, DAG builder creates DIVERT edges, orchestrator injects `_on_write_failure` and resolves failsink references, and ChromaSink migrates its inline filtering to `_divert_row()`.

**Architecture:** SinkExecutor reads `SinkWriteResult.diversions` after `write()`, writes diverted rows to the failsink (or records discard), and records per-token `DIVERTED` outcomes. The orchestrator resolves failsink references at write time (inside `_write_pending_to_sinks()`, from `config.sinks`) and passes them to the executor. ChromaSink's inline metadata filtering is replaced with `_divert_row()` calls.

**Tech Stack:** Python, SQLAlchemy Core (audit trail), pytest, Hypothesis (property tests).

**Spec:** `docs/superpowers/specs/2026-03-26-sink-failsink-design.md`

**Depends on:** Part 1 (contracts & infrastructure) — all types, enums, config fields, BaseSink plumbing, and sink return type migration must be complete.

---

### Task 1: Config validation — validate_sink_failsink_destinations()

**Files:**
- Modify: `src/elspeth/engine/orchestrator/validation.py`
- Create: `tests/unit/engine/test_failsink_validation.py`

- [ ] **Step 1: Write tests for failsink config validation**

```python
# tests/unit/engine/test_failsink_validation.py
"""Tests for sink failsink destination validation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.engine.orchestrator.validation import validate_sink_failsink_destinations


def _stub(on_write_failure: str) -> SimpleNamespace:
    """Minimal stub with on_write_failure attribute."""
    return SimpleNamespace(on_write_failure=on_write_failure)


class TestValidateSinkFailsinkDestinations:
    def test_discard_always_valid(self) -> None:
        """on_write_failure='discard' needs no target sink."""
        validate_sink_failsink_destinations(
            sink_configs={"output": _stub("discard")},
            available_sinks={"output"},
            sink_plugins={"output": "chroma_sink"},
        )  # No error raised

    def test_valid_failsink_reference(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub("csv_failsink"),
                "csv_failsink": _stub("discard"),
            },
            available_sinks={"output", "csv_failsink"},
            sink_plugins={"output": "chroma_sink", "csv_failsink": "csv"},
        )  # No error raised

    def test_json_failsink_valid(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub("json_failsink"),
                "json_failsink": _stub("discard"),
            },
            available_sinks={"output", "json_failsink"},
            sink_plugins={"output": "database", "json_failsink": "json"},
        )  # No error raised

    def test_xml_failsink_valid(self) -> None:
        validate_sink_failsink_destinations(
            sink_configs={
                "output": _stub("xml_failsink"),
                "xml_failsink": _stub("discard"),
            },
            available_sinks={"output", "xml_failsink"},
            sink_plugins={"output": "database", "xml_failsink": "xml"},
        )  # No error raised

    def test_unknown_failsink_raises(self) -> None:
        with pytest.raises(RouteValidationError, match="nonexistent"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub("nonexistent")},
                available_sinks={"output"},
                sink_plugins={"output": "chroma_sink"},
            )

    def test_non_file_failsink_raises(self) -> None:
        """Failsink must be csv, json, or xml."""
        with pytest.raises(RouteValidationError, match="csv, json, or xml"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("db_sink"),
                    "db_sink": _stub("discard"),
                },
                available_sinks={"output", "db_sink"},
                sink_plugins={"output": "chroma_sink", "db_sink": "database"},
            )

    def test_chroma_as_failsink_raises(self) -> None:
        """ChromaSink is not a valid failsink plugin type."""
        with pytest.raises(RouteValidationError, match="csv, json, or xml"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("chroma_backup"),
                    "chroma_backup": _stub("discard"),
                },
                available_sinks={"output", "chroma_backup"},
                sink_plugins={"output": "csv", "chroma_backup": "chroma_sink"},
            )

    def test_failsink_chaining_raises(self) -> None:
        """Failsink targets must have on_write_failure='discard' (no chains)."""
        with pytest.raises(RouteValidationError, match="discard"):
            validate_sink_failsink_destinations(
                sink_configs={
                    "output": _stub("failsink1"),
                    "failsink1": _stub("failsink2"),
                    "failsink2": _stub("discard"),
                },
                available_sinks={"output", "failsink1", "failsink2"},
                sink_plugins={"output": "chroma_sink", "failsink1": "csv", "failsink2": "csv"},
            )

    def test_self_reference_raises(self) -> None:
        """A sink cannot reference itself as failsink."""
        with pytest.raises(RouteValidationError, match="itself"):
            validate_sink_failsink_destinations(
                sink_configs={"output": _stub("output")},
                available_sinks={"output"},
                sink_plugins={"output": "csv"},
            )

    def test_multiple_sinks_mixed_valid(self) -> None:
        """Multiple sinks: some with failsink, some with discard."""
        validate_sink_failsink_destinations(
            sink_configs={
                "chroma_out": _stub("csv_fail"),
                "db_out": _stub("discard"),
                "csv_fail": _stub("discard"),
            },
            available_sinks={"chroma_out", "db_out", "csv_fail"},
            sink_plugins={"chroma_out": "chroma_sink", "db_out": "database", "csv_fail": "csv"},
        )  # No error raised

    def test_all_discard(self) -> None:
        """All sinks using discard — valid."""
        validate_sink_failsink_destinations(
            sink_configs={
                "sink_a": _stub("discard"),
                "sink_b": _stub("discard"),
            },
            available_sinks={"sink_a", "sink_b"},
            sink_plugins={"sink_a": "csv", "sink_b": "json"},
        )  # No error raised
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_failsink_validation.py -v`
Expected: `ImportError: cannot import name 'validate_sink_failsink_destinations'`

- [ ] **Step 3: Implement validate_sink_failsink_destinations()**

In `src/elspeth/engine/orchestrator/validation.py`, add after the existing `validate_source_quarantine_destination()` function:

```python
_ALLOWED_FAILSINK_PLUGINS = frozenset({"csv", "json", "xml"})


def validate_sink_failsink_destinations(
    sink_configs: Mapping[str, Any],
    available_sinks: set[str],
    sink_plugins: Mapping[str, str],
    allowed_failsink_plugins: frozenset[str] = _ALLOWED_FAILSINK_PLUGINS,
) -> None:
    """Validate all sink on_write_failure destinations.

    Called at pipeline initialization, before any rows are processed.
    Parallel to validate_transform_error_sinks() for transform on_error.

    Rules:
    1. 'discard' is always valid
    2. Sink name must exist in available_sinks
    3. Sink cannot reference itself
    4. Target sink must use csv, json, or xml plugin type
    5. Target sink must have on_write_failure='discard' (no chains)

    Args:
        sink_configs: Dict of sink_name -> config object with on_write_failure attr.
        available_sinks: Set of all sink names in the pipeline.
        sink_plugins: Dict of sink_name -> plugin type name (e.g., "csv", "chroma_sink").
        allowed_failsink_plugins: Set of plugin types allowed as failsinks.

    Raises:
        RouteValidationError: If any sink's on_write_failure is invalid.
    """
    for sink_name, config in sink_configs.items():
        dest = config.on_write_failure
        if dest == "discard":
            continue

        # Rule 2: must exist
        if dest not in available_sinks:
            raise RouteValidationError(
                f"Sink '{sink_name}' on_write_failure references unknown sink '{dest}'. "
                f"Available sinks: {sorted(available_sinks)}."
            )

        # Rule 3: no self-reference
        if dest == sink_name:
            raise RouteValidationError(
                f"Sink '{sink_name}' on_write_failure references itself. "
                f"A sink cannot be its own failsink."
            )

        # Rule 4: must be a file sink
        if dest in sink_plugins:
            plugin_type = sink_plugins[dest]
            if plugin_type not in allowed_failsink_plugins:
                raise RouteValidationError(
                    f"Sink '{sink_name}' on_write_failure references '{dest}' "
                    f"(plugin='{plugin_type}'), but failsinks must use csv, json, or xml plugins."
                )

        # Rule 5: no chains — target must use 'discard'
        if dest in sink_configs:
            target_dest = sink_configs[dest].on_write_failure
            if target_dest != "discard":
                raise RouteValidationError(
                    f"Sink '{sink_name}' on_write_failure references '{dest}', "
                    f"but '{dest}' has on_write_failure='{target_dest}'. "
                    f"Failsink targets must have on_write_failure='discard' (no chains)."
                )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_failsink_validation.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/engine/orchestrator/validation.py tests/unit/engine/test_failsink_validation.py
git commit -m "feat(validation): add validate_sink_failsink_destinations()"
```

---

### Task 2: DAG builder __failsink__ DIVERT edges

**Files:**
- Modify: `src/elspeth/core/dag/builder.py:725-763`

- [ ] **Step 1: Write test for __failsink__ edge creation**

```python
# tests/unit/core/dag/test_failsink_edges.py
"""Test __failsink__ DIVERT edge creation in DAG builder."""
from __future__ import annotations

from elspeth.contracts.enums import RoutingMode
from elspeth.core.dag.graph import ExecutionGraph


class TestFailsinkEdges:
    def test_failsink_divert_edge_created(self, dag_with_failsink: ExecutionGraph) -> None:
        """A sink with on_write_failure creates a __failsink__ DIVERT edge."""
        edges = dag_with_failsink.get_edges()
        failsink_edges = [e for e in edges if e.label == "__failsink__"]
        assert len(failsink_edges) == 1
        assert failsink_edges[0].mode == RoutingMode.DIVERT

    def test_discard_no_failsink_edge(self, dag_with_discard: ExecutionGraph) -> None:
        """A sink with on_write_failure='discard' creates no __failsink__ edge."""
        edges = dag_with_discard.get_edges()
        failsink_edges = [e for e in edges if e.label == "__failsink__"]
        assert len(failsink_edges) == 0
```

Note: These tests require DAG construction fixtures. Use the existing `from_plugin_instances()` test patterns in `tests/unit/core/dag/`. The fixtures need a pipeline config with two sinks where one references the other via `on_write_failure`. Check existing DAG test fixtures for the pattern and create similar ones.

- [ ] **Step 2: Add __failsink__ edge creation to DAG builder**

In `src/elspeth/core/dag/builder.py`, find the end of the DIVERT edges section (around line 763, after transform error edges). The function `from_plugin_instances()` needs access to sink settings to know which sinks have `on_write_failure` configured.

Check the current signature of `from_plugin_instances()` — it receives `sinks: Mapping[SinkName, SinkProtocol]`. It needs a way to get `on_write_failure` for each sink. Options:
- Pass the sink_settings separately: `sink_settings: Mapping[str, SinkSettings]`
- Read `_on_write_failure` from the sink instance (already injected by cli_helpers in Task 3)

The simpler approach: read from the sink instance. After the transform error edges block, add:

```python
    # Sink failsink edges
    for sink_name_key, sink_node_id in sink_ids.items():
        sink_instance = sinks[sink_name_key]
        on_write_failure = sink_instance._on_write_failure
        if on_write_failure is not None and on_write_failure != "discard":
            failsink_name = SinkName(on_write_failure)
            if failsink_name in sink_ids:
                graph.add_edge(
                    sink_node_id,
                    sink_ids[failsink_name],
                    label="__failsink__",
                    mode=RoutingMode.DIVERT,
                )
```

**IMPORTANT:** Use direct attribute access `sink_instance._on_write_failure`, NOT `getattr()`. The `_on_write_failure` attribute is defined as a class-level attribute on `BaseSink` with default `None` (Part 1 Task 4), so it exists on every `BaseSink` subclass instance. Using `getattr()` with a default violates CLAUDE.md's offensive programming rules. Test fixtures that use mock sinks must set `_on_write_failure` explicitly.

- [ ] **Step 3: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/core/dag/test_failsink_edges.py -v`
Expected: All PASS.

- [ ] **Step 4: Run existing DAG tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/core/dag/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/core/dag/builder.py tests/unit/core/dag/test_failsink_edges.py
git commit -m "feat(dag): add __failsink__ DIVERT edges for sink failsink routing"
```

---

### Task 3: Orchestrator wiring — inject _on_write_failure + resolve failsink

**Files:**
- Modify: `src/elspeth/cli_helpers.py:107-119`
- Modify: `src/elspeth/engine/orchestrator/core.py:514-585`

- [ ] **Step 1: Inject _on_write_failure in cli_helpers.py**

In `src/elspeth/cli_helpers.py`, find the sink instantiation loop (lines 107-111):

```python
    # Instantiate sinks
    sinks = {}
    for sink_name, sink_config in config.sinks.items():
        sink_cls = manager.get_sink_by_name(sink_config.plugin)
        sinks[sink_name] = sink_cls(dict(sink_config.options))
```

Add after `sinks[sink_name] = sink_cls(dict(sink_config.options))`:

```python
        # Bridge: inject on_write_failure from settings level
        sinks[sink_name]._on_write_failure = sink_config.on_write_failure
```

- [ ] **Step 2: Thread `edge_map` through the call chain and resolve failsink**

**Background (from design spike):** The `edge_map` (`Mapping[tuple[NodeID, str], str]`) already exists in `GraphArtifacts` (built at ~line 1393 of `core.py`). It is available at both call sites of `_flush_and_write_sinks()` via `artifacts.edge_map`. The quarantine system (line 1788) uses the exact same pattern: `edge_map[(node_id, "__quarantine__")]` with a `KeyError → OrchestrationInvariantError` crash guard. Mirror that pattern — do NOT use `.get()`.

**2a. Add `edge_map` parameter to `_flush_and_write_sinks()` signature** (line 1636):

```python
def _flush_and_write_sinks(
    self,
    recorder: LandscapeRecorder,
    run_id: str,
    loop_ctx: LoopContext,
    sink_id_map: Mapping[SinkName, NodeID],
    edge_map: Mapping[tuple[NodeID, str], str],  # NEW — from GraphArtifacts
    interrupted_by_shutdown: bool,
    *,
    on_token_written_factory: _CheckpointFactory | None = None,
    shutdown_checkpoint_source_id: NodeID | None = None,
) -> None:
```

**2b. Update both call sites of `_flush_and_write_sinks()`** to pass `edge_map`:

Call site 1 — `_execute_run()` (~line 2464):
```python
self._flush_and_write_sinks(
    recorder,
    run_id,
    loop_ctx,
    artifacts.sink_id_map,
    artifacts.edge_map,          # NEW
    loop_result.interrupted,
    on_token_written_factory=...,
    shutdown_checkpoint_source_id=artifacts.source_id,
)
```

Call site 2 — `_process_resumed_rows()` (~line 2804):
```python
self._flush_and_write_sinks(
    recorder,
    run_id,
    loop_ctx,
    artifacts.sink_id_map,
    artifacts.edge_map,          # NEW
    interrupted,
    on_token_written_factory=...,
    shutdown_checkpoint_source_id=artifacts.source_id,
)
```

**2c. Add `edge_map` parameter to `_write_pending_to_sinks()` signature** (line 511):

```python
def _write_pending_to_sinks(
    self,
    recorder: LandscapeRecorder,
    run_id: str,
    config: PipelineConfig,
    ctx: PluginContext,
    pending_tokens: PendingTokenMap,
    sink_id_map: dict[SinkName, NodeID],
    edge_map: Mapping[tuple[NodeID, str], str],  # NEW
    sink_step: int,
    *,
    on_token_written_factory: Callable[[str], Callable[[TokenInfo], None]] | None = None,
) -> int:  # NEW return type — total diversion count
```

Change the return type from `None` to `int`. Add `total_diversions = 0` at the top of the method body. Return `total_diversions` at the end.

**2d. Update the call from `_flush_and_write_sinks()` to `_write_pending_to_sinks()`** (~line 1659):

```python
total_diversions = self._write_pending_to_sinks(
    recorder=recorder,
    run_id=run_id,
    config=loop_ctx.config,
    ctx=loop_ctx.ctx,
    pending_tokens=loop_ctx.pending_tokens,
    sink_id_map=dict(sink_id_map),
    edge_map=edge_map,              # NEW
    sink_step=loop_ctx.processor.resolve_sink_step(),
    on_token_written_factory=on_token_written_factory,
)
loop_ctx.counters.rows_diverted += total_diversions
```

(`counters` is already available in `_flush_and_write_sinks` as `loop_ctx.counters`.)

**2e. Resolve failsink inside `_write_pending_to_sinks()`.**

In `src/elspeth/engine/orchestrator/core.py`, find `_write_pending_to_sinks()` (line 511). Inside the loop that calls `sink_executor.write()` (around line 575), resolve the failsink and its edge_id before calling write:

Find:
```python
            for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
                group_tokens = [token for token, _ in group]
                sink_executor.write(
                    sink=sink,
                    tokens=group_tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    on_token_written=on_token_written,
                )
```

Replace with:

```python
            # Resolve failsink reference (if configured and not 'discard')
            # Direct attribute access — _on_write_failure is always present on BaseSink
            # (class-level default None, overwritten by cli_helpers injection)
            failsink: SinkProtocol | None = None
            failsink_config_name: str | None = None
            failsink_edge_id: str | None = None
            on_write_failure = sink._on_write_failure
            if on_write_failure is not None and on_write_failure != "discard":
                # Validation already confirmed this sink exists at startup.
                # If it's missing now, that's an orchestration bug — crash, don't silently degrade.
                if on_write_failure not in config.sinks:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure references '{on_write_failure}' "
                        f"which passed validation but is not in config.sinks at runtime. "
                        f"Available: {sorted(config.sinks.keys())}."
                    )
                failsink = config.sinks[on_write_failure]
                failsink_config_name = on_write_failure
                # Resolve __failsink__ edge_id from the DAG edge_map.
                # Mirrors the quarantine pattern at line 1788: crash guard, not .get().
                # edge_map is keyed by (from_node_id, label).
                failsink_edge_key = (sink_node_id, "__failsink__")
                try:
                    failsink_edge_id = edge_map[failsink_edge_key]
                except KeyError as exc:
                    raise OrchestrationInvariantError(
                        f"Sink '{sink_name}' on_write_failure='{on_write_failure}' "
                        f"but no __failsink__ DIVERT edge exists in DAG for node '{sink_node_id}'. "
                        f"This is a DAG construction bug — on_write_failure should have "
                        f"created a DIVERT edge in from_plugin_instances()."
                    ) from exc

            for pending_outcome, group in groupby(sorted_pairs, key=lambda x: x[1]):
                group_tokens = [token for token, _ in group]
                _, diversion_count = sink_executor.write(
                    sink=sink,
                    tokens=group_tokens,
                    ctx=ctx,
                    step_in_pipeline=step,
                    sink_name=sink_name,
                    pending_outcome=pending_outcome,
                    failsink=failsink,
                    failsink_name=failsink_config_name,
                    failsink_edge_id=failsink_edge_id,
                    on_token_written=on_token_written,
                )
                total_diversions += diversion_count
```

Add `SinkProtocol` to the TYPE_CHECKING imports if not already present. Ensure `OrchestrationInvariantError` is imported from `elspeth.contracts.errors`.

- [ ] **Step 3: Call validate_sink_failsink_destinations() at pipeline init**

Find `validate_source_quarantine_destination()` in `core.py` (lines 1431-1434). Add `validate_sink_failsink_destinations()` immediately after it. The function uses sink instances from `config.sinks` — each sink has `._on_write_failure` (injected by cli_helpers in Step 1) and `.name` (the plugin type).

Build a stub-like wrapper so the validation function can read `on_write_failure` from the instances:

Add `validate_sink_failsink_destinations` to the existing module-level import block (lines 124-128 of `core.py`):

```python
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)
```

Then at the call site (after `validate_source_quarantine_destination()` at line 1434):

```python
# Build validation inputs from instantiated sinks
# Each sink instance has _on_write_failure (injected by cli_helpers)
# and .name (the plugin type name)
sink_validation_stubs = {
    name: SimpleNamespace(on_write_failure=sink._on_write_failure)
    for name, sink in config.sinks.items()
}
sink_plugins = {name: sink.name for name, sink in config.sinks.items()}
validate_sink_failsink_destinations(
    sink_configs=sink_validation_stubs,
    available_sinks=set(config.sinks.keys()),
    sink_plugins=sink_plugins,
)
```

Add `from types import SimpleNamespace` to the imports at the call site. This follows the existing pattern where validation functions at lines 1413-1434 work with `config.sinks` (instantiated plugins), not raw SinkSettings.

The same validation call must be added in the resume path (around line 1622, after the existing `validate_source_quarantine_destination()` call).

- [ ] **Step 4: Run integration tests**

Run: `.venv/bin/python -m pytest tests/integration/ -v --tb=short -q`
Expected: All PASS (or existing failures only — integration tests may need `on_write_failure` added to their YAML fixtures).

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/cli_helpers.py src/elspeth/engine/orchestrator/core.py
git commit -m "feat(orchestrator): wire on_write_failure injection and failsink resolution"
```

---

### Task 4: SinkExecutor failsink routing

**Files:**
- Modify: `src/elspeth/engine/executors/sink.py:94-401`
- Create: `tests/unit/engine/test_sink_executor_diversion.py`

This is the most critical task. The SinkExecutor must handle per-token outcomes based on `SinkWriteResult.diversions`.

- [ ] **Step 1: Write comprehensive tests**

```python
# tests/unit/engine/test_sink_executor_diversion.py
"""Tests for SinkExecutor failsink routing.

Tests the critical path: after sink.write() returns a SinkWriteResult with
diversions, the executor must record correct per-token outcomes and write
diverted rows to the failsink (or record discard).
"""
from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from elspeth.contracts import PendingOutcome, RowOutcome, TokenInfo
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


def _make_token(token_id: str = "tok-1", row_data: dict | None = None) -> MagicMock:
    """Create a minimal TokenInfo mock."""
    token = MagicMock(spec=TokenInfo)
    token.token_id = token_id
    token.row_id = f"row-{token_id}"
    mock_row = MagicMock()
    mock_row.to_dict.return_value = row_data or {"field": "value"}
    mock_row.contract = MagicMock()
    mock_row.contract.merge.return_value = mock_row.contract
    token.row_data = mock_row
    return token


def _make_artifact(path: str = "/tmp/test") -> ArtifactDescriptor:
    return ArtifactDescriptor.for_file(path=path, content_hash="a" * 64, size_bytes=100)


def _make_sink(
    name: str = "primary",
    node_id: str = "node-primary",
    diversions: tuple[RowDiversion, ...] = (),
    on_write_failure: str = "discard",
) -> MagicMock:
    sink = MagicMock()
    sink.name = name
    sink.node_id = node_id
    sink.validate_input = False
    sink.declared_required_fields = frozenset()
    sink.write.return_value = SinkWriteResult(
        artifact=_make_artifact(),
        diversions=diversions,
    )
    sink._on_write_failure = on_write_failure
    sink._reset_diversion_log = MagicMock()
    return sink


def _make_failsink(name: str = "csv_failsink", node_id: str = "node-failsink") -> MagicMock:
    failsink = MagicMock()
    failsink.name = name
    failsink.node_id = node_id
    failsink.write.return_value = SinkWriteResult(artifact=_make_artifact("/tmp/failsink"))
    return failsink


def _make_executor() -> tuple[SinkExecutor, MagicMock]:
    recorder = MagicMock()
    state_counter = [0]

    def _begin_state(**kwargs: Any) -> MagicMock:
        state_counter[0] += 1
        state = MagicMock()
        state.state_id = f"state-{state_counter[0]}"
        return state

    recorder.begin_node_state.side_effect = _begin_state
    recorder.allocate_operation_call_index = MagicMock(return_value=0)
    spans = MagicMock()
    spans.sink_span.return_value.__enter__ = MagicMock(return_value=None)
    spans.sink_span.return_value.__exit__ = MagicMock(return_value=False)
    executor = SinkExecutor(recorder, spans, "run-1")
    return executor, recorder


class TestNoDiversions:
    """Existing behavior preserved when no diversions occur."""

    def test_all_tokens_get_completed_outcome(self) -> None:
        executor, recorder = _make_executor()
        sink = _make_sink()
        tokens = [_make_token("t0"), _make_token("t1"), _make_token("t2")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 3
        for c in outcome_calls:
            assert c.kwargs["outcome"] == RowOutcome.COMPLETED
            assert c.kwargs["sink_name"] == "primary"

    def test_no_failsink_write_called(self) -> None:
        executor, recorder = _make_executor()
        sink = _make_sink()
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink,
        )
        failsink.write.assert_not_called()


class TestDiscardMode:
    """on_write_failure='discard' — diverted rows are dropped with audit record."""

    def test_diverted_tokens_get_diverted_outcome(self) -> None:
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=1, reason="bad metadata", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert len(outcome_calls) == 2
        # t0 (index 0) → COMPLETED
        assert outcome_calls[0].kwargs["outcome"] == RowOutcome.COMPLETED
        assert outcome_calls[0].kwargs["sink_name"] == "primary"
        # t1 (index 1) → DIVERTED
        assert outcome_calls[1].kwargs["outcome"] == RowOutcome.DIVERTED
        assert outcome_calls[1].kwargs["error_hash"] is not None

    def test_all_diverted_all_get_diverted(self) -> None:
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
            RowDiversion(row_index=1, reason="bad", row_data={"x": 2}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert all(c.kwargs["outcome"] == RowOutcome.DIVERTED for c in outcome_calls)


class TestFailsinkMode:
    """on_write_failure=<sink_name> — diverted rows are written to failsink."""

    def test_failsink_write_called_with_enriched_rows(self) -> None:
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=1, reason="invalid metadata", row_data={"doc": "hello"}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink,
        )
        # Failsink should have been called with the diverted row
        failsink.write.assert_called_once()
        failsink_rows = failsink.write.call_args[0][0]
        assert len(failsink_rows) == 1
        assert "__diversion_reason" in failsink_rows[0]
        assert failsink_rows[0]["__diversion_reason"] == "invalid metadata"
        assert failsink_rows[0]["__diverted_from"] == "primary"

    def test_failsink_flush_called(self) -> None:
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink,
        )
        failsink.flush.assert_called_once()

    def test_no_diversions_no_failsink_call(self) -> None:
        executor, recorder = _make_executor()
        sink = _make_sink(diversions=(), on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink,
        )
        failsink.write.assert_not_called()

    def test_diverted_tokens_get_failsink_sink_name(self) -> None:
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0")]
        executor.write(
            sink=sink,
            tokens=tokens,
            ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5,
            sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink,
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        assert outcome_calls[0].kwargs["sink_name"] == "csv_failsink"


class TestFailsinkErrorHandling:
    def test_failsink_write_failure_crashes(self) -> None:
        """If failsink write fails, crash — it's the last resort."""
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink=sink,
                tokens=tokens,
                ctx=MagicMock(run_id="run-1"),
                step_in_pipeline=5,
                sink_name="primary",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                failsink=failsink,
            )
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -v`
Expected: FAIL — SinkExecutor doesn't handle SinkWriteResult diversions yet.

- [ ] **Step 3: Modify SinkExecutor.write() to handle diversions**

**CRITICAL ARCHITECTURAL CHANGE:** The existing `write()` opens `begin_node_state` for ALL tokens at the primary sink's `node_id` BEFORE calling `sink.write()`. But diversions are only known AFTER `write()` returns. The spec requires diverted tokens to have NO node_state at the primary sink — only at the failsink.

**The fix:** Defer `begin_node_state` for diverted tokens. Call `sink.write()` first, determine the diversion set, THEN open node_states only for non-diverted tokens at the primary node_id. For diverted tokens, record a routing_event and open node_states at the failsink's node_id.

**Design spike findings (safe to proceed):** In the current code, `begin_node_state()` calls are already OUTSIDE the `with track_operation()` block (lines 151-164 are before the `with` at line 218). The `with` block wraps only the sink I/O (`write()` + `flush()` + `handle.output_data` assignment). `complete_node_state()`, `register_artifact()`, and `record_token_outcome()` are also already OUTSIDE the `with` block (lines 325-375). The only thing that MUST stay inside the `with` block is `handle.output_data = {...}` (line 320), because the context manager's `finally` block reads it on exit via `recorder.complete_operation()`.

Moving `begin_node_state` from *before* the `with` block to *after* it (Phase 2 below) is safe — it doesn't interact with the operation context. The `ctx.state_id = None` clear (line 212, required for the XOR constraint on the `calls` table) must stay before the `with` block — it ensures sink external calls use `operation_id`, not `state_id`, as their parent.

**Simplified error path:** In the current code, if `sink.write()` throws, `_complete_states_failed()` cleans up already-opened states. In the new flow, primary states aren't opened until after `write()` succeeds (Phase 2), so there's nothing to clean up on primary write failure — the error path is simpler. Cleanup complexity shifts to Phase 3 (failsink write), where opened failsink states need `_complete_states_failed()` if the failsink write fails.

In `src/elspeth/engine/executors/sink.py`:

Add imports at top:
```python
from datetime import UTC, datetime
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import RowOutcome
```

Add `failsink` parameter to `write()` signature (line 94):
```python
    def write(
        self,
        sink: SinkProtocol,
        tokens: list[TokenInfo],
        ctx: PluginContext,
        step_in_pipeline: int,
        *,
        sink_name: str,
        pending_outcome: PendingOutcome,
        failsink: SinkProtocol | None = None,
        failsink_name: str | None = None,
        failsink_edge_id: str | None = None,
        on_token_written: Callable[[TokenInfo], None] | None = None,
    ) -> tuple[Artifact | None, int]:
```

**Revised flow (replaces the current begin→write→flush→complete→register→outcomes sequence):**

The key structural change: `track_operation` wraps ONLY `sink.write()` + `sink.flush()` (external I/O). Node_states and outcomes are recorded OUTSIDE the `with` block, after the diversion set is known. This matches the existing code structure — `begin_node_state` and `complete_node_state` are already outside the `with` block in the current implementation.

```
PHASE 1 — External I/O (inside track_operation):
  1. Reset primary sink's diversion log
  2. Merge contracts (unchanged)
  3. Clear ctx.state_id (unchanged — XOR constraint, must stay BEFORE the with block)
  4. Enter track_operation context
  5. Run centralized input validation (unchanged)
  6. Call sink.write(rows, ctx) → SinkWriteResult
  7. Extract artifact_info = write_result.artifact, diversions = write_result.diversions
  8. Flush primary sink
  9. Set handle.output_data from artifact_info
     *** handle.output_data MUST be assigned inside the with block — the context
         manager's finally reads it on exit via recorder.complete_operation() ***
  10. Exit track_operation context (finally block runs here, reads handle.output_data)

PHASE 2 — Partition and record primary tokens (outside track_operation):
  11. Build diverted_indices = {d.row_index for d in diversions}
  12. Partition: primary_pairs = [(token, idx) for non-diverted], diverted_pairs = [(token, idx) for diverted]
  13. For PRIMARY tokens (if any):
      a. begin_node_state at primary sink's node_id (with try/except for partial-open cleanup)
      b. complete_node_state as COMPLETED with artifact reference
      c. register_artifact for primary (anchored to first primary state)
      d. record_token_outcome COMPLETED with pending_outcome
      e. on_token_written callback per primary token

PHASE 3 — Handle diversions (outside track_operation):
  14. For DIVERTED tokens (if any):

      **If failsink mode (on_write_failure is a sink name, not "discard"):**
      a. Reset failsink's diversion log
      b. Build enriched rows with __diversion_reason, __diverted_from, __diversion_timestamp
      c. Write to failsink: failsink.write(enriched_rows, ctx)
      d. Flush failsink
      e. Assert failsink_write_result.diversions is empty (failsink must not produce diversions)
      f. begin_node_state per diverted token at failsink's node_id
         (with try/except for partial-open cleanup — if begin_node_state fails,
          complete opened failsink states as FAILED, do NOT touch primary states)
      g. Record routing_event per diverted token AFTER its failsink state is opened:
         state_id=failsink_state.state_id, edge_id=failsink_edge_id,
         mode=RoutingMode.DIVERT, reason=RowDiversion.reason
         (NOTE: state_id is the failsink state, not a primary state — diverted tokens
          have no primary state. The routing_event is anchored to the destination.)
      h. complete_node_state as COMPLETED at failsink
      i. Register failsink artifact (anchored to first diverted token's failsink state)
      j. record_token_outcome DIVERTED with error_hash and sink_name=failsink_name

      **If discard mode (on_write_failure == "discard"):**
      a. Do NOT record routing_event (no DAG edge exists for discard, no state_id to anchor to).
         The DIVERTED token_outcome with error_hash is the sole audit record.
      b. record_token_outcome DIVERTED with error_hash and sink_name="__discard__"

      **Both modes:**
      c. Do NOT call on_token_written for diverted tokens

PHASE 4 — Increment diversion counter:
  15. Return (artifact, diversion_count) so _write_pending_to_sinks() can add to counters.rows_diverted
```

**Key implementation notes:**

- `write_result.artifact` and `write_result.diversions` accessed directly — NO isinstance check. All sinks return `SinkWriteResult` after Part 1 Task 6. No legacy compatibility shim (CLAUDE.md No Legacy Code Policy).
- `sink._on_write_failure` accessed directly — NO getattr. Attribute always exists on BaseSink subclasses.
- `sink._reset_diversion_log()` called at step 1, `failsink._reset_diversion_log()` called at step 14.b.i — both before their respective `write()` calls.
- `failsink_name` (the config-level sink name) passed explicitly to `write()`, NOT derived from `failsink.name` (which is the plugin type name, not the pipeline sink name).
- Failsink artifact registered via `self._recorder.register_artifact()` at step 14.e, anchored to first diverted token's failsink state_id.
- **routing_event** recorded at step 14.g (failsink mode only, AFTER failsink state is opened) via `self._recorder.record_routing_event()`. Parameters: `state_id=failsink_state.state_id` (the failsink state, opened at step 14.f), `edge_id=failsink_edge_id` (the `__failsink__` DIVERT edge, resolved by the orchestrator and passed as `failsink_edge_id: str | None` parameter), `mode=RoutingMode.DIVERT`, `reason=RowDiversion.reason`. **For discard mode: NO routing_event** — there is no DAG edge and no state_id. The `DIVERTED` token_outcome with error_hash is the sole audit record. All `record_token_outcome` calls must use kwargs (not positional args) to maintain test assertion compatibility.
- Error handling: if failsink write/flush raises at step 14.b, complete any opened failsink node_states as FAILED before re-raising. Primary states from Phase 2 remain COMPLETED (durable, cannot roll back).
- `on_token_written` NOT called for diverted tokens — they didn't write to the primary sink. This means diverted tokens have no checkpoint records; see resume caveat below.
- **Return type change:** `write()` returns `tuple[Artifact | None, int]` — the artifact and the number of diversions. The existing empty-batch early return (`if not tokens: return None`) must change to `return None, 0`. `_write_pending_to_sinks()` accumulates the diversion count into `counters.rows_diverted`. This is how the counter reaches `RunResult` (N1 fix — `accumulate_row_outcomes()` DIVERTED branch is a guard, not the primary increment path).

**Resume caveat (out of scope but documented):** Diverted tokens receive no `on_token_written` callback, so no checkpoint is created. On resume, these tokens will be replayed and re-diverted, producing duplicate `token_outcome=DIVERTED` records for the same `token_id` and duplicate rows in the failsink file. This is an **audit integrity issue** (duplicate terminal outcomes violate the exactly-once invariant) that must be tracked as a P1 follow-up, not a P3 resume UX improvement.

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Run existing SinkExecutor tests for regression**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_executors.py -v --tb=short -q`
Expected: All PASS — existing behavior preserved when diversions is empty.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/engine/executors/sink.py tests/unit/engine/test_sink_executor_diversion.py
git commit -m "feat(executor): add failsink routing to SinkExecutor.write()"
```

---

### Task 5: ChromaSink migration — replace inline filtering with _divert_row()

**Files:**
- Modify: `src/elspeth/plugins/sinks/chroma_sink.py:164-382`
- Modify: `tests/unit/plugins/sinks/test_chroma_sink.py`

- [ ] **Step 1: Write test for ChromaSink using _divert_row()**

```python
# Add to tests/unit/plugins/sinks/test_chroma_sink.py

class TestChromaSinkDivertRow:
    """Test ChromaSink metadata validation via _divert_row() pattern."""

    def test_invalid_metadata_diverted(self, chroma_sink_with_failsink: ChromaSink) -> None:
        """Rows with non-primitive metadata are diverted, not silently filtered."""
        rows = [
            {"doc_id": "1", "content": "hello", "topic": "valid_string"},
            {"doc_id": "2", "content": "world", "topic": {"nested": "dict"}},
        ]
        result = chroma_sink_with_failsink.write(rows, ctx=mock_ctx())
        assert isinstance(result, SinkWriteResult)
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 1
        assert "topic" in result.diversions[0].reason

    def test_all_valid_no_diversions(self, chroma_sink_with_failsink: ChromaSink) -> None:
        """All valid rows — zero diversions."""
        rows = [
            {"doc_id": "1", "content": "hello", "topic": "valid"},
        ]
        result = chroma_sink_with_failsink.write(rows, ctx=mock_ctx())
        assert result.diversions == ()

    def test_all_invalid_all_diverted(self, chroma_sink_with_failsink: ChromaSink) -> None:
        """All rows have bad metadata — all diverted, nothing written to ChromaDB."""
        rows = [
            {"doc_id": "1", "content": "hello", "topic": [1, 2, 3]},
            {"doc_id": "2", "content": "world", "topic": {"nested": True}},
        ]
        result = chroma_sink_with_failsink.write(rows, ctx=mock_ctx())
        assert len(result.diversions) == 2
```

Note: You'll need to create a `chroma_sink_with_failsink` fixture that sets `_on_write_failure = "csv_failsink"` on the ChromaSink instance. Base it on the existing ChromaSink test fixtures.

- [ ] **Step 2: Modify ChromaSink.write() to use _divert_row()**

In `src/elspeth/plugins/sinks/chroma_sink.py`, replace the inline metadata filtering block (lines 187-218). The current code:

```python
        rejected_metadata: list[dict[str, Any]] = []
        if metadatas is not None:
            valid_indices: list[int] = []
            for i, meta in enumerate(metadatas):
                bad_fields = { ... }
                if bad_fields:
                    rejected_metadata.append({...})
                else:
                    valid_indices.append(i)
            if rejected_metadata and valid_indices:
                ids = [ids[i] for i in valid_indices]
                ...
```

Replace with:

```python
        if metadatas is not None:
            valid_ids: list[str] = []
            valid_documents: list[str] = []
            valid_metadatas: list[dict[str, Any]] = []
            for i, meta in enumerate(metadatas):
                bad_fields = {
                    key: type(value).__name__
                    for key, value in meta.items()
                    if value is not None and not isinstance(value, (str, int, float, bool))
                }
                if bad_fields:
                    # _divert_row signature: (row_data: dict, *, row_index: int, reason: str)
                    # First arg is positional (named row_data in BaseSink), rest are keyword-only.
                    self._divert_row(
                        rows[i],
                        row_index=i,
                        reason=f"Invalid ChromaDB metadata types: {bad_fields}",
                    )
                else:
                    valid_ids.append(ids[i])
                    valid_documents.append(documents[i])
                    valid_metadatas.append(meta)
            ids = valid_ids
            documents = valid_documents
            metadatas = valid_metadatas if valid_metadatas else None
```

Also remove the `rejected_metadata` and `rows_rejected_metadata` variables and their usage throughout the rest of `write()` — they're replaced by `self._get_diversions()`.

Update the return statements to use `SinkWriteResult`:
```python
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_database(...),
            diversions=self._get_diversions(),
        )
```

- [ ] **Step 3: Remove stale rejected_metadata references from audit recording**

The existing code records `rejected_metadata_detail` in `ctx.record_call()` response_data. This is replaced by the diversion log. Clean up all references to `rejected_metadata`, `rows_rejected_metadata`, and `rejected_metadata_detail` in the `write()` method.

- [ ] **Step 4: Update existing ChromaSink tests**

In `tests/unit/plugins/sinks/test_chroma_sink.py`:

- Tests in `TestChromaSinkMetadataTypeValidation` that check `response_data["rows_rejected_metadata"]` need to change to check `result.diversions`
- Tests that check `response_data["rejected_metadata_detail"]` should check `result.diversions[0].reason` instead
- All tests that assert `isinstance(result, ArtifactDescriptor)` change to `isinstance(result, SinkWriteResult)`
- All `result.content_hash` become `result.artifact.content_hash`

- [ ] **Step 5: Run ChromaSink tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/plugins/sinks/test_chroma_sink.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/sinks/chroma_sink.py tests/unit/plugins/sinks/test_chroma_sink.py
git commit -m "feat(chroma): migrate inline metadata filtering to _divert_row() pattern"
```

---

### Task 6: Hypothesis property tests (B5)

**Files:**
- Create: `tests/property/engine/test_sink_executor_diversion_properties.py`

- [ ] **Step 1: Write partition completeness property test**

```python
# tests/property/engine/test_sink_executor_diversion_properties.py
"""Hypothesis property tests for SinkExecutor failsink routing.

These verify invariants that hold across ALL possible batch sizes and
diversion patterns — not just the hand-crafted fixtures in unit tests.

NOTE: These tests verify the single-run invariant only. On resume,
diverted tokens may produce duplicate outcomes (see resume caveat
in the spec). That is a known P1 follow-up, not a property violation.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from elspeth.contracts import PendingOutcome, RowOutcome, TokenInfo
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.engine.executors.sink import SinkExecutor


def _make_token(token_id: str, row_data: dict | None = None) -> MagicMock:
    token = MagicMock(spec=TokenInfo)
    token.token_id = token_id
    token.row_id = f"row-{token_id}"
    mock_row = MagicMock()
    mock_row.to_dict.return_value = row_data or {"field": "value"}
    mock_row.contract = MagicMock()
    mock_row.contract.merge.return_value = mock_row.contract
    token.row_data = mock_row
    return token


def _make_executor() -> tuple[SinkExecutor, MagicMock]:
    recorder = MagicMock()
    state_counter = [0]
    def _begin_state(**kwargs: Any) -> MagicMock:
        state_counter[0] += 1
        state = MagicMock()
        state.state_id = f"state-{state_counter[0]}"
        return state
    recorder.begin_node_state.side_effect = _begin_state
    recorder.allocate_operation_call_index = MagicMock(return_value=0)
    spans = MagicMock()
    spans.sink_span.return_value.__enter__ = MagicMock(return_value=None)
    spans.sink_span.return_value.__exit__ = MagicMock(return_value=False)
    return SinkExecutor(recorder, spans, "run-1"), recorder


def _build_scenario(batch_size: int, diverted_indices: set[int]) -> tuple[list[MagicMock], MagicMock]:
    """Build tokens and a sink mock for a given batch/diversion scenario."""
    tokens = [_make_token(f"t{i}") for i in range(batch_size)]
    diversions = tuple(
        RowDiversion(row_index=i, reason=f"reason-{i}", row_data={"i": i})
        for i in sorted(diverted_indices)
    )
    artifact = ArtifactDescriptor.for_file(path="/tmp/p", content_hash="a" * 64, size_bytes=0)
    sink = MagicMock()
    sink.name = "primary"
    sink.node_id = "node-primary"
    sink.validate_input = False
    sink.declared_required_fields = frozenset()
    sink._on_write_failure = "discard"
    sink._reset_diversion_log = MagicMock()
    sink.write.return_value = SinkWriteResult(artifact=artifact, diversions=diversions)
    return tokens, sink


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_partition_completeness(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Every token gets exactly one outcome: COMPLETED + DIVERTED == total batch."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink = _build_scenario(batch_size, diverted_indices)
    executor, recorder = _make_executor()

    executor.write(
        sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5, sink_name="primary",
        pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
    )

    outcome_calls = recorder.record_token_outcome.call_args_list
    completed_ids = {c.kwargs["token_id"] for c in outcome_calls if c.kwargs["outcome"] == RowOutcome.COMPLETED}
    diverted_ids = {c.kwargs["token_id"] for c in outcome_calls if c.kwargs["outcome"] == RowOutcome.DIVERTED}

    # Partition completeness: every token accounted for
    assert len(completed_ids) + len(diverted_ids) == batch_size
    # Disjoint: no token in both sets
    assert completed_ids & diverted_ids == set()
    # All tokens present
    all_token_ids = {t.token_id for t in tokens}
    assert completed_ids | diverted_ids == all_token_ids


@given(
    batch_size=st.integers(min_value=1, max_value=30),
    diverted_indices_raw=st.lists(st.integers(min_value=0, max_value=29), max_size=30),
)
@settings(max_examples=200)
def test_exactly_once_terminal_state(batch_size: int, diverted_indices_raw: list[int]) -> None:
    """Each token_id appears in exactly one record_token_outcome call."""
    diverted_indices = {i for i in diverted_indices_raw if i < batch_size}
    tokens, sink = _build_scenario(batch_size, diverted_indices)
    executor, recorder = _make_executor()

    executor.write(
        sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
        step_in_pipeline=5, sink_name="primary",
        pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
    )

    outcome_calls = recorder.record_token_outcome.call_args_list
    recorded_token_ids = [c.kwargs["token_id"] for c in outcome_calls]
    # No duplicates
    assert len(recorded_token_ids) == len(set(recorded_token_ids))
    # All input tokens present
    assert set(recorded_token_ids) == {t.token_id for t in tokens}
```

- [ ] **Step 2: Run property tests**

Run: `.venv/bin/python -m pytest tests/property/engine/test_sink_executor_diversion_properties.py -v --hypothesis-show-statistics`
Expected: All PASS with 200 examples each.

- [ ] **Step 3: Commit**

```bash
git add tests/property/engine/test_sink_executor_diversion_properties.py
git commit -m "test(property): add Hypothesis partition-completeness and exactly-once tests for failsink"
```

---

### Task 7: Integration E2E test (B6)

**Files:**
- Create: `tests/integration/plugins/sinks/test_failsink_e2e.py`

- [ ] **Step 1: Write end-to-end integration test**

This test uses `ExecutionGraph.from_plugin_instances()` and real plugin instances (not mocks). It wires a source → transform → ChromaSink (with `on_write_failure=csv_failsink`) → CSVSink failsink.

This test requires real plugin instances wired through the actual graph construction and execution paths. Because integration test fixture patterns vary significantly across the existing `tests/integration/` suite (some use YAML configs, some use direct construction), the implementer must:

1. **Find an existing integration test to use as a template.** Search for a test that uses `ExecutionGraph.from_plugin_instances()` with at least one sink:

   Run: `grep -rn "from_plugin_instances" tests/integration/ --include="*.py" -l`

   Read the first match to understand the fixture pattern (config construction, temp directory setup, landscape recorder initialization, how RunResult is obtained).

2. **Build the test using that pattern.** The test needs:
   - A CSV source with 3 rows, one of which has a metadata field containing a dict (e.g., `{"topic": {"nested": "value"}}`)
   - A ChromaSink using `chromadb.EphemeralClient()` (no persistence needed) with `_on_write_failure = "csv_failsink"`
   - A CSVSink as failsink with `_on_write_failure = "discard"`
   - Both sinks wired through `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()` (or equivalent direct construction that the template test uses)

3. **Assert these concrete outcomes:**

```python
def test_failsink_routes_diverted_row(tmp_path: Path) -> None:
    """Full pipeline: ChromaSink diverts one row to CSV failsink."""
    # ... (fixture setup from template pattern) ...

    # Run pipeline
    result = orchestrator.run(...)  # or however the template executes

    # 1. RunResult counters
    assert result.rows_diverted == 1
    assert result.rows_succeeded == 2

    # 2. Failsink CSV file exists and contains the diverted row
    failsink_path = tmp_path / "failsink" / "chroma_rejects.csv"
    assert failsink_path.exists()
    import csv
    with open(failsink_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert "__diversion_reason" in rows[0]

    # 3. Audit trail: query token_outcomes from the landscape recorder
    # (exact query method depends on what the template test uses)
    # Assert: one DIVERTED outcome, two COMPLETED outcomes

    # 4. DAG has __failsink__ edge
    edges = graph.get_edges()
    failsink_edges = [e for e in edges if e.label == "__failsink__"]
    assert len(failsink_edges) == 1
```

The exact fixture wiring, import paths, and landscape query methods must be adapted from whichever integration test template is found in step 1. Do NOT invent fixture patterns — copy from an existing working test.

- [ ] **Step 2: Run integration test**

Run: `.venv/bin/python -m pytest tests/integration/plugins/sinks/test_failsink_e2e.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/plugins/sinks/test_failsink_e2e.py
git commit -m "test(integration): add end-to-end failsink routing test"
```

---

### Task 8: Additional regression tests (W4, W5)

**Files:**
- Modify: `tests/unit/engine/test_sink_executor_diversion.py`

- [ ] **Step 1: Add failsink write failure cleanup test (W4)**

```python
class TestFailsinkCleanup:
    def test_failsink_write_failure_completes_failsink_states_as_failed(self) -> None:
        """When failsink.write() raises, opened failsink node_states must be FAILED.
        Batch: 1 token, 1 diversion. Expect exactly 1 FAILED state at failsink node."""
        executor, recorder = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError):
            executor.write(
                sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
                step_in_pipeline=5, sink_name="primary",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                failsink=failsink, failsink_name="csv_failsink",
            )
        # Exactly 1 FAILED state (the failsink state for t0), 0 COMPLETED states
        complete_calls = recorder.complete_node_state.call_args_list
        failed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED]
        completed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(failed_calls) == 1
        assert len(completed_calls) == 0  # t0 was diverted, no primary state
        # Verify the FAILED state was opened at the failsink node, not the primary
        begin_calls = recorder.begin_node_state.call_args_list
        assert len(begin_calls) == 1
        assert begin_calls[0].kwargs["node_id"] == failsink.node_id

    def test_failsink_failure_does_not_affect_primary_states(self) -> None:
        """Primary COMPLETED states must remain intact when failsink fails.
        Batch: 2 tokens, 1 diversion at index 1.
        Expect: t0 COMPLETED at primary, t1 FAILED at failsink."""
        executor, recorder = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.write.side_effect = OSError("disk full")
        tokens = [_make_token("t0"), _make_token("t1")]
        with pytest.raises(OSError):
            executor.write(
                sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
                step_in_pipeline=5, sink_name="primary",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                failsink=failsink, failsink_name="csv_failsink",
            )
        complete_calls = recorder.complete_node_state.call_args_list
        # t0: COMPLETED at primary node_id
        completed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.COMPLETED]
        assert len(completed_calls) == 1
        # t1: FAILED at failsink node_id (failsink write crashed)
        failed_calls = [c for c in complete_calls if c.kwargs.get("status") == NodeStateStatus.FAILED]
        assert len(failed_calls) == 1
        # Verify node_id isolation: primary states at primary node, failsink states at failsink node
        begin_calls = recorder.begin_node_state.call_args_list
        primary_begins = [c for c in begin_calls if c.kwargs.get("node_id") == sink.node_id]
        failsink_begins = [c for c in begin_calls if c.kwargs.get("node_id") == failsink.node_id]
        assert len(primary_begins) == 1  # t0 at primary
        assert len(failsink_begins) == 1  # t1 at failsink

    def test_failsink_flush_failure_crashes(self) -> None:
        """If failsink.flush() raises, crash after completing failsink states as FAILED."""
        executor, recorder = _make_executor()
        diversions = (RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        failsink.flush.side_effect = OSError("disk full")
        tokens = [_make_token("t0")]
        with pytest.raises(OSError, match="disk full"):
            executor.write(
                sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
                step_in_pipeline=5, sink_name="primary",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                failsink=failsink, failsink_name="csv_failsink",
            )
```

- [ ] **Step 2: Add non-contiguous diversions test (keyed by token_id, not call order)**

```python
    def test_non_contiguous_diversions(self) -> None:
        """Rows 0 and 2 diverted, row 1 primary. Outcomes correctly partitioned.
        Uses token_id keying, not call ordering — the executor may process
        primary tokens before diverted tokens."""
        executor, recorder = _make_executor()
        diversions = (
            RowDiversion(row_index=0, reason="bad", row_data={"x": 1}),
            RowDiversion(row_index=2, reason="bad", row_data={"x": 3}),
        )
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1"), _make_token("t2")]
        executor.write(
            sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5, sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )
        outcome_calls = recorder.record_token_outcome.call_args_list
        outcomes_by_token = {c.kwargs["token_id"]: c.kwargs["outcome"] for c in outcome_calls}
        assert outcomes_by_token["t0"] == RowOutcome.DIVERTED
        assert outcomes_by_token["t1"] == RowOutcome.COMPLETED
        assert outcomes_by_token["t2"] == RowOutcome.DIVERTED
```

- [ ] **Step 3: Add empty batch test**

```python
class TestEmptyBatch:
    def test_empty_batch_with_failsink_configured(self) -> None:
        """Empty token list with failsink configured — no-op, no crash."""
        executor, recorder = _make_executor()
        sink = _make_sink(on_write_failure="csv_failsink")
        failsink = _make_failsink()
        result = executor.write(
            sink=sink, tokens=[], ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5, sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink, failsink_name="csv_failsink",
        )
        assert result == (None, 0)  # No artifact, 0 diversions
        failsink.write.assert_not_called()
        recorder.record_token_outcome.assert_not_called()
```

- [ ] **Step 4: Add on_token_written tests for both discard and failsink modes**

```python
class TestOnTokenWrittenWithDiversions:
    def test_on_token_written_not_called_for_diverted_discard_mode(self) -> None:
        """on_token_written must not fire for diverted tokens (discard mode)."""
        executor, recorder = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="discard")
        tokens = [_make_token("t0"), _make_token("t1")]
        callback = MagicMock()
        executor.write(
            sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5, sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            on_token_written=callback,
        )
        # callback called once for t0 (primary), NOT for t1 (diverted)
        assert callback.call_count == 1
        assert callback.call_args[0][0].token_id == "t0"

    def test_on_token_written_not_called_for_diverted_failsink_mode(self) -> None:
        """on_token_written must not fire for diverted tokens (failsink mode)."""
        executor, recorder = _make_executor()
        diversions = (RowDiversion(row_index=1, reason="bad", row_data={"x": 1}),)
        sink = _make_sink(diversions=diversions, on_write_failure="csv_failsink")
        failsink = _make_failsink()
        tokens = [_make_token("t0"), _make_token("t1")]
        callback = MagicMock()
        executor.write(
            sink=sink, tokens=tokens, ctx=MagicMock(run_id="run-1"),
            step_in_pipeline=5, sink_name="primary",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            failsink=failsink, failsink_name="csv_failsink",
            on_token_written=callback,
        )
        # callback called once for t0 (primary), NOT for t1 (diverted to failsink)
        assert callback.call_count == 1
        assert callback.call_args[0][0].token_id == "t0"
```

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_sink_executor_diversion.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/engine/test_sink_executor_diversion.py
git commit -m "test: add failsink cleanup, non-contiguous diversions, and on_token_written tests"
```

---

### Task 9: Full suite verification + CI checks

- [ ] **Step 1: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --tb=short -q`
Expected: All PASS.

- [ ] **Step 2: Run mypy**

Run: `.venv/bin/python -m mypy src/`
Expected: PASS (or only pre-existing issues).

- [ ] **Step 3: Run ruff**

Run: `.venv/bin/python -m ruff check src/`
Expected: PASS.

- [ ] **Step 4: Run tier model enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS.

- [ ] **Step 5: Run config contracts check**

Run: `.venv/bin/python -m scripts.check_contracts`
Expected: PASS.

- [ ] **Step 6: Run freeze guard enforcement**

Run: `.venv/bin/python scripts/cicd/enforce_freeze_guards.py`
Expected: PASS — `RowDiversion` uses `freeze_fields()` correctly.

- [ ] **Step 7: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: address lint/type issues from failsink Part 2"
```

- [ ] **Step 8: Close the filigree issue**

```bash
filigree close elspeth-ad6dc0f117 --reason="Implemented: failsink pattern with _divert_row(), SinkWriteResult, DIVERTED outcome, config validation, DAG edges, ChromaSink migration"
```
