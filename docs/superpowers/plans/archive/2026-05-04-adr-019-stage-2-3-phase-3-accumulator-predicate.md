# ADR-019 Stage 2/3 — Phase 3: Accumulator + Predicate + Resume Aggregation + Behaviour Changes

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this phase task-by-task.
>
> **CRITICAL — atomic merge:** This phase is part of a five-phase plan ([overview](2026-05-04-adr-019-stage-2-3-overview.md)). Phase 3 ships TWO operator-visible behaviour changes (discard-mode `RunStatus` flip + GATE_ROUTED/ON_ERROR_ROUTED counter changes) and the bifurcated-predicate simplification — each is non-trivially observable. Phase 4 (cross-table invariants) and Phase 5 (test triage) follow in the same PR.

**Goal:** Ship three coupled changes that ADR-019 § Counter derivation contract (round-4 amendment) requires to land together:

1. **Accumulator behaviour changes** in `outcomes.py::accumulate_row_outcomes` — `(SUCCESS, GATE_ROUTED)` increments BOTH `rows_succeeded` AND `rows_routed_success`; `(FAILURE, ON_ERROR_ROUTED)` increments BOTH `rows_failed` AND `rows_routed_failure`. The accumulator pattern-matches on `(outcome, path)` instead of `RowOutcome`.
2. **Predicate simplification** in `RunResult.__post_init__` and `derive_terminal_run_status` — the bifurcated `success_indicator = rows_succeeded > 0 OR rows_routed_success > 0` becomes `rows_succeeded > 0`; symmetric for the failure_indicator.
3. **Resume aggregation** in `core.py::_derive_resume_terminal_status_from_audit` — pattern-matches on `(outcome, path)` and applies the same counter increments as the accumulator.

The discard-mode flip from Phase 2 is **already landed** at the producer; the accumulator change in this phase is what makes `RunStatus` flip operator-visibly because `(FAILURE, SINK_DISCARDED)` now bumps `rows_failed`.

**TDD discipline:** the operator-visible discard-mode test MUST be RED before the predicate change lands and GREEN after. Same for the GATE_ROUTED counter change. These are the load-bearing behavioural assertions; without them, the phase ships invisibly broken.

**Files touched in this phase:**

- **Create: `tests/integration/_helpers.py`** (NEW — Task 3.0; canonical pipeline-builder factories + `run_pipeline` runner used by every Phase 3 RED test, wired through `instantiate_plugins_from_config()` and `ExecutionGraph.from_plugin_instances()` per CLAUDE.md test-path integrity)
- **Create: `tests/integration/_adr019_test_plugins.py`** (NEW — Task 3.0; config-dict-compatible adapter plugins for the canonical helpers)
- Modify: `src/elspeth/engine/orchestrator/outcomes.py:235-307` (accumulator)
- Modify: `src/elspeth/engine/orchestrator/core.py:450-533, 2277-2371` (resume aggregation + live source-quarantine counter/pending-outcome path)
- Modify: `src/elspeth/engine/executors/sink.py:341, 949-1019` (`SinkExecutor.write()` return shape becomes `DiversionCounts`)
- Modify: `src/elspeth/contracts/run_result.py:60-152, 180-216` (L0 predicate + `derive_terminal_run_status`)
- **Modify: `src/elspeth/web/execution/schemas.py:138-160, 197-285` (L3 Pydantic predicate mirror — `_validate_row_decomposition` and `_check_status_row_count_invariant`).** Discovered 2026-05-05: the Pydantic schema duplicates the L0 predicate logic. Without this update, `/api/runs/{rid}` returns HTTP 500 for any valid run with gate-MOVE or transform-on-error routing because the post-Phase-3 counter doubling makes `rows_succeeded + rows_routed_success` exceed `rows_processed` in `_validate_row_decomposition`'s sum-disjoint formula.
- **Modify: `src/elspeth/web/sessions/protocol.py`, `src/elspeth/web/sessions/service.py:672-679`, `src/elspeth/web/execution/routes.py:669, 686`, `src/elspeth/web/execution/service.py:614-633`, and `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx:19-24` (terminal-status fallout).** The shared session protocol already defines `completed_with_failures` and `empty` as terminal, but WebSocket replay/idle checks, the session write guard, cancellation idempotency, and the sidebar active-run predicate still have old hardcoded status subsets.
- Test (RED-first): `tests/integration/test_adr_019_discard_mode_flip.py` (NEW)
- Test (RED-first): `tests/integration/test_adr_019_counter_changes.py` (NEW)
- Test: `tests/integration/test_adr_019_helpers.py` (NEW — Task 3.0 helper integrity)
- Test (RED-first): `tests/integration/test_adr_019_resume_counter_parity.py` (NEW — Task 3.6 live-vs-resume predicate counter parity)
- Test (unit/integration, direct `SinkExecutor.write()` return-shape fallout): `tests/unit/engine/test_sink_executor_diversion.py`, `tests/unit/engine/test_executors.py`, `tests/integration/plugins/sinks/test_durability.py`
- Test (unit, extend): `tests/unit/web/execution/test_schemas.py::TestCompletedDataDecomposition` (existing — extend with post-Phase-3 Pydantic predicate-mirror acceptance cases)
- Test (unit): `tests/unit/engine/orchestrator/test_outcomes.py` (existing — extend with (outcome, path) match assertions)
- Test (unit): `tests/unit/contracts/test_run_result.py` (existing — flip predicate assertions and add ADR-019 boundary cases)
- Test (unit): `tests/unit/web/execution/test_schemas.py` (existing — Pydantic predicate-mirror tests)
- Test (unit): `tests/unit/web/execution/test_websocket.py` or `tests/unit/web/execution/test_routes.py` (extend with reconnect/idle terminal replay for `completed_with_failures` and `empty`)
- Test (unit): `tests/unit/web/sessions/test_service.py` (extend terminal write/read invariants for `completed_with_failures` and `empty`)
- Test (unit): `tests/unit/web/execution/test_service.py` (extend cancel idempotency for widened terminal set)
- Test (frontend): `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx` or existing nearest frontend test (assert terminal `completed_with_failures` / `empty` does not keep the active-run marker visible)

**Background reading:** ADR-019 § Counter derivation contract (lines 271-326). Behavior Change Notice (lines 386-402). The accumulator's existing `RowOutcome` match at `outcomes.py:235-307` is the source-of-truth for what each path must do today — verify against it as you flip.

---

## The behaviour-change semantics (what flips when this phase lands)

### Change 1: `(SUCCESS, GATE_ROUTED)` becomes a `rows_succeeded` increment

**Before:**
```python
elif result.outcome == RowOutcome.ROUTED:
    counters.rows_routed_success += 1   # only this counter increments
    counters.routed_destinations[sink_name] += 1
    _route_to_sink(...)
```

**After:**
```python
elif (result.outcome, result.path) == (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED):
    counters.rows_succeeded += 1        # NEW: also bumps rows_succeeded
    counters.rows_routed_success += 1
    counters.routed_destinations[sink_name] += 1
    _route_to_sink(...)
```

**Operator visibility:** dashboards reading `rows_succeeded` see higher numbers for runs containing gate MOVE routing. Public field name is preserved.

### Change 2: FAILURE paths make `rows_failed` exhaustive

Symmetric to Change 1 — the `rows_failed` counter now reflects every `TerminalOutcome.FAILURE` lifecycle row that is a predicate input, including transform `on_error` routings and source quarantines. This follows ADR-019's normative mapping for `QUARANTINED` (`rows_quarantined`, `rows_failed`) and makes `failure_indicator = rows_failed > 0` exhaustive without needing `OR rows_routed_failure > 0` or `OR rows_quarantined > 0`.

**Load-bearing live-source nuance:** source quarantine rows do NOT flow through
`accumulate_row_outcomes`; the live path is
`Orchestrator._handle_quarantine_row` in
`src/elspeth/engine/orchestrator/core.py:2277-2371`. Phase 3 must update that
site directly so it increments both `rows_quarantined` and `rows_failed` before
appending the `(FAILURE, QUARANTINED_AT_SOURCE)` `PendingOutcome`. Otherwise the
new L0/L3 subset guard (`rows_quarantined <= rows_failed`) rejects real
source-quarantine runs.

### Change 3: discard-mode `DIVERTED` increments `rows_failed`

This is the operator-visible `RunStatus` flip. A pipeline with a discard sink and otherwise clean rows now reports `RunStatus.COMPLETED_WITH_FAILURES` instead of `RunStatus.COMPLETED`.

The accumulator path is `(FAILURE, SINK_DISCARDED)`. Counter behaviour:

```python
elif (result.outcome, result.path) == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
    counters.rows_failed += 1
    counters.rows_diverted += 1   # structural counter — unchanged from ADR-018
```

(Note: today's accumulator at `outcomes.py:279` raises `OrchestrationInvariantError` for `RowOutcome.DIVERTED` because diversions are recorded at sink-write time inside `SinkExecutor.write()` and counted at the orchestrator's sink-call site (`core.py:2199`: `loop_ctx.counters.rows_diverted += total_diversions`), not in the row-processing loop. The `(FAILURE, SINK_DISCARDED)` and `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` pairs similarly should NOT appear in `accumulate_row_outcomes` — they're emitted by the sink executor and counted at the orchestrator's sink-call site. The operator-visible discard-mode counter bump (`rows_failed`) lands at that orchestrator site too — see Task 3.3 Step 4 below for the SinkExecutor return-shape change and the orchestrator counter-update site.)

### Change 4: predicate drops the bifurcated OR

```python
# OLD:
success_indicator = self.rows_succeeded > 0 or self.rows_routed_success > 0
failure_indicator = (
    self.rows_failed > 0 or self.rows_quarantined > 0
    or self.rows_coalesce_failed > 0 or self.rows_routed_failure > 0
)

# NEW:
success_indicator = self.rows_succeeded > 0
failure_indicator = (
    self.rows_failed > 0 or self.rows_coalesce_failed > 0
)
```

Both bifurcated OR clauses go away because Changes 1 and 2 above make `rows_succeeded` and `rows_failed` exhaustive for terminal success/failure lifecycle rows. `rows_quarantined` remains as a subset counter for quarantine reporting, but it is no longer a separate predicate input. `rows_coalesce_failed` remains separate because it is a coalesce/run-level failure signal rather than a token `TerminalOutcome.FAILURE` row.

---

## Tasks

### Task 3.0: Create canonical integration-test helpers

**Why this task exists:** Tasks 3.1, 3.2, and 3.5 all reference three pipeline-builder helpers (`build_test_pipeline_with_discard_sink`, `build_test_pipeline_with_gate_route`, `build_test_pipeline_with_on_error_route`) and a `run_pipeline` orchestrator-runner. **None of these exist in the codebase today** (verified 2026-05-05: `grep -rn "build_test_pipeline_with_" src/elspeth/testing/ tests/` returns empty). They must be created in a single canonical location BEFORE any of the Phase 3 RED tests can be written, otherwise the tests are non-runnable from the moment they land.

**Canonical location:** `tests/integration/_helpers.py`. NOT `src/elspeth/testing/` — that pack hosts ChaosEngine fixtures shipped to ELSPETH consumers, not integration-test pipeline builders. NOT `tests/conftest.py` — that's pytest fixture machinery, not pipeline-config factories.

**CLAUDE.md test-path integrity (load-bearing):** the helpers MUST exercise both production assembly layers: `instantiate_plugins_from_config()` for settings/YAML-to-plugin construction and `ExecutionGraph.from_plugin_instances()` for graph construction. Do not construct plugin instances directly in these integration helpers. Direct construction is acceptable for isolated unit tests, but these Phase 3 tests assert operator-visible runtime behavior and must pass through the same path as CLI/web execution.

**Files:**
- Create: `tests/integration/_helpers.py`
- Create: `tests/integration/_adr019_test_plugins.py`
- Test: `tests/integration/test_adr_019_helpers.py`

**Step 1: Add `DivertingSink` to `tests/fixtures/plugins.py`**

`(FAILURE, SINK_DISCARDED)` is emitted by `SinkExecutor` when a sink write fails
AND no failsink is configured for that primary sink (the `failsink=None` branch at
`src/elspeth/engine/executors/sink.py:977`). The trigger is a sink that calls
`self._divert_row()` — the `BaseSink` hook at
`src/elspeth/plugins/infrastructure/base.py:894`. No such test fixture currently
exists; add one alongside the existing test plugins.

```python
# In tests/fixtures/plugins.py — add after FailingSink (around line 130)
# Also add to the existing imports block at the top of plugins.py:
#   from elspeth.contracts.diversion import RowDiversion

class DivertingSink(_TestSinkBase):
    """Sink that diverts the first ``divert_count`` rows via direct RowDiversion construction.

    When no failsink is wired to this sink, the executor takes the discard
    branch (``src/elspeth/engine/executors/sink.py:977``) and records each
    diverted token as (FAILURE, SINK_DISCARDED) with sink_name='__discard__'.

    Use this to exercise ADR-019 § Sub-decision 5: discard-mode ``rows_failed``
    increment (Phase 3 accumulator change).

    Args:
        name: Sink name (defaults to 'diverting_sink').
        divert_count: How many rows (from the start of each write batch) to
            divert. Remaining rows are written normally. A value of ``None``
            diverts ALL rows.
    """

    name = "diverting_sink"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        options = dict(config or {})
        self.name = str(options.get("name", self.name))
        self.results: list[dict[str, Any]] = []
        self._divert_count = options.get("divert_count")
        self._artifact_counter = 0

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        limit = self._divert_count if self._divert_count is not None else len(rows)
        diversions: tuple[RowDiversion, ...] = tuple(
            RowDiversion(
                row_index=idx,
                reason="diverting_sink: forced divert for ADR-019 test",
                row_data=dict(row),
            )
            for idx, row in enumerate(rows)
            if idx < limit
        )
        primary_rows = [row for idx, row in enumerate(rows) if idx >= limit]
        self.results.extend(primary_rows)
        self._artifact_counter += 1
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(rows)),
                content_hash=f"hash_{self._artifact_counter}",
            ),
            diversions=diversions,
        )

    def close(self) -> None:
        pass
```

**Diversion handoff contract (verified):** `_TestSinkBase` (at
`tests/fixtures/base_classes.py:135`) does NOT provide `_divert_row()` or
`_get_diversions()` — those are `BaseSink` methods for production plugins only.
The canonical test-fixture pattern constructs `RowDiversion` records directly
and passes them as a `tuple[RowDiversion, ...]` via
`SinkWriteResult(diversions=...)`. The canonical reference is
`DivertSecondRowSink` at
`tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py:34`.
The executor reads `write_result.diversions` from the returned `SinkWriteResult`
at `src/elspeth/engine/executors/sink.py:548` — it does NOT reach into any
sink-side accumulator.

**Step 2: Create config-dict-compatible test plugin registration**

`instantiate_plugins_from_config()` uses the shared `PluginManager` and passes a
single `dict(options)` positional argument into plugin constructors. The existing
fixture classes (`ListSource(data, ...)`, `CollectSink(name, ...)`,
`ConditionalErrorTransform(*, ...)`) are not constructor-compatible with that
path, and `PluginManager` discovers by class-level `name`. Do not register those
classes directly.

Create `tests/integration/_adr019_test_plugins.py` with thin adapter classes and
a pytest install helper. The adapter classes keep the existing fixture behavior
but expose the production constructor shape:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from elspeth.plugins.infrastructure.discovery import create_dynamic_hookimpl
from elspeth.plugins.infrastructure.manager import PluginManager
from tests.fixtures.plugins import (
    CollectSink,
    ConditionalErrorTransform,
    DivertingSink,
    ListSource,
)


class ADR019ListSource(ListSource):
    name = "list_source"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        rows = options.pop("rows")
        super().__init__(
            list(rows),
            name=str(options.pop("name", self.name)),
            on_success=str(options.pop("on_success", "default")),
        )


class ADR019CollectSink(CollectSink):
    name = "collect_sink"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        super().__init__(
            str(options.pop("name", self.name)),
            node_id=options.pop("node_id", None),
        )


class ADR019ConditionalErrorTransform(ConditionalErrorTransform):
    name = "conditional_error"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        options = dict(config or {})
        super().__init__(
            name=options.pop("name", None),
            input_connection=options.pop("input", None),
            on_success=options.pop("on_success", None),
            on_error=options.pop("on_error", None),
        )


class ADR019DivertingSink(DivertingSink):
    name = "diverting_sink"


def make_adr019_plugin_manager() -> PluginManager:
    manager = PluginManager()
    manager.register_builtin_plugins()
    manager.register(create_dynamic_hookimpl([ADR019ListSource], "elspeth_get_source"))
    manager.register(create_dynamic_hookimpl([ADR019ConditionalErrorTransform], "elspeth_get_transforms"))
    manager.register(create_dynamic_hookimpl([ADR019CollectSink, ADR019DivertingSink], "elspeth_get_sinks"))
    return manager


def install_adr019_test_plugin_manager(monkeypatch: pytest.MonkeyPatch) -> PluginManager:
    manager = make_adr019_plugin_manager()
    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: manager,
    )
    return manager
```

Keep the `pytest` import because `install_adr019_test_plugin_manager` exposes the
runtime `pytest.MonkeyPatch` type in its signature.

**Step 3: Create the helper module**

```python
"""Integration-test helpers for ADR-019 behaviour-change verification.

Three pipeline factories that produce minimal end-to-end settings/config objects
plus a single ``run_pipeline()`` runner.

CLAUDE.md test-path integrity: these are integration helpers, so they MUST use
``instantiate_plugins_from_config()`` to construct plugin instances from the same
settings shape the CLI/web paths use, and then build the graph through
``ExecutionGraph.from_plugin_instances()``.
Do not manually instantiate source/transform/sink classes in these helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import ElspethSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.contracts.run_result import RunResult
from elspeth.core.landscape.database import LandscapeDB  # src/elspeth/core/landscape/database.py
from elspeth.core.payload_store import FilesystemPayloadStore  # src/elspeth/core/payload_store.py
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.orchestrator import (
    Orchestrator,       # src/elspeth/engine/orchestrator/core.py:299
    PipelineConfig,     # src/elspeth/engine/orchestrator/types.py:60
)
from tests.integration._adr019_test_plugins import install_adr019_test_plugin_manager


def make_settings_yaml_for_test_plugins(
    *,
    source_plugin: str,
    source_config: dict[str, object],
    sinks: dict[str, dict[str, object]],
    transforms: list[dict[str, object]] | None = None,
    gates: list[dict[str, object]] | None = None,
) -> str:
    """Serialize the minimal production settings shape used by the helpers."""
    source_options = dict(source_config)
    source_on_success = str(source_options.pop("on_success"))
    sink_payload: dict[str, dict[str, object]] = {}
    for sink_name, sink_spec in sinks.items():
        sink_payload[sink_name] = {
            "plugin": sink_spec["plugin"],
            "on_write_failure": sink_spec.get("on_write_failure", "discard"),
            "options": sink_spec.get("options", sink_spec.get("config", {})),
        }
    payload: dict[str, object] = {
        "source": {
            "plugin": source_plugin,
            "on_success": source_on_success,
            "options": source_options,
        },
        "sinks": sink_payload,
    }
    if transforms:
        payload["transforms"] = transforms
    if gates:
        payload["gates"] = gates
    return yaml.safe_dump(payload, sort_keys=False)


def _make_db_and_store(tmp_path: Path) -> tuple[LandscapeDB, FilesystemPayloadStore]:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    store = FilesystemPayloadStore(tmp_path / "payloads")
    return db, store


def _pipeline_from_settings(
    settings: ElspethSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Build a pipeline through the production configuration path."""
    db, store = _make_db_and_store(tmp_path)
    install_adr019_test_plugin_manager(monkeypatch)
    bundle = instantiate_plugins_from_config(settings)
    graph = ExecutionGraph.from_plugin_instances(
        source=bundle.source,
        source_settings=bundle.source_settings,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        coalesce_settings=(list(settings.coalesce) if settings.coalesce else None),
    )
    config = assemble_and_validate_pipeline_config(
        source=bundle.source,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    return config, graph, db, store


def _yaml_for_discard_sink(rows: list[dict[str, object]], discard_row_count: int) -> str:
    """Return settings YAML for source -> DivertingSink -> CollectSink.

    The implementation step must register these fixture plugin names with the
    test plugin registry if current HEAD does not already expose them:
    ``list_source``, ``collect_sink``, and ``diverting_sink``. The RED test in
    Task 3.0 must assert that ``instantiate_plugins_from_config(settings)``
    succeeds for this YAML before any accumulator assertions run.
    """
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        sinks={
            "default": {
                "plugin": "diverting_sink",
                "config": {"divert_count": discard_row_count},
            },
        },
    )


def _yaml_for_gate_route(rows: list[dict[str, object]]) -> str:
    """Return settings YAML for a MOVE gate route plus default sink path."""
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        gates=[
            {
                "name": "gate",
                "input": "default",
                "condition": "row['route'] == 'move'",
                "routes": {"true": "routed", "false": "primary"},
            }
        ],
        sinks={
            "routed": {"plugin": "collect_sink", "config": {}},
            "primary": {"plugin": "collect_sink", "config": {}},
        },
    )


def _yaml_for_on_error_route(rows: list[dict[str, object]]) -> str:
    """Return settings YAML for ConditionalErrorTransform on_error routing."""
    return make_settings_yaml_for_test_plugins(
        source_plugin="list_source",
        source_config={"rows": rows, "on_success": "default"},
        transforms=[
            {
                "name": "maybe_fail",
                "plugin": "conditional_error",
                "input": "default",
                "on_success": "primary",
                "on_error": "error_sink",
            }
        ],
        sinks={
            "primary": {"plugin": "collect_sink", "config": {}},
            "error_sink": {"plugin": "collect_sink", "config": {}},
        },
    )


def build_test_pipeline_with_discard_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    success_row_count: int,
    discard_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline where ``discard_row_count`` rows land as (FAILURE, SINK_DISCARDED).

    Mechanism: ``DivertingSink.write()`` constructs ``RowDiversion`` records
    directly for the first ``discard_row_count`` rows and returns them via
    ``SinkWriteResult(diversions=...)``. Because no failsink is wired, the
    executor takes the discard branch (``src/elspeth/engine/executors/sink.py:977``)
    and records each as (FAILURE, SINK_DISCARDED) with
    sink_name='__discard__'.

    The ``success_row_count`` rows are accepted by ``DivertingSink.write()`` as
    ordinary successful sink writes, so their provisional
    (SUCCESS, DEFAULT_FLOW) outcomes remain after sink diversion reconciliation.

    NOTE: (FAILURE, SINK_DISCARDED) is a SINK-SIDE event, not a transform
    on_error event. The transform here succeeds for all rows; the diversion
    happens when DivertingSink.write() returns RowDiversion records in the
    SinkWriteResult.
    """
    rows = [
        {"id": i, "fail": False, "expected": "discard"}
        for i in range(discard_row_count)
    ] + [
        {"id": discard_row_count + i, "fail": False, "expected": "success"}
        for i in range(success_row_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_discard_sink(rows, discard_row_count))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_gate_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    routed_row_count: int,
    default_flow_row_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline with a ``GateSettings`` MOVE-route exercising (SUCCESS, GATE_ROUTED).

    Gate condition ``row['route'] == 'move'`` routes ``routed_row_count`` rows to
    the 'routed' sink (MOVE semantics → (SUCCESS, GATE_ROUTED)).
    ``default_flow_row_count`` rows fall through to the 'primary' sink as
    (SUCCESS, DEFAULT_FLOW).

    ``GateSettings``: src/elspeth/core/config.py:476
    Routing semantics: condition=True → routes["true"] destination (MOVE to named sink).
    """
    rows = [
        {"id": i, "route": "move"} for i in range(routed_row_count)
    ] + [
        {"id": routed_row_count + i, "route": "default"}
        for i in range(default_flow_row_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_gate_route(rows))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def build_test_pipeline_with_on_error_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_error_routed_count: int,
    success_count: int,
) -> tuple[PipelineConfig, ExecutionGraph, LandscapeDB, FilesystemPayloadStore]:
    """Pipeline where ``on_error_routed_count`` rows land as (FAILURE, ON_ERROR_ROUTED).

    ``ConditionalErrorTransform`` (tests/fixtures/plugins.py:213) returns
    ``TransformResult.error(...)`` when ``row['fail']`` is truthy.  The transform's
    ``on_error`` routes to the 'error_sink'; ``on_success`` routes to 'primary'.

    ``success_count`` rows have ``fail=False`` → (SUCCESS, DEFAULT_FLOW).
    ``on_error_routed_count`` rows have ``fail=True`` → (FAILURE, ON_ERROR_ROUTED).
    """
    rows = [
        {"id": i, "fail": True} for i in range(on_error_routed_count)
    ] + [
        {"id": on_error_routed_count + i, "fail": False}
        for i in range(success_count)
    ]
    settings = load_settings_from_yaml_string(_yaml_for_on_error_route(rows))
    return _pipeline_from_settings(settings, tmp_path, monkeypatch)


def run_pipeline(
    config: PipelineConfig,
    graph: ExecutionGraph,
    db: LandscapeDB,
    store: FilesystemPayloadStore,
) -> RunResult:
    """Run the pipeline through the production code path.

    Canonical pattern from tests/integration/audit/test_recorder_routing_events.py:413
    and tests/fixtures/pipeline.py:98-99:
        Orchestrator(db).run(config, graph=graph, payload_store=store)

    ``Orchestrator.__init__``: src/elspeth/engine/orchestrator/core.py:315
    ``Orchestrator.run``: src/elspeth/engine/orchestrator/core.py:1478
    The graph is already built by ``_pipeline_from_settings`` through
    ExecutionGraph.from_plugin_instances() after instantiate_plugins_from_config().
    """
    return Orchestrator(db).run(
        config,
        graph=graph,
        payload_store=store,
    )
```

Every test that calls one of these helpers must pass the pytest `monkeypatch`
fixture through. This is intentional: `instantiate_plugins_from_config()` has no
manager parameter, so the production lookup path must be patched before the
helper constructs plugins. Do not hide this as global module state in
`tests/integration/_helpers.py`; it makes helper ordering load-bearing.

**Step 4: Verify the helper module imports and production instantiation works**

```bash
.venv/bin/python -c "
from tests.integration._helpers import (
    build_test_pipeline_with_discard_sink,
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    run_pipeline,
)
print('OK')
"
```

Expected: `OK`. (`tests/` must be on `sys.path`; `pytest` ensures this via `pyproject.toml`.)

Add a RED-first helper test that loads each generated settings object and asserts
`instantiate_plugins_from_config(settings)` and `ExecutionGraph.from_plugin_instances(...)`
succeed after `install_adr019_test_plugin_manager(monkeypatch)` runs:

```python
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.integration._helpers import (
    build_test_pipeline_with_discard_sink,
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
)

@pytest.mark.parametrize(
    "builder,args",
    [
        (build_test_pipeline_with_discard_sink, {"success_row_count": 2, "discard_row_count": 1}),
        (build_test_pipeline_with_gate_route, {"routed_row_count": 2, "default_flow_row_count": 1}),
        (build_test_pipeline_with_on_error_route, {"on_error_routed_count": 2, "success_count": 1}),
    ],
)
def test_adr019_helpers_build_through_production_instantiation(
    builder: Callable[..., object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: dict[str, int],
) -> None:
    builder(tmp_path, monkeypatch, **args)
```

If this test fails because a fixture plugin name is absent from the test
registry, fix `_adr019_test_plugins.py`. If it fails with a graph validation
error, fix the generated YAML connection names. Do not fall back to direct
construction in `tests/integration/_helpers.py`.

**Definition of Done:**
- [ ] `DivertingSink` added to `tests/fixtures/plugins.py` using direct `RowDiversion` construction and `SinkWriteResult(diversions=...)` return — canonical pattern from `tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py:34`; constructor accepts `dict(options)` and uses `divert_count`
- [ ] `tests/integration/_adr019_test_plugins.py` registers config-dict-compatible `list_source`, `collect_sink`, `conditional_error`, and `diverting_sink` adapters through `PluginManager.register(...)`
- [ ] `tests/integration/_helpers.py` created with all four helper functions; pipeline builders accept and use pytest `monkeypatch`
- [ ] All three pipeline builders call `instantiate_plugins_from_config(settings)`, then `ExecutionGraph.from_plugin_instances(...)`, then `assemble_and_validate_pipeline_config(...)` per CLAUDE.md test-path integrity
- [ ] Fixture plugins needed by the helpers are registered with the test plugin registry; helper tests prove `instantiate_plugins_from_config(settings)` and graph construction succeed for all three settings shapes
- [ ] Generated helper YAML has coherent connection names (`source.on_success` matches the downstream `input` for gate/transform cases)
- [ ] `run_pipeline` calls `Orchestrator(db).run(config, graph=..., payload_store=...)` — matching the canonical integration-test pattern
- [ ] `from tests.integration._helpers import ...` works for all four names
- [ ] mypy clean on the helper module and the `DivertingSink` addition

---

### Task 3.1: Write the discard-mode behaviour-change RED test

**Files:**
- Create: `tests/integration/test_adr_019_discard_mode_flip.py`

**Step 1: Write the failing test**

```python
"""ADR-019 § Behavior Change Notice: discard-mode DIVERTED flips RunStatus.

A pipeline with a DivertingSink and no failsink wired should report
RunStatus.COMPLETED_WITH_FAILURES (not COMPLETED) because discard-mode is
reclassified as (FAILURE, SINK_DISCARDED) and increments rows_failed.

Mechanism: DivertingSink.write() constructs RowDiversion records directly and
returns them via SinkWriteResult(diversions=...). SinkExecutor takes the
discard branch (failsink=None) and records each as
(FAILURE, SINK_DISCARDED) with sink_name='__discard__'. Phase 3 sink diversion
accounting maps this to
rows_failed += 1 (ADR-019 § Sub-decision 5 / orchestrator site).

This test is RED before Phase 3's accumulator/predicate change lands and
GREEN after. It is the operator-visible assertion that pins the change.
"""

import pytest

from elspeth.contracts.enums import RunStatus
from tests.integration._helpers import build_test_pipeline_with_discard_sink, run_pipeline


@pytest.fixture
def pipeline_with_discard_and_success_sink(tmp_path, monkeypatch):
    """A pipeline where:
    - 3 rows succeed (not diverted by DivertingSink)
    - 2 rows divert to '__discard__' via DivertingSink returning RowDiversion records
    - No quarantines, no upstream transform failures
    """
    return build_test_pipeline_with_discard_sink(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        success_row_count=3,
        discard_row_count=2,
    )


class TestDiscardModeRunStatusFlip:
    def test_discard_with_some_success_yields_completed_with_failures(
        self, pipeline_with_discard_and_success_sink
    ) -> None:
        """ADR-019 § Sub-decision 5: discard-mode is FAILURE, predicate-input."""
        config, graph, db, store = pipeline_with_discard_and_success_sink
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES, (
            f"Expected COMPLETED_WITH_FAILURES (3 success + 2 discard), "
            f"got {result.status}. The discard rows should bump rows_failed "
            f"per ADR-019 § Sub-decision 5."
        )
        assert result.rows_succeeded == 3
        assert result.rows_failed == 2  # NEW: discards now count as failures
        assert result.rows_diverted == 2  # structural counter unchanged

    def test_all_discards_yields_failed(self, tmp_path, monkeypatch) -> None:
        """ADR-019 § Behavior Change Notice: pipelines that ONLY discard yield FAILED."""
        config, graph, db, store = build_test_pipeline_with_discard_sink(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            success_row_count=0,
            discard_row_count=3,
        )
        result = run_pipeline(config, graph, db, store)
        assert result.status == RunStatus.FAILED
        assert result.rows_succeeded == 0
        assert result.rows_failed == 3
```

Add DB-backed audit assertions to the same file; counter/status-only assertions
are not enough because resume and diagnostics read the persisted token outcomes.
Use `RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)`.
Import `DISCARD_SINK_NAME` from `elspeth.contracts.audit` for sentinel checks;
do not compare against a local `"__discard__"` literal. Assert the durable rows
include:

- success rows as `(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, completed=True)`
- discard rows as `(TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED, completed=True, sink_name=DISCARD_SINK_NAME)`
- non-empty `error_hash` on every discard row

The test must fail if discard rows are still persisted as legacy `DIVERTED` or
if the producer path is missing from the audit row.

`build_test_pipeline_with_discard_sink` is the helper created in Task 3.0 at the canonical location `tests/integration/_helpers.py`. It builds a minimal pipeline (``ListSource``, ``DivertingSink``) using the project's fixture plugins — the same path as every other integration test. The ``(config, graph, db, store)`` tuple is unpacked before calling ``run_pipeline``.

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_discard_mode_flip.py -v
```

Expected: BOTH tests fail. The first fails because `result.status == RunStatus.COMPLETED` (under current behaviour). The second fails because the run completes as `COMPLETED` or `EMPTY` rather than `FAILED`.

**Step 3: Confirm the test fixture builds correctly even before the predicate change**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_helpers.py::test_adr019_helpers_build_through_production_instantiation -v
```

Expected: green before and after the predicate change. Helper-build failures are
setup defects and must be fixed in Task 3.0 before using this discard-mode test
as behavioral evidence.

**Definition of Done:**
- [ ] Test file exists and asserts the post-Phase-3 behaviour
- [ ] Both tests fail before Phase 3's accumulator/predicate change lands
- [ ] The four canonical helpers from Task 3.0 are importable from `tests.integration._helpers`

---

### Task 3.2: Write the GATE_ROUTED / ON_ERROR_ROUTED counter-change RED tests

**Files:**
- Create: `tests/integration/test_adr_019_counter_changes.py`

**Step 1: Write the failing tests**

```python
"""ADR-019 § Counter derivation contract: gate-routed/on-error-routed bump
both predicate counters and structural counters.

Before ADR-019: RowOutcome.ROUTED bumps only rows_routed_success;
RowOutcome.ROUTED_ON_ERROR bumps only rows_routed_failure. The predicate
needed `rows_succeeded > 0 OR rows_routed_success > 0` to capture both.

After ADR-019: (SUCCESS, GATE_ROUTED) bumps BOTH rows_succeeded AND
rows_routed_success. (FAILURE, ON_ERROR_ROUTED) bumps BOTH rows_failed AND
rows_routed_failure. The predicate becomes `rows_succeeded > 0` only.
"""

import pytest

from elspeth.contracts.enums import RunStatus
from tests.integration._helpers import (
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    run_pipeline,
)


class TestGateRoutedCounterDoubling:
    def test_gate_routed_bumps_both_succeeded_and_routed_success(
        self, tmp_path, monkeypatch
    ) -> None:
        config, graph, db, store = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            routed_row_count=4,
            default_flow_row_count=0,  # ALL rows get gate-routed
        )
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED, (
            f"With 4 gate-routed rows and 0 failures, expected COMPLETED, got {result.status}"
        )
        # NEW: rows_succeeded reflects the routed rows
        assert result.rows_succeeded == 4
        assert result.rows_routed_success == 4

    def test_on_error_routed_bumps_both_failed_and_routed_failure(
        self, tmp_path, monkeypatch
    ) -> None:
        config, graph, db, store = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            on_error_routed_count=3,
            success_count=2,
        )
        result = run_pipeline(config, graph, db, store)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_failed == 3       # NEW: on-error routes count as failures
        assert result.rows_routed_failure == 3
        assert result.rows_succeeded == 2
```

Add DB-backed audit assertions with
`RecorderFactory(db).query.get_all_token_outcomes_for_run(result.run_id)`:

- gate-routed rows persist as `(TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED, completed=True)` with the routed sink name
- on-error rows persist as `(TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED, completed=True)` with the error sink name and non-empty `error_hash`
- default-flow rows persist as `(TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW, completed=True)`

These assertions are part of the RED/GREEN contract. A run that has correct
counters but wrong durable `(outcome, path)` rows is still broken because resume,
diagnostics, and operator explanations consume the audit rows.

**Step 2: Run RED**

Expected: both tests fail. The gate-routed case fails because under current
code `rows_succeeded == 0` for an all-gate-routed run, even though
`result.status == RunStatus.COMPLETED` via the legacy
`OR rows_routed_success > 0` predicate clause. The on-error case fails because
`rows_failed == 0` while `rows_routed_failure == 3`; status may already be
`COMPLETED_WITH_FAILURES` under the old OR predicate, so the counter assertion is
the load-bearing RED signal.

**Definition of Done:**
- [ ] Test file exists with both counter-doubling assertions
- [ ] Tests fail before Phase 3 lands
- [ ] Tests assert the durable token-outcome `(outcome, path, completed, sink_name, error_hash)` rows, not only `RunResult` counters
- [ ] Fixture helpers exist

---

### Task 3.2a: Add predicate boundary RED tests

**Why this task exists:** Task 3.4 deletes the old bifurcated predicate clauses
(`OR rows_routed_success`, `OR rows_routed_failure`, `OR rows_quarantined`).
The behaviour-change tests cover the main happy paths, but the predicate also
needs explicit boundary coverage for zero-row, all-failure, all-quarantine, and
mixed success/failure shapes. These tests prevent the old subset counters from
quietly remaining predicate inputs after the accumulator starts double-bumping
the base counters, and they reject impossible subset shapes mechanically.

**Files:**
- Modify: `tests/unit/contracts/test_run_result.py`

**Step 1: Add the boundary tests FIRST**

Append this class near the existing `derive_terminal_run_status` tests. Ensure
the file imports `pytest` before adding the `pytest.raises(...)` cases:

```python
class TestADR019PredicateBoundaryCases:
    """ADR-019 predicate boundaries after base counters become exhaustive."""

    def test_zero_row_source_derives_empty(self) -> None:
        """A source that emits no rows remains EMPTY."""
        assert (
            derive_terminal_run_status(
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                rows_coalesce_failed=0,
            )
            == RunStatus.EMPTY
        )
        RunResult(
            run_id="run-empty-boundary",
            status=RunStatus.EMPTY,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
        )

    def test_all_sink_discarded_rows_derive_failed(self) -> None:
        """All rows in (FAILURE, SINK_DISCARDED) have no success path."""
        assert (
            derive_terminal_run_status(
                rows_processed=3,
                rows_succeeded=0,
                rows_failed=3,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=0,
                rows_coalesce_failed=0,
            )
            == RunStatus.FAILED
        )
        RunResult(
            run_id="run-all-discarded",
            status=RunStatus.FAILED,
            rows_processed=3,
            rows_succeeded=0,
            rows_failed=3,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_diverted=3,
        )

    def test_all_quarantined_at_source_derive_failed_from_rows_failed(self) -> None:
        """All source quarantines must be represented by the base failure counter."""
        assert (
            derive_terminal_run_status(
                rows_processed=4,
                rows_succeeded=0,
                rows_failed=4,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=4,
                rows_coalesce_failed=0,
            )
            == RunStatus.FAILED
        )

    def test_quarantine_subset_without_rows_failed_is_rejected(self) -> None:
        """rows_quarantined is impossible unless rows_failed covers it."""
        with pytest.raises(ValueError, match="rows_quarantined"):
            derive_terminal_run_status(
                rows_processed=4,
                rows_succeeded=4,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=4,
                rows_coalesce_failed=0,
            )

        with pytest.raises(ValueError, match="rows_quarantined"):
            RunResult(
                run_id="run-invalid-quarantine-subset",
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=4,
                rows_succeeded=4,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=0,
                rows_quarantined=4,
            )

    def test_on_error_routed_plus_default_flow_uses_rows_failed_not_routed_subset(
        self,
    ) -> None:
        """A mixed DEFAULT_FLOW/ON_ERROR_ROUTED run is mixed only when rows_failed is bumped."""
        assert (
            derive_terminal_run_status(
                rows_processed=5,
                rows_succeeded=2,
                rows_failed=3,
                rows_routed_success=0,
                rows_routed_failure=3,
                rows_quarantined=0,
                rows_coalesce_failed=0,
            )
            == RunStatus.COMPLETED_WITH_FAILURES
        )

        with pytest.raises(ValueError, match="rows_routed_failure"):
            derive_terminal_run_status(
                rows_processed=5,
                rows_succeeded=2,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=3,
                rows_quarantined=0,
                rows_coalesce_failed=0,
            )

        with pytest.raises(ValueError, match="rows_routed_failure"):
            RunResult(
                run_id="run-invalid-routed-failure-subset",
                status=RunStatus.COMPLETED_WITH_FAILURES,
                rows_processed=5,
                rows_succeeded=2,
                rows_failed=0,
                rows_routed_success=0,
                rows_routed_failure=3,
                rows_quarantined=0,
            )
```

**Step 2: Run RED before changing the predicate**

```bash
.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py::TestADR019PredicateBoundaryCases -v
```

Expected before Task 3.4: the simplified-predicate assertions fail because the
old predicate still treats `rows_quarantined` and `rows_routed_failure` as direct
failure indicators, and the impossible-subset `pytest.raises(...)` cases fail
because current L0 code does not yet reject them mechanically. The zero-row,
all-discarded, and correct all-quarantined shapes may already pass; they are
boundary pins that must stay green after the simplification.

**Step 3: GREEN after Task 3.4**

Run the same class after the predicate rewrite. Expected: all boundary tests
pass, proving the base counters (`rows_succeeded`, `rows_failed`) are the only
row-terminal predicate inputs, impossible subset shapes are rejected at L0, and
`rows_coalesce_failed` remains the separate run-level failure signal.

**Definition of Done:**
- [ ] `TestADR019PredicateBoundaryCases` exists in `tests/unit/contracts/test_run_result.py`
- [ ] Invalid `rows_quarantined > rows_failed` and `rows_routed_failure > rows_failed` shapes fail before Task 3.4 because no L0 subset invariant exists yet
- [ ] Zero-row, all-discarded, all-quarantined, mixed default/on-error, and impossible-subset rejection shapes pass after Task 3.4
- [ ] Boundary tests are included in the Phase 3 GREEN command

---

### Task 3.3: Update the accumulator (`outcomes.py::accumulate_row_outcomes`)

**Files:**
- Modify: `src/elspeth/engine/orchestrator/outcomes.py:235-307`

**Step 1: Replace the entire match block**

```python
# OLD (lines 235-307):
for result in results:
    if result.outcome == RowOutcome.COMPLETED:
        counters.rows_succeeded += 1
        sink_name = _require_sink_name(result)
        _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.COMPLETED)
    elif result.outcome == RowOutcome.ROUTED:
        counters.rows_routed_success += 1
        sink_name = _require_sink_name(result)
        counters.routed_destinations[sink_name] += 1
        _route_to_sink(sink_name, pending_tokens, result.token, RowOutcome.ROUTED)
    elif result.outcome == RowOutcome.ROUTED_ON_ERROR:
        # ... handles error_hash construction etc.
    # ... rest of branches

# NEW:
# Each (outcome, path) pair from ADR-019 § Mapping table maps to a counter
# increment block. The two BEHAVIOUR CHANGES from ADR-019 § Counter derivation
# contract are the GATE_ROUTED and ON_ERROR_ROUTED branches: each now bumps
# BOTH the predicate-input counter AND the destination-attribution counter.
for result in results:
    pair = (result.outcome, result.path)
    if pair == (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW):
        counters.rows_succeeded += 1
        sink_name = _require_sink_name(result)
        _route_to_sink(
            sink_name,
            pending_tokens,
            result.token,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
        )
    elif pair == (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED):
        # ADR-019 BEHAVIOUR CHANGE: gate MOVE bumps BOTH rows_succeeded and
        # rows_routed_success. The predicate clause OR rows_routed_success > 0
        # becomes vestigial under this dual-bump.
        counters.rows_succeeded += 1
        counters.rows_routed_success += 1
        sink_name = _require_sink_name(result)
        counters.routed_destinations[sink_name] += 1
        _route_to_sink(
            sink_name,
            pending_tokens,
            result.token,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
        )
    elif pair == (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED):
        # ADR-019 BEHAVIOUR CHANGE: on_error DIVERT bumps BOTH rows_failed and
        # rows_routed_failure (symmetric to GATE_ROUTED above).
        if result.error is None:
            raise OrchestrationInvariantError(
                f"ON_ERROR_ROUTED result missing error (FailureInfo). Token: {result.token}"
            )
        sink_name = _require_sink_name(result)
        error_hash = hashlib.sha256(result.error.message.encode()).hexdigest()[:16]
        counters.rows_failed += 1
        counters.rows_routed_failure += 1
        counters.routed_destinations[sink_name] += 1
        _route_to_sink(
            sink_name,
            pending_tokens,
            result.token,
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.ON_ERROR_ROUTED,
            error_hash=error_hash,
        )
    elif pair == (TerminalOutcome.FAILURE, TerminalPath.UNROUTED):
        counters.rows_failed += 1
    elif pair == (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE):
        counters.rows_quarantined += 1
        counters.rows_failed += 1
    elif pair == (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED):
        counters.rows_succeeded += 1
    elif pair == (TerminalOutcome.SUCCESS, TerminalPath.COALESCED):
        # Terminal coalesce merged rows still reach a sink, but the coalesce
        # identity is not topology-derived here: TokenInfo.join_group_id is the
        # producer-carried witness created by CoalesceExecutor. Sink recording
        # must pass that join_group_id through to record_token_outcome().
        if result.token.join_group_id is None:
            raise OrchestrationInvariantError(
                f"(SUCCESS, COALESCED) result missing token.join_group_id. "
                f"Token: {result.token}"
            )
        sink_name = _require_sink_name(result)
        counters.rows_coalesced += 1
        counters.rows_succeeded += 1
        _route_to_sink(
            sink_name,
            pending_tokens,
            result.token,
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.COALESCED,
        )
    elif pair == (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT):
        counters.rows_forked += 1
    elif pair == (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT):
        counters.rows_expanded += 1
    elif pair == (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED):
        # Aggregated — counted when batch flushes (the batch-result token's
        # SUCCESS terminal carries the lifecycle answer).
        pass
    elif pair == (None, TerminalPath.BUFFERED):
        counters.rows_buffered += 1
    elif pair[1] in (
        TerminalPath.SINK_FALLBACK_TO_FAILSINK,
        TerminalPath.SINK_DISCARDED,
    ):
        # Diversions occur in SinkExecutor and are counted there, not here.
        # Reaching this branch in the orchestrator's accumulator means a
        # producer emitted a diversion outside the SinkExecutor's path —
        # Tier 1 invariant violation.
        raise OrchestrationInvariantError(
            f"Diversion path ({pair}) should not appear in the orchestrator "
            f"accumulator — diversions are counted in SinkExecutor. "
            f"Token: {result.token}"
        )
    else:
        raise OrchestrationInvariantError(
            f"Unhandled (outcome, path) pair: {pair!r}. Token: {result.token}. "
            f"Add an explicit case above; see ADR-019 § Mapping table."
        )
```

**Step 2: Update `_route_to_sink` helper signature**

`_route_to_sink` constructs a `PendingOutcome`. Update it to accept `(outcome, path)` keywords:

```python
# Old signature: _route_to_sink(sink_name, pending_tokens, token, outcome: RowOutcome, *, error_hash=None)
# New signature:
def _route_to_sink(
    sink_name: str,
    pending_tokens: dict[str, list[tuple[TokenInfo, PendingOutcome]]],
    token: TokenInfo,
    *,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    error_hash: str | None = None,
) -> None:
    pending_tokens[sink_name].append(
        (token, PendingOutcome(outcome=outcome, path=path, error_hash=error_hash))
    )
```

Also update every direct `PendingOutcome(...)` construction that bypasses `_route_to_sink`. The live quarantine path in `src/elspeth/engine/orchestrator/core.py` constructs `PendingOutcome(RowOutcome.QUARANTINED, quarantine_error_hash)` directly and increments only `rows_quarantined`. After Phase 1's keyword-only retype and Phase 3's exhaustive-failure counter contract, this source-quarantine branch must become:

```python
# Destination validated - increment counters and proceed with routing.
# ADR-019: (FAILURE, QUARANTINED_AT_SOURCE) contributes to both the
# reporting subset counter and the exhaustive failure predicate counter.
counters.rows_quarantined += 1
counters.rows_failed += 1
```

And the pending outcome append must become:

```python
PendingOutcome(
    outcome=TerminalOutcome.FAILURE,
    path=TerminalPath.QUARANTINED_AT_SOURCE,
    error_hash=quarantine_error_hash,
)
```

Update `reconcile_sink_write_diversions` in `outcomes.py` at the same time: comparisons against `pending_outcome.outcome == RowOutcome.X` must become explicit `(pending_outcome.outcome, pending_outcome.path)` checks so discard-mode and failsink-mode diversions do not collapse into one branch.

**Load-bearing reconciliation matrix:** after Phase 3, routed/quarantined counters are non-disjoint subsets of the base counters. A diverted pending token must remove every provisional counter the accumulator already bumped before the sink write failed:

| Pending pair before sink write | Provisional counters to subtract on diversion |
| --- | --- |
| `(SUCCESS, DEFAULT_FLOW)` | `rows_succeeded` |
| `(SUCCESS, COALESCED)` | `rows_succeeded`, `rows_coalesced`; also require `token.join_group_id` and pass it through the sink recorder |
| `(SUCCESS, GATE_ROUTED)` | `rows_succeeded`, `rows_routed_success`, `routed_destinations[sink_name]` |
| `(FAILURE, ON_ERROR_ROUTED)` | `rows_failed`, `rows_routed_failure`, `routed_destinations[sink_name]` |
| `(FAILURE, QUARANTINED_AT_SOURCE)` | `rows_failed`, `rows_quarantined` |

Only after those provisional counters are reconciled does the orchestrator add sink-diversion counters from `DiversionCounts`: `rows_diverted += total`, and `rows_failed += discard_mode` only for `(FAILURE, SINK_DISCARDED)` discard diversions. Failsink-mode diversions add `rows_diverted` only because `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` is not a predicate input.

Add unit tests in `tests/unit/engine/orchestrator/test_outcomes.py` that
exercise default-flow, gate-routed, on-error-routed, quarantined, and coalesced
pending rows. The coalesced case is mandatory and must assert that a missing
`token.join_group_id` crashes before recording, while a populated
`token.join_group_id` is preserved into the sink-side `record_token_outcome()`
call. Current HEAD already has
coalesced sink-routing coverage in
`tests/unit/engine/orchestrator/test_outcomes.py::TestCoalescedOutcome::test_coalesced_routes_to_sink`
and terminal coalesce tests that return `RowOutcome.COALESCED` with
`sink_name="output"`, so coalesced pending outcomes can reach
`SinkExecutor.write()`.

Required test names:

```python
def test_reconcile_sink_write_diversion_subtracts_default_flow_success() -> None: ...
def test_reconcile_sink_write_diversion_subtracts_gate_routed_success_subset() -> None: ...
def test_reconcile_sink_write_diversion_subtracts_on_error_failure_subset() -> None: ...
def test_reconcile_sink_write_diversion_subtracts_quarantined_failure_subset() -> None: ...
def test_reconcile_sink_write_diversion_subtracts_coalesced_success_and_preserves_join_group() -> None: ...
```

The coalesced regression starts with counters
`rows_succeeded=1, rows_coalesced=1`, a pending pair
`(TerminalOutcome.SUCCESS, TerminalPath.COALESCED)`, a token with
`join_group_id="join-1"`, and a discard-mode sink diversion. After
reconciliation, `rows_succeeded == 0`,
`rows_coalesced == 0`, `rows_diverted == 1`, and `rows_failed == 1` only after
the orchestrator applies `DiversionCounts.discard_mode`.

**Step 3: Preserve `path` in sink-write grouping**

`_write_pending_to_sinks` currently groups pending writes by sink-local outcome
shape before calling `SinkExecutor.write()`. After Phase 1, `PendingOutcome`
contains both `outcome` and `path`; Phase 3 must make `path` part of the
sort/group key. Otherwise `(SUCCESS, DEFAULT_FLOW)` and
`(SUCCESS, GATE_ROUTED)` rows with the same sink and error hash collapse into
one sink write, losing the path witness that the recorder and cross-table
invariants need.

At the sink recording call, preserve coalesce identity mechanically: when
`pending.path == TerminalPath.COALESCED`, pass `join_group_id=token.join_group_id`
to `record_token_outcome()` and crash if that token field is missing. This is how
Phase 3 keeps the Phase 1 recorder contract (`COALESCED` requires
`join_group_id`) without reclassifying terminal coalesce output as default flow
or inventing a topology-derived join ID at the sink.

```python
def pending_group_key(
    pair: tuple[TokenInfo, PendingOutcome | None],
) -> tuple[bool, str, str, str]:
    pending = pair[1]
    if pending is None:
        return (True, "", "", "")
    outcome_value = pending.outcome.value if pending.outcome is not None else ""
    return (
        False,
        outcome_value,
        pending.path.value,
        pending.error_hash or "",
    )


sorted_pairs = sorted(token_outcome_pairs, key=pending_group_key)
for _group_key, group in groupby(sorted_pairs, key=pending_group_key):
    group_pairs = list(group)
    pending_outcome = group_pairs[0][1]
    group_tokens = [token for token, _pending in group_pairs]
    ...
```

Add a focused unit test that sends mixed pending rows for the same sink:
`(SUCCESS, DEFAULT_FLOW)` and `(SUCCESS, GATE_ROUTED)` with the same
`error_hash`. The test must prove `SinkExecutor.write()` is invoked once per
distinct `(outcome, path, error_hash)` group and that the recorder receives the
original path for every row.

**Step 4: Update diversion-counter accumulation at the orchestrator (NOT inside SinkExecutor)**

**Reality check:** `SinkExecutor` has no `_counters` attribute — counter accumulation happens at the orchestrator level. `SinkExecutor.write()` (`sink.py:341`) returns `tuple[Artifact | None, int]`; the int is `diversion_count`. `Orchestrator._run_main_processing_loop` at `core.py:2199` accumulates it via `loop_ctx.counters.rows_diverted += total_diversions`. The discard-mode `rows_failed` bump must live at that orchestrator site, AND `SinkExecutor.write()` must return enough information for the orchestrator to know how many of the diversions were discard-mode versus failsink-mode (the orchestrator can no longer treat `total_diversions` as opaque).

**Sub-step 4a: Change the SinkExecutor.write() return shape to distinguish diversion flavors**

Add a small frozen dataclass alongside the existing return contract:

```python
# In src/elspeth/engine/executors/sink.py near the top of the file:

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class DiversionCounts:
    """ADR-019 § Sub-decision 5: SinkExecutor.write() reports diversion counts
    split by flavor so the orchestrator can apply the correct predicate-bucket
    increments.

    failsink_mode: (TRANSIENT, SINK_FALLBACK_TO_FAILSINK) — bumps only the
        structural rows_diverted counter (TRANSIENT is not a predicate input).
    discard_mode: (FAILURE, SINK_DISCARDED) — bumps BOTH rows_diverted
        (structural, unchanged from ADR-018) AND rows_failed (predicate input,
        NEW under ADR-019). Operator-visible RunStatus flip.
    """

    failsink_mode: int = 0
    discard_mode: int = 0

    @property
    def total(self) -> int:
        return self.failsink_mode + self.discard_mode
```

Update `SinkExecutor.write()` (signature at line 341, return statement at line 1019) to return `tuple[Artifact | None, DiversionCounts]`. Inside the failsink-mode block (after the loop at lines 949-957) and the discard-mode block (after the loop at lines 980-1003), increment local `failsink_count` and `discard_count` integers; assemble the `DiversionCounts` for the return statement.

```python
# Old return at line ~1019:
return artifact, diversion_count

# New return:
return artifact, DiversionCounts(failsink_mode=failsink_count, discard_mode=discard_count)
```

**Sub-step 4b: Update orchestrator to accumulate the split counts**

```python
# In core.py around line 2199 — OLD:
loop_ctx.counters.rows_diverted += total_diversions

# NEW:
# ADR-019 § Sub-decision 5: discard-mode diversions bump rows_failed
# (predicate input). Failsink-mode diversions bump only the structural
# rows_diverted (TRANSIENT, not a predicate input).
loop_ctx.counters.rows_diverted += diversion_counts.total
loop_ctx.counters.rows_failed += diversion_counts.discard_mode
```

The local variable that today holds `total_diversions` (returned from the sink-write call site) becomes `diversion_counts: DiversionCounts`. Update the direct `SinkExecutor.write()` destructuring site and the later orchestrator counter-update site that consumes the value — find them via `rg -n "rows_diverted|sink_executor.*write|\\.write\\(" src/elspeth/engine/orchestrator/core.py` and update each.

**Sub-step 4c: Update any other callers of `SinkExecutor.write()`**

```bash
rg -n "SinkExecutor\\(|sink_executor\\.write\\(|executor\\.write\\(|artifact, diversion_count|diversion_count" src/elspeth tests
```

Each caller that destructured `(artifact, diversion_count)` becomes
`(artifact, diversion_counts)` and uses the dataclass fields. Test fixtures that
assert on the return shape need fixture updates in this commit
(schema-dependent per Phase 5 triage). As of the review that produced this
fixup, direct call sites existed in at least:

- `tests/unit/engine/test_sink_executor_diversion.py`
- `tests/unit/engine/test_executors.py`
- `tests/integration/plugins/sinks/test_durability.py`
- source call sites under `src/elspeth/engine/orchestrator/core.py`

Do not stop at a `src/elspeth/`-only grep. The full test gate in Task 3.7 is a
commit blocker, so return-shape fallout in tests is part of this phase.

**Partial-failure timing note:** `_write_pending_to_sinks` may durably record
some sink diversions before a later sink group raises. If the phase keeps the
current "return aggregate counts after all writes" shape, exception-bounded
failures rely on the audit rows as the authoritative partial-write evidence and
the final counter update may not run. Document that explicitly in the code
comment beside the aggregation, or refactor the counter update to apply after
each successful sink group before moving to the next group. Do not leave the
timing assumption implicit.

**Step 5: Update imports in `outcomes.py`**

Replace `from elspeth.contracts.enums import RowOutcome` with `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`.

**Step 6: Verify**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/test_outcomes.py -v`

Expected: existing accumulator unit tests pass. The integration tests from Tasks 3.1 and 3.2 should now go GREEN.

**Definition of Done:**
- [ ] Accumulator pattern-matches on `(outcome, path)`
- [ ] GATE_ROUTED and ON_ERROR_ROUTED branches bump both counters
- [ ] `_route_to_sink` signature updated
- [ ] `_write_pending_to_sinks` sort/group key includes `pending.path` and uses the same key function for both `sorted(...)` and `groupby(...)`
- [ ] Mixed default-flow/gate-routed pending rows for the same sink do not collapse into one write group
- [ ] Direct quarantine branch in `Orchestrator._handle_quarantine_row` increments BOTH `rows_quarantined` and `rows_failed`
- [ ] Direct quarantine `PendingOutcome(...)` construction updated with keyword-only `(outcome, path, error_hash)`
- [ ] `reconcile_sink_write_diversions` compares explicit `(outcome, path)` pairs instead of `RowOutcome`
- [ ] `reconcile_sink_write_diversions` subtracts every provisional base and subset counter in the matrix above; routed destinations are decremented for both routed paths
- [ ] `DiversionCounts` dataclass added in `sink.py`; `SinkExecutor.write()` return shape changed from `tuple[Artifact | None, int]` to `tuple[Artifact | None, DiversionCounts]`
- [ ] All `SinkExecutor.write()` call sites updated to destructure the dataclass where needed (verified by the all-tree `rg` command in Sub-step 4c, not by a source-only grep)
- [ ] Direct sink-executor tests and plugin sink durability tests are updated or explicitly verified to ignore the return shape safely
- [ ] Orchestrator counter-update site at `core.py:~2199` increments BOTH `rows_diverted` (total) AND `rows_failed` (discard_mode subset)
- [ ] Discard-mode integration test (Task 3.1) passes
- [ ] Counter-doubling integration test (Task 3.2) passes
- [ ] mypy clean
- [ ] No `self._counters` reference inside SinkExecutor (verified by `grep -n "_counters" src/elspeth/engine/executors/sink.py` returning empty)

---

### Task 3.4: Update the predicate (`run_result.py::__post_init__`, `derive_terminal_run_status`)

**Files:**
- Modify: `src/elspeth/contracts/run_result.py:60-152` (`__post_init__`)
- Modify: `src/elspeth/contracts/run_result.py:180-216` (`derive_terminal_run_status`)

**Step 1: Update existing predicate unit tests FIRST (RED)**

Existing tests in `tests/unit/contracts/test_run_result.py` assert the bifurcated predicate. Update them to assert the simplified predicate:

```python
# Update assertions like:
def test_completed_status_with_only_routed_success(self) -> None:
    """A run with only gate-routed rows and rows_routed_success > 0 is COMPLETED.

    UNDER ADR-019: this test changes — gate-routed bumps rows_succeeded too,
    so the test fixture's rows_succeeded value also increments.
    """
    # OLD assertion: RunResult(status=COMPLETED, rows_succeeded=0, rows_routed_success=4)
    # NEW assertion: RunResult(status=COMPLETED, rows_succeeded=4, rows_routed_success=4)
    result = RunResult(
        status=RunStatus.COMPLETED,
        rows_processed=4,
        rows_succeeded=4,         # NEW: gate-routed bumps this too
        rows_routed_success=4,
        rows_failed=0,
        ...
    )
    assert result.status == RunStatus.COMPLETED
```

Run RED test against the existing run_result.py.

**Step 1a: Add the L0 subset invariant helper**

Before editing the predicate, add a named helper in
`src/elspeth/contracts/run_result.py` and call it from both
`RunResult.__post_init__` and `derive_terminal_run_status(...)`. This is a
Tier-1 mechanical guard, not a documentation-only expectation:

```python
def _validate_counter_subsets(
    *,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """ADR-019: structural counters are subsets of base terminal counters."""
    if rows_routed_success > rows_succeeded:
        raise ValueError(
            "rows_routed_success must be <= rows_succeeded under ADR-019; "
            f"got rows_routed_success={rows_routed_success}, "
            f"rows_succeeded={rows_succeeded}"
        )
    if rows_routed_failure > rows_failed:
        raise ValueError(
            "rows_routed_failure must be <= rows_failed under ADR-019; "
            f"got rows_routed_failure={rows_routed_failure}, "
            f"rows_failed={rows_failed}"
        )
    if rows_quarantined > rows_failed:
        raise ValueError(
            "rows_quarantined must be <= rows_failed under ADR-019; "
            f"got rows_quarantined={rows_quarantined}, "
            f"rows_failed={rows_failed}"
        )
```

Call this helper before status classification in `derive_terminal_run_status(...)`
so callers cannot derive a clean terminal status from corrupt counter shapes. Call
it in `RunResult.__post_init__` after integer validation and before
`_check_status_invariant()`.

**Step 2: Apply the predicate edits**

```python
# OLD (lines 71-74):
success_indicator = self.rows_succeeded > 0 or self.rows_routed_success > 0
failure_indicator = (
    self.rows_failed > 0 or self.rows_quarantined > 0 or self.rows_coalesce_failed > 0 or self.rows_routed_failure > 0
)

# NEW:
# ADR-019 § Counter derivation contract — public API field names preserved.
# Under the round-3-resolved two-axis model, `(SUCCESS, GATE_ROUTED)` bumps
# BOTH rows_succeeded AND rows_routed_success at the accumulator (and
# symmetrically (FAILURE, ON_ERROR_ROUTED) bumps both rows_failed and
# rows_routed_failure). The bifurcated OR clauses are vestigial.
success_indicator = self.rows_succeeded > 0
failure_indicator = (
    self.rows_failed > 0 or self.rows_coalesce_failed > 0
)
```

Update the error-message strings in the `case` arms below to match the simplified predicate. Most messages reference `rows_succeeded > 0 or rows_routed_success > 0`; trim to `rows_succeeded > 0`.

```python
# Same edit for derive_terminal_run_status (lines 208-216):
success_indicator = rows_succeeded > 0
failure_indicator = rows_failed > 0 or rows_coalesce_failed > 0
```

**Step 3: Keep subset counters as guard-only inputs**

`derive_terminal_run_status` takes `rows_routed_success`, `rows_routed_failure`,
and `rows_quarantined` parameters. After ADR-019 they are no longer predicate
inputs, but they MUST remain in the function signature because
`_validate_counter_subsets(...)` uses them to reject corrupt counter shapes before
status classification. Treat them as guard-only inputs:

- Keep the named parameters in `derive_terminal_run_status(...)`.
- Call `_validate_counter_subsets(...)` before computing `success_indicator` and
  `failure_indicator`.
- Do not OR any routed/quarantine subset counter into the simplified predicate.
- Keep all callers passing the routed/quarantine counters by keyword so the guard
  remains mechanically enforced at every terminal-status derivation site.

This is intentionally not an API-cleanup task. Dropping the parameters would make
`derive_terminal_run_status(...)` capable of deriving a clean status from an
impossible shape such as `rows_quarantined > rows_failed`, which is the exact
Tier-1 corruption guard Phase 3 is adding.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py tests/integration/test_adr_019_*.py -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] `_validate_counter_subsets` added in `run_result.py` and called from both `RunResult.__post_init__` and `derive_terminal_run_status(...)`
- [ ] L0 rejects `rows_routed_success > rows_succeeded`, `rows_routed_failure > rows_failed`, and `rows_quarantined > rows_failed`
- [ ] `__post_init__` predicate simplified
- [ ] `derive_terminal_run_status` predicate simplified
- [ ] Error-message strings updated to match new predicate
- [ ] `derive_terminal_run_status` keeps `rows_routed_success`, `rows_routed_failure`, and `rows_quarantined` as guard-only parameters; callers keep passing them by keyword
- [ ] All existing tests pass after fixture updates
- [ ] Both ADR-019 integration tests pass
- [ ] mypy clean

---

### Task 3.5: Update L3 Pydantic predicate mirror in `web/execution/schemas.py`

**Why this task exists:** `web/execution/schemas.py` carries TWO duplicates of the L0 predicate logic that ship as part of the response-validation layer for `/api/runs/{rid}`:

1. **`_validate_row_decomposition` (line 138)** — asserts `rows_processed >= rows_succeeded + rows_failed + rows_routed_success + rows_routed_failure + rows_quarantined`. This formula assumes `rows_succeeded` and `rows_routed_success` are DISJOINT — true under ADR-018, **false after Phase 3's counter doubling**. A run with N gate-routed rows now has `rows_succeeded=N` AND `rows_routed_success=N`; the right-hand sum is `2N`, the left-hand `rows_processed=N`, so the inequality fails → `ValueError` → HTTP 500 on every gate-MOVE / on_error-routed run.

2. **`_check_status_row_count_invariant` (line ~218)** — Pydantic mirror of the L0 status/count biconditional with the same bifurcated-OR predicate the L0 layer is dropping. Functionally still valid post-Phase-3 (the OR clauses remain True when `rows_succeeded > 0`), but the error-message strings reference the dropped clauses and will mislead operators when they DO see a validation failure for an unrelated reason.

**This is a Phase 3 regression** because the bug is triggered by Phase 3's accumulator counter-doubling. The L0 predicate change in Task 3.4 must land in lockstep with the L3 mirror change here. ADR-019 § Counter derivation contract preserves public API field NAMES but changes the semantic of which fields are bumped — the L3 schema mirror must match.

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py:138-160` (`_validate_row_decomposition`)
- Modify: `src/elspeth/web/execution/schemas.py:197-285` (`_check_status_row_count_invariant`)
- Test: `tests/unit/web/execution/test_schemas.py` — extend `TestCompletedDataDecomposition` (EXISTING); no new test file needed

**Test shape rationale (shape c — schema-layer unit tests):**

The bug is a Pydantic validator invariant (`_validate_row_decomposition`). Testing it at the `CompletedData` / `RunStatusResponse` constructor level is both sufficient and the established pattern — `TestCompletedDataDecomposition` already lives in `tests/unit/web/execution/test_schemas.py` and exercises exactly this predicate.

A web-layer integration test using `TestClient(app)` and `/api/runs/{run_id}` would additionally require: `create_app()` with a test-scoped `WebSettings`
(no module-level `app` object is exported — `src/elspeth/web/app.py:293` exports only
`def create_app(settings: WebSettings | None = None) -> FastAPI`), an auth token
(the route enforces ownership via `_verify_run_ownership`), a run seeded through the
full web `ExecutionService`, and `asgi_lifespan.LifespanManager`. That is end-to-end
coverage of the wrong layer: the exercise is already provided by
`tests/integration/web/test_execute_pipeline.py::TestGateRoutedPipelineExecution`
(gate-routed pipeline, `/api/runs/{run_id}`, `rows_routed_success > 0` assertions).
Duplicating it here is CYA, not regression value.

**Step 1: Write the RED-first regression tests — extend `TestCompletedDataDecomposition`**

Add the following methods to `tests/unit/web/execution/test_schemas.py::TestCompletedDataDecomposition`:

```python
# Imports already at the top of test_schemas.py:
#   from elspeth.web.execution.schemas import (CompletedData, RunStatusResponse, ...)
# Source symbols:
#   _validate_row_decomposition  src/elspeth/web/execution/schemas.py:138
#   CompletedData                src/elspeth/web/execution/schemas.py:288

def test_gate_routed_shape_accepted_post_phase3(self) -> None:
    """ADR-019 B4 regression: gate-MOVE rows bump BOTH rows_succeeded and
    rows_routed_success after Phase 3's accumulator change. Pre-fix,
    _validate_row_decomposition added rows_routed_success to the RHS sum,
    producing sum=8 against rows_processed=4, raising ValueError (HTTP 500).
    Post-fix the RHS is rows_succeeded + rows_failed only. Quarantines are
    included in rows_failed and also reported separately via rows_quarantined.

    # _validate_row_decomposition: src/elspeth/web/execution/schemas.py:138
    """
    data = CompletedData(
        status="completed",
        rows_processed=4,
        rows_succeeded=4,
        rows_failed=0,
        rows_routed_success=4,   # non-disjoint SUBSET of rows_succeeded post-Phase-3
        rows_routed_failure=0,
        rows_quarantined=0,
        landscape_run_id="lscape-gate-phase3",
    )
    assert data.rows_succeeded == 4
    assert data.rows_routed_success == 4

def test_on_error_routed_shape_accepted_post_phase3(self) -> None:
    """Symmetric B4 regression for transform on_error routing. Pre-fix,
    rows_failed=3 AND rows_routed_failure=3 produced sum=8 against
    rows_processed=5, raising ValueError (HTTP 500).

    # _validate_row_decomposition: src/elspeth/web/execution/schemas.py:138
    """
    data = CompletedData(
        status="completed_with_failures",
        rows_processed=5,
        rows_succeeded=2,
        rows_failed=3,
        rows_routed_success=0,
        rows_routed_failure=3,   # non-disjoint SUBSET of rows_failed post-Phase-3
        rows_quarantined=0,
        landscape_run_id="lscape-on-error-phase3",
    )
    assert data.rows_failed == 3
    assert data.rows_routed_failure == 3

def test_routed_success_cannot_exceed_rows_succeeded_post_phase3(self) -> None:
    """rows_routed_success is a subset of rows_succeeded, not a sibling bucket.

    This RED test constructs an impossible schema payload directly. A future
    accumulator regression that bumps rows_routed_success without the paired
    rows_succeeded bump must be caught at the Pydantic boundary instead of
    relying on construction discipline alone.
    """
    with pytest.raises(pydantic.ValidationError, match="rows_routed_success"):
        CompletedData(
            status="completed",
            rows_processed=5,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=2,   # impossible subset relationship
            rows_routed_failure=0,
            rows_quarantined=0,
            landscape_run_id="lscape-routed-subset",
        )

def test_routed_failure_cannot_exceed_rows_failed_post_phase3(self) -> None:
    """rows_routed_failure is a subset of rows_failed."""
    with pytest.raises(pydantic.ValidationError, match="rows_routed_failure"):
        CompletedData(
            status="completed_with_failures",
            rows_processed=5,
            rows_succeeded=2,
            rows_failed=1,
            rows_routed_success=0,
            rows_routed_failure=2,   # impossible subset relationship
            rows_quarantined=0,
            landscape_run_id="lscape-routed-failure-subset",
        )

def test_quarantined_cannot_exceed_rows_failed_post_phase3(self) -> None:
    """rows_quarantined is a reporting subset of rows_failed."""
    with pytest.raises(pydantic.ValidationError, match="rows_quarantined"):
        CompletedData(
            status="completed_with_failures",
            rows_processed=5,
            rows_succeeded=2,
            rows_failed=1,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=2,   # impossible subset relationship
            landscape_run_id="lscape-quarantine-subset",
        )
```

**Also update `test_double_counted_routed_success_rejected` (line 455) in the same class.** That test currently asserts that `rows_processed=1, rows_succeeded=1, rows_routed_success=1` raises a `ValidationError` with "decomposition mismatch". Post-Phase-3, this is the VALID counter shape for a single gate-routed row — the test is pinned to the pre-Phase-3 invariant and will need to be updated (or converted to a positive-case assertion) in the same commit as the predicate change.

```python
# OLD (locked to pre-Phase-3 invariant — DELETE this test):
def test_double_counted_routed_success_rejected(self) -> None:
    with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
        CompletedData(
            status="completed",
            rows_processed=1,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=1,
            rows_routed_failure=0,
            rows_quarantined=0,
            landscape_run_id="lscape-routed",
        )

# NEW (post-Phase-3 valid shape — replace with positive assertion):
def test_gate_routed_single_row_accepted(self) -> None:
    """Post-Phase-3: a single gate-routed row legitimately increments both
    rows_succeeded and rows_routed_success. The old test asserted this shape
    raises ValidationError; after the predicate fix it must be accepted.

    # _validate_row_decomposition: src/elspeth/web/execution/schemas.py:138
    """
    data = CompletedData(
        status="completed",
        rows_processed=1,
        rows_succeeded=1,
        rows_failed=0,
        rows_routed_success=1,
        rows_routed_failure=0,
        rows_quarantined=0,
        landscape_run_id="lscape-routed",
    )
    assert data.rows_routed_success == 1
    assert data.rows_succeeded == 1
```

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py::TestCompletedDataDecomposition -v
```

Expected: `test_gate_routed_shape_accepted_post_phase3` and `test_on_error_routed_shape_accepted_post_phase3` FAIL with `pydantic.ValidationError` ("decomposition mismatch") because `_validate_row_decomposition` still includes `rows_routed_*` in the RHS sum. `test_double_counted_routed_success_rejected` passes (it's pinned to the old invariant), and `test_gate_routed_single_row_accepted` fails (same formula rejection). The subset-bound tests also fail in the RED sense because the current schema accepts impossible `rows_routed_success > rows_succeeded`, `rows_routed_failure > rows_failed`, and `rows_quarantined > rows_failed` payloads. Together these failures confirm the predicate mirror is still in its pre-Phase-3 shape and lacks the named subset invariant.

**Step 3: Patch `_validate_row_decomposition` (line 138-160)**

```python
# OLD:
def _validate_row_decomposition(
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """Enforce rows_processed >= succeeded + failed + routed_success + routed_failure + quarantined.

    elspeth-5069612f3c — rows_routed split into rows_routed_success (MOVE) and
    rows_routed_failure (DIVERT). Both contribute to terminal-state counts.
    ...
    """
    sum_terminal = (
        rows_succeeded + rows_failed + rows_routed_success
        + rows_routed_failure + rows_quarantined
    )
    if rows_processed < sum_terminal:
        raise ValueError(...)

# NEW (Blocker B1 fix — ONLY succeeded + failed participate in decomposition):
def _validate_row_decomposition(
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
) -> None:
    """Enforce rows_processed >= rows_succeeded + rows_failed.

    ADR-019 § Counter derivation contract: under the two-axis terminal model,
    ``rows_succeeded`` is exhaustive for SUCCESS lifecycles (DEFAULT_FLOW,
    GATE_ROUTED, FILTER_DROPPED, COALESCED) and ``rows_failed`` is exhaustive
    for FAILURE lifecycles (UNROUTED, ON_ERROR_ROUTED, QUARANTINED_AT_SOURCE,
    SINK_DISCARDED). ``rows_quarantined`` is a non-disjoint SUBSET of
    ``rows_failed`` — counting it in a decomposition sum double-counts source
    quarantines. The structural counters ``rows_routed_success`` and
    ``rows_routed_failure`` are non-disjoint SUBSETS of ``rows_succeeded`` /
    ``rows_failed`` — counting them in a decomposition sum double-counts
    gate-routed and on-error-routed rows. These subset counters are display-only
    breakdown fields at the web/API layer and must be checked by the named
    subset invariant, not by this terminal-decomposition sum.

    The pre-ADR-019 formula included rows_routed_* in the sum because
    ADR-018 made gate-routed rows bump ONLY rows_routed_success (disjoint
    from rows_succeeded). ADR-019's accumulator change at outcomes.py:235
    bumps both, so the structural counters drop OUT of the formula here.

    NARROW INVARIANT (elspeth-31d53c7493 carry-forward, preserved). The
    inequality not equality form remains: DAG nodes (aggregation, fork,
    expansion, coalesce) emit TRANSIENT bookkeeping tokens that increment
    rows_processed without contributing to any of the three predicate
    buckets. The full DAG-aware balance is tracked in elspeth-cf84eb1b52.
    """
    sum_terminal = rows_succeeded + rows_failed
    if rows_processed < sum_terminal:
        raise ValueError(
            f"Row count decomposition mismatch (over-counting): rows_processed="
            f"{rows_processed} < rows_succeeded({rows_succeeded}) + "
            f"rows_failed({rows_failed}) "
            f"= {sum_terminal}. Tier 1 anomaly: orchestrator emitted more "
            f"predicate-bucket counts than input rows. ADR-019 § Counter "
            f"derivation contract: rows_routed_success and rows_routed_failure "
            f"are non-disjoint subsets and excluded from the decomposition sum."
        )
```

**Required shape:** drop `rows_routed_success`, `rows_routed_failure`, and
`rows_quarantined` from `_validate_row_decomposition`. They are no longer
terminal decomposition inputs after Phase 3; they are display-only breakdown
fields and named subset-invariant inputs. Keeping them as accepted-but-ignored
parameters is not allowed because it invites a future patch to reintroduce the
double-counting formula.

Run `grep -rn "_validate_row_decomposition(" src/elspeth/` and update every
caller so it passes only `(rows_processed, rows_succeeded, rows_failed)`.
Pass the routed/quarantine counters to `_validate_response_counter_subsets`
instead.

**Step 4: Add a named response-counter subset invariant**

After `rows_routed_*` drops out of the decomposition sum, the schema layer still
must enforce the Tier 1 subset relationships by name. Mirror the L0 helper from
Task 3.4, including quarantine:

```python
def _validate_response_counter_subsets(
    *,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """ADR-019: response structural counters are subsets of base counters."""
    if rows_routed_success > rows_succeeded:
        raise ValueError(
            "rows_routed_success must be <= rows_succeeded under ADR-019; "
            f"got rows_routed_success={rows_routed_success}, "
            f"rows_succeeded={rows_succeeded}"
        )
    if rows_routed_failure > rows_failed:
        raise ValueError(
            "rows_routed_failure must be <= rows_failed under ADR-019; "
            f"got rows_routed_failure={rows_routed_failure}, "
            f"rows_failed={rows_failed}"
        )
    if rows_quarantined > rows_failed:
        raise ValueError(
            "rows_quarantined must be <= rows_failed under ADR-019; "
            f"got rows_quarantined={rows_quarantined}, "
            f"rows_failed={rows_failed}"
        )
```

Call this helper from every Pydantic response model that carries both base
counters and structural subset counters (`ProgressData`, `CompletedData`,
`CancelledData`, `RunStatusResponse`, and `RunResultsResponse` if all are present
in current HEAD). This is a schema-boundary invariant, not just an accumulator
construction property.

**Step 5: Patch `_check_status_row_count_invariant` (line ~218)**

The functional check must match the L0 predicate's simplified shape:

```python
# OLD line 218:
success_indicator = rows_succeeded > 0 or rows_routed_success > 0
failure_indicator = rows_failed > 0 or rows_quarantined > 0 or rows_routed_failure > 0

# NEW:
# ADR-019 § Counter derivation contract: rows_succeeded is exhaustive for
# SUCCESS lifecycles; rows_failed is exhaustive for FAILURE lifecycles.
# rows_quarantined is a reporting subset of rows_failed, not a separate
# predicate input. The bifurcated OR clauses become vestigial.
success_indicator = rows_succeeded > 0
failure_indicator = rows_failed > 0
```

Update the four error-message strings (lines ~248, ~258, ~273, ~282) to drop the `or rows_routed_success` / `or rows_routed_failure` references. The function signature still receives those parameters because callers pass them, but they're no longer used in the predicate.

**Step 6: Update callers of `_validate_row_decomposition`**

```bash
grep -n "_validate_row_decomposition(" src/elspeth/web/execution/schemas.py
```

Each call site (CompletedData, CancelledData, RunStatusResponse, ProgressData if applicable) drops the two `rows_routed_*` arguments and `rows_quarantined` from `_validate_row_decomposition`. Those counters still feed `_validate_response_counter_subsets`; they are display-only breakdown/subset fields, not decomposition inputs. The `_check_status_row_count_invariant` function may still accept routed/quarantine parameters if its callers expose them, but it must treat them as non-predicate display/subset fields and must not OR them into terminal status.

**Step 7: GREEN**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/execution/test_schemas.py \
  tests/integration/test_adr_019_counter_changes.py \
  tests/integration/test_adr_019_discard_mode_flip.py \
  tests/integration/web/test_execute_pipeline.py::TestGateRoutedPipelineExecution \
  -v
```

Expected: all green. `test_double_counted_routed_success_rejected` must have been replaced by `test_gate_routed_single_row_accepted` in this commit — if the old test still exists, it will spuriously fail (it was pinned to the pre-Phase-3 invariant). `TestGateRoutedPipelineExecution` in `tests/integration/web/test_execute_pipeline.py` provides the end-to-end `/api/runs/{run_id}` coverage that pins this regression at the HTTP layer; no additional API-layer test file is needed.

**Definition of Done:**
- [ ] `_validate_row_decomposition` formula simplified — drops `rows_routed_*` and `rows_quarantined` from the RHS sum (new formula: `rows_succeeded + rows_failed`)
- [ ] `_validate_response_counter_subsets` added and wired into every response model that exposes base counters plus routed/quarantined subset counters
- [ ] `_check_status_row_count_invariant` predicate aligned with L0 (drops bifurcated OR; `success_indicator = rows_succeeded > 0`, `failure_indicator = rows_failed > 0`)
- [ ] Error messages updated to match the new predicate shape
- [ ] All `_validate_row_decomposition` callers updated to pass only `rows_processed`, `rows_succeeded`, and `rows_failed`; routed/quarantine counters are routed to `_validate_response_counter_subsets`
- [ ] `test_gate_routed_shape_accepted_post_phase3` and `test_on_error_routed_shape_accepted_post_phase3` pass in `TestCompletedDataDecomposition`
- [ ] Subset-boundary tests reject `rows_routed_success > rows_succeeded`, `rows_routed_failure > rows_failed`, and `rows_quarantined > rows_failed`
- [ ] `test_double_counted_routed_success_rejected` removed; replaced with `test_gate_routed_single_row_accepted` (positive assertion for the post-Phase-3 valid shape)
- [ ] `TestGateRoutedPipelineExecution` (already in `tests/integration/web/test_execute_pipeline.py`) passes — provides end-to-end `/api/runs/{run_id}` coverage
- [ ] All pre-existing `test_schemas.py` tests pass after any fixture updates
- [ ] mypy clean across `src/elspeth/web/execution/`

---

### Task 3.5a: Update web terminal-status replay, session persistence, cancellation, and frontend active-run predicates

**Why this task exists:** `src/elspeth/web/sessions/protocol.py` already defines `SESSION_TERMINAL_RUN_STATUS_VALUES = {"completed", "completed_with_failures", "failed", "empty", "cancelled"}` and `RunRecord.__post_init__` requires `finished_at` for every terminal status. However, several web/backend/frontend surfaces still hard-code the old terminal subset `("completed", "failed", "cancelled")`: `web/execution/routes.py` WebSocket seed/idle checks, `web/sessions/service.py::update_run_status` finished-at stamping, `web/execution/service.py::cancel` idempotency, and `SessionSidebar.tsx` active-run visibility. These stale terminal subsets become more visible once ADR-019 makes `completed_with_failures` and `empty` common operator-facing results.

**Files:**
- Modify: `src/elspeth/web/sessions/protocol.py` (add shared operator-completion status constant)
- Modify: `src/elspeth/web/execution/routes.py:669, 686`
- Modify: `src/elspeth/web/sessions/service.py:672-679`
- Modify: `src/elspeth/web/execution/service.py:614-633`
- Modify: `src/elspeth/web/frontend/src/types/index.ts` (add terminal-status type guard)
- Modify: `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.tsx:19-24`
- Test: `tests/unit/web/execution/test_websocket.py` or `tests/unit/web/execution/test_routes.py`
- Test: `tests/unit/web/sessions/test_service.py` or nearest existing SessionService test file
- Test: `tests/unit/web/execution/test_service.py` or nearest existing ExecutionService cancel test file
- Test: create `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx` if no closer existing colocated test file exists

**Step 1: Backend RED tests**

Add tests for all backend terminal-set consumers:

1. A client connects after the run is already `completed_with_failures`; the route sends `_build_terminal_run_event(...)` and closes.
2. A client idles, status is rechecked as `empty`; the route sends `_build_terminal_run_event(...)` and closes.
3. `SessionService.update_run_status(..., status="completed_with_failures")` stores `finished_at` and the returned `RunRecord` does not raise the terminal-null `AuditIntegrityError`.
4. `SessionService.update_run_status(..., status="empty")` stores `finished_at` and the returned `RunRecord` does not raise the terminal-null `AuditIntegrityError`.
5. `SessionService.update_run_status(..., status="completed")`, `status="completed_with_failures"`, and `status="empty"` each raise `ValueError` when neither the new call nor the existing row has a `landscape_run_id`.
6. `ExecutionService.cancel(run_id)` is idempotent for existing runs with `status="completed_with_failures"` and `status="empty"`; it must not attempt a transition to `cancelled`.

Use `SESSION_TERMINAL_RUN_STATUS_VALUES` in assertions so the test catches future local hardcoded subsets.

**Step 2: Backend fixes**

Add a shared operator-completion subset next to the existing terminal set in
`web/sessions/protocol.py`, then replace every hardcoded backend terminal subset
with shared protocol constants:

```python
OperatorCompletionSessionRunStatus = Literal["completed", "completed_with_failures", "empty"]
OPERATOR_COMPLETION_RUN_STATUS_VALUES: frozenset[str] = frozenset(
    get_args(OperatorCompletionSessionRunStatus)
)

from elspeth.web.sessions.protocol import (
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
)

# routes.py seed/idle checks:
if current.status in SESSION_TERMINAL_RUN_STATUS_VALUES:
    event = _build_terminal_run_event(current)
    await websocket.send_json(event.model_dump(mode="json"))
    await websocket.close(code=1000)
    return

# web/sessions/service.py::update_run_status:
if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and not (
    landscape_run_id or current.landscape_run_id
):
    raise ValueError(f"{status} status requires landscape_run_id")

if status in SESSION_TERMINAL_RUN_STATUS_VALUES:
    values["finished_at"] = now

# web/execution/service.py::cancel:
if run.status not in SESSION_TERMINAL_RUN_STATUS_VALUES:
    await self._session_service.update_run_status(run_id, status="cancelled")
```

The imports must come from `web.sessions.protocol`; do not invent second local
terminal or operator-completion sets in `routes.py`, `web/sessions/service.py`,
or `web/execution/service.py`. The write guard must mirror
`RunRecord.__post_init__`: all operator-completion statuses
(`completed`, `completed_with_failures`, `empty`) require `landscape_run_id`;
every terminal status requires `finished_at`.

**Step 3: Frontend RED test + fix**

Add or extend `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx` so `hasActiveRun` is false for every terminal status, including `completed_with_failures` and `empty`. If a closer colocated test exists in current HEAD, extend that file instead and update this task before implementation; do not leave the frontend assertion implicit. Then add an exported type guard beside the existing frontend terminal taxonomy in `src/elspeth/web/frontend/src/types/index.ts`:

```ts
export function isTerminalRunStatus(status: RunStatus): status is TerminalRunStatus {
  return (TERMINAL_RUN_STATUS_VALUES as readonly RunStatus[]).includes(status);
}
```

Replace the local three-comparison predicate with that guard:

```ts
import { isTerminalRunStatus, type Session } from "@/types/index";

const hasActiveRun =
  !!activeRunId &&
  !!progress &&
  !isTerminalRunStatus(progress.status);
```

Do not add a second terminal-status set inside `SessionSidebar.tsx`; future status additions must flow through the central `TERMINAL_RUN_STATUS_VALUES` export. Use the type guard instead of calling `.includes(progress.status)` directly because `TERMINAL_RUN_STATUS_VALUES` is a terminal-only `as const` tuple while `progress.status` is the wider `RunStatus` type under strict TypeScript.
Replace the existing type-only `Session` import with the combined import shown
above so the component keeps the repo's `@/types/index` alias convention.

**Step 4: GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/execution/test_websocket.py tests/unit/web/execution/test_routes.py -v
.venv/bin/python -m pytest tests/unit/web/sessions tests/unit/web/execution/test_service.py -v
cd src/elspeth/web/frontend
npm run test -- SessionSidebar
npm run build
```

**Definition of Done:**
- [ ] `routes.py` uses `SESSION_TERMINAL_RUN_STATUS_VALUES` for both seed and idle terminal checks
- [ ] `web/sessions/service.py::update_run_status` stamps `finished_at` for every status in `SESSION_TERMINAL_RUN_STATUS_VALUES`
- [ ] `web/sessions/service.py::update_run_status` rejects `completed`, `completed_with_failures`, and `empty` when `landscape_run_id` would remain absent
- [ ] `web/execution/service.py::cancel` treats every status in `SESSION_TERMINAL_RUN_STATUS_VALUES` as idempotently terminal
- [ ] WebSocket tests cover `completed_with_failures` and `empty`
- [ ] SessionService tests prove `completed_with_failures` and `empty` persist with non-null `finished_at` and require `landscape_run_id`
- [ ] ExecutionService cancel tests prove `completed_with_failures` and `empty` are no-ops
- [ ] `src/elspeth/web/frontend/src/types/index.ts` exports `isTerminalRunStatus(status: RunStatus): status is TerminalRunStatus`
- [ ] `SessionSidebar` imports `isTerminalRunStatus` and treats `completed_with_failures` and `empty` as terminal
- [ ] `src/elspeth/web/frontend/src/components/sessions/SessionSidebar.test.tsx` (or the updated colocated replacement named in this task) covers all terminal statuses
- [ ] Frontend `npm run test -- SessionSidebar` and `npm run build` pass

---

### Task 3.6: Update resume aggregation (`core.py::_derive_resume_terminal_status_from_audit`)

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py:450-533`

**Step 1: Apply the edit**

The resume aggregator pattern-matches on `RowOutcome` and tallies per-counter. Flip to `(outcome, path)` matching with the same two BEHAVIOUR CHANGES applied to the **predicate-input counters** so resume status agrees with a live run. Per ADR-019 § Resume nuance, do **not** re-derive structural counters from token outcomes in this helper: `rows_diverted`, `rows_coalesced`, `rows_forked`, `rows_expanded`, and `rows_buffered` remain owned by the live run summary / existing audit surfaces until a later ADR explicitly makes them replayable from first principles.

```python
# OLD (lines 456-501):
for outcome in outcomes:
    if not outcome.is_terminal:
        continue
    match outcome.outcome:
        case RowOutcome.COMPLETED | RowOutcome.COALESCED | RowOutcome.DROPPED_BY_FILTER:
            rows_succeeded += 1
            rows_processed += 1
        case RowOutcome.ROUTED:
            rows_routed_success += 1
            rows_processed += 1
        # ... etc

# NEW:
for outcome_record in outcomes:
    if not outcome_record.completed:
        continue
    pair = (outcome_record.outcome, outcome_record.path)
    match pair:
        case (TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW) \
             | (TerminalOutcome.SUCCESS, TerminalPath.COALESCED) \
             | (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED):
            rows_succeeded += 1
            rows_processed += 1
        case (TerminalOutcome.SUCCESS, TerminalPath.GATE_ROUTED):
            # ADR-019 BEHAVIOUR CHANGE: bump BOTH rows_succeeded and rows_routed_success.
            rows_succeeded += 1
            rows_routed_success += 1
            rows_processed += 1
        case (TerminalOutcome.FAILURE, TerminalPath.ON_ERROR_ROUTED):
            # ADR-019 BEHAVIOUR CHANGE: bump BOTH rows_failed and rows_routed_failure.
            rows_failed += 1
            rows_routed_failure += 1
            rows_processed += 1
        case (TerminalOutcome.FAILURE, TerminalPath.UNROUTED):
            rows_failed += 1
            rows_processed += 1
        case (TerminalOutcome.FAILURE, TerminalPath.QUARANTINED_AT_SOURCE):
            rows_quarantined += 1
            rows_failed += 1
            rows_processed += 1
        case (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
            # ADR-019 § Sub-decision 5: discard-mode is FAILURE and therefore
            # a predicate input. Do not re-derive rows_diverted here; it is a
            # structural counter owned by the live sink-write path.
            rows_failed += 1
            rows_processed += 1
        case (TerminalOutcome.TRANSIENT, TerminalPath.SINK_FALLBACK_TO_FAILSINK):
            # Sink-write diversion reached a failsink. This is structural
            # diversion evidence, not a predicate input. Count it as processed
            # for resume completion/status, but do not re-derive rows_diverted.
            rows_processed += 1
        case (TerminalOutcome.TRANSIENT, TerminalPath.FORK_PARENT) \
             | (TerminalOutcome.TRANSIENT, TerminalPath.EXPAND_PARENT) \
             | (TerminalOutcome.TRANSIENT, TerminalPath.BATCH_CONSUMED):
            # Parent-token / batch bookkeeping is neutral in resume, matching
            # live-run accumulation: children / batch-result tokens carry the
            # row-level lifecycle answer.
            pass
        case _:
            raise AssertionError(
                f"Unhandled (outcome, path) pair in resume aggregation: {pair!r}. "
                f"Add a case here; see ADR-019 § Mapping table and the live "
                f"accumulator in engine/orchestrator/outcomes.py."
            )
```

**Step 2: Preserve the existing return shape**

`_derive_resume_terminal_status_from_audit` currently feeds the resume completion branch around `core.py:3250-3304`. Keep the return tuple scoped to replayable predicate/status counters:

```python
return (
    terminal_status,
    rows_processed,
    rows_succeeded,
    rows_failed,
    rows_routed_success,
    rows_routed_failure,
    rows_quarantined,
)
```

Do not add `rows_diverted` to the tuple in this ADR. A sink-fallback token outcome is `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)`: it proves a processed lifecycle checkpoint for resume status, but not enough by itself to reconstruct every structural sink-diversion count with the same authority as the live sink-write path.

**Step 3: Update imports**

Replace `RowOutcome` with `TerminalOutcome, TerminalPath` in the imports.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/test_resume.py -v` (or whichever resume test file exists).

Add explicit live-vs-resume parity tests for gate-routed, on-error-routed, quarantine, failsink-mode, and discard-mode runs. The parity assertions are for terminal status and predicate-input counters (`rows_processed`, `rows_succeeded`, `rows_failed`, `rows_routed_success`, `rows_routed_failure`, `rows_quarantined`). For sink-diversion cases, assert that structural `rows_diverted` is **not** re-derived by `_derive_resume_terminal_status_from_audit`; leave any existing live-run structural counter source untouched.

Create `tests/integration/test_adr_019_resume_counter_parity.py` with concrete
same-file helpers before the parameterized tests. Do not leave
`run_live_scenario(...)` or `run_interrupt_then_resume_scenario(...)` as
undefined placeholders.

Required helper contract:

```python
@dataclass(frozen=True, slots=True)
class ScenarioResult:
    result: RunResult
    db: LandscapeDB
    run_id: str


def _predicate_counter_tuple(result: RunResult) -> tuple[RunStatus, int, int, int, int, int, int]:
    return (
        result.status,
        result.rows_processed,
        result.rows_succeeded,
        result.rows_failed,
        result.rows_routed_success,
        result.rows_routed_failure,
        result.rows_quarantined,
    )


def _resume_counter_tuple_from_audit(db: LandscapeDB, run_id: str) -> tuple[RunStatus, int, int, int, int, int, int]:
    factory = RecorderFactory(db)
    status, processed, succeeded, failed, routed_success, routed_failure, quarantined = (
        Orchestrator._derive_resume_terminal_status_from_audit(factory, run_id)
    )
    return (status, processed, succeeded, failed, routed_success, routed_failure, quarantined)
```

`run_live_scenario(tmp_path, monkeypatch, scenario)` must build and run the
named scenario through `run_pipeline(...)`, returning `ScenarioResult`. It may
reuse Task 3.0 builders for `gate_routed_success`, `on_error_routed_failure`,
and `discard_mode_diversion`, but it must define local builders in this test
file for `quarantine_failure` and `failsink_mode_diversion` if Task 3.0 does not
create those helpers. The local builders still go through production
`instantiate_plugins_from_config()` and `ExecutionGraph.from_plugin_instances()`;
do not synthesize `RunResult` objects.

`run_interrupt_then_resume_scenario(tmp_path, monkeypatch, scenario)` must use
the real checkpoint/resume path: `CheckpointManager`, `RecoveryManager`,
`RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True,
frequency="every_row"))`, `Orchestrator.run(..., shutdown_event=...)`, and
`Orchestrator.resume(...)`. Reuse the `InterruptAfterN` / shutdown-event
construction pattern from
`tests/integration/pipeline/orchestrator/test_graceful_shutdown.py`; do not
invent a separate fake resume harness. If a scenario cannot be interrupted
deterministically through the existing helper plugins, define the smallest
test-only transform/source adapter in `_adr019_test_plugins.py` and register it
through the same plugin-manager patching path as the other ADR-019 helpers.

Then add the explicit RED-first cases:

```python
@pytest.mark.parametrize(
    "scenario",
    [
        "gate_routed_success",
        "on_error_routed_failure",
        "quarantine_failure",
        "failsink_mode_diversion",
        "discard_mode_diversion",
    ],
)
def test_resume_counter_shape_matches_live_predicate_counters(
    scenario: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-019 resume parity: replayed predicate counters match live counters."""
    live = run_live_scenario(tmp_path / "live", monkeypatch, scenario)
    resumed = run_interrupt_then_resume_scenario(tmp_path / "resume", monkeypatch, scenario)

    assert _predicate_counter_tuple(resumed.result) == _predicate_counter_tuple(live.result)
    assert _resume_counter_tuple_from_audit(live.db, live.run_id) == _predicate_counter_tuple(live.result)
```

For `failsink_mode_diversion` and `discard_mode_diversion`, add a second
assertion block that documents the structural-counter boundary:

```python
derived = Orchestrator._derive_resume_terminal_status_from_audit(factory, run_id)
assert len(derived) == 7  # tuple stays scoped to predicate/status counters
```

**Definition of Done:**
- [ ] Resume match block flipped to `(outcome, path)`
- [ ] Both BEHAVIOUR CHANGES applied symmetrically with the live accumulator
- [ ] Transient resume paths are explicit: `SINK_FALLBACK_TO_FAILSINK` increments
      `rows_processed` only, while `FORK_PARENT`, `EXPAND_PARENT`, and
      `BATCH_CONSUMED` remain neutral bookkeeping paths
- [ ] `(FAILURE, SINK_DISCARDED)` increments `rows_failed` and `rows_processed` only
- [ ] Resume completion branch return tuple is not extended with `rows_diverted`
- [ ] Live-vs-resume parity tests cover gate-routed, on-error-routed, quarantine, failsink-mode, and discard-mode predicate/status counters; structural `rows_diverted` is explicitly documented as not replayed here
- [ ] `tests/integration/test_adr_019_resume_counter_parity.py` runs through real interrupt/resume construction, not synthetic `RunResult` instances
- [ ] Resume tests pass
- [ ] mypy clean

---

### Task 3.7: Atomic Stage 2/3 commit (Phases 1-3)

**Step 1: Run the hard atomic Stage 2/3 gate**

```bash
.venv/bin/python -m pytest \
    tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
    tests/unit/core/landscape/test_database_compatibility_guards.py \
    tests/unit/contracts/ \
    tests/unit/core/landscape/test_data_flow_repository.py::TestRecordTokenOutcomeTwoAxis \
    tests/unit/core/landscape/test_model_loaders.py::TestTokenOutcomeLoaderTwoAxis \
    tests/unit/mcp/test_diagnose_quarantine_count.py \
    tests/unit/mcp/test_outcome_analysis.py \
    tests/unit/mcp/analyzers/test_reports.py \
    tests/unit/web/execution/test_diagnostics.py \
    tests/unit/web/execution/test_discard_summary.py \
    tests/unit/core/landscape/test_exporter.py \
    tests/unit/core/landscape/test_lineage.py \
    tests/unit/core/landscape/test_formatters.py \
    tests/unit/telemetry/ \
    tests/unit/engine/orchestrator/ \
    tests/unit/engine/test_sink_executor_diversion.py \
    tests/unit/engine/test_executors.py \
    tests/integration/plugins/sinks/test_durability.py \
    tests/integration/test_adr_019_discard_mode_flip.py \
    tests/integration/test_adr_019_counter_changes.py \
    tests/integration/test_adr_019_helpers.py \
    tests/integration/test_adr_019_resume_counter_parity.py \
    -q

.venv/bin/python - <<'PY'
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.processor import RowProcessor

print("adr019-stage23-import-smoke: OK", Orchestrator.__name__, RowProcessor.__name__)
PY

.venv/bin/python -m mypy src/elspeth
.venv/bin/python -m ruff check src/ tests/ scripts/
.venv/bin/python -m ruff format --check src/ tests/ scripts/
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model --exclude "**/__pycache__/*"
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
.venv/bin/python scripts/cicd/enforce_contract_manifest.py check --allowlist config/cicd/enforce_contract_manifest
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
.venv/bin/python scripts/cicd/enforce_frozen_annotations.py check --root src/elspeth --allowlist config/cicd/enforce_frozen_annotations
.venv/bin/python scripts/cicd/adr019_symbol_inventory.py check \
    --root src/elspeth \
    --allowlist config/cicd/adr019_symbol_inventory
.venv/bin/python scripts/cicd/forbid_new_row_outcome.py check --root . --allowlist config/cicd/forbid_new_row_outcome
cd src/elspeth/web/frontend
npm run test
npm run build
cd /home/john/elspeth
```

This focused gate is the mechanical enforcement that makes Phases 1-3 one
atomic execution unit. It proves every source/test surface intentionally touched
by Phases 1-3, the import/runtime smoke, frontend build, and policy checks. It
does **not** run `pytest tests/ -q`; Phase 5 owns the remaining repo-wide
schema-dependent/assertion-only/direct-DB-read test triage and is the first
full-suite gate. If any command here fails, do not commit; return to the phase
task that owns the failure and patch it before rerunning this whole gate.

**Step 2: Operator-visible RunStatus flip is now visible**

Both Task 3.1 and Task 3.2 integration tests are GREEN. Re-run them to confirm:

```bash
.venv/bin/python -m pytest \
    tests/integration/test_adr_019_discard_mode_flip.py \
    tests/integration/test_adr_019_counter_changes.py \
    -v
```

**Step 3: Commit Phases 1-3 together**

```bash
git add src/elspeth/contracts/ \
        src/elspeth/core/landscape/ \
		        src/elspeth/testing/ \
			    src/elspeth/mcp/ \
			    src/elspeth/telemetry/ \
			    src/elspeth/web/execution/ \
			    src/elspeth/web/sessions/ \
			    src/elspeth/web/frontend/ \
			    src/elspeth/engine/ \
			    src/elspeth/core/checkpoint/recovery.py \
			    docs/architecture/adr/019-two-axis-terminal-model.md \
			    docs/operator/migrations/adr-019.md \
			    config/cicd/forbid_new_row_outcome/migration_files.yaml \
			    config/cicd/adr019_symbol_inventory \
			    scripts/cicd/adr019_symbol_inventory.py \
			    tests/fixtures/cicd/adr019_symbol_inventory/ \
			    tests/fixtures/plugins.py \
			    tests/unit/scripts/cicd/test_adr019_symbol_inventory.py \
			    tests/unit/contracts/ \
		        tests/unit/core/landscape/ \
		        tests/unit/mcp/ \
		        tests/unit/telemetry/ \
		        tests/unit/web/execution/ \
	        tests/unit/web/sessions/ \
	        tests/unit/engine/orchestrator/ \
	        tests/unit/engine/test_sink_executor_diversion.py \
	        tests/unit/engine/test_executors.py \
	        tests/integration/plugins/sinks/test_durability.py \
	        tests/integration/test_adr_019_discard_mode_flip.py \
	        tests/integration/test_adr_019_counter_changes.py \
	        tests/integration/test_adr_019_helpers.py \
	        tests/integration/test_adr_019_resume_counter_parity.py \
	        tests/integration/_helpers.py \
	        tests/integration/_adr019_test_plugins.py

git commit -m "$(cat <<'EOF'
feat(adr-019): atomic stage 2-3 two-axis outcome migration

ADR-019 Stage 2/3 Phases 1-3 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).
This is the first legal commit boundary for the migration: Phase 1 and Phase 2
are local checkpoints only and are intentionally not buildable in isolation.

Schema + recorder + loader + consumer surface:
- token_outcomes: rename is_terminal → completed; add path column; outcome
  column nullability flipped to True (NULL means non-terminal/BUFFERED).
- TokenOutcome, RowResult, PendingOutcome, and TokenCompleted carry
  (outcome, path) instead of RowOutcome.
- record_token_outcome signature and TokenOutcomeLoader cross-checks use the
  new two-axis model.
- MCP, Web execution diagnostics, and core landscape exporters/formatters read
  completed/path instead of is_terminal-only schema.

Producer flip:
- processor.py, transform.py, sink.py, coalesce_executor.py, and recovery.py
  emit/read canonical (TerminalOutcome, TerminalPath) pairs.

Accumulator + predicate + resume:
- engine/orchestrator/outcomes.py::accumulate_row_outcomes pattern-matches on
  (outcome, path) and ships the two behaviour changes:
  (SUCCESS, GATE_ROUTED) bumps rows_succeeded and rows_routed_success;
  (FAILURE, ON_ERROR_ROUTED) bumps rows_failed and rows_routed_failure.
- contracts/run_result.py drops the bifurcated OR clauses from the terminal
  predicate while retaining routed/quarantine counters as guard-only subset
  inputs to derive_terminal_run_status.
- web/execution/schemas.py mirrors the L0 predicate and decomposition formula.
- _derive_resume_terminal_status_from_audit reads the new columns and produces
  the same counter shape as live runs.
- Source-quarantine routing in Orchestrator._handle_quarantine_row now bumps
  both rows_quarantined and rows_failed before routing to the quarantine sink.
- Sink discard-mode rows now contribute to rows_failed, so discard-only
  pipelines no longer report plain COMPLETED.

New RED-first tests cover discard-mode RunStatus flip, routed counter changes,
schema acceptance, and accumulator/predicate behaviour.

Refs: elspeth-edb60744f0 (Stage 3 ticket — producer + accumulator)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md § Counter derivation contract

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] Accumulator + predicate + resume aggregation flipped
- [ ] Live source-quarantine branch increments both rows_quarantined and rows_failed
- [ ] SinkExecutor counter increments updated for discard-mode
- [ ] Two RED-first integration tests now GREEN
- [ ] Focused Phases 1-3 pytest gate passes; full `pytest tests/ -q --timeout=120` remains owned by Phase 5
- [ ] Engine import/runtime smoke passes
- [ ] mypy, ruff check, ruff format, contracts, tier-model, plugin-hash, contract-manifest, freeze-guard, ADR-019 inventory, RowOutcome guard, and frontend gates pass
- [ ] Atomic Phases 1-3 commit landed
- [ ] Phase 4 starts in the next session/checkpoint
