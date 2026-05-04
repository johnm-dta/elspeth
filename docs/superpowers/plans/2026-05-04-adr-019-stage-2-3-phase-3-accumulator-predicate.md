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

- **Create: `tests/integration/_helpers.py`** (NEW — Task 3.0; canonical pipeline-builder factories + `run_pipeline` runner used by every Phase 3 RED test, mandated by CLAUDE.md test-path integrity to use `instantiate_plugins_from_config + ExecutionGraph.from_plugin_instances`)
- Modify: `src/elspeth/engine/orchestrator/outcomes.py:235-307` (accumulator)
- Modify: `src/elspeth/engine/orchestrator/core.py:450-533` (resume aggregation)
- Modify: `src/elspeth/contracts/run_result.py:60-152, 180-216` (L0 predicate + `derive_terminal_run_status`)
- **Modify: `src/elspeth/web/execution/schemas.py:138-160, 197-285` (L3 Pydantic predicate mirror — `_validate_row_decomposition` and `_check_status_row_count_invariant`).** Discovered 2026-05-05: the Pydantic schema duplicates the L0 predicate logic. Without this update, `/api/runs/{rid}` returns HTTP 500 for any valid run with gate-MOVE or transform-on-error routing because the post-Phase-3 counter doubling makes `rows_succeeded + rows_routed_success` exceed `rows_processed` in `_validate_row_decomposition`'s sum-disjoint formula.
- Test (RED-first): `tests/integration/test_adr_019_discard_mode_flip.py` (NEW)
- Test (RED-first): `tests/integration/test_adr_019_counter_changes.py` (NEW)
- Test (RED-first): `tests/integration/test_adr_019_api_runs_endpoint.py` (NEW — regression test for the `/api/runs/{rid}` 500)
- Test (unit): `tests/unit/engine/orchestrator/test_outcomes.py` (existing — extend with (outcome, path) match assertions)
- Test (unit): `tests/unit/contracts/test_run_result.py` (existing — flip predicate assertions)
- Test (unit): `tests/unit/web/execution/test_schemas.py` (existing — Pydantic predicate-mirror tests)

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

### Change 2: `(FAILURE, ON_ERROR_ROUTED)` becomes a `rows_failed` increment

Symmetric to Change 1 — the `rows_failed` counter now reflects transform `on_error` routings. This makes `failure_indicator = rows_failed > 0` exhaustive without needing `OR rows_routed_failure > 0`.

### Change 3: discard-mode `DIVERTED` increments `rows_failed`

This is the operator-visible `RunStatus` flip. A pipeline with a discard sink and otherwise clean rows now reports `RunStatus.COMPLETED_WITH_FAILURES` instead of `RunStatus.COMPLETED`.

The accumulator path is `(FAILURE, SINK_DISCARDED)`. Counter behaviour:

```python
elif (result.outcome, result.path) == (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
    counters.rows_failed += 1
    counters.rows_diverted += 1   # structural counter — unchanged from ADR-018
```

(Note: today's accumulator at `outcomes.py:279` raises `OrchestrationInvariantError` for `RowOutcome.DIVERTED` because diversions are recorded at sink-write time inside `SinkExecutor.write()` and counted at the orchestrator's sink-call site (`core.py:2199`: `loop_ctx.counters.rows_diverted += total_diversions`), not in the row-processing loop. The `(FAILURE, SINK_DISCARDED)` and `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` pairs similarly should NOT appear in `accumulate_row_outcomes` — they're emitted by the sink executor and counted at the orchestrator's sink-call site. The operator-visible discard-mode counter bump (`rows_failed`) lands at that orchestrator site too — see Step 3 below for the SinkExecutor return-shape change and the orchestrator counter-update site.)

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
    self.rows_failed > 0 or self.rows_quarantined > 0
    or self.rows_coalesce_failed > 0
)
```

Both OR clauses go away because Changes 1 and 2 above make `rows_succeeded` and `rows_failed` exhaustive. `rows_quarantined` and `rows_coalesce_failed` remain because they're separate failure signals not folded into `rows_failed` (per ADR-018's existing semantics, preserved by ADR-019).

---

## Tasks

### Task 3.0: Create canonical integration-test helpers

**Why this task exists:** Tasks 3.1, 3.2, and 3.5 all reference three pipeline-builder helpers (`build_test_pipeline_with_discard_sink`, `build_test_pipeline_with_gate_route`, `build_test_pipeline_with_on_error_route`) and a `run_pipeline` orchestrator-runner. **None of these exist in the codebase today** (verified 2026-05-05: `grep -rn "build_test_pipeline_with_" src/elspeth/testing/ tests/` returns empty). They must be created in a single canonical location BEFORE any of the Phase 3 RED tests can be written, otherwise the tests are non-runnable from the moment they land.

**Canonical location:** `tests/integration/_helpers.py`. NOT `src/elspeth/testing/` — that pack hosts ChaosEngine fixtures shipped to ELSPETH consumers, not integration-test pipeline builders. NOT `tests/conftest.py` — that's pytest fixture machinery, not pipeline-config factories.

**CLAUDE.md test-path integrity (load-bearing):** the helpers MUST construct pipelines via `instantiate_plugins_from_config()` + `ExecutionGraph.from_plugin_instances()` from `cli_helpers.py:43` and `engine/__init__.py:25`. Per CLAUDE.md "Critical Implementation Patterns": *"Never bypass production code paths in tests — integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`"*. Bypassing these would let Phase 3's behaviour-change tests pass against a fake codepath that isn't exercised in production — defeating the whole point of the RED-first verification.

**Files:**
- Create: `tests/integration/_helpers.py`

**Step 1: Create the helper module**

```python
"""Integration-test helpers for ADR-019 behaviour-change verification.

Three pipeline factories that produce minimal end-to-end pipeline configs
plus a single ``run_pipeline()`` runner that goes through the production
code path: instantiate_plugins_from_config + ExecutionGraph.from_plugin_instances
+ Orchestrator. Per CLAUDE.md Critical Implementation Patterns: integration
tests MUST exercise the real code path, never a bypass.

The pipelines produced here are deliberately minimal — JSON source, one
transform, one or two sinks. Just enough surface to exercise the producer
emit sites for (SUCCESS, GATE_ROUTED), (FAILURE, ON_ERROR_ROUTED), and
(FAILURE, SINK_DISCARDED) in the accumulator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts.run_result import RunResult
from elspeth.core.config import load_settings
from elspeth.core.dag import ExecutionGraph
from elspeth.engine import Orchestrator, PipelineConfig


@dataclass(frozen=True)
class TestPipelineSpec:
    """A self-contained pipeline configuration ready to feed run_pipeline()."""

    config_path: Path
    audit_db_url: str


def _write_input(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(rows))
    return input_file


def _write_settings(tmp_path: Path, settings_dict: dict[str, Any]) -> Path:
    """Write a settings.yaml suitable for load_settings() to consume."""
    import yaml

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(settings_dict))
    return settings_path


def build_test_pipeline_with_discard_sink(
    tmp_path: Path,
    *,
    success_row_count: int,
    discard_row_count: int,
) -> TestPipelineSpec:
    """Pipeline with one transform whose ``on_error: discard`` causes
    ``discard_row_count`` rows to hit the discard sink path
    (sink_name='__discard__'). The remaining ``success_row_count`` rows
    flow through to the primary success sink as (SUCCESS, DEFAULT_FLOW)."""
    rows = [
        {"id": i, "should_fail": False} for i in range(success_row_count)
    ] + [
        {"id": success_row_count + i, "should_fail": True}
        for i in range(discard_row_count)
    ]
    input_path = _write_input(tmp_path, rows)
    output_path = tmp_path / "output.json"
    audit_db_url = f"sqlite:///{tmp_path / 'audit.db'}"

    settings = {
        "source": {
            "plugin": "json",
            "on_success": "fail_if_flagged",
            "options": {
                "path": str(input_path),
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "should_fail: bool"],
                },
            },
        },
        "transforms": [
            {
                "name": "fail_if_flagged",
                "plugin": "raise_on_field",  # Stage-1-shipped test plugin or equivalent
                "input": "fail_if_flagged",
                "on_success": "output",
                "on_error": "discard",
                "options": {"field": "should_fail", "raise_when": True},
            },
        ],
        "sinks": {
            "output": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {"path": str(output_path)},
            },
        },
        "landscape": {"url": audit_db_url},
    }
    config_path = _write_settings(tmp_path, settings)
    return TestPipelineSpec(config_path=config_path, audit_db_url=audit_db_url)


def build_test_pipeline_with_gate_route(
    tmp_path: Path,
    *,
    routed_row_count: int,
    default_flow_row_count: int,
) -> TestPipelineSpec:
    """Pipeline with a gate that MOVE-routes ``routed_row_count`` rows to
    a secondary sink, and lets ``default_flow_row_count`` rows pass through
    to the primary sink as (SUCCESS, DEFAULT_FLOW). The MOVE-routed rows
    land as (SUCCESS, GATE_ROUTED) and exercise the Phase 3 counter doubling.
    """
    rows = [
        {"id": i, "route": "move"} for i in range(routed_row_count)
    ] + [
        {"id": routed_row_count + i, "route": "default"}
        for i in range(default_flow_row_count)
    ]
    input_path = _write_input(tmp_path, rows)
    primary_output = tmp_path / "primary.json"
    routed_output = tmp_path / "routed.json"
    audit_db_url = f"sqlite:///{tmp_path / 'audit.db'}"

    settings = {
        "source": {
            "plugin": "json",
            "on_success": "route_gate",
            "options": {"path": str(input_path), "schema": {"mode": "observed"}},
        },
        "gates": [
            {
                "name": "route_gate",
                "plugin": "expression_gate",  # canonical gate; replace if pack differs
                "input": "route_gate",
                "on_success": "primary",
                "options": {
                    "rules": [
                        {
                            "when": "row.route == 'move'",
                            "action": "route_to_sink",
                            "destination": "routed",
                        },
                    ],
                },
            },
        ],
        "sinks": {
            "primary": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {"path": str(primary_output)},
            },
            "routed": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {"path": str(routed_output)},
            },
        },
        "landscape": {"url": audit_db_url},
    }
    config_path = _write_settings(tmp_path, settings)
    return TestPipelineSpec(config_path=config_path, audit_db_url=audit_db_url)


def build_test_pipeline_with_on_error_route(
    tmp_path: Path,
    *,
    on_error_routed_count: int,
    success_count: int,
) -> TestPipelineSpec:
    """Pipeline with a transform whose ``on_error: error_sink`` routes
    failing rows to a named error sink. ``on_error_routed_count`` rows
    raise inside the transform and land as (FAILURE, ON_ERROR_ROUTED);
    ``success_count`` rows succeed as (SUCCESS, DEFAULT_FLOW)."""
    rows = [
        {"id": i, "should_fail": True} for i in range(on_error_routed_count)
    ] + [
        {"id": on_error_routed_count + i, "should_fail": False}
        for i in range(success_count)
    ]
    input_path = _write_input(tmp_path, rows)
    primary_output = tmp_path / "primary.json"
    error_output = tmp_path / "errors.json"
    audit_db_url = f"sqlite:///{tmp_path / 'audit.db'}"

    settings = {
        "source": {
            "plugin": "json",
            "on_success": "fail_if_flagged",
            "options": {"path": str(input_path), "schema": {"mode": "observed"}},
        },
        "transforms": [
            {
                "name": "fail_if_flagged",
                "plugin": "raise_on_field",
                "input": "fail_if_flagged",
                "on_success": "primary",
                "on_error": "error_sink",
                "options": {"field": "should_fail", "raise_when": True},
            },
        ],
        "sinks": {
            "primary": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {"path": str(primary_output)},
            },
            "error_sink": {
                "plugin": "json",
                "on_write_failure": "discard",
                "options": {"path": str(error_output)},
            },
        },
        "landscape": {"url": audit_db_url},
    }
    config_path = _write_settings(tmp_path, settings)
    return TestPipelineSpec(config_path=config_path, audit_db_url=audit_db_url)


def run_pipeline(spec: TestPipelineSpec) -> RunResult:
    """Run the pipeline through the production code path.

    Mandated by CLAUDE.md "Critical Implementation Patterns": integration
    tests must use ``instantiate_plugins_from_config()`` +
    ``ExecutionGraph.from_plugin_instances()`` so the test exercises the
    same wiring the CLI does. Bypassing these (e.g., constructing
    PipelineConfig manually with mocked plugins) would let behaviour-change
    tests pass against a fake codepath.
    """
    settings = load_settings(spec.config_path)
    config = PipelineConfig.from_settings(settings)
    plugins = instantiate_plugins_from_config(config)
    graph = ExecutionGraph.from_plugin_instances(config=config, plugins=plugins)
    orchestrator = Orchestrator(config=config, graph=graph)
    return orchestrator.execute()
```

**Plugin-name caveats (verify before relying on the helper):**

The pipeline configs above name three plugins (`json` source/sink, `raise_on_field` transform, `expression_gate`). Verify each is registered in the codebase before writing the tests:

```bash
.venv/bin/python -c "from elspeth.plugins import list_plugins; print('\\n'.join(sorted(list_plugins())))" | grep -E "^(json|raise_on_field|expression_gate)$"
```

If a name differs (e.g., `json_source` / `json_sink` are separate, or `raise_on_field` doesn't exist as a test plugin), substitute the registered name. The Phase 3 RED tests assume the helper produces VALID pipeline configs that pass `instantiate_plugins_from_config()` — bad plugin names would fail there, masking the actual behaviour-change assertion.

If `raise_on_field` doesn't exist as a registered plugin, add it under `src/elspeth/testing/` as a small test-only transform that raises on a configured field-equals condition. This is one-off scaffolding for the ADR-019 RED tests; document the addition in the Phase 5 test-strategy doc.

**Step 2: Verify the helper module imports cleanly**

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

Expected: `OK`. (Note: `tests/integration/_helpers.py` must be importable from the test runner's working directory; `pytest` does this by default if `tests/` is on `sys.path` per `pyproject.toml` configuration.)

**Definition of Done:**
- [ ] `tests/integration/_helpers.py` created with all four helper functions
- [ ] All three pipeline builders use `instantiate_plugins_from_config + ExecutionGraph.from_plugin_instances` per CLAUDE.md test-path integrity (no bypassed code paths)
- [ ] Plugin names verified against the project's registered plugin set
- [ ] `from tests.integration._helpers import ...` works for all four names
- [ ] mypy clean on the helper module

---

### Task 3.1: Write the discard-mode behaviour-change RED test

**Files:**
- Create: `tests/integration/test_adr_019_discard_mode_flip.py`

**Step 1: Write the failing test**

```python
"""ADR-019 § Behavior Change Notice: discard-mode DIVERTED flips RunStatus.

A pipeline with a discard sink ("__discard__") and otherwise clean rows
should report RunStatus.COMPLETED_WITH_FAILURES (not COMPLETED) because
discard-mode is reclassified as (FAILURE, SINK_DISCARDED) and increments
rows_failed.

This test is RED before Phase 3's accumulator/predicate change lands and
GREEN after. It is the operator-visible assertion that pins the change.
"""

import pytest

from elspeth.contracts.enums import RunStatus
from tests.integration._helpers import build_test_pipeline_with_discard_sink, run_pipeline


@pytest.fixture
def pipeline_with_discard_and_success_sink(tmp_path):
    """A pipeline where:
    - 3 rows succeed via the primary sink (DEFAULT_FLOW)
    - 2 rows divert to "__discard__" via a transform's on_error config
    - No quarantines, no upstream failures
    """
    return build_test_pipeline_with_discard_sink(
        tmp_path=tmp_path,
        success_row_count=3,
        discard_row_count=2,
    )


class TestDiscardModeRunStatusFlip:
    def test_discard_with_some_success_yields_completed_with_failures(
        self, pipeline_with_discard_and_success_sink
    ) -> None:
        """ADR-019 § Sub-decision 5: discard-mode is FAILURE, predicate-input."""
        result = run_pipeline(pipeline_with_discard_and_success_sink)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES, (
            f"Expected COMPLETED_WITH_FAILURES (3 success + 2 discard), "
            f"got {result.status}. The discard rows should bump rows_failed "
            f"per ADR-019 § Sub-decision 5."
        )
        assert result.rows_succeeded == 3
        assert result.rows_failed == 2  # NEW: discards now count as failures
        assert result.rows_diverted == 2  # structural counter unchanged

    def test_all_discards_yields_failed(self, tmp_path) -> None:
        """ADR-019 § Behavior Change Notice: pipelines that ONLY discard yield FAILED."""
        pipeline = build_test_pipeline_with_discard_sink(
            tmp_path=tmp_path,
            success_row_count=0,
            discard_row_count=3,
        )
        result = run_pipeline(pipeline)
        assert result.status == RunStatus.FAILED
        assert result.rows_succeeded == 0
        assert result.rows_failed == 3
```

`build_test_pipeline_with_discard_sink` is the helper created in Task 3.0 at the canonical location `tests/integration/_helpers.py`. It builds a minimal pipeline (JSON source, a transform that raises on a configured field, primary success sink + discard sink) using `instantiate_plugins_from_config` + `ExecutionGraph.from_plugin_instances` per CLAUDE.md test-path integrity.

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_discard_mode_flip.py -v
```

Expected: BOTH tests fail. The first fails because `result.status == RunStatus.COMPLETED` (under current behaviour). The second fails because the run completes as `COMPLETED` or `EMPTY` rather than `FAILED`.

**Step 3: Confirm the test fixture builds correctly even before the predicate change**

```python
# Quick sanity check: the test pipeline runs without errors and produces a RunResult.
.venv/bin/python -c "
from tests.integration._helpers import build_test_pipeline_with_discard_sink, run_pipeline
import tempfile, pathlib
with tempfile.TemporaryDirectory() as t:
    pipeline = build_test_pipeline_with_discard_sink(pathlib.Path(t), 3, 2)
    result = run_pipeline(pipeline)
    print(result.status, result.rows_succeeded, result.rows_failed)
"
```

Expected pre-Phase-3: `RunStatus.COMPLETED 3 0`. Post-Phase-3: `RunStatus.COMPLETED_WITH_FAILURES 3 2`.

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
        self, tmp_path
    ) -> None:
        pipeline = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            routed_row_count=4,
            default_flow_row_count=0,  # ALL rows get gate-routed
        )
        result = run_pipeline(pipeline)

        assert result.status == RunStatus.COMPLETED, (
            f"With 4 gate-routed rows and 0 failures, expected COMPLETED, got {result.status}"
        )
        # NEW: rows_succeeded reflects the routed rows
        assert result.rows_succeeded == 4
        assert result.rows_routed_success == 4

    def test_on_error_routed_bumps_both_failed_and_routed_failure(
        self, tmp_path
    ) -> None:
        pipeline = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            on_error_routed_count=3,
            success_count=2,
        )
        result = run_pipeline(pipeline)

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_failed == 3       # NEW: on-error routes count as failures
        assert result.rows_routed_failure == 3
        assert result.rows_succeeded == 2
```

**Step 2: Run RED**

Expected: both tests fail because under current code, `rows_succeeded == 0` (all rows are gate-routed) and `result.status == RunStatus.COMPLETED` only via the `OR rows_routed_success > 0` predicate clause.

**Definition of Done:**
- [ ] Test file exists with both counter-doubling assertions
- [ ] Tests fail before Phase 3 lands
- [ ] Fixture helpers exist

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
    elif pair == (TerminalOutcome.SUCCESS, TerminalPath.FILTER_DROPPED):
        counters.rows_succeeded += 1
    elif pair == (TerminalOutcome.SUCCESS, TerminalPath.COALESCED):
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

**Step 3: Update diversion-counter accumulation at the orchestrator (NOT inside SinkExecutor)**

**Reality check:** `SinkExecutor` has no `_counters` attribute — counter accumulation happens at the orchestrator level. `SinkExecutor.write()` (`sink.py:341`) returns `tuple[Artifact | None, int]`; the int is `diversion_count`. `Orchestrator._run_main_processing_loop` at `core.py:2199` accumulates it via `loop_ctx.counters.rows_diverted += total_diversions`. The discard-mode `rows_failed` bump must live at that orchestrator site, AND `SinkExecutor.write()` must return enough information for the orchestrator to know how many of the diversions were discard-mode versus failsink-mode (the orchestrator can no longer treat `total_diversions` as opaque).

**Sub-step 3a: Change the SinkExecutor.write() return shape to distinguish diversion flavors**

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

**Sub-step 3b: Update orchestrator to accumulate the split counts**

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

The local variable that today holds `total_diversions` (returned from the sink-write call site) becomes `diversion_counts: DiversionCounts`. Both call sites in `core.py` that consume the return value need updating — find them via `grep -n "rows_diverted\\|sink_executor.*write\\|\\.write(" src/elspeth/engine/orchestrator/core.py` and update each.

**Sub-step 3c: Update any other callers of `SinkExecutor.write()`**

```bash
grep -rn "sink_executor\.write\|SinkExecutor.*write\|self\._sink_executor\.write" src/elspeth/
```

Each caller that destructured `(artifact, diversion_count)` becomes `(artifact, diversion_counts)` and uses the dataclass fields. Test fixtures that assert on the return shape need fixture updates in this commit (schema-dependent per Phase 5 triage).

**Step 4: Update imports in `outcomes.py`**

Replace `from elspeth.contracts.enums import RowOutcome` with `from elspeth.contracts.enums import TerminalOutcome, TerminalPath`.

**Step 5: Verify**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/test_outcomes.py -v`

Expected: existing accumulator unit tests pass. The integration tests from Tasks 3.1 and 3.2 should now go GREEN.

**Definition of Done:**
- [ ] Accumulator pattern-matches on `(outcome, path)`
- [ ] GATE_ROUTED and ON_ERROR_ROUTED branches bump both counters
- [ ] `_route_to_sink` signature updated
- [ ] `DiversionCounts` dataclass added in `sink.py`; `SinkExecutor.write()` return shape changed from `tuple[Artifact | None, int]` to `tuple[Artifact | None, DiversionCounts]`
- [ ] All `SinkExecutor.write()` call sites updated to destructure the dataclass (verified by `grep -rn "sink_executor\.write\|SinkExecutor.*write" src/elspeth/`)
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
    self.rows_failed > 0 or self.rows_quarantined > 0 or self.rows_coalesce_failed > 0
)
```

Update the error-message strings in the `case` arms below to match the simplified predicate. Most messages reference `rows_succeeded > 0 or rows_routed_success > 0`; trim to `rows_succeeded > 0`.

```python
# Same edit for derive_terminal_run_status (lines 208-216):
success_indicator = rows_succeeded > 0
failure_indicator = rows_failed > 0 or rows_quarantined > 0 or rows_coalesce_failed > 0
```

**Step 3: Drop unused parameters**

`derive_terminal_run_status` takes `rows_routed_success` and `rows_routed_failure` parameters that are no longer read by the predicate. They remain in the function signature because the resume aggregator + caller still pass them as positional args, but mark them with comments that they are no longer load-bearing for the predicate (they remain useful for the structural-counter view in `RunResult`'s output).

Actually — clean approach: drop them from the predicate function signature entirely. Audit the callers and confirm none rely on the parameter for predicate purposes. Update each caller to omit the unused arguments.

**Step 4: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py tests/integration/test_adr_019_*.py -v`

Expected: all tests pass.

**Definition of Done:**
- [ ] `__post_init__` predicate simplified
- [ ] `derive_terminal_run_status` predicate simplified
- [ ] Error-message strings updated to match new predicate
- [ ] Unused parameters dropped (or callers updated)
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
- Test: `tests/integration/test_adr_019_api_runs_endpoint.py` (NEW)

**Step 1: Write the RED-first regression test for the 500**

Create: `tests/integration/test_adr_019_api_runs_endpoint.py`

```python
"""ADR-019 B4 regression: /api/runs/{rid} returns 200 (not 500) for runs
that exercise gate-MOVE routing or transform on_error routing.

Pre-fix bug: web/execution/schemas.py::_validate_row_decomposition asserts a
sum-disjoint inequality that breaks after Phase 3's accumulator counter
doubling. A run with N gate-routed rows has rows_succeeded=N AND
rows_routed_success=N; the formula's right-hand sum exceeds rows_processed
and fires ValueError, surfacing as HTTP 500.
"""

from fastapi.testclient import TestClient

from elspeth.contracts.enums import RunStatus
from tests.integration._helpers import (
    build_test_pipeline_with_gate_route,
    build_test_pipeline_with_on_error_route,
    run_pipeline,
)
from elspeth.web.app import app  # the FastAPI app


class TestAdr019ApiRunsEndpoint:
    def test_gate_routed_run_returns_200_not_500(self, tmp_path) -> None:
        """A run with 4 gate-routed rows produces a valid CompletedData payload."""
        pipeline = build_test_pipeline_with_gate_route(
            tmp_path=tmp_path,
            routed_row_count=4,
            default_flow_row_count=0,
        )
        run_id = run_pipeline(pipeline).run_id

        client = TestClient(app)
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200, (
            f"Expected 200 OK, got {response.status_code}. "
            f"Body: {response.json()}. The Pydantic CompletedData validator "
            f"must accept Phase-3 counter shapes where rows_succeeded and "
            f"rows_routed_success both reflect gate-routed rows."
        )
        body = response.json()
        assert body["status"] == "completed"
        assert body["rows_succeeded"] == 4
        assert body["rows_routed_success"] == 4

    def test_on_error_routed_run_returns_200_not_500(self, tmp_path) -> None:
        """Symmetric: transform on_error routing produces valid response."""
        pipeline = build_test_pipeline_with_on_error_route(
            tmp_path=tmp_path,
            on_error_routed_count=3,
            success_count=2,
        )
        run_id = run_pipeline(pipeline).run_id

        client = TestClient(app)
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed_with_failures"
        assert body["rows_failed"] == 3
        assert body["rows_routed_failure"] == 3
```

**Step 2: Run RED**

```bash
.venv/bin/python -m pytest tests/integration/test_adr_019_api_runs_endpoint.py -v
```

Expected: BOTH tests fail with HTTP 500 because `_validate_row_decomposition` rejects the post-Phase-3 counter shape.

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

# NEW:
def _validate_row_decomposition(
    rows_processed: int,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_success: int,
    rows_routed_failure: int,
    rows_quarantined: int,
) -> None:
    """Enforce rows_processed >= rows_succeeded + rows_failed + rows_quarantined.

    ADR-019 § Counter derivation contract: under the two-axis terminal model,
    ``rows_succeeded`` is exhaustive for SUCCESS lifecycles (DEFAULT_FLOW,
    GATE_ROUTED, FILTER_DROPPED, COALESCED) and ``rows_failed`` is exhaustive
    for FAILURE lifecycles other than QUARANTINED_AT_SOURCE (UNROUTED,
    ON_ERROR_ROUTED, SINK_DISCARDED). ``rows_quarantined`` covers the third
    failure bucket (QUARANTINED_AT_SOURCE). The structural counters
    ``rows_routed_success`` and ``rows_routed_failure`` are non-disjoint
    SUBSETS of ``rows_succeeded`` / ``rows_failed`` — counting them in a
    decomposition sum double-counts gate-routed and on-error-routed rows.

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
    sum_terminal = rows_succeeded + rows_failed + rows_quarantined
    if rows_processed < sum_terminal:
        raise ValueError(
            f"Row count decomposition mismatch (over-counting): rows_processed="
            f"{rows_processed} < rows_succeeded({rows_succeeded}) + "
            f"rows_failed({rows_failed}) + rows_quarantined({rows_quarantined}) "
            f"= {sum_terminal}. Tier 1 anomaly: orchestrator emitted more "
            f"predicate-bucket counts than input rows. ADR-019 § Counter "
            f"derivation contract: rows_routed_success and rows_routed_failure "
            f"are non-disjoint subsets and excluded from the decomposition sum."
        )
```

**Note:** The function signature still accepts `rows_routed_success` and `rows_routed_failure` because callers pass them positionally. Inside the function they're now unused. Either:
- **(a)** Drop the parameters from the signature and update all four call sites (lines 308, 621, 665, plus the one I haven't seen yet).
- **(b)** Keep the parameters as accepted-but-ignored; mark with `_` prefix or a docstring note.

Prefer (a) — clean signature change matches the project's "no legacy code" policy. The grep `grep -rn "_validate_row_decomposition(" src/elspeth/` will surface all callers; update each.

**Step 4: Patch `_check_status_row_count_invariant` (line ~218)**

The functional check is fine (the bifurcated OR clauses remain logically satisfied post-Phase-3) but the error messages reference dropped concepts. Update to match the L0 predicate's simplified shape:

```python
# OLD line 218:
success_indicator = rows_succeeded > 0 or rows_routed_success > 0
failure_indicator = rows_failed > 0 or rows_quarantined > 0 or rows_routed_failure > 0

# NEW:
# ADR-019 § Counter derivation contract: rows_succeeded is exhaustive for
# SUCCESS lifecycles; rows_failed for non-quarantine FAILURE lifecycles.
# The bifurcated OR clauses become vestigial.
success_indicator = rows_succeeded > 0
failure_indicator = rows_failed > 0 or rows_quarantined > 0
```

Update the four error-message strings (lines ~248, ~258, ~273, ~282) to drop the `or rows_routed_success` / `or rows_routed_failure` references. The function signature still receives those parameters because callers pass them, but they're no longer used in the predicate.

**Step 5: Update callers of `_validate_row_decomposition`**

```bash
grep -n "_validate_row_decomposition(" src/elspeth/web/execution/schemas.py
```

Each call site (CompletedData, CancelledData, RunStatusResponse, ProgressData if applicable) drops the two `rows_routed_*` arguments. The `_check_status_row_count_invariant` function does the same parameter cleanup OR keeps the params as accepted-but-ignored — same choice as Step 3.

**Step 6: GREEN**

```bash
.venv/bin/python -m pytest \
  tests/integration/test_adr_019_api_runs_endpoint.py \
  tests/integration/test_adr_019_counter_changes.py \
  tests/integration/test_adr_019_discard_mode_flip.py \
  tests/unit/web/execution/test_schemas.py \
  -v
```

Expected: all green. Any pre-existing tests in `test_schemas.py` that asserted the old decomposition-sum shape need fixture updates in this commit — they're schema-dependent.

**Definition of Done:**
- [ ] `_validate_row_decomposition` formula simplified — drops `rows_routed_*` from the sum
- [ ] `_check_status_row_count_invariant` predicate aligned with L0 (drops bifurcated OR)
- [ ] Error messages updated to match the new predicate shape
- [ ] All `_validate_row_decomposition` callers updated (or parameters preserved-and-ignored consistently)
- [ ] B4 regression test (`/api/runs/{rid}` returns 200 for gate-routed and on_error-routed runs) passes
- [ ] Pre-existing schemas tests pass after fixture updates
- [ ] mypy clean across `src/elspeth/web/execution/`

---

### Task 3.6: Update resume aggregation (`core.py::_derive_resume_terminal_status_from_audit`)

**Files:**
- Modify: `src/elspeth/engine/orchestrator/core.py:450-533`

**Step 1: Apply the edit**

The resume aggregator pattern-matches on `RowOutcome` and tallies per-counter. Flip to `(outcome, path)` matching with the same two BEHAVIOUR CHANGES applied (so resume produces the same counter shape as a live run).

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
            rows_processed += 1
        case (TerminalOutcome.FAILURE, TerminalPath.SINK_DISCARDED):
            # ADR-019 § Sub-decision 5: discard-mode is FAILURE.
            rows_failed += 1
            rows_processed += 1
        case (TerminalOutcome.TRANSIENT, _):
            # Parent-token / batch-absorption / failsink-fallback bookkeeping —
            # children/batch-result tokens carry the lifecycle answer.
            rows_processed += 1
        case _:
            raise AssertionError(
                f"Unhandled (outcome, path) pair in resume aggregation: {pair!r}. "
                f"Add a case here; see ADR-019 § Mapping table and the live "
                f"accumulator in engine/orchestrator/outcomes.py."
            )
```

**Step 2: Update imports**

Replace `RowOutcome` with `TerminalOutcome, TerminalPath` in the imports.

**Step 3: GREEN**

Run: `.venv/bin/python -m pytest tests/unit/engine/orchestrator/test_resume.py -v` (or whichever resume test file exists).

**Definition of Done:**
- [ ] Resume match block flipped to `(outcome, path)`
- [ ] Both BEHAVIOUR CHANGES applied symmetrically with the live accumulator
- [ ] Resume tests pass
- [ ] mypy clean

---

### Task 3.7: Phase 3 commit

**Step 1: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -q --timeout=120
```

Most tests pass. Some unit tests for the accumulator and predicate may have stale RowOutcome assertions that survived Phase 1's contract-test updates — those are now fully retyped in this phase. Update each in the same commit.

**Step 2: Operator-visible RunStatus flip is now visible**

Both Task 3.1 and Task 3.2 integration tests are GREEN. Re-run them to confirm:

```bash
.venv/bin/python -m pytest \
    tests/integration/test_adr_019_discard_mode_flip.py \
    tests/integration/test_adr_019_counter_changes.py \
    -v
```

**Step 3: Commit**

```bash
git add src/elspeth/engine/orchestrator/outcomes.py \
        src/elspeth/engine/orchestrator/core.py \
        src/elspeth/engine/executors/sink.py \
        src/elspeth/contracts/run_result.py \
        src/elspeth/web/execution/schemas.py \
        tests/integration/test_adr_019_discard_mode_flip.py \
        tests/integration/test_adr_019_counter_changes.py \
        tests/integration/test_adr_019_api_runs_endpoint.py \
        tests/integration/_helpers.py \
        tests/unit/contracts/test_run_result.py \
        tests/unit/web/execution/test_schemas.py \
        tests/unit/engine/orchestrator/

git commit -m "$(cat <<'EOF'
feat(adr-019): phase 3 — accumulator + L0 predicate + L3 Pydantic mirror + resume + behaviour changes

ADR-019 Stage 2/3 Phase 3 of 5 (see docs/superpowers/plans/2026-05-04-adr-019-stage-2-3-overview.md).

Four coupled changes ship together per ADR-019 § Counter derivation contract:

1. Accumulator (engine/orchestrator/outcomes.py::accumulate_row_outcomes):
   pattern-matches on (outcome, path); ships the two BEHAVIOUR CHANGES:
   - (SUCCESS, GATE_ROUTED) bumps BOTH rows_succeeded AND rows_routed_success
   - (FAILURE, ON_ERROR_ROUTED) bumps BOTH rows_failed AND rows_routed_failure

2. L0 predicate (contracts/run_result.py::__post_init__ + derive_terminal_run_status):
   bifurcated OR clauses removed. success_indicator = rows_succeeded > 0;
   failure_indicator = rows_failed > 0 or rows_quarantined > 0 or
   rows_coalesce_failed > 0 (drops the rows_routed_failure clause).

3. L3 Pydantic mirror (web/execution/schemas.py::_validate_row_decomposition +
   _check_status_row_count_invariant): the decomposition formula drops the
   rows_routed_* terms (now non-disjoint subsets of rows_succeeded /
   rows_failed); the predicate mirror aligns with L0. Without this update,
   /api/runs/{rid} returns HTTP 500 for any run with gate-MOVE or
   transform-on-error routing because the post-Phase-3 counter doubling
   makes rows_succeeded + rows_routed_success exceed rows_processed in the
   Pydantic decomposition check.

4. Resume aggregation (engine/orchestrator/core.py::_derive_resume_terminal_status_from_audit):
   pattern-matches on (outcome, path) with the same two behaviour changes
   applied; resume produces the same counter shape as a live run.

Plus the SinkExecutor counter increment for the discard-mode flip:
sink.py adds rows_failed += 1 alongside the existing rows_diverted += 1
for (FAILURE, SINK_DISCARDED). Operator-visible: a pipeline with discards
and otherwise clean rows now reports COMPLETED_WITH_FAILURES instead of
COMPLETED.

New integration tests pin each behaviour change as RED-first → GREEN-after:
- tests/integration/test_adr_019_discard_mode_flip.py
- tests/integration/test_adr_019_counter_changes.py
- tests/integration/test_adr_019_api_runs_endpoint.py (B4 regression — confirms
  /api/runs/{rid} returns 200 for gate-routed and on_error-routed runs)

Refs: elspeth-edb60744f0 (Stage 3 ticket — producer + accumulator)
ADR: docs/architecture/adr/019-two-axis-terminal-model.md § Counter derivation contract

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

**Definition of Done:**
- [ ] Accumulator + predicate + resume aggregation flipped
- [ ] SinkExecutor counter increments updated for discard-mode
- [ ] Two RED-first integration tests now GREEN
- [ ] All unit tests pass after fixture updates
- [ ] mypy clean
- [ ] Phase 3 commit landed
- [ ] Phase 4 starts in the next session/checkpoint
