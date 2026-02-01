# Contract Boundary Consolidation Implementation Plan

**Status:** ✅ IMPLEMENTED (2026-01-31)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move cross-boundary types to `contracts/` and use existing TypedDicts in Landscape recorder, eliminating whitelist workarounds and strengthening type safety at subsystem boundaries.

**Architecture:** Types that cross subsystem boundaries (engine↔plugins, engine↔telemetry) must live in `contracts/`. The whitelist was created as a workaround; this plan fixes the root cause by moving 4 types to their proper locations and updating Landscape recorder signatures to use existing `RoutingReason` TypedDict.

**Tech Stack:** Python dataclasses, TypedDict, mypy for verification

**Review Status:** ✅ Reviewed 2026-01-30
- Critical gap fixed: Added Step 6e to update `telemetry/__init__.py` (prevents ImportError)
- Step 6a clarified: Extend existing import instead of adding new line
- Task 6 scope note added: Documents partial type safety limitation
- Summary table updated to reflect all affected files

**Execution Status:** ✅ Completed 2026-01-31
- All 7 tasks executed successfully using subagent-driven development
- CI validation passed (mypy, ruff, pre-commit hooks)
- Follow-up work tracked in bead `elspeth-rapid-m76` for eliminating remaining `cast()` calls

---

## Implementation Summary

- Cross-boundary contracts centralized (`ExceptionResult` in `src/elspeth/contracts/results.py`, core telemetry events in `src/elspeth/contracts/events.py`).
- Landscape recorder now consumes typed `RoutingReason` without whitelist workarounds (`src/elspeth/core/landscape/recorder.py`).
- Import surfaces cleaned and exports updated (`src/elspeth/contracts/__init__.py`, `src/elspeth/telemetry/__init__.py`).

## Phase 1: Move Cross-Boundary Types to Contracts

### Task 1: Move ExceptionResult to contracts/results.py

**Files:**
- Modify: `src/elspeth/contracts/results.py` (add ExceptionResult)
- Modify: `src/elspeth/contracts/__init__.py` (export ExceptionResult)
- Modify: `src/elspeth/engine/batch_adapter.py` (import from contracts)
- Modify: `src/elspeth/plugins/batching/mixin.py` (update imports)
- Modify: `src/elspeth/plugins/batching/ports.py` (update imports)
- Test: `tests/contracts/test_results.py`

**Step 1: Write failing test for ExceptionResult in contracts**

```python
# tests/contracts/test_results.py (add to existing file)
def test_exception_result_in_contracts():
    """ExceptionResult should be importable from contracts."""
    from elspeth.contracts import ExceptionResult

    err = ExceptionResult(
        exception=ValueError("test"),
        traceback="Traceback...",
    )
    assert err.exception.args == ("test",)
    assert err.traceback == "Traceback..."
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/contracts/test_results.py::test_exception_result_in_contracts -v`
Expected: FAIL with "cannot import name 'ExceptionResult' from 'elspeth.contracts'"

**Step 3: Add ExceptionResult to contracts/results.py**

Add at end of `src/elspeth/contracts/results.py`:

```python
@dataclass
class ExceptionResult:
    """Wrapper for exceptions that should propagate through async pattern.

    When a worker thread encounters an uncaught exception (plugin bug),
    it wraps the exception in this container. The waiter then re-raises
    the original exception in the orchestrator thread, ensuring plugin
    bugs crash the pipeline as intended.

    Used by:
    - engine/batch_adapter.py: Wraps exceptions in worker threads
    - plugins/batching/mixin.py: Creates ExceptionResult on worker failure
    - plugins/batching/ports.py: Type hint in BatchOutputPort protocol
    """

    exception: BaseException
    traceback: str
```

**Step 4: Export ExceptionResult from contracts/__init__.py**

In `src/elspeth/contracts/__init__.py`:

1. Add to imports from results:
```python
from elspeth.contracts.results import (
    ArtifactDescriptor,
    ExceptionResult,  # ADD THIS
    FailureInfo,
    ...
)
```

2. Add to `__all__` list (in results section):
```python
    # results
    "ArtifactDescriptor",
    "ExceptionResult",  # ADD THIS
    "FailureInfo",
```

**Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/contracts/test_results.py::test_exception_result_in_contracts -v`
Expected: PASS

**Step 6: Update batch_adapter.py to import from contracts**

In `src/elspeth/engine/batch_adapter.py`:

1. Remove the ExceptionResult class definition (lines 48-59)
2. Add import at top (after TYPE_CHECKING block, around line 44):
```python
from elspeth.contracts import ExceptionResult
```

**Step 7: Update batching/mixin.py imports**

In `src/elspeth/plugins/batching/mixin.py`:

1. Line 33: Change from:
```python
    from elspeth.engine.batch_adapter import ExceptionResult
```
To:
```python
    from elspeth.contracts import ExceptionResult
```

2. Line 237: Change from:
```python
            from elspeth.engine.batch_adapter import ExceptionResult
```
To:
```python
            from elspeth.contracts import ExceptionResult
```

3. Line 293: Change from:
```python
                from elspeth.engine.batch_adapter import ExceptionResult
```
To:
```python
                from elspeth.contracts import ExceptionResult
```

**Step 8: Update batching/ports.py imports**

In `src/elspeth/plugins/batching/ports.py`:

Line 24: Change from:
```python
    from elspeth.engine.batch_adapter import ExceptionResult
```
To:
```python
    from elspeth.contracts import ExceptionResult
```

**Step 9: Run full test suite for batching**

Run: `.venv/bin/python -m pytest tests/plugins/batching/ tests/engine/test_batch_adapter.py -v`
Expected: All PASS

**Step 10: Commit**

```bash
git add src/elspeth/contracts/results.py src/elspeth/contracts/__init__.py \
        src/elspeth/engine/batch_adapter.py src/elspeth/plugins/batching/mixin.py \
        src/elspeth/plugins/batching/ports.py tests/contracts/test_results.py
git commit -m "refactor(contracts): move ExceptionResult to contracts/results.py

ExceptionResult crosses engine↔plugins boundary and belongs in contracts.
This eliminates the need for whitelist entry.

- Move dataclass from engine/batch_adapter.py to contracts/results.py
- Export from contracts/__init__.py
- Update all importers to use contracts path"
```

---

### Task 2: Move Telemetry Events to contracts/events.py

**Files:**
- Modify: `src/elspeth/contracts/events.py` (add TransformCompleted, GateEvaluated, TokenCompleted)
- Modify: `src/elspeth/contracts/__init__.py` (export new events)
- Modify: `src/elspeth/telemetry/events.py` (re-export from contracts)
- Modify: `src/elspeth/engine/processor.py` (update imports)
- Test: `tests/contracts/test_events.py`

**Step 1: Write failing test for telemetry events in contracts**

Create `tests/contracts/test_events.py`:

```python
"""Tests for contracts/events.py exports."""

from datetime import UTC, datetime


def test_transform_completed_in_contracts():
    """TransformCompleted should be importable from contracts."""
    from elspeth.contracts import TransformCompleted
    from elspeth.contracts.enums import NodeStateStatus

    event = TransformCompleted(
        timestamp=datetime.now(UTC),
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        node_id="node-1",
        plugin_name="test",
        status=NodeStateStatus.COMPLETED,
        duration_ms=100.0,
        input_hash="abc",
        output_hash="def",
    )
    assert event.run_id == "run-1"


def test_gate_evaluated_in_contracts():
    """GateEvaluated should be importable from contracts."""
    from elspeth.contracts import GateEvaluated
    from elspeth.contracts.enums import RoutingMode

    event = GateEvaluated(
        timestamp=datetime.now(UTC),
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        node_id="gate-1",
        plugin_name="test_gate",
        routing_mode=RoutingMode.MOVE,
        destinations=("sink1",),
    )
    assert event.destinations == ("sink1",)


def test_token_completed_in_contracts():
    """TokenCompleted should be importable from contracts."""
    from elspeth.contracts import TokenCompleted
    from elspeth.contracts.enums import RowOutcome

    event = TokenCompleted(
        timestamp=datetime.now(UTC),
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        outcome=RowOutcome.COMPLETED,
        sink_name="output",
    )
    assert event.outcome == RowOutcome.COMPLETED


def test_telemetry_events_inherit_from_contracts_base():
    """Telemetry-specific events should inherit from contracts TelemetryEvent.

    After the refactor, TelemetryEvent lives in contracts but telemetry-specific
    events (RunStarted, RowCreated, etc.) remain in telemetry/events.py. They
    must still inherit from the contracts TelemetryEvent for type consistency.
    """
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.telemetry.events import (
        ExternalCallCompleted,
        PhaseChanged,
        RowCreated,
        RunCompleted,
        RunStarted,
    )

    # All telemetry-specific events must be subclasses of TelemetryEvent
    assert issubclass(RunStarted, TelemetryEvent)
    assert issubclass(RunCompleted, TelemetryEvent)
    assert issubclass(PhaseChanged, TelemetryEvent)
    assert issubclass(RowCreated, TelemetryEvent)
    assert issubclass(ExternalCallCompleted, TelemetryEvent)
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/contracts/test_events.py -v`
Expected: FAIL with "cannot import name 'TransformCompleted' from 'elspeth.contracts'"

**Step 3: Add imports and TelemetryEvent base to contracts/events.py**

At the top of `src/elspeth/contracts/events.py`, add:

```python
from datetime import datetime
```

After existing imports, add:

```python
from elspeth.contracts.enums import (
    NodeStateStatus,
    RoutingMode,
    RowOutcome,
)
```

At the end of `src/elspeth/contracts/events.py`, add:

```python
# =============================================================================
# Telemetry Events (Row-Level Observability)
# =============================================================================
# These events are emitted by the engine and consumed by telemetry exporters.
# They provide operational visibility alongside the Landscape audit trail.


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Base class for all telemetry events.

    All events include:
    - timestamp: When the event occurred (UTC)
    - run_id: Pipeline run this event belongs to

    Events are immutable (frozen) for thread-safety and to prevent
    accidental modification during export.
    """

    timestamp: datetime
    run_id: str


@dataclass(frozen=True, slots=True)
class TransformCompleted(TelemetryEvent):
    """Emitted when a transform finishes processing a row.

    Attributes:
        row_id: Source row identity
        token_id: Token instance being processed
        node_id: DAG node that processed the row
        plugin_name: Name of the transform plugin
        status: Processing result (completed, failed)
        duration_ms: Transform execution time in milliseconds
        input_hash: Hash of transform input for lineage
        output_hash: Hash of transform output for lineage
    """

    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    status: NodeStateStatus
    duration_ms: float
    input_hash: str
    output_hash: str


@dataclass(frozen=True, slots=True)
class GateEvaluated(TelemetryEvent):
    """Emitted when a gate makes a routing decision.

    Attributes:
        row_id: Source row identity
        token_id: Token instance being routed
        node_id: Gate node that made the decision
        plugin_name: Name of the gate plugin
        routing_mode: How routing was performed (move, copy)
        destinations: Tuple of destination node/sink names
    """

    row_id: str
    token_id: str
    node_id: str
    plugin_name: str
    routing_mode: RoutingMode
    destinations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TokenCompleted(TelemetryEvent):
    """Emitted when a token reaches its terminal state.

    Attributes:
        row_id: Source row identity
        token_id: Token instance that completed
        outcome: Terminal outcome (completed, routed, failed, etc.)
        sink_name: Destination sink if applicable, None otherwise
    """

    row_id: str
    token_id: str
    outcome: RowOutcome
    sink_name: str | None
```

**Step 4: Export new events from contracts/__init__.py**

In `src/elspeth/contracts/__init__.py`:

1. Update imports from events:
```python
from elspeth.contracts.events import (
    GateEvaluated,  # ADD
    PhaseAction,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompleted,
    RunCompletionStatus,
    TelemetryEvent,  # ADD
    TokenCompleted,  # ADD
    TransformCompleted,  # ADD
)
```

2. Update `__all__` list (in events section):
```python
    # events
    "GateEvaluated",  # ADD
    "PhaseAction",
    "PhaseCompleted",
    "PhaseError",
    "PhaseStarted",
    "PipelinePhase",
    "RunCompleted",
    "RunCompletionStatus",
    "TelemetryEvent",  # ADD
    "TokenCompleted",  # ADD
    "TransformCompleted",  # ADD
```

**Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/contracts/test_events.py -v`
Expected: PASS

**Step 6: Restructure telemetry/events.py to re-export from contracts**

This step requires surgical changes to `src/elspeth/telemetry/events.py`. The file currently defines all events locally; we need to:
1. Import and re-export the 4 classes that moved to contracts
2. Delete the local definitions of those 4 classes
3. Keep the telemetry-specific classes that inherit from `TelemetryEvent`

**IMPORTANT:** Do NOT simply "replace lines 31-177" — that range includes classes we must keep (`RunStarted`, `RunCompleted`, `PhaseChanged`, `RowCreated`).

**Step 6a: Extend existing contracts import (line 28)**

Line 28 already imports from `elspeth.contracts.events`. Extend it to include `TelemetryEvent`:

```python
# BEFORE:
from elspeth.contracts.events import PhaseAction, PipelinePhase

# AFTER:
from elspeth.contracts.events import PhaseAction, PipelinePhase, TelemetryEvent
```

**Note:** We do NOT re-export `TransformCompleted`, `GateEvaluated`, `TokenCompleted` from telemetry. Per the No Legacy Code Policy, callers must import these directly from `elspeth.contracts`. The engine already imports from contracts (Step 7), so no compatibility shim is needed.

**Step 6b: Delete local class definitions that moved to contracts**

Delete these class definitions (they now come from contracts):

1. Delete `TelemetryEvent` class (lines 31-44):
```python
# DELETE THIS ENTIRE BLOCK:
@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """Base class for all telemetry events.
    ...
    """
    timestamp: datetime
    run_id: str
```

2. Delete `TransformCompleted` class (lines 117-139)
3. Delete `GateEvaluated` class (lines 142-160)
4. Delete `TokenCompleted` class (lines 163-177)

**Step 6c: Verify kept classes still work**

The following classes remain in `telemetry/events.py` and continue to inherit from `TelemetryEvent` (now imported from contracts):
- `RunStarted` (lines 52-62) — telemetry-specific lifecycle event
- `RunCompleted` (lines 65-77) — telemetry version, different from contracts.events.RunCompleted
- `PhaseChanged` (lines 80-94) — telemetry-specific lifecycle event
- `RowCreated` (lines 102-114) — telemetry-specific row event
- `ExternalCallCompleted` (lines 185-207) — telemetry-specific call event

These classes inherit from `TelemetryEvent`, which is now imported from contracts. No changes needed to their definitions — Python resolves the import automatically.

**Step 6d: Update or add `__all__` export list**

At the top of the file (after imports), add or update the `__all__` list:

```python
__all__ = [
    # Base class imported from contracts (for inheritance)
    "TelemetryEvent",
    # Telemetry-specific events (defined here, NOT in contracts)
    "RunStarted",
    "RunCompleted",
    "PhaseChanged",
    "RowCreated",
    "ExternalCallCompleted",
]
```

**Note:** `TransformCompleted`, `GateEvaluated`, `TokenCompleted` are NOT exported from here. Callers must import them from `elspeth.contracts` directly. No backwards compatibility shims.

**Step 6e: Update telemetry/__init__.py to stop re-exporting moved classes**

**CRITICAL:** The package `__init__.py` currently re-exports all events from `telemetry/events.py`. After deleting the moved classes from `events.py`, the package init will fail to import them. We must update it.

In `src/elspeth/telemetry/__init__.py`:

1. Update imports (lines 49-59) — remove the 3 moved classes:

```python
# BEFORE:
from elspeth.telemetry.events import (
    ExternalCallCompleted,
    GateEvaluated,        # REMOVE
    PhaseChanged,
    RowCreated,
    RunCompleted,
    RunStarted,
    TelemetryEvent,
    TokenCompleted,       # REMOVE
    TransformCompleted,   # REMOVE
)

# AFTER:
from elspeth.telemetry.events import (
    ExternalCallCompleted,
    PhaseChanged,
    RowCreated,
    RunCompleted,
    RunStarted,
    TelemetryEvent,
)
```

2. Update `__all__` list (lines 65-81) — remove the 3 moved classes:

```python
# REMOVE these 3 entries from __all__:
"GateEvaluated",
"TokenCompleted",
"TransformCompleted",
```

3. Update docstring example (lines 29-31) to remove the moved classes from the usage example.

**Step 6f: Verify the restructured files**

Run: `.venv/bin/python -c "from elspeth.telemetry import TelemetryEvent, RunStarted; from elspeth.contracts import TransformCompleted; print('OK')"`
Expected: "OK" (no import errors — telemetry events from telemetry, moved events from contracts)

**Step 7: Find and update ALL callers importing moved events from telemetry**

Run: `grep -rn "from elspeth.telemetry.events import" src/ --include="*.py"`

Update ALL files that import `TransformCompleted`, `GateEvaluated`, or `TokenCompleted` from `telemetry.events` to import from `contracts` instead.

Known callers (verify with grep):
- `src/elspeth/engine/processor.py` (lines 202, 245, 280)

**Step 8: Update engine/processor.py imports**

In `src/elspeth/engine/processor.py`:

Line 202: Change from:
```python
        from elspeth.telemetry.events import TransformCompleted
```
To:
```python
        from elspeth.contracts import TransformCompleted
```

Line 245: Change from:
```python
        from elspeth.telemetry.events import GateEvaluated
```
To:
```python
        from elspeth.contracts import GateEvaluated
```

Line 280: Change from:
```python
        from elspeth.telemetry.events import TokenCompleted
```
To:
```python
        from elspeth.contracts import TokenCompleted
```

**Step 9: Run telemetry and engine tests (including inheritance verification)**

Run: `.venv/bin/python -m pytest tests/telemetry/ tests/engine/test_processor*.py tests/contracts/test_events.py -v`
Expected: All PASS

This includes `test_telemetry_events_inherit_from_contracts_base` which verifies that telemetry-specific events (`RunStarted`, `RowCreated`, etc.) correctly inherit from the `TelemetryEvent` base class now defined in contracts.

**Step 10: Commit**

```bash
git add src/elspeth/contracts/events.py src/elspeth/contracts/__init__.py \
        src/elspeth/telemetry/events.py src/elspeth/engine/processor.py \
        tests/contracts/test_events.py
git commit -m "refactor(contracts): move telemetry events to contracts/events.py

TransformCompleted, GateEvaluated, TokenCompleted cross engine↔telemetry
boundary and belong in contracts. TelemetryEvent base class also moved.

- Add TelemetryEvent base and 3 row-level events to contracts/events.py
- Export from contracts/__init__.py
- Remove definitions from telemetry/events.py (import TelemetryEvent for inheritance)
- Update all callers to import from contracts directly"
```

---

### Task 3: Resolve RunCompleted Naming Collision and Remove Defensive Aliases

**Problem:** Two types named `RunCompleted` exist with different purposes, requiring aliasing workarounds. Additionally, the orchestrator uses defensive `Telemetry*` aliases for clarity.

**Solution:** Establish clear naming convention:
- `contracts.events.RunCompleted` → `RunSummary` (CLI summary with exit codes, routing breakdown)
- `telemetry.events.RunCompleted` → `RunFinished` (pairs with `RunStarted`)
- Remove all `Telemetry*` aliases in orchestrator

**Files:**
- Modify: `src/elspeth/contracts/events.py` (rename RunCompleted → RunSummary)
- Modify: `src/elspeth/contracts/__init__.py` (update export)
- Modify: `src/elspeth/telemetry/events.py` (rename RunCompleted → RunFinished)
- Modify: `src/elspeth/telemetry/__init__.py` (update export)
- Modify: `src/elspeth/telemetry/filtering.py` (update usage)
- Modify: `src/elspeth/cli.py` (update RunCompleted → RunSummary)
- Modify: `src/elspeth/engine/orchestrator.py` (update both, remove aliases)
- Modify: Multiple test files

**Step 1: Rename contracts.events.RunCompleted → RunSummary**

In `src/elspeth/contracts/events.py`, line 103:

```python
# BEFORE
class RunCompleted:
    """Emitted when pipeline run finishes (success or failure).

    Provides final summary for CI integration.
    ...
    """

# AFTER
class RunSummary:
    """Summary emitted when pipeline run finishes (success or failure).

    Provides final metrics for CI integration: exit codes, row counts,
    routing breakdown.
    ...
    """
```

**Step 2: Update contracts/__init__.py export**

Change:
```python
from elspeth.contracts.events import (
    ...
    RunCompleted,  # CHANGE TO: RunSummary
    ...
)

__all__ = [
    ...
    "RunCompleted",  # CHANGE TO: "RunSummary"
    ...
]
```

**Step 3: Rename telemetry.events.RunCompleted → RunFinished**

In `src/elspeth/telemetry/events.py`, line 66:

```python
# BEFORE
class RunCompleted(TelemetryEvent):
    """Emitted when a pipeline run finishes (success or failure).
    ...
    """

# AFTER
class RunFinished(TelemetryEvent):
    """Emitted when a pipeline run finishes (success or failure).

    Pairs with RunStarted for telemetry lifecycle tracking.
    ...
    """
```

**Step 4: Update telemetry/__init__.py export**

Change all occurrences of `RunCompleted` to `RunFinished`.

**Step 5: Update telemetry/filtering.py**

Line 19 and 58: Change `RunCompleted` to `RunFinished`.

**Step 6: Update cli.py (RunCompleted → RunSummary)**

Run: `grep -n "RunCompleted" src/elspeth/cli.py`

Update all occurrences (~15 locations) to use `RunSummary`:
- Import (line 24)
- Type hints in `_format_run_completed_json` and `_format_run_completed` functions
- Event bus subscriptions

**Step 7: Update engine/orchestrator.py and REMOVE aliases**

1. Line 39: Change import from `RunCompleted` to `RunSummary`

2. Lines 544-550: Remove aliases, import directly:
```python
# BEFORE
from elspeth.telemetry import (
    PhaseChanged as TelemetryPhaseChanged,
)
from elspeth.telemetry import (
    RunCompleted as TelemetryRunCompleted,
)
from elspeth.telemetry import (
    RunStarted as TelemetryRunStarted,
)

# AFTER
from elspeth.telemetry import (
    PhaseChanged,
    RunFinished,
    RunStarted,
)
```

3. Lines 775-781: Remove aliases:
```python
# BEFORE
from elspeth.telemetry import (
    PhaseChanged as TelemetryPhaseChanged,
)
from elspeth.telemetry import (
    RowCreated as TelemetryRowCreated,
)

# AFTER
from elspeth.telemetry import (
    PhaseChanged,
    RowCreated,
)
```

4. Update all usages:
   - `TelemetryRunCompleted` → `RunFinished`
   - `TelemetryRunStarted` → `RunStarted`
   - `TelemetryPhaseChanged` → `PhaseChanged`
   - `TelemetryRowCreated` → `RowCreated`
   - `RunCompleted(` (contracts) → `RunSummary(`

**Step 8: Update test files**

Run: `grep -rn "RunCompleted" tests/`

Update all test files. Key files:
- `tests/telemetry/` - change to `RunFinished`
- `tests/engine/test_orchestrator_telemetry.py` - change to `RunFinished`
- `tests/unit/telemetry/` - change to `RunFinished`

**Step 9: Run verification**

```bash
# Type check
.venv/bin/python -m mypy src/

# Run telemetry tests
.venv/bin/python -m pytest tests/telemetry/ tests/engine/test_orchestrator*.py -v

# Run CLI tests (if any use RunCompleted/RunSummary)
.venv/bin/python -m pytest tests/ -k "cli" -v
```

**Step 10: Commit**

```bash
git add src/elspeth/contracts/events.py src/elspeth/contracts/__init__.py \
        src/elspeth/telemetry/events.py src/elspeth/telemetry/__init__.py \
        src/elspeth/telemetry/filtering.py src/elspeth/cli.py \
        src/elspeth/engine/orchestrator.py tests/
git commit -m "refactor(events): resolve RunCompleted naming collision

Establish clear naming convention for run completion events:
- contracts.events.RunSummary: CLI summary with exit codes, routing
- telemetry.events.RunFinished: Observability metrics (pairs with RunStarted)

Also remove defensive Telemetry* aliases in orchestrator - no longer
needed since names are now distinct.

This eliminates import aliasing workarounds and improves code clarity."
```

---

### Task 4: Update Whitelist to Remove Migrated Entries

**Files:**
- Modify: `config/cicd/contracts-whitelist.yaml`

**Step 1: Remove the 4 type boundary entries**

In `config/cicd/contracts-whitelist.yaml`, remove these entries from `allowed_external_types`:

```yaml
  # REMOVE THESE 4 ENTRIES:
  - "engine/batch_adapter:ExceptionResult"
  - "telemetry/events:TransformCompleted"
  - "telemetry/events:GateEvaluated"
  - "telemetry/events:TokenCompleted"
```

**Step 2: Run contract enforcement to verify**

Run: `.venv/bin/python scripts/check_contracts.py`
Expected: All checks pass (no violations)

**Step 3: Commit**

```bash
git add config/cicd/contracts-whitelist.yaml
git commit -m "chore(whitelist): remove migrated type entries

Types are now in contracts/, no longer need whitelist exceptions."
```

---

### Task 5: Run Full CI Validation

**Step 1: Run linting**

Run: `.venv/bin/python -m ruff check src/ tests/`
Expected: No errors

**Step 2: Run type checking**

Run: `.venv/bin/python -m mypy src/`
Expected: No errors

**Step 3: Run full test suite**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/examples/test_llm_examples.py`
Expected: All PASS

**Step 4: Run contract enforcement**

Run: `.venv/bin/python scripts/check_contracts.py`
Expected: All checks pass

**Step 5: Run no-bug-hiding check**

Run: `.venv/bin/python scripts/cicd/no_bug_hiding.py check --root src/elspeth --allowlist config/cicd/no_bug_hiding.yaml`
Expected: All checks pass

---

## Phase 2: Use Existing TypedDicts in Landscape Recorder

### Task 6: Update Landscape Recorder to Use RoutingReason TypedDict

**Files:**
- Modify: `src/elspeth/core/landscape/recorder.py`
- Test: `tests/core/landscape/test_recorder.py`

**Scope Note (Partial Type Safety):** This task updates the recorder's type signature to use `RoutingReason` TypedDict, but this is a **partial improvement**:

- `RoutingAction.reason` (in `contracts/routing.py`) remains `Mapping[str, Any]`
- Executors pass `dict(action.reason)` to the recorder, which becomes `dict[str, Any]` at runtime
- The TypedDict signature improves documentation and IDE support but doesn't enforce field presence at runtime

A follow-up task could strengthen `RoutingAction` itself to use `RoutingReason`, but that would affect all gate plugins and is out of scope for this refactor.

**Step 1: Write test for typed routing reason**

Add to `tests/core/landscape/test_recorder.py`:

```python
def test_record_routing_event_accepts_routing_reason_typeddict():
    """record_routing_event should accept RoutingReason TypedDict."""
    from elspeth.contracts import RoutingReason

    # This test verifies the type signature accepts RoutingReason
    # The actual recording is tested elsewhere; this is for type coverage
    reason: RoutingReason = {
        "rule": "field_equals",
        "matched_value": "category_a",
        "field": "category",
        "comparison": "==",
    }
    # Type checker should accept this without error
    assert reason["rule"] == "field_equals"
```

**Step 2: Update record_routing_event signature**

In `src/elspeth/core/landscape/recorder.py`:

1. Add import at top (in TYPE_CHECKING block or regular imports):
```python
from elspeth.contracts import RoutingReason
```

2. Line 1156-1161: Change from:
```python
    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: dict[str, Any] | None = None,
```
To:
```python
    def record_routing_event(
        self,
        state_id: str,
        edge_id: str,
        mode: RoutingMode,
        reason: RoutingReason | None = None,
```

**Note:** Per the No Legacy Code Policy, we use `RoutingReason` directly without a `dict[str, Any]` fallback. All callers must use the TypedDict. This provides type safety and documentation at the boundary.

**Step 3: Update record_routing_events signature**

In `src/elspeth/core/landscape/recorder.py`:

Line 1216-1220: Change from:
```python
    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: dict[str, Any] | None = None,
```
To:
```python
    def record_routing_events(
        self,
        state_id: str,
        routes: list[RoutingSpec],
        reason: RoutingReason | None = None,
```

**Step 4: Find and update all callers of record_routing_event(s)**

Run: `grep -rn "record_routing_event" src/elspeth/ --include="*.py" | grep -v "def record_routing_event"`

For each caller passing a `dict` to the `reason` parameter, update to use `RoutingReason` TypedDict:

```python
# BEFORE (untyped dict)
recorder.record_routing_event(
    state_id=state_id,
    edge_id=edge_id,
    mode=mode,
    reason={"rule": "threshold", "matched_value": score},
)

# AFTER (typed RoutingReason)
from elspeth.contracts import RoutingReason

reason: RoutingReason = {
    "rule": "threshold",
    "matched_value": score,
}
recorder.record_routing_event(
    state_id=state_id,
    edge_id=edge_id,
    mode=mode,
    reason=reason,
)
```

**Note:** The `RoutingReason` TypedDict has required fields `rule` and `matched_value`, with optional fields `threshold`, `field`, `comparison`. Update callers to match this structure.

**Step 5: Run type checker**

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py`
Expected: No errors

**Step 6: Run landscape tests**

Run: `.venv/bin/python -m pytest tests/core/landscape/ -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/elspeth/core/landscape/recorder.py tests/core/landscape/test_recorder.py \
        <any files modified in Step 4>
git commit -m "refactor(landscape): use RoutingReason TypedDict in recorder signatures

Update record_routing_event() and record_routing_events() to require
RoutingReason TypedDict. All callers updated to use typed dict.

This provides type safety and documentation at the audit boundary."
```

---

### Task 7: Final Validation

**Step 1: Run full CI suite**

Run: `.venv/bin/python -m pytest tests/ -v --ignore=tests/examples/test_llm_examples.py`
Expected: All PASS

**Step 2: Run all contract enforcement**

Run: `.venv/bin/python scripts/check_contracts.py && .venv/bin/python scripts/cicd/no_bug_hiding.py check --root src/elspeth --allowlist config/cicd/no_bug_hiding.yaml`
Expected: All checks pass

**Step 3: Squash commits for clean history (optional)**

If desired, squash the phase commits into logical units:
- Phase 1: "refactor(contracts): consolidate cross-boundary types"
- Phase 2: "refactor(landscape): use typed contracts in recorder"

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Move ExceptionResult to contracts | 5 files |
| 2 | Move telemetry events to contracts | 6+ files (contracts/events.py, contracts/__init__.py, telemetry/events.py, telemetry/__init__.py, + all callers) |
| 3 | Resolve RunCompleted collision, remove aliases | 7+ files (contracts, telemetry, cli, orchestrator, tests) |
| 4 | Update whitelist | 1 file |
| 5 | Full CI validation | (verification only) |
| 6 | Use RoutingReason in Landscape (partial type safety) | 2+ files (recorder + all callers) |
| 7 | Final validation | (verification only) |

**Total:** ~20+ files modified, 4 whitelist entries eliminated, naming collision resolved, stronger type safety at boundaries.

**Naming Convention Established:**
- `contracts.events.RunSummary` - CLI summary with exit codes, routing breakdown
- `telemetry.events.RunFinished` - Observability metrics (pairs with `RunStarted`)

**Note:** File counts for Tasks 2, 3, and 6 depend on how many callers exist. Use `grep` to find them before starting.
