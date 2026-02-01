# CLI Observability Design

**Date:** 2026-01-23
**Status:** ✅ Final - Ready for Implementation
**Problem:** CLI appears to "hang" in containers/CI with no output during setup phases

**Review Summary:** Reviewed by architecture critic and code review agents. All critical issues addressed:
- ✅ Type safety: Added `PipelinePhase`, `PhaseAction`, `RunCompletionStatus` enums
- ✅ Protocol pattern: `NullEventBus` no longer inherits (prevents silent bugs)
- ✅ Error context: `PhaseError` stores `BaseException` with target field
- ✅ Phase coverage: Added `aggregations`, `schema_validation`, `export` phases
- ✅ Migration clarity: Removed `on_progress` callback, EventBus subscription is the single mechanism
- ✅ Memory efficiency: All event dataclasses use `slots=True`
- ✅ Explicit behavior: EventBus uses dict + `.get()`, not `defaultdict`
- ✅ CLAUDE.md compliance: No backwards compatibility shims, clean migration

## Problem Statement

The ELSPETH CLI provides no feedback during pipeline setup phases. Users in container/CI environments see nothing for extended periods and cannot tell if the process is working or stuck.

Current issues:
1. **Silent setup phases** - Config loading, graph building, plugin init, database connection produce no output
2. **Row-based progress only** - Progress emits every 100 rows, useless for slow LLM pipelines (each row spawns 12 LLM queries, ~30+ seconds per row) or small datasets (<100 rows)
3. **No structured output for CI** - Log aggregators can't parse the current output format

## Design Goals

1. Visibility into every pipeline phase
2. Regular progress feedback regardless of row processing speed
3. Human-readable output by default, structured JSON for CI (`--json-logs`)
4. Clean separation between domain events and presentation
5. Testable without CLI dependencies

## Architecture

### Event Bus Pattern

```
┌─────────────────────────────────────────────────────────────┐
│                         cli.py                               │
│                                                              │
│  ┌─────────────────┐    ┌──────────────────────────────┐   │
│  │ ConsoleFormatter│◄───│ Subscribe to domain events   │   │
│  │ or JsonFormatter│    │ Format for output            │   │
│  └─────────────────┘    └──────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────┘
                               │ creates & injects
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      orchestrator.py                         │
│                                                              │
│  ┌─────────────────┐                                        │
│  │ EventBus        │ ← emits PhaseStarted, ProgressEvent,  │
│  │ (injected)      │   PhaseCompleted, PhaseError,         │
│  │                 │   RunCompleted                         │
│  └─────────────────┘                                        │
│                                                              │
│  No knowledge of formatting, logging backends, or CLI       │
└─────────────────────────────────────────────────────────────┘
```

### Domain Events

Located in `src/elspeth/contracts/events.py` (new file):

```python
from dataclasses import dataclass
from enum import Enum

class PipelinePhase(str, Enum):
    """Pipeline lifecycle phases for observability events.

    Uses (str, Enum) pattern for consistency with existing codebase
    (see contracts/enums.py RunStatus).
    """
    CONFIG = "config"
    GRAPH = "graph"
    PLUGINS = "plugins"
    AGGREGATIONS = "aggregations"
    DATABASE = "database"
    SCHEMA_VALIDATION = "schema_validation"
    SOURCE = "source"
    PROCESS = "process"
    EXPORT = "export"


class PhaseAction(str, Enum):
    """Actions within a pipeline phase."""
    LOADING = "loading"
    VALIDATING = "validating"
    BUILDING = "building"
    CONNECTING = "connecting"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    EXPORTING = "exporting"


class RunCompletionStatus(str, Enum):
    """Final status for RunCompleted events."""
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class PhaseStarted:
    """Emitted when a pipeline phase begins.

    Phases represent major lifecycle stages:
    - config: Loading and validating settings
    - graph: Building and validating execution graph
    - plugins: Instantiating source, transforms, and sinks
    - aggregations: Instantiating aggregation plugins
    - database: Connecting to Landscape database
    - schema_validation: Validating plugin schemas
    - source: Loading source data
    - process: Processing rows through transforms
    - export: Exporting results (when enabled)

    Attributes:
        phase: The lifecycle phase starting
        action: What's happening (e.g., "loading", "validating")
        target: Optional target (e.g., file path, plugin name)
    """
    phase: PipelinePhase
    action: PhaseAction
    target: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseCompleted:
    """Emitted when a pipeline phase completes successfully."""
    phase: PipelinePhase
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PhaseError:
    """Emitted when a pipeline phase fails.

    Stores the full exception object to preserve traceback, exception type,
    and chained causes for debugging and audit trail integrity.
    """
    phase: PipelinePhase
    error: BaseException
    target: str | None = None  # What failed (plugin name, file path, etc.)

    @property
    def error_message(self) -> str:
        """Human-readable error message for formatting."""
        return str(self.error)


@dataclass(frozen=True, slots=True)
class RunCompleted:
    """Emitted when pipeline run finishes (success or failure).

    Provides final summary for CI integration.
    """
    run_id: str
    status: RunCompletionStatus
    total_rows: int
    succeeded: int
    failed: int
    quarantined: int
    duration_seconds: float
    exit_code: int  # 0=success, 1=partial failure, 2=total failure
```

**Note:** The existing `ProgressEvent` in `contracts/cli.py` will be reused for progress ticks. No new `ProgressTick` class needed.

### EventBus Implementation

Located in `src/elspeth/core/events.py` (new file):

```python
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")


class EventBusProtocol(Protocol):
    """Protocol for event bus implementations.

    Allows both EventBus and NullEventBus to satisfy the interface
    without inheritance, preventing accidental substitution bugs.
    """

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe a handler to an event type."""
        ...

    def emit(self, event: T) -> None:
        """Emit an event to all subscribers."""
        ...


class EventBus:
    """Simple synchronous event bus for pipeline observability.

    Events are dispatched synchronously to all subscribers. Handler
    exceptions propagate to the caller - formatters are "our code"
    per CLAUDE.md, so bugs should crash immediately.
    """

    def __init__(self) -> None:
        self._subscribers: dict[type, list[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """Subscribe a handler to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def emit(self, event: T) -> None:
        """Emit an event to all subscribers.

        Handlers are called synchronously in subscription order.
        Exceptions propagate - handlers are system code, not user code.

        Events with no subscribers are silently ignored - this is
        intentional for decoupling. Formatters subscribe to events
        they care about.
        """
        handlers = self._subscribers.get(type(event), [])
        for handler in handlers:
            handler(event)


class NullEventBus:
    """No-op event bus for library use where no CLI is present.

    IMPORTANT: Does NOT inherit from EventBus. Calling subscribe() on
    this is a no-op by design - use only when you genuinely don't want
    event observability (e.g., programmatic API usage, testing).

    Per CLAUDE.md: "A defective plugin that silently produces wrong
    results is worse than a crash." If someone subscribes expecting
    callbacks, inheritance would hide the bug. Protocol-based design
    makes the no-op behavior explicit.
    """

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """No-op subscription - handler will never be called."""
        pass  # Intentional no-op - no subscribers expected

    def emit(self, event: T) -> None:
        """No-op emission - no handlers to call."""
        pass  # Intentional no-op
```

### Progress Timing

Hybrid approach: emit on whichever comes first:
- First row (immediate feedback)
- Every 100 rows (fast pipelines)
- Every 5 seconds (slow pipelines)

Injectable clock for testing determinism.

```python
# In orchestrator.py
from elspeth.core.events import EventBusProtocol, NullEventBus

PROGRESS_ROW_INTERVAL = 100
PROGRESS_TIME_INTERVAL = 5.0  # seconds

class Orchestrator:
    def __init__(
        self,
        db: LandscapeDB,
        *,
        event_bus: EventBusProtocol = NullEventBus(),  # Explicit default, no coercion
        clock: Callable[[], float] = time.perf_counter,
        canonical_version: str = "sha256-rfc8785-v1",
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_settings: CheckpointSettings | None = None,
    ) -> None:
        self._events = event_bus  # No coercion needed
        self._clock = clock
        # ... existing init preserved
```

**Clock injection scope:** The `clock` parameter is injected specifically for progress timing (to test the hybrid "100 rows OR 5 seconds" logic deterministically). Other timing in the orchestrator (run duration, phase duration) continues using `time.perf_counter()` directly—these don't need injection because they're measured in real time for the audit trail. Only the progress emission logic requires deterministic testing.

### CLI Formatters

Console formatter (default):
```
[CONFIG] Loading settings.yaml
[CONFIG] Done (0.02s)
[GRAPH] Building execution graph
[GRAPH] Done (0.01s)
[PLUGINS] Initializing source: csv
[PLUGINS] Initializing transforms: azure_multi_query_llm
[PLUGINS] Done (0.05s)
[AGGREGATIONS] Initializing aggregations
[AGGREGATIONS] Done (0.01s)
[DATABASE] Connecting to sqlite:///runs/audit.db
[DATABASE] Run started: run-abc123
[DATABASE] Done (0.03s)
[SCHEMA_VALIDATION] Validating plugin schemas
[SCHEMA_VALIDATION] Done (0.01s)
[SOURCE] Loading from input.csv
[SOURCE] Done (0.01s)
[PROCESS] Starting row processing
[PROCESS] 1 rows | 0 rows/sec | ✓0 ✗0 ⚠0
[PROCESS] 2 rows | 0 rows/sec | ✓2 ✗0 ⚠0
[PROCESS] Done: 8 rows in 4m 32s
[RUN] Completed: run-abc123 (exit 0)
```

JSON formatter (`--json-logs`):
```json
{"ts":"2026-01-23T10:00:00Z","event":"PhaseStarted","phase":"config","action":"loading","target":"settings.yaml"}
{"ts":"2026-01-23T10:00:00Z","event":"PhaseCompleted","phase":"config","duration_seconds":0.02}
{"ts":"2026-01-23T10:00:01Z","event":"PhaseStarted","phase":"aggregations","action":"initializing"}
{"ts":"2026-01-23T10:00:01Z","event":"PhaseCompleted","phase":"aggregations","duration_seconds":0.01}
{"ts":"2026-01-23T10:00:01Z","event":"PhaseStarted","phase":"schema_validation","action":"validating"}
{"ts":"2026-01-23T10:00:01Z","event":"PhaseCompleted","phase":"schema_validation","duration_seconds":0.01}
{"ts":"2026-01-23T10:00:32Z","event":"ProgressEvent","rows_processed":1,"rows_succeeded":0,"elapsed_seconds":32.1}
{"ts":"2026-01-23T10:05:00Z","event":"RunCompleted","run_id":"abc123","status":"completed","exit_code":0}
```

### File Changes

| File | Change |
|------|--------|
| `src/elspeth/contracts/events.py` | **NEW** - Domain event dataclasses and enums |
| `src/elspeth/contracts/__init__.py` | Export new events and enums |
| `src/elspeth/core/events.py` | **NEW** - EventBus, NullEventBus, EventBusProtocol |
| `src/elspeth/core/__init__.py` | Export EventBus, EventBusProtocol, NullEventBus |
| `src/elspeth/engine/orchestrator.py` | **BREAKING:** Remove `on_progress` param, accept `event_bus`, emit events at phases, hybrid progress timing |
| `src/elspeth/cli.py` | Create EventBus, subscribe formatters, pass to orchestrator (remove on_progress usage) |

**contracts/__init__.py additions:**

```python
from elspeth.contracts.events import (
    PhaseAction,
    PhaseCompleted,
    PhaseError,
    PhaseStarted,
    PipelinePhase,
    RunCompleted,
    RunCompletionStatus,
)

# Add to __all__:
"PhaseAction",
"PhaseCompleted",
"PhaseError",
"PhaseStarted",
"PipelinePhase",
"RunCompleted",
"RunCompletionStatus",
```

**core/__init__.py additions:**

```python
from elspeth.core.events import EventBus, EventBusProtocol, NullEventBus
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All rows succeeded |
| 1 | Partial success (some rows failed/quarantined) |
| 2 | Run failed to start or crashed |

## Testing Strategy

1. **Unit tests for EventBus**
   - Subscribe and emit behavior
   - Multiple subscribers to same event type
   - Unsubscribed events silently ignored (no crash)
   - **Handler exception propagation** - verify that formatter bugs crash immediately (validates CLAUDE.md "formatters are our code" contract)

2. **Unit tests for NullEventBus**
   - Subscribe is no-op (doesn't crash)
   - Emit is no-op (doesn't crash)
   - Satisfies EventBusProtocol (type checking)

3. **Unit tests for formatters**
   - Console formatter: each event type produces expected output
   - JSON formatter: each event type produces valid JSON
   - Formatter output comparison: same events, different formats

4. **Integration test with injectable clock**
   - Verify hybrid timing logic (first row, 100 rows, 5 seconds)
   - Deterministic progress emission with mocked clock

5. **Phase ordering tests**
   - Verify phases emit in correct sequence (config → graph → plugins → ... → process)
   - Verify PhaseError interrupts sequence appropriately

6. **Container smoke test** - verify output appears in CI (existing test remains)

## Migration

**Breaking change:** The existing `on_progress: Callable[[ProgressEvent], None] | None` parameter is removed from `Orchestrator.__init__()`. This eliminates dual mechanisms for observability.

**Migration path for callers:**
- **CLI usage:** No changes needed - CLI will create EventBus and subscribe formatters
- **Programmatic/library usage:** Pass `event_bus=NullEventBus()` (or omit, it's the default) if you don't need observability
- **Programmatic usage with progress callbacks:** Subscribe to events explicitly:
  ```python
  bus = EventBus()
  bus.subscribe(ProgressEvent, lambda evt: print(f"Progress: {evt.rows_processed}"))
  orchestrator = Orchestrator(db, event_bus=bus)
  ```

**Rationale:** Per CLAUDE.md's "No Legacy Code Policy," keeping both `on_progress` callback and EventBus subscription would create dual mechanisms doing the same thing. The EventBus pattern is more flexible (multiple subscribers, multiple event types) and `on_progress` is subsumed by it. Clean migration eliminates technical debt.

## Alternatives Considered

1. **Callback-based approach** - Rejected: couples orchestrator to CLI presentation
2. **Pure structlog** - Rejected: harder to format human-readable output
3. **Row-only progress** - Rejected: silent for slow LLM pipelines
4. **Time-only progress** - Rejected: loses determinism for testing

## Open Questions

None - all review feedback addressed, design is ready for implementation.

## Review History

**2026-01-23 - Initial Review:**
- Reviewed by `axiom-system-architect:architecture-critic` (agent a20930c)
- Reviewed by `pr-review-toolkit:code-reviewer` (agent a313b7c)
- **Architecture quality score:** 4/5
- **Findings:** 0 critical, 0 high, 2 medium, 3 low severity issues
- **All findings addressed in this revision**
