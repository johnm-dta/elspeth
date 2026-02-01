# Type Safety Lockdown for RC2

**Status:** ✅ IMPLEMENTED (2026-02-01)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all avoidable `Any` types and `# type: ignore` directives to achieve strict type safety before RC2 release. After this plan, the only remaining `Any` usage should be genuinely dynamic data (row data, external API responses, plugin configs).

**Scope:** Address all type safety gaps identified in the comprehensive audit:
1. PayloadStore forward reference typing (8 files)
2. CLI type ignores from protocol structural typing (3 locations)
3. Blob source `row: Any` parameter
4. Telemetry lazy import typing (2 files)

**Bead:** TBD (create with `bd create --title="Type safety lockdown for RC2" --type=task --priority=1`)

---

## Implementation Summary

- PayloadStore typing wired through core entry points (`src/elspeth/core/landscape/recorder.py`, `src/elspeth/engine/processor.py`, `src/elspeth/engine/tokens.py`, `src/elspeth/engine/orchestrator.py`, `src/elspeth/cli.py`).
- Telemetry exporter lazy imports typed (`src/elspeth/telemetry/exporters/otlp.py`, `src/elspeth/telemetry/exporters/azure_monitor.py`).
- CLI protocol instantiation no longer relies on `# type: ignore` for plugin collections (`src/elspeth/cli.py`).

## Task 1: Fix PayloadStore Forward References

**Files:**
- `src/elspeth/core/landscape/recorder.py:119`
- `src/elspeth/engine/processor.py:108`
- `src/elspeth/engine/tokens.py:46`
- `src/elspeth/engine/orchestrator.py:519, 751, 1853, 1960`
- `src/elspeth/cli.py:1335`

**Context:** These files use `payload_store: Any` to avoid circular imports. The `PayloadStore` protocol is defined in `contracts/payload_store.py` and can be imported safely using `TYPE_CHECKING` blocks.

### Step 1: Update recorder.py

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
```

Change line 119 from:
```python
    def __init__(self, db: LandscapeDB, *, payload_store: Any | None = None) -> None:
```

To:
```python
    def __init__(self, db: LandscapeDB, *, payload_store: PayloadStore | None = None) -> None:
```

### Step 2: Update processor.py

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
```

Change line 108 from:
```python
        payload_store: Any = None,
```

To:
```python
        payload_store: PayloadStore | None = None,
```

### Step 3: Update tokens.py

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
```

Change line 46 from:
```python
    def __init__(self, recorder: LandscapeRecorder, *, payload_store: Any = None) -> None:
```

To:
```python
    def __init__(self, recorder: LandscapeRecorder, *, payload_store: PayloadStore | None = None) -> None:
```

### Step 4: Update orchestrator.py

Add to imports (if not already present):
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
```

Change lines 519, 751, 1853, 1960 from:
```python
        payload_store: Any = None,
```

To:
```python
        payload_store: PayloadStore | None = None,
```

### Step 5: Update cli.py

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
```

Change line 1335 from:
```python
    payload_store: Any,
```

To:
```python
    payload_store: PayloadStore | None,
```

### Step 6: Remove unused Any imports

In each file, check if `Any` is still used elsewhere. If not, remove it from the typing imports.

### Step 7: Verify

Run: `.venv/bin/python -m mypy src/elspeth/core/landscape/recorder.py src/elspeth/engine/processor.py src/elspeth/engine/tokens.py src/elspeth/engine/orchestrator.py src/elspeth/cli.py --no-error-summary`
Expected: No new errors

### Step 8: Commit

```bash
git add src/elspeth/core/landscape/recorder.py src/elspeth/engine/processor.py src/elspeth/engine/tokens.py src/elspeth/engine/orchestrator.py src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
refactor: replace payload_store: Any with proper PayloadStore typing

Use TYPE_CHECKING blocks and forward references to properly type
payload_store parameters without circular imports.

Files updated:
- recorder.py: PayloadStore | None
- processor.py: PayloadStore | None
- tokens.py: PayloadStore | None
- orchestrator.py: PayloadStore | None (4 locations)
- cli.py: PayloadStore | None

All payload_store parameters now have proper protocol typing.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: ~~Fix Blob Source Row Parameter~~ SKIPPED

**Status:** ⏭️ **SKIPPED - Intentionally Correct**

**Analysis:** The `row: Any` parameter in `_validate_and_yield` is **correct** because:

1. **Tier 3 Trust Boundary** - This is a source plugin processing external data
2. **SourceRow.row: Any** - The contract explicitly uses `Any` with comment:
   *"row is Any (not dict) because quarantined rows from external data may not be dicts (e.g., JSON arrays containing primitives like numbers)"*
3. **Method docstring** confirms: *"May be non-dict for malformed external data"*
4. **Valid JSON can be non-dict**: `[1,2,3]` (list), `"string"`, `42`, `null`

The source must accept non-dict inputs to quarantine them properly. Changing to `dict[str, Any]` would **break** the ability to handle malformed JSON.

**No action required.**

---

## Task 3: Fix Telemetry Lazy Import Typing

**Files:**
- `src/elspeth/telemetry/exporters/otlp.py:108`
- `src/elspeth/telemetry/exporters/azure_monitor.py:64`

**Context:** These files use `Any` for lazy-imported exporter types. We can use `TYPE_CHECKING` blocks to provide proper types.

### Step 1: Update otlp.py

Check what type `OTLPSpanExporter` should be:

```bash
grep -n "OTLPSpanExporter" src/elspeth/telemetry/exporters/otlp.py
```

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
```

Change line 108 from:
```python
        self._span_exporter: Any = None  # OTLPSpanExporter, typed as Any for lazy import
```

To:
```python
        self._span_exporter: OTLPSpanExporter | None = None
```

### Step 2: Update azure_monitor.py

Check what type `AzureMonitorTraceExporter` should be:

```bash
grep -n "AzureMonitorTraceExporter" src/elspeth/telemetry/exporters/azure_monitor.py
```

Add to imports:
```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
```

Change line 64 from:
```python
        self._azure_exporter: Any = None  # AzureMonitorTraceExporter
```

To:
```python
        self._azure_exporter: AzureMonitorTraceExporter | None = None
```

### Step 3: Verify

Run: `.venv/bin/python -m mypy src/elspeth/telemetry/exporters/otlp.py src/elspeth/telemetry/exporters/azure_monitor.py --no-error-summary`
Expected: No errors (or only import-not-found if packages not installed)

### Step 4: Commit

```bash
git add src/elspeth/telemetry/exporters/otlp.py src/elspeth/telemetry/exporters/azure_monitor.py
git commit -m "$(cat <<'EOF'
refactor(telemetry): type lazy-imported exporters properly

Use TYPE_CHECKING blocks to provide proper types for lazy-imported
telemetry exporters:
- otlp.py: OTLPSpanExporter | None
- azure_monitor.py: AzureMonitorTraceExporter | None

This maintains lazy import behavior while providing type safety.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Investigate CLI Type Ignores

**Files:**
- `src/elspeth/cli.py:503, 532, 558, 566-568, 805, 829-831, 1375-1381`

**Context:** The CLI has many `# type: ignore` directives when building `PipelineConfig`. The issue is that:
1. Plugin manager returns `TransformProtocol`
2. `PipelineConfig.transforms` expects `list[RowPlugin]` where `RowPlugin = TransformProtocol | GateProtocol`
3. Mypy can't verify structural protocol conformance for concrete plugin classes

### Step 1: Analyze the root cause

The type ignores fall into categories:

**Category A: Transform list append (lines 532, 558, 805, 1375)**
```python
transforms.append(transform_cls(plugin_options))  # type: ignore[arg-type]
```
Issue: `transform_cls(...)` returns instance but mypy can't verify it's `RowPlugin`

**Category B: PipelineConfig construction (lines 566-568, 829-831, 1379-1381)**
```python
pipeline_config = PipelineConfig(
    source=source,  # type: ignore[arg-type]
    transforms=transforms,  # type: ignore[arg-type]
    sinks=sinks,  # type: ignore[arg-type]
```
Issue: Mypy can't verify protocol conformance

**Category C: Sink assignment (line 503)**
```python
sinks[sink_name] = sink_cls(sink_options)  # type: ignore[assignment]
```
Issue: Same as above

### Step 2: Fix with explicit protocol typing

The cleanest solution is to type the local variables explicitly:

**For transforms list:**
```python
transforms: list[RowPlugin] = []
```

**For sinks dict:**
```python
sinks: dict[str, SinkProtocol] = {}
```

**For source:**
```python
source: SourceProtocol = source_cls(source_options)
```

Then the PipelineConfig construction should work without ignores.

### Step 3: Update cli.py - instantiate_plugins_from_config function

Find the function (around line 460) and update variable declarations:

```python
def instantiate_plugins_from_config(
    config: ElspethSettings,
    manager: PluginManager,
    *,
    graph: ExecutionGraph | None = None,
) -> PipelineConfig:
    """Instantiate all plugins from configuration."""

    # Explicitly typed collections for mypy
    transforms: list[RowPlugin] = []
    sinks: dict[str, SinkProtocol] = {}

    # ... rest of function
```

Remove the `# type: ignore` comments as they become unnecessary.

### Step 4: Update cli.py - resume_run function

Find the function and apply the same pattern.

### Step 5: Update cli.py - _verify_run function

Find the function and apply the same pattern.

### Step 6: Verify

Run: `.venv/bin/python -m mypy src/elspeth/cli.py --no-error-summary`
Expected: Fewer type ignores (ideally zero for these patterns)

### Step 7: Commit

```bash
git add src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): eliminate type ignores with explicit protocol typing

Add explicit type annotations to local variables in plugin
instantiation functions:
- transforms: list[RowPlugin]
- sinks: dict[str, SinkProtocol]
- source: SourceProtocol

This allows mypy to verify PipelineConfig construction without
type ignore directives.

Removed ~15 type ignore comments.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Audit Remaining Type Ignores

**Files:**
- All files in `src/elspeth/`

**Context:** After Tasks 1-4, audit remaining `# type: ignore` comments to ensure they're all justified.

### Step 1: List all remaining type ignores

Run:
```bash
grep -rn "# type: ignore" src/elspeth/ --include='*.py' | grep -v ".pyc" | wc -l
```

### Step 2: Categorize remaining ignores

**Acceptable categories (should remain):**
- `# type: ignore[empty-body]` - Pluggy hook specs
- `# type: ignore[misc]` - MCP/external library decorators
- `# type: ignore[import-not-found]` - Optional imports
- `# type: ignore[attr-defined]` - Dynamic plugin discovery
- `# type: ignore[override]` - Intentional signature override

**Should investigate:**
- Any `# type: ignore[arg-type]` that wasn't addressed
- Any `# type: ignore[assignment]` that wasn't addressed
- Undocumented type ignores

### Step 3: Document remaining ignores

For each remaining type ignore, ensure there's a comment explaining why it's necessary:

```python
# type: ignore[empty-body]  # Pluggy hookspec - body intentionally empty
```

### Step 4: Commit documentation updates

```bash
git add src/elspeth/
git commit -m "$(cat <<'EOF'
docs: document remaining type ignore directives

All remaining # type: ignore comments now have explanatory comments
describing why they are necessary:
- Pluggy hookspecs (empty-body)
- MCP library untyped decorators (misc)
- Optional imports (import-not-found)
- Dynamic plugin discovery (attr-defined)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Run Full Type Check

**Context:** Verify the entire codebase passes mypy with no avoidable errors.

### Step 1: Run mypy on entire codebase

Run: `.venv/bin/python -m mypy src/elspeth/ --no-error-summary`

### Step 2: Fix any new errors

If mypy reports new errors from the changes, fix them.

### Step 3: Run tests

Run: `.venv/bin/python -m pytest tests/ -v --tb=short`
Expected: PASS

### Step 4: Commit any fixes

```bash
git add src/elspeth/
git commit -m "$(cat <<'EOF'
fix: resolve mypy errors from type safety improvements

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final Verification

### Step 1: Count remaining Any usage

Run:
```bash
grep -rn ": Any" src/elspeth/ --include='*.py' | grep -v "dict\[str, Any\]" | grep -v "# Legitimate" | wc -l
```

Document remaining `Any` usage and verify each is legitimate.

### Step 2: Count remaining type ignores

Run:
```bash
grep -rn "# type: ignore" src/elspeth/ --include='*.py' | wc -l
```

Target: Reduced from 40+ to <30 (only genuinely necessary ones)

### Step 3: Run ruff

Run: `.venv/bin/python -m ruff check src/elspeth/`
Expected: No errors

### Step 4: Close bead

```bash
bd close <bead-id> --reason="Type safety lockdown complete: PayloadStore properly typed, CLI type ignores eliminated, telemetry exporters typed, blob source fixed. Remaining type ignores are justified and documented."
```

---

## Summary

| Task | Files | Change | Impact |
|------|-------|--------|--------|
| Task 1 | 5 files | PayloadStore forward refs | -8 `Any` usages |
| Task 2 | — | ~~blob_source row param~~ | ⏭️ SKIPPED (correct as-is) |
| Task 3 | 2 files | Telemetry exporter types | -2 `Any` usages |
| Task 4 | 1 file | CLI explicit typing | -15 type ignores |
| Task 5 | All | Document remaining | Clarity |
| Task 6 | All | Full verification | Correctness |

**Expected Outcome:**
- `Any` usage reduced to only genuinely dynamic data
- `# type: ignore` reduced to only external library limitations
- Full mypy compliance with strict typing

**Remaining Acceptable `Any` Usage After This Plan:**
1. `row: dict[str, Any]` - Pipeline row data (inherently dynamic)
2. `config: dict[str, Any]` - Plugin configuration (inherently dynamic)
3. External API responses at Tier 3 boundaries
4. SQLAlchemy session parameters (library limitation)
5. Canonical JSON normalization (polymorphic by design)

**Remaining Acceptable Type Ignores After This Plan:**
1. Pluggy hookspecs (`empty-body`)
2. MCP library decorators (`misc`, `untyped-decorator`)
3. Optional imports (`import-not-found`)
4. NetworkX graph storage (`assignment`)
5. Intentional signature overrides (`override`)
