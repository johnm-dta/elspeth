# Root Cause Analysis: P0 Source Row Payloads Never Persisted

**Date:** 2026-01-22
**Bug:** P0-2026-01-22-source-row-payloads-never-persisted
**Analysis Method:** Systematic Debugging (Phase 1-2)

---

## Phase 1: Root Cause Investigation

### Evidence Gathered

#### 1. Working Example: `resume` Command

**File:** `src/elspeth/cli.py` (lines 918-941)

```python
def resume(...):
    # Get payload store from settings
    from elspeth.core.payload_store import FilesystemPayloadStore

    payload_path = settings_config.payload_store.base_path
    payload_store = FilesystemPayloadStore(payload_path)  # ✓ CREATED

    # ...

    result = orchestrator.resume(
        resume_point=resume_point,
        config=pipeline_config,
        graph=graph,
        payload_store=payload_store,  # ✓ PASSED TO ORCHESTRATOR
        settings=settings_config,
    )
```

**Status:** ✓ PayloadStore successfully created and wired

#### 2. Broken Example: `run` Command

**File:** `src/elspeth/cli.py` (lines 269-396)

```python
def _execute_pipeline(config: ElspethSettings, verbose: bool = False) -> ExecutionResult:
    # ... source, transforms, sinks instantiated ...

    # Execute via Orchestrator
    orchestrator = Orchestrator(db)
    result = orchestrator.run(
        pipeline_config,
        graph=graph,
        settings=config,
        on_progress=_print_progress,
        # ✗ NO payload_store parameter passed
    )
```

**Status:** ✗ PayloadStore never instantiated, never passed

#### 3. Orchestrator Signature Comparison

**Working:** `orchestrator.resume()` signature
**File:** `src/elspeth/engine/orchestrator.py` (lines 1130-1138)

```python
def resume(
    self,
    resume_point: "ResumePoint",
    config: PipelineConfig,
    graph: ExecutionGraph,
    *,
    payload_store: Any = None,  # ✓ ACCEPTS payload_store
    settings: "ElspethSettings | None" = None,
) -> RunResult:
```

**Broken:** `orchestrator.run()` signature
**File:** `src/elspeth/engine/orchestrator.py` (lines 398-405)

```python
def run(
    self,
    config: PipelineConfig,
    graph: ExecutionGraph | None = None,
    settings: "ElspethSettings | None" = None,
    batch_checkpoints: dict[str, dict[str, Any]] | None = None,
    on_progress: Callable[[ProgressEvent], None] | None = None,
    # ✗ NO payload_store parameter
) -> RunResult:
```

**Status:** ✗ `run()` does not accept `payload_store` at all

#### 4. TokenManager Call Chain

**File:** `src/elspeth/engine/tokens.py` (lines 73-78)

```python
def create_initial_token(
    self,
    run_id: str,
    source_node_id: str,
    row_index: int,
    row_data: dict[str, Any],
) -> TokenInfo:
    row = self._recorder.create_row(
        run_id=run_id,
        source_node_id=source_node_id,
        row_index=row_index,
        data=row_data,
        # ✗ payload_ref NOT passed (defaults to None)
    )
```

`TokenManager` doesn't have access to `PayloadStore` because `Orchestrator.run()` never passes it down.

---

## Phase 2: Pattern Analysis

### Comparison Table: Working vs Broken

| Step | `resume` (Working) | `run` (Broken) | Status |
|------|-------------------|----------------|--------|
| **1. CLI instantiates PayloadStore** | ✓ Line 918-925 | ✗ Not done | Missing |
| **2. Orchestrator accepts payload_store** | ✓ Line 1136 | ✗ No parameter | Missing |
| **3. Orchestrator passes to TokenManager** | ✓ (presumably) | ✗ Can't pass what it doesn't have | Missing |
| **4. TokenManager stores payload** | ✓ (presumably) | ✗ Never happens | Missing |
| **5. payload_ref passed to create_row** | ✓ (presumably) | ✗ Always None | Missing |

### Configuration Evidence

Both commands use the same settings schema:

**File:** `src/elspeth/core/config.py`

```python
class PayloadStoreSettings(BaseModel):
    backend: Literal["filesystem"] = "filesystem"
    base_path: Path = Path("./state/payloads")
    retention_days: int = 90
```

The configuration exists and is loaded by both `run` and `resume`. Only `resume` uses it.

### Key Architectural Finding

The `PayloadStore` infrastructure is **complete and working**:

- ✓ `FilesystemPayloadStore` implementation exists
- ✓ `store()` and `retrieve()` methods work
- ✓ Used successfully by `resume` command
- ✓ Used successfully by `purge` command
- ✓ Configuration schema exists

**The ONLY missing piece:** Wiring it into the `run` command path.

---

## Root Cause Statement

**The P0 bug is caused by incomplete implementation of the `run` command.**

The `PayloadStore` was implemented for recovery scenarios but never integrated into the normal run path. This is evident from:

1. `Orchestrator.resume()` has `payload_store` parameter
2. `Orchestrator.run()` does NOT have `payload_store` parameter
3. CLI `resume` instantiates and passes `PayloadStore`
4. CLI `run` never instantiates `PayloadStore`

This is a **wiring gap**, not a logic bug. The fix requires:
1. Adding `payload_store` parameter to `Orchestrator.run()`
2. Instantiating `PayloadStore` in CLI `run` command
3. Passing it through the call chain to `TokenManager`
4. Calling `payload_store.store()` before `create_row()`

---

## CLAUDE.md Violation

From `CLAUDE.md` line 23:

> **Data storage points (non-negotiable):**
> 1. **Source entry** - Raw data stored before any processing

This is explicitly listed as the FIRST non-negotiable storage point. The current implementation violates this requirement for normal runs.

---

## Impact Confirmation

- **Audit Trail Incomplete:** Every run has rows with NULL `source_data_ref`
- **Resume Fails:** Recovery expects payloads to exist but they don't
- **explain() Broken:** Cannot show raw source data, only hashes
- **Compliance Risk:** Violates documented audit requirements

This confirms P0 severity.

---

## Next Steps (Phase 3-4)

1. **Create failing test** demonstrating `source_data_ref` is NULL after run
2. **Implement fix** by wiring `PayloadStore` into `run` path
3. **Verify** test passes and `source_data_ref` is populated
