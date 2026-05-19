# Test Infrastructure Audit — 2026-03-01

## Executive Summary

**Problem:** Tests create their own mocks, landscape databases, plugin contexts, and plugin classes instead of using the existing factory/fixture infrastructure. This causes tests to break whenever production code changes, because each test has its own brittle coupling to internal APIs.

**Scale:** ~900+ violations across ~120+ files out of 499 total test files.

**Root cause:** The factory infrastructure (`make_context()`, `make_operation_context()`, `ListSource`, `CollectSink`, etc.) exists and is well-designed, but was not adopted consistently during the test suite v2 migration.

**Triggering incident:** `test_resume_validate_then_write_succeeds` was constructing `PluginContext(...)` directly with a fabricated `operation_id`, bypassing the FK chain required by the Landscape database. A single production change to enforce FK constraints broke it. Using `make_operation_context()` would have prevented this entirely.

---

## Available Infrastructure (What Tests SHOULD Use)

| Factory/Fixture | Location | Purpose |
|---|---|---|
| `make_context()` | `tests/fixtures/factories.py` | PluginContext with Mock landscape (most common need) |
| `make_source_context()` | `tests/fixtures/factories.py` | PluginContext with real landscape + FK chain (run→node) |
| `make_operation_context()` | `tests/fixtures/factories.py` | Full FK chain (run→node→operation) for `record_call()` |
| `make_landscape_db()` | `tests/fixtures/landscape.py` | `LandscapeDB.in_memory()` wrapper |
| `make_recorder()` | `tests/fixtures/landscape.py` | `LandscapeRecorder` factory |
| `landscape_db` fixture | `tests/fixtures/landscape.py` | Pytest fixture for LandscapeDB |
| `recorder` fixture | `tests/fixtures/landscape.py` | Pytest fixture for LandscapeRecorder |
| `MockPayloadStore` | `tests/fixtures/stores.py` | In-memory PayloadStore |
| `MockClock` | `tests/fixtures/stores.py` | Deterministic clock |
| `ListSource` | `tests/fixtures/plugins.py` | Source from list |
| `CollectSink` | `tests/fixtures/plugins.py` | Sink that collects results |
| `PassTransform` | `tests/fixtures/plugins.py` | Identity transform |
| `FailTransform` | `tests/fixtures/plugins.py` | Always-error transform |
| `ConditionalErrorTransform` | `tests/fixtures/plugins.py` | Error on truthy `fail` key |
| `CountingTransform` | `tests/fixtures/plugins.py` | Counts invocations |
| `ErrorOnNthTransform` | `tests/fixtures/plugins.py` | Error on Nth call |
| `build_linear_pipeline()` | `tests/fixtures/pipeline.py` | Production-path pipeline builder |
| `build_fork_pipeline()` | `tests/fixtures/pipeline.py` | Fork/join pipeline builder |
| `wire_transforms()` | `tests/fixtures/factories.py` | WiredTransform list builder |
| `create_observed_contract()` | `tests/fixtures/base_classes.py` | SchemaContract from row dict |

---

## Findings by Category

### Category A: Direct `PluginContext(...)` Construction (359 occurrences across 60 files)

Tests that call `PluginContext(run_id=..., config={}, landscape=...)` directly instead of `make_context()`, `make_source_context()`, or `make_operation_context()`.

**Severity: HIGH** — This is the exact pattern that caused the triggering incident.

| Directory | Files | Occurrences | Key Offenders |
|---|---|---|---|
| `unit/plugins/` | 25+ files | ~160 | `test_openrouter.py` (54), `test_azure.py` (21), `test_batching/test_batch_transform_mixin.py` (14), `test_azure_multi_query.py` (13), `test_sink_display_headers.py` (10), `test_azure_multi_query_retry.py` (10) |
| `unit/engine/` | 5 files | ~65 | `test_processor.py` (52 alone), `test_batch_token_identity.py` (3) |
| `unit/core/` | 1 file | 10 | `test_validation_error_noncanonical.py` |
| `unit/contracts/` | 7 files | ~30 | `test_csv_sink_contract.py` (11), `test_csv_source_contract.py` (7) |
| `unit/regression/` | 1 file | 5 | `test_phase8_sweep_d_validation.py` |
| `integration/` | 8 files | ~30 | `test_error_persistence.py` (7), `test_retry.py` (3), `test_fixes.py` (2), `test_rate_limit/test_integration.py` (3) |
| `performance/` | 2 files | ~23 | `test_llm_retry.py` (22), `conftest.py` (1) |
| `property/` | 2 files | ~22 | `test_operations_properties.py` (20), `test_csv_sink_properties.py` (2) |

**Legitimate exclusions (tests that test PluginContext itself):**

- `tests/unit/plugins/test_context.py` (35 occurrences) — tests PluginContext behavior directly
- `tests/unit/contracts/test_context_protocols.py` (2 occurrences) — tests protocol compliance
- `tests/fixtures/factories.py` (3 occurrences) — factory implementations

These files should NOT be refactored to use factories — direct construction is the system under test or the factory itself.

#### PluginContext Field Usage Analysis

Before bulk-replacing `PluginContext(...)` with `make_context()`, we must verify that `make_context()` exposes all fields tests actually need. Analysis of the 8 highest-volume files:

| Field | make_context() has it? | Files using it | Occurrences |
|---|---|---|---|
| `run_id` | Yes | 7 | 173 |
| `config` | Yes | 7 | 173 |
| `landscape` | Yes | 7 | 172 |
| `state_id` | Yes | 2 | 30 |
| `token` | Yes | 2 | 29 |
| **`node_id`** | **No** | 1 | 10 |
| **`operation_id`** | **No** (but `make_operation_context()` covers this) | 1 | 8 |

**Finding:** `make_context()` covers ~95% of violations as-is. The remaining ~5% need `node_id` support. The fix is to extend `make_context()` with `node_id: str | None = None` (see Prerequisite P0.5 below), not to create a new factory.

**Fix:** Replace with `make_context()` (mock landscape) or `make_context(landscape=recorder)` (real recorder). For tests needing `node_id`, use the extended `make_context(node_id="source")`.

### Category B: Inline `LandscapeDB.in_memory()` + `LandscapeRecorder(...)` (550 occurrences across 105 files)

Tests that construct the landscape stack inline instead of using `make_landscape_db()`, `make_recorder()`, or the shared pytest fixtures. This category covers both `LandscapeDB.in_memory()` (550 occurrences across 105 files, excluding `tests/fixtures/`) and companion `LandscapeRecorder(db)` calls (478 across 98 files, excluding `tests/fixtures/`), which are nearly always co-located.

**Severity: MEDIUM-HIGH** — Duplicates 3-4 lines of setup that break if LandscapeDB/LandscapeRecorder constructors change.

| Directory | Files | `in_memory()` | `Recorder(...)` | Notes |
|---|---|---|---|---|
| `unit/plugins/` | 22 files | ~45 | ~45 | Co-located with Category A violations |
| `unit/engine/` | 7 files | ~35 | ~30 | `test_tokens.py` (21 alone!) |
| `unit/core/landscape/` | 13 files | ~65 | ~65 | Landscape subsystem tests |
| `unit/core/` (other) | 3 files | 6 | — | `test_token_outcomes.py`, `test_explicit_sink_routing_safeguards.py`, `test_config_alignment.py` |
| `unit/contracts/` | 4 files | ~20 | ~20 | `test_csv_sink_contract.py` (11), `test_csv_source_contract.py` (7) |
| `unit/mcp/` | 1 file | 6 | 6 | `test_diagnostics.py` |
| `integration/audit/` | 16 files | ~152 | ~152 | **Worst cluster** — every `test_recorder_*.py` file (tokens: 23, contracts: 18, runs: 14, nodes: 14, node_states: 14, batches: 12). **Note:** requires scoping pass before migration (see P2 prerequisite) |
| `integration/pipeline/` | 10+ files | ~55 | ~20 | `test_quarantine_routing.py` (13), `test_t18_characterization.py` (11), orchestrator tests |
| `e2e/` | 8 files | ~15 | ~15 | Pipeline + recovery tests |
| `performance/` | 8 files | ~16 | ~6 | Benchmarks + scalability |
| `property/` | 12 files | ~108 | ~58 | Hypothesis `@given` (see note below) |

**Tier-level breakdown:**

| Tier | `in_memory()` occurrences | Files |
|---|---|---|
| `tests/unit/` | 206 | 55 |
| `tests/integration/` | 220 | 28 |
| `tests/property/` | 108 | 13 |
| `tests/performance/` | 16 | 9 |

#### Common Setup Pattern (80%+ of violations)

Analysis of 451 `begin_run()` and 500 `register_node()` calls reveals a dominant pattern:

```python
# The 80% pattern — identical across most tests
db = LandscapeDB.in_memory()
recorder = LandscapeRecorder(db)
run = recorder.begin_run(config={}, canonical_version="v1")
source = recorder.register_node(
    run_id=run.run_id,
    plugin_name="source",
    node_type=NodeType.SOURCE,
    plugin_version="1.0",
    config={},
    schema_config=DYNAMIC_SCHEMA,
)
# Tests then use: db, recorder, run.run_id, source.node_id
```

Parameters are effectively constant: `config={}`, `canonical_version="v1"`, `plugin_version="1.0"`, `schema_config=DYNAMIC_SCHEMA`. The only variation is `plugin_name` and whether `node_id` is explicit (for deterministic tests) or auto-generated.

**Variant patterns (the 20%):**
- **Multi-node:** Register 2-5 additional nodes (aggregation, transform, coalesce, sink) — same `run_id`
- **With rows/tokens:** After register_node, call `create_row()` + `create_token()` — mainly integration/audit tests
- **Helper factories:** `_make_recorder()` in `test_processor.py`, `_setup()` in 8 landscape test files

**Fix for integration/unit:** Use `landscape_db`/`recorder` fixtures or `make_landscape_db()`/`make_recorder()` for simple cases. Use the new `make_recorder_with_run()` factory (see F5 below) for the 80% pattern that needs run + source node.

**Fix for property:** While `@given` tests cannot receive pytest fixtures as parameters, they CAN call factory functions inside the test body:

```python
@given(st.integers())
def test_something(n):
    db = make_landscape_db()       # Works inside @given
    recorder = make_recorder(db)   # Works inside @given
```

Property tests should use `make_landscape_db()` / `make_recorder()` rather than inline `LandscapeDB.in_memory()` / `LandscapeRecorder(db)`. This doesn't reduce line count, but routes all construction through factories so constructor changes propagate automatically. Lower priority than unit/integration fixes.

### Category C: Inline Plugin Class Definitions (20 duplicate classes across 12 files)

Tests defining their own Source/Sink/Transform classes when `ListSource`, `CollectSink`, `PassTransform`, etc. would suffice.

**Severity: MEDIUM** — Creates drift between test infrastructure and production plugin protocols.

**Verified duplicates (behavioral equivalence confirmed):**

#### PassTransform duplicates (8 classes → 0 after fix)

| Inline Class | File | Lines | Differences from `PassTransform` |
|---|---|---|---|
| `IdentityTransform` | `test_orchestrator_core.py` | 82 | None — identical logic |
| `IdentityTransform` (×4) | `test_orchestrator_checkpointing.py` | 82, 137, 183, 415 | None — repeated 4 times |
| `PassthroughTransform` | `test_orchestrator_checkpointing.py` | 231 | None — identical logic, different name |
| `PassthroughTransform` (×3) | `test_completed_outcome_timing.py` | 133, 227, 306 | None — identical logic |
| `PassthroughTransform` | `test_telemetry_contracts.py` | 107 | Minor: PipelineRow type handling |
| `SimpleTransform` | `test_wiring.py` (telemetry) | 80 | None — identical logic |
| `IdentityTransform` | `test_explicit_sink_routing.py` | 40 | None — identical logic |

#### ListSource duplicates (3 replaceable, 1 special case)

| Inline Class | File | Lines | Differences from `ListSource` |
|---|---|---|---|
| `SimpleSource` | `test_concurrency.py` | 231 | None — identical interface |
| `SimpleSource` | `test_wiring.py` (telemetry) | 64 | None — identical |
| `SimpleSource` | `test_telemetry_contracts.py` | 73 | None — identical |
| `ListSource` | `test_base.py` | 260 | **Different:** reads from `config["data"]` — keep as-is |

#### CollectSink duplicates (5 replaceable)

| Inline Class | File | Lines | Differences from `CollectSink` |
|---|---|---|---|
| `SimpleSink` | `test_concurrency.py` | 244 | Attribute named `written` instead of `results` |
| `SimpleSink` | `test_wiring.py` (telemetry) | 93 | None — identical |
| `SimpleSink` | `test_telemetry_contracts.py` | 89 | None — identical |
| `CollectingSink` | `test_aggregation_checkpoint_bug.py` | 91 | None — identical |
| `MemorySink` | `test_integration.py` | 69 | Attribute named `rows` instead of `results` |

#### Classes needing new fixtures (no existing equivalent)

| Inline Class | File | Lines | Behavior | Proposed Fixture |
|---|---|---|---|---|
| `FailingSink` (×3) | `test_completed_outcome_timing.py` | — | `write()` raises RuntimeError | F2 |
| `FailingSource` | `test_orchestrator_cleanup.py`, `test_export_partial_semantics.py` | — | `load()` raises RuntimeError | F3 |
| `FlakyTransform` | `test_retry.py` | 51 | Fails N times then succeeds | DEFERRED (1 occurrence, below threshold) |

**Excluded from count (legitimate test infrastructure, NOT duplicates):**
- Schema classes (`PluginSchema` subclasses like `SourceOutput`, `SinkInput`) — test schema validation
- Protocol test base classes (`SourceContractTestBase`, `SinkContractTestBase`) — test infrastructure
- Plugin discovery classes (`SourceOne`, `SourceTwo`) — test pluggy registration
- Behavioral test transforms (`DoubleTransform`, `AddOneTransform`, `ExpandingTransform`) — test-specific logic that fixtures cannot generalize

### Category D: Local Mock Context Helpers (6 definitions, 20+ downstream calls)

Tests defining their own `make_mock_context()` / `make_plugin_context()` helper instead of importing `make_context()` from `tests/fixtures/factories.py`.

**Severity: LOW-MEDIUM** — The `make_context()` factory already does exactly what these helpers do.

**Helper definitions found:**

| File | Function | Downstream Calls | Notes |
|---|---|---|---|
| `tests/unit/plugins/llm/conftest.py` | `make_plugin_context()` | Imported by LLM tests | **Functionally identical to `make_context()`** |
| `tests/performance/stress/conftest.py` | `make_plugin_context()` | Imported by stress tests | Same pattern |
| `tests/unit/plugins/transforms/test_keyword_filter.py` | `make_mock_context()` | 20 calls within file | Same pattern |
| `tests/unit/plugins/transforms/azure/test_content_safety.py` | `make_mock_context()` | Within file | Same pattern |
| `tests/unit/plugins/transforms/azure/test_prompt_shield.py` | `make_mock_context()` | Within file | Duplicate of content_safety |
| `tests/unit/plugins/llm/test_openrouter_multi_query.py` | Local helper | Within file | Duplicate of conftest |

**Why these are redundant:** The conftest `make_plugin_context()` (lines 39-53) constructs `PluginContext(run_id="run-123", landscape=mock, state_id=state_id, config={}, token=token)` — identical to `make_context(state_id=state_id, token=token)`.

**Fix:** Delete all 6 helper definitions. Replace calls with `from tests.fixtures.factories import make_context`.

### Category E: Manual `ExecutionGraph()` in Integration/E2E (HIGH SEVERITY) - FIXED

Tests that bypass `ExecutionGraph.from_plugin_instances()` — the BUG-LINEAGE-01 prevention rule. These have been fixed.

**Non-exempt violations (must fix):**

1. `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py:406-413` — `_setup_failed_run()` - FIXED
2. `tests/integration/plugins/sources/test_payload_storage.py:47-68` — `_build_simple_graph()` - FIXED
3. `tests/integration/pipeline/test_concurrency.py` — synthetic `DAGTraversalContext` - FIXED

**Documented exemptions (acceptable):**

- `test_topology_validation.py` — testing hashing algorithms specifically
- `test_recovery.py` — checkpoint tests need stable node IDs
- `test_aggregation_contracts.py` — testing DAG edge validation logic
- `test_resume_comprehensive.py` — checkpoint node ID matching
- Various e2e recovery tests — all carry inline documentation

---

## New Factories Needed

Based on repeated boilerplate (3+ occurrences), these new factories should be added.

**Note:** F1 from the original audit (`make_process_context`) has been **removed** — `make_context()` already accepts the same parameters (`state_id`, `token`, `landscape`, `run_id`, `config`). Instead, `make_context()` should be extended with `node_id` support (see below).

### Prerequisite: Extend `make_context()` with `node_id`

**Before any Category A remediation**, add `node_id` to `make_context()`:

```python
def make_context(
    *,
    run_id: str = "test-run",
    state_id: str = "state-123",
    token: Any | None = None,
    config: dict[str, Any] | None = None,
    landscape: Any | None = None,
    node_id: str | None = None,           # NEW — needed by 10 tests
) -> PluginContext:
```

This is a non-breaking additive change — all existing callers are unaffected.

### F2: `FailingSink` — Add to `tests/fixtures/plugins.py`

**Pattern:** Sink whose `write()` always raises `RuntimeError`.
**Currently duplicated in:** `test_completed_outcome_timing.py` (×3), `test_orchestrator_checkpointing.py`.
**Proposed:**

```python
class FailingSink(_TestSinkBase):
    name = "failing_sink"
    def write(self, rows, ctx):
        raise RuntimeError("Sink write failed")
```

### F3: `FailingSource` — Add to `tests/fixtures/plugins.py`

**Pattern:** Source whose `load()` always raises.
**Currently duplicated in:** `test_orchestrator_cleanup.py`, `test_export_partial_semantics.py`.
**Proposed:**

```python
class FailingSource(ListSource):
    name = "failing_source"
    def load(self, ctx):
        raise RuntimeError("Source failed intentionally")
```

### ~~F4: `FlakyTransform`~~ — DEFERRED

**Deferred from P0.5 scope.** Only 1 occurrence (`tests/integration/pipeline/test_retry.py`), which is below the 3+ duplication threshold for new fixtures. `ErrorOnNthTransform` already exists in `tests/fixtures/plugins.py` with similar semantics (errors on exactly call N). The semantic difference — `ErrorOnNthTransform` errors on exactly call N while `FlakyTransform` errors on calls 1 through N — is genuine but does not justify a new fixture with only 1 consumer. If retry testing creates more demand in future, revisit.

### F5: `make_recorder_with_run()` — Add to `tests/fixtures/landscape.py`

**Pattern:** `LandscapeDB.in_memory()` + `LandscapeRecorder(db)` + `begin_run()` + `register_node()`.
**Currently duplicated in:** `test_processor.py` (`_make_recorder()`), `test_tokens.py` (21 inline), `test_batch_token_identity.py` (3 inline), `test_plugin_detection.py`, and 8+ landscape test files with `_setup()` helpers. Covers the 80% setup pattern identified across 451 `begin_run()` calls.

**Distinction from `make_source_context()` / `make_operation_context()`:** Those factories return a `PluginContext` — the recorder and DB are internal and not accessible. `make_recorder_with_run()` returns a named result for tests that need to query the DB directly or call recorder methods beyond what PluginContext exposes.

**Proposed:**

```python
@dataclass
class RecorderSetup:
    """Test scaffolding — plain @dataclass (not frozen) to match PipelineResult convention.

    Note: db and recorder are mutable objects; frozen=True would only prevent
    reference reassignment without providing an immutability guarantee.
    """

    db: LandscapeDB
    recorder: LandscapeRecorder
    run_id: str
    source_node_id: str


def make_recorder_with_run(
    *,
    run_id: str | None = None,
    source_node_id: str | None = None,
    source_plugin_name: str = "source",
    canonical_version: str = "v1",
) -> RecorderSetup:
    """Create LandscapeDB + Recorder + run + source node in one call.

    Covers the 80% setup pattern: db → recorder → begin_run → register_node(SOURCE).
    Tests needing additional nodes (transforms, sinks, aggregations) can call
    recorder.register_node() on the returned recorder.

    Args:
        run_id: Explicit run ID for deterministic tests. Auto-generated if None.
        source_node_id: Explicit source node ID. Auto-generated if None.
        source_plugin_name: Plugin name for the source node (default "source").
        canonical_version: Version string for begin_run (default "v1").
            Some tests (e.g., test_processor.py) use "sha256-rfc8785-v1".
    """
```

**Design rationale:** The factory covers the common 4-step boilerplate but does NOT register transforms/gates/sinks — those are test-specific. Tests needing multi-node setups call `recorder.register_node()` on the returned recorder with the same `run_id`.

**Internal deduplication (P0.5b):** After `make_recorder_with_run()` is implemented (P0.5a), `make_source_context()` and `make_operation_context()` in `tests/fixtures/factories.py` should be refactored to delegate to it internally in a **separate PR** (P0.5b). Both currently duplicate the same 4-step boilerplate. Without this consolidation, a change to `begin_run()` or `register_node()` signatures would need updating in three factories independently — the exact class of breakage this audit exists to prevent. The separate PR isolates the risk: if the delegation introduces a regression, it affects only the 44 existing callers of these two factories, not the new factories.

**Companion helper for multi-node setup (the 20% pattern):**

```python
def register_test_node(
    recorder: LandscapeRecorder,
    run_id: str,
    node_id: str,
    *,
    node_type: NodeType = NodeType.TRANSFORM,
    plugin_name: str = "transform",
) -> str:
    """Register an additional test node with sensible defaults.

    Defaults plugin_version="1.0", config={}, schema_config=DYNAMIC_SCHEMA.
    Returns the node_id for convenience.
    """
```

This completes the solution for the 20% variant pattern where tests need 2-5 additional nodes after `make_recorder_with_run()` creates the source.

### F6: `run_audit_pipeline()` — Add to `tests/fixtures/pipeline.py`

**Pattern:** Full pipeline execution with file-based SQLite + payload store, returning named fields for audit verification.
**Currently duplicated in:** 4 e2e audit test files (`test_full_lineage.py`, `test_attributability.py`, `test_export_reimport.py`, `test_purge_integrity.py`). All 4 share an identical `_run_pipeline()` helper returning `tuple[str, LandscapeDB, FilesystemPayloadStore, CollectSink]`.

**Distinction from `make_recorder_with_run()` (F5):** F5 creates an in-memory DB for unit/integration tests. F6 creates a file-based SQLite DB at `tmp_path/audit.db` with a real `FilesystemPayloadStore`, then actually executes the pipeline via `Orchestrator.run()`. F6 is the e2e audit factory; F5 is the unit/integration setup factory.

**Proposed:**

```python
@dataclass
class AuditPipelineResult:
    """Result from run_audit_pipeline() for e2e audit verification.

    Plain @dataclass (not frozen) — test scaffolding, not audit records.
    Matches PipelineResult convention in pipeline.py.
    """

    run_id: str
    db: LandscapeDB
    payload_store: FilesystemPayloadStore
    sink: CollectSink


def run_audit_pipeline(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
    transforms: list[Any] | None = None,
) -> AuditPipelineResult:
    """Execute a linear pipeline with file-based audit trail for e2e verification.

    Creates a file-based SQLite DB (not in-memory) and FilesystemPayloadStore,
    builds a production-path pipeline via build_linear_pipeline(), runs it via
    Orchestrator.run(), and asserts RunStatus.COMPLETED.

    For tests that need the pipeline to fail, call build_linear_pipeline() and
    Orchestrator directly — this factory is for the success path only.

    Args:
        tmp_path: Pytest tmp_path fixture for DB and payload files.
        source_data: Rows to feed through the pipeline.
        transforms: Optional transforms (default: [PassTransform()]).

    Returns:
        AuditPipelineResult with run_id, db, payload_store, and sink.
    """
```

**Design rationale:** The 4 existing `_run_pipeline()` helpers are functionally identical: create file-based DB, create payload store, call `build_linear_pipeline()`, build `PipelineConfig`, run via `Orchestrator`, assert completion, return the 4-tuple. The factory eliminates the bare tuple return (which requires positional indexing) in favor of named fields via `AuditPipelineResult`.

### F7: `create_observed_contract()` — Already exists but underused

**Pattern:** `SchemaContract(mode="OBSERVED", fields=(...), locked=True)` from a row dict.
**Already in:** `tests/fixtures/base_classes.py` as `create_observed_contract(row)`.
**Duplicated in:** `test_aggregation_recovery.py`, `test_durability.py` (both define their own `_make_contract()`). Note: `test_payload_storage.py` was previously duplicated but has already been fixed to import from `tests.fixtures.base_classes`.

---

## Priority Order for Remediation

### P0 — Critical (BUG-LINEAGE-01 violations) — DONE

3 non-exempt `ExecutionGraph()` manual constructions in integration tests. All fixed.

### P0.5a — Prerequisite (Create new factories + smoke tests)

Add all new factories, extend `make_context()`, and create comprehensive smoke tests. This PR adds new code without modifying existing factories. **Hard gate: P0.5a must be merged before any P1 PR can land.** This is enforced by PR review, not CI — reviewers must verify `tests/fixtures/test_factories.py` exists before approving any P1 PR.

Without this step, P1 remediation will stall on the ~10 tests that need `node_id` and the Category C deduplication will have nowhere to point.

**Scope:**
1. Extend `make_context()` with `node_id` parameter
2. Add `FailingSink`, `FailingSource` to `tests/fixtures/plugins.py`
3. Add `RecorderSetup` dataclass, `make_recorder_with_run()`, and `register_test_node()` to `tests/fixtures/landscape.py`
4. Add `AuditPipelineResult` dataclass and `run_audit_pipeline()` to `tests/fixtures/pipeline.py`
5. Delete unused `PipelineResult` dataclass from `tests/fixtures/pipeline.py` (zero callers — dead code under no-legacy-code policy)
6. Add `tests/fixtures/test_factories.py` — comprehensive factory tests (see Factory Testing below)

### P0.5b — Prerequisite (Refactor existing factories to delegate)

Refactor `make_source_context()` and `make_operation_context()` to delegate to `make_recorder_with_run()` internally. **Separate PR from P0.5a** because this changes the internal implementation of two factories already used in ~44 locations across ~15 files. Bundling with new factory creation would make regressions hard to localize.

**Scope:**
1. Refactor `make_source_context()` to use `make_recorder_with_run()` internally
2. Refactor `make_operation_context()` to use `make_recorder_with_run()` internally
3. Add regression tests for both refactored factories to `tests/fixtures/test_factories.py` (see Factory Testing below)

**Factory Testing (`tests/fixtures/test_factories.py`):**

Factory tests MUST be created before any migration begins. A bug in a centralized factory would silently corrupt every downstream test, producing hundreds of failures with no signal about the root cause.

**P0.5a tests (new factories):**

- `make_context(node_id="x")` — verify `node_id` passes through to PluginContext
- `make_context(run_id="my-run")` — verify explicit `run_id` passes through
- `make_recorder_with_run()` — verify: (a) returned `run_id` is identical to what `begin_run()` stored (round-trip invariant), (b) `source_node_id` is registered in the DB, (c) calling `recorder.register_node()` on the returned recorder succeeds
- `make_recorder_with_run(canonical_version="sha256-rfc8785-v1")` — verify non-default `canonical_version` passes through
- `make_recorder_with_run(source_plugin_name="custom")` — verify non-default plugin name is stored
- `register_test_node()` — verify registered node exists in DB with correct `node_type`
- `FailingSink.write()` — verify it raises `RuntimeError`
- `FailingSource.load()` — verify it raises `RuntimeError`
- `run_audit_pipeline()` — verify it returns a completed run with rows in the sink
- `make_recorder_with_run()` internal assertion — factory includes a self-check (`assert setup.run_id == ...`) consistent with CLAUDE.md offensive programming patterns, so factory bugs crash immediately with context

**P0.5b tests (refactored factories):**

- `make_source_context()` — verify returned PluginContext has a real landscape (not Mock), correct `run_id`, correct `node_id`, and that `ctx.landscape.create_row()` succeeds (round-trip through delegated `make_recorder_with_run()`)
- `make_operation_context()` — verify returned PluginContext has `operation_id`, and that `ctx.landscape.record_call()` succeeds on the returned context

### P1 — High (FK chain violations like the triggering incident)

359 direct `PluginContext(...)` constructions across 60 files. Any of these could break on the next production change to PluginContext, LandscapeRecorder, or the FK schema.

**Also includes Category D:** Delete 6 local `make_mock_context()`/`make_plugin_context()` helpers and replace all call sites with `make_context()`.

### P2 — Medium-High (Inline landscape setup in non-property tests)

~442 inline `LandscapeDB.in_memory()` + `LandscapeRecorder(...)` constructions across ~92 non-property files (550 total minus ~108 property occurrences).

**Prerequisite — `integration/audit/` scoping pass:** Before migrating the `integration/audit/test_recorder_*.py` cluster (~152 occurrences across 16 files), a scoping pass must classify each file as EXEMPT or MIGRATE.

**Owner:** The scoping pass is performed during P0.5a (before any migration begins) and committed as part of the CI enforcer allowlist YAML. This is not deferred to P2 — an unresolved prerequisite with no owner or criteria is a stall point, not a prerequisite.

**Classification criteria:**
- **EXEMPT:** The test's primary assertion exercises `begin_run()`, `register_node()`, `create_row()`, `create_token()`, or other `LandscapeRecorder` methods directly. The inline `LandscapeDB.in_memory() → LandscapeRecorder(db)` construction IS the system under test. Criterion: does the test assert on recorder return values, DB state after recorder calls, or recorder error behavior?
- **MIGRATE:** The test uses the recorder only as a dependency for testing something else (e.g., creating a landscape context for an executor test). The recorder setup is incidental boilerplate.

**Expected outcome:** Based on examination of `test_recorder_tokens.py`, the likely result is that most or all 16 files are EXEMPT — because `begin_run()` / `register_node()` / `create_row()` ARE the subject of every recorder test. The scoping pass will confirm this. If the outcome is "all 16 exempt," the audit's effective Category B count for P2 drops from ~442 to ~290 (a 34% reduction). This is fine — the accurate count matters more than a large number.

**Deliverable:** A committed `config/cicd/enforce_test_factories/integration_audit.yaml` (or equivalent) with per-file EXEMPT entries and inline justification, before P2 begins.

This distinction does not apply to other P2 clusters (`unit/plugins/`, `unit/engine/`, etc.) where the landscape is always a dependency, never the system under test.

**Additional P2 scoping requirement:** Before migrating Category B targets, audit all `canonical_version` values. The proposed `make_recorder_with_run()` defaults to `canonical_version="v1"`, but `test_processor.py:_make_recorder()` uses `"sha256-rfc8785-v1"`. Tests with non-default `canonical_version` must use the explicit parameter (`make_recorder_with_run(canonical_version="sha256-rfc8785-v1")`) to avoid silently changing hash-sensitive assertions. Similarly, check for non-default arguments to `LandscapeDB.in_memory()` — `make_landscape_db()` does not forward arguments, so any non-default call sites must be documented.

### P3 — Medium (Inline plugin class deduplication)

Mechanically replace inline plugin class definitions with imports from `tests/fixtures/plugins.py`:
- 8 PassTransform duplicates → `from tests.fixtures.plugins import PassTransform`
- 3 ListSource duplicates → `from tests.fixtures.plugins import ListSource`
- 5 CollectSink duplicates → `from tests.fixtures.plugins import CollectSink`
- 2 classes using new fixtures (FailingSink, FailingSource)

### P4 — Low (Property test `@given` setup)

~108 `LandscapeDB.in_memory()` + ~58 `LandscapeRecorder(...)` constructions in property tests (13 files). Use `make_landscape_db()` / `make_recorder()` factory calls instead of direct construction. Lower priority because these are less likely to break (simpler setup patterns) and the fix is purely routing through factories, not reducing code.

---

## Migration Strategy

### Recommended Approach: Directory-at-a-Time

For ~900 violations, manual file-by-file editing is impractical. The recommended approach:

1. **Phase by directory, starting with highest-density clusters:**
   - `unit/plugins/llm/` (54 + 21 + 14 + 13 + 10 = 112 occurrences) — do first
   - `unit/engine/test_processor.py` (52 occurrences, single file) — second
   - `integration/audit/test_recorder_*.py` (~152 occurrences, 16 files) — third, **after** the scoping pass (see P2 prerequisite)
   - Remaining directories in descending occurrence count

2. **Each directory is one PR** to keep diffs reviewable (typically 200–800 lines of diff per directory). PRs must be sequential, not parallel — conftest files span directories and concurrent modification causes merge conflicts.

3. **Mechanical replacement pattern** for ~80% of violations:

   ```python
   # BEFORE (Category A)
   ctx = PluginContext(run_id="test-run", config={}, landscape=Mock())

   # AFTER
   ctx = make_context()
   ```

   ```python
   # BEFORE (Category B)
   db = LandscapeDB.in_memory()
   recorder = LandscapeRecorder(db)
   run = recorder.begin_run(config={}, canonical_version="v1")

   # AFTER
   setup = make_recorder_with_run()
   db, recorder, run_id = setup.db, setup.recorder, setup.run_id
   ```

4. **Validate each PR** before merging:
   - `.venv/bin/python -m pytest tests/` — 0 failures (xfails allowed)
   - `.venv/bin/python -m mypy src/` — no new errors
   - `.venv/bin/python -m ruff check src/` — clean
   - **Test count baseline:** Run `.venv/bin/python -m pytest --co -q` before and after the migration PR. The collected test count must not decrease. A decrease indicates a silently deleted or merged test. This catches the scenario where a migration PR accidentally drops an assertion or merges two test methods.

### What NOT to Automate

- Tests where `PluginContext` is constructed with unusual field combinations (manual review needed — see decision checklist below)
- Tests where the inline plugin class has behavioral differences from the fixture (noted in Category C tables)
- Category D helpers in conftest files (deletion requires verifying all importers)
- Performance tests (`tests/performance/`) — these have custom multi-step setup integrated with ChaosLLM infrastructure (real `LandscapeRecorder` with full FK chain, ChaosLLM server lifecycle, profiling hooks) that cannot be trivially replaced with `make_recorder_with_run()`. The `tests/performance/stress/conftest.py:make_plugin_context()` uses a real recorder, not a Mock — the exemption is for setup complexity, not Mock avoidance. These should be added to the CI enforcement allowlist with this accurate rationale.
- Tests that pass `landscape=None` intentionally to test null-landscape error paths (see decision checklist item 5)

**Category D atomic PR constraint:** The deletion of a conftest helper and ALL of its call-site replacements MUST be in a single PR. `tests/unit/plugins/llm/conftest.py:make_plugin_context()` is imported by files across `tests/unit/plugins/llm/` — deleting the definition before all importers are updated breaks test collection for the entire LLM test cluster.

**Conftest surgery clarification:** Only the `make_plugin_context()` function is deleted from the conftest — NOT the entire file. The LLM conftest also exports `make_token`, `DYNAMIC_SCHEMA`, `chaosllm_server`, and `_build_chaosllm_response`, all of which remain needed. The conftest file itself persists; only the redundant helper is removed.

**LLM directory single-PR commitment:** The `unit/plugins/llm/` directory (160+ violations) MUST be processed as a single PR, not split, because the Category D conftest helper deletion is atomic with all importers. If the diff exceeds reviewable size, the PR can be split into two sequential sub-PRs: (1) update all LLM test files to import `make_context` from `tests.fixtures.factories` alongside the existing conftest import; (2) delete `make_plugin_context()` from conftest and remove the now-unused imports. This preserves test collection at every intermediate commit.

**Factory scoping guidance:** `make_recorder_with_run()` is a plain function, not a pytest fixture. Tests using `setup_class()` that call it at class scope will share a single `LandscapeDB` across all test methods, creating ordering-dependent failures. Always call `make_recorder_with_run()` inside individual test methods or `setup_method()`, never `setup_class()`. For class-based tests that currently use `setup_class()`, continue using direct construction or convert to `setup_method()`.

**Decision checklist for Category A manual review:**

When migrating a `PluginContext(...)` call, check:
1. Does the test assert on a specific `run_id` value (e.g., `assert ctx.run_id == "specific-id"`)? → Keep explicit `run_id` via `make_context(run_id="specific-id")`
2. Does the test pass `ctx` to production code that reads `ctx.node_id`? → Use `make_context(node_id="...")`
3. Does the test assert on error messages containing `ctx.run_id`? → Keep explicit `run_id` to avoid breaking message assertions
4. Does the test use `ctx.landscape` for direct DB queries after pipeline execution? → Use `make_source_context()` or `make_operation_context()` (real recorder, not mock)
5. Does the test pass `landscape=None` intentionally to test a null-landscape error path? → Do NOT migrate — add to CI enforcer allowlist. `make_context()` unconditionally replaces `None` with a `Mock()` (factories.py lines 91–95), which would silently invalidate the test. Known instances: `test_sink_display_headers.py:145`, `test_csv_sink_headers.py:233`.
6. Does the test verify behavior when a required field is _absent_ (e.g., testing `FrameworkBugError` when `node_id is None`)? → Use explicit construction with the field absent, or use `make_context(node_id=None)` to preserve visible intent. Tests that assert on missing-field error paths must not have their intent obscured by factory defaults.
7. Does the test pass a specific `config={"key": "value"}` that affects behavior? → Carry the override through: `make_context(config={"key": "value"})`

---

## Regression Prevention

After remediation, new violations must be prevented from being introduced. The following measures should be added:

1. **AST-based CI enforcement.** Add a Python script at `scripts/cicd/enforce_test_factories.py`, modeled on the existing `enforce_tier_model.py`. The script should:
   - Walk `tests/**/*.py` with Python `ast`
   - Detect `PluginContext(` call nodes (not string/comment matches — grep has false positives)
   - Detect `LandscapeDB.in_memory()` and `LandscapeRecorder(` call nodes outside `tests/fixtures/`
   - Detect inline plugin class definitions that duplicate existing fixtures (Category C)
   - Support an explicit allowlist of `(file, function/class)` pairs for legitimate direct construction
   - Support stale/expired entry detection matching `enforce_tier_model.py`'s allowlist lifecycle management (prevents allowlist from becoming permanent technical debt)
   - Fail if any non-allowlisted match is found

   **Why AST over grep:** The project already uses AST-based enforcement for the tier model (`enforce_tier_model.py`). Grep-based checks have known failure modes: false positives from comments/strings/docstrings, no per-site allowlist mechanism (only whole-file exclusion), and false negatives via import aliasing (`from ... import PluginContext as PC`). The existing `enforce_tier_model.py` is ~1,100 lines (including allowlist YAML loading, fingerprint-based keying, expiry dates, stale detection, AST visitor classes, and multi-format output). A comparable enforcer for test factories will be **~300-500 lines** — smaller because it has fewer rule types, but still substantial if it supports per-site allowlist with fingerprints. This is a realistic scope for a single focused PR.

   **Known limitation: import aliasing.** Static AST analysis cannot resolve `from ... import PluginContext as PC; ctx = PC(...)` — the enforcer detects call sites by name node, not resolved binding. This is documented as a known blind spot. The pragmatic mitigation is code review: aliasing `PluginContext` to bypass the enforcer requires deliberate intent, which code review catches. If aliasing becomes a pattern, a follow-up pass using `ast` import tracking can be added.

   **Initial allowlist entries:**
   - `tests/unit/plugins/test_context.py` — tests PluginContext behavior directly
   - `tests/unit/contracts/test_context_protocols.py` — tests protocol compliance
   - `tests/fixtures/factories.py` — factory implementations
   - `tests/fixtures/landscape.py` — factory implementations
   - `tests/fixtures/test_factories.py` — factory smoke tests
   - `tests/performance/` — custom ChaosLLM-integrated setup (not simple Mock avoidance)
   - `tests/unit/plugins/sinks/test_sink_display_headers.py:145` — intentional `landscape=None` test
   - `tests/unit/plugins/sinks/test_csv_sink_headers.py:233` — intentional `landscape=None` test
   - Exempt `integration/audit/test_recorder_*.py` entries from P0.5a scoping pass deliverable

2. **Convention documentation.** Add a `tests/fixtures/README.md` documenting which factory to use for each scenario:
   - "I need a PluginContext for a transform test" → `make_context()`
   - "I need a PluginContext that can record calls" → `make_operation_context()`
   - "I need a PluginContext for a source with validation errors" → `make_source_context()`
   - "I need a recorder + DB to query directly" → `make_recorder_with_run()`
   - "I need additional nodes after `make_recorder_with_run()`" → `register_test_node()`
   - "I need a full e2e pipeline with audit trail" → `run_audit_pipeline()`
   - "I need a test source/sink/transform" → use `tests/fixtures/plugins.py`
   - "I need `landscape=None` to test a null-landscape error path" → construct `PluginContext` directly (add to CI allowlist)

   **Scoping guidance:** `make_recorder_with_run()` is a plain function, not a pytest fixture. Always call it inside individual test methods or `setup_method()`, never `setup_class()`. Class-scope calls share a single `LandscapeDB` across test methods, creating ordering-dependent failures.

3. **Enforcement rollout timing.** The enforcement script should be deployed in two phases:
   - **During P1-P2 remediation:** Run in "audit mode" (`--audit`) — reports violations but does not fail CI. This allows migration PRs to land without fighting the ratchet.
   - **After P2 completes:** Switch to "enforce mode" (`--enforce`) — fails CI on any non-allowlisted violation.
   - This mirrors the rollout pattern used for `enforce_tier_model.py`.

   **Enforce mode switch trigger:** The switch happens when Category A+B violations in non-exempt, non-property files reach zero, as verified by the enforcer's own audit output. This is defined by the enforcer's output, not by a specific PR boundary — the trigger is "zero violations reported" regardless of which PR achieves it. The switch PR adds `--enforce` to `ci.yaml` and must itself pass the enforcer.

   **In-flight PR freeze protocol:** Before the enforce-mode activation PR is merged, all open migration PRs (P1/P2) must be merged or closed. An open P1 PR that lands after enforce mode activates will fail CI if it contains any remaining violations. The migration author must coordinate: announce "freeze — no new migration PRs until enforce switch lands" for the final transition.

   **Audit mode monitoring:** During P1-P2, the enforcer's audit output is included in each migration PR's CI log. The migration author reviews the audit report on each PR merge to catch new violations introduced by concurrent feature work. If new violations accumulate during the migration period, they are resolved in the next migration PR rather than deferred to the enforce switch.

---

## Metrics

| Metric | Count |
|---|---|
| Total test files audited | 499 |
| Files with violations | ~120+ |
| Category A (direct PluginContext) | 359 occurrences across 60 files |
| Category B (inline `in_memory()`, excl. fixtures) | 550 occurrences across 105 files |
| Category B (inline `Recorder(...)`, excl. fixtures) | 478 occurrences across 98 files |
| Category B — unit tier | 206 `in_memory()` / 193 `Recorder()` across 55/53 files |
| Category B — integration tier | 220 `in_memory()` / 204 `Recorder()` across 28/36 files |
| Category B — property tier | 108 `in_memory()` / 58 `Recorder()` across 13/6 files |
| Category B — performance tier | 16 `in_memory()` / 6 `Recorder()` across 9/3 files |
| Category C (inline plugin classes) | 20 verified duplicates across 12 files |
| Category D (local mock context helpers) | 6 definitions, 20+ downstream calls |
| Category E (manual ExecutionGraph) | 3 non-exempt (FIXED) |
| New factories to create | 5 (F2, F3, F5–F7) + `register_test_node()` + 1 `make_context()` extension |
| Factory tests | `tests/fixtures/test_factories.py` — smoke + regression tests for all new and refactored factories |
| Internal factory refactor | `make_source_context()` + `make_operation_context()` delegate to `make_recorder_with_run()` (separate PR from new factories) |
| Dead code removal | `PipelineResult` dataclass (zero callers) |
| Estimated lines of churn | 1,800–2,700 (replaced by factory calls) |

---

## Review Log

**2026-03-01 — 4-agent review (Reality, Architecture, Quality, Systems)**

Verdict: APPROVE WITH CONDITIONS. All conditions resolved in-document.

Changes made from review:

| # | Change | Source |
|---|--------|--------|
| B1 | F6 (`run_audit_pipeline`) fully specified: `AuditPipelineResult` dataclass, complete function signature, design rationale | Architecture + Quality |
| B2 | Factory smoke tests (`tests/fixtures/test_factories.py`) added to P0.5 scope | Quality + Systems |
| B3 | `RecorderSetup` proposed as `@dataclass(frozen=True, slots=True)` (later reverted to plain `@dataclass` in review #2 — W1) | Architecture |
| B4 | CI enforcement rollout timing specified: audit mode during P1-P2, enforce mode after P2 | Systems + Quality |
| B5 | `integration/audit/` scoping pass added as P2 prerequisite — split exempt vs. migrate | Systems |
| W1 | CI grep checks replaced with AST-based enforcement (`enforce_test_factories.py`) | Architecture |
| W2 | F4 (`FlakyTransform`) deferred from P0.5 — below 3+ threshold, `ErrorOnNthTransform` may suffice | Architecture |
| W3 | Internal factory deduplication: `make_source_context`/`make_operation_context` delegate to `make_recorder_with_run` | Architecture |
| W4 | Performance tests exempted from landscape factory migration | Systems |
| W5 | `register_test_node()` helper added for 20% multi-node pattern | Architecture |
| W6 | Validation criteria specified: pytest + mypy + ruff per PR | Quality |
| I1 | `unit/core/landscape/` count corrected: ~65 across 13 files (was ~55 across 14) | Reality |
| I2 | `integration/audit/` count corrected: ~152 across 16 files (was ~165 across 15+) | Reality |
| I3 | F7 stale claim removed: `test_payload_storage.py` already imports from fixtures | Reality |
| — | Category D atomic PR constraint documented | Systems |
| — | Decision checklist for Category A manual review added | Quality |
| — | Migration ordering updated: `unit/plugins/llm/` first, `integration/audit/` third (after scoping) | Systems |
| — | PRs must be sequential (conftest conflict risk documented) | Architecture |

**2026-03-01 — 4-agent review #2 (Reality, Architecture, Quality, Systems)**

Verdict: CHANGES_REQUESTED → resolved. 4 blocking issues, 13 warnings addressed in-document.

| # | Change | Source |
|---|--------|--------|
| B1 | Decision checklist extended: items 5 (landscape=None), 6 (missing-field error paths), 7 (explicit config) | Quality + Systems |
| B2 | P0.5a/P0.5b hard gate: `test_factories.py` must exist before any P1 PR lands (PR review enforced) | Quality |
| B3 | Regression tests for `make_source_context()`/`make_operation_context()` refactoring added to P0.5b scope | Quality |
| B4 | `integration/audit/` scoping pass: owner (migration author), criteria (subject-under-test vs boilerplate), deliverable (committed allowlist YAML), timing (during P0.5a) | Systems |
| W1 | `RecorderSetup` and `AuditPipelineResult` changed from `frozen=True, slots=True` to plain `@dataclass` — test scaffolding, not audit records | Architecture |
| W2 | P0.5 split into P0.5a (new factories + tests) and P0.5b (refactor existing factories + regression tests) | Architecture |
| W3 | CI enforcer complexity estimate revised: ~100-150 lines → ~300-500 lines (realistic for per-site allowlist) | Architecture |
| W4 | `canonical_version` parameter added to `make_recorder_with_run()` — `test_processor.py` uses `"sha256-rfc8785-v1"` | Systems |
| W5 | Performance test exemption rationale corrected: ChaosLLM-integrated setup, not Mock avoidance | Systems |
| W6 | Factory smoke tests expanded: round-trip invariant, non-default parameters, internal assertions | Quality + Systems |
| W7 | Factory scoping guidance added: never call `make_recorder_with_run()` at class scope | Quality |
| W8 | `landscape=None` tests added to CI enforcer initial allowlist (`test_sink_display_headers.py:145`, `test_csv_sink_headers.py:233`) | Quality |
| W9 | Import aliasing documented as known CI enforcer limitation | Quality + Architecture |
| W10 | Stale entry detection added to CI enforcer requirements | Systems |
| W11 | Test count baseline (`--co -q` comparison) added to per-PR validation | Quality |
| W12 | Enforce mode switch trigger defined by enforcer output (zero violations), not PR boundary | Systems |
| W13 | In-flight PR freeze protocol added for enforce-mode transition | Quality + Systems |
| W14 | Audit mode monitoring: migration author reviews audit reports on each merge | Systems |
| W15 | LLM directory single-PR commitment with safe two-phase alternative documented | Systems |
| W16 | Conftest surgery clarification: only `make_plugin_context()` deleted, not entire conftest | Architecture |
| W17 | Convention documentation expanded: `landscape=None` scenario, scoping guidance | Quality |
| W18 | `PipelineResult` dead code deletion added to P0.5a scope | Architecture |
| W19 | `canonical_version` pre-audit added to P2 scoping requirements | Systems |
| W20 | `make_landscape_db()` argument forwarding gap documented in P2 scoping | Quality |

**2026-03-01 — Post-review: `populate_run()` deletion (Architecture finding)**

The 4-agent review identified `populate_run()` in `tests/fixtures/factories.py` as a BUG-LINEAGE-01-class violation: it bypassed `LandscapeRecorder` entirely, inserting directly into `runs_table`, `nodes_table`, `rows_table`, `tokens_table`, and `token_outcomes_table` via raw SQL. This is the same architectural anti-pattern (bypassing production code paths in test infrastructure) that this audit exists to prevent.

Investigation found `populate_run()` was **dead code** — zero callers across the entire test suite. The documentation (`docs/guides/test-system.md`) incorrectly claimed it had "many" users and listed it in `fixtures/landscape.py` (it was actually in `fixtures/factories.py`). The docs also referenced two non-existent companion functions (`populate_fork_run()`, `assert_lineage_complete()`).

**Actions taken:**

| # | Action | Files |
|---|--------|-------|
| 1 | Deleted `populate_run()` (119 lines of dead code with raw SQL inserts) | `tests/fixtures/factories.py` |
| 2 | Removed `populate_run()` definition from docs (92 lines), replaced with 2-line removal note | `docs/guides/test-system.md` |
| 3 | Removed references to non-existent `populate_fork_run()` and `assert_lineage_complete()` | `docs/guides/test-system.md` |
| 4 | Updated 3 remaining doc references (table row, tree diagram, description list) | `docs/guides/test-system.md` |

**2026-03-01 — P0.5a implementation: COMPLETE**

All 6 scope items implemented and verified:

| # | Scope Item | Status | Details |
|---|-----------|--------|---------|
| 1 | Extend `make_context()` with `node_id` | ✅ Done | `tests/fixtures/factories.py` — additive, non-breaking |
| 2 | Add `FailingSink`, `FailingSource` | ✅ Done | `tests/fixtures/plugins.py` — `FailingSink(_TestSinkBase)`, `FailingSource(ListSource)` |
| 3 | Add `RecorderSetup`, `make_recorder_with_run()`, `register_test_node()` | ✅ Done | `tests/fixtures/landscape.py` — plain `@dataclass`, `canonical_version` param, offensive assertions |
| 4 | Add `AuditPipelineResult`, `run_audit_pipeline()` | ✅ Done | `tests/fixtures/pipeline.py` — file-based SQLite + FilesystemPayloadStore, asserts COMPLETED |
| 5 | Delete `PipelineResult` dead code | ✅ Done | `tests/fixtures/pipeline.py` — zero callers confirmed via grep |
| 6 | Add `tests/fixtures/test_factories.py` | ✅ Done | 33 smoke tests covering all 10 plan items (lines 466-476) |

Stale doc references to `PipelineResult` and `run_pipeline()` updated in `docs/guides/test-system.md`.

**Verification:**
- pytest: 10,366 passed, 17 skipped, 3 xfailed, 0 failures
- mypy: clean (no errors)
- ruff: clean (import sorting fixed)

**2026-03-01 — P0.5b implementation: COMPLETE**

All 3 scope items implemented and verified:

| # | Scope Item | Status | Details |
|---|-----------|--------|---------|
| 1 | Refactor `make_source_context()` to delegate to `make_recorder_with_run()` | ✅ Done | `tests/fixtures/factories.py` — clean delegation, removed 4 function-level imports |
| 2 | Refactor `make_operation_context()` to delegate to `make_recorder_with_run()` | ✅ Done | `tests/fixtures/factories.py` — SOURCE path delegates directly, non-SOURCE path uses `register_test_node()` for the actual node with throwaway source |
| 3 | Add regression tests for refactored factories | ✅ Done | `tests/fixtures/test_factories.py` — 4 tests: `test_landscape_is_real_recorder`, `test_create_row_round_trip`, `test_record_call_round_trip`, `test_sink_record_call_round_trip` |

Code review finding: mypy type narrowing added to `test_create_row_round_trip` (`assert ctx.landscape is not None` / `assert ctx.node_id is not None`) — consistent with offensive programming patterns.

**Verification:**
- pytest: 10,370 passed, 17 skipped, 3 xfailed, 0 failures (4 new tests)
- mypy (`src/`): clean (no errors)
- ruff: clean

**2026-03-01 — P1 implementation: COMPLETE**

53 test files modified, 735 insertions, 1,038 deletions (net -303 lines). 21 parallel subagents + 3 code review agents.

| Metric | Count |
|---|---|
| Files modified | 53 (55 total incl. 2 pre-existing doc changes) |
| `PluginContext(...)` constructions replaced | ~350+ |
| Category D helpers deleted | 7 (`make_plugin_context` x3, `make_mock_context` x2, `_make_ctx` x2) |
| Category D call sites replaced | ~100+ |
| Import additions (`from tests.fixtures.factories import make_context`) | 48 files |
| Import removals (`PluginContext`, `Mock`, `LandscapeDB`, `LandscapeRecorder`) | ~30 files |
| Dead `if TYPE_CHECKING: pass` blocks removed | 2 files |

**Exceptions — files with `PluginContext(...)` intentionally preserved:**

| File | Reason | Count |
|---|---|---|
| `tests/fixtures/factories.py` | Factory implementation (EXEMPT) | 3 |
| `tests/unit/plugins/test_context.py` | Tests PluginContext behavior directly (EXEMPT) | all |
| `tests/unit/contracts/test_context_protocols.py` | Protocol compliance tests (EXEMPT) | all |
| `tests/performance/stress/conftest.py` | `make_plugin_context()` delegates to `make_context()` — no direct `PluginContext()` construction remains (MIGRATED in supplementary fix) | 0 |
| `tests/property/core/test_operations_properties.py` | Uses `FakePluginContext` (local dataclass, not PluginContext) | 0 (false positive) |
| `tests/unit/plugins/sinks/test_sink_display_headers.py:142` | Intentional `landscape=None` error path test | 1 |
| `tests/unit/plugins/sinks/test_csv_sink_headers.py:234` | Intentional `landscape=None` error path test | 1 |
| `tests/unit/plugins/batching/test_batch_transform_mixin.py:105` | Intentional `token=None` contract violation test — `make_context()` always creates a default token | 1 |
| `tests/unit/plugins/llm/test_azure.py:439,460` | Intentional `state_id=None` and `token=None` error path tests | 2 |
| `tests/unit/plugins/llm/test_azure.py:243,589` | MIGRATED in supplementary fix — `make_context(landscape=recorder)` handles FK chain | 0 |
| `tests/unit/plugins/llm/test_openrouter.py:576` | Intentional `state_id=None` error path test | 1 |
| `tests/unit/plugins/llm/test_azure_batch.py:66` | `_make_batch_ctx()` uses real `LandscapeRecorder` with full FK chain | 1 |
| `tests/unit/plugins/llm/test_azure_batch.py:1487,1591` | `MagicMock` landscape + no token — `make_context()` injects default token causing `record_call()` token mismatch validation failure. Reverted to direct construction. | 2 |
| `tests/unit/plugins/transforms/test_keyword_filter.py:233` | MIGRATED in supplementary fix — replaced with `make_context()` | 0 |
| `tests/unit/plugins/transforms/test_field_mapper.py:577` | Uses `contract=` parameter not supported by `make_context()` | 1 |
| `tests/unit/plugins/transforms/test_web_scrape.py` | Uses `rate_limit_registry=` and `payload_store=` params not supported by `make_context()` | 2 |
| `tests/unit/plugins/transforms/test_web_scrape_security.py` | Uses `rate_limit_registry=` and `payload_store=` params not supported by `make_context()` | 2 |
| `tests/unit/contracts/transform_contracts/test_web_scrape_contract.py` | Uses `payload_store=` param not supported by `make_context()` | 1 |
| `tests/integration/rate_limit/test_integration.py` | Uses `rate_limit_registry=` param not supported by `make_context()` | all |
| `tests/integration/plugins/sources/test_contract.py` | Uses `_TestablePluginContext(PluginContext)` subclass | all |
| `tests/integration/plugins/transforms/test_contract.py` | Uses `_TestablePluginContext(PluginContext)` subclass | all |
| `tests/integration/plugins/sources/test_trust_boundary.py` | `_TestablePluginContext` subclass remains (custom `__init__` + overridden methods); `_make_audited_context` delegates to `make_context()` (MIGRATED in supplementary fix) | subclass only |
| `tests/integration/plugins/llm/test_openrouter_batch_integration.py` | Real `LandscapeRecorder` needed for audit trail | all |
| `tests/unit/contracts/test_record_call_guards.py` | Tests `record_call()` defensive guards — intentionally passes invalid PluginContext combinations (landscape=None, XOR violations, token mismatches) | 6 |
| `tests/unit/engine/orchestrator/test_resume_failure.py` | Tests orchestrator cleanup when plugins raise `FrameworkBugError`/`AuditIntegrityError` — requires direct PluginContext construction | 3 |

**Bug found and fixed during migration:**

The `test_azure_batch.py::TestAzureBatchLLMTransformMissingResults` tests (2 tests) created `PluginContext(...)` without a `token` parameter (defaulting to `None`). `make_context()` always creates a default token via `make_token_info()`. The `PluginContext.record_call()` method (added in P2-2026-02-14) validates `ctx.token.token_id` against `landscape.get_node_state(state_id).token_id` — with a `MagicMock` landscape, the auto-generated `.token_id` attribute is a Mock object, not a string, causing `FrameworkBugError`. These 2 tests were reverted to direct `PluginContext(...)` construction. *(Note 2026-05-06: this paragraph is historical — `test_azure_batch.py` was deleted alongside the retirement of `AzureBatchLLMTransform` per ADR-020.)*

**Code review results (3 agents):**

| Reviewer | Scope | Verdict | Issues Found |
|---|---|---|---|
| Engine reviewer | 6 files, 269 tests | Clean | None — all parameter mappings correct |
| LLM reviewer | 9 files | 1 critical | `test_azure_batch.py` token injection (fixed above) |
| Plugin/sink/transform reviewer | 25 files, 548 tests | 1 minor | Unused `PluginContext` import in `test_base.py` (fixed) |

**Verification:**
- pytest: 10,370 passed, 17 skipped, 87 deselected, 3 xfailed, 0 failures
- Test count: unchanged from P0.5b baseline (10,370)

**2026-03-02 — P2 prerequisites: COMPLETE (3 scoping deliverables)**

Three P2 entry prerequisites resolved. All are analysis/documentation — no code changes required.

#### Prerequisite 1: `integration/audit/` Scoping Pass

**Result: All 14 `test_recorder_*.py` files classified as EXEMPT.** Every file's primary assertions exercise `LandscapeRecorder` methods directly — the inline `LandscapeDB.in_memory() → LandscapeRecorder(db)` construction IS the system under test. This confirms the audit's prediction ("most or all 16 exempt") and drops P2's effective Category B count from ~442 to ~290 (a 34% reduction).

| File | Classification | `in_memory()` | `Recorder(` | Justification |
|---|---|---|---|---|
| `test_recorder_contracts.py` | EXEMPT | 18 | 18 | Asserts on `begin_run` with schema_contract, `update_run_contract`, `get_run_contract`, `get_node_contracts`, contract hash integrity |
| `test_recorder_batches.py` | EXEMPT | 0 | 12 | Asserts on `create_batch`, `add_batch_member`, `complete_batch`, `get_batches_by_status`, `retry_batch`. Uses `landscape_db` fixture for DB, inline `Recorder()` |
| `test_recorder_artifacts.py` | EXEMPT | 7 | 7 | Asserts on `register_artifact`, `get_artifacts`, `get_rows`, `get_tokens`, `get_node_states`, idempotency keys |
| `test_recorder_queries.py` | EXEMPT | 7 | 7 | Asserts on `get_row`, `get_token`, `get_token_parents`, `get_routing_events`, `get_row_data` return values |
| `test_recorder_nodes.py` | EXEMPT | 12 | 12 | Asserts on `register_node`, `register_edge`, `get_nodes`, `get_edges`, schema recording |
| `test_recorder_routing_events.py` | EXEMPT | 4 | 4 | Asserts on `record_routing_event` persistence and reason payload storage |
| `test_recorder_node_states.py` | EXEMPT | 14 | 14 | Asserts on `begin_node_state`, `complete_node_state`, retry increments, hash correctness, integrity validation, `context_after` round-trip |
| `test_recorder_calls.py` | EXEMPT | 8 | 8 | Asserts on `record_call` field persistence, payload refs, duplicate rejection, request hash determinism, cross-run isolation |
| `test_recorder_runs.py` | EXEMPT | 14 | 14 | Asserts on `begin_run`, `complete_run`, `get_run`, `list_runs`, field resolution JSON Tier-1 integrity |
| `test_recorder_explain.py` | EXEMPT | 9 | 9 | Asserts on `explain_row` graceful degradation, payload status, run_id mismatch rejection |
| `test_recorder_row_data.py` | EXEMPT | 8 | 9 | Asserts on `get_row_data` across 5 explicit states + Tier-1 corruption handling |
| `test_recorder_tokens.py` | EXEMPT | 23 | 23 | Asserts on `create_row`, `create_token`, `fork_token`, `coalesce_tokens`, `expand_token`, hash correctness, atomic outcomes |
| `test_recorder_grades.py` | EXEMPT | 9 | 8 | Asserts on `compute_reproducibility_grade`, `finalize_run`, grade degradation after purge |
| `test_recorder_errors.py` | EXEMPT | 9 | 9 | Asserts on `record_transform_error` return values and persistence, `set_export_status` enum coercion |
| **TOTAL** | **14 EXEMPT** | **142** | **154** | |

**Non-recorder files in `integration/audit/` (MIGRATE candidates):**

| File | `in_memory()` | `Recorder(` | Classification | Notes |
|---|---|---|---|---|
| `test_contract_audit.py` | 7 | 7 | MIGRATE | Recorder is dependency for CSVSource contract audit trail testing |
| `test_fixes.py` | 2 | 2 | MIGRATE | Broader integration tests; only 2 of 8 tests use inline construction |
| `test_export.py` | 1 | 1 | MIGRATE | SUT is export system, not recorder |
| `test_error_persistence.py` | 0 | 0 | N/A | Already uses `landscape_db`/`recorder` fixtures |
| `test_tier1_integrity.py` | 0 | 0 | EXEMPT | Uses fixtures but asserts directly on recorder Tier-1 behavior |
| `test_sqlcipher_pipeline.py` | 0 | 1 | MIGRATE | Uses `LandscapeDB.from_url()`, not `.in_memory()` |
| `test_not_null_constraints.py` | 0 | 0 | N/A | Tests raw schema constraints, no recorder usage |

#### Prerequisite 2: `canonical_version` Pre-Audit

**Result: `canonical_version` is stored metadata, not an input to hash computation.** The `stable_hash()` and `canonical_json()` functions use the hardcoded `"sha256-rfc8785-v1"` algorithm regardless of what is stored in `runs.canonical_version`. Migration risk is limited to tests that assert on the stored `canonical_version` value itself.

**Factory default mismatch (informational):** `make_recorder_with_run()` defaults to `canonical_version="v1"`, while production uses `CANONICAL_VERSION = "sha256-rfc8785-v1"` (from `contracts/hashing.py:26`). The other factory `make_run_record()` defaults to `"sha256-rfc8785-v1"`. The ~300 existing `make_recorder_with_run()` call sites already run with the non-production `"v1"` value without affecting hash correctness.

**Distinct `canonical_version` values found in test suite:**

| Value | Occurrences | Matches factory default? |
|---|---|---|
| `"v1"` | ~300 | Yes |
| `"1.0"` | ~74 | No |
| `"sha256-rfc8785-v1"` | ~32 | No (matches production constant) |
| `"1.0.0"` | ~8 | No |
| `"test-v1"` | 4 | No |
| `"v2"` | 1 | No |
| `"test"` | 1 | No |
| `CANONICAL_VERSION` (variable) | ~12 | No (resolves to `"sha256-rfc8785-v1"`) |
| Hypothesis-generated | ~2 tests | No (random values) |

**Files requiring explicit `canonical_version` parameter if migrated to `make_recorder_with_run()`:**

| File | Current Value | Why Explicit Required |
|---|---|---|
| `tests/unit/core/landscape/test_run_lifecycle_repository.py` | `"v2"` | Line 68 asserts `canonical_version == "v2"` — tests non-default storage |
| `tests/property/core/test_landscape_recording_properties.py` | Hypothesis-generated | Line 97 asserts round-trip fidelity of arbitrary values |
| `tests/integration/audit/test_tier1_integrity.py` | `CANONICAL_VERSION` | Line 294 asserts `canonical_version == CANONICAL_VERSION` — verifies production constant (EXEMPT file) |
| `tests/fixtures/test_factories.py` | `"sha256-rfc8785-v1"` | Line 238 tests factory parameter passthrough (EXEMPT — factory tests) |

**All other files:** `canonical_version` is passed to `begin_run()` as metadata but never asserted against a specific literal value. Hash-sensitive assertions in these files compare computed-vs-computed (e.g., `assert config_hash_1 == config_hash_2`), not against literal hash strings. Safe to migrate without explicit `canonical_version` parameter.

#### Prerequisite 3: `make_landscape_db()` Argument Forwarding Gap

**Result: No gap exists.** `LandscapeDB.in_memory()` accepts zero user-facing parameters (only `cls`). All 533 calls across 98 test files use the bare `LandscapeDB.in_memory()` form with no arguments. `make_landscape_db()` is a drop-in replacement for every existing call site. No factory modifications needed.

**Updated P2 effective scope:**

| Cluster | Original Count | Adjusted Count | Notes |
|---|---|---|---|
| `integration/audit/test_recorder_*.py` | ~152 | **0** | All 14 files EXEMPT |
| `integration/audit/` (other) | ~10 | **~10** | 3 MIGRATE files |
| `unit/plugins/` | ~90 | ~90 | Unchanged |
| `unit/engine/` | ~70 | ~70 | Unchanged |
| `unit/core/landscape/` | ~130 | ~130 | Requires per-file scoping (some may be EXEMPT) |
| Other directories | ~100 | ~100 | Unchanged |
| **Total** | **~442** | **~290** | **34% reduction** |

**2026-03-02 — P1 supplementary fix: 27 missed instances across 5 files**

Independent 4-agent verification audit (Explore agents searching from different angles: remaining `PluginContext(` patterns, `PluginContext` imports in tests, `make_context(` adoption completeness, and audit report exception claims) identified 27 `PluginContext(...)` constructions that survived the initial P1 migration. 22 of 27 were concentrated in the performance stress tier (`test_llm_retry.py`), which was marked EXEMPT in the original P1 but contained simple `make_context()`-compatible patterns.

**Files fixed (27 instances total):**

| File | Instances | Pattern |
|---|---|---|
| `tests/performance/stress/test_llm_retry.py` | 22 | 14 identical `start_ctx` lines, 2 inline ctx blocks, 1 `_feed_rows` helper, removed `# type: ignore[arg-type]` |
| `tests/unit/plugins/llm/test_azure.py` | 2 | Lines 243, 589 — `make_context(landscape=recorder)` handles FK chain |
| `tests/performance/stress/conftest.py` | 1 | `make_plugin_context()` body now delegates to `make_context()` via local import; `PluginContext` moved to `TYPE_CHECKING` |
| `tests/unit/plugins/transforms/test_keyword_filter.py` | 1 | Line 233 — replaced with `make_context()`, removed unused `LandscapeDB`/`LandscapeRecorder` imports |
| `tests/integration/plugins/sources/test_trust_boundary.py` | 1 | `_make_audited_context()` body now delegates to `make_context()` via local import |

**Exceptions table updated:** 4 entries changed from preserved to MIGRATED; `test_trust_boundary.py` updated to reflect that only `_TestablePluginContext` subclass remains (not `_make_audited_context`).

**Code review (1 agent, 32 tool calls):**

| Reviewer | Scope | Verdict | Issues Found |
|---|---|---|---|
| Independent code reviewer | 5 files, 27 substitutions | Clean | None — all parameter mappings correct, import changes verified, intentionally unmigrated calls confirmed (test_azure.py:439,460 error paths, _TestablePluginContext subclass) |

Semantic preservation confirmed: `config={}` correctly dropped (factory default), `run_id` forwarded where non-default, `state_id`/`token`/`landscape`/`node_id` all correctly mapped. `# type: ignore[arg-type]` on `landscape=recorder` correctly removed (factory accepts `Any`).

**Verification:**
- pytest (non-stress): 107 passed, 0 failures
- pytest (stress collection): 19 tests collected successfully
- ruff: clean
- mypy: pre-existing errors only (5 in factories.py/test_azure.py — none introduced)

**Updated P1 totals:** ~377 `PluginContext(...)` constructions replaced across 58 files (original 350+ across 53, plus supplementary 27 across 5).

**2026-03-02 — P2: Inline landscape setup replacement — COMPLETE**

Replaced inline `LandscapeDB.in_memory()` and `LandscapeRecorder(...)` constructions with centralized factory calls across 76 files. Net: 685 insertions, 1,400 deletions (−715 lines).

**Scope:** 76 non-property, non-exempt test files. Excluded: 14 EXEMPT `test_recorder_*.py` files, property tests (out of P2 scope), `test_context.py`/`test_context_protocols.py` (test PluginContext directly), factory infrastructure files.

**Replacement patterns applied:**

| Pattern | Description | Count |
|---|---|---|
| Pattern 1 | `LandscapeDB.in_memory()` → `make_landscape_db()` | 177 |
| Pattern 2 | `LandscapeRecorder(db)` → `make_recorder(db)` | 156 |
| Pattern 3 | 4-step boilerplate → `make_recorder_with_run()` | 45 |
| Pattern 4 | Helper body → delegates to `make_recorder_with_run()` | ~10 (subset of Pattern 3) |
| `register_test_node()` | Additional node registration after Pattern 3 | 30 |

**Execution:** 29 subagents, each assigned 1–3 files (≤20 tests). Written instructions at `.claude/prompts/p2-migration-instructions.md` defined all 4 patterns, import changes, exempt criteria, and factory API reference. Each agent made mechanical replacements only.

**Residual (intentionally not migrated):**

| Category | Remaining | Files | Reason |
|---|---|---|---|
| `payload_store=` kwarg | 27 | 9 files | `make_recorder()` does not accept `payload_store` — factory gap |
| `performance/` tests | 16 `in_memory()` + 6 `Recorder(` | 8 files | Out of P2 scope (performance tests — different lifecycle) |
| EXEMPT recorder tests | 142 `in_memory()` + 154 `Recorder(` | 14 files | SUT is recorder itself — Pattern 3 would hide the code being tested |
| EXEMPT context/factory tests | 13 `in_memory()` + 14 `Recorder(` | 3 files | SUT is context/factory infrastructure |
| Property tests | 108 `in_memory()` + 58 `Recorder(` | 13 files | Out of P2 scope |
| `test_processor.py` docstring | 1 | 1 file | Comment, not code |

**Issues found and fixed during migration:**

1. **Import sorting (9 files):** Subagents inserted `tests.fixtures.landscape` imports between `elspeth.*` imports. Auto-fixed with `ruff check tests/ --fix`.
2. **Unused `db` variable (1 file, 3 occurrences):** `test_batch_token_identity.py` destructured `db` from `RecorderSetup` but never used it. Fixed by removing `db` from destructuring.

**Code review results (4 agents):**

| Reviewer | Scope | Verdict | Issues Found |
|---|---|---|---|
| High-density reviewer | 3 files, 193 tests | Clean | None |
| Medium-density reviewer | 4 files, 158 tests | Clean | None — EXEMPT files correctly Pattern 1+2 only |
| Engine/processor reviewer | 5 files, 101 tests | Clean | None — `payload_store` calls correctly preserved |
| Integration/E2E reviewer | 10 files | Clean | None — `payload_store` calls correctly left untouched |

**Verification:**
- pytest: 10,370 passed, 17 skipped, 87 deselected, 3 xfailed, 0 failures
- Test count: unchanged from P1 baseline (10,370)
- ruff: all checks passed
- mypy: no new errors introduced (pre-existing only)

**P2 coverage summary:**

| Metric | Before P2 | After P2 | Delta |
|---|---|---|---|
| `LandscapeDB.in_memory()` in migrateable scope | ~251 | 0 | −251 |
| `LandscapeRecorder(` in migrateable scope (excl. `payload_store`) | ~201 | 0 | −201 |
| Files touched | — | 76 | — |
| Net lines | — | −715 | — |

**2026-03-02 — P3: Inline plugin class deduplication — COMPLETE**

Replaced 19 inline plugin class definitions across 13 test files with imports from `tests/fixtures/plugins.py`. Implementation split across two commits: 276ec733 (core deduplication, 13 files, −327 net lines) and 3fe16f86 (code review follow-up: test relocation + fixture consolidation).

**Replacements applied:**

| Original Class | Replaced With | Files Affected | Count |
|---|---|---|---|
| `IdentityTransform` | `PassTransform` | `test_orchestrator_core.py`, `test_orchestrator_checkpointing.py` (×4), `test_explicit_sink_routing.py` | 6 |
| `PassthroughTransform` | `PassTransform` | `test_orchestrator_checkpointing.py`, `test_completed_outcome_timing.py` (×3), `test_telemetry_contracts.py` | 5 |
| `SimpleTransform` | `PassTransform` | `test_wiring.py` (telemetry) | 1 |
| `SimpleSource` | `ListSource` | `test_concurrency.py`, `test_wiring.py` (telemetry), `test_telemetry_contracts.py` | 3 |
| `SimpleSink` | `CollectSink` | `test_concurrency.py`, `test_wiring.py` (telemetry), `test_telemetry_contracts.py` | 3 |
| `CollectingSink` | `CollectSink` | `test_aggregation_checkpoint_bug.py` | 1 |
| Inline `FailingSink` (×3) | `FailingSink` (fixture) | `test_completed_outcome_timing.py`, `test_orchestrator_checkpointing.py` | 2 files |
| Inline `FailingSource` | `FailingSource` (fixture) | `test_orchestrator_cleanup.py`, `test_export_partial_semantics.py` | 2 files |
| **Total** | | **13 files** | **19 class definitions removed** |

**Additional improvements (code review follow-up, 3fe16f86):**

| # | Change | Details |
|---|--------|---------|
| I1 | Relocated `test_completed_outcome_timing.py` | `tests/unit/engine/` → `tests/integration/pipeline/orchestrator/` — tests use full `Orchestrator.run()`, belong in integration tier |
| I3 | Consolidated telemetry fixtures | `tests/unit/telemetry/fixtures.py` (240 lines) deleted → content merged into `tests/fixtures/telemetry.py` |
| — | Updated `test_manager.py` import | `from tests.unit.telemetry.fixtures` → `from tests.fixtures.telemetry` |

**Audit misclassification resolved:**

The audit listed `MemorySink` in `tests/unit/plugins/test_integration.py:69` as a `CollectSink` duplicate. Investigation found it is NOT functionally equivalent:
- Extends `BaseSink` directly (not `_TestSinkBase`)
- Uses typed `PluginSchema` (`EnrichedSchema`) for schema validation
- Returns `ArtifactDescriptor` from `write()`
- Uses `ClassVar[list]` for class-level shared storage
- Registered through pluggy `@hookimpl` in a `TestPlugin` class for integration lifecycle testing
- Audit description ("Attribute named `rows`") does not match actual attribute name (`collected`)

This class tests plugin registration infrastructure, not sink collection behavior. Correctly classified as exempt — should have been in the audit's "Excluded from count" section alongside `DoubleTransform`, `AddOneTransform`, and other behavioral test classes.

**Residual inline plugin classes (correctly preserved):**

| Class | File | Reason |
|---|---|---|
| `MemorySink` | `tests/unit/plugins/test_integration.py` | Plugin registration lifecycle test (see above) |
| `MemorySink` | `tests/unit/plugins/test_base.py` | Tests `BaseSink` base class behavior directly |
| `MemorySink` | `tests/unit/plugins/test_protocols.py` | Tests `SinkProtocol` structural compliance |
| `SimpleTransform` | `tests/unit/plugins/test_base.py` | Tests `BaseTransform` base class behavior |
| `ListSource` | `tests/unit/plugins/test_base.py` | Reads from `config["data"]` (different interface from fixture `ListSource`) |

**Code review results (5 verification agents):**

| Agent | Scope | Verdict | Issues Found |
|---|---|---|---|
| P3 plugin dedup verifier | All 20 audit targets | Complete | 1 misclassification (MemorySink — resolved above) |
| P1 PluginContext verifier | 19 files with remaining constructions | Complete | None — all documented exemptions |
| P2 landscape verifier | All `in_memory()` / `Recorder(` residuals | Complete | None — exact match on documented residuals |
| P0.5 factory verifier | All factory files + 37 smoke tests | Complete | None — full spec compliance |
| Test suite health verifier | Full pytest + ruff + mypy | Green | 10,476 tests passed, 0 lint/type errors |

**Verification:**
- pytest: 10,476 collected, 10,389 passed, 17 skipped, 87 deselected, 3 xfailed, 0 failures
- Test count: +106 from P2 baseline (10,370 → 10,476) — growth from code review additions (I2, I4-I8)
- Factory tests: 37/37 passed
- ruff: all checks passed
- mypy: 0 errors in 243 source files

**P3 coverage summary:**

| Metric | Before P3 | After P3 | Delta |
|---|---|---|---|
| Inline PassTransform/IdentityTransform/PassthroughTransform duplicates | 12 | 0 | −12 |
| Inline SimpleSource duplicates | 3 | 0 | −3 |
| Inline SimpleSink/CollectingSink duplicates | 4 | 0 | −4 |
| Inline FailingSink/FailingSource (now shared fixtures) | 5 | 0 | −5 |
| Files touched (dedup) | — | 13 | — |
| Files touched (review follow-up) | — | 36 | — |
| Net lines (dedup commit) | — | −327 | — |
| Net lines (review follow-up) | — | +720 | — |

**2026-03-02 — P4: Property test factory adoption — COMPLETE**

Replaced inline `LandscapeDB.in_memory()` and `LandscapeRecorder(...)` constructions with centralized factory calls across 13 property test files. Net: 183 insertions, 187 deletions (−4 lines).

**Scope:** 13 property test files containing 108 `LandscapeDB.in_memory()` + 58 `LandscapeRecorder(...)` constructions. All `@given`-decorated tests use factory functions inside the test body (property tests cannot receive pytest fixtures as parameters).

**Replacement patterns applied:**

| Pattern | Description | Count |
|---|---|---|
| Pattern 1 | `LandscapeDB.in_memory()` → `make_landscape_db()` | 107 |
| Pattern 2 | `LandscapeRecorder(db)` → `make_recorder(db)` | 58 |
| Pattern 3 | 4-step boilerplate → `make_recorder_with_run()` | 1 |
| **Total** | | **166** |

**Execution:** 8 subagents, each assigned 1–3 files. Written instructions at `.claude/prompts/p4-migration-instructions.md` defined all patterns, import changes, and factory API reference. Each agent made mechanical replacements only.

**Per-file breakdown:**

| File | `in_memory()` | `Recorder(` | Patterns | Notes |
|---|---|---|---|---|
| `audit/test_recorder_properties.py` | 24 | 24 | P1+P2 | Hypothesis-generated `config` — Pattern 3 inapplicable |
| `core/test_landscape_recording_properties.py` | 24 | 23 | P1+P2+P3 | `_setup()` helper → `make_recorder_with_run(canonical_version="1.0", source_plugin_name="test_source")` |
| `engine/test_processor_properties.py` | 16 | 0 | P1 | No recorders — `Orchestrator(db)` manages internally |
| `core/test_reproducibility_properties.py` | 8 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `core/test_retention_monotonicity.py` | 7 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `engine/test_token_lifecycle_state_machine.py` | 6 | 6 | P1+P2 | State machine `__init__` + `@given` tests |
| `audit/test_terminal_states.py` | 6 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `audit/test_fork_coalesce_flow.py` | 5 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `audit/test_fork_join_balance.py` | 4 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `engine/test_sink_routing_invariant.py` | 3 | 0 | P1 | `LandscapeDB` retained for type annotations |
| `sinks/test_csv_sink_properties.py` | 2 | 2 | P1+P2 | Both direct imports removed |
| `sinks/test_json_sink_properties.py` | 2 | 2 | P1+P2 | Both direct imports removed |
| `engine/test_processor_coalesce_equivalence_properties.py` | 1 | 1 | P1+P2 | Both direct imports removed |
| **Total** | **108** | **58** | | **166 replacements** |

**Pattern 3 applicability note:** Only 1 of 166 violations qualified for `make_recorder_with_run()`. Property tests typically create the DB and recorder separately because they pass Hypothesis-generated configs to `begin_run()`, which the factory does not accept. All agents correctly identified this constraint and applied Pattern 1+2 only.

**Residual (intentionally not migrated):**

| Category | Remaining | Files | Reason |
|---|---|---|---|
| Docstring reference | 1 `LandscapeDB.in_memory()` | 1 file | Comment text, not constructor call |

**Import decisions:**

| Decision | Files | Reason |
|---|---|---|
| `LandscapeDB` import retained | 10 files | Used in type annotations for helper functions |
| `LandscapeDB` import removed | 3 files | No remaining type annotation references |
| `LandscapeRecorder` import retained | 1 file | Used in return type annotation for `_setup()` |
| `LandscapeRecorder` import removed | 5 files | No remaining references after Pattern 2 migration |

**Import sorting (auto-fixed):** 3 files had `I001` ruff violations from `tests.fixtures.landscape` imports placed after `tests.fixtures.stores` (alphabetical sort required `landscape` before `stores`). Auto-fixed with `ruff check tests/property/ --fix`.

**Code review results (3 agents):**

| Reviewer | Scope | Verdict | Issues Found |
|---|---|---|---|
| High-density reviewer | 3 files, 111 replacements | Clean | None |
| Medium-density reviewer | 5 files, 37 replacements | Clean | None |
| Small-density reviewer | 5 files, 18 replacements | Clean | None |

**Verification:**
- pytest: 1,160 passed, 13 deselected, 0 failures
- Test count: unchanged from P3 baseline (1,160 property tests)
- ruff: all checks passed (after auto-fix)
- Remaining `LandscapeDB.in_memory()` in property tests: 0 code, 1 docstring
- Remaining `LandscapeRecorder(` in property tests: 0

**Cumulative audit impact (P0 through P4):**

| Phase | Violations Resolved | Files Touched | Net Lines |
|---|---|---|---|
| P0 (ExecutionGraph) | 3 | 3 | ~−50 |
| P0.5a (new factories) | 6 scope items | 4 | +350 |
| P0.5b (factory delegation) | 2 refactors | 1 | ~−30 |
| P1 (PluginContext) | ~377 | 58 | −303 |
| P2 (landscape setup) | ~452 | 76 | −715 |
| P3 (inline plugins) | 19 class defs | 13 | −327 |
| P4 (property tests) | 166 | 13 | −4 |
| **Total** | **~1,023** | **~168 unique** | **~−1,079** |

**2026-03-02 — Post-P4 review: Gap remediation**

8-agent parallel verification audit identified 3 minor gaps. All resolved:

| # | Gap | Fix | Files |
|---|-----|-----|-------|
| 1 | P1 exceptions table missing 2 legitimate files (9 `PluginContext(...)` calls) | Added `test_record_call_guards.py` (6) and `test_resume_failure.py` (3) to exceptions table | `docs/audits/test-infrastructure-audit-2026-03-01.md` |
| 2 | F7 `_make_contract()` duplication — 2 files defined local helpers instead of importing `create_observed_contract()` | Deleted local helpers, replaced with `from tests.fixtures.base_classes import create_observed_contract` | `tests/integration/pipeline/test_aggregation_recovery.py`, `tests/integration/plugins/sinks/test_durability.py` |
| 3 | 6 ruff lint violations across 4 files on branch | Fixed: F401 unused imports (2), RUF043 raw string regex (2), SIM117 nested with (1), I001 import sort (1) | `test_record_call_guards.py`, `test_operations.py`, `test_aggregation.py`, `test_validation.py` |
