# Architect Handover Document: ELSPETH

**Handover Date:** 2026-01-20
**Analysis Type:** Architect-Ready (Option C)
**Status:** Ready for improvement planning

---

## Purpose

This document provides actionable recommendations for an architect taking ownership of ELSPETH. It synthesizes findings from the architecture analysis into prioritized improvement roadmaps.

---

## Current State Summary

| Aspect | Status | Confidence |
|--------|--------|------------|
| Core Architecture | ✅ Sound | High |
| Audit Infrastructure | ✅ Production-ready | High |
| Engine | ⏳ Near-complete (resume pending) | High |
| Plugin Framework | ✅ Functional | High |
| CLI/TUI | ✅ Functional | Medium |
| Documentation | ✅ Exceptional | High |
| Technical Debt | ✅ Minimal | High |

**Overall:** Ready for RC-1 with minor fixes.

---

## Improvement Roadmap

### Phase 1: RC-1 Preparation (1-2 weeks)

These items should be completed before release candidate:

#### 1.1 Complete Resume Functionality

**Location:** `src/elspeth/engine/orchestrator.py:897`

**Current State:**
- Checkpoint creation works
- Recovery detection works
- Batch recovery works
- **Row-level resume is TODO**

**Recommended Approach:**
```python
# Current (line 897)
# TODO: Continue processing unprocessed rows

# Implementation needed:
# 1. Query last completed row_index from Landscape
# 2. Configure source to skip already-processed rows
# 3. Restore aggregation state from checkpoints
# 4. Continue from last position
```

**Effort:** Medium (2-3 days)
**Impact:** High - critical for production reliability

#### 1.2 Fix SQLite Pragma Handling

**Bug Document:** `docs/bugs/open/2026-01-19-sqlite-pragmas-missing-from-url.md`

**Problem:** SQLAlchemy URL parsing doesn't pass pragmas to SQLite.

**Recommended Fix:**
```python
# In core/landscape/database.py
# Parse pragmas from URL and apply via connection events
from sqlalchemy import event

def _configure_sqlite_pragmas(dbapi_conn, record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

event.listen(engine, "connect", _configure_sqlite_pragmas)
```

**Effort:** Low (0.5 days)
**Impact:** Medium - affects production performance

#### 1.3 Verify Coalesce Configuration

**Bug Document:** `docs/bugs/open/2026-01-19-coalesce-config-ignored.md`

**Task:** Audit coalesce code paths to ensure config is respected.

**Effort:** Low (0.5 days)
**Impact:** Medium - correctness for fork/join pipelines

---

### Phase 2: Post-RC-1 Improvements (2-4 weeks)

#### 2.1 Dynamic Plugin Discovery

**Current:** Hardcoded TRANSFORM_PLUGINS dict in CLI

**Target:**
```python
# Plugin discovery via entry points
plugins = pluggy.PluginManager("elspeth")
plugins.load_setuptools_entrypoints("elspeth")
```

**Benefits:**
- Plugins installable via pip
- No CLI code changes for new plugins
- Third-party plugin ecosystem (future)

**Effort:** Medium (3-5 days)
**Impact:** Low - convenience, extensibility

#### 2.2 Improve Batch Transform Typing

**Current:**
```python
result = transform.process(rows, ctx)  # type: ignore[arg-type]
```

**Target:** Clean typing without ignores

**Options:**
1. **Protocol overload** - Multiple signatures
2. **Union type** - `dict | list[dict]`
3. **Separate method** - `process_batch()` for batch transforms

**Recommendation:** Option 3 (cleanest separation)

**Effort:** Medium (2-3 days)
**Impact:** Low - code clarity

#### 2.3 Observability Metrics

**Current:** OpenTelemetry spans only

**Target:** Add metrics for:
- Rows processed/second
- Transform latency percentiles
- Error rates by type
- Aggregation buffer sizes

**Implementation:**
```python
from opentelemetry import metrics

meter = metrics.get_meter("elspeth")
rows_processed = meter.create_counter("elspeth.rows.processed")
transform_latency = meter.create_histogram("elspeth.transform.latency")
```

**Effort:** Medium (3-5 days)
**Impact:** Medium - operational visibility

---

### Phase 3: LLM Integration (Phase 6 of project)

This is the next major feature milestone.

#### 3.1 LLMTransform Plugin

**Architecture Considerations:**
- Use LiteLLM for provider abstraction
- Record full request/response in audit trail
- Handle rate limiting via existing rate_limit subsystem
- Support record/replay for testing

**Audit Trail:**
```python
# External calls recorded with full payloads
call = recorder.record_call(
    state_id=state_id,
    call_type="llm",
    request_hash=hash(prompt),
    request_ref=payload_store.store(prompt),
    response_hash=hash(response),
    response_ref=payload_store.store(response),
)
```

#### 3.2 Reproducibility Considerations

- LLM calls are `Determinism.NONDETERMINISTIC`
- Run grade automatically becomes `REPLAY_REPRODUCIBLE`
- Record/replay mode needed for testing

---

## Architecture Principles to Preserve

### 1. Audit-First Design

**Rule:** Record before acting, crash on audit violations.

**Implementation Pattern:**
```python
# Always: begin_state → action → complete_state
state = recorder.begin_node_state(...)
result = perform_action(...)
recorder.complete_node_state(...)
```

### 2. Three-Tier Trust Model

**Rule:** Never coerce Tier 1/2 data. Only Sources coerce.

**Enforcement:**
- Transforms/Sinks receive type-valid data
- Type mismatches are upstream bugs, not defensive coding targets
- Wrap operations on row values, not type checks

### 3. No Legacy Code

**Rule:** When something changes, delete old code completely.

**Enforcement:**
- No backwards compatibility shims
- No deprecated code retention
- Update all call sites in same commit

### 4. System-Owned Plugins

**Rule:** All plugins are system code.

**Enforcement:**
- Plugin bugs crash immediately
- No user-provided plugins (sandboxing needed first)
- Test plugins with same rigor as engine

---

## Key Files for Modification

### Adding a New Transform

1. Create: `src/elspeth/plugins/transforms/my_transform.py`
2. Register: Add to TRANSFORM_PLUGINS in `cli.py` (until dynamic discovery)
3. Test: `tests/plugins/transforms/test_my_transform.py`

### Modifying the Engine

**Critical files:**
- `engine/orchestrator.py` - Run lifecycle
- `engine/processor.py` - Row processing
- `engine/executors.py` - Plugin execution

**Always:**
- Update audit recording calls
- Preserve token identity
- Test with DAG pipelines (forks/joins)

### Modifying the Audit Trail

**Critical files:**
- `core/landscape/recorder.py` - Recording API
- `core/landscape/schema.py` - Table definitions

**Always:**
- Add Alembic migration for schema changes
- Preserve backwards compatibility for existing runs
- Update exporter if new fields

---

## Risk Mitigation

### High-Risk Changes

| Change | Risk | Mitigation |
|--------|------|------------|
| Schema changes | Data loss | Alembic migrations, backup before |
| Token identity changes | Audit corruption | Comprehensive test coverage |
| Canonical JSON changes | Hash mismatches | Version field, detect changes |

### Medium-Risk Changes

| Change | Risk | Mitigation |
|--------|------|------------|
| New terminal states | Incomplete handling | Grep for RowOutcome usage |
| New plugin types | Interface gaps | Protocol tests |
| Config changes | Silent failures | Pydantic validation |

---

## Testing Requirements

### Before Any Release

1. **Full test suite passes**
   ```bash
   .venv/bin/python -m pytest tests/
   ```

2. **Type checking passes**
   ```bash
   .venv/bin/python -m mypy src/
   ```

3. **Linting passes**
   ```bash
   .venv/bin/python -m ruff check src/
   ```

4. **Integration tests pass**
   ```bash
   .venv/bin/python -m pytest tests/integration/
   ```

### For Specific Changes

| Change Type | Required Tests |
|-------------|----------------|
| Engine | `tests/engine/`, integration |
| Plugins | `tests/plugins/`, integration |
| Landscape | `tests/core/landscape/` |
| CLI | `tests/cli/`, manual validation |

---

## Contact Points

### Code Ownership

| Area | Primary Owner |
|------|---------------|
| Engine | TBD |
| Landscape | TBD |
| Plugins | TBD |
| CLI/TUI | TBD |

### Decision Records

- Architecture decisions: `docs/adr/` (if exists)
- Bug tracking: `docs/bugs/`
- Plans: `docs/plans/`

---

## Next Steps for Architect

1. **Review this document** with team leads
2. **Prioritize Phase 1 items** for RC-1
3. **Assign owners** to improvement tasks
4. **Create tracking issues** for roadmap items
5. **Schedule architecture review** post-RC-1

---

## Appendix: Quick Reference

### Common Commands

```bash
# Development setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests
.venv/bin/python -m pytest tests/

# Run pipeline
elspeth run --settings settings.yaml --execute

# Query lineage
elspeth explain --run <run_id> --token <token_id>

# Validate config
elspeth validate --settings settings.yaml
```

### Key Constants

```python
CANONICAL_VERSION = "sha256-rfc8785-v1"
MAX_WORK_QUEUE_ITERATIONS = 10_000
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| ELSPETH_SETTINGS_FILE | Default settings file |
| ELSPETH_LANDSCAPE_URL | Database URL |
| ELSPETH_PAYLOAD_DIR | Payload storage directory |
| ELSPETH_SIGNING_KEY | Export signing key |
| ELSPETH_FINGERPRINT_KEY | Secret fingerprinting key |
