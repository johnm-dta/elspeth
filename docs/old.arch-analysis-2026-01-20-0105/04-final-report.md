# Architecture Analysis Report: ELSPETH

**Analysis Date:** 2026-01-20
**Analyst:** Claude (Architecture Analysis Agent)
**Scope:** src/elspeth/ (~88 files, ~10,600 LOC)
**Confidence:** High

---

## Executive Summary

ELSPETH is a **well-architected, audit-first data pipeline framework** approaching production readiness (RC-1). The codebase demonstrates:

- **Clear subsystem boundaries** with minimal coupling
- **Consistent design patterns** (SDA model, three-tier trust, token identity)
- **Strong audit guarantees** through the Landscape subsystem
- **Mature infrastructure** (canonical JSON, DAG validation, retry management)

The architecture is fundamentally sound. The remaining work is primarily completion (resume, LLM integration) rather than architectural changes.

---

## Key Findings

### 1. Architecture Quality: Excellent

| Aspect | Rating | Notes |
|--------|--------|-------|
| Modularity | ★★★★★ | 7 subsystems with clear boundaries |
| Cohesion | ★★★★★ | Each subsystem has single responsibility |
| Coupling | ★★★★☆ | Clean dependencies, some forward refs |
| Consistency | ★★★★★ | Patterns applied uniformly |
| Testability | ★★★★☆ | Good separation, mock boundaries clear |
| Documentation | ★★★★☆ | CLAUDE.md is exceptional |

### 2. Core Strengths

#### 2.1 Audit-First Design
The entire framework is built around the axiom that **every decision must be traceable**. This isn't bolted on—it's the foundation.

```python
# Every operation records to Landscape BEFORE side effects
state = recorder.begin_node_state(token_id, node_id, step_index, input_data)
result = transform.process(row, ctx)  # Actual work
recorder.complete_node_state(state_id, status, output_data)
```

#### 2.2 Three-Tier Trust Model
Clear delineation of trust boundaries eliminates ambiguity in error handling:

| Tier | Trust Level | Error Response |
|------|-------------|----------------|
| Audit DB | Full | Crash immediately |
| Pipeline Data | Elevated | Expect types, wrap operations |
| External Data | Zero | Coerce, validate, quarantine |

#### 2.3 Token Identity System
Sophisticated row tracking through DAG execution:
- `row_id`: Stable source identity
- `token_id`: Instance in specific path
- Fork/join relationships preserved via `token_parents`

#### 2.4 Plugin Architecture
Clean pluggy-based extensibility:
- Protocols for type checking
- Base classes for convenience
- System-owned plugins (no user code risk)

### 3. Design Decisions That Stand Out

#### Canonical JSON (RFC 8785)
Two-phase canonicalization ensures deterministic hashing:
1. Normalize: pandas/numpy → JSON primitives
2. Serialize: RFC 8785/JCS standard

**NaN/Infinity are strictly rejected**, not silently converted. This prevents audit integrity issues.

#### Aggregation is Structural
The engine buffers rows, not plugins. This separates:
- **Buffer management** (engine responsibility)
- **Batch processing** (plugin responsibility)

```python
# Engine does buffering
self._aggregation_executor.buffer_row(node_id, current_token)
if self._aggregation_executor.should_flush(node_id):
    rows, tokens = self._aggregation_executor.flush_buffer(node_id)
    result = transform.process(rows, ctx)  # Plugin processes batch
```

#### Work Queue for DAG Traversal
RowProcessor uses a work queue instead of recursion to handle forks:
```python
work_queue: deque[_WorkItem] = deque([initial_item])
while work_queue:
    item = work_queue.popleft()
    result, children = self._process_single_token(item)
    work_queue.extend(children)  # Fork children added to queue
```
This prevents stack overflow on deep forks and allows iteration guards.

### 4. Identified Concerns

| Concern | Severity | Location | Recommendation |
|---------|----------|----------|----------------|
| Plugin registry hardcoded | Low | `cli.py` | Phase 4 - dynamic discovery planned |
| Resume incomplete | Medium | `orchestrator.py` | Complete row-level resume |
| SQLite pragmas ignored | Medium | `database.py` | Fix URL parameter parsing |
| Coalesce config ignored | Medium | Various | Verify coalesce paths |
| Some type: ignore | Low | `processor.py` | Improve batch-aware typing |

### 5. Technical Debt Assessment

**Minimal technical debt.** The codebase follows its own "No Legacy Code Policy" strictly:
- No backwards compatibility code
- No deprecated code retention
- Clean module boundaries

The git status shows documentation for known issues (bugs in `docs/bugs/open/`), indicating proper tracking.

---

## Architecture Patterns

### Pattern: Discriminated Union
Used for NodeState (Open | Completed | Failed):
```python
if status == NodeStateStatus.COMPLETED:
    return NodeStateCompleted(...)
elif status == NodeStateStatus.FAILED:
    return NodeStateFailed(...)
```

### Pattern: Factory Methods
TransformResult uses static factory methods:
```python
TransformResult.success(row)
TransformResult.error(details)
TransformResult.success_multi(rows)
```

### Pattern: Context Object
PluginContext bundles runtime dependencies:
```python
@dataclass
class PluginContext:
    run_id: str
    config: dict
    landscape: LandscapeRecorder
    payload_store: PayloadStore | None
    tracer: Tracer | None
```

### Pattern: Graceful Degradation
explain() handles purged payloads:
```python
def explain_row(self, run_id, row_id) -> RowLineage:
    # Returns lineage even if payload is purged
    # Hash is always preserved for integrity verification
```

---

## Subsystem Maturity

| Subsystem | Maturity | Status |
|-----------|----------|--------|
| Contracts | ★★★★★ | Production ready |
| Core | ★★★★★ | Production ready |
| Landscape | ★★★★★ | Production ready |
| Engine | ★★★★☆ | Near complete (resume pending) |
| Plugins | ★★★★☆ | Near complete (dynamic discovery pending) |
| CLI | ★★★★☆ | Near complete |
| TUI | ★★★☆☆ | Functional, less tested |

---

## Risk Assessment

### Low Risk
- **Code quality**: Consistently high
- **Testing coverage**: Appears comprehensive (test files visible)
- **Documentation**: Exceptional CLAUDE.md

### Medium Risk
- **Resume functionality**: Partial implementation could cause confusion
- **SQLite pragma handling**: May affect production deployments

### No High Risks Identified

---

## Recommendations

### Immediate (Before RC-1)

1. **Complete resume implementation** - Row-level resume is critical for production
2. **Fix SQLite pragma parsing** - Already documented as bug
3. **Verify coalesce paths** - Already documented as bug

### Near-Term (Post RC-1)

4. **Dynamic plugin discovery** - Replace hardcoded TRANSFORM_PLUGINS
5. **Improve batch-aware typing** - Remove type: ignore comments
6. **TUI testing** - Ensure explain functionality is robust

### Long-Term

7. **Observability metrics** - OpenTelemetry spans exist, add metrics
8. **LLM integration** - Phase 6 is next major feature

---

## Conclusion

ELSPETH is a **high-quality, well-designed framework** that achieves its stated goals:

1. ✅ **Auditable**: Every decision traceable to source
2. ✅ **Domain-agnostic**: SDA model applies to any pipeline
3. ✅ **Integrity**: Hash-based verification survives payload deletion
4. ✅ **Extensible**: Clean plugin architecture
5. ⏳ **Resilient**: Resume functionality partially complete

The architecture is **ready for production use** with minor fixes. The remaining work is feature completion, not architectural remediation.

**Overall Assessment:** Recommend proceeding to RC-1 after addressing the three documented bugs.

---

## Appendix: File Statistics

| Subsystem | Files | ~LOC |
|-----------|-------|------|
| CLI | 1 | 300 |
| Contracts | 11 | 1,200 |
| Core | 16 | 2,500 |
| Landscape | 9 | 2,800 |
| Engine | 10 | 2,200 |
| Plugins | 18 | 1,400 |
| TUI | 7 | 200 |
| **Total** | **~88** | **~10,600** |
