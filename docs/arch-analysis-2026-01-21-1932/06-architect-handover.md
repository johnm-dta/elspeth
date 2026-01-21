# Architect Handover Document

**Analysis Date:** 2026-01-21
**Prepared By:** Claude Code (Opus 4.5)
**Purpose:** Enable transition to architecture improvement planning

---

## Document Purpose

This handover document synthesizes the architecture analysis findings into actionable improvement opportunities. It is designed to provide a system architect with:

1. **Prioritized improvement backlog** with effort/impact assessment
2. **Refactoring roadmap** for identified technical debt
3. **Risk register** for areas requiring attention
4. **Decision log** of architectural choices discovered

---

## 1. Improvement Backlog

### 1.1 Critical Path Items (Before RC-1)

| ID | Item | Subsystem | Effort | Impact | Risk if Deferred |
|----|------|-----------|--------|--------|------------------|
| **C-1** | Complete TUI widget wiring | CLI/TUI | 2-4h | HIGH | `explain` command non-functional |
| **C-2** | Verify Alembic migrations work | Landscape | 1-2h | HIGH | Database portability unknown |
| **C-3** | Test checkpoint/recovery path | Production Ops | 2-4h | HIGH | Resume reliability unverified |

### 1.2 High Priority (Post RC-1, Sprint 1-2)

| ID | Item | Subsystem | Effort | Impact | Rationale |
|----|------|-----------|--------|--------|-----------|
| **H-1** | Split `recorder.py` by entity | Landscape | 4-8h | HIGH | 2,571 LOC, maintenance burden |
| **H-2** | Extract orchestrator components | Engine | 4-8h | MEDIUM | 1,622 LOC, complex state |
| **H-3** | Consolidate LLM pooling patterns | Plugins | 2-4h | MEDIUM | Azure/OpenRouter duplication |
| **H-4** | Repository layer decision | Landscape | 2h | LOW | Remove or use consistently |

### 1.3 Medium Priority (Post RC-1, Sprint 3-4)

| ID | Item | Subsystem | Effort | Impact | Rationale |
|----|------|-----------|--------|--------|-----------|
| **M-1** | Split `config.py` into modules | Core | 4-8h | MEDIUM | 1,186 LOC, many settings models |
| **M-2** | Add API reference generation | Documentation | 4-8h | MEDIUM | No Sphinx/auto-generated docs |
| **M-3** | Performance benchmark suite | Testing | 8-16h | LOW | No baseline metrics |
| **M-4** | Dependency vulnerability scanning | CI/CD | 1-2h | MEDIUM | Security baseline |

### 1.4 Nice to Have (Backlog)

| ID | Item | Subsystem | Effort | Impact | Rationale |
|----|------|-----------|--------|--------|-----------|
| **N-1** | Tutorial documentation | Docs | 4-8h | LOW | Onboarding friction |
| **N-2** | Troubleshooting guide | Docs | 2-4h | LOW | Support burden |
| **N-3** | Schema versioning in Landscape | Landscape | 4-8h | LOW | Migration tracking |
| **N-4** | Distributed execution design | Engine | 16-32h | LOW | Future scalability |

---

## 2. Refactoring Roadmap

### 2.1 Recorder Split (H-1)

**Current State:** `landscape/recorder.py` at 2,571 LOC handles all audit recording.

**Target Architecture:**

```
landscape/
├── recorder/
│   ├── __init__.py      # Public API (LandscapeRecorder class)
│   ├── runs.py          # Run lifecycle recording
│   ├── tokens.py        # Token state recording
│   ├── batches.py       # Batch/aggregation recording
│   ├── external.py      # External call recording
│   ├── artifacts.py     # Artifact recording
│   └── export.py        # Export functionality
├── models.py            # SQLAlchemy table definitions (unchanged)
├── schemas.py           # Pydantic schemas (unchanged)
└── repositories.py      # Repository layer (decide: use or remove)
```

**Migration Strategy:**

1. Create `recorder/` directory with `__init__.py` re-exporting public API
2. Extract entity-specific recording into separate modules
3. Keep `LandscapeRecorder` class as facade
4. Update imports throughout codebase
5. Remove old `recorder.py`

**Breaking Changes:** None if facade preserves interface.

### 2.2 Orchestrator Decomposition (H-2)

**Current State:** `engine/orchestrator.py` at 1,622 LOC handles:
- Pipeline execution
- Resume/checkpoint coordination
- Export orchestration
- Validation logic

**Target Architecture:**

```
engine/
├── orchestrator.py      # Core pipeline execution only (~800 LOC)
├── resume.py            # Checkpoint recovery logic
├── export.py            # Export orchestration
├── validation.py        # Pipeline validation
└── strategies/          # Execution strategies (future)
```

**Migration Strategy:**

1. Identify method clusters by responsibility
2. Extract `ResumeCoordinator` class
3. Extract `ExportOrchestrator` class
4. Update Orchestrator to compose these
5. Ensure no behavioral changes

### 2.3 LLM Pattern Consolidation (H-3)

**Current State:** Similar pooled execution patterns in:
- `plugins/llm/azure.py` (597 LOC)
- `plugins/llm/openrouter.py` (~400 LOC)
- `plugins/azure/content_safety.py` (~300 LOC)
- `plugins/azure/prompt_shield.py` (~300 LOC)

**Target Architecture:**

```python
# plugins/llm/base_pooled.py
class PooledLLMBase(TransformBase):
    """Base class for LLM transforms with pooled execution."""

    def __init__(self, config: PooledLLMConfig):
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._aimd = AIMDController(config.aimd_settings)

    @abstractmethod
    def _execute_single(self, row: dict, ctx: PluginContext) -> LLMResult:
        """Subclass implements actual API call."""
        ...

    def process_batch(self, rows: list[dict], ctx: PluginContext) -> list[TransformResult]:
        """Pooled execution with AIMD throttling."""
        futures = [self._pool.submit(self._execute_single, row, ctx) for row in rows]
        return [f.result() for f in as_completed(futures)]
```

**Migration Strategy:**

1. Create `PooledLLMBase` in `plugins/llm/base_pooled.py`
2. Refactor `azure.py` to extend base
3. Refactor `openrouter.py` to extend base
4. Apply similar pattern to Azure safety transforms

---

## 3. Risk Register

### 3.1 Technical Risks

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|------------|--------|------------|-------|
| **R-1** | TUI incomplete for RC-1 | HIGH | MEDIUM | Complete C-1 before release | TBD |
| **R-2** | Resume path untested at scale | MEDIUM | HIGH | Add integration tests (C-3) | TBD |
| **R-3** | pyrate-limiter cleanup race | LOW | LOW | Workaround documented and in place | N/A |
| **R-4** | Large file maintenance burden | MEDIUM | MEDIUM | Schedule H-1, H-2 post-RC-1 | TBD |

### 3.2 Operational Risks

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|------------|--------|------------|-------|
| **R-5** | No dependency scanning | MEDIUM | MEDIUM | Add `pip-audit` to CI (M-4) | TBD |
| **R-6** | No performance baseline | LOW | LOW | Add benchmark suite (M-3) | TBD |
| **R-7** | Single-process bottleneck | LOW | MEDIUM | Document limitation; N-4 for future | TBD |

### 3.3 Documentation Risks

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|----|------|------------|--------|------------|-------|
| **R-8** | No onboarding tutorial | MEDIUM | LOW | Create after RC-1 (N-1) | TBD |
| **R-9** | No API reference docs | MEDIUM | MEDIUM | Add Sphinx generation (M-2) | TBD |

---

## 4. Architectural Decisions Discovered

### 4.1 Decisions with Clear Rationale

| Decision | Rationale | Evidence |
|----------|-----------|----------|
| **pluggy for plugins** | Battle-tested (pytest uses it), clean hooks | CLAUDE.md, plugin architecture |
| **SQLAlchemy Core (not ORM)** | Multi-backend without ORM overhead | CLAUDE.md, Landscape implementation |
| **RFC 8785 for canonical JSON** | Standards-compliant deterministic hashing | CLAUDE.md, canonical.py |
| **Three-tier trust model** | Clear handling rules for data provenance | CLAUDE.md, implemented consistently |
| **No legacy code policy** | Prevents technical debt accumulation | CLAUDE.md, enforced in codebase |
| **Token-based DAG tracking** | Enables fork/join lineage | Engine design, token_id/row_id split |

### 4.2 Implicit Decisions (Not Documented)

| Decision | Current Choice | Alternatives | Recommendation |
|----------|----------------|--------------|----------------|
| **Repository layer** | Defined but unused | Use consistently OR remove | Decide and execute |
| **Config file split** | Single 1,186 LOC file | Split by concern | Split post-RC-1 |
| **Recorder structure** | Monolithic 2,571 LOC | Entity-based modules | Split post-RC-1 |
| **Export in orchestrator** | Embedded | Separate module | Extract post-RC-1 |

### 4.3 Decisions Requiring Validation

| Decision | Question | Impact | Recommended Action |
|----------|----------|--------|-------------------|
| **Alembic migrations** | Are they configured and working? | Database portability | Verify before RC-1 |
| **TUI architecture** | Is widget wiring complete? | CLI functionality | Complete before RC-1 |
| **Checkpoint atomicity** | Does two-phase commit work? | Data integrity | Test before RC-1 |

---

## 5. Quality Metrics Summary

### 5.1 Current Scores

| Dimension | Score | Target | Gap |
|-----------|-------|--------|-----|
| Architecture Design | 95/100 | 90+ | ✓ |
| Code Organization | 85/100 | 90+ | -5 (large files) |
| Type Safety | 95/100 | 90+ | ✓ |
| Testing | 90/100 | 90+ | ✓ |
| Documentation | 90/100 | 90+ | ✓ |
| Technical Debt | 90/100 | 90+ | ✓ |
| Security | 90/100 | 90+ | ✓ |
| Maintainability | 85/100 | 90+ | -5 (large files) |

### 5.2 Improvement Targets

After completing H-1 and H-2:
- Code Organization: 85 → 92 (+7)
- Maintainability: 85 → 92 (+7)
- **Weighted Average: 90 → 93**

---

## 6. Transition Checklist

### 6.1 Pre-RC-1 Checklist

- [ ] **C-1**: TUI widget wiring complete
- [ ] **C-2**: Alembic migrations verified
- [ ] **C-3**: Checkpoint/recovery tested
- [ ] All tests passing
- [ ] No critical lint warnings
- [ ] Documentation updated for any changes

### 6.2 Post-RC-1 Sprint 1 Checklist

- [ ] **H-1**: Recorder split complete
- [ ] **H-2**: Orchestrator decomposition complete
- [ ] **H-4**: Repository layer decision made
- [ ] Quality scores re-evaluated

### 6.3 Post-RC-1 Sprint 2 Checklist

- [ ] **H-3**: LLM pooling consolidated
- [ ] **M-4**: Dependency scanning added to CI
- [ ] **M-2**: API reference generation implemented

---

## 7. Knowledge Transfer Notes

### 7.1 Key Files to Understand

| File | Why | Complexity |
|------|-----|------------|
| `contracts/__init__.py` | Load-bearing import order prevents cycles | MEDIUM |
| `landscape/recorder.py` | Audit trail backbone | HIGH |
| `engine/orchestrator.py` | Pipeline execution core | HIGH |
| `core/canonical.py` | Deterministic hashing (audit integrity) | MEDIUM |
| `plugins/protocols.py` | All plugin interfaces | MEDIUM |

### 7.2 Non-Obvious Dependencies

| Component | Depends On | Why |
|-----------|------------|-----|
| CLI | All subsystems | Integration point |
| Engine | Landscape | Audit recording |
| Engine | Plugins | Plugin invocation |
| Plugins | Contracts | Shared types |
| Landscape | PayloadStore | Blob separation |

### 7.3 Testing Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Property tests | `tests/property/` | Hypothesis for invariants |
| Contract tests | `tests/contracts/` | Plugin protocol compliance |
| Integration tests | `tests/integration/` | Cross-subsystem |
| System tests | `tests/system/` | End-to-end flows |

### 7.4 Configuration Gotchas

1. **Precedence**: CLI > env > suite.yaml > profile > pack defaults > system defaults
2. **Expressions**: `${VAR}` for env, `${provider.field}` for cross-reference
3. **Validation**: Pydantic validates after merging all sources
4. **Secrets**: Never stored; HMAC fingerprints used for audit

---

## 8. Recommended Next Steps

### Immediate (This Week)

1. **Review this analysis** with stakeholders
2. **Validate C-1, C-2, C-3** items are on RC-1 checklist
3. **Assign owners** to risk register items

### Short-Term (Next Sprint)

1. **Plan H-1, H-2** refactoring with story points
2. **Add M-4** (dependency scanning) to CI/CD backlog
3. **Decide H-4** (repository layer: use or remove)

### Medium-Term (Next Quarter)

1. **Execute refactoring roadmap** (H-1 through H-3)
2. **Establish quality gates** based on scores
3. **Consider N-4** (distributed execution) if scale demands

---

## Appendix A: File Size Inventory

Files over 500 LOC requiring attention:

| File | LOC | Recommendation |
|------|-----|----------------|
| `recorder.py` | 2,571 | Split by entity (H-1) |
| `orchestrator.py` | 1,622 | Extract components (H-2) |
| `config.py` | 1,186 | Split settings models (M-1) |
| `azure.py` (LLM) | 597 | Consolidate patterns (H-3) |
| `dag.py` | 579 | Acceptable - cohesive |

## Appendix B: Test Coverage Gaps

Based on analysis, verify coverage for:

1. **Resume path** - Multiple failure scenarios
2. **Fork/join execution** - Complex DAG topologies
3. **Aggregation triggers** - All trigger conditions
4. **Export with purged payloads** - Grade degradation

## Appendix C: Related Analysis Artifacts

| Document | Purpose |
|----------|---------|
| `00-coordination.md` | Analysis orchestration plan |
| `01-discovery-findings.md` | Initial holistic assessment |
| `02-subsystem-catalog.md` | Detailed subsystem documentation |
| `03-diagrams.md` | Enhanced C4 architecture diagrams |
| `04-final-report.md` | Executive summary and analysis |
| `05-quality-assessment.md` | Code quality evaluation |
| `06-architect-handover.md` | This document |

---

**End of Architect Handover Document**
