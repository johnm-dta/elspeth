# ELSPETH Architect Handover Document

**Date:** 2026-01-27
**Prepared by:** Claude Opus 4.5 (Architecture Analysis)
**For:** Implementation team transitioning to improvement work
**Prerequisites:** Read `04-final-report.md` and `06-technical-debt-catalog.md`

---

## Purpose

This document provides a structured transition from architecture analysis to improvement implementation. It includes:
1. Prioritized improvement roadmap
2. Specific implementation guidance for critical fixes
3. Architectural decisions requiring team consensus
4. Success criteria for each phase

---

## Improvement Roadmap

### Phase 1: Critical Fixes (Week 1-2)

**Goal:** Unblock production deployment

| Item | Days | Owner | Acceptance Criteria |
|------|------|-------|---------------------|
| TD-001: Wire rate limiting | 3-5 | Backend | LLM calls respect configured limits; integration test proves it |
| TD-003: Coalesce timeout | 2-3 | Backend | Test with deliberate branch failure times out correctly |
| TD-002: Azure batch boundary validation | 1-2 | Backend | Missing keys produce specific error messages |
| TD-008: HTTP JSON parse error | 1 | Backend | JSON parse failures logged with body preview |

**Phase 1 Exit Criteria:**
- [ ] `RateLimitRegistry` instantiated in Orchestrator
- [ ] `check_timeouts()` called in processor main loop
- [ ] No `.get()` chains on external API responses
- [ ] All 4 items have integration tests

### Phase 2: Core Feature Completion (Week 3-4)

**Goal:** Deliver core value proposition

| Item | Days | Owner | Acceptance Criteria |
|------|------|-------|---------------------|
| TD-004: Implement `explain` | 5-10 | Full-stack | `elspeth explain --run X --row Y` returns lineage tree |
| TD-009: Wire TUI widgets | 3-5 | Frontend | ExplainApp displays real data in LineageTree/NodeDetail |
| TD-005: Implement checkpoints | 5-10 | Backend | Long pipeline can resume from checkpoint |

**Phase 2 Exit Criteria:**
- [ ] `explain` command works for completed runs
- [ ] TUI shows actual lineage data, not placeholders
- [ ] Checkpoints written periodically during processing
- [ ] `get_latest_checkpoint()` returns valid checkpoint

### Phase 3: Production Hardening (Week 5-6)

**Goal:** Make it reliable at scale

| Item | Days | Owner | Acceptance Criteria |
|------|------|-------|---------------------|
| TD-006: Fix exporter N+1 | 3-5 | Backend | Export 10k rows in < 60s |
| TD-007: Fix memory leak | 1 | Backend | Long pipeline doesn't grow `_completed_keys` |
| TD-017: Graceful shutdown | 3-5 | Backend | SIGTERM creates checkpoint, flushes pending |
| Add circuit breaker | 3-5 | Backend | Dead endpoint fails fast after N failures |

**Phase 3 Exit Criteria:**
- [ ] Export performance test passes
- [ ] Memory profiler shows stable usage over 100k rows
- [ ] SIGTERM test shows clean shutdown
- [ ] Circuit breaker test shows fast failure

### Phase 4: Architecture Cleanup (Ongoing)

**Goal:** Reduce maintenance burden

| Item | Days | Owner | Acceptance Criteria |
|------|------|-------|---------------------|
| TD-010: Add BaseCoalesce, audit drift | 3-5 | Backend | All plugin types have matching Protocol/Base |
| TD-011: Consolidate PayloadStore protocols | 1 | Backend | Single protocol in contracts/ |
| TD-013: Fix LSP violation | 3-5 | Backend | LLM transforms don't extend BaseTransform OR don't reject process() |
| TD-014: Extract CLI formatters | 1-2 | Backend | Single set of formatter functions called 3x |
| TD-026: Fix layer violations | 3-5 | Backend | No imports from lower to higher layers |
| TD-015: Fix test path integrity | 5-10 | Testing | No `graph._` in tests; all use `from_plugin_instances()` |

---

## Implementation Guidance

### TD-001: Wire Rate Limiting to Engine

**Current state:**
- `src/elspeth/core/rate_limit/registry.py` - Complete implementation
- `src/elspeth/core/rate_limit/limiter.py` - Working rate limiter
- `src/elspeth/engine/orchestrator.py` - No rate limiting imports

**Implementation approach:**

1. **Add rate_limit parameter to Orchestrator**
```python
# orchestrator.py:55+
def __init__(
    self,
    ...
    rate_limit_registry: RateLimitRegistry | None = None,
):
    self._rate_limit_registry = rate_limit_registry
```

2. **Pass registry to PluginContext**
```python
# context.py - add field
rate_limiter: RateLimitRegistry | None = None
```

3. **Use in LLM transforms**
```python
# azure.py - before making API call
if ctx.rate_limiter:
    ctx.rate_limiter.acquire("azure_openai", weight=1)
```

4. **Wire from CLI**
```python
# cli.py - in _execute_pipeline
if config.rate_limits:
    registry = RateLimitRegistry.from_config(config.rate_limits)
else:
    registry = None
orchestrator = Orchestrator(..., rate_limit_registry=registry)
```

**Test:**
```python
def test_rate_limiting_actually_limits():
    registry = RateLimitRegistry({"azure": {"requests_per_second": 1}})
    # Make 10 calls, verify they take ~10 seconds
```

---

### TD-003: Call Coalesce Timeout in Processor Loop

**Current state:**
- `coalesce_executor.py:371-440` - `check_timeouts()` exists
- `processor.py` - Never calls it

**Implementation approach:**

1. **Add timeout check interval config**
```python
# config.py
class ProcessorSettings(BaseModel):
    coalesce_timeout_check_interval_ms: int = 1000
```

2. **Check timeouts periodically in _process_loop**
```python
# processor.py - in main loop
last_timeout_check = time.monotonic()
while work_queue:
    # ... existing processing ...

    if time.monotonic() - last_timeout_check > timeout_check_interval:
        for coalesce_name in self._coalesce_configs:
            outcomes = self._coalesce_executor.check_timeouts(coalesce_name, current_step)
            for outcome in outcomes:
                self._handle_coalesce_outcome(outcome)
        last_timeout_check = time.monotonic()
```

**Test:**
```python
def test_coalesce_timeout_fires_during_processing():
    # Configure 3-branch fork with 1s timeout
    # Process 2 branches, delay 3rd
    # Verify timeout fires before end-of-source
```

---

### TD-002: Replace Defensive `.get()` with Boundary Validation

**Current state (azure_batch.py:768-774):**
```python
response = result.get("response", {})
body = response.get("body", {})
choices = body.get("choices", [])
```

**Correct pattern (per CLAUDE.md):**
```python
# Validate at boundary IMMEDIATELY
if "response" not in result:
    return TransformResult.error({
        "reason": "malformed_api_response",
        "detail": "missing 'response' key",
        "available_keys": list(result.keys())
    })

response = result["response"]
if not isinstance(response, dict) or "body" not in response:
    return TransformResult.error({
        "reason": "malformed_api_response",
        "detail": "invalid response structure",
        "response_type": type(response).__name__
    })

body = response["body"]
# Continue with validated data - no more .get() needed
```

**Apply to all LLM transforms:**
- `azure_batch.py`
- `azure.py`
- `azure_multi_query.py`

---

### TD-004: Implement `explain` Command

**Current state:**
- `cli.py:291-365` - Returns "not_implemented"
- `tui/screens/explain_screen.py` - 314 LOC of working screen
- `tui/widgets/lineage_tree.py` - 198 LOC of working widget
- `core/landscape/lineage.py` - Query logic exists

**Implementation approach:**

1. **JSON output mode (simplest, do first)**
```python
# cli.py - in explain command
lineage = explain_token_lineage(landscape, run_id, token_id)
if json_output:
    typer.echo(json.dumps(lineage.to_dict(), indent=2))
    return
```

2. **Wire TUI widgets (after JSON works)**
```python
# explain_app.py - replace placeholders
def compose(self) -> ComposeResult:
    yield Header()
    yield LineageTree(self.lineage_data, id=WidgetIDs.LINEAGE_TREE)
    yield NodeDetailPanel(id=WidgetIDs.DETAIL_PANEL)
    yield Footer()
```

3. **Add query methods if missing**
```python
# lineage.py - add if not present
def explain_token_lineage(
    landscape: LandscapeDB,
    run_id: str,
    token_id: str | None = None,
    row_id: str | None = None,
) -> LineageResult:
    # Query source row, tokens, node_states, calls, routing_events
    # Return structured LineageResult
```

**Test:**
```python
def test_explain_returns_complete_lineage():
    # Run simple pipeline
    # Call explain on a token
    # Verify source_row, tokens, node_states, calls present
```

---

## Architectural Decisions Required

The following items require team discussion before implementation:

### 1. Protocol/Base Class Duality (TD-010)

**Options:**
a) **Keep both, add generation** - Generate Base from Protocol (or vice versa)
b) **Keep both, add sync CI** - CI check that they match
c) **Merge to one** - Protocols only, no base classes
d) **Status quo** - Accept maintenance burden

**Recommendation:** Option (b) - Add CI check. Low effort, prevents drift.

### 2. LLM Transform Execution Model (TD-013)

**Options:**
a) **Separate hierarchy** - `BaseStreamingTransform` for accept()-based transforms
b) **Remove inheritance** - LLM transforms don't extend BaseTransform
c) **Make process() work** - Implement process() as accept() wrapper
d) **Document and accept** - LSP violation is intentional

**Recommendation:** Option (a) - Separate hierarchy clarifies the contract.

### 3. Layer Violation Resolution (TD-026)

**Current violations:**
- `contracts/results.py` imports `MaxRetriesExceeded` from `engine/retry.py`
- `core/config.py` imports `ExpressionParser` from `engine/expression_parser.py`

**Options:**
a) **Move to contracts** - `MaxRetriesExceeded` becomes a contract type
b) **Move to core** - `ExpressionParser` moves to core/
c) **Create shared layer** - New layer between contracts and engine

**Recommendation:** Option (a) for exceptions, Option (b) for parser.

### 4. Checkpoint Strategy (TD-005)

**Questions:**
- How often to checkpoint? (Every N rows? Every N seconds?)
- What to checkpoint? (Token positions? Coalesce state? Aggregation buffers?)
- Where to store? (Database? Filesystem? Both?)

**Recommendation:** Start with database storage, every 1000 rows or 60 seconds (whichever first).

---

## Success Criteria Summary

### Phase 1 Complete When:
- [ ] Production pipeline runs without Azure rate-limit errors
- [ ] Pipeline with failed branch times out correctly (not hangs)
- [ ] API response schema changes produce actionable error messages
- [ ] All changes have integration tests

### Phase 2 Complete When:
- [ ] `elspeth explain --run <id> --row <id>` returns meaningful output
- [ ] TUI shows lineage tree that can be navigated
- [ ] Long-running pipeline can be stopped and resumed

### Phase 3 Complete When:
- [ ] 10,000 row export completes in < 60 seconds
- [ ] 100,000 row pipeline maintains stable memory
- [ ] SIGTERM results in clean shutdown with checkpoint
- [ ] 1000 calls to dead endpoint completes in < 10 seconds (circuit breaker)

### Phase 4 Complete When:
- [ ] No layer violation warnings in CI
- [ ] No test files with `graph._` access
- [ ] Single PayloadStore protocol
- [ ] All plugin types have matching Protocol/Base

---

## Risk Mitigation

### Risk: Rate Limiting Changes Break Existing Pipelines

**Mitigation:**
- Rate limiting remains optional (None if not configured)
- Existing configs without rate_limits continue to work
- Add deprecation warning if LLM plugin used without rate limiting

### Risk: Coalesce Timeout Changes Break Fork/Join Logic

**Mitigation:**
- Add feature flag for timeout behavior
- Default to current behavior (timeout only at end-of-source)
- New behavior opt-in via config

### Risk: Explain Implementation Takes Too Long

**Mitigation:**
- JSON output first (simplest)
- TUI wiring second
- Incremental PRs, not one big change

---

## Handover Checklist

Before starting implementation:

- [ ] Read `04-final-report.md` for context
- [ ] Read `06-technical-debt-catalog.md` for item details
- [ ] Read `03-diagrams.md` to understand architecture
- [ ] Discuss architectural decisions with team
- [ ] Set up integration test environment
- [ ] Create tracking issues for Phase 1 items

---

## Contact

This analysis was performed by Claude Opus 4.5 using the axiom-system-archaeologist skill pack. The full analysis workspace is at:

```
docs/arch-analysis-2026-01-27-2132/
├── 00-coordination.md        # Analysis coordination log
├── 01-discovery-findings.md  # 47 issues from 17 agents
├── 03-diagrams.md            # C4 architecture diagrams
├── 04-final-report.md        # Synthesized findings
├── 05-quality-assessment.md  # Evidence-based critique
├── 06-technical-debt-catalog.md  # 28 prioritized items
├── 07-architect-handover.md  # This document
└── temp/                     # Individual agent analysis files
    ├── engine-analysis.md
    ├── landscape-analysis.md
    ├── plugins-analysis.md
    ├── core-analysis.md
    ├── contracts-cli-analysis.md
    ├── silent-failures-analysis.md
    └── test-gaps-analysis.md
```

For questions about specific findings, reference the relevant `temp/*.md` file.
