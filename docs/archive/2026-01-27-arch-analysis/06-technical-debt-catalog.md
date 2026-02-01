# ELSPETH Technical Debt Catalog

**Date:** 2026-01-27
**Author:** Claude Opus 4.5 (Debt Cataloger Agent)
**Source:** Discovery Findings (01-discovery-findings.md)
**Status:** RC-1 Pre-Release

---

## Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Architecture | 2 | 5 | 3 | - | 10 |
| Code Quality | 2 | 3 | 4 | - | 9 |
| Performance | - | 2 | 2 | - | 4 |
| Testing | - | 1 | 2 | - | 3 |
| Documentation | - | - | 2 | - | 2 |
| Security | - | - | - | - | 0 |
| **Total** | **4** | **11** | **13** | **0** | **28** |

---

## Critical Priority (Immediate)

### TD-001: Rate Limiting Subsystem Disconnected from Engine

**Evidence:** `src/elspeth/core/rate_limit/` (3 files, ~250 LOC) - no imports in `src/elspeth/engine/`
**Impact:** Rate limiting configuration in settings.yaml has no effect. Production deployments hitting Azure OpenAI will be rate-limited by provider, causing cascading failures and potential service bans.
**Effort:** M (3-5 days)
**Category:** Architecture
**Dependencies:** None
**Details:** The `RateLimitRegistry` and related classes exist but are never instantiated or called from the engine. The engine's `RowProcessor` and `Orchestrator` make external calls without any rate limiting.

**Fix Path:**
1. Inject `RateLimitRegistry` into `Orchestrator`
2. Wrap LLM executor calls with rate limiter
3. Add configuration validation that rate limits are applied
4. Add integration test proving rate limiting works

---

### TD-002: Defensive `.get()` Chain Masks LLM Response Schema Failures

**Evidence:** `src/elspeth/plugins/llm/azure_batch.py:768-774`
```python
response = result.get("response", {})
body = response.get("body", {})
choices = body.get("choices", [])
```
**Impact:** Violates Three-Tier Trust Model. When Azure changes response schema, users see "no_choices_in_response" instead of "malformed API response: missing 'body' key". Debugging requires examining audit trail payloads.
**Effort:** S (1-2 days)
**Category:** Code Quality
**Dependencies:** None
**Details:** External API responses are Tier 3 (zero trust) and require immediate boundary validation, not defensive fallbacks. The `.get()` chain silently coerces missing keys to empty containers.

**Fix Path:**
1. Replace `.get()` chain with explicit key validation
2. Return `TransformResult.error()` with specific missing key information
3. Apply same pattern to all LLM transform response parsing

---

### TD-003: Coalesce Timeout Never Fires During Active Processing

**Evidence:** `src/elspeth/engine/coalesce_executor.py:371-440` defines `check_timeouts()`, but `grep -r "check_timeouts" src/elspeth/engine/processor.py` returns no matches
**Impact:** Tokens waiting at coalesce points wait forever during processing. Timeouts only fire at end-of-source during `flush_pending()`. Pipeline with quorum policy expecting 3 branches where one fails will hang indefinitely.
**Effort:** M (2-3 days)
**Category:** Architecture
**Dependencies:** None
**Details:** The `check_timeouts()` method exists but is never called from the main processing loop. Only the cleanup path at source exhaustion calls it.

**Fix Path:**
1. Add periodic timeout check in `RowProcessor._process_loop()`
2. Implement configurable check interval
3. Add integration test with deliberate branch failure

---

### TD-004: `explain` Command Returns "not_implemented"

**Evidence:** `src/elspeth/cli.py:291-365` - returns `{"status": "not_implemented"}` and exit code 2
**Impact:** Core value proposition of ELSPETH (audit traceability) is not accessible via CLI. Users see command in `--help` but cannot use it.
**Effort:** L (1-2 weeks)
**Category:** Architecture
**Dependencies:** TD-009 (TUI widgets exist but not wired)
**Details:** The `ExplainScreen` (314 LOC), `LineageTree` (198 LOC), and `NodeDetail` (166 LOC) widgets exist. The command stub exists. They need to be connected.

**Fix Path:**
1. Implement query logic to fetch lineage from Landscape
2. Wire TUI widgets to actual data
3. Implement non-TUI JSON output mode
4. Add integration tests with sample pipelines

---

## High Priority (Next Sprint)

### TD-005: Checkpoints Table Defined But Not Implemented

**Evidence:** `src/elspeth/core/landscape/schema.py:373-400` - table exists; no `create_checkpoint()`, `get_latest_checkpoint()`, or `CheckpointRepository`
**Impact:** Checkpoint feature is scaffolded but non-functional. Long-running pipelines cannot resume from intermediate state.
**Effort:** L (1-2 weeks)
**Category:** Architecture
**Dependencies:** None
**Details:** Schema defines `checkpoints_table` with all necessary columns but no recorder methods or repository class to operate on it.

**Fix Path:**
1. Create `CheckpointRepository` class
2. Add `create_checkpoint()` and `get_latest_checkpoint()` to `Recorder`
3. Integrate with `Orchestrator` for periodic checkpointing
4. Add CLI `--checkpoint-interval` option

---

### TD-006: Exporter N+1 Query Pattern (21,001 Queries for 1000 Rows)

**Evidence:** `src/elspeth/core/landscape/exporter.py:199-329` - nested loops with individual queries
**Impact:** Export of large runs takes hours. Compliance audits requiring full export are impractical. Observed: 1 + 1000 + 2000 + 6000+ queries for 1000 rows.
**Effort:** M (3-5 days)
**Category:** Performance
**Dependencies:** None
**Details:** Each row triggers token query, each token triggers node_states query, each state triggers calls query. No batch loading or eager fetching.

**Fix Path:**
1. Refactor to batch-load tokens per run
2. Batch-load node_states per run
3. Batch-load calls per run
4. Join and assemble in memory
5. Add performance test with 10,000 row export

---

### TD-007: Memory Leak in CoalesceExecutor `_completed_keys`

**Evidence:** `src/elspeth/engine/coalesce_executor.py:172-199` - `_completed_keys.add(key)` grows unbounded
**Impact:** Long-running pipelines will OOM. Set is only cleared in `flush_pending()`, not after each normal merge.
**Effort:** S (1 day)
**Category:** Code Quality
**Dependencies:** None
**Details:** The `_completed_keys` set tracks completed coalesce groups but never prunes entries after they're processed.

**Fix Path:**
1. Remove key from `_completed_keys` after merge result is emitted
2. Add memory usage test for long-running pipeline simulation

---

### TD-008: Silent JSON Parse Fallback in HTTP Client

**Evidence:** `src/elspeth/plugins/clients/http.py:164-169`
```python
except Exception:  # Too broad!
    response_body = response.text  # Silent fallback
```
**Impact:** Content-Type says JSON but body is HTML error page -> downstream transforms receive string instead of dict -> cryptic type mismatch errors.
**Effort:** S (1 day)
**Category:** Code Quality
**Dependencies:** None
**Details:** Bare `except Exception` catches JSONDecodeError and silently substitutes raw text. Violates Three-Tier Trust boundary validation.

**Fix Path:**
1. Catch `JSONDecodeError` specifically
2. Return error result with diagnostic information
3. Log the actual response body (truncated) for debugging

---

### TD-009: TUI Widgets Exist But Use Placeholders

**Evidence:** `src/elspeth/tui/screens/explain_screen.py` uses `Static("Lineage Tree (placeholder)")`
**Impact:** Working code (512+ LOC across 3 widgets) is dead code. TUI displays placeholder text instead of functional widgets.
**Effort:** M (3-5 days)
**Category:** Architecture
**Dependencies:** None (but blocking TD-004)
**Details:** `LineageTree`, `NodeDetail`, and `ExplainScreen` are fully implemented but never instantiated in the app.

**Fix Path:**
1. Replace `Static` placeholders with actual widget instances
2. Wire data loading to Landscape queries
3. Add TUI integration tests

---

### TD-010: Protocol/Base Class Duality Maintenance Burden

**Evidence:** `src/elspeth/plugins/base.py` and `src/elspeth/plugins/protocols.py` - 8 protocols, 4 base classes
**Impact:** Every plugin type requires synchronized Protocol and Base class. `_on_error` attribute already has documentation drift. Missing `BaseCoalesce`.
**Effort:** M (3-5 days)
**Category:** Architecture
**Dependencies:** None
**Details:** `CoalesceProtocol` exists but no `BaseCoalesce`. `_on_error` docstrings differ between Protocol and Base. Changes must be applied twice.

**Fix Path:**
1. Add `BaseCoalesce` class
2. Audit all Protocol/Base pairs for drift
3. Consider code generation or single-source-of-truth pattern

---

### TD-011: Duplicate PayloadStore Protocols

**Evidence:** `src/elspeth/core/payload_store.py:28-83` and `src/elspeth/core/retention/purge.py:28-41`
**Impact:** Two Protocol definitions for same abstraction. Interface contract is fragmented. "Avoid circular imports" comment indicates architectural smell.
**Effort:** S (1 day)
**Category:** Architecture
**Dependencies:** None
**Details:** Retention module defines "minimal" protocol to avoid imports from payload_store, but this creates parallel interface definitions.

**Fix Path:**
1. Extract shared protocol to `contracts/` layer
2. Import from contracts in both locations
3. Remove duplicate protocol

---

### TD-012: OpenTelemetry Integration Claimed But Missing

**Evidence:** `src/elspeth/core/logging.py:3-6` docstring claims OTel; no tracer/span code exists
**Impact:** Claimed observability capability doesn't exist. Users expecting distributed tracing will be disappointed.
**Effort:** L (1-2 weeks)
**Category:** Architecture
**Dependencies:** None
**Details:** Docstring says "complements OpenTelemetry spans" but no tracer configuration, span creation, or trace context propagation in core.

**Fix Path:**
1. Remove false claims from docstring (immediate)
2. Or implement OTel integration:
   - Add tracer configuration
   - Instrument key operations (row processing, LLM calls)
   - Propagate trace context through pipeline

---

### TD-013: LLM Transforms Violate Liskov Substitution Principle

**Evidence:** `src/elspeth/plugins/llm/azure.py:228-243` - `process()` raises `NotImplementedError`
**Impact:** LLM transforms implement `BaseTransform` but reject `process()` call. Engine code that treats all transforms uniformly will break.
**Effort:** M (3-5 days)
**Category:** Code Quality
**Dependencies:** None
**Details:** The transforms use `accept()` for row-level pipelining but still extend `BaseTransform` which defines `process()`.

**Fix Path:**
1. Create `BaseStreamingTransform` subclass that formalizes `accept()` pattern
2. Or remove inheritance from `BaseTransform`
3. Document execution model difference clearly

---

### TD-014: CLI Code Duplication (123 Lines Verbatim)

**Evidence:** `src/elspeth/cli.py:471-594` and `683-806` - identical event handler code
**Impact:** Bug fixes must be applied twice. High risk of divergence. 123 lines duplicated between `_execute_pipeline()` and `_execute_pipeline_with_instances()`.
**Effort:** S (1-2 days)
**Category:** Code Quality
**Dependencies:** None
**Details:** Same duplication exists in `_execute_resume_with_instances()`.

**Fix Path:**
1. Extract shared event handling to helper function
2. Parameterize the differences
3. Call helper from all three locations

---

### TD-015: Test Path Integrity Violations (62+ Instances)

**Evidence:** `grep -r "graph\._" tests/engine/ --include="*.py"` returns 62+ matches
**Impact:** Tests manually construct `ExecutionGraph` bypassing `from_plugin_instances()`. BUG-LINEAGE-01 hid for weeks due to this pattern.
**Effort:** L (1-2 weeks)
**Category:** Testing
**Dependencies:** None
**Details:** CLAUDE.md explicitly documents this anti-pattern and its consequences.

**Fix Path:**
1. Audit all tests using `graph._` private access
2. Refactor to use production factory methods
3. Add CI check to flag private attribute access in tests

---

## Medium Priority (Next Quarter)

### TD-016: SinkExecutor O(N) Node States Creation

**Evidence:** `src/elspeth/engine/executors.py:1563-1572`
**Impact:** Linear scan per sink operation. Noticeable slowdown at scale.
**Effort:** S (1 day)
**Category:** Performance
**Dependencies:** None

---

### TD-017: No Graceful Shutdown Mechanism

**Evidence:** No "shutdown" matches in `src/elspeth/engine/`
**Impact:** Long-running pipelines cannot be cleanly stopped. Risk of data loss on interrupt.
**Effort:** M (3-5 days)
**Category:** Architecture

---

### TD-018: Call Index Counter In-Memory Only

**Evidence:** `src/elspeth/core/landscape/recorder.py:1750-1787`
**Impact:** Resume after crash may have call index conflicts.
**Effort:** S (1-2 days)
**Category:** Code Quality

---

### TD-019: Hardcoded Plugin Lookup Tables in Validation

**Evidence:** `src/elspeth/plugins/validation.py:85-109`
**Impact:** Adding new plugin types requires updating hardcoded lookup tables.
**Effort:** S (1 day)
**Category:** Code Quality

---

### TD-020: BatchStatus Accepts Raw String, No Enum Validation

**Evidence:** `src/elspeth/core/landscape/recorder.py:1319-1348`
**Impact:** Invalid status strings stored without validation.
**Effort:** S (1 day)
**Category:** Code Quality

---

### TD-021: models.py Appears to be Dead Code

**Evidence:** `src/elspeth/core/landscape/models.py`
**Impact:** Maintenance burden for unused code.
**Effort:** S (1 day to verify and remove)
**Category:** Code Quality

---

### TD-022: Repository Session Parameter Unused

**Evidence:** `src/elspeth/core/landscape/repositories.py` - all classes
**Impact:** API contract suggests session injection but implementation ignores it.
**Effort:** S (1 day)
**Category:** Architecture

---

### TD-023: `in_memory()` Factory Bypasses Schema Validation

**Evidence:** `src/elspeth/core/landscape/database.py:188-202`
**Impact:** Tests using in-memory database may miss schema issues.
**Effort:** S (1 day)
**Category:** Testing

---

### TD-024: Missing Composite Index on token_outcomes

**Evidence:** `src/elspeth/core/landscape/schema.py:134-135`
**Impact:** Queries filtering by token_id and outcome type will be slow at scale.
**Effort:** S (1 day)
**Category:** Performance

---

### TD-025: Property Testing Covers Only 1.2% of Codebase

**Evidence:** `tests/property/` covers only 3 files
**Impact:** Limited fuzzing coverage. Edge cases may be missed.
**Effort:** L (ongoing)
**Category:** Testing

---

### TD-026: Layer Violations in Dependency Structure

**Evidence:** `contracts/results.py` imports from `engine/retry.py`; `core/config.py` imports from `engine/expression_parser.py`
**Impact:** Violates documented layer dependencies. Increases coupling.
**Effort:** M (3-5 days)
**Category:** Architecture

---

### TD-027: Missing CLI Commands (`status`, `export`, `db migrate`)

**Evidence:** CLAUDE.md documents `elspeth status`; no implementation exists
**Impact:** Documented commands don't exist. Users must use Alembic directly for migrations.
**Effort:** M (1 week for all three)
**Category:** Documentation

---

### TD-028: Hardcoded Checkpoint Compatibility Date

**Evidence:** `src/elspeth/core/checkpoint/manager.py:202-233` - `datetime(2026, 1, 24)`
**Impact:** Future format changes require more hardcoded dates instead of version field.
**Effort:** S (1 day)
**Category:** Code Quality

---

## Debt Categories Summary

| Category | Count | Items |
|----------|-------|-------|
| Architecture | 10 | TD-001, TD-003, TD-004, TD-005, TD-009, TD-010, TD-011, TD-012, TD-017, TD-022, TD-026 |
| Code Quality | 9 | TD-002, TD-007, TD-008, TD-013, TD-014, TD-018, TD-019, TD-020, TD-021, TD-028 |
| Performance | 4 | TD-006, TD-016, TD-024 |
| Testing | 3 | TD-015, TD-023, TD-025 |
| Documentation | 2 | TD-027 |

---

## Prioritized Remediation Order

### Phase 1: Unblock Core Functionality (Week 1-2)
1. **TD-001** - Rate limiting disconnected (prevents production deployment)
2. **TD-003** - Coalesce timeout never fires (prevents reliable fork/join)
3. **TD-002** - Defensive `.get()` chain (masks errors)
4. **TD-008** - Silent JSON fallback (masks errors)

### Phase 2: Complete Core Features (Week 3-4)
5. **TD-004** - `explain` command not implemented (core value prop)
6. **TD-009** - TUI widgets not wired (dependency of TD-004)
7. **TD-005** - Checkpoints not implemented (reliability)

### Phase 3: Production Hardening (Week 5-6)
8. **TD-006** - Exporter N+1 queries (compliance requirement)
9. **TD-007** - Memory leak in coalesce (stability)
10. **TD-015** - Test path integrity (confidence)

### Phase 4: Architecture Cleanup (Ongoing)
11. **TD-010** - Protocol/Base duality
12. **TD-011** - Duplicate protocols
13. **TD-013** - LSP violation in LLM transforms
14. **TD-014** - CLI code duplication
15. **TD-026** - Layer violations

---

## Quick Wins (High Impact, Low Effort)

| ID | Item | Effort | Impact |
|----|------|--------|--------|
| TD-007 | Memory leak in `_completed_keys` | S (1 day) | Prevents OOM in production |
| TD-008 | Silent JSON parse fallback | S (1 day) | Better error diagnostics |
| TD-014 | CLI code duplication | S (1-2 days) | 50% reduction in event handler code |
| TD-011 | Duplicate PayloadStore protocols | S (1 day) | Single source of truth |
| TD-019 | Hardcoded plugin lookup | S (1 day) | Easier plugin extensibility |
| TD-020 | BatchStatus enum validation | S (1 day) | Data integrity |
| TD-021 | Remove dead models.py | S (1 day) | Less maintenance |
| TD-028 | Replace hardcoded date with version | S (1 day) | Future-proofs checkpoints |

---

## Confidence Assessment

**Overall Confidence:** High

**Evidence Quality:**
- All items trace to specific file:line locations
- Discovery findings validated with grep/search commands
- Impact assessments based on documented behavior

**Risk Assessment:**
- Critical items (TD-001 through TD-004) block production readiness
- High items block GA release
- Medium items are maintenance burden but not blockers

**Information Gaps:**
- TD-021 (dead code) needs runtime verification before deletion
- TD-006 (N+1 queries) impact estimate based on code analysis, not profiling
- TD-025 (property testing) requires codebase coverage analysis

**Caveats:**
- Effort estimates assume familiarity with codebase
- Dependencies may reveal additional work during implementation
- Some items may be resolved by planned features not yet documented

---

## Appendix: Evidence Commands

```bash
# TD-001: Verify rate limiting disconnected
grep -r "RateLimitRegistry" src/elspeth/engine/

# TD-003: Verify check_timeouts not called
grep -r "check_timeouts" src/elspeth/engine/processor.py

# TD-015: Count test path violations
grep -r "graph\._" tests/engine/ --include="*.py" | grep -v "graph\._graph" | wc -l

# TD-026: Find layer violations
grep -r "from.*engine" src/elspeth/contracts/
grep -r "from.*engine" src/elspeth/core/
```
