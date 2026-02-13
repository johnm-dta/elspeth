# ELSPETH RC-3 Remediation Plan

**Date:** 2026-02-13
**Lineage:** Remaining items from RC-2 Comprehensive Remediation Plan (2026-01-27)
**Source:** Architecture Analysis by 17+ parallel agents (archive/2026-01-27-arch-analysis/)
**Original Issues:** 75+ | **Resolved Piecemeal:** ~65 | **Remaining:** 10

---

## Context

The RC-2 remediation plan identified 75+ issues across 6 phases. Over the course of the
RC2.x bug sprints, routing trilogy, test suite v2 migration, telemetry implementation,
and RC3-quality-sprint, approximately 65 items were resolved. This plan captures only
the items that remain unaddressed.

### What Was Completed (not repeated here)

**Phase 0 Quick Wins:** 11/12 done (QW-01 through QW-09, QW-11, QW-12)
**Phase 1 Critical Fixes:** 3/4 done (CRIT-01 rate limiting, CRIT-03 coalesce timeout, CRIT-04 HTTP JSON parse)
**Phase 2 Core Features:** 3/6 done (FEAT-01 explain command, FEAT-02 TUI widgets, FEAT-03 checkpoints)
**Phase 3 Production Hardening:** 12/12 done (all PERF items, all SAFE items including SAFE-05 call index seeding)
**Phase 4 Architecture:** 16/18 done (ARCH-02/03/04/05/07/08/09/10/11/12/13/15/16/17/18 + ARCH-01/08 N/A)
**Phase 5 Quality:** 10+ done (TEST-01/02/03/04, OBS-01/02/03, SEC-01, DOC-03/04/05)
**N/A by design:** ARCH-01 BaseCoalesce, ARCH-08 gate discovery (gate plugins removed entirely)

### Recently Completed (v2.0 → v2.1)

| Item | Commit | What |
|------|--------|------|
| TEST-01 | `1c31869b` | 13 public setters + 5 method renames on ExecutionGraph; 60+ private accesses eliminated |
| ARCH-02 | `dd7cf9bc` | LLM execution models documented (accept vs process, batch lifecycle) |
| ARCH-07 | `dd7cf9bc` | Plugin lifecycle hooks documented (on_start/on_complete/close ordering) |
| ARCH-13 | `b60cdcd9` | Unused session parameter removed from all Repository classes |
| SAFE-05 | `b60cdcd9` | Call index seeded from MAX(call_index) on resume; prevents UNIQUE violations |
| DOC-03 | `b60cdcd9` | Access control limitations documented in release guarantees |
| DOC-04 | `b60cdcd9` | Pre-2026-01-24 checkpoint incompatibility documented |
| DOC-05 | `b60cdcd9` | Audit export signing recommendation documented |

---

## Remaining Phase 1: LLM Boundary Validation

### CRIT-02: Replace `.get()` Chains on External API Responses
**Source:** TD-002 | **Priority:** MEDIUM (downgraded — partial progress made)

LLM plugins partially cleaned up, but `.get("usage") or {}` chains on external API
response structures persist in 4 files. These are at Tier 3 boundaries (external data),
so the pattern is technically correct (defensive on external data) but produces silent
empty-dict fallbacks instead of actionable error messages.

**Known remaining instances:**
- `src/elspeth/plugins/llm/openrouter.py:635` — `data.get("usage") or {}`
- `src/elspeth/plugins/llm/openrouter_batch.py:737` — `data.get("usage") or {}`
- `src/elspeth/plugins/llm/openrouter_multi_query.py:316` — `data.get("usage") or {}`
- `src/elspeth/plugins/llm/azure_batch.py:1207` — `body.get("usage", {})`

**Fix:** Replace with explicit key checks that return `TransformResult.error()` with
diagnostic info on unexpected response shapes, or at minimum log a structured warning
when usage metadata is absent (since usage is genuinely optional in some API responses).

---

## Remaining Phase 2: Feature Gaps

### FEAT-04: Add Missing CLI Commands
**Source:** TD-027 | **Priority:** MEDIUM

Current CLI has: `run`, `explain`, `validate`, `plugins list`, `purge`, `resume`, `health`.

**Missing:**
- `elspeth status` — show run status summary
- `elspeth export` — export audit trail (landscape exporter exists but no CLI surface)
- `elspeth db migrate` — run Alembic migrations (Alembic configured but no CLI command)

---

### FEAT-05: Add Graceful Shutdown — DONE
**Source:** TD-017 | **Priority:** HIGH

No signal handlers (SIGTERM, SIGINT) in orchestrator. Long-running pipelines cannot be
stopped cleanly.

**Implementation:** Cooperative shutdown via `threading.Event`. Signal handler sets event,
processing loop checks between rows. On shutdown: flush aggregation buffers, write pending
tokens to sinks, create checkpoint, mark run INTERRUPTED. Resumable via `elspeth resume`.
Second Ctrl-C force-kills. CLI exits with code 3. 16 tests (9 unit + 7 integration).

---

### FEAT-06: Add Circuit Breaker to Retry Logic
**Source:** Discovery Findings | **Priority:** MEDIUM

RetryManager uses tenacity exponential backoff but no circuit breaker. 10,000 rows against
a dead service = hours of retries before all rows fail.

**Fix:** After N consecutive failures to same endpoint, fail fast for M seconds.

---

## Remaining Phase 4: Architecture Cleanup

### ARCH-06: Consolidate AggregationExecutor Parallel Dictionaries
**Source:** Engine Analysis | **Priority:** MEDIUM

`engine/executors/aggregation.py` maintains 6 parallel dictionaries (`_buffers`,
`_buffer_tokens`, `_batch_ids`, `_member_counts`, `_trigger_evaluators`,
`_restored_states`) that must stay synchronized.

**Fix:** Consolidate into a single `AggregationNodeState` dataclass keyed by node_id.

---

### ARCH-14: Fix Resume Schema Verification Gap
**Source:** Core Analysis | **Priority:** MEDIUM

`core/checkpoint/recovery.py` requires `source_schema_class` but cannot verify it matches
the original run's schema. A schema change between runs could silently corrupt data.

**Fix:** Record a schema fingerprint (canonical hash of field names + types) in the
checkpoint metadata during the original run. On resume, compare the fingerprint against
the current `source_schema_class` and fail fast on mismatch.

---

## Remaining Phase 5: Quality & Documentation

### QW-10: Extract CLI Event Formatters
**Source:** TD-014 | **Priority:** LOW

Event handling logic in `cli.py` (~2,000 lines) is monolithic. Extract shared event
formatting to a helper module to reduce duplication.

---

### DOC-01: Add Missing Example READMEs — DONE
**Source:** Examples Analysis | **Priority:** LOW

All 12 missing READMEs created. Master `examples/README.md` index added. 5 new
gap-filling examples created (fork_coalesce, checkpoint_resume, database_sink,
rate_limited_llm, retention_purge). Committed `8318c2e4` + `863d882e`.

---

### DOC-02: Add Fork/Coalesce Examples — DONE (partial)
**Source:** Examples Analysis | **Priority:** LOW

Fork/coalesce example created (`examples/fork_coalesce/`). However, during implementation
a **feature gap was discovered**: the DAG builder wires fork branches directly to coalesce
nodes with no support for per-branch intermediate transforms (see ARCH-15 below).
Tracked as `elspeth-rapid-jyvr`.

---

### ARCH-15: Support Per-Branch Transforms Between Fork and Coalesce
**Source:** Discovery during DOC-02 | **Priority:** MEDIUM

The DAG builder (`core/dag/builder.py`) wires fork branches directly to coalesce nodes.
Two constraints prevent per-branch transforms:

1. Fork branch wiring only checks `branch_to_coalesce` and `sink_ids` — no transform routing
2. Coalesce branches must exactly match `fork_to` paths — can't use transform output connections

This means the fork/coalesce pattern is a merge barrier only. The canonical use case
(fork to sentiment API + entity API, merge results) requires per-branch transforms.

**Fix:** Allow fork branches to wire through transforms by treating branch names as
consumable connections. Token branch identity (already tracked) can be used for
coalesce correlation instead of requiring branch-name matching.

**Tracked:** `elspeth-rapid-jyvr`

---

### OBS-04: Add Metrics/Prometheus Integration
**Source:** Observability Analysis | **Priority:** LOW

Telemetry subsystem covers tracing and event streaming. No Prometheus-style pull metrics
for operational dashboards (request rates, error rates, processing latency histograms).

---

## Effort Summary

| Category | Items | Estimated Effort |
|----------|-------|------------------|
| Feature Gaps (FEAT-04/06) | 2 | 5-10 days |
| LLM Boundary Validation (CRIT-02) | 1 | 0.5-1 day |
| Architecture (ARCH-06/14) | 2 | 3-5 days |
| Quality/Docs (QW-10, DOC-01/02, OBS-04) | 4 | 3-6 days |
| **Total** | **9** | **~12-22 days** |

---

## Priority Ordering for RC-3

**Should-have:**
1. ARCH-14: Resume schema verification (silent corruption risk)
2. FEAT-04: CLI surface for export/status/migrate
3. ARCH-06: AggregationExecutor state consolidation
4. CRIT-02: LLM boundary validation audit

**Nice-to-have:**
5. FEAT-06: Circuit breaker (operational improvement, not correctness)
6. QW-10: CLI event formatter extraction (code quality)
7. DOC-01/02: Example documentation (user experience)
8. OBS-04: Prometheus metrics (operational visibility)

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-27 | 1.0 | Initial RC-2 plan from 17+ agent analysis (75+ items) |
| 2026-02-13 | 2.0 | Rebaselined for RC-3: removed ~57 completed items, 18 remain |
| 2026-02-13 | 2.1 | Updated: 8 more items completed (TEST-01, ARCH-02/07/13, SAFE-05, DOC-03/04/05), 10 remain |
