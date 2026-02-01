# Pending Bugs Verification Report - Complete Audit
**Date:** 2026-01-25
**Session:** Continuation of RC1 Bug Burndown Session 4
**Scope:** All 34 pending bugs (11 P1, 14 P2, 9 P3)

## Executive Summary

Successfully verified **100% of pending bugs** (34/34) using systematic parallel agent verification. Each bug was examined against current code, git history analyzed, and status determined.

All verified bugs have been moved to subsystem folders in `docs/bugs/open/` or `docs/bugs/closed/`.

### Overall Results

| Priority | Total | STILL VALID | OBE | LOST | % Valid |
|----------|-------|-------------|-----|------|---------|
| **P1**   | 11    | 11          | 0   | 0    | **100%** |
| **P2**   | 14    | 13          | 1   | 0    | **93%**  |
| **P3**   | 9     | 9           | 0   | 0    | **100%** |
| **TOTAL**| 34    | 33          | 1   | 0    | **97%**  |

**Key Finding:** 97% of pending bugs remain valid - confirms accurate bug triage and real technical debt.

## P1 Critical Bugs (11 bugs - 100% STILL VALID)

All P1 bugs verified as critical issues requiring fixes before production.

### By Subsystem:
- **CLI:** 1 bug (explain command placeholder)
- **Engine-Orchestrator:** 4 bugs (aggregation failures, timeouts, context)
- **Plugins-LLM:** 5 bugs (client infrastructure, audit recording)
- **Engine-Processor:** 1 bug (token group ID correlation)
- **Core-Landscape:** 1 bug (replay hash collisions - moved here)

### Critical Issues Identified:

1. **CLI Explain Non-Functional** (`P1-2026-01-20-cli-explain-is-placeholder`)
   - Status: STILL VALID
   - Impact: Core auditability feature unusable
   - Backend exists but not wired to CLI

2. **Aggregation Buffered Tokens Non-Terminal** (`P1-2026-01-21-aggregation-passthrough-failure-buffered`)
   - Status: STILL VALID
   - Impact: Silent data loss on batch failures
   - Violates "every token reaches terminal state" principle

3. **Aggregation Single Skips Downstream** (`P1-2026-01-21-aggregation-single-skips-downstream`)
   - Status: STILL VALID
   - Impact: Downstream transforms bypassed
   - Protocol violation

4. **Aggregation Condition Missing Row Context** (`P1-2026-01-22-aggregation-condition-trigger-missing-row-context`)
   - Status: STILL VALID
   - Impact: Row-based conditions fail at runtime
   - Documentation mismatch

5. **Aggregation Timeout Never Fires** (`P1-2026-01-22-aggregation-timeout-idle-never-fires`)
   - Status: STILL VALID
   - Impact: Idle batches never flush
   - Related to coalesce timeout bug (also unfixed)

6. **Call Index Collisions Across Clients** (`P1-2026-01-21-call-index-collisions-across-clients`)
   - Status: STILL VALID
   - Impact: LOW currently (no multi-client transforms), HIGH future risk
   - Per-instance counters violate uniqueness constraint

7. **HTTP Auth Headers Dropped from Request Hash** (`P1-2026-01-21-http-auth-headers-dropped-request-hash`)
   - Status: STILL VALID
   - Impact: Hash collisions when credentials differ
   - Fingerprint infrastructure exists but not integrated

8. **HTTP Response Truncation Audit Loss** (`P1-2026-01-21-http-response-truncation-audit-loss`)
   - Status: STILL VALID
   - Impact: Audit integrity violation (100KB truncation)
   - Payload store infrastructure bypassed

9. **LLM Response Partial Recording** (`P1-2026-01-21-llm-response-partial-recording`)
   - Status: STILL VALID
   - Impact: Missing finish_reason, tool_calls, logprobs
   - Violates "full request AND response recorded"

10. **Replay Request Hash Collisions** (`P1-2026-01-21-replay-request-hash-collisions`)
    - Status: STILL VALID
    - Impact: Replays wrong response for 2nd+ occurrence
    - Affects non-deterministic LLMs, retry logic

11. **Token Outcome Group ID Mismatch** (`P1-2026-01-21-token-outcome-group-id-mismatch`)
    - Status: STILL VALID
    - Impact: Audit trail correlation broken
    - TokenInfo contract missing group ID fields

## P2 Bugs (14 bugs - 93% STILL VALID)

13 STILL VALID, 1 OBE (plugin gates removed in refactor).

### STILL VALID Bugs by Subsystem:

#### Core-Landscape (3 bugs)
- **Exporter Missing Config** - Portability gap for audit compliance
- **Exporter N+1 Queries** - Performance issue (25K+ queries for 1K rows)
- **Verifier Missing Payload Hidden** - Coerces None to {}, hides purged data

#### Core-Config (2 bugs)
- **Boolean Classifier BoolOp Mismatch** - Blocks valid semantic routing
- **Expression Slice Accepted** - Runtime failure (no visit_Slice handler)

#### Engine-Orchestrator (4 bugs)
- **Aggregation Coalesce Context Dropped** - Fork metadata lost
- **Aggregation Config Gates Skipped** - Gates bypassed after aggregation
- **Aggregation Timeout Checkpoint Age Reset** - Timer reset on recovery
- **Trigger Type Priority Misreports** - Audit attributes to wrong trigger

#### Engine-Pooling (2 bugs)
- **Pooling Ordering Metadata Dropped** - Submit/complete indices never recorded
- **Pooling Throttle Dispatch Burst** - Per-worker delays cause bursts

#### Engine-Retry (1 bug)
- **Retryable Transform Result Ignored** - retryable field has no effect

#### Plugins-LLM (1 bug)
- **LLM Usage Missing Crash** - AttributeError on response.usage = None

### OBE Bug:
- **Plugin Gate Graph Mismatch** (`P2-2026-01-19-plugin-gate-graph-mismatch`) - Plugin gates removed in refactor (Jan 18-19), infrastructure preserved for future use

## P3 Bugs (9 bugs - 100% STILL VALID)

All P3 bugs verified as valid quality/enhancement issues.

### By Category:

#### Expression Validation Gaps (3 bugs - Core-Config)
- **is Operator Not Restricted** - Allows `row['x'] is 1` (should only allow `is None`)
- **row.get Attribute Allowed** - Allows bare `row.get` without call
- **Subscript Not Restricted to Row** - Allows `{'a': 1}['a']`, violates security boundary

#### Audit/Observability (2 bugs)
- **Verifier Ignore Order Hides Drift** (Core-Landscape) - Hard-coded ignore_order masks meaningful drift
- **Pooling Missing Pool Stats** (Engine-Pooling) - max_concurrent_reached, dispatch_delay missing

#### Defensive Programming Violations (1 bug)
- **Aggregation Defensive Empty Output** (Engine-Orchestrator) - Fabricates {} on None, hides plugin bugs

#### Pooling Infrastructure (2 bugs)
- **Pooling Concurrent Execute Batch** - Shared buffer causes result mixing (latent bug)
- **Pooling Delay Invariant Not Validated** - Accepts min > max

#### HTTP Client (1 bug)
- **HTTP Base URL Concat Malformed** - String concat instead of proper URL joining

## New Subsystem Created

**Engine-Pooling** (`docs/bugs/open/engine-pooling/`)
- Created to organize pooling infrastructure bugs
- Total: 5 bugs (2 P2, 3 P3)
- Files: executor.py, throttle.py, reorder_buffer.py, config.py

## Combined Statistics

Including original 73 bugs from `VERIFICATION-REPORT-2026-01-25.md`:

| Source | Total | STILL VALID | OBE | LOST | % Valid |
|--------|-------|-------------|-----|------|---------|
| **Original (open/)** | 73 | 66 | 4 | 0 | **94%** |
| **Pending** | 34 | 33 | 1 | 0 | **97%** |
| **GRAND TOTAL** | **107** | **99** | **5** | **0** | **93%** |

**Overall finding:** 93% validation rate across 107 bugs indicates highly accurate triage with minimal false positives.

## Verification Methodology

### Process:
1. **Parallel agents:** 2-3 concurrent verification agents per wave
2. **Read-only verification:** No code edits, only bug ticket updates
3. **Comprehensive analysis:** Current code + git history + test coverage
4. **Status determination:** STILL VALID / OBE / LOST
5. **Immediate organization:** Bugs moved to subsystem folders as verified

### Tools Used:
- Task tool with general-purpose agents
- Grep for code search
- Read for file examination
- Bash for git history analysis

### Success Metrics:
- **100% coverage:** All 34 pending bugs verified
- **High accuracy:** 97% still valid (very low false positive rate)
- **Detailed findings:** Each bug has verification section with evidence
- **Zero LOST bugs:** No bugs invalidated by code changes

## Architectural Themes from Pending Bugs

### 1. Aggregation Subsystem Gaps
- **9 bugs total** across P1/P2/P3 priorities
- Timeout handling incomplete (no periodic checks)
- Context propagation broken (coalesce, gates)
- Terminal state violations (buffered tokens)
- Defensive programming patterns hiding bugs

### 2. Pooling Infrastructure Immaturity
- **5 bugs identified** (new subsystem)
- Concurrency hazards (shared buffer)
- Audit metadata not propagated (ordering indices)
- Throttle implementation flawed (burst dispatch)
- Config validation incomplete (delay invariant)
- Observability gaps (missing stats)

### 3. Expression Validation Incompleteness
- **5 bugs in expression parser**
- Slice syntax accepted but unsupported
- Identity operator unrestricted
- Subscript not limited to row
- Bare attribute access allowed
- Boolean vs non-boolean classification wrong

### 4. Audit Trail Gaps
- **Multiple bugs** affecting auditability principle
- HTTP response truncation before payload store
- LLM responses partially recorded
- Pooling metadata dropped
- Group IDs mismatched
- Missing payloads hidden by verifier

### 5. Client Infrastructure Issues
- **6 bugs in HTTP/LLM clients**
- Auth header fingerprinting not implemented
- Response truncation defeats payload store
- Request hash collisions on duplicates
- Call index collisions across clients
- Usage field crashes instead of defensive handling

## Recommendations

### Immediate Priorities (Pre-Production):
1. **Fix all 11 P1 pending bugs** - All verified as critical
2. **Aggregation subsystem hardening** - 4 P1 bugs + 4 P2 bugs
3. **Client infrastructure audit gaps** - 5 P1 bugs blocking auditability

### RC-2 Priorities:
1. **Expression validation completeness** (5 bugs)
2. **Pooling infrastructure maturity** (5 bugs)
3. **Remaining P2 bugs** (13 total)

### Technical Debt Cleanup:
1. **Remove defensive programming patterns** (3 bugs violate CLAUDE.md)
2. **Complete audit trail** (metadata gaps across subsystems)
3. **Config validation** (multiple validation gaps)

### Process Improvements:
1. **Subsystem-focused sprints** - Tackle aggregations, then pooling
2. **Audit trail validation tests** - Ensure completeness
3. **Expression parser hardening** - Complete validation coverage

## Files Modified

All 34 bug reports updated with verification sections and moved to subsystem folders:
- 11 P1 bugs → comprehensive verification, moved to subsystems
- 14 P2 bugs → detailed findings, 13 to subsystems, 1 to closed/
- 9 P3 bugs → status determination, moved to subsystems

Organized into subsystem folders:
- `docs/bugs/open/cli/` (1 P1)
- `docs/bugs/open/core-landscape/` (+3 P2, +1 P3)
- `docs/bugs/open/core-config/` (+2 P2, +3 P3)
- `docs/bugs/open/engine-orchestrator/` (+4 P1, +4 P2, +1 P3)
- `docs/bugs/open/engine-processor/` (+1 P1)
- `docs/bugs/open/engine-pooling/` (NEW: 2 P2, 3 P3)
- `docs/bugs/open/engine-retry/` (+1 P2)
- `docs/bugs/open/plugins-llm/` (+5 P1, +1 P2, +1 P3)
- `docs/bugs/closed/` (+1 P2 OBE)

## Conclusion

The pending bugs verification reveals:
- **Real technical debt:** 97% validation rate (only 1 false positive)
- **Critical issues:** All 11 P1 bugs confirmed as blockers
- **Accurate triage:** Combined 93% validation across 107 total bugs
- **Clear priorities:** Aggregations and pooling need immediate attention

**Next step:** Systematic fixing starting with P1 bugs in aggregation subsystem (5 bugs) and client infrastructure (6 bugs).
