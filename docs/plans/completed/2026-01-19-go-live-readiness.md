# ELSPETH Go-Live Readiness Report

> **Date:** 2026-01-19
> **Status:** Post-Plugin-Refactor Assessment
> **Audit Method:** 7 parallel explore agents verified 245 requirements against codebase

## Executive Summary

ELSPETH is **85% complete** (208/245 requirements implemented). The core audit infrastructure is production-ready. This report triages the remaining 37 items into blockers, should-haves, and nice-to-haves.

**Bottom line:** Fix 4 blockers (2 work packages, ~3-5 days), then production-ready for CSV/JSON pipelines.

### Plugin Refactor Impact

The recent plugin refactor (January 2026) restructured the plugin system to properly separate:
- **Plugins** (Source/Transform/Sink) - touch row contents
- **System Operations** (Gate/Fork/Coalesce/Aggregation) - touch token wrappers

This refactor may have removed or broken previously-delivered Phase 5 features (governance, access control, redaction). The explore agents found no trace of this code. If these features are needed, check git history before rebuilding.

---

## Current State

| Category | Count | Percentage |
|----------|-------|------------|
| ✅ Implemented | 208 | 85% |
| ⚠️ Partial | 22 | 9% |
| ❌ Not Implemented | 15 | 6% |

### What's Working

- **Canonical JSON** (11/11) - Deterministic hashing with RFC 8785
- **Landscape Tables** (17/17) - Complete audit schema
- **Audit Recording** (9/9) - Full traceability
- **Plugin System** (13/13) - Source, Transform, Sink contracts
- **System Operations** (26/26) - Gate, Fork, Coalesce, Aggregation
- **DAG Execution** (5/5) - NetworkX-based validation
- **Token Identity** (6/6) - row_id/token_id/parent_token_id lineage
- **Retry Integration** (7/7) - tenacity-based with attempt tracking

---

## Triage

### 🚫 BLOCKERS - Must Fix Before Production

These items affect **audit trail completeness** for error cases. The happy path is fully auditable; these ensure the unhappy path is equally traceable.

| ID | Requirement | Issue | Impact | Effort |
|----|-------------|-------|--------|--------|
| **SDA-029** | QuarantineEvent recorded for discard | `ctx.record_validation_error()` exists but not persisted to DB | Source validation failures not queryable via `explain()` | Medium |
| **SDA-030** | QuarantineEvent: run_id, source_id, row_index | Fields in context signature; DB mapping incomplete | Same as above | Part of SDA-029 |
| **SDA-031** | QuarantineEvent: raw_row, failure_reason, field_errors | Fields in context signature; DB mapping incomplete | Same as above | Part of SDA-029 |
| **SDA-015** | TransformErrorEvent recorded on error | `ctx.record_transform_error()` exists but not persisted | Transform errors not queryable via `explain()` | Medium |
| **ENG-007** | Aggregation crash recovery | Checkpoints exist; recovery path not fully verified | Potential data loss on mid-batch crash | Medium |

#### Work Packages

**WP-BLOCKER-1: Error Event Persistence** (Est: 2-3 days)
- Wire `ctx.record_validation_error()` to landscape `quarantine_events` table
- Wire `ctx.record_transform_error()` to landscape `transform_error_events` table
- Add integration tests for error path auditability
- Verify `explain()` returns complete lineage for failed rows

**WP-BLOCKER-2: Crash Recovery Verification** (Est: 1-2 days)
- Test aggregation recovery from checkpoint after simulated crash
- Verify batch state transitions: draft → executing → completed/failed
- Document recovery procedure in runbook

---

### ⚠️ SHOULD HAVE - Fix Soon After Launch

These items represent operational or compliance risks that should be addressed within the first month of production.

| ID | Requirement | Issue | Risk if Deferred | Effort |
|----|-------------|-------|------------------|--------|
| **GOV-001** | Secrets NEVER stored - HMAC fingerprint only | Exporter uses HMAC; no `secret_fingerprint()` utility | API keys may appear in audit trail | Low |
| **GOV-002** | `secret_fingerprint()` function | Not implemented | Same as above | Low |
| **PLD-006** | Retention policies | Config exists; purge job not scheduled | Unbounded DB growth | Low |
| **PRD-005** | Concurrent processing | Pool size configurable; no thread executor | Single-threaded throughput limit | Medium |
| **CFG-017** | Env var interpolation `${VAR}` | Dynaconf `ELSPETH_*` works; `${VAR}` syntax TBD | Cannot externalize secrets in YAML | Low |

#### Work Package

**WP-SHOULDHAVE: Operational Hardening** (Est: 3-5 days)
- Implement `secret_fingerprint()` utility
- Add retention purge cron job / CLI command
- Add `${VAR}` interpolation support
- (Optional) Thread pool executor for concurrent processing

---

### 📋 NICE TO HAVE - Defer to Future Phases

#### Phase 5: Governance & Access Control

> ⚠️ **Note:** Phase 5 was previously marked as delivered, but the plugin refactor may have removed or broken some of this functionality. These items were not found by the explore agents, suggesting they were either never fully implemented or were removed during refactoring.

| ID | Requirement | Notes | Status |
|----|-------------|-------|--------|
| CFG-014 | `landscape.redaction.profile` config | Multi-tenant feature | Not found |
| GOV-003 | Fingerprint key from environment | Part of secrets story | Not found |
| GOV-004 | Configurable redaction profiles | Multi-tenant feature | Not found |
| GOV-005 | Access levels: Operator (redacted) | RBAC | Not found |
| GOV-006 | Access levels: Auditor (full) | RBAC | Not found |
| GOV-007 | Access levels: Admin (retention/purge) | Purge exists; no auth layer | Partial |
| GOV-008 | `explain --full` requires ELSPETH_AUDIT_ACCESS | Auth layer | Not found |
| PRD-004 | Redaction profiles | Multi-tenant feature | Not found |

**Rationale for deferral:** ELSPETH is system-operated by trusted teams, not multi-tenant. Access control adds complexity without value for current use case.

**Recovery note:** If Phase 5 governance features are needed, check git history for prior implementation before rebuilding from scratch.

#### Phase 6: External Calls (LLM, APIs)

| ID | Requirement | Notes |
|----|-------------|-------|
| SDA-016 | LLM query transform | No LLM workloads yet |
| EXT-002 | Record model/version | Part of LLM story |
| EXT-006 | Run modes: live, replay, verify | Advanced reproducibility |
| EXT-007 | Verify mode uses DeepDiff | Part of replay story |

**Rationale:** Current pipelines are deterministic (CSV→Transform→CSV). Build LLM infrastructure when first LLM workload arrives.

#### Future Plugin Packs

| ID | Requirement | Notes |
|----|-------------|-------|
| SDA-005 | Database source plugin | Add when needed |
| SDA-006 | HTTP API source plugin | Phase 6 |
| SDA-007 | Message queue source | Phase 6+ |
| SDA-026 | Webhook sink plugin | Phase 6 |
| PLD-004 | S3/blob storage backend | Phase 7 |
| PLD-005 | Inline backend | Not planned |
| PLD-008 | Optional compression | Not planned |

**Rationale:** Current workloads use CSV/JSON sources and sinks. Add plugins when workloads require them.

#### Convenience Features

| ID | Requirement | Notes |
|----|-------------|-------|
| CFG-016 | Profile system (`--profile` flag) | Dynaconf supports; not integrated |
| CFG-019 | Pack defaults (`packs/*/defaults.yaml`) | Not needed without plugin packs |
| CFG-020 | Suite configuration (`suite.yaml`) | Single settings file per run works |
| CLI-002 | `elspeth --profile <name>` | Part of profile story |
| CLI-004 | `elspeth explain --full` | Has `--json` and `--no-tui` |
| CLI-007 | `elspeth status` | Query landscape directly |
| CLI-008 | Human-readable + `--json` for all commands | `explain` has it; others TBD |

**Rationale:** Pipeline works without these. Operator convenience, not correctness.

#### Acceptable Divergences

| ID | Original Spec | Actual Implementation | Verdict |
|----|---------------|----------------------|---------|
| CFG-010/011 | `backend` + `path` | SQLAlchemy URL | ✅ Better |
| CFG-012/013 | Split retention by type | Unified `retention_days` | ✅ Simpler |
| FAI-011 | (run_id, row_id, seq, attempt) | (token_id, node_id, attempt) | ✅ Same semantics |
| SDA-022 | Idempotency key format | Schema ready, runtime wiring | ⚠️ Partial |
| EXT-001 | Provider identifier | CallType enum exists | ⚠️ Partial |

---

## Recommendation

### Immediate (Week 1)

1. **Fix blockers** - Error event persistence + crash recovery verification
2. **Validate** - Run integration tests for error path auditability

### Short-term (Month 1)

3. **Operational hardening** - Secret fingerprinting, retention purge, env var interpolation

### Medium-term (As Needed)

4. **Add plugins** - Database source, webhook sink when workloads require
5. **Phase 6** - LLM infrastructure when first LLM workload arrives

---

## Go-Live Checklist

```
PRE-LAUNCH (Must Complete)
─────────────────────────
[x] WP-BLOCKER-1: QuarantineEvent + TransformErrorEvent persisted ✅ DONE
[x] WP-BLOCKER-2: Aggregation crash recovery verified ✅ DONE (8 commits, 2026-01-19)
[x] Integration tests pass for error path auditability ✅ 8 tests passing
[x] explain() returns complete lineage for failed rows ✅ DONE
[x] Runbook documented ✅ docs/runbooks/aggregation-crash-recovery.md

POST-LAUNCH (First Month)
─────────────────────────
[ ] secret_fingerprint() utility implemented
[ ] Retention purge scheduled
[ ] ${VAR} interpolation working

DEFERRED (Future Phases)
────────────────────────
[ ] Access control / redaction (Phase 5)
[ ] LLM transforms / replay modes (Phase 6)
[ ] Additional plugin packs (as needed)
```

---

*Report generated from requirements audit by 7 parallel explore agents on 2026-01-19*
*Source: docs/design/requirements.md aligned with docs/contracts/plugin-protocol.md*
