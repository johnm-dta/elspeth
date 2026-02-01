# Bug Triage Report - 2026-02-01

## Summary

- Removed **17 open bug files** that were duplicates of entries already under `docs/bugs/closed/`.
- Moved **1 implemented bug** to `closed/` (verifier order handling).
- Normalized **4 misfiled bugs** into subsystem folders with standard filenames.
- Updated `docs/bugs/open/README.md` and added notes in `docs/bugs/README.md` to reflect the cleanup.
- Completed a P2 verification pass and closed 1 fixed P2 (coalesce failure recording).
- Closed `P3-2026-01-22-engine-artifacts-legacy-shim` after removing the legacy shim and migrating imports.
- Closed `P3-2026-01-31-payload-store-legacy-reexport` after removing the compatibility re-export and updating imports.
- Closed `P3-2026-01-31-rowresult-legacy-accessors` after removing the compatibility accessors and updating imports.

## P1 Verification Pass (2026-02-01)

- Verified all remaining P1s and tightened evidence sections with current line references.
- Closed two P1s as fixed:
  - `docs/bugs/closed/engine-orchestrator/P1-2026-01-30-payload-store-optional.md`
  - `docs/bugs/closed/engine-executors/P1-2026-01-31-sink-flush-failure-leaves-open-states.md`

## P2 Verification Pass (2026-02-01)

- Verified remaining P2s in engine/orchestrator, engine/executors, engine/tokens, engine/triggers, MCP, and Azure/LLM plugins.
- Closed one P2 as fixed:
  - `docs/bugs/closed/engine-coalesce/P2-2026-01-22-coalesce-timeout-failures-unrecorded.md`
- Re-verified older P2/P3 entries from the 2026-01-25 snapshot and updated verification dates.

## P3 Verification + Priority Review (2026-02-01)

- Re-verified all open P3s; closed three P3s as fixed.
- Added 2026-02-01 verification sections to the remaining P3s and validated that their priority remains P3.
- Closed `docs/bugs/closed/core-config/P3-2026-01-22-engine-artifacts-legacy-shim.md` (shim removed, imports updated).
- Closed `docs/bugs/closed/core-payload/P3-2026-01-31-payload-store-legacy-reexport.md` (compat re-export removed, imports updated).
- Closed `docs/bugs/closed/contracts/P3-2026-01-31-rowresult-legacy-accessors.md` (compat accessors removed, imports updated).

## Already Closed (Duplicate Open Entries Removed)

| Open path | Closed path |
| --- | --- |
| `docs/bugs/open/core-checkpoint/P1-2026-01-31-recovery-missing-payload-hash-verification.md` | `docs/bugs/closed/core-checkpoint/P1-2026-01-31-recovery-missing-payload-hash-verification.md` |
| `docs/bugs/open/core-checkpoint/P1-2026-01-31-running-status-blocks-resume.md` | `docs/bugs/closed/core-checkpoint/P1-2026-01-31-running-status-blocks-resume.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-json-formatter-nan-coercion.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-json-formatter-nan-coercion.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-compute-grade-no-determinism-validation.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-compute-grade-no-determinism-validation.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-fetchone-multi-row-silent-truncation.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-fetchone-multi-row-silent-truncation.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-nodestate-repo-missing-invariant-checks.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-nodestate-repo-missing-invariant-checks.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-routing-reason-payload-not-persisted.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-routing-reason-payload-not-persisted.md` |
| `docs/bugs/open/core-landscape/P1-2026-01-31-token-outcome-repo-is-terminal-coercion.md` | `docs/bugs/closed/core-landscape/P1-2026-01-31-token-outcome-repo-is-terminal-coercion.md` |
| `docs/bugs/open/core-security/P1-2026-01-31-sanitized-webhook-url-fragment-secrets.md` | `docs/bugs/closed/core-security/P1-2026-01-31-sanitized-webhook-url-fragment-secrets.md` |
| `docs/bugs/open/core-security/P1-2026-01-31-http-client-records-raw-urls-with-secrets.md` | `docs/bugs/closed/core-security/P1-2026-01-31-http-client-records-raw-urls-with-secrets.md` |
| `docs/bugs/open/engine-pooling/P1-2026-01-31-batching-mixin-unbound-local-error.md` | `docs/bugs/closed/engine-pooling/P1-2026-01-31-batching-mixin-unbound-local-error.md` |
| `docs/bugs/open/engine-pooling/P1-2026-01-31-row-reorder-buffer-deadlock.md` | `docs/bugs/closed/engine-pooling/P1-2026-01-31-row-reorder-buffer-deadlock.md` |
| `docs/bugs/open/plugins-llm/P1-2026-01-31-llm-response-payload-dropped-on-parse-failure.md` | `docs/bugs/closed/plugins-llm/P1-2026-01-31-llm-response-payload-dropped-on-parse-failure.md` |
| `docs/bugs/open/plugins-sinks/P1-2026-01-31-azure-blob-sink-no-audit-calls.md` | `docs/bugs/closed/plugins-sinks/P1-2026-01-31-azure-blob-sink-no-audit-calls.md` |
| `docs/bugs/open/plugins-sources/P1-2026-01-31-azure-csv-bad-lines-skipped-no-quarantine.md` | `docs/bugs/closed/plugins-sources/P1-2026-01-31-azure-csv-bad-lines-skipped-no-quarantine.md` |
| `docs/bugs/open/plugins-sources/P1-2026-01-31-azure-json-errors-crash-instead-quarantine.md` | `docs/bugs/closed/plugins-sources/P1-2026-01-31-azure-json-errors-crash-instead-quarantine.md` |
| `docs/bugs/open/plugins-sources/P1-2026-01-31-azure-json-accepts-nan-infinity.md` | `docs/bugs/closed/plugins-sources/P1-2026-01-31-azure-json-accepts-nan-infinity.md` |

### Priority Validation (Duplicates)

All removed duplicates are **P1** and still correctly prioritized:
- **core-landscape/core-checkpoint**: audit integrity + resume correctness (legal/audit risk).
- **core-security**: secret leakage in recorded URLs (security severity).
- **engine-pooling**: deadlock + runtime crashes (pipeline stability).
- **plugins-llm/sources/sinks**: missing audit payloads or quarantine handling (audit trail completeness).

## Implemented/Overshadowed Bugs Closed

- `docs/bugs/closed/core-landscape/P3-2026-01-21-verifier-ignore-order-hides-drift.md`
  - **Reason:** Phase 1 implementation is already in place; moved from open to closed.
  - **Priority validated:** P3 (quality/verification fidelity improvement).

## Additional Cleanup

- Removed a duplicate open entry:
  - `docs/bugs/open/P1/sink-flush-leaves-open-states.md` (duplicate of `docs/bugs/closed/engine-executors/P1-2026-01-31-sink-flush-failure-leaves-open-states.md`).
- Normalized misfiled bugs into subsystem folders with standard filenames:
  - `docs/bugs/closed/engine-orchestrator/P1-2026-01-30-payload-store-optional.md`
  - `docs/bugs/open/engine-coalesce/P1-2026-01-30-require-all-timeout-ignored.md`
  - `docs/bugs/open/core-dag/P2-2026-01-30-coalesce-schema-identity-check.md`
  - `docs/bugs/open/engine-orchestrator/P2-2026-01-30-orchestrator-cleanup-suppressed.md`

## Current Open Totals (Post-Triage)

- **P1:** 7
- **P2:** 43
- **P3:** 21
- **Total:** 71

See `docs/bugs/open/README.md` for the updated breakdown.
