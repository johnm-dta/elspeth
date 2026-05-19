# Post-Mortem: Aggregation Checkpoint Restoration IndexError (P1-2026-01-21)

> **ARCHIVED — Post-mortem captured at January–February 2026 (RC-2 hot-fix).**
> This document records the post-mortem for a single defect closed in RC-2. The system has progressed through RC-3, RC-4, and RC-5 since; the checkpoint subsystem has been substantially extended.
>
> **For the current state, see** [`../elspeth-progress-rc1-to-rc5.md`](../elspeth-progress-rc1-to-rc5.md).

**Incident ID:** P1-2026-01-21
**Severity:** P1 (audit-integrity-adjacent — crashes on resume from a recorded checkpoint)
**Audience:** Engineering team — post-mortem reader; future engineers maintaining the checkpoint subsystem
**Register:** Technical / incident-review

## Summary

Aggregation executors crashed with `IndexError` when flushing aggregation buffers after restoring from a checkpoint. The checkpoint format stored only token IDs, requiring database queries to reconstruct full `TokenInfo` objects during restoration; when prior-run token entries had been removed from the database, the buffer-to-token length mismatch caused the flush to index past the end of the token list.

## Timeline

| Time (approx) | Event |
|---|---|
| Pre-incident | Aggregation checkpoint format established as `{"rows": [...], "token_ids": [...], "batch_id": ...}` — minimal token info stored, expecting database reconstruction on resume |
| 2026-01-21 | First reproduction of the `IndexError` during resume testing |
| Day 0 | Incident filed as P1-2026-01-21; root cause identified the same day (token-info reconstruction depends on database state that may have changed between checkpoint write and resume) |
| Day 0 → Day 2 | Three-task fix sequence implemented (Task 1: store full TokenInfo; Task 2: reconstruct from checkpoint; Task 3: defensive guard) |
| Day 2 → Day 3 | Size-validation and edge-case tests added (Tasks 3.5–4) |
| Resolution | All 16 new tests passing including `test_checkpoint_roundtrip` |

## Root Cause

The aggregation checkpoint format stored token IDs but not the token data, on the assumption that the database would always be the authoritative source for token data at resume time. Three failures cascaded from that assumption:

1. **Performance.** Reconstructing N tokens required N database queries (N+1 query pattern), making resume linear in batch size where it could have been constant.
2. **Data loss on schema or row churn.** If the database no longer had token entries from the previous run (because of retention purge, schema migration, or unrelated cleanup), the reconstruction silently produced fewer tokens than the buffer expected.
3. **Type-of-crash, not crash-or-not.** The buffer/token length mismatch surfaced only at flush time, as an `IndexError` on a list access, rather than at restore time as a clear "data is missing" error. The crash site was distant from the root cause.

The deeper architectural fault was a **trust-boundary inversion**: the checkpoint subsystem trusted the database to be the source of truth for data the checkpoint itself was responsible for preserving. A checkpoint exists precisely to be independent of the database; the original design defeated that purpose.

## Detection

The crash was detected by integration tests during resume scenarios — specifically, by tests that exercised resume after data churn in the audit database. The audit-trail discipline ("every row reaches a recorded terminal state") meant that buffered rows in a checkpointed run had to flush before resume could complete; the flush is where the inversion bit.

The error message at the time was a bare `IndexError` from the executor's flush loop, with no context. This is itself a finding (see *Lessons Learned* below).

## Fix

### Format migration

| | Old format | New format |
|---|---|---|
| Storage | `{"rows": [...], "token_ids": [...], "batch_id": ...}` | `{"tokens": [{"token_id", "row_id", "branch_name", "row_data"}], "batch_id": ...}` |
| Restore cost | O(N) database queries | O(1) — read from checkpoint |
| Database dependency | Required | None |
| Behaviour on corruption | Silent length mismatch → IndexError at flush | Explicit error at restore with corruption message |

### Implementation

1. **Store complete `TokenInfo` in checkpoints** (10 MB hard limit; 1 MB warning).
2. **Reconstruct `TokenInfo` from the checkpoint** rather than from the database — eliminating the trust-boundary inversion.
3. **Defensive guard in `execute_flush()`** to detect corruption explicitly rather than as a downstream IndexError.

**Commits**

- `260b9a7`: Store full TokenInfo with size validation (Task 1)
- `3e25073`: Restore TokenInfo without DB queries (Task 2)
- `54edba7`, `59bb35f`: Add defensive guard for buffer/token mismatch (Task 3)
- `30de5e6`, `91c95cd`: Add size validation and edge case tests (Tasks 3.5–4)

### Testing

16 new tests covering format migration round-trip, size validation (1 MB warning, 10 MB error), edge cases, and defensive guard behaviour. Critical regression test: `test_checkpoint_roundtrip`.

## Impact

- Checkpoint restoration is O(1) rather than O(N).
- Checkpoints are now portable — no database dependency at restore.
- Corruption produces a clear error message rather than an opaque IndexError at an unrelated site.
- **Breaking change:** all checkpoints written before 2026-01-24 are invalid. (At the time, the project had no external users, so no migration tooling was provided.)

## Lessons Learned

1. **Checkpoints must be self-contained.** A checkpoint exists to be a self-sufficient record of state at a point in time. Depending on a separate authoritative store at restore time defeats the purpose of the checkpoint. *Generalisation:* any record that exists for crash recovery must be readable without depending on a system that may also have crashed or been mutated since.
2. **Length-mismatch IndexErrors are a symptom, not a root cause.** When a fixed-shape data structure fails on an indexing operation, the failure is the discovery point — not the bug. The bug is upstream where the shape became inconsistent. Wrapping the indexing site in a `try/except IndexError` would have buried this defect; the right response was to add a guard *plus* migrate the format so the inconsistency could not occur.
3. **Distance between root cause and crash site matters.** The defect was a contract violation at restore (fewer tokens reconstructed than expected); the crash was at flush, an unrelated subsystem. Make crashes loud at the trust boundary, not at the consumer of the corrupted data.
4. **Trust-boundary inversions are easy to miss in design review.** The original design "store token IDs, look up the rest" looks economical on paper and only fails under data churn. Reviewers should ask, for any cross-subsystem reference: *"What does this assume about the state of the referenced system at the time of reference resolution?"*

## Prevention

Concrete controls that landed alongside or shortly after this fix:

- **Construction-time validation** on checkpoint dataclasses (`__post_init__` invariants).
- **Defensive guard in `execute_flush()`** for buffer-to-token length symmetry.
- **Size limits with explicit warnings** so checkpoints that approach the limit are flagged before they become incident-shaped.
- **Format-versioning convention** for audit-adjacent serialisation, so future incompatibilities are diagnosable at restore.

Longer-arc controls that came later in the project and would have prevented or limited a similar defect:

- **Deep-immutability discipline (RC-3.4)** — `freeze_fields()` API and CI guards prevent ad-hoc mutation of frozen audit records.
- **Construction-time invariants** on every audit-adjacent dataclass (RC-3.4).
- **Guard-symmetry CI scanner (RC-5.0)** — every Landscape write site must have a corresponding read guard; a similar discipline now applies to checkpoint write/restore symmetry.

## Migration Guide

**Breaking change:** all checkpoints created before 2026-01-24 are invalid due to node ID format changes introduced in the routing refactor that landed at the same time. A pre-2026-01-24 checkpoint will fail restoration with a clear error message. Delete old checkpoint files and re-run affected pipelines.

**For developers:** see `src/elspeth/engine/executors.py` for the current checkpoint format and `tests/engine/test_executors.py` for usage examples.
