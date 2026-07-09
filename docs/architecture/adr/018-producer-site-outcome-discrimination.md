# ADR-018: Producer-Site Outcome Discrimination

**Date:** 2026-05-02
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** contracts, audit, row-outcomes, public-api

## Context

The `rows_routed` counter split in `elspeth-5069612f3c` exposed a broader
design rule. Before the split, `RowOutcome.ROUTED` represented both
intentional gate `route_to_sink` MOVE rows and transform `on_error` DIVERT
rows. The terminal-status predicate could not distinguish those producer
circumstances, so gate-only pipelines were misclassified as failed.

The producer already knows which circumstance occurred. Deferring that
knowledge to a later accumulator, graph lookup, or optional discriminator
field makes Tier-1 status accounting depend on convention rather than a
mechanical type signal.

## Decision

When a producer emits a terminal row outcome and the producer-known
circumstance changes audit obligations, status-predicate contribution, or
operator meaning, encode that circumstance as a distinct `RowOutcome` enum
variant at the producer site.

Do not encode producer-known terminal circumstances as optional discriminator
fields on a shared outcome variant unless a separate ADR justifies the
exception. Optional discriminator fields recreate the invalid state "outcome
known, producer circumstance unknown" and force every consumer to remember a
secondary field.

### Naming Rule

`RowOutcome` variants name the producer/audit circumstance. Aggregate row
counters that feed run-status predicates name the predicate role.

The current canonical mapping is recorded below so a future producer can
apply the rule mechanically rather than re-deriving it from prose. Sources
of truth: live accumulator at
`engine/orchestrator/outcomes.py::accumulate_row_outcomes`, resume
aggregation at `engine/orchestrator/core.py` (the `match` over
`outcome.outcome`), and the predicate definitions on
`contracts/run_result.py::RunResult` (`success_indicator` / `failure_indicator`).

| `RowOutcome` variant | Aggregate counter(s) incremented | Predicate role |
| --- | --- | --- |
| `COMPLETED` | `rows_succeeded` | success indicator |
| `ROUTED` | `rows_routed_success` (+ `routed_destinations[sink]`) | success indicator |
| `ROUTED_ON_ERROR` | `rows_routed_failure` (+ `routed_destinations[sink]`) | failure indicator |
| `DROPPED_BY_FILTER` | `rows_succeeded` | success indicator |
| `COALESCED` | `rows_succeeded`, `rows_coalesced` | success indicator (via `rows_succeeded`); `rows_coalesced` is structural |
| `FAILED` | `rows_failed` | failure indicator |
| `QUARANTINED` | `rows_quarantined` | failure indicator |
| `DIVERTED` | `rows_diverted` (incremented in `SinkExecutor` at the sink-write boundary; the row-outcome accumulator raises if it ever sees `DIVERTED`) | structural — sink-write fallback recorded for visibility, not a predicate input |
| `FORKED` | `rows_forked` | structural — parent-token bookkeeping; child tokens carry their own terminal outcomes |
| `EXPANDED` | `rows_expanded` | structural — parent-token bookkeeping; child tokens carry their own terminal outcomes |
| `CONSUMED_IN_BATCH` | (deferred — counted at batch flush via the batch-result token's terminal outcome) | structural — no producer-site counter; the flush-time outcome carries the predicate role |
| `BUFFERED` (non-terminal) | `rows_buffered` | structural — non-terminal hold; final outcome is reassigned at flush time |

`rows_coalesce_failed` is a failure-indicator counter that is **not** mapped
to a single `RowOutcome` variant. It is incremented at coalesce-flush time
when a barrier fails quorum (`outcomes.py::flush_coalesce_pending` and the
inline coalesce flush), and the consumed tokens additionally bump
`rows_failed`. Future producer-site outcomes that need similar quorum-fail
accounting must add their own structural counter and register it in the
`failure_indicator` predicate at the same time.

**Resume nuance.** The "Aggregate counter(s) incremented" column reflects
the in-flight live-accumulator path. The resume-time aggregation in
`engine/orchestrator/core.py::_derive_resume_terminal_status_from_audit`
deliberately restores **only the predicate-input counters** needed to feed
the biconditional in `contracts/run_result.py::RunResult`
(`rows_succeeded`, `rows_routed_success`, `rows_failed`, `rows_quarantined`,
`rows_routed_failure`, `rows_coalesce_failed`, `rows_processed`). Structural
counters (`rows_coalesced`, `rows_forked`, `rows_expanded`, `rows_buffered`,
`rows_diverted`) are not re-derived from `token_outcomes` on the
all-rows-already-processed branch and reflect only activity in the resumed
segment. Future producers that introduce a structural counter must decide
explicitly whether resume-time restoration is required and, if so, extend
the audit-replay aggregation alongside the producer site.

For this PR (`elspeth-5069612f3c`), the rule shows up as:

- `RowOutcome.ROUTED` means intentional gate MOVE and contributes to
  `rows_routed_success`.
- `RowOutcome.ROUTED_ON_ERROR` means transform `on_error` reroute and
  contributes to `rows_routed_failure`.

The enum and counter names are intentionally not lexically isomorphic. The
enum answers "what producer circumstance happened to this token?" The counter
answers "how does this aggregate bucket contribute to the run-status
predicate?" Future ADRs must not cite this decision as "make every
outcome/counter pair have the same word"; the pattern is producer-site
outcome discrimination plus predicate-role aggregate naming.

When adding a new `RowOutcome` variant: add a row to the table above, decide
its counter (existing predicate-role counter, new counter feeding an existing
predicate, or new structural counter), and update both the live accumulator
and the resume aggregation in the same change. The closed-set partition
assertion at `contracts/enums.py` (`_TERMINAL_ROW_OUTCOMES` /
`_NON_TERMINAL_ROW_OUTCOMES`) and the `case _:` exhaustiveness guards in both
accumulators will fail loudly at import or replay time if any of these steps
is skipped.

### Public API Naming

The web API exposes `rows_routed_success` and `rows_routed_failure` directly
on the relevant Pydantic response models. These field names are stable for the
current public API horizon. Do not add `rows_moved`, `rows_error_routed`, or a
transitional `rows_routed` alias in this PR. A future rename would be a
breaking API decision requiring its own ADR/API migration plan and OpenAPI
schema test updates.

## Consequences

### Positive Consequences

- Consumer code gets a mechanical prompt to handle new producer
  circumstances through enum branches.
- Audit records preserve producer intent directly in
  `token_outcomes.outcome`.
- L0 contracts, L3 Pydantic response models, and frontend types can compare
  the same predicate-role counter names without translation drift.

### Negative Consequences

- Adding a new producer circumstance requires updating every relevant
  `RowOutcome` branch, even when the transport path is otherwise shared.
- Aggregate counter names may not be lexical siblings of enum variant names.
  The naming rule must be read before adding future counters.
- Public API field names inherit engine predicate vocabulary by design.

## Alternatives Considered

### Alternative 1: Shared outcome variant plus discriminator field

Use `RowOutcome.ROUTED` for both MOVE and DIVERT and add a secondary
`routing_intent: Literal["move", "divert"] | None` field.

Rejected because it creates an optional field every consumer must remember to
read, does not force existing `RowOutcome.ROUTED` branches to change, and
allows the invalid state "routed but unknown intent" to be represented.

### Alternative 2: Accumulator graph lookup

Keep the producer emission unchanged and have the accumulator infer MOVE vs
DIVERT from the graph edge or `RoutingMode`.

Rejected because the producer already knows the answer, while a graph lookup
is a defensive inference path at the Tier-1 counter boundary.

### Alternative 3: Rename public API fields away from engine vocabulary

Expose names such as `rows_moved` and `rows_error_routed` in the web API while
using different L0/L2 names internally.

Rejected for this PR because the bug came from predicate mirror drift. Keeping
L0, L3, and frontend predicate-role fields identical is the mechanical guard.

## Related Decisions

- ADR-004: Explicit Sink Routing
- Historical implementation plan: `2026-05-02-rows-routed-counter-split.md`
  (retained in git history, not active docs)
- Filigree issue `elspeth-5069612f3c`
