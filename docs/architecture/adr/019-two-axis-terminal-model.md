# ADR-019: Two-Axis Terminal Model — Lifecycle, Outcome, and Path

**Date:** 2026-05-04
**Status:** Proposed
**Deciders:** ELSPETH maintainers
**Tags:** contracts, audit, row-outcomes, public-api, supersedes-adr-018

## Context

ADR-018 (2026-05-02, Accepted) codified producer-site outcome discrimination: every
producer-known terminal circumstance gets its own `RowOutcome` enum variant.
Filigree ticket `elspeth-d376b8e681` (2026-05-03) surfaced a question ADR-018
defers: when a gate-routed row durably reaches a sink, ADR-018 records its
audit terminal as `ROUTED`, never `COMPLETED`. The ticket's user-stated
invariant — "successfully sunk tokens must be COMPLETED, not sunk — otherwise
what's the point in COMPLETED?" — exposes that `RowOutcome` is doing two jobs:

1. **Lifecycle answer.** Did the row succeed, fail, or end as parent-token
   bookkeeping?
2. **Provenance answer.** How did the row reach its terminal — default flow,
   gate MOVE-routed, transform `on_error` DIVERT, source-side quarantine, sink-
   write fallback?

ADR-018 collapses these two axes into one enum. The collapse forces three
recurring smells:

- **Bifurcated success predicate.** `contracts/run_result.py:208`:
  `success_indicator = rows_succeeded > 0 OR rows_routed_success > 0`. The OR is
  there because routed-and-sunk rows are not in `rows_succeeded` even though
  they durably succeeded. Path A (RC5-UX, shipped) made `RowOutcome.ROUTED`
  bump both counters to fix the operator-facing "0 succeeded | →N routed"
  display. The bifurcation in the predicate is the residue.
- **`COALESCED` already does what `ROUTED` doesn't.** `outcomes.py:310-314`
  routes coalesced rows with `RowOutcome.COMPLETED` as their audit terminal,
  recording the coalesce fact via the structural counter `rows_coalesced`. ADR-
  018 line 53 justifies this by classing `COALESCED` as "structural" rather
  than producer-circumstance. But the same shape (lifecycle answer + structural
  fact) applies to `ROUTED` — and is not applied — because the line between
  "structural" and "producer-circumstance" was drawn around what was needed for
  the rows-routed split (`elspeth-5069612f3c`), not as a complete model.
- **Three-read attributability for routed-and-sunk.** Under ADR-018, an auditor
  asking "did row 42 succeed?" must read the audit terminal (`ROUTED`), the
  gate event (was it MOVE or DIVERT?), and the sink event (was the write
  durable?) to confirm. CLAUDE.md's Auditability Standard and the
  attributability test (`explain(recorder, run_id, token_id)` must prove
  complete lineage in one call) assume the audit terminal carries the
  lifecycle answer in one read.

The team has already factored lifecycle and provenance as two axes on the
**counter** side (`ExecutionCounters.rows_routed_success` is provenance;
`rows_succeeded` is lifecycle). ADR-019 propagates the same factoring into
`RowOutcome` itself, removing the bifurcation at its source.

## Decision

Replace the single `RowOutcome` enum with three orthogonal fields on the
`token_outcomes` audit row:

1. **`completed: bool`** — has the row reached a terminal state? Equivalent
   to "row is no longer in flight." Today's `BUFFERED` is `completed=False`;
   every other current `RowOutcome` variant maps to `completed=True`.
2. **`outcome: TerminalOutcome | None`** — the lifecycle answer when
   `completed=True`. Three values:
   - `SUCCESS` — row contributed positively to the run.
   - `FAILURE` — row's processing did not succeed; the run's failure
     indicator should reflect this row.
   - `STRUCTURAL` — parent-token or sink-fallback bookkeeping; child tokens
     or related rows carry the actual lifecycle answer. Not a predicate input.
   When `completed=False`, `outcome` is `NULL` (the row hasn't decided yet).
3. **`path: TerminalPath`** — the provenance answer. How did the row reach
   its terminal? Always populated (BUFFERED rows have `path=BUFFERED`).

The `completed` field is materially redundant with `outcome IS NOT NULL` but
preserved for query ergonomics and to honor the operator vocabulary the model
is being designed around. Tier 1 cross-checks at the recorder boundary MUST
verify the bool–outcome consistency (`completed XOR (outcome IS NULL)`) at
both write and read time, mirroring the existing `is_terminal` cross-check at
`core/landscape/model_loaders.py:539`.

### Naming Rule

`TerminalOutcome` answers "what was the lifecycle answer?" — three values,
predicate-role aligned: `SUCCESS` and `FAILURE` are predicate inputs;
`STRUCTURAL` is not.

`TerminalPath` answers "how did the row get there?" — provenance enum,
producer-known and producer-emitted, never inferred from graph topology or
counter context. New paths require an enum addition at the producer site.

This continues ADR-018's mechanical-prompt discipline: a producer that emits a
new circumstance must add a `TerminalPath` value AND classify it in the
mapping table below.

### Mapping table (current `RowOutcome` → new model)

| Current `RowOutcome` | `completed` | `outcome` | `path` | Predicate counter(s) |
| --- | --- | --- | --- | --- |
| `COMPLETED` | `True` | `SUCCESS` | `DEFAULT_FLOW` | `rows_succeeded` |
| `ROUTED` | `True` | `SUCCESS` | `GATE_ROUTED` | `rows_succeeded`, `rows_routed_success` |
| `ROUTED_ON_ERROR` | `True` | `FAILURE` | `ON_ERROR_ROUTED` | `rows_failed`, `rows_routed_failure` |
| `DROPPED_BY_FILTER` | `True` | `SUCCESS` | `FILTER_DROPPED` | `rows_succeeded` |
| `COALESCED` | `True` | `SUCCESS` | `COALESCED` | `rows_succeeded`, `rows_coalesced` (structural) |
| `FAILED` | `True` | `FAILURE` | `UNROUTED` | `rows_failed` |
| `QUARANTINED` | `True` | `FAILURE` | `QUARANTINED_AT_SOURCE` | `rows_quarantined`, `rows_failed` |
| `DIVERTED` | `True` | `STRUCTURAL` | `SINK_FALLBACK` | `rows_diverted` (structural) |
| `FORKED` | `True` | `STRUCTURAL` | `FORK_PARENT` | `rows_forked` (structural) |
| `EXPANDED` | `True` | `STRUCTURAL` | `EXPAND_PARENT` | `rows_expanded` (structural) |
| `CONSUMED_IN_BATCH` | `True` | `STRUCTURAL` | `BATCH_CONSUMED` | (deferred — counted at flush) |
| `BUFFERED` | `False` | `NULL` | `BUFFERED` | `rows_buffered` (structural, non-terminal) |

The bifurcated success predicate at `contracts/run_result.py:208` becomes:

```text
success_indicator  = COUNT(outcome = SUCCESS) > 0
failure_indicator  = COUNT(outcome = FAILURE) > 0
```

Equivalently: `rows_succeeded > 0` and `rows_failed > 0`. The `OR rows_routed_success > 0`
clause goes away because gate-routed-and-sunk rows already increment
`rows_succeeded` (they're `SUCCESS`/`GATE_ROUTED`).

### Why `STRUCTURAL` exists as a third outcome value

The user's natural framing — "completed is a bool, outcome is success or
failure" — admits two outcome values. Four current `RowOutcome` variants
(`FORKED`, `EXPANDED`, `CONSUMED_IN_BATCH`, `DIVERTED`) do not fit cleanly
into `SUCCESS` or `FAILURE`:

- **Parent-token bookkeeping** (`FORKED`, `EXPANDED`): the parent row's
  lifecycle is "spawned children" — not a success and not a failure. Children
  carry the actual lifecycle. Forcing `outcome=SUCCESS` on the parent would
  double-count successes; forcing `outcome=FAILURE` would mis-attribute.
- **Batch consumption** (`CONSUMED_IN_BATCH`): the row was absorbed into an
  aggregate. The batch-result token carries the lifecycle answer at flush
  time. The consumed row's outcome is bookkeeping.
- **Sink fallback** (`DIVERTED`): ADR-018 line 56 already classes this as
  "structural — sink-write fallback recorded for visibility, not a predicate
  input." Preserving that classification.

`STRUCTURAL` makes these visible in `token_outcomes` with a path explaining the
circumstance, but explicitly excludes them from the success and failure
predicates. The closed-set assertion at module-import time enforces the
partition: every new `TerminalPath` value must be classified as predicate
input (with a counter) or structural (without one).

### Public API field-name preservation

Wire schemas (`ProgressEvent`, `RunSummary`, `CompletedData`) continue to
expose the existing predicate-role counter names: `rows_succeeded`,
`rows_routed_success`, `rows_routed_failure`, `rows_failed`, `rows_quarantined`,
`rows_coalesce_failed`, plus structural counters. Per ADR-018 line 109-114,
counter renames are a separate breaking-API decision. ADR-019 redefines the
underlying engine model; a follow-on ADR-020 may revisit counter names if the
team wants to align them with the new vocabulary. Frontend counter widgets and
operator dashboards do not require changes for ADR-019.

The `RowOutcome` enum at `contracts/enums.py:160` is removed. Code paths that
read `outcome.outcome == RowOutcome.X` flip to read the new field pair
(`outcome.outcome == TerminalOutcome.SUCCESS AND outcome.path == TerminalPath.X`).
The closed-set partition at `contracts/enums.py:238` is replaced by the
mapping table above, enforced by import-time exhaustiveness assertions over
`(TerminalOutcome × TerminalPath)`.

### Resume nuance

Resume-time aggregation in `engine/orchestrator/core.py::_derive_resume_terminal_status_from_audit`
must read the new field pair from `token_outcomes` and re-derive the
predicate-input counters. Structural counters (`rows_coalesced`, `rows_forked`,
`rows_expanded`, `rows_buffered`, `rows_diverted`) are not re-derived on the
all-rows-already-processed branch (preserving ADR-018 line 70-82 nuance).

### Migration policy

ELSPETH's "delete the DB on schema change" policy
(`MEMORY.md::project_db_migration_policy`) applies. There is no in-place
migration: operators discard old `audit.db` and `sessions.db` files when ADR-019
ships. The recorder, wire schemas, frontend counter readers, and integration
tests flip in a single coordinated commit. Tests asserting `outcome ==
RowOutcome.X` (~72 hits across `src/` and `tests/` per a panel-1 grep) update
to assert the new field pair in the same commit.

## Consequences

### Positive

- **Single-read attributability** for lifecycle questions. An auditor asking
  "did row 42 succeed?" reads `outcome=SUCCESS` from one row. No join, no gate-
  event traversal, no inference from sink-event existence.
- **Predicate simplification.** `success_indicator = rows_succeeded > 0` and
  `failure_indicator = rows_failed > 0`. The bifurcated OR at
  `run_result.py:208` and Path A's counter-symmetry workaround are no longer
  load-bearing for correctness — they become consequences of the new model
  rather than patches on top of the old one.
- **Provenance preservation.** Every fact ADR-018 protected (intentional MOVE
  vs error DIVERT, source quarantine vs unrouted failure, default flow vs
  gate-driven dispatch) survives in `TerminalPath`. The discriminant is more
  visible because it has its own column rather than being embedded in the
  same column as the lifecycle answer.
- **The COALESCED asymmetry resolves.** `COALESCED` is no longer special-cased
  in `outcomes.py`; it's just `(SUCCESS, COALESCED)` and the structural
  `rows_coalesced` counter increments. The team's existing two-axis intuition
  (line 51-53 of ADR-018) becomes the model.

### Negative

- **Two-column reads.** Consumers that want both lifecycle and provenance
  (e.g., MCP analyzer queries grouping by outcome distribution) read two
  columns instead of one. SQL filters that previously matched a single
  `RowOutcome` value now match a (outcome, path) pair.
- **Migration surface is large.** Recorder, repository load/store, resume
  aggregation, ~20+ integration tests, the closed-set partition assertion,
  and the explain() lineage report all flip in the same commit. The "delete
  the DB" policy contains the data side; the code side is real work.
- **Three outcome values, not two.** The user's natural mental model
  ("success or failure") admits a `STRUCTURAL` middle ground for parent-token
  bookkeeping. This is a justified deviation (see "Why `STRUCTURAL` exists"
  above) but it is a deviation.
- **Supersedes a 48-hour-old ADR.** ADR-018 was Accepted on 2026-05-02 and
  has a complete predicate-role table that downstream tickets cite. Marking
  it Superseded so quickly is governance churn; the discipline is to treat
  ADR-018's table as the input that ADR-019 generalizes, not as a mistake.

### Neutral

- The non-terminal partition retains exactly one member (`BUFFERED`); the
  `_TERMINAL_ROW_OUTCOMES` / `_NON_TERMINAL_ROW_OUTCOMES` partition is replaced
  by the `completed` bool plus the mapping-table exhaustiveness assertion.
- Public API counter names are unchanged. Frontend widgets reading
  `rows_routed_success` / `rows_routed_failure` continue to function.
- The `RowOutcome.is_terminal` property and the audit-DB `is_terminal` column
  are subsumed by the `completed` field.

## Alternatives Considered

### Alternative 1: Keep ADR-018, ship Path A only (counter symmetry)

**Description:** Accept the bifurcated success predicate and the three-read
attributability for routed-and-sunk rows. Close `elspeth-d376b8e681` with no
model change.

**Rejected because:** the residue is real and recurring — the same ticket has
been filed at least once, the predicate at `run_result.py:208` already shows
the smell, and future producers will keep hitting the line between "structural"
and "producer-circumstance" that ADR-018 drew narrowly.

### Alternative 2: Path B as the original ticket framed it

**Description:** Recorder rewrites `RowOutcome.ROUTED` to `RowOutcome.COMPLETED`
on successful sink-write. Keep the single-axis enum.

**Rejected because:** it erases the producer-circumstance discriminant
(intentional gate MOVE vs default flow) without putting it anywhere else. The
audit row loses information; consumers must reconstruct routing from
`routing_events`. ADR-019 instead moves the discriminant to a dedicated column
(`path`) where it is at least as visible as before, and more so for queries
that group by provenance.

### Alternative 3: Two-value outcome (`SUCCESS | FAILURE` only, no `STRUCTURAL`)

**Description:** Force every row to `SUCCESS` or `FAILURE`. Parent-token
bookkeeping (`FORKED`, `EXPANDED`, `CONSUMED_IN_BATCH`) maps to a separate
non-terminal lifecycle (`completed=False, lifecycle=parent_bookkeeping`).
Sink-fallback (`DIVERTED`) maps to `FAILURE`.

**Rejected because:** "non-terminal terminal" is a contradiction; FORKED rows
HAVE reached a terminal from the parent's POV, they just don't have a
lifecycle answer of their own. Forcing `DIVERTED` to `FAILURE` overloads
"failure" with "sink-write fallback for visibility," which ADR-018 line 56
explicitly excluded. `STRUCTURAL` is the smallest deviation that keeps the
model honest.

### Alternative 4: Tagged-union dataclass instead of two enums

**Description:** Replace `RowOutcome` with a `TerminalState` dataclass
carrying `outcome` and `path` together; absence-of-`TerminalState` means
non-terminal.

**Rejected because:** at the audit-DB layer, columns are the natural primitive
and a tagged union serializes back to two columns anyway. The dataclass
abstraction adds Python-side ergonomics but no audit-DB benefit. A future
refactor can introduce a Python-level dataclass over the column pair without
needing a new ADR.

### Alternative 5: Default-failure-sink, eliminating `path=UNROUTED`

**Description:** Engine implies a "default failure sink" for transform throws
without `on_error` and coalesce-quorum-fail. Every failure has a routing
destination; `UNROUTED` is unnecessary.

**Rejected for ADR-019:** would require pipeline-config additions and
implicit-default semantics that ELSPETH does not have today. Defer to a
separate decision if operators ever need it; `UNROUTED` is the honest
representation of "the engine had nowhere to route this failure" until that
config burden is taken on.

## Open Questions for the Maintainer

These are deliberate sub-decisions ADR-019 does not pre-empt. Each affects
implementation but not the core model. Resolve before moving the ADR to
Accepted.

1. **Should `QUARANTINED` and `FAILED` share a single `path=UNROUTED` value,
   or stay separate (`QUARANTINED_AT_SOURCE` vs `UNROUTED`)?** ADR-019 keeps
   them separate because the source-side coercion failure (Tier 3 boundary,
   CLAUDE.md "Trust Flow") is distinct from a transform-time unrouted
   failure. If the operator UI never distinguishes them, collapsing is
   defensible.

2. **Does `DROPPED_BY_FILTER` belong as `SUCCESS` or `STRUCTURAL`?** ADR-018
   line 52 puts it in `rows_succeeded`. ADR-019 follows that. But a transform
   that intentionally emits zero rows is arguably structural (the row never
   "succeeded" — it was dropped). The current classification is preserved for
   continuity; revisit if the predicate role feels wrong.

3. **Should `completed` be a materialized column or a derived view?**
   Materializing is the recommendation (query ergonomics, mirrors existing
   `is_terminal` column at `model_loaders.py:539`). Deriving (`completed = outcome
   IS NOT NULL`) is cheaper but loses ergonomics. Tier 1 cross-check covers
   either choice.

4. **Should the wire-schema counter rename happen in a follow-on ADR-020?**
   Recommendation: yes, separately. ADR-019 is large enough; the wire-schema
   stability promise from ADR-018 line 109-114 is its own conversation.

## Related Decisions

- **Supersedes:** ADR-018 (Producer-Site Outcome Discrimination, 2026-05-02)
- ADR-004: Explicit Sink Routing
- Filigree issue `elspeth-d376b8e681`: the deferred Path B question that
  motivated this ADR.
- Filigree issue `elspeth-5069612f3c`: the rows-routed counter split that
  motivated ADR-018.

## Implementation Notes

The implementation surface, in order of dependency:

1. `contracts/enums.py`: introduce `TerminalOutcome` and `TerminalPath`,
   delete `RowOutcome` and `_TERMINAL_ROW_OUTCOMES` / `_NON_TERMINAL_ROW_OUTCOMES`,
   add the new closed-set assertion over the mapping table.
2. `core/landscape/data_flow_repository.py` and `core/landscape/model_loaders.py`:
   migrate the audit-row read/write to two columns plus `completed` bool.
   Tier 1 cross-checks update.
3. `engine/orchestrator/outcomes.py`: replace the per-`RowOutcome` branches in
   `accumulate_row_outcomes` with a (outcome, path) match. Counter logic is
   unchanged in shape but reads from the new fields.
4. `engine/orchestrator/core.py`: `_derive_resume_terminal_status_from_audit`
   reads the new column pair.
5. `engine/executors/sink.py:635`: write the new column pair instead of
   `RowOutcome`.
6. Producer sites that emit `RowOutcome.X`: each site emits the (outcome, path)
   pair instead.
7. Tests (~72 hits): update assertions in the same commit per "no legacy
   code" policy.
8. Frontend types (`web/frontend/src/types/index.ts`): no change required for
   ADR-019 (counter names preserved); update if ADR-020 ships.
