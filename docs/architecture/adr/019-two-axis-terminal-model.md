# ADR-019: Two-Axis Terminal Model — Lifecycle, Outcome, and Path

**Date:** 2026-05-04
**Status:** Accepted (panel-reviewed 2026-05-04, all sub-decisions resolved)
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

- **Vestigial bifurcated success predicate.** `contracts/run_result.py:208`:
  `success_indicator = rows_succeeded > 0 OR rows_routed_success > 0`. The OR
  is no longer load-bearing — Path A (RC5-UX, shipped) made `RowOutcome.ROUTED`
  bump both counters in both the live accumulator (`outcomes.py:251-265`) and
  the resume aggregation (`core.py:463-476`), so `rows_succeeded > 0` alone is
  sufficient under the current shipped code. The OR remains as defence-in-depth
  residue from before Path A. ADR-019 makes the OR removable *by construction*
  rather than by counter-symmetry convention: under the two-axis model,
  `success_indicator = rows_succeeded > 0` derives directly from
  `outcome=SUCCESS`, and the predicate stops needing the attribution counter
  as a fallback.
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
   - `TRANSIENT` — parent-token or sink-fallback bookkeeping; child tokens
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
`TRANSIENT` is not.

`TerminalPath` answers "how did the row get there?" — provenance enum,
producer-known and producer-emitted, never inferred from graph topology or
counter context. New paths require an enum addition at the producer site.

This continues ADR-018's mechanical-prompt discipline: a producer that emits a
new circumstance must add a `TerminalPath` value AND classify it in the
mapping table below.

### Mapping table (current `RowOutcome` → new model)

| Current `RowOutcome` | `completed` | `outcome` | `path` | Predicate counter(s) | Resume re-derive? |
| --- | --- | --- | --- | --- | --- |
| `COMPLETED` | `True` | `SUCCESS` | `DEFAULT_FLOW` | `rows_succeeded` | Yes |
| `ROUTED` | `True` | `SUCCESS` | `GATE_ROUTED` | `rows_succeeded`, `rows_routed_success` | Yes |
| `ROUTED_ON_ERROR` | `True` | `FAILURE` | `ON_ERROR_ROUTED` | `rows_failed`, `rows_routed_failure` | Yes |
| `DROPPED_BY_FILTER` | `True` | `SUCCESS` | `FILTER_DROPPED` | `rows_succeeded` | Yes |
| `COALESCED` | `True` | `SUCCESS` | `COALESCED` | `rows_succeeded`, `rows_coalesced` (structural) | Predicate only (`rows_coalesced` not re-derived) |
| `FAILED` | `True` | `FAILURE` | `UNROUTED` | `rows_failed` | Yes |
| `QUARANTINED` | `True` | `FAILURE` | `QUARANTINED_AT_SOURCE` | `rows_quarantined`, `rows_failed` | Yes |
| `DIVERTED` (failsink) | `True` | `TRANSIENT` | `SINK_FALLBACK_TO_FAILSINK` | `rows_diverted` (structural) | No |
| `DIVERTED` (discard) | `True` | `FAILURE` | `SINK_DISCARDED` | `rows_failed`, `rows_diverted` (structural) | Predicate only (`rows_diverted` not re-derived) |
| `FORKED` | `True` | `TRANSIENT` | `FORK_PARENT` | `rows_forked` (structural) | No |
| `EXPANDED` | `True` | `TRANSIENT` | `EXPAND_PARENT` | `rows_expanded` (structural) | No |
| `CONSUMED_IN_BATCH` | `True` | `TRANSIENT` | `BATCH_CONSUMED` | (deferred — counted at flush) | N/A — flush-time outcome carries the predicate role |
| `BUFFERED` | `False` | `NULL` | `BUFFERED` | `rows_buffered` (structural, non-terminal) | No |

> **Note on `DIVERTED` two-flavor split.** `RowOutcome.DIVERTED` is a single
> enum value today used for two materially different cases. **Failsink mode**
> (`engine/executors/sink.py:952`, `sink_name=failsink_name`) writes a
> `DIVERTED` token_outcome alongside a successful failsink token_outcome for
> the same `token_id` — the failsink record carries the lifecycle answer, so
> the `DIVERTED` record is bookkeeping (genuinely transient). **Discard mode**
> (`engine/executors/sink.py:998`, `sink_name="__discard__"`) writes only the
> `DIVERTED` record alongside `NodeStateStatus.FAILED` (`sink.py:991`); no
> paired record exists, so the `DIVERTED` record IS the permanent audit
> answer for the row. ADR-018 line 56 implicitly classed both flavors as
> non-predicate-input. ADR-019 corrects this: discard-mode is `(FAILURE,
> SINK_DISCARDED)` and a predicate input (panel-resolved 2026-05-04). The
> `TRANSIENT` invariant — "the lifecycle answer lives on a different token" —
> holds for failsink-mode but is false for discard-mode (no other token
> exists), and the engine itself already classifies discard at the node-state
> layer as `FAILED`. This is a deliberate semantic correction; see
> Consequences/Negative for the user-visible behavior change.

The bifurcated success predicate at `contracts/run_result.py:208` becomes:

```text
success_indicator  = COUNT(outcome = SUCCESS) > 0
failure_indicator  = COUNT(outcome = FAILURE) > 0
```

Equivalently: `rows_succeeded > 0` and `rows_failed > 0`. The `OR rows_routed_success > 0`
clause goes away because gate-routed-and-sunk rows already increment
`rows_succeeded` (they're `SUCCESS`/`GATE_ROUTED`).

### Why `TRANSIENT` exists as a third outcome value

The natural framing — "completed is a bool, outcome is success or failure" —
admits two outcome values. The remaining `RowOutcome` variants (`FORKED`,
`EXPANDED`, `CONSUMED_IN_BATCH`, and at least one flavor of `DIVERTED`) don't
carry a row's lifecycle answer; they are **transient tokens** — present in the
audit trail as bookkeeping markers while the row's actual lifecycle answer
lives on a different token. Permanent tokens traverse the full graph and reach
a real success or failure terminal as themselves; transient tokens are
temporary, and forcing them into `SUCCESS` or `FAILURE` would either
double-count or mis-attribute.

- **Parent-token bookkeeping** (`FORKED`, `EXPANDED`): the parent row's
  lifecycle ended in "spawned children" — not a success and not a failure.
  Children carry the lifecycle. Forcing `outcome=SUCCESS` on the parent would
  double-count successes; forcing `outcome=FAILURE` would mis-attribute.
- **Batch consumption** (`CONSUMED_IN_BATCH`): the row was absorbed into an
  aggregate. The batch-result token at flush time carries the lifecycle
  answer; the consumed row's record is bookkeeping for "where the absorption
  happened."
- **Sink fallback to a failsink** (`DIVERTED` with `sink_name=failsink_name`,
  per `engine/executors/sink.py:952`): two `token_outcomes` records exist for
  the same `token_id` — the failsink's record carries the lifecycle answer;
  the original-sink `DIVERTED` record is bookkeeping for "the intended
  sink-write didn't work, see the failsink record." Genuinely transient on
  the original-sink side. **Discard-mode `DIVERTED`** (`sink_name="__discard__"`,
  per `sink.py:998`) is materially different: there is no paired record
  carrying the lifecycle answer; the `DIVERTED` record IS the permanent audit
  answer for a row that never reached any destination. Its classification
  (transient or failure) is an explicit panel-review decision — see the
  mapping-table footnote and "Open Questions" below.

`TRANSIENT` makes these visible in `token_outcomes` with a path explaining the
circumstance, but explicitly excludes them from the success and failure
predicates. The closed-set assertion at module-import time enforces the
partition: every new `TerminalPath` value must be classified as predicate
input (with a counter) or structural (without one).

### `TRANSIENT` outcome vs "structural counter" — related but distinct

`TRANSIENT` (an outcome value) and "structural counter" (a counter
classification) are related but distinct, and conflating them is the
load-bearing future-reader trap. A `TRANSIENT` token typically increments a
structural counter (`rows_forked`, `rows_expanded`, `rows_diverted` on the
failsink-mode side, `rows_buffered` for the non-terminal case). But not every
structural counter is incremented by a `TRANSIENT` token: `rows_coalesced` is
a structural counter and the `COALESCED` row's outcome is `SUCCESS`, not
`TRANSIENT`. The vocabulary inherits from ADR-018's predicate-role table:
"structural" describes *counters* that record activity for visibility without
contributing to the run-status biconditional; "transient" describes the
*token's nature* — its audit record is bookkeeping while the actual lifecycle
answer lives on a different token.

The mechanical test for `TRANSIENT` is: **does another `token_outcomes`
record (or batch-result token) exist that carries this row's lifecycle
answer?** If yes, this token is `TRANSIENT`. If no, this token IS the
lifecycle answer and must be `SUCCESS` or `FAILURE`. This test is what
classifies failsink-mode `DIVERTED` (paired record exists → TRANSIENT) and
discard-mode `DIVERTED` (no paired record → FAILURE) into different cells.

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
tests flip together. Migration surface (verified by `grep -rn 'RowOutcome\.'
src/ tests/`): **779 total references across 81 files (13 in `src/`, 68 in
`tests/`)**, of which **143 are `outcome == RowOutcome` test assertions** that
must flip to assert the new field pair. This is too large for a single
coordinated commit; the implementation rolls out as a sequenced PR series with
the closed-set partition assertion gating each stage:

1. **Contract layer** (`contracts/enums.py`): introduce `TerminalOutcome` and
   `TerminalPath`, retain `RowOutcome` temporarily as a derivable view over
   the new fields. Closed-set assertion runs over the new mapping.
2. **Recorder + repository** (`core/landscape/`): migrate the audit-row
   read/write to `(completed, outcome, path)`. Tier 1 cross-checks update
   per the invariant-translation table in Implementation Notes.
3. **Producers + accumulator** (`engine/orchestrator/`, `engine/executors/`):
   producer sites emit `(outcome, path)` pairs; accumulator reads from the
   new fields.
4. **Test migration** (`tests/`): the 143 assertion sites flip to the new
   field pair. The temporary `RowOutcome` derivable view is removed.
5. **Final sweep**: delete the temporary `RowOutcome` view; closed-set
   assertion is now the only enforcement of the partition.

The "delete the DB" policy (`MEMORY.md::project_db_migration_policy`)
contains the data-side migration: operators discard old `audit.db` and
`sessions.db` between stages 1 and 5. No in-place migration; no schema
versioning.

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
- **Migration surface is large.** 779 `RowOutcome.X` references across 81
  files (143 test assertions); recorder, repository load/store, resume
  aggregation, the closed-set partition assertion, and the explain() lineage
  report all participate. The "delete the DB" policy contains the data side;
  the code side is a sequenced 5-stage rollout (see "Migration policy"
  section), not a single commit.
- **Discard-sink behavior change (deliberate).** Pipelines using discard
  sinks (`sink_name="__discard__"`, `engine/executors/sink.py:998`) currently
  produce no failure indicator for discarded rows — discarded rows increment
  `rows_diverted` (structural) but neither `rows_succeeded` nor `rows_failed`.
  Under ADR-019, discard-mode `DIVERTED` becomes `(FAILURE, SINK_DISCARDED)`
  and increments `rows_failed`, flipping `failure_indicator` for any run
  with discarded rows. Run status changes from `COMPLETED` to
  `COMPLETED_WITH_FAILURES` (or `FAILED` if all rows discard) for affected
  pipelines. This is the right correction (a row that reached no destination
  is a failed lifecycle, and the engine already classes discard as
  `NodeStateStatus.FAILED` at `sink.py:991` — the token-outcome layer was
  silently disagreeing with the node-state layer). Operators relying on
  discard as silent housekeeping must reconfigure or accept the new run-
  status semantics.
- **Three outcome values, not two.** The user's natural mental model
  ("success or failure") admits a `TRANSIENT` middle ground for parent-token
  bookkeeping. This is a justified deviation (see "Why `TRANSIENT` exists"
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

### Alternative 3: Two-value outcome (`SUCCESS | FAILURE` only, no `TRANSIENT`)

**Description:** Force every row to `SUCCESS` or `FAILURE`. Parent-token
bookkeeping (`FORKED`, `EXPANDED`, `CONSUMED_IN_BATCH`) maps to a separate
non-terminal lifecycle (`completed=False, lifecycle=parent_bookkeeping`).
Sink-fallback (`DIVERTED`) maps to `FAILURE`.

**Rejected because:** "non-terminal terminal" is a contradiction; FORKED rows
HAVE reached a terminal from the parent's POV, they just don't have a
lifecycle answer of their own. Forcing `DIVERTED` to `FAILURE` overloads
"failure" with "sink-write fallback for visibility," which ADR-018 line 56
explicitly excluded. `TRANSIENT` is the smallest deviation that keeps the
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

## Sub-decisions Resolved by Panel Review (2026-05-04)

The five sub-decisions enumerated below were originally tabled as "Open
Questions for the Maintainer." A three-agent review panel
(`axiom-solution-architect:solution-design-reviewer`,
`axiom-system-architect:architecture-critic`,
`yzmir-systems-thinking:pattern-recognizer`) reviewed ADR-019 against primary
source on 2026-05-04 and resolved all five **unanimously**. The verdicts are
recorded here as the binding interpretation of the mapping table above; if a
future reader disagrees, an ADR-020 amendment is the vehicle, not local
re-litigation.

1. **`QUARANTINED` path naming.** **Verdict: distinct (`QUARANTINED_AT_SOURCE`
   vs `UNROUTED`).** Source-side coercion failure is a Tier 3 trust-boundary
   outcome (CLAUDE.md "Three-Tier Trust Model"); transform-time unrouted
   failure is a Tier 2 outcome. Collapsing them erases the trust-tier
   provenance an auditor needs to distinguish "the input failed" from "our
   routing config has a gap." Operator-UI argument for collapsing is L9
   (parameter) and does not drive an L6 (information-structure) decision.

2. **`DROPPED_BY_FILTER`.** **Verdict: `outcome=SUCCESS, path=FILTER_DROPPED`.**
   The transform's intent was to drop the row — the row succeeded at being
   filtered. There is no paired token carrying a separate lifecycle answer,
   so the `TRANSIENT` invariant ("the lifecycle answer lives on a different
   token") does not apply. Preserves ADR-018 line 52's classification.

3. **`completed`: materialized column.** Mirrors the existing `is_terminal`
   column at `model_loaders.py:539`; the Tier 1 cross-check pattern (compare
   stored bool to derived bool, raise on mismatch) is the audit-integrity
   mechanism the project already runs. Deriving (`outcome IS NOT NULL`)
   removes the integrity gate to save a column — wrong cost-benefit in a
   system where the audit trail is the legal record.

4. **Counter rename: defer to ADR-020.** ADR-018 line 109-114 made an
   explicit wire-schema stability promise. Bundling counter renames into
   ADR-019 expands blast radius without adding correctness; separating them
   keeps each ADR's revert path clean.

5. **Discard-mode `DIVERTED`: `(FAILURE, SINK_DISCARDED)`, predicate input.**
   `sink.py:998` writes only the `DIVERTED` record alongside
   `NodeStateStatus.FAILED` (line 991). No paired token carries a separate
   lifecycle answer, so the `TRANSIENT` invariant fails for discard-mode.
   Classifying as `TRANSIENT` would assert the audit-row layer disagrees
   with the node-state layer about whether the row failed — a Tier 1
   inconsistency. ADR-018 line 56's blanket "non-predicate-input"
   classification was a single-flavor decision based on the dominant
   failsink case; ADR-019 corrects an under-specified classification. See
   Consequences/Negative for the user-visible run-status change for
   pipelines using discard sinks.

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
   migrate the audit-row read/write to `(completed, outcome, path)` plus
   the existing supporting columns (`sink_name`, `error_hash`, `fork_group_id`,
   `join_group_id`, etc.). Tier 1 cross-checks update per the
   invariant-translation table below. Every existing per-`RowOutcome` guard
   at `model_loaders.py:547-580` must be re-expressed in (outcome, path)
   terms in the same stage:

   | Existing guard (`model_loaders.py`) | New (outcome, path) guard |
   | --- | --- |
   | `COMPLETED requires sink_name` | `(SUCCESS, DEFAULT_FLOW)` requires `sink_name` |
   | `ROUTED requires sink_name` | `(SUCCESS, GATE_ROUTED)` requires `sink_name` |
   | `ROUTED_ON_ERROR requires sink_name AND error_hash` | `(FAILURE, ON_ERROR_ROUTED)` requires `sink_name` AND `error_hash` |
   | `FORKED requires fork_group_id` | `(TRANSIENT, FORK_PARENT)` requires `fork_group_id` |
   | `EXPANDED requires fork_group_id` | `(TRANSIENT, EXPAND_PARENT)` requires `fork_group_id` |
   | `COALESCED requires join_group_id AND sink_name` | `(SUCCESS, COALESCED)` requires `join_group_id` AND `sink_name` |
   | `QUARANTINED requires quarantine_reason` | `(FAILURE, QUARANTINED_AT_SOURCE)` requires `quarantine_reason` |
   | `DIVERTED requires sink_name AND error_hash` (failsink) | `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)` requires `sink_name` AND `error_hash` |
   | (DIVERTED / discard mode, currently same guard as failsink) | `(FAILURE, SINK_DISCARDED)` requires `sink_name="__discard__"` AND `error_hash` |
   | `is_terminal` cross-check (`model_loaders.py:539`) | `completed XOR (outcome IS NULL)` cross-check |

   Producers must populate the new columns at write time; the cross-check
   guards crash on inconsistency rather than coercing.
3. `engine/orchestrator/outcomes.py`: replace the per-`RowOutcome` branches in
   `accumulate_row_outcomes` with a (outcome, path) match. Counter logic is
   unchanged in shape but reads from the new fields.
4. `engine/orchestrator/core.py`: `_derive_resume_terminal_status_from_audit`
   reads the new column pair.
5. `engine/executors/sink.py`: write the new column pair instead of
   `RowOutcome` at every recorder write site — the primary terminal at line
   635, and the diversion sites at line 952 (failsink mode) and line 998
   (discard mode). The discard-mode site changes predicate behavior if the
   panel adopts ADR-019's recommendation to class discard-mode `DIVERTED` as
   `(FAILURE, SINK_DISCARDED)` rather than transient.
6. Producer sites that emit `RowOutcome.X`: each site emits the (outcome, path)
   pair instead.
7. Tests (~72 hits): update assertions in the same commit per "no legacy
   code" policy.
8. Frontend types (`web/frontend/src/types/index.ts`): no change required for
   ADR-019 (counter names preserved); update if ADR-020 ships.
