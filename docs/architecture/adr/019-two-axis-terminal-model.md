# ADR-019: Two-Axis Terminal Model — Lifecycle, Outcome, and Path

**Date:** 2026-05-04
**Status:** Accepted (re-accepted 2026-05-04 against round-3 corrected text; supersedes ADR-018; round-2 acceptance was reverted earlier 2026-05-04 after the round-3 panel caught a mechanical-test reasoning error — see "Round-2 Correction" subsection for the honest record; round-4 amendment 2026-05-04: §"Public API field-name preservation" rewritten as §"Counter derivation contract — public API field names preserved" to make the `(outcome, path) → counter` mapping at lines 99-115 explicitly normative and to enumerate the two accumulator behaviour changes the migration must ship — no Re-Accept, contract clarified)
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
> (`engine/executors/sink.py:952`, `sink_name=failsink_name`) writes a single
> `DIVERTED` `token_outcomes` row, paired with a `NodeStateStatus.COMPLETED`
> `node_state` for the failsink (`sink.py:898-903`) and a registered
> `artifacts` row (`sink.py:938-946`). The failsink's durable artifact carries
> the row's lifecycle answer; the `DIVERTED` `token_outcomes` row is
> bookkeeping for "see the failsink artifact." Genuinely transient on the
> original-sink side. **Discard mode** (`sink.py:998`,
> `sink_name="__discard__"`) writes the `DIVERTED` `token_outcomes` row
> alongside `NodeStateStatus.FAILED` (`sink.py:991`); no failsink node_state,
> no `artifacts` row. The `DIVERTED` row IS the permanent audit answer for a
> row that reached no destination. ADR-018 line 56 implicitly classed both
> flavors as non-predicate-input. ADR-019 corrects this: discard-mode is
> `(FAILURE, SINK_DISCARDED)` and a predicate input. The corrected mechanical
> framing — producer declares; topology cannot derive (`ROUTED_ON_ERROR` is
> topologically identical to failsink-mode `DIVERTED`, so the audit DB cannot
> distinguish FAILURE from TRANSIENT without producer declaration) — is
> developed in "Classification is producer-declared, not topology-derivable"
> below. This is a deliberate semantic correction; see Behavior Change
> Notice immediately above Consequences for the user-visible run-status
> impact.

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
  per `engine/executors/sink.py:952`): a paired `NodeStateStatus.COMPLETED`
  `node_state` for the same `token_id` exists at the failsink node, with a
  registered row in the `artifacts` table (`sink.py:898-946`). The failsink's
  durable artifact carries the lifecycle answer; the original-sink `DIVERTED`
  `token_outcomes` record is bookkeeping for "the intended sink-write didn't
  work, see the failsink artifact." Genuinely transient on the original-sink
  side. (Note: only ONE `token_outcomes` row is written for the diverted
  token — the DIVERTED row at `sink.py:952`. The lifecycle answer lives in
  `node_states` + `artifacts`, not in a paired `token_outcomes` record. An
  earlier draft of this ADR claimed otherwise; see "Round-2 Correction"
  below.) **Discard-mode `DIVERTED`** (`sink_name="__discard__"`, per
  `sink.py:998`) is materially different: no paired failsink `node_state`,
  no `artifacts` row, primary `node_state` completed at
  `NodeStateStatus.FAILED` (`sink.py:991`). The row reached no destination
  and the `DIVERTED` `token_outcomes` record IS the permanent audit answer.
  Classified `(FAILURE, SINK_DISCARDED)` and a predicate input — see
  Behavior Change Notice for the user-visible run-status consequence.

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

### Classification is producer-declared, not topology-derivable

The mapping table above is canonical; **producers declare the (outcome, path)
pair at the emit site.** The audit tier records the declaration; cross-checks
(below) verify structural consistency for the `TRANSIENT` direction. Topology
cannot derive the classification in the `SUCCESS`-vs-`FAILURE` direction:
`ROUTED_ON_ERROR` (transform threw, on-error sink received the row) and
`SINK_FALLBACK_TO_FAILSINK` (original sink-write failed, failsink absorbed
for visibility) produce structurally identical artefacts at the audit
layer — both create a paired `NodeStateStatus.COMPLETED` `node_state` for the
same `token_id` at a different node, plus a registered `artifacts` row. Only
the producer knows whether "transform's work failed but routing succeeded"
versus "original sink-write failed but failsink absorbed." The audit DB
cannot recover that distinction. CLAUDE.md's Auditability Standard ("no
inference — if it's not recorded, it didn't happen") forecloses the
reconstruction.

**Classification rationale (descriptive, not derivational):** a `TRANSIENT`
token has its row's lifecycle answer durably recorded *elsewhere* — either
(a) on a paired `token_outcomes` record reachable via the same `row_id`
lineage (`FORK_PARENT`, `EXPAND_PARENT`), (b) through the consuming batch row's
`batches.status == COMPLETED` witness reached by `batch_id` (`BATCH_CONSUMED`),
or (c) on a paired `NodeStateStatus.COMPLETED` `node_state` for the same
`token_id` at a different node, with a registered row in the `artifacts` table
(`SINK_FALLBACK_TO_FAILSINK`). A `SUCCESS` or `FAILURE` token IS its row's
lifecycle answer.

### Cross-check invariants (verify producer declaration; do not derive)

The recorder Tier 1 cross-checks below verify *some* structural facts in the
`TRANSIENT` direction. They catch violations of the producer's declared
classification; they do not derive the classification. Where the cross-check
cannot run in real time (children/batch-result tokens land later), the
invariant is a *deferred* obligation rather than a write-time assertion.

- **I1a (lineage-paired):** `(TRANSIENT, FORK_PARENT)` and
  `(TRANSIENT, EXPAND_PARENT)` require ≥1 child `token_outcomes` row with
  `parent_token_id == this.token_id`. *Deferred* — children complete after
  the parent.
- **I1b (aggregate-paired):** `(TRANSIENT, BATCH_CONSUMED)` requires the
  consuming batch row to reach `BatchStatus.COMPLETED` by end of run. The
  batch-result token created at flush time does not carry `batch_id` in its own
  `token_outcomes` row; the invariant witness is the `batches.status` row linked
  from the `BATCH_CONSUMED` token's `batch_id`. *Deferred*.
- **I1c (sink-fallback-paired):** `(TRANSIENT, SINK_FALLBACK_TO_FAILSINK)`
  requires a paired `NodeStateStatus.COMPLETED` `node_state` for the same
  `token_id` at the failsink node, AND that node_state has a registered
  `artifacts` row. *Real-time verifiable* — `engine/executors/sink.py:898-946`
  registers the artifact and completes the failsink node_state before
  `record_token_outcome()` at line 952.
- **I3 (discard-FAILURE):** `(FAILURE, SINK_DISCARDED)` requires
  `sink_name="__discard__"` AND no paired `NodeStateStatus.COMPLETED` sink
  `node_state` for the same `token_id` (no failsink absorbed). *Real-time
  verifiable* — `sink.py:977-1003` does not call `register_artifact()` and
  completes the primary node_state at `NodeStateStatus.FAILED` (line 991).

I1a/I1b/I1c/I3 are cross-row/cross-table invariants — strictly stronger than
the single-row scalar guards at `model_loaders.py:547-580` that the
implementation will need to extend. The single-row guards remain in force
for required-field constraints (`sink_name`, `error_hash`, `fork_group_id`);
the new invariants add structural consistency between `token_outcomes` and
`node_states`/`artifacts`.

### Counter derivation contract — public API field names preserved

Wire schemas (`ProgressEvent`, `RunSummary`, `CompletedData`) continue to
expose the existing predicate-role counter names: `rows_succeeded`,
`rows_routed_success`, `rows_routed_failure`, `rows_failed`, `rows_quarantined`,
`rows_coalesce_failed`, plus structural counters (`rows_coalesced`,
`rows_forked`, `rows_expanded`, `rows_buffered`, `rows_diverted`). Per
ADR-018 line 109-114, counter *renames* are a separate breaking-API decision
and remain out of scope for ADR-019. The frontend counter widgets and
operator dashboards do not require changes for ADR-019.

What ADR-019 *does* normatively specify is the **`(outcome, path) →
counter increment` mapping** — the bridge between producer emissions under
the two-axis model and the public counter surface that
`RunResult.__post_init__` (`src/elspeth/contracts/run_result.py:77-142`)
consumes to derive `RunStatus` (the four-value taxonomy made publicly
visible at `/api/runs/{rid}` by commit `cc895589`). The Mapping table at
lines 99-115 above IS this contract; read it by `(outcome, path)` as the
key. The accumulator at
`src/elspeth/engine/orchestrator/outcomes.py:235-307` is the authoritative
implementation; any deviation between the table and the accumulator is a
bug to fix in the accumulator, not a permitted hidden behavior.

#### Behavior change at the contract layer

Two existing accumulator behaviors change under the new model to align
producer emissions with the canonical predicate
(`success_indicator = rows_succeeded > 0`,
`failure_indicator = rows_failed > 0`). Live code on RC5 does NOT yet
match the contract for these two cases:

- **`(SUCCESS, GATE_ROUTED)` (was `RowOutcome.ROUTED`).** Under the new
  model, a gate-routed-and-sunk row increments BOTH `rows_succeeded` AND
  `rows_routed_success`. Live code today only increments
  `rows_routed_success` (`outcomes.py:245`). The migration's atomic
  recorder/producer flip MUST update the accumulator. This is the
  positive-side change that makes the bifurcated
  `OR rows_routed_success > 0` predicate clause vestigial and removable.
- **`(FAILURE, ON_ERROR_ROUTED)` (was `RowOutcome.ROUTED_ON_ERROR`).**
  Symmetric: increments BOTH `rows_failed` AND `rows_routed_failure`.
  Live code today only increments `rows_routed_failure`
  (`outcomes.py:266`). The migration updates the accumulator together
  with the predicate.

These two accumulator updates are necessary AND sufficient for
`success_indicator = rows_succeeded > 0` and
`failure_indicator = rows_failed > 0` to be the canonical predicates
under the new model. The migration plan's Stage 2/3 (merged) PR ships
the accumulator change in lockstep with the `RunResult.__post_init__`
predicate rewrite — neither edit is safe in isolation.

The third behaviour change is the discard-mode flip already documented
in §Behavior Change Notice: `(FAILURE, SINK_DISCARDED)` now increments
`rows_failed`, where `RowOutcome.DIVERTED` (discard flavor) previously
incremented neither succeeded nor failed counters. This change is
operator-visible via `RunStatus`.

A follow-on ADR-020 *may* revisit counter names (e.g., aligning
`rows_routed_success` with the `(SUCCESS, GATE_ROUTED)` vocabulary)
if the team wants to. That is an independent breaking-API decision,
not a prerequisite for ADR-019 implementation.

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
(`MEMORY.md::project_db_migration_policy`) applies to the Landscape audit store.
There is no in-place audit-store migration: operators replace old pre-ADR-019
`audit.db` / Landscape audit-store schemas when ADR-019 ships. ADR-019 does not
change the web session schema; operators preserve `sessions.db` unless a
separate web-session compatibility check proves that database stale and the
operator runbook is amended with explicit session backup/restore steps. The
recorder, wire schemas, frontend counter readers, and integration tests flip
together. Migration surface is approximately 700–800 `RowOutcome.X`
references across ~80 files (precise count depends on commit; both
pre-composer-audit and post-composer-audit grep totals are in this band). Of
these, on the order of 90–150 are `outcome == RowOutcome` test assertion sites
that must flip to assert the new field pair. Re-grep at implementation-stage
start with `grep -rn 'RowOutcome\.' src/ tests/ | wc -l` and
`grep -rn 'outcome == RowOutcome\|\.outcome.*== RowOutcome' src/ tests/ | wc -l`
for the contemporary number. This is too large for a single coordinated
commit; the implementation rolls out as a sequenced PR series with the
closed-set partition assertion gating each stage:

1. **Contract layer** (`contracts/enums.py`): introduce `TerminalOutcome` and
   `TerminalPath`, retain `RowOutcome` temporarily as a derivable view over
   the new fields. Closed-set assertion runs over the new mapping.
2. **Recorder + repository** (`core/landscape/`): migrate the audit-row
   read/write to `(completed, outcome, path)`. Tier 1 cross-checks update
   per the invariant-translation table in Implementation Notes.
3. **Producers + accumulator** (`engine/orchestrator/`, `engine/executors/`):
   producer sites emit `(outcome, path)` pairs; accumulator reads from the
   new fields.
4. **Test migration** (`tests/`): the test assertion sites flip to the new
   field pair (count is in the 90–150 range described in Migration policy
   above; re-grep at stage start for the contemporary number). The
   temporary `RowOutcome` derivable view is removed.
5. **Final sweep**: delete the temporary `RowOutcome` view; closed-set
   assertion is now the only enforcement of the partition.

The "delete the DB" policy (`MEMORY.md::project_db_migration_policy`)
contains the data-side migration for the Landscape audit store only: operators
replace old `audit.db` / audit-store schemas between stages 1 and 5. No
in-place audit-store migration; no Alembic path for this ADR.

## Behavior Change Notice (operator-visible)

> ⚠ **Pipelines using discard sinks (`sink_name="__discard__"`) will see run
> status flip from `COMPLETED` to `COMPLETED_WITH_FAILURES` (or `FAILED` if
> all rows discard).** ADR-019 reclassifies discard-mode `DIVERTED` as
> `(FAILURE, SINK_DISCARDED)`, a predicate input. Today the engine already
> classifies discard at the node-state layer as `NodeStateStatus.FAILED`
> (`engine/executors/sink.py:991`); the token-outcome layer was silently
> disagreeing, leaving `failure_indicator` unset for runs with discarded
> rows. ADR-019 reconciles both layers.
>
> **Operator action required:** if your pipeline uses discard as silent
> housekeeping (rows you intend to drop without affecting run status),
> reconfigure to route those rows to a no-op success sink. If you are
> comfortable with the new semantics (discarded rows count toward
> `rows_failed` and flip `failure_indicator`), no action needed beyond
> re-baselining dashboards.

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
- **Migration surface is large.** ~700–800 `RowOutcome.X` references across
  ~80 files; recorder, repository load/store, resume aggregation, the
  closed-set partition assertion, and the explain() lineage report all
  participate. The "delete the DB" policy contains the data side; the code
  side is a sequenced 5-stage rollout (see "Migration policy" section), not
  a single commit.
- **Discard-sink behavior change (deliberate).** See "Behavior Change
  Notice" callout above Consequences for the operator-visible summary. The
  short form: discard-mode `DIVERTED` becomes `(FAILURE, SINK_DISCARDED)`
  and a predicate input, flipping `failure_indicator` for runs with
  discarded rows. The engine already classifies discard at the node-state
  layer as `NodeStateStatus.FAILED` (`sink.py:991`); the token-outcome
  layer was silently disagreeing. ADR-019 reconciles both layers.
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
source on 2026-05-04. The panel ran twice — see "Round-2 Correction"
subsection below for an honest account of why. After round-3, all five
sub-decisions are resolved **unanimously**. The verdicts are recorded here as
the binding interpretation of the mapping table above; if a future reader
disagrees, an ADR-020 amendment is the vehicle, not local re-litigation.

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
   `sink.py:998` writes the `DIVERTED` `token_outcomes` row alongside
   `NodeStateStatus.FAILED` (line 991). No paired failsink `node_state` and
   no `artifacts` row exist for the diverted token (`sink.py:977-1003`
   contains no `register_artifact()` call). The discard `DIVERTED` row IS
   the row's lifecycle answer; classifying as `TRANSIENT` would assert that
   the audit-row layer disagrees with the node-state layer about whether
   the row failed — a Tier 1 inconsistency. ADR-018 line 56's blanket
   "non-predicate-input" classification was a single-flavor decision based
   on the dominant failsink case; ADR-019 corrects an under-specified
   classification via cross-layer agreement (cross-check invariant I3 in
   the "Cross-check invariants" subsection above). See Behavior Change
   Notice for the user-visible run-status impact.

### Round-2 Correction

The first panel review (round-2, also 2026-05-04) accepted ADR-019 with a
mechanical test for `TRANSIENT` classification phrased as: *"does another
`token_outcomes` record exist for this token's row's lifecycle answer? If
yes, TRANSIENT."* Verdict 5 (discard-mode → FAILURE) was endorsed on the
grounds that no paired `token_outcomes` record existed for discard-mode but
one did exist for failsink-mode.

The maintainer's primary-source check disconfirmed the test. Failsink mode
writes exactly one `record_token_outcome` call (`engine/executors/sink.py:952`).
The "paired record" the round-2 panel relied on was the failsink's
`NodeStateStatus.COMPLETED` `node_state` (`sink.py:898-903`), which is in a
*different table* (`node_states`, not `token_outcomes`). All three
round-2 reviewers had endorsed the test on reasoning-coherence grounds
without verifying it pair-by-pair against `sink.py`.

Round-3 (also 2026-05-04) reconvened the panel as an actual collaborative
team rather than parallel reviewers. `panel-ac` walked the mapping table
pair-by-pair and surfaced a structural ambiguity: `ROUTED_ON_ERROR`
(`outcomes.py:266-291` — transform threw, on-error sink received the row)
produces a paired `NodeStateStatus.COMPLETED` `node_state` plus registered
`artifacts` row at a different node — *topologically identical* to
failsink-mode `DIVERTED` at the audit layer. Topology cannot derive the
classification: only producer declaration distinguishes "transform's work
failed but routing succeeded" (FAILURE) from "original sink-write failed
but failsink absorbed for visibility" (TRANSIENT). The panel converged on
**producer-declared classification, with audit-tier cross-checks verifying
structural consistency in the `TRANSIENT` direction only** — see the
"Classification is producer-declared, not topology-derivable" and
"Cross-check invariants" subsections above.

**Sub-decision conclusions are unchanged from round-2.** The reasoning
chain underneath sub-decision 5 is rebuilt: discard-mode → FAILURE rests
on `sink.py:991` (FAILED `node_state`), absence of `register_artifact()`
in `sink.py:977-1003`, and cross-check invariant I3 (cross-layer agreement
between `node_states` and `token_outcomes`) — *not* on the round-2 claim
about a "paired `token_outcomes` record."

**Panel-process finding (recorded for future ADR reviews):** for any
mechanical test or derivation invariant in ADR text, each reviewer must
independently verify the claim against primary source for *every row in
the relevant mapping table* before endorsing. Reasoning-coherence checks
are insufficient when the claim references audit-table sequencing.

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
   | `EXPANDED requires expand_group_id` | `(TRANSIENT, EXPAND_PARENT)` requires `expand_group_id` |
   | `COALESCED requires join_group_id AND sink_name` | `(SUCCESS, COALESCED)` requires `join_group_id` AND `sink_name` |
   | `QUARANTINED requires error_hash` | `(FAILURE, QUARANTINED_AT_SOURCE)` requires `error_hash` |
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
   (discard mode). The discard-mode site changes predicate behavior because
   `(FAILURE, SINK_DISCARDED)` is the canonical classification per
   sub-decision 5 (round-3 panel-resolved); see Behavior Change Notice for
   the operator-visible run-status impact.
6. Producer sites that emit `RowOutcome.X`: each site emits the `(outcome, path)`
   pair instead.
7. Tests (90–150 assertion sites; re-grep at stage start): update assertions
   in the stage-4 commit per the 5-stage Migration policy. The "no legacy
   code" policy applies once the temporary `RowOutcome` derivable view is
   removed in stage 5.
8. Frontend types (`web/frontend/src/types/index.ts`): no change required for
   ADR-019 (counter names preserved); update if ADR-020 ships.
