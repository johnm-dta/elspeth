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

For this PR, that means:

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
- `docs/superpowers/plans/2026-05-02-rows-routed-counter-split.md`
- Filigree issue `elspeth-5069612f3c`
