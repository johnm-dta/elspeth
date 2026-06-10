# ADR-028: QUEUE and COALESCE Are Not Duplicates — Leave Them Separate

**Date:** 2026-06-11
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** dag, queue, coalesce, barrier, schema-contract, multi-source

## Context

The fork/coalesce architectural assessment
(`notes/fork-coalesce-architecture-assessment-2026-06-10.md`, finding F5)
examined whether the QUEUE node type duplicates the coalesce barrier. The
two can look alike from a distance: both are DAG nodes at which multiple
upstream paths converge into one downstream path.

They are not alike on inspection.

- **QUEUE** (`src/elspeth/core/dag/builder.py:274-291`) is pass-through
  multi-source fan-in. It coordinates scheduling across sources; it does
  not merge fields, hold tokens against a condition, or synthesize
  guarantees across its inputs. Its output schema is **deliberately
  observed** (`SchemaConfig(mode="observed", fields=None)`) precisely
  because V1 queue semantics promise nothing about the union of source
  shapes.
- **COALESCE** (`src/elspeth/engine/coalesce_executor.py`) is a
  field-merging barrier over fork siblings of a single `row_id`. It holds
  tokens until a merge policy is satisfied, merges their fields, and emits
  one token under a **computed, typed merged contract** (the DAG builder
  precomputes policy-aware union semantics via `merge_union_fields`).

Different layers (build-time graph coordination versus runtime token
barrier), opposite schema postures (deliberately observed versus computed
and enforced). A future Barrier abstraction effort (assessment finding
F3-long-term) is the foreseeable moment someone might mistake these for a
unifiable pair.

## Decision

QUEUE and COALESCE remain separate. Do not unify them, and do not include
QUEUE in the scope of any future Barrier abstraction. The pair that
abstraction targets is aggregation + coalesce (the structural twins
documented in [barrier-machinery.md](../barrier-machinery.md)), not queue
+ coalesce.

## Consequences

- The Barrier abstraction effort has an explicit non-goal recorded before
  it starts; scoping it to include QUEUE is a design error, not an open
  question.
- QUEUE's observed schema posture stays intact. Any future proposal to
  give queues merged or synthesized schemas is a new decision requiring
  its own ADR, not a cleanup.
- A small amount of surface-level resemblance (two fan-in node types)
  persists in the codebase by design.

## Alternatives Considered

**Unify QUEUE into the coalesce/barrier machinery.** Rejected. The
resemblance is topological only. Coalesce correlates fork siblings by
`row_id` and owes its consumers a typed merged contract; a queue
correlates nothing and deliberately promises nothing. Forcing them under
one abstraction would either weaken coalesce's schema guarantees or
attach barrier state to a node that must remain pass-through.

## Related Decisions

- [ADR-025](025-multi-source-ingestion.md) — multi-source ingestion, the
  feature QUEUE fan-in serves.
- [ADR-026](026-durable-token-scheduler.md) — the scheduler within which
  coalesce barrier state is being made durable.
- [barrier-machinery.md](../barrier-machinery.md) — the twin pair
  (aggregation + coalesce) that *is* the legitimate unification target.

## References

- `notes/fork-coalesce-architecture-assessment-2026-06-10.md`, finding F5
  (QUEUE vs COALESCE checked deliberately; verdict: not duplicates).
