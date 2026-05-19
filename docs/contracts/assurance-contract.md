# ELSPETH Assurance Contract

**Status:** Current RC-5.2 contract summary
**Audience:** operators, plugin developers, engine developers, and reviewers

This document states the promises ELSPETH treats as product contracts. It is the
visible replacement for the old RC-3 release guarantees snapshot. Detailed
mechanics live in the linked contract, architecture, and runbook documents.

## Core Promise

Every output ELSPETH produces must be attributable to its source data, runtime
configuration, plugin contracts, execution path, and terminal outcome.

For a completed run, an operator must be able to answer:

- which source row or token produced this output
- which transforms, gates, aggregations, forks, coalesce steps, and sinks touched it
- which external calls were made on its behalf
- which configuration, secret references, and validation commitments were active
- whether the token succeeded, failed, was routed, was quarantined, or ended in a structural terminal path
- which artifact or sink write contains the result

If that answer cannot be reconstructed from the Landscape audit record and the
run artifacts, the system has an audit-integrity defect.

## Must-Hold Guarantees

### Audit Primacy

The Landscape audit record is the source of truth. Telemetry and logs support
operations, but they do not replace audit evidence.

- Audit writes happen before telemetry.
- Row-level decisions belong in Landscape records, not logger messages.
- A corrupted audit record is a Tier 1 failure and should crash loudly.
- Retention may remove payload bytes, but metadata and hashes must remain
  sufficient to explain what happened and why.

See [Landscape System](../architecture/landscape.md),
[Landscape Entry Points](../architecture/landscape-entry-points.md), and
[Data Trust and Error Handling](../guides/data-trust-and-error-handling.md).

### Token Outcome Accountability

Every token must reach exactly one terminal outcome. Transient outcomes such as
buffering must be followed by a terminal outcome before the run is complete.

The current contract is the two-axis terminal model:

- lifecycle result: success, failure, transient, or structural
- path/provenance: default flow, gate route, on-error route, discard, fork,
  batch consumption, coalesce, expansion, quarantine, or unrouteable failure

This lets the system answer both "did this row succeed?" and "how did it get
there?" without overloading one enum.

See [Token Outcome Assurance](token-outcomes/README.md) and
[ADR-019](../architecture/adr/019-two-axis-terminal-model.md).

### Runtime Validation Before Execution

Invalid pipelines fail before they run whenever the invariant can be checked
without touching row data.

- Plugin ids and options validate against real plugin schemas.
- Route targets and graph structure validate before execution.
- Source, transform, and sink contracts validate at the earliest responsible boundary.
- Runtime validation commitments are recorded so later review can see which
  checks were active.

See [Execution Graph](execution-graph.md), [Plugin Protocol](plugin-protocol.md),
and [System Operations](system-operations.md).

### Deterministic Identity And Hashing

Audit identity and hashes must be stable.

- Canonical JSON rejects NaN and Infinity instead of normalising them silently.
- Token and row identity survive forks, coalesce, aggregation, expansion, and resume.
- Payload hashes must verify on read.
- Secret values are not stored raw; audit records carry safe fingerprints or
  references.

See [Token Lifecycle](../architecture/token-lifecycle.md) and
[Configuration Reference](../reference/configuration.md).

### Boundary Validation

External data is Tier 3 until validated at the boundary.

- Sources validate and normalise incoming rows.
- Transforms that call external systems validate responses before using them.
- Sinks receive validated pipeline data and should fail loudly on upstream type
  violations.
- User-visible validation errors should be specific enough to repair the pipeline.

See [Data Trust and Error Handling](../guides/data-trust-and-error-handling.md).

### Recovery And Operational Evidence

Interrupted work must either resume from a valid checkpoint or fail with an
explicit, diagnosable reason.

- Checkpoints preserve enough state to continue without reprocessing completed work.
- Resume rejects incompatible or stale schema shapes.
- Web run status, diagnostics, and artifacts must agree with the underlying
  Landscape run.
- Operational reset procedures must treat coupled databases as one unit when a
  cross-database invariant requires it.

See [Resume Failed Run](../runbooks/resume-failed-run.md),
[Staging Session DB Recreation](../runbooks/staging-session-db-recreation.md),
and [Troubleshooting](../guides/troubleshooting.md).

## Non-Guarantees

ELSPETH records and explains execution. It does not guarantee everything around
the execution environment.

- It does not guarantee LLM answer quality or third-party API availability.
- It does not make unsupported connectors available through the composer.
- It does not provide high-throughput streaming semantics.
- Aggregation timeouts are checked when the next row arrives or when the source
  completes; they are not true idle timers without a heartbeat source.
- Authentication, authorization, network isolation, and secret provisioning are
  deployment responsibilities unless a specific Web subsystem implements them
  for that deployment.

## Contract Violation Rule

Treat any broken must-hold guarantee as a bug, not as documentation drift.

Examples:

- a row enters execution and no terminal outcome is recorded
- an `explain()` query cannot reconstruct lineage for a produced output
- an external call influences a row without a corresponding audit path
- a runtime validation commitment is claimed but not enforced
- a secret value is persisted raw where only a fingerprint or reference belongs
- a web run reports success while Landscape records failure

File or promote those defects in Filigree with enough evidence for another
agent to reproduce the missing guarantee.
