# Semi-Autonomous Platform Design

**Status:** Historical design draft; superseded by RC5 web composer architecture
**Last reviewed:** 2026-05-20

This document used to describe a proposed semi-autonomous platform built around
a separate conversation service, Temporal workflow service, preview workers,
execution workers, and sealed `PipelineArtifact` objects. That design was not
implemented as written and is no longer the current architecture.

Current source-of-truth documents:

- [Root architecture](../../../ARCHITECTURE.md) — current system architecture
- [README web composer path](../../../README.md#web-composer-path) — current
  operator-facing composer flow
- [Composer UX specification](../../../docs-archive/2026-06-28-docs-cleanout/docs/design/composer-ux-spec.md) — current
  composer north-star and execution model
- [Composer UX redesign set](../../../docs-archive/2026-06-28-docs-cleanout/docs/composer/ux-redesign-2026-05/) — RC5.2
  composer planning and implementation phase documents

Retained historical points:

- The semi-autonomous layer must not relax ELSPETH's audit, trust, lineage, or
  failure semantics.
- Model output is advisory; deterministic validation and auditable state changes
  remain authoritative.
- Preview evidence must come from real ELSPETH execution paths, not UI-only
  simulation.
- Approval and review artifacts must bind to inspectable pipeline state rather
  than an unstructured chat transcript.

Use this file only as historical context for the design direction that preceded
the current web composer work. Do not use it as an implementation plan.
