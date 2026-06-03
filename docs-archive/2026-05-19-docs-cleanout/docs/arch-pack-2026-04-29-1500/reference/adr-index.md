# ADR Index

Authoritative index of accepted Architecture Decision Records, derived
from `docs/architecture/adr/` at this pack's HEAD.

This index resolves the documentation-correctness defect captured as
[R10](../07-improvement-roadmap.md#r10): `ARCHITECTURE.md`'s ADR table
covers ADR-001 through ADR-006 only, but ADRs 007 through 017 are also
accepted.

> **Source of truth:** `docs/architecture/adr/`. Read each ADR for full
> context, alternatives, and consequences.

> **Discharge note for [R10](../07-improvement-roadmap.md#r10):** the
> table below is structured to be copyable into `ARCHITECTURE.md`'s
> ADR section. Doing so closes the "ADR-007..017 unindexed" defect
> directly; the rest of R10 can be addressed independently.

---

## Status overview

**17 ADRs accepted; 0 deprecated; 0 superseded.**

| # | Title | Status | Theme |
|--:|-------|--------|-------|
| [001](../../architecture/adr/001-plugin-level-concurrency.md) | Plugin-Level Concurrency (Push Complexity to the Edges) | Accepted | Concurrency |
| [002](../../architecture/adr/002-routing-copy-mode-limitation.md) | COPY Mode Limited to FORK_TO_PATHS Only | Accepted | DAG semantics |
| [003](../../architecture/adr/003-schema-validation-lifecycle.md) | Schema Validation Lifecycle | Accepted | Validation |
| [004](../../architecture/adr/004-adr-explicit-sink-routing.md) | Replace `default_sink` with Explicit Per-Transform Sink Routing | Approved (with conditions) | DAG semantics |
| [005](../../architecture/adr/005-adr-declarative-dag-wiring.md) | Declarative DAG Wiring — Explicit Input/Output Connections on All Nodes | Approved (P3 future extension) | DAG semantics |
| [006](../../architecture/adr/006-layer-dependency-remediation.md) | Layer Dependency Remediation — Enforcing Strict 4-Layer Import Direction | Accepted | Layer model |
| [007](../../architecture/adr/007-pass-through-contract-propagation.md) | Pass-through Contract Propagation — Declaration, Semantics, and Composer Parity | Accepted | Declaration trust |
| [008](../../architecture/adr/008-runtime-contract-cross-check.md) | Runtime Contract Cross-Check in `TransformExecutor` | Accepted | Declaration trust |
| [009](../../architecture/adr/009-pass-through-pathway-fusion.md) | Pass-through Pathway Fusion and Runtime-VAL Completeness | Accepted | Declaration trust |
| [010](../../architecture/adr/010-declaration-trust-framework.md) | Declaration-Trust Framework — Generalised Contract Protocol for Plugin Declarations | Accepted (amended 2026-04-20 — Amendment A3) | Declaration trust |
| [011](../../architecture/adr/011-declared-output-fields-contract.md) | Declared Output Fields Contract | Accepted | Declaration trust adopter |
| [012](../../architecture/adr/012-can-drop-rows-contract.md) | `can_drop_rows` Governance Contract | Accepted | Declaration trust adopter |
| [013](../../architecture/adr/013-declared-required-fields-contract.md) | Declared Required Input Fields Contract | Accepted | Declaration trust adopter |
| [014](../../architecture/adr/014-schema-config-mode-contract.md) | Schema Config Mode Contract | Accepted | Declaration trust adopter |
| [015](../../architecture/adr/015-creates-tokens-contract.md) | `creates_tokens` Remains a Permission Flag, Not a Production Declaration Contract | Accepted | Declaration trust scope |
| [016](../../architecture/adr/016-source-guaranteed-fields-contract.md) | Source Guaranteed Fields Contract | Accepted | Declaration trust adopter |
| [017](../../architecture/adr/017-sink-required-fields-contract.md) | Sink Required Fields Contract | Accepted | Declaration trust adopter |

---

## Themes at a glance

### Layer model (1 ADR)

ADR-006 establishes the strict 4-layer import direction enforced by
`scripts/cicd/enforce_tier_model.py`. Foundation for everything else
in this pack.

### Declaration-trust framework (11 ADRs)

ADRs 007 through 017 collectively define the framework that locks
plugin behavioural declarations to runtime enforcement.

- ADR-010 is the **framework spec**: the 4-site dispatcher
  (`pre_emission_check`, `post_emission_check`, `batch_flush_check`,
  `boundary_check`) and the declaration-contract protocol that adopters
  implement.
- ADRs 007, 008, 009 establish the pass-through contract surface that
  the framework generalised.
- ADRs 011, 012, 013, 014, 015, 016, 017 are the **seven contract
  adopters**. Each names a behavioural property a plugin can declare,
  and the framework enforces it at the relevant dispatch sites.

The 4-site × 7-adopter mapping is the surface the engine cluster's
AST-scanning drift test
(`tests/unit/engine/test_declaration_contract_bootstrap_drift.py`) locks
against unauthorised modification.

### Concurrency (1 ADR)

ADR-001 — plugins own their own concurrency surface; the engine does
not centralise it.

### DAG semantics (3 ADRs)

ADRs 002, 004, 005 — routing modes, explicit sink routing, declarative
DAG wiring.

### Validation (1 ADR)

ADR-003 — schema validation lifecycle (when validation runs relative to
plugin instantiation).
