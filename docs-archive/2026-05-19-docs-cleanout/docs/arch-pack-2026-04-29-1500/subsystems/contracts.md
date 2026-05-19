# `contracts/` — L0 Leaf

**Layer:** L0 — leaf, no upward outbound permitted.
**Size:** 63 files, 17,403 LOC.
**Composite:** triggered by ≥10k LOC and ≥20 files.
**Quality score:** **5 / 5**.

The L0 leaf is the system's strongest cluster. It is mechanically
verified to import nothing above; its responsibility discipline is
coherent; its CI gates are stable; and the cross-cluster handshakes
against engine, core, and plugins are aligned.

---

## §1 Responsibility

`contracts/` owns the **shared vocabulary** that every higher layer
imports:

- Shared types, protocols, enums.
- Errors (raiseable exceptions and frozen audit DTOs).
- Frozen-dataclass primitives (`freeze.py`, `freeze_fields`).
- Audit-evidence DTOs (`AuditEvidenceBase` ABC and concrete subclasses).
- The `DeclarationContract` 4-site framework (the L0 surface of
  ADR-010).
- Schema-contract types and validators.
- Configuration alignment, defaults, protocols, and runtime types.
- Hashing primitives.
- Security primitives (secret types, plugin context).

Outbound dependencies are **empty by construction**. The leaf invariant
is mechanically verified by the L3 import oracle (zero outbound edges)
and by the layer-conformance scan (rule-`L1` count zero, rule-`TC`
count zero).

---

## §2 Internal sub-areas

| Sub-area | Files | Purpose |
|----------|------:|---------|
| Top-level modules | ~30 | `audit_evidence`, `declaration_contracts`, `plugin_context`, `plugin_protocols`, `errors`, `freeze`, `hashing`, `schema_contract`, `security`, … |
| `config/` sub-package | several | Configuration alignment, defaults, protocols, runtime |

---

## §3 Dependencies

| Direction | Edges |
|-----------|-------|
| **Outbound** | None (L0 leaf invariant) |
| **Inbound** | All other subsystems: `{core, engine, plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` |

The most consequential inbound surface is engine's consumption of the
ADR-010 declaration-contracts payload TypedDicts (`PostEmissionInputs`,
`PreEmissionInputs`, `derive_effective_input_fields`) at
`engine/executors/transform.py:16-23`. This is the contract that lets
the engine implement the 4-site dispatcher.

---

## §4 Findings

### K1 — `contracts/errors.py` mixes Tier-1 and Tier-2 surfaces · **Medium**

`contracts/errors.py` (1,566 LOC) holds Tier-1 raiseable exceptions,
Tier-2 frozen audit DTOs, structured-reason TypedDicts, and re-exported
`FrameworkBugError` in a single file. The Tier-1 / Tier-2 distinction
is currently encoded by inline comments, not by file split.

A CI-enforced split (e.g., `errors_tier1.py` versus `errors_dtos.py`)
would mechanise the discipline. The file relies on convention today.

**Recommendation:** [R11](../07-improvement-roadmap.md#r11). Split when
the file next requires material edits — don't split-for-the-sake-of-splitting.

### K2 — `plugin_context.py:31` TYPE_CHECKING smell · **Medium**

`contracts/plugin_context.py:31` carries the cluster's only cross-layer
reference: a TYPE_CHECKING import of `core.rate_limit.RateLimitRegistry`.
This is an ADR-006d Violation #11 candidate; an extracted
`RateLimitRegistryProtocol` in `contracts.config.protocols` would
eliminate the TYPE_CHECKING block.

The runtime is not coupled (annotation-only), but TYPE_CHECKING imports
are the canonical marker of a deferred structural fix, and ADR-006d's
"never lazy-import" rule forbids the pattern.

**Recommendation:** [R12](../07-improvement-roadmap.md#r12).

### K3 — `schema_contract` sub-package promotion · **Low**

The `schema_contract` cluster (8 files, ~3,500 LOC) has high internal
cohesion; promoting it to a `contracts/schema_contracts/` sub-package
would mirror the `config/` partition. **Organisational hygiene only.**

**Recommendation:** defer until a near-term ADR motivates it.

### K4 — Citation editorial defect · **Low**

Ten KNOW-A* citation IDs in the institutional knowledge map resolve
correctly but inline rationales mismatch. **Documentation correctness
only.**

**Recommendation:** [R10](../07-improvement-roadmap.md#r10).

---

## §5 Strengths

- **L0 leaf invariant is mechanically confirmed.** Zero outbound edges
  in [`../reference/l3-import-graph.json`](../reference/l3-import-graph.json);
  layer-conformance scan empty for both `L1` (upward import) and `TC`
  (TYPE_CHECKING upward import) findings. The leaf is a leaf, verifiably.
- **ADR-010 declaration-trust framework's L0 surface is complete.**
  - `AuditEvidenceBase` ABC.
  - `@tier_1_error` decorator with frozen registry.
  - `DeclarationContract` 4-site framework with bundle types and
    payload-schema H5 enforcement.
  - Secret-scrub last-line-of-defence.

  All present, all consumed by engine via the contracts-defined
  protocols.
- **Frozen-dataclass deep-immutability is enforced.**
  `contracts/freeze.py:freeze_fields` is the canonical pattern; CI
  detects forbidden anti-patterns (`MappingProxyType(self.x)` views,
  shallow wrapping, isinstance shortcuts) at
  `scripts/cicd/enforce_freeze_guards.py`.

---

## §6 Cross-cluster handshakes

For the synthesis-level discussion of how `contracts/` interfaces with
each other cluster, see:

- [`engine.md`](engine.md) §4 — declaration-contracts payload TypedDicts; `pipeline_runner` Protocol.
- [`core.md`](core.md) §4 — 50+ identifiers imported from `contracts/` (errors, payload protocols, freeze primitives, schema/security types, audit DTOs, checkpoint family, enums).
- [`plugins.md`](plugins.md) §4 — protocols and base classes for the plugin ecosystem.
- [`../05-cross-cutting-concerns.md#6-data-integrity-audit-trail`](../05-cross-cutting-concerns.md#6-data-integrity-audit-trail) — the L0 audit DTOs that thread the audit backbone.
