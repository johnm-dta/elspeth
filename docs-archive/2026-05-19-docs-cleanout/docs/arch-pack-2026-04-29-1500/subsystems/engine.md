# `engine/` — L2 SDA Execution

**Layer:** L2 — outbound subset of `{contracts, core}`.
**Size:** 36 files, 17,425 LOC.
**Composite:** triggered by ≥10k LOC and ≥20 files.
**Quality score:** **4 / 5**.

The engine cluster is the L2 SDA execution tier. Its layer position is
unambiguous; its outbound edges are confined to `{contracts, core}` and
verified clean. The cluster ships no console scripts of its own — it
is invoked through `elspeth.cli` and exposes a 25-name `__all__` from
`engine/__init__.py` as the stable contract for L3 callers.

---

## §1 Responsibility

The L2 SDA execution layer:

- **Orchestrator** (run lifecycle, DAG execution per row / token).
- **RowProcessor** (per-row transformation pipeline).
- **Executors** — transform, coalesce, pass-through.
- **RetryManager** (Tenacity-backed; audit-hook contract).
- **ArtifactPipeline** (manages execution artefacts).
- **SpanFactory** (telemetry spans).
- **Triggers** (commencement gates, expression evaluators).
- **TokenManager** (token-identity façade over `DataFlowRepository`).

The cluster's defining architectural commitment is the **ADR-010
declaration-trust framework**: a 4-site dispatcher
(`pre_emission_check`, `post_emission_check`, `batch_flush_check`,
`boundary_check`) drives 7 contract adopters mapped 1:1 to ADRs 007 /
008 / 011 / 012 / 013 / 014 / 016 / 017.

---

## §2 Internal sub-areas

| Sub-area | Files | Notes |
|----------|------:|-------|
| `orchestrator/` | 6 | Decomposed from a single 3,000-LOC module into six focused siblings (`core.py`, `types.py`, `validation.py`, `export.py`, `aggregation.py`, `outcomes.py`) |
| `executors/` | several | Includes `state_guard.py:NodeStateGuard`, `declaration_dispatch.py`, transform / coalesce / pass-through |
| Top-level modules | ~20 | `processor.py`, `coalesce_executor.py`, `retry.py`, `tokens.py`, `bootstrap.py`, `triggers.py`, `commencement.py`, `dependency_resolver.py`, … |

---

## §3 Dependencies

| Direction | Edges |
|-----------|-------|
| **Outbound** | `{contracts, core}` (per layer model + clean enforcer) |
| **Inbound** | `{plugins, web, mcp, composer_mcp, telemetry, tui, testing, cli}` |

The engine has no nodes in any L3 import-graph SCC — its position is L2,
and the L3 oracle excludes engine internals from edge enumeration.
Internal coupling within `engine/` is described per sub-area in the
analysis catalogues; it is not redrawn here.

---

## §4 Findings

### E1 — ADR-010 dispatcher audit-completeness · **Resolved in prior pass**

The prior assessment closed this finding by direct read of
`src/elspeth/engine/executors/declaration_dispatch.py:120–172`.

- Both `except DeclarationContractViolation` (lines 137–141) and
  `except PluginContractViolation` (lines 142–150) branches **append
  to the violations list**; neither swallows.
- Post-loop logic correctly distinguishes 0 / 1 / N≥2 cases with
  reference-equality preservation at N=1 (the non-regression invariant).
- Docstring (lines 1–26) accurately describes the behaviour as
  audit-complete-with-aggregation per ADR-010 §Semantics.
- Test coverage: 1,923 LOC across:
  - `tests/unit/engine/test_declaration_dispatch.py` (642 LOC)
  - `tests/property/engine/test_declaration_dispatch_properties.py` (1,183 LOC)
  - `tests/integration/pipeline/orchestrator/test_declaration_contract_aggregate.py` (98 LOC)

**No remediation needed.**

### E2 — `processor.py` cohesion is unverified · **Medium**

`engine/processor.py` (2,700 LOC) carries a docstring claiming one
cohesive responsibility (`RowProcessor` end-to-end), but its imports
span six visible concerns:

- DAG navigation
- Retry classification
- Terminal-state assignment
- ADR-009b cross-check
- Batch error handling
- Quarantine routing

Whether the LOC reflects **essential complexity** (one responsibility
honestly large) or **accidental concentration** (multiple responsibilities
accreted) cannot be answered without a per-file deep-dive.

**Impact:** maintenance burden and onboarding friction; not a runtime
risk. **Recommendation:** [R5](../07-improvement-roadmap.md#r5).

### E3 — Engine integration tests have no in-cluster directory · **Medium**

There is no `tests/integration/engine/`. Engine integration coverage
exists in the integration tree but is not locatable within the engine
cluster's scope. `CLAUDE.md` requires that integration tests use
`ExecutionGraph.from_plugin_instances()` and
`instantiate_plugins_from_config()`; the rule is currently
un-auditable from inside engine.

**Recommendation:** [R4](../07-improvement-roadmap.md#r4) — a
cross-cluster integration-tier audit (distinct from any single L2
cluster pass).

---

## §5 Strengths

### Terminal-state-per-token invariant is structurally guaranteed

`engine/executors/state_guard.py:NodeStateGuard` implements "every row
reaches exactly one terminal state" as a **context-manager pattern**,
locked by:

- `tests/unit/engine/test_state_guard_audit_evidence_discriminator.py`
- `tests/unit/engine/test_row_outcome.py`

Context-manager-as-invariant for safety properties is genuinely good
architectural practice — the type system cooperates with the runtime to
make the invariant non-bypassable.

### ADR-010 dispatch surface is drift-resistant by construction

The 4-site × 7-adopter mapping is locked by an AST-scanning unit test
(`tests/unit/engine/test_declaration_contract_bootstrap_drift.py`).
Adding a new adopter without registering it fails CI.

### `orchestrator/core.py` (3,281 LOC) is in-progress decomposition

The orchestrator was previously a single 3,000-LOC module. It has been
refactored into six focused siblings (`core.py`, `types.py`,
`validation.py`, `export.py`, `aggregation.py`, `outcomes.py`) with a
stable public API. **Active remediation visible in the tree**, not
stagnant debt waiting for someone.

### `coalesce_executor.py` (1,603 LOC) is essential complexity

Four policies × three strategies × branch-loss handling × late-arrivals
× checkpoint resume genuinely populates the LOC. Resolved as essential
in the prior assessment; no further action needed.

---

## §6 Token identity: a three-locus split

The engine's token-identity story spans the engine and core clusters:

| Locus | What it does |
|-------|-------------|
| `engine/tokens.py` (`TokenManager`, 399 LOC) | Engine-side façade for token lifecycle (create / fork / coalesce / update). The docstring (`tokens.py:1-5`) names this explicitly as "a simplified interface over `DataFlowRepository`." |
| `core/landscape/data_flow_repository.py` (1,590 LOC, out of cluster) | The persistence of token identity. `tokens.py:19` is the cross-layer import that wires the two together. |
| `engine/processor.py` + `engine/orchestrator/core.py` | The call sites where tokens are minted at fork and consumed at coalesce. |

Token identity is therefore a shared concern of `engine` and `core`,
not engine-only.

---

## §7 Cross-cluster handshakes

| Partner | Direction | Shape |
|---------|-----------|-------|
| `core/landscape/data_flow_repository` | engine → core | `tokens.py:19` imports `DataFlowRepository` (TokenManager façade) |
| `contracts/declaration_contracts` | engine → contracts | `executors/transform.py:16-23` imports `PostEmissionInputs`, `PreEmissionInputs`, `derive_effective_input_fields` (ADR-010 payload TypedDicts) |
| `core/expression_parser` | engine → core | Three sites consume the parser: `triggers.py:24`, `commencement.py:12`, `dependency_resolver.py:14` |
| `contracts.pipeline_runner` Protocol | engine → contracts | `bootstrap.py` and `dependency_resolver.py` consume `PipelineRunner` (contracts-defined, engine-implemented at orchestration scope) |

For the audit-trail backbone view that combines engine, core, and
contracts, see [`../04-component-view.md#3-the-audit-trail-backbone`](../04-component-view.md#3-the-audit-trail-backbone).
