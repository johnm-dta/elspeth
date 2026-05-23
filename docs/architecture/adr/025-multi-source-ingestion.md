# ADR-025: Multi-Source Ingestion — Source Surface Is Plural

**Date:** 2026-05-23
**Status:** Accepted
**Deciders:** John Morrissey, Claude Opus
**Tags:** sources, config, resume, schema-contract, no-legacy-code,
          rc6, multi-source-token-scheduler

## Context

Through RC5.2 the pipeline source surface was singular by contract and by
code. The CLAUDE.md project overview still records the rule verbatim:
*"Source: Load data — exactly 1 per run."* `ElspethSettings` carried a
single `source: SourceSettings`, `ExecutionGraph.from_plugin_instances`
took one `source` keyword argument, and every downstream consumer
(orchestrator, processor, audit trail, composer, redaction policy,
runbooks, runtime contract) modelled the pipeline as exactly one
producer feeding zero-or-more transforms feeding one-or-more sinks.

The RC6 branch (`feat/multi-source-token-scheduler`) adds first-class
multi-source ingestion: a pipeline may declare N named sources, fan-in
to ordinary processing nodes is mandatory through a new `NodeType.QUEUE`
graph node, per-source quarantine edges attribute failure to the source
that produced the row, and resume reconstructs each row's schema
contract from a per-source audit record (`run_sources`) so that
mixed-contract resume is sound.

The branch shipped multi-source *beside* single-source, not *in place of*
it. The structural state is unambiguous across four review lenses:

- `ElspethSettings.sources: dict[str, SourceSettings]` is the canonical
  surface. `ElspethSettings.source: SourceSettings` is preserved as an
  explicit transition shim, populated by a `model_validator(mode="after")`
  that takes `next(iter(self.sources.values()))`
  (`src/elspeth/core/config.py:1474-1479`). The field's description in the
  model already names itself: *"Transition shim for legacy single-source
  callers; canonical configuration is sources."*
- `build_execution_graph` accepts both shapes and routes the singular
  path through a `legacy_single_source_invocation = sources is None`
  branch (`src/elspeth/core/dag/builder.py:135-140`). The branch forces
  the source name to the literal string `"source"` to preserve
  pre-RC6 checkpoint identity. The comment is explicit: *"keeps the
  RC5.2 plugin-name/raw-config identity for checkpoint compatibility."*
- `ResumeState` carries both a singular `schema_contract: SchemaContract`
  field and a plural `schema_contracts_by_source: Mapping[NodeID,
  SchemaContract]` field
  (`src/elspeth/engine/orchestrator/types.py:519-535`). The two are not
  redundant: the plural map is authoritative when present, and the
  singular field is populated from
  `next(iter(schema_contracts_by_source.values()))` when it is — an
  arbitrary pick of one source's contract as "the" run contract
  (`src/elspeth/engine/orchestrator/core.py:3512`).
- The 26+ engine call sites that still read `config.source.*` rather
  than `config.sources[...]` carry the singular assumption into every
  downstream consumer. Examples: `_record_schema_contract` at
  `engine/orchestrator/core.py:2480-2515`, the singular-source branch in
  `_reconstruct_resume_state` at `orchestrator/core.py:3513-3545`,
  on-start attribution, redaction policy, audit-readiness source
  surfaces, blob-source guards, composer state, MCP read surfaces.

### Why this matters now

The architecture review filed 64 tickets under the
`multi-source-token-scheduler-audit` label (see
`/home/john/elspeth/notes/RC6-large-list.md`). The structural family
splits into four clusters:

1. **Dual-truth surface** — `config.source` / `config.sources`,
   singular / plural ResumeState fields, dual writers for source schema
   contract (`runs.contract_json` *and* `run_sources`), 3-tuple|4-tuple
   union for unprocessed rows, dual drain paths in `RowProcessor`.
2. **Named-source feature gaps** — blob-backed sources lose the plural
   map on rewrite (G1.1), wire_secret_ref cannot target named sources
   (P2), composer state-claim grounding ignores later named sources
   (P2), the audit-readiness panel omits non-primary sources (P2), the
   diff view collapses named-source-only changes (P2), inline source
   projection hides later inline sources (P2), the redaction policy
   doesn't carry per-source provenance (P3).
3. **Composer cannot author multi-source** — the composer LLM skill
   still teaches *"Every pipeline needs: one source"* (G16 /
   elspeth-86de46bcd4); the feature is unreachable to its primary
   persona until the skill catches up.
4. **Doc / governance corpus stale** — `CLAUDE.md`, `docs/release/
   guarantees.md §7.1`, `docs/reference/configuration.md`,
   `docs/contracts/system-operations.md`, the runbook surface, the RC1–
   RC5 progress note, and (until this ADR) the architecture-decision
   record itself all model the pipeline as single-source.

The structural decision recorded here precedes the implementation work
in clusters (1) and (2) so that downstream PRs do not ping-pong on the
question *"does the singular surface stay?"*

### Two paths were on the table

The branch-review consolidation surfaced two paths:

- **Conservative (invariant-check) path:** keep the singular
  `schema_contract` and `config.source` shapes; defend them with a
  runtime invariant that all configured sources share an identical
  schema contract; reject multi-source resume that violates the
  invariant. The legacy facade survives indefinitely.
- **Structural (plural) path:** delete the singular surface and the
  legacy facade outright; teach every consumer to look the row's
  contract up via `source_node_id`; allow mixed-contract pipelines as
  a first-class capability.

The operator chose the structural path. The decision below records
that choice.

## Decision

ELSPETH treats the pipeline source surface as **plural by contract and
by code**. The singular `source` surface is deleted, not deprecated.

### Concretely

1. **`ElspethSettings.sources: dict[str, SourceSettings]` is the only
   declared shape.** The `source: SourceSettings` transition shim, the
   `normalize_legacy_source` mode-before validator, and the
   `populate_legacy_source_view` mode-after validator are deleted in
   the same commit that lands the structural fix. A YAML that supplies
   `source:` instead of `sources:` is a configuration error, not a
   shim activation. ELSPETH has no users yet; no compatibility path
   is preserved (CLAUDE.md: *No Legacy Code Policy*).

2. **`build_execution_graph` takes plural sources only.** The
   `legacy_single_source_invocation` branch
   (`src/elspeth/core/dag/builder.py:135-140`,
   `:244-250`) is removed. `source: SourceProtocol | None` and
   `source_settings: SourceSettings | None` keyword arguments are
   removed from the signature. Source node identity always includes
   `source_name` in the config hash; the `"source"` literal name
   reserved for legacy checkpoint identity is no longer special.

3. **`ResumeState.schema_contract` is removed.** The replacement is
   `schema_contracts_by_source: Mapping[NodeID, SchemaContract]`, which
   is non-optional and never empty (resume rejects rather than picks).
   Every consumer of `schema_contract` migrates to look up the row's
   contract by `source_node_id`. The `next(iter(...))` arbitrary-pick
   at `orchestrator/core.py:3512` is deleted — the single dim2 sentence
   the reviewers all converged on: *"silently validates rows under the
   wrong schema — Tier 1 evidence tampering, the exact failure mode the
   trust model exists to prevent."* (Closes G2 / elspeth-01942858c3.)

4. **`unprocessed_rows` is a single shape, not a discriminated
   union.** The 3-tuple|4-tuple union at
   `engine/orchestrator/types.py:530` discriminated by `len()`
   collapses into a `ResumedRow` dataclass with
   `(row_id, row_index, source_node_id, row_data)`. `source_node_id`
   is non-optional. Closes G10 / elspeth-5335eb63e4.

5. **The singular contract writer goes away.** `runs.contract_json` is
   no longer written; `run_sources.schema_contract_json` is the single
   writer for per-source schema contracts. Audit-trail reads consult
   `run_sources` exclusively. Closes G6 / elspeth-2e2f2184ab.

6. **`get_source_id()` is deleted.** The function raises
   `GraphValidationError` on multi-source graphs by design; callers
   that need source identity look up the row's
   `source_node_id` directly (durable on every row), and callers that
   need the full source set iterate `source_ids: Mapping[str, NodeID]`.
   Closes G8 / elspeth-bdc43c911e.

7. **Composer state mirrors plural.** `CompositionState.source`
   singular field is deleted; the UI layer reads
   `CompositionState.sources` (G9 / elspeth-1ed6db3db4).

8. **Per-source quarantine, per-source on-start attribution, per-source
   blob ownership, per-source redaction provenance.** The structural
   move catches the family of P1/P2 named-source tickets
   (elspeth-1e3ae62d5e, elspeth-1e162ad261, elspeth-4548f560da,
   elspeth-9d30da4325, elspeth-d8cc46680d, elspeth-37c1dde240,
   elspeth-c33c71aafe, elspeth-a4b0c3e00f, elspeth-80671c0eb3,
   elspeth-d42360f518, elspeth-84d4680346, elspeth-6db715b7c3,
   elspeth-9738349228, elspeth-99f992f8bd, elspeth-9e083c57fe,
   elspeth-af612d0470, elspeth-dde60f76b4, elspeth-543ee35ed3,
   elspeth-3fc847c4be, elspeth-8fe6fc5f24, elspeth-7ef5d9ff67,
   elspeth-04980fc019, elspeth-336e3f704f) one by one — but they all
   share the precondition that the engine call sites consult the
   plural surface. This ADR commits to the precondition; the
   per-ticket work is downstream.

9. **`NodeType.QUEUE` is the only structural fan-in primitive for
   ordinary processing nodes.** Multi-source pipelines that MOVE-fan
   into a transform / aggregation / gate must route through an
   explicit QUEUE node; the graph refuses to validate otherwise
   (`src/elspeth/core/dag/graph.py:340-353`). Sinks remain exempt by
   policy (closes G30 / elspeth-30e7ac9571 — the policy is recorded
   here; the contract documentation work in
   `docs/contracts/system-operations.md` is downstream).

### What this is NOT

- This is not a commitment to *concurrent* multi-source execution.
  The orchestrator processes sources sequentially within a run (G12 /
  elspeth-bc81207798 — parallel ingest is downstream RC6 work). The
  determinism property recorded in ADR-001 — orchestrator is single-
  threaded — is preserved; ADR-026 (the durable token scheduler)
  records the queue primitive that makes future multi-worker
  ingestion safe to add without revisiting this ADR.
- This is not a claim that all sources in a pipeline share a
  schema contract. They explicitly may not. Mixed-contract pipelines
  are a first-class capability; each row's schema is recovered by
  `source_node_id` lookup against `run_sources`.
- This is not a guarantee that single-source pipelines pay no
  syntactic cost. A YAML file with one source must spell it
  `sources:\n  primary: {...}` (or similar), not `source: {...}`. The
  composer skill must teach the plural form from the first lesson
  (G16 / elspeth-86de46bcd4 — downstream).

## Consequences

### Positive Consequences

- The audit trail's per-row `source_node_id` becomes load-bearing,
  not advisory. An auditor querying "which schema contract validated
  row 42?" gets a deterministic answer from `run_sources` joined on
  `rows.source_node_id` — not "the first contract this run happened
  to record," which is what the singular `runs.contract_json` reader
  would have answered for a multi-source resume under the pre-ADR
  shape.
- The dual-truth surface collapses. A reviewer reading the engine no
  longer has to ask "is this site reading the canonical plural
  shape, the transition shim, or the legacy-facade branch?" There is
  one shape.
- Mixed-contract fan-in is a first-class capability. A pipeline that
  ingests `Person` rows from one source and `Org` rows from another,
  joins them through a queue, and writes to a single sink, can do so
  without a contract-merging compromise at the source boundary.
- The structural fix unlocks the named-source ticket family
  (cluster 2 in *Context*) by removing the precondition obstacle.
  Each of the ~25 named-source P1/P2 tickets becomes a focused
  per-feature fix rather than a precondition-and-fix coupling.
- The composer skill update (G16) becomes a single coherent rewrite
  rather than a *"teach plural but accept singular fallback"*
  hedge.

### Negative Consequences

- **Existing audit databases become unreadable on resume.** Runs
  recorded under the singular `runs.contract_json` writer cannot be
  resumed by the post-ADR engine. The operator policy is *delete the
  old DB* (project memory `project_db_migration_policy`), which is
  appropriate here because ELSPETH has no users yet. The runbook
  update (G19 / elspeth-559bce3459) must record the resume
  precondition explicitly.
- **Every engine call site that reads `config.source.*` is touched
  in one commit.** Roughly 26 sites; the change is mechanical but
  large. The CICD allowlist will see a corresponding shift
  (`tier-model` fingerprint rotation per the AST-shift discipline —
  memory `feedback_ast_shift_fingerprint_rotation`).
- **The composer LLM must be retaught.** Until the composer skill
  update (G16) lands, the composer authors a YAML the engine
  rejects. This is a load-bearing dependency between this ADR and
  G16; G16 cannot be deferred to a separate release window.
- **Documentation update is non-trivial.** CLAUDE.md, the release
  guarantees, the configuration reference, the contracts
  documentation, and the lineage runbook all carry the singular
  framing; the omnibus G14 / elspeth-e4cf92586c covers six files
  enumerated in the consolidation note.
- **The legacy `"source"` checkpoint identity reservation is gone.**
  Any external tooling that grepped for the literal string
  `"source"` as a node-id prefix breaks. The project has no such
  tooling today.

### Neutral Consequences

- The `dict[str, SourceSettings]` ordering is insertion-ordered
  (Python ≥ 3.7 guarantee). YAML preserves declaration order;
  `next(iter(sources.values()))` is well-defined when it's *needed*
  (e.g., audit-readiness "primary source" display), but is no
  longer load-bearing on the resume path.
- `source_settings_map` in `build_execution_graph` becomes a single
  required keyword argument rather than one half of a
  singular/plural pair. The signature shrinks by two parameters.
- The `node_id("source", source_name, source_node_config)` identity
  rule is universal; the `node_id("source", source_instance.name,
  source_config)` legacy form for single-source runs disappears.
  Future named-source pipelines and existing single-source
  pipelines share one identity scheme.

## Alternatives Considered

### Alternative 1: Keep the singular `schema_contract` + invariant-check

**Description:** Retain `ResumeState.schema_contract: SchemaContract`
and `ElspethSettings.source: SourceSettings`. Add a runtime invariant
asserting that every configured source declares an identical
`SchemaContract` (compared by canonical hash). Reject multi-source
configurations that violate the invariant at validation time. Resume
continues to use the singular contract because, by invariant, every
source produced rows under it.

**Rejected because:** The reviewers' four-lens diagnosis is that
mixed-contract fan-in is a legitimate use case and the only credible
reason to ship multi-source at all. A user joining `Person` rows from
HR and `Org` rows from procurement through a queue is the canonical
shape; forcing both sources to a synthetic super-schema at config
time discards information the audit trail then can't recover. The
invariant-check path also lies about the singular surface — the
engine would still read `config.source.*` at 26 sites under the
invariant, which means the dual-truth surface persists and every
named-source ticket in cluster 2 of *Context* remains blocked. The
invariant defends a fiction that the rest of the system has already
moved past.

### Alternative 2: Per-source separate runs

**Description:** Reject pipelines with N>1 sources at config time.
Operators wanting fan-in execute N separate single-source runs
upstream and pipe their outputs through a join sink to a downstream
single-source run.

**Rejected because:** Cross-source coalesce is core to the SDA model —
the existing Coalesce primitive (`docs/contracts/system-operations.md`)
explicitly merges results from parallel paths within one DAG, and
audit lineage is traced through one `run_id` per logical pipeline.
Splitting into N runs fragments the lineage: an auditor asking *"which
upstream rows produced this sink row?"* gets N+1 runs to join across,
rather than one. The audit-readiness panel, the lineage explorer
(`elspeth explain`), and the MCP failure-context surface all assume
one `run_id` per pipeline; per-source-run splitting would force a
parallel surface for each consumer. The cost is in the wrong place
(everywhere downstream) for a question that should be answered
upstream once (the DAG can hold N sources cleanly).

### Alternative 3: Retain legacy facade indefinitely as a documented escape hatch

**Description:** Keep the `legacy_single_source_invocation` branch in
`build_execution_graph`, the `normalize_legacy_source` model
validator, and the `populate_legacy_source_view` shim. Document the
singular surface as the "simple path" for tutorials and the plural
surface as the "advanced path" for fan-in.

**Rejected because:** The *No Legacy Code Policy* (CLAUDE.md) is
unconditional and the project's compensating control is *we have no
users yet*. A documented escape hatch is exactly the *"both old and
new branches"* anti-pattern the policy forbids; the existence of two
paths through `build_execution_graph` is the source of the dual-truth
critique the reviewers raised. The composer-skill consequence is
also asymmetric: teaching the LLM that the singular form *is also
valid* leaks back into engine call sites that defensively accept
both, which reproduces the present state.

## Tickets this ADR covers / unblocks

### Directly closes (after implementation lands)

- **G17 / elspeth-57d0031a14** — *No architectural doc explains the
  multi-source / scheduler design.* This ADR + ADR-026 close it.

### Architecturally anchors (implementation downstream, each becomes a
focused per-feature fix once this ADR's precondition holds)

- **G2 / elspeth-01942858c3** — arbitrary `schema_contract` pick on
  multi-source resume (the dim4 test plan from
  elspeth-d5f0194fc8 lands with the fix; P0).
- **G4 / elspeth-af87655cdb** — `ElspethSettings.source` field is a
  documented legacy shim.
- **G5 / elspeth-781e042709** — `legacy_single_source_invocation`
  facade in `build_execution_graph`.
- **G6 / elspeth-2e2f2184ab** — dual writer for source schema
  contract (`runs.contract_json` + `run_sources`).
- **G7 / elspeth-b680e81bce** — dual drain paths in `RowProcessor`.
- **G8 / elspeth-bdc43c911e** — `get_source_id` raises on
  multi-source graphs.
- **G9 / elspeth-1ed6db3db4** — `CompositionState.source` singular
  field is a UI-layer compatibility shim.
- **G10 / elspeth-5335eb63e4** — `unprocessed_rows` 3-tuple|4-tuple
  union discriminated by `len()`.
- **G11 / elspeth-11a4ed2630** — `TokenManager source_row_index /
  ingest_sequence` accept `None` defaults.
- **G30 / elspeth-30e7ac9571** — SINK exempt from "multi-producer
  requires QUEUE" rule (policy recorded; doc-tier follow-up
  unblocked).

### Named-source feature family (the structural precondition becomes
true; each ticket is then a focused fix)

- elspeth-1e162ad261 — Forked blob-backed sources rewrite path
- elspeth-336e3f704f — Forked multi-source states blob drops/leaks
- elspeth-4548f560da — Execution blob ownership for non-primary sources
- elspeth-9d30da4325 — Active-run blob guard for named sources
- elspeth-a7aa07b7ce — Blob-backed file source storage paths leak
  through redaction
- elspeth-d8cc46680d — Named source blob refs validated only on legacy
- elspeth-6ebb263e61 — `headers: original` sinks reuse wrong contract
  on multi-source
- elspeth-84d4680346 — Composer state-claim grounding ignores later
  named sources
- elspeth-6db715b7c3 — Tutorial runtime normalization drops named
  sources
- elspeth-1e3ae62d5e — `on_start` attribution uses first-source
  context for every source
- elspeth-37c1dde240 — `wire_secret_ref` cannot target named sources
- elspeth-c33c71aafe — Source secret wiring `target_id` ignored for
  named sources
- elspeth-a4b0c3e00f — Audit-readiness source surfaces omit named
  sources
- elspeth-80671c0eb3 — `diff_pipeline` source summary collapses
  named-source-only changes
- elspeth-d42360f518 — Pipeline graph view: named sources beyond
  compatibility invisible
- elspeth-9738349228 — Inline source projection hides later inline
  sources
- elspeth-99f992f8bd — `set_source_from_blob` invalid source names
  crash plugins
- elspeth-9e083c57fe — `set_source_from_blob` affected nodes use
  noncanonical component ids
- elspeth-af612d0470 — Named blob sources duplicate refs fail
  run-setup link insertion
- elspeth-3fc847c4be — Synthesised cache audit writer rejects
  multi-source topologies
- elspeth-543ee35ed3 — MCP and export read surfaces drop
  `resolved_prompt_template_hash`
- elspeth-9b17af34ca — Shared review snapshots without `sources` fail
  validation
- elspeth-e22a476972 — Shared inspect client validation omits
  `sources` from snapshot contract
- elspeth-7ef5d9ff67 — Run outputs panel stale artifact manifest
  across run switches
- elspeth-04980fc019 — Proof repair gate skips forced repair for
  named-source blob diagnostics
- elspeth-dde60f76b4 — Redaction policy doesn't carry per-source
  provenance
- elspeth-8fe6fc5f24 — Identity passthrough advisory: named-source
  producers ignored

### Documentation / governance follow-ups (RC6 publish gate, not merge
gate — per `project_multi_source_token_scheduler_rc6`)

- **G13 / elspeth-2409a7c7bf** — `CLAUDE.md "exactly 1 source per
  run"` (correct for RC5.2 today; update on RC6 ship).
- **G14 / elspeth-e4cf92586c** — Single-source doc corpus stale
  (omnibus, 6 files enumerated in consolidation note).
- **G15 / elspeth-bc91898548** — `docs/release/guarantees.md §7.1`
  "single-threaded in RC-3" contradicted.
- **G16 / elspeth-86de46bcd4** — Composer skill teaches "every
  pipeline needs one source."
- **G18 / elspeth-06aecb78a0** — `docs/architecture/landscape.md`
  missing `run_sources` schema.
- **G19 / elspeth-559bce3459** — No runbook for scheduler /
  multi-source resume (cross-cuts ADR-026).
- **G20 / elspeth-c2aa936ad8** — `docs/contracts/system-operations.md`
  Coalesce assumes single-source `row_id`.
- **G21 / elspeth-8c4ca2d89c** — RC1–RC5 progress note missing
  multi-source delivery.
- **G22 / elspeth-7f3ac1ac65** — *"Do not fabricate source_row_index"*
  lives only in an exception string.

### Tests anchored by this ADR

- **G25a / elspeth-71dcedcb66** — Zero crash-and-resume coverage for
  multi-source (P1).
- **G25b / elspeth-6116873e3b** — Source isolation under concurrent
  multi-source execution untested (P1).
- **G25g / elspeth-e51eaed773** — No invariant test that
  `rows.source_node_id` is durably attributed.

## Open questions / future work

- **Concurrent multi-source ingestion.** This ADR does not commit to
  parallel iteration of N sources. ADR-001 (plugin-level concurrency)
  records the orchestrator's single-threaded determinism property and
  is not amended here. G12 / elspeth-bc81207798 (multi-source
  pipelines run sequentially) is RC6 follow-up work and would
  warrant its own ADR if it changes the concurrency contract.
- **Cross-source row identity coalescing.** When two sources happen
  to emit `row_id = "12345"`, the audit trail today treats them as
  distinct because `(source_node_id, row_id)` is the durable
  identity. An explicit "compound row identity" doc is downstream
  work (covered by G22 / elspeth-7f3ac1ac65's lint and doc-rule
  remediation).
- **Composer ingestion of dynamic sources.** The composer's
  "dynamic-source-from-chat" workflow
  (memory `project_composer_dynamic_source_from_chat`) creates one
  source from chat text. Whether the composer ever authors *multiple*
  dynamic sources in one pipeline is a UX question, not an engine
  question; deferred.

## Related Decisions

- **ADR-001** (plugin-level concurrency) — preserved; the orchestrator
  remains single-threaded within a run. The scheduler primitive
  recorded in ADR-026 is designed to support future multi-worker
  ingestion without revisiting ADR-001's determinism property.
- **ADR-010** (declaration-trust framework) — preserved; the
  per-source schema contract continues to be a declaration-trust
  artifact, and the runtime VAL manifest drift check at
  `orchestrator/core.py:3482-3491` already gates resume on
  registry-identity match. The plural ResumeState shape doesn't
  weaken the check.
- **ADR-019** (two-axis terminal model) — preserved; per-source
  quarantine edges attribute failure to the originating source, but
  the terminal-state axis is unchanged.
- **ADR-021** (sources and sinks uniformly boundary) — preserved and
  reinforced. Every named source remains a Tier-3 boundary; the
  classification predicate (`kind in ("source", "sink")` in
  `web/audit_readiness/service._build_plugin_trust_row`) is
  independent of source count.
- **ADR-024** (delivery governance for single-maintainer mode) —
  preserved; this ADR is itself the governance artifact ADR-024
  contemplates for a structural change of this size.
- **ADR-026** (durable token scheduler) — companion. ADR-025 records
  *what* the source surface looks like; ADR-026 records *how* tokens
  produced by that surface survive crash and resume.

## References

### Engine and contract sites

- `src/elspeth/core/config.py:1340-1479` — `ElspethSettings`,
  `sources` field, `source` transition shim, `normalize_legacy_source`,
  `populate_legacy_source_view`.
- `src/elspeth/core/dag/builder.py:116-260` — `build_execution_graph`,
  `legacy_single_source_invocation` branch, source-name identity
  rule.
- `src/elspeth/core/dag/graph.py:283-353` — DAG validation including
  the QUEUE fan-in requirement.
- `src/elspeth/engine/orchestrator/types.py:518-540` — `ResumeState`
  dataclass, dual singular/plural schema fields.
- `src/elspeth/engine/orchestrator/core.py:3427-3555` —
  `_reconstruct_resume_state`, the multi-source branch and the
  `next(iter(...))` arbitrary pick.
- `src/elspeth/engine/orchestrator/core.py:2480-2515` —
  `_record_schema_contract` reading `config.source.get_schema_contract()`.
- `src/elspeth/contracts/scheduler.py` — `TokenWorkItem`,
  `TokenWorkStatus` (companion ADR-026).
- `src/elspeth/core/landscape/scheduler_repository.py` — durable
  scheduler primitive (companion ADR-026).

### Review notes

- `notes/branch-review-multi-source-token-scheduler-architecture-2026-05-22.md`
  (in the main checkout; orientation note).
- `notes/branch-review-multi-source-token-scheduler-consolidation-2026-05-22.md`
  (in the main checkout; tier consolidation, the *"upgrade around the
  scheduler is unfinished"* framing).
- `.worktrees/multi-source-token-scheduler/notes/multi-source-audit-dedup-map.md`
  (execution-detail dedup map).
- `notes/RC6-large-list.md` (in the main checkout; canonical RC6
  ticket enumeration).

### Project policies

- `CLAUDE.md` — *No Legacy Code Policy*, *Three-Tier Trust Model* (the
  Tier-1 audit-integrity rule that the arbitrary `schema_contract`
  pick violated).
- Project memory `project_db_migration_policy` — *delete the old DB*
  rather than migrate; rationale for the *negative consequences:
  existing audit databases unreadable* line above.
- Project memory `project_multi_source_token_scheduler_rc6` — this
  branch targets RC6, not RC5.2; rationale for the Tier-3 doc tickets
  being publish-gate rather than merge-gate.

## Notes

This ADR is the governance record for the structural fix. The
implementation work — deletion of the legacy facade, migration of the
26+ engine call sites, ResumeState field removal, composer skill
rewrite, doc corpus update, tests for crash-and-resume — is downstream
and tracked by the tickets enumerated above. The ADR commits to the
shape; the tickets execute against it.
