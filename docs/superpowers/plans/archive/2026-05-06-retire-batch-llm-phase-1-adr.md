# Phase 1 — ADR-020: Retire Batch-LLM Transforms

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author ADR-020 documenting the retirement decision, mark `2026-04-21-batch-llm-invariant-follow-on-design.md` as superseded, and link the ADR from the main filigree retirement issue (`elspeth-2799f6ec22`).

**Architecture:** Pure documentation phase. Produces no code change. The ADR sets the governance frame for Phases 2–5 and gives reviewers a single canonical reference.

**Tech Stack:** Markdown only. ADR template at `docs/architecture/adr/000-template.md`. ADR-020 is the next free number after ADR-019.

---

## Task 1: Author ADR-020

**Files:**
- Create: `docs/architecture/adr/020-retire-batch-llm-transforms.md`

- [ ] **Step 1: Read the ADR template and a recent ADR for tone**

```bash
cat docs/architecture/adr/000-template.md
cat docs/architecture/adr/019-two-axis-terminal-model.md | head -60
```

Expected: Template has sections Context / Decision / Consequences / Alternatives / Related Decisions / References. ADR-019 shows current tone (formal, evidence-cited, with ticket references).

- [ ] **Step 2: Create ADR-020 with the full content below**

Write `docs/architecture/adr/020-retire-batch-llm-transforms.md` with this body:

````markdown
# ADR-020: Retire Batch-LLM Transforms (`azure_batch_llm`, `openrouter_batch_llm`)

**Date:** 2026-05-06
**Status:** Accepted
**Deciders:** ELSPETH maintainers
**Tags:** plugins, llm, batch, conops, supersedes-design-doc-2026-04-21

## Context

Two LLM batch transforms have shipped in `src/elspeth/plugins/transforms/llm/`:

- `azure_batch_llm` (1,662 LOC) — uses Azure OpenAI's true async batch API. Submits a job, persists `batch_id`, returns `BatchPendingError` to the orchestrator, and resumes hours later via a checkpoint stored in `PluginAuditContext._batch_checkpoints`.
- `openrouter_batch_llm` (909 LOC) — issues a buffered set of HTTP calls in parallel via `ThreadPoolExecutor`. Synchronous from the engine's POV but bypasses the per-row dispatch contract because the engine receives a list of rows, not a single row.

A 2026-05-06 dogfood-run (filigree `elspeth-2799f6ec22`) confirmed both transforms hard-fail at config-instantiation in the current `RC5-UX` codebase. Two `*_batched.yaml` examples raise:

```
FrameworkBugError: Transform 'openrouter_batch_llm' declares declared_input_fields [...]
but is batch-aware. No batch-pre-execution dispatch site exists; ADR-013 scopes
DeclaredRequiredFieldsContract to non-batch transforms until an ADR-010 amendment
lands.
```

`docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md` describes a *fix* path (truthful sentinel population for error rows, no-network probe seams). That fix has not landed and would not address the deeper conops critique below.

### The conops critique

ELSPETH's audit-trail discipline (CLAUDE.md "Auditability Standard") requires per-row attribution within a single run: every output traceable to source data, configuration, and code version, with `explain(recorder, run_id, token_id)` proving complete lineage. Two structural problems make the batch-LLM transforms incompatible:

1. **`azure_batch_llm` bifurcates row lifetime across runs.** A row enters the pipeline in run A, gets buffered into a batch submission, the run ends with `BatchPendingError`, and (hours later) run B resumes with the same `batch_id`. Audit attribution must span those two runs to be meaningful — a property no other ELSPETH transform requires. The `BatchCheckpointState` / `BatchPendingError` infrastructure exists *solely* to support this lifecycle.
2. **`openrouter_batch_llm` bypasses the per-row dispatch contract.** ADR-013 (`DeclaredRequiredFieldsContract`) is enforced at the per-row pre-execution dispatch site, which a batch-aware transform never visits. The "batch-pre-execution dispatch site" referenced in the error message has never been built and would require an ADR-010 amendment to design.

### The "no users" datum

A 2026-05-06 inventory across `examples/`, internal projects, and external integrations confirmed the only references to either plugin are inside the broken `examples/*_batched.yaml` files. There are no production deployments depending on these plugins.

## Decision

Retire both `azure_batch_llm` and `openrouter_batch_llm` in a single coherent change. Per ELSPETH's "no legacy code" policy, no shims, no deprecation period, no compatibility flags. Removed APIs are removed.

The retirement also removes async-batch primitives that exist solely to serve `azure_batch_llm`'s submit-and-poll lifecycle:

- `BatchPendingError` (in `src/elspeth/contracts/errors.py`)
- `BatchCheckpointState`, `RowMappingEntry` (in `src/elspeth/contracts/batch_checkpoint.py` — file deleted)
- `_checkpoint`, `_batch_checkpoints` fields and `get_checkpoint`/`set_checkpoint`/`clear_checkpoint` methods on `PluginAuditContext`
- `get_checkpoint`/`set_checkpoint` on `PluginContextProtocol`
- `batch_checkpoints` parameter on `PipelineOrchestrator._execute_run`, `run`, and `_initialize_run_context`
- The two `except BatchPendingError:` handlers in `engine/executors/aggregation.py` and the matching one in `engine/orchestrator/core.py`

The retirement does **not** remove:

- `BatchTransformProtocol` (in `contracts/plugin_protocols.py`) — kept for `batch_replicate` and `batch_stats`
- `is_batch_aware` flag on `TransformProtocol` — same
- `engine/batch_adapter.py` (BatchTransformMixin and dispatch infrastructure) — same
- `batch_replicate` (1-to-N row expansion plugin)
- `batch_stats` (group-by statistics plugin)
- The aggregation executor's batch-aware dispatch path (minus the `BatchPendingError` handlers)

These retain a clear conops mapping: `batch_replicate` and `batch_stats` are synchronous, single-run, per-row attributable, and serve statistical-aggregation use cases that ELSPETH's audit framework supports natively.

### What we are NOT doing

- **Not** keeping `BatchPendingError` "in case some future async non-LLM transform needs it." Per the principle that load-bearing infrastructure must be justified by current consumers, speculative primitives are removed and reintroduced only when an actual use case lands.
- **Not** bumping the audit-DB schema epoch. Batch checkpoint state was always in-process (`PluginAuditContext._batch_checkpoints`), never persisted to the Landscape audit DB. Verified by grep on `src/elspeth/core/landscape/` (zero hits for `BatchCheckpoint`/`batch_checkpoint`).
- **Not** writing a migration helper for old configs. Configs that reference `azure_batch_llm` or `openrouter_batch_llm` will fail at value-source enforcement with a clear error pointing at the missing transform name. This is consistent with how stale model identifiers surface today (catalog enforcement at config-instantiation time).

## Consequences

### Positive Consequences

- **`PluginAuditContext` shrinks meaningfully.** ~80 lines of checkpoint plumbing removed. The `PluginContextProtocol` interface becomes thinner.
- **The `contracts` package loses three exports** (`BatchCheckpointState`, `RowMappingEntry`, `BatchPendingError`) and one file (`batch_checkpoint.py`, 193 lines).
- **The orchestrator simplifies.** `batch_checkpoints` parameter removed from three method signatures. Resume flow no longer threads typed checkpoint state through the call graph.
- **Two engine-side `except BatchPendingError:` handlers go.** The aggregation executor's batch-dispatch path becomes simpler — every batch transform now returns either success or a real exception, no control-flow signals.
- **The `elspeth-be398f0bcb` migration epic shrinks.** Its skip-set explicitly named both retired transforms; retirement removes them from the migration scope without further effort.
- **The conops critique is structurally resolved.** No transform in the codebase will straddle two runs after this change.
- **Composer/MCP catalog shrinks** by two transforms. Agents using the catalog get the new shape on next refresh.

### Negative Consequences

- **Loss of Azure Batch API support.** ELSPETH no longer offers the documented "50% cost savings" path for high-volume Azure OpenAI work. Operators wanting that economic profile must run regular `llm` transforms with cost-aware throughput tuning.
- **Loss of OpenRouter parallel-batch path.** High-throughput OpenRouter use cases must rely on the existing pooled (`pool_size: N`) executor on the regular `llm` transform.
- **`BatchPendingError` removal is a public-API break for `elspeth.contracts`.** Any external code importing it will fail. Per "no users" inventory, this is theoretical.

### Neutral Consequences

- **`BatchTransformProtocol` stays.** Anyone who has built a custom batch-aware transform on top of it remains supported; only the LLM specializations go.
- **Resume CLI behaviour simplifies.** `elspeth resume <run_id>` no longer carries `batch_checkpoints` through the orchestrator; only row-level checkpointing remains. Behavioural simplification, not a feature loss for non-batch-LLM users (i.e., everyone).

## Alternatives Considered

### Alternative 1: Fix the contract violation (the design doc's path)

**Description:** Implement the truthful-sentinel approach from `2026-04-21-batch-llm-invariant-follow-on-design.md`. Populate `*_usage` and `*_model` on every error-bearing row with `None` sentinels, characterise via unit tests, add no-network probe seams.

**Rejected because:** Addresses only the ADR-013 contract violation surface, not the conops mismatch. `azure_batch_llm`'s submit-and-poll lifecycle would still bifurcate row lifetime across runs. The `BatchPendingError` / `BatchCheckpointState` plumbing would remain in the engine. Net cost (engineering + ongoing maintenance burden) exceeds net value given the "no users" datum.

### Alternative 2: Build the ADR-010 amendment (batch-pre-execution dispatch site)

**Description:** Extend ADR-010's declaration framework to add a batch-pre-execution dispatch site so `DeclaredRequiredFieldsContract` and friends apply to batch-aware transforms cleanly.

**Rejected because:** Significant ADR-level design work to support transforms that the operator-facing surface (composer, examples, docs) lists prominently but no one uses. Resource allocation to design contract semantics for unused transforms is the wrong priority.

### Alternative 3: Keep `BatchPendingError` and friends as primitives for hypothetical future async transforms

**Description:** Delete the LLM plugins but keep the `BatchPendingError` / `BatchCheckpointState` infrastructure for "when we need it later."

**Rejected because:** Speculative infrastructure rots faster than code with consumers. The handler call sites cannot be exercised without a consumer, so they would either silently break or accumulate test scaffolding for shapes nothing uses. When a real async-transform use case lands, designing the primitives against actual requirements is straightforwardly preferable to evolving a guess made now.

## Related Decisions

- **ADR-010** (Declaration Trust Framework): defines the per-row dispatch site that batch-aware transforms bypass.
- **ADR-013** (Declared Required Fields Contract): explicitly scopes itself to non-batch transforms, citing the missing batch-pre-execution dispatch site. The contract violation surfaced in the dogfood run originates here.
- **ADR-019** (Two-Axis Terminal Model): batch-LLM transforms had no novel interaction with the terminal-model migration; retirement does not affect the ADR-019 work.
- **Supersedes:** `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md` — that spec proposed the fix path; this ADR retires the targets instead.

## References

- Filigree ticket: `elspeth-2799f6ec22` (P1 task — Retire batch-LLM transforms)
- Implementation plan overview: `docs/superpowers/plans/2026-05-06-retire-batch-llm-overview.md`
- Per-phase plans: `docs/superpowers/plans/2026-05-06-retire-batch-llm-phase-{1..5}-*.md`
- Dogfood run that surfaced the violation: 2026-05-06 session, observed in `examples/openrouter_sentiment/settings_batched.yaml` and `examples/template_lookups/settings_batched.yaml`
- Migration epic context (scope reduction): `elspeth-be398f0bcb`

## Notes

The retirement is intentionally aggressive on the scope of removal — the principle "no speculative primitives" is the load-bearing constraint. Reviewers should treat any temptation to retain `BatchPendingError` "for safety" as the symptom of optionality bias and reject it.
````

- [ ] **Step 3: Verify the file is well-formed Markdown**

Run: `markdown-link-check docs/architecture/adr/020-retire-batch-llm-transforms.md` (if available) or simply `head -200 docs/architecture/adr/020-retire-batch-llm-transforms.md`

Expected: file reads top-to-bottom, internal markdown structure intact, no truncated headers.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture/adr/020-retire-batch-llm-transforms.md
git commit -m "docs(adr): adr-020 retire batch-LLM transforms"
```

---

## Task 2: Mark superseded design doc

**Files:**
- Modify: `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md` (top of file)

- [ ] **Step 1: Read the current top of the file**

Run: `head -10 docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md`

Expected output (roughly):
```
# Batch LLM Invariant Follow-on Design — Azure Batch and OpenRouter Batch

**Status:** Spike
**Date:** 2026-04-21
...
```

- [ ] **Step 2: Replace the status line and add a superseded-by banner**

Use the Edit tool. Change:

```
**Status:** Spike
```

to:

```
**Status:** Superseded by ADR-020 (`docs/architecture/adr/020-retire-batch-llm-transforms.md`) — 2026-05-06. The fix path described below was not pursued; the targets were retired instead.
```

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md
git commit -m "docs(specs): mark batch-LLM invariant follow-on superseded by ADR-020"
```

---

## Task 3: Sanity check — no other ADR or spec references the targets as future work

- [ ] **Step 1: Grep for stale "future work" references**

Run:
```bash
grep -rn "azure_batch_llm\|openrouter_batch_llm" docs/architecture/adr/ docs/superpowers/specs/ docs/superpowers/plans/ 2>&1 | grep -v "020-retire-batch-llm\|2026-04-21-batch-llm-invariant\|2026-05-06-retire-batch-llm"
```

Expected: at most a couple of incidental archive references. If any current spec/plan describes the retired transforms as live work, flag — those need updating in Phase 4.

- [ ] **Step 2: If new findings, append them to Phase 4's documentation sweep checklist**

Edit `docs/superpowers/plans/2026-05-06-retire-batch-llm-phase-4-ci-and-docs.md` and add the file paths to its Task 5 (or wherever the documentation list lives).

- [ ] **Step 3: Commit (if Step 2 produced any change)**

```bash
git add docs/superpowers/plans/2026-05-06-retire-batch-llm-phase-4-ci-and-docs.md
git commit -m "docs(plans): expand phase-4 doc sweep with stale spec/plan references"
```

(Skip if Step 1 found nothing to add.)

---

## Phase 1 Exit Criteria

- [ ] `docs/architecture/adr/020-retire-batch-llm-transforms.md` exists and reads well
- [ ] `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md` shows superseded-by banner at top
- [ ] No active spec/plan describes either retired transform as live work
- [ ] Git log shows 1–2 commits scoped to `docs/`

Phase 2 picks up the surface-deletion sweep.
