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

A prior 2026-04-21 batch-LLM invariant follow-on design described a *fix* path
(truthful sentinel population for error rows, no-network probe seams). That
historical design is retained in git history, not active docs. The fix did not
land and would not address the deeper conops critique below.

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
- The two `except BatchPendingError:` handlers in `engine/executors/aggregation.py` and the matching ones in `engine/orchestrator/core.py`

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
- **Multiple engine-side `except BatchPendingError:` handlers go.** The aggregation executor's batch-dispatch path becomes simpler — every batch transform now returns either success or a real exception, no control-flow signals.
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
- **Supersedes:** the historical 2026-04-21 batch-LLM invariant follow-on design
  — that spec proposed the fix path; this ADR retires the targets instead.

## References

- Filigree ticket: `elspeth-2799f6ec22` (P1 task — Retire batch-LLM transforms)
- Historical implementation plan overview and per-phase plans are retained in
  git history, not active docs.
- Dogfood run that surfaced the violation: 2026-05-06 session, observed in `examples/openrouter_sentiment/settings_batched.yaml` and `examples/template_lookups/settings_batched.yaml`
- Migration epic context (scope reduction): `elspeth-be398f0bcb`

## Notes

The retirement is intentionally aggressive on the scope of removal — the principle "no speculative primitives" is the load-bearing constraint. Reviewers should treat any temptation to retain `BatchPendingError` "for safety" as the symptom of optionality bias and reject it.
