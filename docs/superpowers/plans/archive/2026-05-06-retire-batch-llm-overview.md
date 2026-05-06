# Retire Batch-LLM Transforms — Overview Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire `azure_batch_llm` and `openrouter_batch_llm` transforms entirely from ELSPETH, plus the async-batch primitives (`BatchPendingError`, `BatchCheckpointState`, `RowMappingEntry`, `get_checkpoint`/`set_checkpoint` plumbing) that exist solely to serve their submit-and-poll lifecycle. Keep `batch_replicate`, `batch_stats`, the `BatchTransformProtocol`, and the rest of the batch-aware infrastructure that statistical aggregation depends on.

**Architecture:** Layered demolition. The two plugin files come out cleanly (~2,571 LOC). After they are gone, `BatchCheckpointState` and `BatchPendingError` have zero non-test consumers, so they and their plumbing in `PluginAuditContext`, the orchestrator, and the executors can be removed in the same change. The `BatchTransformProtocol` and `is_batch_aware` flag stay — they are load-bearing for `batch_replicate`/`batch_stats`. No audit-DB schema change is needed (batch checkpoint state was always in-process, never persisted). No legacy shims, no deprecation period — per ELSPETH's "no legacy code" policy, removed APIs are removed in a single coherent change.

**Tech Stack:** Python 3.13, ruff, mypy, pytest, pluggy, structlog, tenacity. Tier-model enforcement via `scripts/cicd/enforce_tier_model.py`. Filigree for issue tracking. Filigree issue: `elspeth-2799f6ec22`.

---

## Phase Map

This work is split into five phases, each producing a buildable, committable state with green tests at completion. Each phase has its own plan file.

| Phase | Plan file | Purpose | Estimated commits |
|---|---|---|---|
| **1 — ADR** | `2026-05-06-retire-batch-llm-phase-1-adr.md` | Author ADR-020 (decision record), mark `2026-04-21-batch-llm-invariant-follow-on-design.md` superseded | 1 |
| **2 — Surface deletion** | `2026-05-06-retire-batch-llm-phase-2-surface.md` | Delete plugin source files, plugin-specific tests, example settings, composer skill entries, plugin-discovery test counts | 4–5 |
| **3 — Engine demolition** | `2026-05-06-retire-batch-llm-phase-3-engine.md` | Remove `BatchPendingError`/`BatchCheckpointState`/`RowMappingEntry`, `get_checkpoint`/`set_checkpoint` plumbing, `batch_checkpoints` parameter, and all engine handlers | 5–6 |
| **4 — CI + docs sweep** | `2026-05-06-retire-batch-llm-phase-4-ci-and-docs.md` | Trim CI allowlists, sweep documentation, mark superseded specs, update reference docs | 3–4 |
| **5 — Verification + close-out** | `2026-05-06-retire-batch-llm-phase-5-verification.md` | Full test sweep, tier-model check, dogfood smoke run, filigree close-out, comment on `elspeth-be398f0bcb` skip-set | 1 |

**Total estimate:** 14–17 commits, ~1 day of focused work, ~5,000 LOC delta (deletion-heavy).

---

## Decision Record (rationale summary)

1. **Conops misalignment.** ELSPETH's audit-trail discipline requires per-row attribution within a single run. `azure_batch_llm` uses Azure's true async batch API (submit, persist `batch_id`, return `BatchPendingError`, resume hours later via checkpoint) — bifurcating row lifetime across runs. `openrouter_batch_llm` is concurrent-not-async, but still bypasses the per-row dispatch contract.

2. **ADR-013 violation (the trigger).** Both transforms emit `declared_input_fields` while declaring `is_batch_aware=True`, which raises `FrameworkBugError`. ADR-013 explicitly scopes `DeclaredRequiredFieldsContract` to non-batch transforms; the batch-pre-execution dispatch site has never been built.

3. **No users.** Confirmed only references are in `examples/`. No production deployments depend on these plugins.

4. **No speculative primitives.** Per project principle, infrastructure should be justified by current consumers, not hypothetical future ones. After azure_batch removal, `BatchPendingError` + `BatchCheckpointState` have zero consumers — they go.

5. **No audit-DB epoch bump.** Batch checkpoint state was always in-process (`PluginAuditContext._batch_checkpoints`), never persisted to the Landscape audit DB. Removal does not change the audit schema.

---

## What Stays vs What Goes

### Stays (load-bearing for statistical aggregation)

- `src/elspeth/plugins/transforms/batch_replicate.py` — 1-to-N row expansion (used by `examples/deaggregation`)
- `src/elspeth/plugins/transforms/batch_stats.py` — group-by statistics (used by `examples/batch_aggregation`)
- `src/elspeth/contracts/plugin_protocols.py:BatchTransformProtocol`
- `src/elspeth/contracts/plugin_protocols.py:TransformProtocol.is_batch_aware`
- `src/elspeth/engine/batch_adapter.py` (BatchTransformMixin / dispatch infrastructure)
- `src/elspeth/engine/executors/aggregation.py` batch-aware dispatch (minus the two `except BatchPendingError` blocks)

### Goes

| File | Lines (approx) | Purpose |
|---|---:|---|
| `src/elspeth/plugins/transforms/llm/openrouter_batch.py` | 909 | OpenRouterBatchLLMTransform plugin |
| `src/elspeth/plugins/transforms/llm/azure_batch.py` | 1,662 | AzureBatchLLMTransform plugin |
| `src/elspeth/contracts/batch_checkpoint.py` | 193 | `BatchCheckpointState`, `RowMappingEntry` |
| `examples/openrouter_sentiment/settings_batched.yaml` | — | Broken example |
| `examples/template_lookups/settings_batched.yaml` | — | Broken example |
| Plugin-specific tests | several files | See Phase 2 |
| `BatchPendingError` class | ~40 | In `contracts/errors.py:645` |
| `_checkpoint`/`_batch_checkpoints` fields + `get_checkpoint`/`set_checkpoint` methods | ~80 | In `contracts/plugin_context.py:130–219` |
| `get_checkpoint`/`set_checkpoint` on `PluginContextProtocol` | 3 | In `contracts/contexts.py:127–129` |
| `batch_checkpoints` parameter | several call sites | Orchestrator core, executors, tests |

### Filigree housekeeping

- Main issue: `elspeth-2799f6ec22` (P1 task) — closes when verification passes
- `elspeth-be398f0bcb` epic — comment on scope reduction (its skip-set explicitly names both retired transforms)
- Children of `elspeth-be398f0bcb` (six tickets) — review for redundancy with retirement; close-as-superseded any that are batch-LLM-scoped

---

## Execution Conventions

- Commit messages follow `<type>(<scope>): <subject>` (matches existing repo style):
  - `chore(batch-llm): delete openrouter_batch.py + azure_batch.py`
  - `refactor(contracts): remove BatchPendingError + BatchCheckpointState`
  - `docs(adr): adr-020 retire batch-LLM transforms`
  - `test(llm): drop batch-LLM test files`
- Run after every commit: `pytest tests/unit -x -q 2>&1 | tail -20` to confirm nothing collateral broke.
- After Phase 3 (the riskiest), also run `mypy src/` and `ruff check src/` before committing — engine demolition will surface dangling type references.
- Tier-model enforcement: `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` — must pass at end of each phase.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Some test imports `BatchCheckpointState` for shape coverage of *kept* code | Phase 4 step explicitly greps every consumer; trim tests don't rely on the type — they rely on the *protocol*, which stays |
| Documentation drift — old plans still reference batch-LLM as future work | Phase 4 sweeps; CHANGELOG-RC1/RC2 left intact (history) |
| Composer/MCP `list_transforms` cache on agents using the catalog | Schema bump on next composer reload — agents pick up new catalog automatically; old configs referencing the names will fail at value-source enforcement (consistent with how stale model identifiers surfaced this week) |
| `elspeth-be398f0bcb` epic children blocked on the retirement | Phase 5 closes the relevant ones; `codex` (the assignee) gets a comment |
| The `RC5-UX` branch already has 25 unpushed commits; risk of merge conflict on `main` | Land each phase as its own commit so individual reverts work; do not push until verified |

---

## Phase 1 — ADR

See: `2026-05-06-retire-batch-llm-phase-1-adr.md`

---

## Phase 2 — Surface deletion

See: `2026-05-06-retire-batch-llm-phase-2-surface.md`

---

## Phase 3 — Engine demolition

See: `2026-05-06-retire-batch-llm-phase-3-engine.md`

---

## Phase 4 — CI + docs sweep

See: `2026-05-06-retire-batch-llm-phase-4-ci-and-docs.md`

---

## Phase 5 — Verification + close-out

See: `2026-05-06-retire-batch-llm-phase-5-verification.md`
