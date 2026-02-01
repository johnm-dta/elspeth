# Bug Report: Coalesce config is accepted but ignored (and COALESCED outcomes would be dropped)

## Summary

- `ElspethSettings.coalesce` is validated and persisted into the resolved config, but **never influences the execution graph or runtime**. No coalesce nodes/edges are created, no `CoalesceExecutor` is wired into the engine, and `RowOutcome.COALESCED` is not handled by `Orchestrator`.
- Result: coalesce is effectively **non-functional** (silent config no-op today), and any future wiring that causes `RowProcessor` to emit `COALESCED` will likely cause **silent token drops** unless Orchestrator handling is added.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `370fc20862d6bab1bb77ebfe8c49527a12fa2aa8` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1 (repo targets >=3.11)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into project, identify a bug, do RCA, write bug report
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `config.py`, `dag.py`, `orchestrator.py`, `processor.py`, `coalesce_executor.py`

## Steps To Reproduce

1. Create a settings YAML (or `ElspethSettings`) with a non-empty `coalesce:` section.
2. Build the execution graph via `ExecutionGraph.from_config(settings)`.
3. Observe the graph: there are nodes for source/transforms/aggregations/config-gates/sinks, but **no coalesce nodes or edges**.

(Optional runtime repro once dependencies are installed)
1. Configure a gate that forks to branches `path_a` and `path_b`, and configure `coalesce` expecting those branches to merge.
2. Run a pipeline and inspect the Landscape:
   - no node with `node_type='coalesce'`
   - no join tokens (`join_group_id`) and no `COALESCED` terminal outcomes

## Expected Behavior

- A non-empty `coalesce` configuration either:
  - (A) is compiled into the DAG (coalesce node + edges) and executed at runtime, producing merged tokens, OR
  - (B) fails fast with a clear validation error stating coalesce is unsupported/unwired.
- No token outcomes are silently ignored (per “no silent drops” invariant).

## Actual Behavior

- `coalesce` is accepted at config time but has **no effect** on graph construction or runtime execution.
- `RowOutcome.COALESCED` is not handled by `Orchestrator` (so if/when coalesce is wired, merged results risk being silently dropped).

## Evidence

- Config layer supports coalesce and preserves it into resolved config:
  - `src/elspeth/core/config.py:654-660` (`ElspethSettings.coalesce`)
  - `tests/core/test_config.py:1539-1602` (expects coalesce to validate + be present in `resolve_config`)
- Graph compilation ignores coalesce:
  - `src/elspeth/core/dag.py:245-413` (`ExecutionGraph.from_config()` never references `config.coalesce`)
- Runtime does not wire coalesce:
  - `src/elspeth/engine/orchestrator.py:548-564` (`RowProcessor(...)` constructed without `coalesce_executor` / `coalesce_node_ids`)
  - `src/elspeth/engine/orchestrator.py:624-706` (result handling has no `RowOutcome.COALESCED` branch)
- RowProcessor has a coalesce path that can emit `COALESCED` when a coalesce executor is present:
  - `src/elspeth/engine/processor.py:571-606`
- Coalesce lifecycle cleanup exists but is unused:
  - `src/elspeth/engine/coalesce_executor.py:382-463` (`flush_pending`)
  - `rg -n "flush_pending\\(" src/elspeth` shows no call sites

## Impact

- User-facing impact: Coalesce configurations are silently ignored; users cannot build fork/join pipelines as designed.
- Data integrity / security impact: Audit trail lacks coalesce nodes/events and join lineage; violates “no silent drops / every token terminal” expectations if coalesce is partially enabled.
- Performance or cost impact: Downstream systems may process duplicated branch outputs instead of a merged result.

## Root Cause Hypothesis

- Coalesce was implemented as an isolated executor + config model, but the **DAG compiler and orchestrator wiring were not updated** to place coalesce in the execution plan and to drive/flush coalesce state.
- Additional integration gap: Orchestrator’s result handling still assumes a limited set of terminal outcomes and doesn’t fail fast for unknown outcomes (allowing silent drops).

## Proposed Fix

- Code changes (modules/files):
  - **Fail fast (minimal safety fix):**
    - Reject non-empty `settings.coalesce` in `ExecutionGraph.from_config()` or `ElspethSettings` validation until coalesce is fully supported.
    - Add an explicit hard error if `RowProcessor` emits an unhandled `RowOutcome` in `Orchestrator._execute_run()`.
  - **Full integration (feature fix):**
    - Define how coalesce is placed in the pipeline (likely as an explicit step, per `docs/contracts/plugin-protocol.md`), then:
      - Extend `ExecutionGraph.from_config()` to add `coalesce` nodes/edges.
      - Instantiate/register `CoalesceExecutor` in `Orchestrator`, pass it into `RowProcessor`, and plumb `coalesce_at_step`/`coalesce_name`.
      - Handle `RowOutcome.COALESCED` in Orchestrator and route merged token to the appropriate sink.
      - Ensure end-of-source triggers `flush_pending()` and records failures per policy.
- Config or schema changes:
  - If coalesce placement is intended to be in-order, update config schema (and docs) so coalesce is expressible as a pipeline stage, not just a top-level list.
- Tests to add/update:
  - Add a `tests/core/test_dag.py` test asserting coalesce nodes are present (or that config is rejected until implemented).
  - Add an orchestrator/processor integration test ensuring `COALESCED` is handled and no tokens are silently dropped.
- Risks or migration steps:
  - Introducing coalesce nodes changes DAG topology and step indexing; ensure step indices remain stable for audit queries and checkpoint semantics.

## Architectural Deviations

- Spec or doc reference:
  - `docs/contracts/plugin-protocol.md#Coalesce-(Token-Merging)` (coalesce as a pipeline step)
  - `docs/design/requirements.md` SOP-012..SOP-018 (marked ✅ but currently unwired)
- Observed divergence:
  - Coalesce config exists but is not compiled/executed; `COALESCED` outcome is not handled at orchestration level.
- Reason (if known):
  - Config refactor and incremental DAG work left coalesce integration incomplete.
- Alignment plan or decision needed:
  - Decide whether coalesce is a pipeline stage (recommended) or a top-level feature with explicit attachment points (gate/coalesce mapping).

## Acceptance Criteria

- A pipeline with configured coalesce produces:
  - a registered `coalesce` node in Landscape,
  - join token lineage (`token_parents` and `join_group_id`),
  - `COALESCED` terminal outcomes for consumed tokens and a merged token that reaches a sink.
- Orchestrator never silently ignores a `RowOutcome`; unknown outcomes hard-fail with a clear error.

## Tests

- Suggested tests to run:
  - `pytest tests/core/test_dag.py`
  - `pytest tests/engine/test_coalesce_executor.py`
  - `pytest tests/engine/test_orchestrator.py`
- New tests required: yes (DAG compilation and orchestrator handling)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`, `docs/design/architecture.md`, `docs/contracts/plugin-protocol.md`

## Resolution

- **Fixed in:**
  - `c088732` feat(dag): add coalesce node creation in from_config()
  - `21fed53` feat(orchestrator): wire CoalesceExecutor into RowProcessor
  - `de3e5ef` feat(orchestrator): handle COALESCED outcome in result loop
  - `15f379f` feat(orchestrator): call flush_pending at end of source
  - `eff5062` feat(processor): link fork children to coalesce points
  - `40e8fa0` feat(orchestrator): compute coalesce step positions
  - `7e085dd` feat(orchestrator): add coalesce_settings to PipelineConfig
  - `29f3c8e` test(integration): add fork/coalesce pipeline integration tests
- **Date:** 2026-01-20
- **Resolution:** Full coalesce integration implemented:
  - DAG compiler creates coalesce nodes (Task 1)
  - Orchestrator wires CoalesceExecutor (Task 2)
  - COALESCED outcome handled (Task 3)
  - flush_pending called at end-of-source (Task 4)
  - Fork children linked to coalesce points (Tasks 5-6)
  - coalesce_settings added to PipelineConfig (Task 7)
  - Integration tests verify end-to-end flow (Task 8)
