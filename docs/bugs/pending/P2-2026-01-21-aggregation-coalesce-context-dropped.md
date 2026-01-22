# Bug Report: Aggregation requeue drops coalesce metadata for forked branches

## Summary

- When a forked branch passes through a passthrough/transform aggregation and the batch flush requeues tokens for further processing, the new work items omit `coalesce_at_step` and `coalesce_name`. Forked tokens then bypass coalesce entirely.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Configure a gate that forks to two branches with a coalesce point configured for those branches.
2. Place a batch-aware transform with `output_mode: passthrough` or `transform` in each branch, with additional transforms after the aggregation.
3. Trigger a batch flush and observe whether tokens coalesce.

## Expected Behavior

- Forked branch tokens should carry coalesce metadata through the aggregation flush and merge at the coalesce point.

## Actual Behavior

- Tokens requeued after aggregation lack coalesce metadata, so the coalesce check never runs and tokens bypass the join.

## Evidence

- Passthrough requeue omits coalesce info in work items: `src/elspeth/engine/processor.py:271-285`.
- Transform-mode requeue omits coalesce info in work items: `src/elspeth/engine/processor.py:352-359`.
- Coalesce check requires `coalesce_at_step`/`coalesce_name`: `src/elspeth/engine/processor.py:955-964`.

## Impact

- User-facing impact: Fork/join pipelines silently skip coalesce, producing unmerged outputs.
- Data integrity / security impact: Audit lineage for parallel branches is broken.
- Performance or cost impact: Downstream duplication and inflated outputs.

## Root Cause Hypothesis

- `_process_batch_aggregation_node()` builds new `_WorkItem` objects without propagating coalesce metadata.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add a fork→aggregation→coalesce integration test ensuring coalesce executes.
- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/subsystems/00-overview.md` (fork/coalesce flow).
- Observed divergence: Coalesce metadata is dropped mid-pipeline.
- Reason (if known): Work items do not carry coalesce fields.
- Alignment plan or decision needed: Propagate coalesce metadata when requeuing tokens.

## Acceptance Criteria

- Aggregated branch tokens still coalesce when coalesce is configured.
- Tests confirm coalesce outcomes and join lineage for aggregated branches.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k coalesce`
- New tests required: Yes (aggregation + coalesce integration).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/subsystems/00-overview.md`
