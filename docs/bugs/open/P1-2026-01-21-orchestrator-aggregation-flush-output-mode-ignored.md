# Bug Report: End-of-source aggregation flush ignores output_mode semantics

## Summary

- `_flush_remaining_aggregation_buffers()` branches on `TransformResult` shape (row vs rows) and always uses `buffered_tokens[0]` or `expand_token`, ignoring the aggregation `output_mode`.
- In `passthrough` mode, this creates new tokens tied to the first buffered row_id and leaves the original buffered tokens in a non-terminal state, breaking lineage and audit attribution.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: aggregation with output_mode=passthrough that flushes at end-of-source

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/engine/orchestrator.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation with `output_mode=passthrough` and a trigger that will not fire before end-of-source (e.g., count=100 with 2 rows).
2. Run a pipeline and let end-of-source flush handle the remaining buffered rows.
3. Inspect `tokens` and `token_outcomes` for the buffered rows and the emitted outputs.

## Expected Behavior

- End-of-source flush should respect `output_mode`:
  - `passthrough`: reuse each buffered token_id/row_id, update row_data, and emit those tokens.
  - `transform`: always create new tokens via `expand_token` (even for single-row output) while buffered tokens remain CONSUMED_IN_BATCH.

## Actual Behavior

- End-of-source flush uses `buffered_tokens[0]` for single-row output and `expand_token` for multi-row output regardless of `output_mode`.
- In passthrough mode, all outputs inherit the first buffered row_id and new token_ids, while original buffered tokens never receive a terminal outcome.

## Evidence

- Output handling ignores `output_mode` and always uses `buffered_tokens[0]` or `expand_token` in `src/elspeth/engine/orchestrator.py:1582-1606`.
- `expand_token` assigns the parent row_id to all expanded tokens, so every output inherits the first buffered row_id in `src/elspeth/engine/tokens.py:229-245`.
- Aggregation `output_mode` is stored in the DAG config but is not consulted here (`src/elspeth/core/dag.py:309-324`).

## Impact

- User-facing impact: passthrough aggregations produce misattributed outputs with incorrect row lineage.
- Data integrity / security impact: audit trail breaks token identity and terminal state guarantees.
- Performance or cost impact: none.

## Root Cause Hypothesis

- End-of-source aggregation flush re-implements aggregation output handling without consulting `output_mode`, diverging from `RowProcessor._process_batch_aggregation_node()` semantics.

## Proposed Fix

- Code changes (modules/files):
  - In `src/elspeth/engine/orchestrator.py`, branch on `agg_settings.output_mode`.
  - For `passthrough`, zip `buffered_tokens` with `flush_result.rows` and reuse each token_id/row_id.
  - For `transform`, always use `expand_token` (even for single-row output) and ensure buffered tokens are terminal CONSUMED_IN_BATCH.
  - Consider refactoring to reuse `RowProcessor` aggregation handling to avoid drift.
- Config or schema changes: N/A
- Tests to add/update:
  - End-of-source flush for passthrough preserves token_id/row_id and records terminal outcomes.
  - End-of-source flush for transform (single-row output) creates new token_id via expand_token.
- Risks or migration steps:
  - Verify downstream gate routing and sink writes remain deterministic.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "Terminal Row States" and "Attributability Test".
- Observed divergence: buffered tokens never reach terminal outcomes; output tokens lose row identity.
- Reason (if known): output handling keyed to result shape instead of output_mode.
- Alignment plan or decision needed: align end-of-source flush semantics with RowProcessor aggregation semantics.

## Acceptance Criteria

- Passthrough end-of-source flush preserves token identity and row_id for each buffered token.
- Transform end-of-source flush always creates new tokens for output rows.
- Buffered tokens have consistent terminal outcomes after flush.

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py -k aggregation -v`
- New tests required: yes, end-of-source passthrough and transform flush coverage.

## Notes / Links

- Related issues/PRs: P1-2026-01-21-aggregation-input-hash-mismatch
- Related design docs: CLAUDE.md auditability standard
