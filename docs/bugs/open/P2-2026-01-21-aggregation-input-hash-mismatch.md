# Bug Report: Aggregation flush uses inconsistent input hashes (node_state vs result)

## Summary

- `AggregationExecutor.execute_flush()` computes `input_hash` from `buffered_rows`, but `begin_node_state()` computes the stored input hash from a wrapped dict (`{"batch_rows": buffered_rows}`).
- The result's `input_hash` therefore does not match the node_state input hash, breaking audit consistency and traceability for aggregation inputs.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (main)
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/engine/executors.py` for bugs and create reports
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of `src/elspeth/engine/executors.py` and recorder hashing logic

## Steps To Reproduce

1. Run an aggregation flush and inspect the recorded `node_states.input_hash` in Landscape.
2. Compare it to `TransformResult.input_hash` returned by `execute_flush()` for the same flush.

## Expected Behavior

- The node_state input hash and the result's input_hash match, since they describe the same input payload.

## Actual Behavior

- The node_state input hash is computed from `{"batch_rows": buffered_rows}` while `result.input_hash` is computed from `buffered_rows`, so the hashes differ.

## Evidence

- `input_hash` computed from list of rows:
  - `src/elspeth/engine/executors.py:893`
  - `src/elspeth/engine/executors.py:964`
- Node state input hash computed from wrapped dict (via `begin_node_state` hashing input_data):
  - `src/elspeth/engine/executors.py:907`
  - `src/elspeth/engine/executors.py:909`
  - `src/elspeth/core/landscape/recorder.py:1019`

## Impact

- User-facing impact: audit UI and exports show inconsistent input hashes for the same aggregation flush.
- Data integrity / security impact: breaks hash-based verification and traceability guarantees for aggregation inputs.
- Performance or cost impact: none direct.

## Root Cause Hypothesis

- Aggregation flush uses two different representations of the input when hashing.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`: either compute `input_hash` from the same `batch_input` dict passed to `begin_node_state`, or pass `buffered_rows` directly as `input_data` so both hashes align.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test asserting `result.input_hash == node_state.input_hash` for aggregation flushes.
- Risks or migration steps: none.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): auditability standard in `CLAUDE.md` (transform boundaries should be traceable by hash).
- Observed divergence: mismatch between state and result hashes.
- Reason (if known): input wrapper for node_state differs from result hash input.
- Alignment plan or decision needed: standardize hash input representation for aggregation flushes.

## Acceptance Criteria

- Aggregation flush results and node_states share identical input_hash values.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor.py -k aggregation_input_hash`
- New tests required: yes (input hash consistency).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
