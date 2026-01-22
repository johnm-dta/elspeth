# Bug Report: Checkpoint aggregation_state_json bypasses canonical normalization

## Summary

- `CheckpointManager.create_checkpoint()` serializes `aggregation_state` with raw `json.dumps` (no canonical normalization, no NaN/Infinity rejection), while aggregation buffers store raw row dicts; this can raise `TypeError` for supported non-JSON primitives (datetime/Decimal/bytes/numpy/pandas) or persist invalid JSON (NaN/Infinity), breaking crash recovery and violating Tier‑1 audit integrity.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: Unknown
- Related run/issue ID: Unknown

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-2 @ 81a0925d7d6de0d0e16fdd2d535f63d096a7d052
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit on /home/john/elspeth-rapid/src/elspeth/core/checkpoint/manager.py
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): sandbox_mode=read-only, approval_policy=never, network restricted
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: reviewed `src/elspeth/core/checkpoint/manager.py`, `src/elspeth/core/canonical.py`, `src/elspeth/engine/executors.py`

## Steps To Reproduce

1. Prepare an aggregation buffer that includes a row with a non-JSON primitive (e.g., `datetime`, `Decimal`, `bytes`, numpy/pandas scalar) or a non-finite float (NaN/Infinity).
2. Call `CheckpointManager.create_checkpoint(..., aggregation_state=AggregationExecutor.get_checkpoint_state())`.
3. Observe `TypeError` from `json.dumps` or a stored `aggregation_state_json` containing `NaN`/`Infinity`.

## Expected Behavior

- Aggregation checkpoint state is normalized using the same canonical rules as audit data, rejects NaN/Infinity explicitly, and is always stored as valid JSON.

## Actual Behavior

- `json.dumps` is called directly and either throws on supported non-JSON primitives or serializes NaN/Infinity into invalid JSON.

## Evidence

- `src/elspeth/core/checkpoint/manager.py:57` uses raw `json.dumps(aggregation_state)` without normalization or `allow_nan=False`.
- `src/elspeth/engine/executors.py:1065` stores raw buffered rows in checkpoint state (`"rows": list(self._buffers[node_id])`).
- `src/elspeth/core/canonical.py:9` and `src/elspeth/core/canonical.py:48` enforce strict NaN/Infinity rejection and normalize common non-JSON types, but checkpoint serialization bypasses this.

## Impact

- User-facing impact: checkpoint creation can crash for valid pipeline data types; resume may be impossible.
- Data integrity / security impact: invalid JSON may be written into Tier‑1 audit storage.
- Performance or cost impact: failed runs and forced full reprocessing.

## Root Cause Hypothesis

- `CheckpointManager` serializes aggregation state without canonical normalization or strict JSON rules, allowing unsupported types and non-finite floats through.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/core/checkpoint/manager.py` should normalize `aggregation_state` via `core.canonical._normalize_for_canonical` or `canonical_json`, then `json.dumps(..., allow_nan=False)`; raise on normalization errors before insert.
- Config or schema changes: None.
- Tests to add/update: add unit tests covering checkpoint serialization of datetime/Decimal/bytes/numpy/pandas and explicit NaN/Infinity rejection.
- Risks or migration steps: ensure existing checkpoints with invalid JSON are handled or flagged before upgrade.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/core/canonical.py:9` (strict NaN/Infinity rejection)
- Observed divergence: checkpoint serialization does not enforce canonical normalization or NaN rejection.
- Reason (if known): Unknown
- Alignment plan or decision needed: normalize/validate checkpoint payloads using canonical rules.

## Acceptance Criteria

- `create_checkpoint()` successfully serializes aggregation state containing canonical-supported types (datetime/Decimal/bytes/numpy/pandas).
- `create_checkpoint()` raises a clear error for NaN/Infinity before writing to DB.
- `aggregation_state_json` is always valid JSON and round-trips via `json.loads()`.

## Tests

- Suggested tests to run: `pytest tests/engine/test_executors.py -k checkpoint`
- New tests required: Yes; add checkpoint serialization tests for non-JSON primitives and NaN rejection.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
