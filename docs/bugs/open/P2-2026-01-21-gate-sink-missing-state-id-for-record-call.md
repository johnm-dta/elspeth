# Bug Report: Gate and sink execution never set ctx.state_id, breaking ctx.record_call

## Summary

- `PluginContext.record_call()` requires `ctx.state_id` and increments a per-state call index.
- `GateExecutor.execute_gate()` and `SinkExecutor.write()` never set `ctx.state_id` or reset `ctx._call_index`.
- Any gate or sink that attempts to record external calls via `ctx.record_call()` will raise `RuntimeError`, preventing audit of external calls at those nodes.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (local)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into contents of `src/elspeth/plugins` and create bug tickets
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: static inspection of `src/elspeth/engine/executors.py` and `src/elspeth/plugins/context.py`

## Steps To Reproduce

1. Implement a `BaseGate` or `BaseSink` that calls `ctx.record_call(...)` during execution.
2. Run a pipeline that executes the gate or sink.
3. Observe `RuntimeError: Cannot record call: state_id not set`.

## Expected Behavior

- Gate and sink executions should set `ctx.state_id` (and reset call_index) so external calls can be recorded in the audit trail.

## Actual Behavior

- `ctx.state_id` is never set for gates or sinks, so `ctx.record_call()` always raises.

## Evidence

- `PluginContext.record_call()` raises when `state_id` is None: `src/elspeth/plugins/context.py`.
- `GateExecutor.execute_gate()` does not set `ctx.state_id` or `ctx._call_index`: `src/elspeth/engine/executors.py`.
- `SinkExecutor.write()` does not set `ctx.state_id` or `ctx._call_index`: `src/elspeth/engine/executors.py`.

## Impact

- User-facing impact: any gate/sink that uses external services cannot record calls and will crash if it tries.
- Data integrity / security impact: external call audit trail is incomplete for gates/sinks, violating auditability requirements.
- Performance or cost impact: failed runs when gates/sinks attempt to record external calls.

## Root Cause Hypothesis

- `ctx.state_id` initialization was added for transform execution paths but not mirrored for gates or sinks.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/executors.py`:
    - In `GateExecutor.execute_gate()`, set `ctx.state_id = state.state_id`, `ctx.node_id = gate.node_id`, and reset `ctx._call_index = 0` before calling `gate.evaluate()`.
    - In `SinkExecutor.write()`, decide on a representative state for external call recording (e.g., the first token’s node_state), set `ctx.state_id` and reset `ctx._call_index` before `sink.write()`.
    - Alternatively, add a sink/gate-specific call recording helper that accepts an explicit `state_id`.
- Tests to add/update:
  - Add a gate test that calls `ctx.record_call()` and asserts the call is recorded under the gate’s node_state.
  - Add a sink test that calls `ctx.record_call()` during `write()` and asserts a valid call record exists.
- Risks or migration steps:
  - For sinks that write batches, confirm which node_state should own the call records (document the chosen policy).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md` auditability standard (external calls recorded).
- Observed divergence: gate/sink external calls cannot be recorded via PluginContext.
- Reason (if known): state_id setup exists only for transform execution paths.
- Alignment plan or decision needed: define gate/sink call recording semantics (single representative state vs per-token state).

## Acceptance Criteria

- Gates and sinks can call `ctx.record_call()` without raising.
- External calls from gates/sinks are recorded against a valid `node_states.state_id`.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_executors.py`
- New tests required: yes (gate/sink external call recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
