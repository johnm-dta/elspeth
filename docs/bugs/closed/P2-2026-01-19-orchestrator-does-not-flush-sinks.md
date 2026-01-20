# Bug Report: Orchestrator never calls SinkProtocol.flush() for pipeline sinks

## Summary

- `SinkProtocol.flush()` is documented as being called “periodically and at end of run,” but `Orchestrator` never calls `flush()` for pipeline sinks (only `close()`).
- This can cause lost or partial outputs for sinks that buffer data internally and rely on explicit flush (e.g., network sinks, database sinks, async writers).

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: any pipeline using buffered sinks
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into system 5 (engine) and look for bugs
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: protocol review + orchestrator code inspection

## Steps To Reproduce

1. Use a sink implementation that buffers data and only persists on `flush()`.
2. Run a pipeline that writes some rows.
3. Observe that outputs are missing or incomplete if `close()` does not implicitly flush (or if the sink requires flush semantics separate from close).

## Expected Behavior

- Engine calls `sink.flush()` after writing buffered tokens and again during finalization before `sink.close()`.

## Actual Behavior

- Engine calls `sink.close()` but not `sink.flush()` for pipeline sinks.

## Evidence

- Sink writes occur via `SinkExecutor.write(...)`:
  - `src/elspeth/engine/orchestrator.py:676`
  - `src/elspeth/engine/orchestrator.py:684`
- Finalization closes sinks without flushing:
  - `src/elspeth/engine/orchestrator.py:707`
- SinkProtocol specifies flush should be called:
  - `src/elspeth/plugins/protocols.py:434`

## Impact

- User-facing impact: buffered sinks may not persist data reliably.
- Data integrity / security impact: missing sink artifacts undermine audit record completeness.
- Performance or cost impact: reruns required; buffered sinks may accumulate memory unnecessarily.

## Root Cause Hypothesis

- Orchestrator relies on `close()` to handle final persistence, but the protocol explicitly distinguishes `flush()` and `close()`.

## Proposed Fix

- Code changes (modules/files):
  - Call `sink.flush()` after each `sink_executor.write(...)` (best-effort with suppression) and in `finally:` before `sink.close()`.
  - Consider adding periodic flushing policy for long runs.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test sink whose `close()` does not flush; verify orchestrator calls flush and data is persisted.
- Risks or migration steps:
  - Ensure flush failures are handled carefully (plugin bugs should likely crash, per trust model).

## Architectural Deviations

- Spec or doc reference: `src/elspeth/plugins/protocols.py` sink lifecycle contract
- Observed divergence: flush is never called for pipeline sinks.
- Alignment plan or decision needed: confirm whether close is required to imply flush or keep explicit flush calls in engine.

## Acceptance Criteria

- Sinks receive `flush()` at least once at end-of-run (and optionally after writes).

## Tests

- Suggested tests to run: `pytest tests/engine/test_orchestrator.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
