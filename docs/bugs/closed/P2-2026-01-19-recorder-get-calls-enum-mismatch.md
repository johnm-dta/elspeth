# Bug Report: `LandscapeRecorder.get_calls()` returns `Call.call_type`/`Call.status` as raw strings (violates strict enum contracts)

## Summary

- The strict audit contract requires `Call.call_type: CallType` and `Call.status: CallStatus`.
- `LandscapeRecorder.get_calls()` returns these fields as raw DB strings (`"llm"`, `"success"`, etc.) without coercion, undermining type-safety and making contract usage inconsistent across the codebase (repositories coerce; recorder doesn’t).

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: minimal repro inserts into `calls_table` + code inspection

## Steps To Reproduce

1. Run the following Python snippet (inserts a call row manually, then reads via `get_calls()`):
   - `python - <<'PY'`
   - `from datetime import UTC, datetime`
   - `from elspeth.contracts.schema import SchemaConfig`
   - `from elspeth.core.landscape.database import LandscapeDB`
   - `from elspeth.core.landscape.recorder import LandscapeRecorder`
   - `from elspeth.core.landscape.schema import calls_table`
   - `from elspeth.contracts.enums import CallType, CallStatus`
   - `schema = SchemaConfig.from_dict({"fields": "dynamic"})`
   - `db = LandscapeDB.in_memory()`
   - `rec = LandscapeRecorder(db)`
   - `run = rec.begin_run(config={}, canonical_version="v1")`
   - `node = rec.register_node(run_id=run.run_id, plugin_name="t", node_type="transform", plugin_version="1", config={}, schema_config=schema)`
   - `row = rec.create_row(run_id=run.run_id, source_node_id=node.node_id, row_index=0, data={})`
   - `token = rec.create_token(row_id=row.row_id)`
   - `state = rec.begin_node_state(token_id=token.token_id, node_id=node.node_id, step_index=0, input_data={})`
   - `with db.connection() as conn:`
   - `    conn.execute(calls_table.insert().values(call_id="c1", state_id=state.state_id, call_index=0, call_type=CallType.LLM.value, status=CallStatus.SUCCESS.value, request_hash="rh", created_at=datetime.now(UTC)))`
   - `call = rec.get_calls(state.state_id)[0]`
   - `print(type(call.call_type), call.call_type, type(call.status), call.status)`
   - `PY`
2. Observe `call.call_type` and `call.status` are `str`, not enums.

## Expected Behavior

- `get_calls()` should return `Call` objects where:
  - `call.call_type` is a `CallType`
  - `call.status` is a `CallStatus`
- Invalid enum strings in the audit DB should raise (Tier 1 invariant).

## Actual Behavior

- `get_calls()` returns raw strings for `call_type` and `status`.

## Evidence

- Contract requires enums:
  - `src/elspeth/contracts/audit.py:204-223`
- Recorder does not coerce:
  - `src/elspeth/core/landscape/recorder.py:1922-1957`
- Repository layer does coerce (inconsistent behavior across APIs):
  - `src/elspeth/core/landscape/repositories.py:173-194`

## Impact

- User-facing impact: TUI/export/explain features can behave inconsistently depending on whether they use recorder methods or repository layer.
- Data integrity / security impact: moderate. Invalid enum strings can silently flow through as plain strings, delaying crash and increasing ambiguity.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- `LandscapeRecorder` duplicates model construction and missed enum coercion for call fields (unlike `CallRepository.load()`).

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - In `get_calls()`, coerce:
      - `call_type=CallType(r.call_type)`
      - `status=CallStatus(r.status)`
  - Ensure this matches the strict “crash on invalid Tier 1 data” policy.
- Config or schema changes: none.
- Tests to add/update:
  - Add a recorder-level test asserting enum types from `get_calls()`.
- Risks or migration steps:
  - Existing DBs with invalid call enum strings will now crash on read (desired); provide actionable error messaging if needed.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“Tier 1: crash on invalid audit DB values”)
- Observed divergence: recorder returns non-contract types for call records.
- Reason (if known): missed conversion during recorder implementation.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `get_calls()` returns `CallType`/`CallStatus` enums for all call records.
- Invalid stored enum strings crash immediately with a clear error.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py tests/core/landscape/test_repositories.py`
- New tests required: yes (recorder call enum coercion)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md` (external call capture)
