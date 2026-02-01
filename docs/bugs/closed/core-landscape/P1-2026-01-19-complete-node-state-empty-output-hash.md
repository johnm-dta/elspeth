# Bug Report: `complete_node_state()` treats empty output/error/context as absent, causing NULL hashes and crashes for valid empty outputs

## Summary

- `LandscapeRecorder.complete_node_state()` computes `output_hash` with `stable_hash(output_data) if output_data else None`, so an empty dict/list output (`{}` or `[]`) produces `output_hash=None`.
- When status is `completed`, `_row_to_node_state()` enforces `output_hash` is non-NULL and raises `ValueError`, so completing a node state with an empty-but-valid output crashes immediately.
- Similar truthiness bugs drop audit data:
  - `error_json = canonical_json(error) if error else None` loses `{}` error payloads.
  - `context_json = canonical_json(context_after) if context_after else None` loses `{}` context payloads.

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
- Notable tool calls or steps: minimal Python repro + code inspection

## Steps To Reproduce

1. Run the following Python snippet:
   - `python - <<'PY'`
   - `from elspeth.core.landscape.database import LandscapeDB`
   - `from elspeth.core.landscape.recorder import LandscapeRecorder`
   - `from elspeth.contracts.schema import SchemaConfig`
   - `schema = SchemaConfig.from_dict({"fields": "dynamic"})`
   - `db = LandscapeDB.in_memory()`
   - `rec = LandscapeRecorder(db)`
   - `run = rec.begin_run(config={}, canonical_version="v1")`
   - `node = rec.register_node(run_id=run.run_id, plugin_name="t", node_type="transform", plugin_version="1", config={}, schema_config=schema)`
   - `row = rec.create_row(run_id=run.run_id, source_node_id=node.node_id, row_index=0, data={})`
   - `token = rec.create_token(row_id=row.row_id)`
   - `state = rec.begin_node_state(token_id=token.token_id, node_id=node.node_id, step_index=0, input_data={})`
   - `rec.complete_node_state(state_id=state.state_id, status="completed", output_data={}, duration_ms=1.0)`
   - `PY`
2. Observe failure: `ValueError: COMPLETED state ... has NULL output_hash - audit integrity violation`.

## Expected Behavior

- Empty outputs are still valid outputs and must be hashed and recorded:
  - `{}` / `[]` should produce a non-NULL `output_hash`.
- Empty structured payloads should still be recorded:
  - `{}` error/context should serialize to `"{}"` (not `NULL`).

## Actual Behavior

- Empty outputs are treated as missing output, leading to `NULL output_hash` and a crash when reading the state back.
- Empty error/context payloads are silently dropped (`NULL` in DB), reducing audit completeness.

## Evidence

- Truthiness bug in hashing/serialization:
  - `src/elspeth/core/landscape/recorder.py:1078-1082`
- Enforcement that makes this crash immediately for completed states:
  - `src/elspeth/core/landscape/recorder.py:131-156` (`row.output_hash is None -> raise`)

## Impact

- User-facing impact: pipelines can crash if a transform/gate/sink produces an empty dict/list output (a legitimate outcome for “no fields changed” or “no items produced”).
- Data integrity / security impact: moderate. Empty-but-meaningful error/context payloads are lost, increasing inference risk.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- The implementation uses truthiness checks (`if output_data`) rather than explicit `is not None` checks, conflating “empty” with “absent”.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - Replace truthiness checks with `is not None` checks:
      - `output_hash = stable_hash(output_data) if output_data is not None else None`
      - `error_json = canonical_json(error) if error is not None else None`
      - `context_json = canonical_json(context_after) if context_after is not None else None`
- Config or schema changes: none.
- Tests to add/update:
  - Add tests ensuring:
    - completed state with `output_data={}` succeeds and has non-NULL `output_hash`
    - failed state with `error={}` stores `"{}"` (or explicitly documented behavior)
    - context snapshots `{}` are persisted, not dropped
- Risks or migration steps:
  - None; this tightens correctness without changing schema.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (“Transform boundaries - input AND output captured at every transform”)
- Observed divergence: valid outputs can be treated as absent, breaking the audit trail.
- Reason (if known): use of truthiness checks for optional payloads.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `complete_node_state(..., status="completed", output_data={})` completes successfully and persists a non-NULL `output_hash`.
- Empty dict/list payloads are preserved when explicitly provided.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py -k node_state`
- New tests required: yes (empty output/error/context coverage)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`
