# Bug Report: TriggerType exists and DB has batches.trigger_type, but Batch contract/recorder never set it

## Summary

- `TriggerType` is defined as a DB-serialized enum “for batches.trigger_type”, and the Landscape schema includes a `batches.trigger_type` column.
- The `contracts.audit.Batch` dataclass does not include `trigger_type`.
- `LandscapeRecorder.create_batch()` / `update_batch_status()` / `complete_batch()` / `get_batch()` never write or read `trigger_type`, so the column remains NULL.
- The engine has trigger evaluation logic that can identify the trigger cause, but this information is lost from the audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Steps To Reproduce

1. Configure an aggregation (batch-aware transform) with a trigger (count/timeout/condition).
2. Run a pipeline until a batch flush occurs.
3. Inspect the `batches` table: `trigger_reason` may be set, but `trigger_type` remains NULL.

## Expected Behavior

- When a batch is triggered, the audit trail records both:
  - `trigger_type` (one of `TriggerType` enum values)
  - `trigger_reason` (human-readable/details)

## Actual Behavior

- `trigger_type` is never recorded or surfaced through the recorder/contract API.

## Evidence

- Enum defined for DB serialization:
  - `src/elspeth/contracts/enums.py:56-74` (`TriggerType`, doc mentions `batches.trigger_type`)
- DB schema includes column:
  - `src/elspeth/core/landscape/schema.py:221-236` (`Column("trigger_type", String(32))`)
- Batch contract missing field:
  - `src/elspeth/contracts/audit.py:258-274`
- Recorder writes/reads batches without trigger_type:
  - `src/elspeth/core/landscape/recorder.py:1253-1296` (`create_batch()` insert omits trigger_type)
  - `src/elspeth/core/landscape/recorder.py:1331-1399` (`update_batch_status()`/`complete_batch()` omit trigger_type)
  - `src/elspeth/core/landscape/recorder.py:1400-1475` (`get_batch()`/`get_batches()` don’t read trigger_type)
- Engine has trigger-type logic:
  - `src/elspeth/engine/triggers.py:135` (`get_trigger_type()`)

## Impact

- User-facing impact: explain/export cannot accurately answer “why did this batch flush?” beyond a free-form reason.
- Data integrity / security impact: medium: audit trail is missing a structured causal field for a critical control-flow event.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- TriggerType column and enum were introduced, but batch lifecycle plumbing was not updated to propagate trigger type from trigger evaluation to recorder and contract.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/contracts/audit.py`: add `trigger_type: TriggerType | None` to `Batch`.
  - `src/elspeth/core/landscape/recorder.py`: add trigger_type handling in batch update/completion paths (store `.value` in DB; coerce back to enum on reads).
  - `src/elspeth/engine/executors.py` / aggregation flush path: pass the trigger type from the trigger evaluator into recorder updates.
- Tests to add/update:
  - Add integration/unit tests asserting `batches.trigger_type` is set for:
    - end-of-source flush
    - count trigger
    - timeout trigger (if supported in tests)

## Architectural Deviations

- Spec or doc reference: “Trigger types stored in batches.trigger_type” (enum doc)
- Observed divergence: column exists but is unused/unobservable
- Alignment plan or decision needed: decide where trigger_type is set (likely at transition draft/executing or executing/completed)

## Acceptance Criteria

- Batches created and completed by the engine have `trigger_type` populated in DB and surfaced via `LandscapeRecorder.get_batch()` / `get_batches()`.
- Export/explain outputs include trigger_type.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine -k batch`
- New tests required: yes
