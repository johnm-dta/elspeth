# Bug Report: JSONExplode silently “explodes” strings (and other iterables) into garbage rows instead of treating them as upstream type bugs

## Summary

- JSONExplode’s contract states that type violations (wrong type for `array_field`) indicate an upstream bug and should crash to preserve audit integrity.
- Implementation relies on `len(array_value)` and `enumerate(array_value)`, so *iterable but wrong* types like `str` and `dict` do not crash and instead produce nonsensical outputs (one row per character / key).
- This is explicitly documented by an existing test, but the behavior is still a correctness/auditability bug: bad types can silently produce incorrect pipeline outputs.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis + existing test evidence)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: any row where `array_field` is a string/dict

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: inspected transform implementation + tests

## Steps To Reproduce

1. Configure a pipeline that uses the `json_explode` transform with `array_field: items`.
2. Provide a row where `items` is a string, e.g. `{"id": 1, "items": "abc"}`.
3. Run the transform (or pipeline).

## Expected Behavior

- The transform should treat `items` being a string as an upstream type violation and crash (or otherwise fail loudly) rather than emitting incorrect rows.

## Actual Behavior

- The transform returns `success_multi` and emits one row per character.

## Evidence

- Implementation enumerates `array_value` without type checking: `src/elspeth/plugins/transforms/json_explode.py:125-151`
- Test explicitly documents current behavior as “almost certainly wrong”: `tests/plugins/transforms/test_json_explode.py:227-257`

## Impact

- User-facing impact: pipelines can silently produce incorrect records without crashing, making debugging and trust difficult.
- Data integrity / security impact: violates the “plugin bugs crash immediately” posture by allowing invalid types to pass through in a misleadingly “successful” way.
- Performance or cost impact: can multiply outputs unexpectedly (e.g., long strings => many tokens).

## Root Cause Hypothesis

- Python’s iteration semantics allow `str`/`dict` to behave like sequences, so “wrong type” isn’t guaranteed to raise `TypeError`.

## Proposed Fix

- Code changes (modules/files):
  - Add explicit runtime type enforcement in `JSONExplode.process()`:
    - require `array_value` to be a `list` (or `Sequence` excluding `str`/`bytes`/`dict`)
    - raise `TypeError` (upstream bug) when the type is not acceptable
  - Alternatively (or additionally), extend schema config/type system to support list types and validate `array_field` at the source boundary.
- Config or schema changes:
  - Consider requiring explicit “array_field is list” schema support, or documenting that JSONExplode requires upstream enforcement beyond current SchemaConfig capabilities.
- Tests to add/update:
  - Update the “string iterates over characters” test to assert a crash instead of documenting garbage output (once behavior changes).
- Risks or migration steps:
  - Behavior change may break pipelines that relied on implicit string iteration; treat that as correct breakage.

## Architectural Deviations

- Spec or doc reference: module docstring in `src/elspeth/plugins/transforms/json_explode.py:6-19` (type violations should crash)
- Observed divergence: wrong iterable types do not crash and instead produce “successful” but invalid outputs.
- Reason (if known): missing explicit type checks + schema system can’t express container types.
- Alignment plan or decision needed: define container typing support in schema system vs explicit checks in plugins.

## Acceptance Criteria

- When `array_field` is not a list/array, JSONExplode fails loudly (TypeError or equivalent), never producing “successful” garbage rows.

## Tests

- Suggested tests to run: `pytest tests/plugins/transforms/test_json_explode.py`
- New tests required: yes (or update existing documented-behavior test)

## Notes / Links

- Related docs: `CLAUDE.md` ("plugin bugs crash immediately", "no silent wrong result")

---

## CLOSED: 2026-01-23

**Resolution:** Fixed via explicit type check in `JSONExplode.process()`.

**Implementation:**
- Added `isinstance(array_value, list)` check immediately after extracting array_field from row (line 138-142)
- Raises TypeError with clear message when array_field is not a list
- Updated test `test_string_value_iterates_over_characters` → `test_string_value_crashes_with_type_error`
- Added tests: `test_dict_value_crashes_with_type_error`, `test_tuple_value_crashes_with_type_error`

**Files Changed:**
- `src/elspeth/plugins/transforms/json_explode.py` - Added type enforcement
- `tests/plugins/transforms/test_json_explode.py` - Updated and added tests

**Test Results:** All 18 tests in `test_json_explode.py` pass.

**Architecture Review:** Confirmed by axiom-system-architect:architecture-critic as correct approach. The isinstance check is contract enforcement (not defensive programming). Prevents audit trail fraud where garbage output is marked as "successfully processed."

**Follow-Up Work:** File ticket to add container type support (`list`, `dict`) to SchemaConfig to enable declarative validation at source boundary. Transform-level check remains as defense-in-depth.
