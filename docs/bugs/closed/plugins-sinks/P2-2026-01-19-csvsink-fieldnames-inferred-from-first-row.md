# Bug Report: CSVSink infers headers from the first row, ignoring configured schema; later rows with additional (valid) fields can crash

## Summary

- `CSVSink` lazily initializes its `csv.DictWriter` with `fieldnames=list(rows[0].keys())`.
- `csv.DictWriter` defaults `extrasaction="raise"`, so if a later row includes a key not present in the first row's keys, `writer.writerow(row)` raises `ValueError`.
- This is especially problematic for explicit schemas with optional fields: the first row may omit an optional field, but later rows can legally include it.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A (static analysis)

## Environment

- Commit/branch: `main` @ `8cfebea78be241825dd7487fed3773d89f2d7079`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into subsystem 6 (plugins), identify bugs, create tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (no runtime execution)

## Steps To Reproduce

1. Configure `CSVSink` with an explicit schema that includes an optional field, e.g. `["id: int", "score: float?"]`.
2. Call `sink.write([{"id": 1}], ctx)` (first batch lacks `score` key).
3. Call `sink.write([{"id": 2, "score": 1.0}], ctx)` (second batch includes optional `score` key).
4. Observe `ValueError` from `csv.DictWriter.writerow` due to unexpected field.

## Expected Behavior

- The sink should treat schema-defined fields (including optional ones) as valid across all batches and not crash simply because the first row omitted them.

## Actual Behavior

- Header/fieldnames are derived from the first row only; later valid keys can crash the sink.

## Evidence

- Fieldnames derived from first row keys: `src/elspeth/plugins/sinks/csv_sink.py:114-126`
- Schema config exists and is validated, but not used to determine fieldnames: `src/elspeth/plugins/sinks/csv_sink.py:64-78`

## Impact

- User-facing impact: pipelines can crash mid-run depending on row ordering (first row shape influences sink behavior).
- Data integrity / security impact: auditability suffers if sinks fail nondeterministically based on input ordering.
- Performance or cost impact: reruns and debugging time.

## Root Cause Hypothesis

- Sink uses data-driven inference for header fields instead of schema-driven definition.

## Proposed Fix

- Code changes (modules/files):
  - If `schema_config` is explicit (not dynamic), derive `fieldnames` from `schema_config.fields` (stable ordering) instead of `rows[0].keys()`.
  - For dynamic schema, decide a clear policy:
    - either enforce stable columns (fail fast if keys vary), or
    - support evolving columns (requires rewriting header / buffering), or
    - ignore extras (`extrasaction="ignore"`) and record a warning/audit event.
- Config or schema changes:
  - Potentially document/require "stable field set" for CSV sinks.
- Tests to add/update:
  - Add a test with explicit schema + optional field that appears only in later rows and assert sink does not crash (or fails fast with a clear, deterministic error if that's the chosen policy).
- Risks or migration steps:
  - Changing header generation affects output CSV column order; document this for consumers.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` (deterministic, auditable behavior; "no silent wrong results")
- Observed divergence: sink behavior depends on incidental first-row shape rather than declared schema.
- Reason (if known): convenience lazy initialization without schema integration.
- Alignment plan or decision needed: define canonical CSV column ordering strategy (schema-driven preferred).

## Acceptance Criteria

- With an explicit schema, `CSVSink` writes consistent headers and accepts rows that include any schema-defined fields in any batch order.

## Tests

- Suggested tests to run: `pytest tests/plugins/sinks/test_csv_sink.py`
- New tests required: yes

## Notes / Links

- Related file: `src/elspeth/contracts/schema.py` (optional field support via `?`)

---

## Resolution

**Status:** CLOSED
**Date:** 2026-01-21
**Resolved by:** Claude Opus 4.5

### Root Cause Confirmed

The `_open_file` method set `self._fieldnames = list(rows[0].keys())` instead of using the stored `_schema_config.fields` when an explicit schema was configured.

### Fix Applied

Modified `src/elspeth/plugins/sinks/csv_sink.py`:

1. Added new method `_get_fieldnames_from_schema_or_row()`:
   - When schema is explicit (not dynamic) and has fields, returns field names from `schema_config.fields`
   - When schema is dynamic, falls back to inferring from first row keys (original behavior preserved)

2. Updated `_open_file` to use this new method instead of directly accessing `rows[0].keys()`

3. Updated class docstring to reflect the new behavior.

### Tests Added

Two new tests in `tests/plugins/sinks/test_csv_sink.py`:

1. `test_explicit_schema_creates_all_headers_including_optional` - Verifies optional fields present in headers even when first row doesn't include them
2. `test_dynamic_schema_still_infers_from_row` - Verifies backward compatibility for dynamic schemas

### Verification

- All 16 CSV sink tests pass
- All 6 append mode tests pass
- Type checking (mypy) passes
- Linting (ruff) passes

### Related Fix

This bug was a sibling of P1-2026-01-19-databasesink-schema-inferred-from-first-row, which had the same root cause pattern. Both were fixed in the same session using the same approach.
