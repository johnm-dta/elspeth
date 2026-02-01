# Bug Report: Resume Drops Row Data By Validating Through NullSource Schema

## RESOLUTION (2026-01-27)

**Status: ALREADY FIXED**

This bug was fixed in commit `b2a3518` (2026-01-23) before this report was filed. The report was based on static analysis of outdated code.

### Evidence of Fix:
1. `orchestrator.py:1511-1520` retrieves schema from audit trail via `recorder.get_source_schema(run_id)`, NOT from `config.source._schema_class`
2. `orchestrator.py:1354-1359` validates schema has fields and raises clear error if empty
3. 110 resume-related tests pass, including type fidelity tests for datetime/Decimal
4. `test_resume_comprehensive.py` covers datetime, Decimal, array, and nested object restoration

### Verification:
```bash
.venv/bin/python -m pytest tests/ -k "resume" -v  # 110 passed
```

---

## Summary (Original Report)

- Resume replaces the real source with `NullSource`, so type restoration validates against an empty schema and drops all row fields, producing empty `row_data` during resume.

## Severity

- Severity: ~~critical~~ **RESOLVED**
- Priority: ~~P0~~ **CLOSED**

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 8635789 (fix/rc1-bug-burndown-session-4)
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `/home/john/elspeth-rapid/src/elspeth/cli.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run a pipeline with a real source schema (e.g., CSV source with explicit fields) and force a failure so a checkpoint exists.
2. Execute `elspeth resume <run_id> --execute`.
3. Observe transforms receiving empty or missing row fields (or crashing on missing keys).

## Expected Behavior

- Resume restores full row data with correct types using the original source schema.

## Actual Behavior

- Resume validates payloads with `NullSourceSchema`, which ignores all fields, resulting in empty `row_data` for resumed rows.

## Evidence

- `src/elspeth/cli.py:1436` replaces the real source with `NullSource` during resume.
- `src/elspeth/engine/orchestrator.py:1343` uses `config.source._schema_class` for type restoration in resume.
- `src/elspeth/plugins/sources/null_source.py:50` sets `_schema_class = NullSourceSchema` (no fields).
- `src/elspeth/contracts/data.py:43` configures `PluginSchema` with `extra="ignore"`, so an empty schema drops all fields.
- `src/elspeth/core/checkpoint/recovery.py:140` documents that source schema is REQUIRED to preserve type fidelity.

## Impact

- User-facing impact: Resume can crash or produce incorrect outputs due to missing fields.
- Data integrity / security impact: Resumed rows lose original data, breaking auditability and lineage.
- Performance or cost impact: Failed resume can require full re-runs and manual recovery.

## Root Cause Hypothesis

- CLI overwrites the source with `NullSource` without preserving the original source schema, so resume uses an empty schema that discards all fields.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/cli.py` to preserve the original source schema when constructing `NullSource`, e.g., copy `_schema_class` (and `output_schema` if needed) from `plugins["source"]` onto `null_source`, or keep the original source instance in `PipelineConfig` while still avoiding `load()`.
- Config or schema changes: None.
- Tests to add/update:
  - Add a resume test that uses a typed source schema (datetime/Decimal) and asserts resumed `row_data` retains fields and types.
- Risks or migration steps:
  - Ensure resume still does not call `source.load()`; only schema should be reused.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/core/checkpoint/recovery.py:140`
- Observed divergence: Resume validates payloads with an empty schema, violating the documented type fidelity requirement.
- Reason (if known): CLI swaps the source to `NullSource` without preserving schema metadata.
- Alignment plan or decision needed: Preserve original source schema during resume.

## Acceptance Criteria

- Resume uses the original source schema for type restoration.
- Resumed rows retain all fields and expected types.
- Resume completes without missing-field errors in downstream transforms.

## Tests

- Suggested tests to run: `pytest tests/test_resume_type_fidelity.py`
- New tests required: yes, resume preserves row fields/types across checkpoint recovery

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/core/checkpoint/recovery.py`
---
# Bug Report: Resume Forces mode=append For All Sinks, Breaking Non-CSV Sinks

## Summary

- Resume unconditionally injects `mode="append"` into every sink config, which violates sink config contracts for JSON/Database sinks and causes resume to fail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 8635789 (fix/rc1-bug-burndown-session-4)
- OS: Unknown
- Python version: Unknown
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of `/home/john/elspeth-rapid/src/elspeth/cli.py`
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with a `json` or `database` sink (no `mode` field).
2. Create a failed run with a checkpoint.
3. Execute `elspeth resume <run_id> --execute`.

## Expected Behavior

- Resume should either apply sink-specific append semantics or refuse to resume with a clear, targeted error.

## Actual Behavior

- Resume injects `mode="append"` for all sinks, causing config validation errors for sinks that do not support `mode`.

## Evidence

- `src/elspeth/cli.py:1425` sets `sink_options["mode"] = "append"` for every sink during resume.
- `src/elspeth/plugins/config_base.py:42` forbids unknown config fields (extra=forbid).
- `src/elspeth/plugins/sinks/json_sink.py:28` defines `JSONSinkConfig` without `mode`.
- `src/elspeth/plugins/sinks/database_sink.py:42` defines `DatabaseSinkConfig` with `if_exists`, not `mode`.
- `src/elspeth/plugins/sinks/csv_sink.py:29` shows `mode` is only defined for CSV sinks.

## Impact

- User-facing impact: Resume fails for pipelines that use JSON or Database sinks.
- Data integrity / security impact: Recovery is blocked; operators may re-run pipelines and risk duplication.
- Performance or cost impact: Extra operator time and potential reprocessing cost.

## Root Cause Hypothesis

- CLI assumes a universal `mode` setting across sinks, but sink config schemas are plugin-specific and forbid unknown fields.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/cli.py` to apply append behavior only for sinks that support it (e.g., `csv` uses `mode="append"`, `database` uses `if_exists="append"`), or explicitly block resume with a clear message for non-appendable sinks (e.g., JSON array mode).
- Config or schema changes: None.
- Tests to add/update:
  - Add resume tests covering CSV, JSON, and Database sinks to validate append-mode handling.
- Risks or migration steps:
  - Ensure any new behavior does not silently alter sink semantics; prefer explicit errors for unsupported append.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/config_base.py:42`
- Observed divergence: CLI injects unsupported config fields, violating strict config contracts.
- Reason (if known): Resume logic assumes a single append flag for all sinks.
- Alignment plan or decision needed: Enforce sink-specific append semantics or fail fast with a targeted error.

## Acceptance Criteria

- Resume does not inject unsupported config fields.
- CSV/Database sinks resume with append semantics as intended.
- JSON sinks either resume safely (JSONL append) or error with a clear message.

## Tests

- Suggested tests to run: `pytest tests/test_resume_sinks.py`
- New tests required: yes, resume sink append behavior per sink type

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `src/elspeth/plugins/config_base.py`
