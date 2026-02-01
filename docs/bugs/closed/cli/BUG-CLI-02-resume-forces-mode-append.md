# Bug Report: Resume Forces mode=append For All Sinks, Breaking Non-CSV Sinks

## Status: RESOLVED ✅

**Resolution Date:** 2026-01-26
**Fixed In:** PR #5 (fix/rc1-bug-burndown-session-5)
**Resolution:** Implemented polymorphic resume capability - sinks declare `supports_resume` and self-configure via `configure_for_resume()`

---

## Summary

- Resume unconditionally injects `mode="append"` into every sink config, which violates sink config contracts for JSON/Database sinks and causes resume to fail with config validation errors.

## Severity

- Severity: critical
- Priority: P0

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-CLI-02

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Any pipeline with JSON or Database sink

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit of changed files on branch
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure a pipeline with a `json` or `database` sink (no `mode` field in config).
2. Run pipeline and force a failure to create a checkpoint.
3. Execute `elspeth resume <run_id> --execute`.

## Expected Behavior

- Resume should apply sink-specific append semantics (e.g., `if_exists="append"` for database sinks, JSONL append for JSON), or refuse to resume with a clear error for non-appendable sinks.

## Actual Behavior

- Resume injects `mode="append"` for ALL sinks, causing config validation errors for sinks that do not support `mode` field. Resume fails with Pydantic validation error.

## Evidence

- `src/elspeth/cli.py:1425` unconditionally sets `sink_options["mode"] = "append"` for every sink during resume:
  ```python
  sink_options["mode"] = "append"  # Line 1425 - applied to ALL sinks
  ```
- `src/elspeth/plugins/config_base.py:42` forbids unknown config fields with `extra="forbid"` in BasePluginConfig.
- `src/elspeth/plugins/sinks/json_sink.py:28` defines `JSONSinkConfig` without `mode` field.
- `src/elspeth/plugins/sinks/database_sink.py:42` defines `DatabaseSinkConfig` with `if_exists` field, not `mode`.
- `src/elspeth/plugins/sinks/csv_sink.py:29` shows `mode` is only defined for CSV sinks.

## Impact

- User-facing impact: Resume fails completely for pipelines using JSON or Database sinks. Error message is cryptic Pydantic validation failure.
- Data integrity / security impact: Resume is blocked; operators must choose between losing checkpoint progress or manually modifying configs, risking data duplication.
- Performance or cost impact: Lost checkpoint progress means full pipeline re-run, wasting compute and time.

## Root Cause Hypothesis

- CLI resume logic assumes a universal `mode="append"` setting works across all sinks, but sink config schemas are plugin-specific and forbid unknown fields via `extra="forbid"`.

## Proposed Fix

- Code changes (modules/files):
  - Update `src/elspeth/cli.py:1425` to apply sink-type-specific append logic:

  ```python
  # Option A: Sink-specific append semantics
  sink_type = sink_config["type"]
  if sink_type == "csv":
      sink_options["mode"] = "append"
  elif sink_type == "database":
      sink_options["if_exists"] = "append"
  elif sink_type == "json":
      # JSONL can append, JSON array cannot
      format = sink_config.get("format", "json")
      if format == "jsonl":
          pass  # File-level append works for JSONL
      else:
          raise ValueError(
              f"Cannot resume with JSON array sink '{sink_name}'. "
              "JSON array format does not support append. "
              "Use JSONL format or re-run from start."
          )
  elif sink_type == "azure_blob":
      # Azure blob doesn't support append - must overwrite or error
      raise ValueError(
          f"Cannot resume with Azure Blob sink '{sink_name}'. "
          "Blob storage does not support append. "
          "Re-run from start or use appendable sink."
      )
  else:
      # Unknown sink type - fail fast
      raise ValueError(
          f"Cannot resume with sink type '{sink_type}'. "
          "Resume behavior not defined for this sink type."
      )
  ```

- Config or schema changes: None.

- Tests to add/update:
  - `test_resume_csv_sink_uses_mode_append()` - Verify CSV sink gets `mode="append"`
  - `test_resume_database_sink_uses_if_exists_append()` - Verify DB sink gets `if_exists="append"`
  - `test_resume_jsonl_sink_appends()` - Verify JSONL resume works
  - `test_resume_json_array_sink_rejects()` - Verify JSON array resume rejected with clear error
  - `test_resume_azure_blob_sink_rejects()` - Verify blob sink resume rejected

- Risks or migration steps:
  - Breaking change: Some sink types will now reject resume with clear errors (JSON array, Azure blob)
  - This is acceptable per CLAUDE.md RC policy - fail fast better than silent data corruption
  - Document known limitation: Not all sink types support resume

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/plugins/config_base.py:42` (extra="forbid")
- Observed divergence: CLI injects unsupported config fields, violating strict config schema contracts enforced by Pydantic.
- Reason (if known): Resume logic assumes uniform append semantics across heterogeneous sink plugins.
- Alignment plan or decision needed: Enforce sink-specific append semantics or fail fast with targeted error messages.

## Acceptance Criteria

- Resume does not inject unsupported config fields into any sink.
- CSV sinks resume with `mode="append"`.
- Database sinks resume with `if_exists="append"`.
- JSONL sinks resume successfully (file append mode).
- JSON array sinks reject resume with clear, actionable error message.
- Azure Blob sinks reject resume with clear error or implement append semantics.

## Tests

- Suggested tests to run: `pytest tests/integration/test_resume_sinks.py`
- New tests required: yes, resume sink append behavior per sink type (5 tests listed above)

## Notes / Links

- Related issues/PRs: PR #5
- Related design docs:
  - `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md` - Bug triage report
  - `CLAUDE.md` - Three-Tier Trust Model, Plugin System
  - `src/elspeth/plugins/config_base.py` - Config validation

---

## Resolution Evidence

### Actual Fix (Polymorphic Approach)

Instead of the proposed sink-type-specific conditionals in the CLI, we implemented a **polymorphic pattern** that respects plugin encapsulation:

1. **SinkProtocol** now declares resume capability:
   ```python
   supports_resume: bool  # Can this sink append on resume?
   def configure_for_resume(self) -> None: ...  # Self-configure for append
   ```

2. **Each sink implements its own resume logic**:
   - `CSVSink`: `supports_resume=True`, sets `_mode="append"`
   - `DatabaseSink`: `supports_resume=True`, sets `_if_exists="append"`
   - `JSONSink`: `supports_resume=True` for JSONL (can append lines), `False` for JSON array (must rewrite)
   - `AzureBlobSink`: `supports_resume=False` (Azure Blobs are immutable)

3. **CLI queries capability instead of injecting fields**:
   ```python
   if not sink.supports_resume:
       typer.echo(f"Error: Cannot resume with sink '{sink_name}'...", err=True)
       raise typer.Exit(1)
   sink.configure_for_resume()  # Let sink self-configure
   ```

### Files Changed

- `src/elspeth/plugins/protocols.py` - Added `supports_resume` and `configure_for_resume` to SinkProtocol
- `src/elspeth/plugins/base.py` - Added default implementation to BaseSink
- `src/elspeth/plugins/sinks/csv_sink.py` - Implemented resume capability
- `src/elspeth/plugins/sinks/database_sink.py` - Implemented resume capability
- `src/elspeth/plugins/sinks/json_sink.py` - Implemented format-dependent resume
- `src/elspeth/plugins/azure/blob_sink.py` - Declared no resume support with helpful error
- `src/elspeth/cli.py` - Replaced hardcoded `mode=append` injection with polymorphic calls

### Tests Added (41 total)

- `tests/plugins/sinks/test_csv_sink_resume.py` (3 tests)
- `tests/plugins/sinks/test_database_sink_resume.py` (3 tests)
- `tests/plugins/sinks/test_json_sink_resume.py` (9 tests)
- `tests/plugins/azure/test_blob_sink_resume.py` (2 tests)
- `tests/integration/test_cli_resume_sink_capability.py` (12 tests)
- `tests/integration/test_cli_resume_sink_append.py` (8 tests, updated)
- `tests/contracts/sink_contracts/test_sink_protocol.py` (2 tests)
- `tests/plugins/test_base_sink.py` (2 tests)

### Acceptance Criteria Verification

| Criterion | Status |
|-----------|--------|
| Resume does not inject unsupported config fields | ✅ No field injection - uses `configure_for_resume()` |
| CSV sinks resume with `mode="append"` | ✅ `CSVSink.configure_for_resume()` sets `_mode="append"` |
| Database sinks resume with `if_exists="append"` | ✅ `DatabaseSink.configure_for_resume()` sets `_if_exists="append"` |
| JSONL sinks resume successfully | ✅ `JSONSink` with JSONL format supports resume |
| JSON array sinks reject with clear error | ✅ Raises `NotImplementedError` with guidance |
| Azure Blob sinks reject with clear error | ✅ Raises `NotImplementedError` explaining immutability |
