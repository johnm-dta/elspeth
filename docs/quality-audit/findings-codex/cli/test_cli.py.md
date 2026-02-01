# Test Defect Report

## Summary

- CLI run tests execute pipelines with Landscape enabled but never validate audit trail tables/hashes, so audit regressions can pass undetected.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/cli/test_cli.py:89`
  ```python
  landscape:
    enabled: true
    backend: sqlite
    url: "sqlite:///{audit_db}"
  ```
- `tests/cli/test_cli.py:95`
  ```python
  result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])
  assert result.exit_code == 0
  assert output_file.exists()
  ```
- `tests/cli/test_cli.py:151`
  ```python
  result = runner.invoke(app, ["run", "-s", str(config_file), "--execute", "-v"])
  assert output_file.exists()
  assert "new_name" in output_content
  ```

## Impact

- Audit recording regressions (missing `node_states`, `token_outcomes`, `artifacts`, wrong hashes/lineage) could ship while CLI tests still pass.
- Violates the Auditability Standard: outputs appear correct but no verified audit trail exists, creating false confidence in traceability.

## Root Cause Hypothesis

- Tests prioritize CLI success and output file content; auditability checks were not added or standardized for CLI integration tests.

## Recommended Fix

- After each `run` invocation in these tests, query the Landscape DB and assert expected audit records and hash integrity.
- Example pattern:
  ```python
  from sqlalchemy import select
  from elspeth.core.canonical import stable_hash
  from elspeth.core.landscape import LandscapeDB
  from elspeth.core.landscape.schema import artifacts_table, node_states_table, token_outcomes_table

  db = LandscapeDB.from_url(f"sqlite:///{audit_db}")
  with db.connection() as conn:
      states = conn.execute(select(node_states_table)).fetchall()
      assert states
      assert all(s.status == "completed" for s in states)
      assert all(s.input_hash and s.output_hash for s in states)
      assert all(s.error_json in (None, "") for s in states)

      outcomes = conn.execute(select(token_outcomes_table)).fetchall()
      assert any(o.outcome == "completed" and o.is_terminal == 1 for o in outcomes)

      artifacts = conn.execute(select(artifacts_table)).fetchall()
      assert artifacts
      assert all(a.content_hash for a in artifacts)

  # Optional: compute stable_hash on input/output rows and compare to node_states hashes.
  ```
- This directly verifies `node_states`, `token_outcomes`, and `artifacts` rows and hash fields for the executed run.
---
# Test Defect Report

## Summary

- `test_purge_dry_run` uses a vacuous `"0"` substring check that can pass even when dry-run output is incorrect.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_cli.py:188`
  ```python
  assert "would delete" in result.stdout.lower() or "0" in result.stdout
  ```
- `src/elspeth/cli.py:1055`
  ```python
  typer.echo(f"No payloads older than {effective_retention_days} days found.")
  ```

## Impact

- The test can pass if the output contains any "0" (e.g., from "90" days), even if the CLI fails to show dry-run deletion info, reducing detection of regressions in purge messaging.

## Root Cause Hypothesis

- Assertion was loosened to avoid brittle output matching, but it became too permissive.

## Recommended Fix

- Assert explicit expected output for the no-payload case, or create expired payloads and assert the exact dry-run delete count.
- Example:
  ```python
  assert f"No payloads older than {expected_days} days found." in result.stdout
  # or, with fixtures that create expired payloads:
  assert f"Would delete {expected_count} payload(s)" in result.stdout
  ```
