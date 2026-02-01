# Test Defect Report

## Summary

- Using `using-quality-engineering` (test-maintenance-patterns) because this is a test quality audit; explain CLI tests rely on weak assertions (loose substring checks, no exit-code/JSON validation), so regressions can slip through.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Weak Assertions]

## Evidence

- `tests/cli/test_explain_command.py:15`, `tests/cli/test_explain_command.py:16`, `tests/cli/test_explain_command.py:17` only assert non-zero exit and generic substrings, allowing unrelated failures to pass.
  ```python
  result = runner.invoke(app, ["explain"])
  assert result.exit_code != 0
  assert "missing" in result.output.lower() or "required" in result.output.lower()
  ```
- `tests/cli/test_explain_command.py:23`, `tests/cli/test_explain_command.py:24`, `tests/cli/test_explain_command.py:26` allow `"not found"` and only check stdout, which can mask import/errors; the command writes to stderr and exits with code 2.
  ```python
  result = runner.invoke(app, ["explain", "--run", "test-run", "--no-tui"])
  assert "not yet implemented" in result.output.lower() or "not found" in result.output.lower()
  ```
  ```python
  typer.echo("Note: The explain --no-tui command is not yet implemented.", err=True)
  ...
  raise typer.Exit(2)
  ```
  (`src/elspeth/cli.py:321`, `src/elspeth/cli.py:325`)
- `tests/cli/test_explain_command.py:32`, `tests/cli/test_explain_command.py:34` only check a leading `{`/`[` and never validate JSON or the expected fields/exit code despite explicit contract in CLI.
  ```python
  result = runner.invoke(app, ["explain", "--run", "test-run", "--json"])
  assert result.output.strip().startswith("{") or result.output.strip().startswith("[")
  ```
  ```python
  result = {
      "run_id": run_id,
      "status": "not_implemented",
      ...
  }
  typer.echo(json_module.dumps(result, indent=2))
  raise typer.Exit(2)
  ```
  (`src/elspeth/cli.py:305`, `src/elspeth/cli.py:311`, `src/elspeth/cli.py:316`, `src/elspeth/cli.py:317`)

## Impact

- Tests can pass even when the CLI returns the wrong exit code, writes to the wrong stream, or emits malformed/non-contract JSON.
- Regressions in explain’s error handling or output format could slip through unnoticed, giving false confidence in CLI behavior.
- Permissive substring checks allow unrelated failures (e.g., missing imports) to be treated as success.

## Root Cause Hypothesis

- Placeholder behavior led to minimal, permissive assertions without aligning to the explicit CLI contract or stderr handling.
- No explicit test harness configuration for stderr mixing, so stdout-only checks were used as a shortcut.

## Recommended Fix

- Assert exact exit codes and expected content to match the CLI contract: `result.exit_code == 2` for `--json`/`--no-tui`, and `result.exit_code == 2` with a concrete “Missing option '--run'” message for missing run ID.
- Explicitly capture stderr (`CliRunner(mix_stderr=True)` or `result.stderr`) and assert the exact message for `--no-tui` instead of allowing `"not found"`.
- Parse JSON output with `json.loads(result.output)` and validate keys/values such as `status == "not_implemented"` and `run_id == "test-run"`.
- Priority justification: tightening assertions prevents false positives in CLI tests and ensures explain output contracts are enforced before Phase 4 lineage work.
