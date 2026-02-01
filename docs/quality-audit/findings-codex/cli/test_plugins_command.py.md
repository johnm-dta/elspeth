# Test Defect Report

## Summary

- plugins list tests rely on broad stdout/exit-code checks instead of section-specific output and error assertions.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_plugins_command.py:103` and `tests/cli/test_plugins_command.py:104` assert only substrings in the entire stdout:
  ```python
  assert "csv" in result.stdout.lower()
  assert "json" in result.stdout.lower()
  ```
- `src/elspeth/cli.py:916` and `src/elspeth/cli.py:923` show output aggregates all plugin types into one stdout stream:
  ```python
  for ptype in types_to_show:
      ...
      typer.echo(f"  {plugin.name:20} - {plugin.description}")
  ```
- `src/elspeth/plugins/sinks/csv_sink.py:62` and `src/elspeth/plugins/sinks/json_sink.py:53` show sink plugin names include "csv"/"json", so the "sources" test can pass even if sources are missing:
  ```python
  name = "csv"
  ```
  ```python
  name = "json"
  ```
- `tests/cli/test_plugins_command.py:139` only checks `exit_code != 0` for invalid type, so unrelated crashes pass:
  ```python
  assert result.exit_code != 0
  ```

## Impact

- Missing or misrouted sources could go undetected because sinks share the same names.
- Error-path regressions could be masked by any nonzero exit, creating false confidence in CLI validation.
- Output formatting/sectioning bugs in `plugins list` are not reliably caught.

## Root Cause Hypothesis

- Tests favor minimal string checks over parsing the structured CLI output and error streams.

## Recommended Fix

- Parse stdout by section headers (e.g., `SOURCES:`/`SINKS:`) and assert plugin names appear under the correct section, not just anywhere in the output.
- Assert the exact invalid-type error message and expected exit code.
- Example:
  ```python
  result = runner.invoke(app, ["plugins", "list"])
  assert result.exit_code == 0
  text = result.stdout.lower()
  assert "\nsources:\n" in text
  assert "\n  csv" in text
  assert "\n  json" in text

  result = runner.invoke(app, ["plugins", "list", "--type", "invalid"])
  assert result.exit_code == 1
  err = (result.stderr or result.output).lower()
  assert "invalid type" in err
  assert "valid types" in err
  ```
---
# Test Defect Report

## Summary

- plugins list tests do not cover transform output or the `--type transform` filter, leaving a supported CLI path unverified.

## Severity

- Severity: trivial
- Priority: P3

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/cli.py:890` and `src/elspeth/cli.py:901` define transforms as a supported plugin type and `--type` option value:
  ```python
  "transform": [PluginInfo(name=cls.name, description=get_plugin_description(cls)) for cls in manager.get_transforms()],
  ```
  ```python
  help="Filter by plugin type (source, transform, sink).",
  ```
- `tests/cli/test_plugins_command.py:120` and `tests/cli/test_plugins_command.py:121` only assert "source"/"sink" sections, and `tests/cli/test_plugins_command.py:128` only filters by `--type source`:
  ```python
  assert "source" in result.stdout.lower()
  assert "sink" in result.stdout.lower()
  ...
  result = runner.invoke(app, ["plugins", "list", "--type", "source"])
  ```

## Impact

- A regression that omits transforms from the list or breaks `--type transform` filtering would not be detected.
- CLI contract coverage is incomplete for a documented option.

## Root Cause Hypothesis

- Initial tests focused on sources/sinks and didn't expand to the full set of plugin types.

## Recommended Fix

- Add tests that assert the `TRANSFORMS:` section exists and that a known transform (or one obtained from the plugin manager) is listed.
- Add a filter test for `--type transform` (and optionally `--type sink`) to ensure output is scoped correctly.
- Example:
  ```python
  result = runner.invoke(app, ["plugins", "list", "--type", "transform"])
  assert result.exit_code == 0
  assert "TRANSFORMS:" in result.stdout
  assert "passthrough" in result.stdout.lower()
  ```
