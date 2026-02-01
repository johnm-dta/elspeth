# Test Defect Report

## Summary

- Substring-based assertions (`"valid"`/`"error"`) allow false positives and do not verify the intended error/success messaging

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_plugin_errors.py:196` and `tests/cli/test_plugin_errors.py:266` use substring checks that also match `"invalid"`, so a wrong success message could still pass.
```python
# tests/cli/test_plugin_errors.py:196-200
result = runner.invoke(app, ["validate", "--settings", str(config_file)])
assert result.exit_code == 0
assert "valid" in result.output.lower()
```
```python
# tests/cli/test_plugin_errors.py:266-270
result = runner.invoke(app, ["validate", "--settings", str(config_file)])
assert result.exit_code == 0
assert "valid" in result.output.lower()
```
- `tests/cli/test_plugin_errors.py:99` asserts only `"error"` for a test that claims to verify clear plugin init errors; it does not check that the plugin init failure (missing `path`/`on_validation_failure`) is surfaced.
```python
# tests/cli/test_plugin_errors.py:99-104
result = runner.invoke(app, ["validate", "--settings", str(config_file)])
assert result.exit_code != 0
assert "error" in result.output.lower()
```

## Impact

- Tests can pass when the CLI prints `"invalid"` or any unrelated error string, masking regressions in user-visible validation messaging
- Plugin initialization failures could be misreported (or replaced by different errors) without test detection
- Creates false confidence that the CLI output is clear and correct

## Root Cause Hypothesis

- Assertions were kept intentionally loose to avoid brittleness in CLI messaging, but the substring checks are too permissive for the stated test intent
- Pattern of minimal string checks suggests these were added as scaffolding and never tightened

## Recommended Fix

- Replace `"valid"` substring checks with a specific, unambiguous phrase, e.g. `"pipeline configuration valid"` or a word-boundary regex, and optionally assert `"invalid"` is not present.
- Strengthen `test_plugin_initialization_error` to assert the error mentions the plugin/config issue (e.g., `"Error instantiating plugins"` plus `"CSVSourceConfig"` and missing field names like `"path"` or `"on_validation_failure"`).
- Priority justification: improves correctness of CLI-facing tests and reduces false positives on core validation behavior without major refactors.
