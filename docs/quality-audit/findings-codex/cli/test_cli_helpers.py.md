# Test Defect Report

## Summary

- `test_instantiate_plugins_from_config` only checks key presence and base-class types, so it doesn’t verify correct plugin selection or option propagation.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/cli/test_cli_helpers.py:45` and `tests/cli/test_cli_helpers.py:52` only assert presence/`isinstance` and never validate plugin identity or config values.
```python
# Verify structure
assert "source" in plugins
...
assert isinstance(plugins["sinks"]["output"], BaseSink)
```
- `src/elspeth/cli_helpers.py:32` and `src/elspeth/cli_helpers.py:51` show plugin constructors receive config options, but the test never checks those options are preserved.
```python
source = source_cls(dict(config.datasource.options))
...
sinks[sink_name] = sink_cls(dict(sink_config.options))
```

## Impact

- Regressions that mis-map plugin names or drop/mutate options (e.g., wrong path/schema) can pass, creating false confidence in CLI instantiation correctness.

## Root Cause Hypothesis

- The test was written as a smoke check for instantiation and never expanded to assert correctness of instantiated plugin identity/config.

## Recommended Fix

- Strengthen assertions in `tests/cli/test_cli_helpers.py` to validate plugin identity and option propagation.
```python
assert plugins["source"].name == "csv"
assert plugins["source"].config["path"] == "test.csv"
assert plugins["transforms"][0].name == "passthrough"
assert plugins["sinks"]["output"].name == "csv"
assert plugins["sinks"]["output"].config["path"] == "output.csv"
```
