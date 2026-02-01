# Test Defect Report

## Summary

- Missing tests for `${VAR}` / `${VAR:-default}` interpolation in config loading, leaving `_expand_env_vars` behavior unverified in `tests/core/test_config.py`

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `src/elspeth/core/config.py:727` and `src/elspeth/core/config.py:1179` implement and invoke `${VAR}` expansion in `load_settings`.
```python
# src/elspeth/core/config.py
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")
...
raw_config = _expand_env_vars(raw_config)
```
- `tests/core/test_config.py:105` and `tests/core/test_config.py:131` show `TestLoadSettings` only exercising Dynaconf env overrides, not `${VAR}` interpolation.
```python
# tests/core/test_config.py
def test_load_with_env_override(...):
    ...
    monkeypatch.setenv("ELSPETH_DATASOURCE__PLUGIN", "json")
    settings = load_settings(config_file)
    assert settings.datasource.plugin == "json"
```

## Impact

- `${VAR}` expansion regressions (e.g., defaults not applied, nested values not expanded) can ship without detection, causing mis-resolved paths/credentials and incomplete audit configuration data.

## Root Cause Hypothesis

- Env-var interpolation was added as a helper in `load_settings` without dedicated tests; coverage focused on Dynaconf overrides and schema validation.

## Recommended Fix

- Add tests in `tests/core/test_config.py` that load YAML containing `${VAR}` and `${VAR:-default}` values (including nested dict/list cases) and assert resolved values.
- Example:
```python
def test_load_expands_env_vars(self, tmp_path, monkeypatch):
    from elspeth.core.config import load_settings

    monkeypatch.setenv("INPUT_PATH", "data/input.csv")
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: ${INPUT_PATH}
sinks:
  output:
    plugin: csv
output_sink: output
""")
    settings = load_settings(config_file)
    assert settings.datasource.options["path"] == "data/input.csv"

def test_load_expands_env_vars_with_default(self, tmp_path):
    from elspeth.core.config import load_settings

    config_file = tmp_path / "settings.yaml"
    config_file.write_text("""
datasource:
  plugin: csv
  options:
    path: ${MISSING_PATH:-fallback.csv}
sinks:
  output:
    plugin: csv
output_sink: output
""")
    settings = load_settings(config_file)
    assert settings.datasource.options["path"] == "fallback.csv"
```
---
# Test Defect Report

## Summary

- Tests use defensive `.get()` and redundant `isinstance()` checks on system outputs, weakening strictness and violating the no-defensive-programming policy

## Severity

- Severity: trivial
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/core/test_config.py:1765` uses `.get()` to access required keys instead of direct indexing.
```python
fingerprint = audit_config["datasource"]["options"].get("api_key_fingerprint")
assert fingerprint is not None
```
- `tests/core/test_config.py:2130` and `tests/core/test_config.py:2155` use `.get()` for required secret fields.
```python
assert result.get("api_key") == "sk-secret"
...
assert settings.datasource.options.get("api_key") == "sk-secret-key"
```
- `tests/core/test_config.py:810` and `tests/core/test_config.py:852` use `isinstance()` checks rather than direct access.
```python
assert isinstance(resolved, dict)
...
assert isinstance(json_str, str)
```

## Impact

- `.get()` suppresses immediate KeyError failures and can mask missing keys when assertions are incomplete; `isinstance()` checks add noise without increasing correctness. This reduces test strictness and conflicts with the repository’s anti-defensive rules.

## Root Cause Hypothesis

- Convenience-driven assertions were used without aligning to the project’s “no defensive patterns” testing policy.

## Recommended Fix

- Replace `.get()` with direct indexing and drop redundant `isinstance()` checks so missing keys crash immediately.
- Example:
```python
fingerprint = audit_config["datasource"]["options"]["api_key_fingerprint"]
assert len(fingerprint) == 64

assert result["api_key"] == "sk-secret"
assert settings.datasource.options["api_key"] == "sk-secret-key"

# Remove isinstance checks; direct access already proves dict structure
assert resolved["datasource"]["plugin"] == "csv"
```
