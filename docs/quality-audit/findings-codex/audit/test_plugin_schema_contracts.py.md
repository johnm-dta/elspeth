# Test Defect Report

## Summary

- Schema contract tests silently skip plugins that raise init/config errors, so those plugins' schema assignments are never validated.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/audit/test_plugin_schema_contracts.py:20`
  ```python
  try:
      instance = plugin_cls({"path": "test.csv", "schema": {"fields": "dynamic"}, "on_validation_failure": "quarantine"})
  except (TypeError, PluginConfigError):
      # Some sources may require different config - skip validation
      continue
  ```
- `tests/audit/test_plugin_schema_contracts.py:41` and `tests/audit/test_plugin_schema_contracts.py:60` repeat the same skip pattern for transforms and sinks.

## Impact

- Built-in plugins with additional required config (e.g., Azure sources/sinks, database sink) are untested for schema contract compliance.
- Any regression that raises `TypeError`/`PluginConfigError` during init is hidden, producing false confidence in schema enforcement.

## Root Cause Hypothesis

- A single "minimal config" was applied to all plugins and exceptions were swallowed to avoid per-plugin setup, sacrificing contract coverage.

## Recommended Fix

- Provide explicit minimal configs per plugin name and fail the test if a plugin cannot be instantiated; remove the `try/except ... continue`.
- If a plugin truly cannot be exercised in unit tests, use `pytest.skip` with a specific reason so the gap is visible.
- Example pattern:
  ```python
  config = MINIMAL_CONFIGS[plugin_cls.name]
  instance = plugin_cls(config)  # let failures surface
  ```
---
# Test Defect Report

## Summary

- Schema checks rely on `hasattr`, a prohibited defensive pattern that does not validate schema behavior or type.

## Severity

- Severity: major
- Priority: P1

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/audit/test_plugin_schema_contracts.py:26`
  ```python
  assert hasattr(instance, "output_schema"), f"Source {plugin_cls.name} missing output_schema attribute"
  ```
- `tests/audit/test_plugin_schema_contracts.py:46`, `tests/audit/test_plugin_schema_contracts.py:48`, `tests/audit/test_plugin_schema_contracts.py:66`, and `tests/audit/test_plugin_schema_contracts.py:86` repeat the same `hasattr` pattern for transforms, sinks, and the I/O check.
- `tests/audit/test_plugin_schema_contracts.py:28` asserts schema can be None for dynamic, but dynamic schemas are always real `PluginSchema` classes (see `src/elspeth/plugins/schema_factory.py:70`).

## Impact

- A plugin can set `output_schema`/`input_schema` to `None` or a non-`PluginSchema` object and still pass, undermining schema contract enforcement.
- The tests violate the codebase "no defensive patterns" rule, masking interface violations instead of letting them crash.

## Root Cause Hypothesis

- The tests were written as runtime existence checks rather than behavioral schema validations, contrary to CLAUDE.md guidance.

## Recommended Fix

- Replace `hasattr` with direct attribute usage and behavioral validation (e.g., call `model_validate` on a representative row) so missing/wrong schemas raise immediately.
- Example pattern for dynamic schemas:
  ```python
  instance.output_schema.model_validate({"any": "row"})
  instance.input_schema.model_validate({"any": "row"})
  ```
- Remove the "schema can be None" assumption; dynamic schemas are still `PluginSchema` subclasses.
