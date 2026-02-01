# Test Defect Report

## Summary

- Tests only assert 1–2 fields per model, leaving most required fields/defaults unverified and making regressions in model field mapping or defaults easy to miss.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/core/landscape/test_models.py:14` constructs a `Run` with multiple fields but only asserts `run_id` and `status` (`tests/core/landscape/test_models.py:22`, `tests/core/landscape/test_models.py:23`), leaving `config_hash`, `settings_json`, `canonical_version`, and `started_at` unchecked.
  ```python
  run = Run(
      run_id="run-001",
      started_at=datetime.now(UTC),
      config_hash="abc123",
      settings_json="{}",
      canonical_version="sha256-rfc8785-v1",
      status=RunStatus.RUNNING,
  )
  assert run.run_id == "run-001"
  assert run.status == RunStatus.RUNNING
  ```
- `tests/core/landscape/test_models.py:33` builds a `Node` with many fields but only asserts `node_type` and `determinism` (`tests/core/landscape/test_models.py:44`, `tests/core/landscape/test_models.py:45`), leaving `plugin_name`, `plugin_version`, `config_hash`, `config_json`, and `registered_at` unvalidated.
- `tests/core/landscape/test_models.py:54` and `tests/core/landscape/test_models.py:71` show `Row`/`Token` created with multiple required fields, but assertions only check `row_index` (`tests/core/landscape/test_models.py:62`) and `token_id` (`tests/core/landscape/test_models.py:76`), leaving `run_id`, `source_node_id`, `source_data_hash`, and `created_at` unverified.

## Impact

- These tests can pass even if constructors silently mutate or mis-assign critical audit fields like `config_hash`, `settings_json`, or `created_at`.
- Regressions in required fields/defaults for core audit models can slip through without detection, giving false confidence in model integrity.
- Weak coverage discourages catching Tier 1 data integrity issues early in the test suite.

## Root Cause Hypothesis

- The file appears to be early-stage smoke tests that were never expanded as the model schema grew.
- Reliance on dataclass behavior led to minimal assertions rather than explicit field validation.

## Recommended Fix

- Expand each test to assert all constructor inputs and key defaults (`None` for optional fields) to ensure full field mapping is verified.
- Add negative tests for missing required fields (expect `TypeError`) and invalid enum values where applicable; use fixtures to reduce repetition.
- Priority justification: P2 because these are core audit models; stronger assertions reduce the risk of silent schema regressions.
