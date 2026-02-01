# Test Defect Report

## Summary

- Dynamic schema tests call structural validation and duplicate detection logic, so schema-compatibility behavior for dynamic schemas is never exercised.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/test_dag.py:2094` and `tests/core/test_dag.py:2124` call `graph.validate()` in the dynamic schema tests, which only checks structure:
```python
graph.add_edge("source", "sink", label="continue")
# Should NOT raise - validation is skipped for dynamic schemas
graph.validate()
```
- `src/elspeth/core/dag.py:151` documents that `validate()` does not check schema compatibility, while schema checks are implemented in `src/elspeth/core/dag.py:667`:
```python
def validate(self) -> None:
    ...
    Does NOT check schema compatibility - plugins validate their own
    schemas during construction.
```
- `tests/core/test_dag.py:2158` defines a local `is_dynamic_schema` helper instead of exercising production logic:
```python
def is_dynamic_schema(schema: type | None) -> bool:
    if schema is None:
        return True
    return len(schema.model_fields) == 0 and schema.model_config.get("extra") == "allow"
```

## Impact

- Dynamic schema handling in `validate_edge_compatibility()` can regress without any test failing.
- Pipelines using dynamic schemas could start rejecting valid edges or skip required checks with no detection.
- The current tests give false confidence that dynamic schema validation is covered.

## Root Cause Hypothesis

- Confusion between structural validation (`validate`) and schema validation (`validate_edge_compatibility`) led to tests calling the wrong method.
- Avoidance of private methods resulted in duplicating logic in tests instead of exercising production behavior.

## Recommended Fix

- Replace `graph.validate()` with `graph.validate_edge_compatibility()` in `TestDynamicSchemaDetection` so schema logic is exercised.
- Remove the local `is_dynamic_schema` helper; instead, assert behavior by running `validate_edge_compatibility()` on dynamic vs explicit schema edges and add one explicit mismatch case that must raise.
- Priority justification: dynamic schemas are common in configs; schema-validation regressions would affect core DAG correctness and should be caught at test time.
