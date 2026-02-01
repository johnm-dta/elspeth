# Test Defect Report

## Summary

- KeywordFilter contract tests configure an error-producing transform without `on_error`, so the contract suite does not enforce the required error-routing configuration.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/contracts/transform_contracts/test_keyword_filter_contract.py:30` configures `KeywordFilter` with `blocked_patterns` but omits `on_error` (same pattern in the error contract setup).
```python
# tests/contracts/transform_contracts/test_keyword_filter_contract.py:30-35
return KeywordFilter(
    {
        "fields": ["content"],
        "blocked_patterns": [r"\btest\b"],
        "schema": {"fields": "dynamic"},
    }
)
```
- `tests/contracts/transform_contracts/test_keyword_filter_contract.py:50` does the same for the error-path fixture.
```python
# tests/contracts/transform_contracts/test_keyword_filter_contract.py:50-55
return KeywordFilter(
    {
        "fields": ["content"],
        "blocked_patterns": [r"\bblocked\b"],
        "schema": {"fields": "dynamic"},
    }
)
```
- The transform contract explicitly requires error-routing configuration when errors can occur: `src/elspeth/plugins/transforms/keyword_filter.py:46`–`54` and `src/elspeth/plugins/protocols.py:129`–`132`.
```python
# src/elspeth/plugins/transforms/keyword_filter.py:46-54
# ... Rows with matches are routed to the on_error sink ...
# ... on_error: Sink for blocked rows (required when patterns might match)

# src/elspeth/plugins/protocols.py:129-132
# Transforms that can return TransformResult.error() must set _on_error
# ... If _on_error is None and the transform returns an error, the executor raises RuntimeError.
```

## Impact

- Tests pass with a configuration that would raise `RuntimeError` in the executor when blocked content is encountered, so the suite can miss a critical runtime failure in real pipelines.
- Provides false confidence that the KeywordFilter contract is satisfied while violating the error-routing requirement.

## Root Cause Hypothesis

- Contract fixtures favor minimal config and do not encode the protocol requirement that error-capable transforms must set `_on_error`.
- The shared contract base does not enforce error-routing configuration, so each transform contract must supply it explicitly.

## Recommended Fix

- Update `tests/contracts/transform_contracts/test_keyword_filter_contract.py` to include a valid `on_error` value in both `transform` fixtures (e.g., `"on_error": "quarantine_sink"`).
- Add a test in `TestKeywordFilterErrorContract` that asserts `transform._on_error` is not `None` for error-producing transforms to enforce the protocol contract locally.
- Priority justification: this prevents a pipeline-level crash from slipping through contract tests.
