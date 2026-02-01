# Test Defect Report

## Summary

- Lifecycle hook contract tests skip calling hooks when they are missing, so a source without `on_start`/`on_complete` can pass contract tests even though the engine calls these hooks unconditionally.

## Severity

- Severity: major
- Priority: P1

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/source_contracts/test_source_protocol.py:151` skips `on_start` if the attribute is missing:
```python
if hasattr(source, "on_start"):
    source.on_start(ctx)
```
- `tests/contracts/source_contracts/test_source_protocol.py:160` skips `on_complete` if the attribute is missing:
```python
if hasattr(source, "on_complete"):
    list(source.load(ctx))
    source.on_complete(ctx)
```
- `src/elspeth/engine/orchestrator.py:783` and `src/elspeth/engine/orchestrator.py:1139` call the hooks unconditionally in production:
```python
config.source.on_start(ctx)
...
config.source.on_complete(ctx)
```

## Impact

- A source missing lifecycle hooks passes tests but will raise `AttributeError` at runtime when the orchestrator calls the hooks.
- Contract tests give false confidence about lifecycle compliance and compatibility with the engine.

## Root Cause Hypothesis

- Tests treat lifecycle hooks as optional and use defensive guards, diverging from engine behavior and the “system-owned code must crash on bugs” policy.

## Recommended Fix

- In `tests/contracts/source_contracts/test_source_protocol.py`, remove the `hasattr` guards and call `source.on_start(ctx)` and `source.on_complete(ctx)` unconditionally so missing hooks fail the contract tests.
- Optional enhancement: add an explicit assertion that the attributes exist and are callable before invoking, but avoid skipping the call.
---
# Test Defect Report

## Summary

- The `output_schema` contract test only checks that it is a type, not that it is a `PluginSchema` subclass, allowing invalid schemas to pass.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/source_contracts/test_source_protocol.py:78` only validates that `output_schema` is a class:
```python
assert isinstance(source.output_schema, type)
```
- `src/elspeth/plugins/protocols.py:66` specifies `output_schema: type["PluginSchema"]`, which requires a `PluginSchema` subclass.
- `src/elspeth/contracts/data.py:28` defines the `PluginSchema` base class expected by the protocol.

## Impact

- A source can set `output_schema` to an arbitrary class (e.g., `dict`) and still pass contract tests, causing downstream schema validation failures or undefined behavior.
- Tests do not enforce a core schema contract, reducing confidence in plugin correctness.

## Root Cause Hypothesis

- The assertion focuses on “is a class” and overlooks the stronger requirement that it be a `PluginSchema` subclass.

## Recommended Fix

- In `tests/contracts/source_contracts/test_source_protocol.py`, import `PluginSchema` and assert subclassing:
```python
assert isinstance(source.output_schema, type)
assert issubclass(source.output_schema, PluginSchema)
```
- Keep the existing type check to provide a clearer failure message if `output_schema` is not a class.
