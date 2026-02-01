# Test Defect Report

## Summary

- Determinism contract tests do not compare hashes across runs or data, so they never actually verify determinism despite claiming to

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/contracts/sink_contracts/test_sink_protocol.py:259` shows a determinism test that only asserts hash length after a single write, with no comparison to a second run or different data.
- `tests/contracts/sink_contracts/test_sink_protocol.py:282` shows a “different data” test that only asserts non-null length, not a different hash.
- `tests/contracts/sink_contracts/test_sink_protocol.py:7` states the contract guarantee “Same data MUST produce same content_hash,” which is not actually enforced.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py:259
result = sink.write(sample_rows, ctx)
first_hash = result.content_hash
...
assert len(first_hash) == 64, "Content hash must be SHA-256 hex"

# tests/contracts/sink_contracts/test_sink_protocol.py:282
result = sink.write(sample_rows, ctx)
assert result.content_hash is not None
assert len(result.content_hash) == 64
```

## Impact

- A sink could return a constant or random hash and still pass these tests.
- Determinism regressions (critical for audit verification) would not be detected.
- Creates false confidence in audit integrity for sinks that claim determinism.

## Root Cause Hypothesis

- The base contract tests avoid creating a fresh sink instance and were reduced to placeholder assertions rather than enforcing the contract.

## Recommended Fix

- Require a `sink_factory` or `fresh_sink` fixture and compare hashes across two fresh runs; also compare against a modified data set.
- If determinism is not required for a given sink, explicitly skip based on `sink.determinism` rather than weakening the assertion.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py
@pytest.fixture
@abstractmethod
def sink_factory(self) -> Callable[[], SinkProtocol]:
    ...

def test_same_data_same_hash(self, sink_factory, sample_rows, ctx):
    first = sink_factory().write(sample_rows, ctx).content_hash
    second = sink_factory().write(sample_rows, ctx).content_hash
    assert first == second
```
---
# Test Defect Report

## Summary

- Lifecycle hook tests skip missing methods via `hasattr`, allowing SinkProtocol violations to pass silently

## Severity

- Severity: minor
- Priority: P2

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/contracts/sink_contracts/test_sink_protocol.py:232` skips `on_start` if missing.
- `tests/contracts/sink_contracts/test_sink_protocol.py:241` skips `on_complete` if missing.
- `src/elspeth/plugins/protocols.py:461` includes `on_start`/`on_complete` in `SinkProtocol`, so missing methods should fail the contract.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py:232
if hasattr(sink, "on_start"):
    sink.on_start(ctx)

# tests/contracts/sink_contracts/test_sink_protocol.py:241
if hasattr(sink, "on_complete"):
    sink.write(sample_rows, ctx)
    sink.on_complete(ctx)
```

## Impact

- A sink missing required lifecycle hooks would pass contract tests.
- Engine code that assumes these hooks exist could raise `AttributeError` at runtime.
- Normalizes a prohibited defensive pattern (skipping required interface checks).

## Root Cause Hypothesis

- Hooks were treated as “optional” in tests, leading to defensive checks instead of enforcing the protocol.

## Recommended Fix

- Remove the `hasattr` guard and call hooks directly; missing methods should fail the test.
- If hooks are truly optional, update the contract text and assert explicit absence as a supported behavior.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py
def test_on_start_does_not_raise(self, sink, ctx):
    sink.on_start(ctx)

def test_on_complete_does_not_raise(self, sink, sample_rows, ctx):
    sink.write(sample_rows, ctx)
    sink.on_complete(ctx)
```
---
# Test Defect Report

## Summary

- The input schema contract only checks for any `type`, not that it is a `PluginSchema` subclass as required by the protocol

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/sink_contracts/test_sink_protocol.py:86` only asserts `isinstance(..., type)` for `input_schema`.
- `src/elspeth/plugins/protocols.py:405` specifies `input_schema: type["PluginSchema"]`.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py:86
assert hasattr(sink, "input_schema")
assert isinstance(sink.input_schema, type)

# src/elspeth/plugins/protocols.py:405
input_schema: type["PluginSchema"]
```

## Impact

- Sinks can pass the contract while declaring `input_schema = dict` or `int`, which breaks schema validation and DAG compatibility checks later.
- Reduces the contract test’s ability to enforce correct plugin interfaces.

## Root Cause Hypothesis

- The test was written with a generic type check and never updated to reflect the `PluginSchema` requirement.

## Recommended Fix

- Assert `issubclass(sink.input_schema, PluginSchema)` and fail fast for invalid types.

```python
# tests/contracts/sink_contracts/test_sink_protocol.py
from elspeth.contracts import PluginSchema

assert isinstance(sink.input_schema, type)
assert issubclass(sink.input_schema, PluginSchema)
```
