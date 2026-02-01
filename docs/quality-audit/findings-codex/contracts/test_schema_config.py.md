Using skill: using-quality-engineering (contract testing guidance for schema config contract audit).

# Test Defect Report

## Summary

- SchemaConfig serialization/round-trip is untested: no coverage for `SchemaConfig.to_dict()` or the serialized dynamic form (`mode="dynamic", fields=None`), leaving audit-log schema snapshots and reloads unverified.

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/contracts/schema.py:195` and `src/elspeth/contracts/schema.py:251` implement the serialized-dynamic branch and audit-log serialization that are never exercised by tests.
```python
if config.get("mode") == "dynamic":
    return cls(mode=None, fields=None, is_dynamic=True)

def to_dict(self) -> dict[str, Any]:
    if self.is_dynamic:
        return {"mode": "dynamic", "fields": None}
```
- `tests/contracts/test_schema_config.py:86` only validates parsing for `fields="dynamic"` and never calls `to_dict` or provides `mode="dynamic"` inputs, so the serialization contract and round-trip path are uncovered.
```python
config = SchemaConfig.from_dict({"fields": "dynamic"})
assert config.is_dynamic is True
assert config.mode is None
```

## Impact

- Regressions in schema serialization could silently corrupt audit trail snapshots (wrong `mode`/`fields` representation) or break reload/round-trip of persisted config without any test signal.
- The audit trail is core to ELSPETH’s accountability guarantees; missing coverage here creates false confidence in audit log correctness.

## Root Cause Hypothesis

- Tests were added to cover YAML parsing issues and explicit field parsing, but serialization and round-trip behavior were overlooked because they are not part of the parsing flow.

## Recommended Fix

- Add tests in `tests/contracts/test_schema_config.py` that explicitly verify serialization and round-trip:
```python
config = SchemaConfig.from_dict({"fields": "dynamic"})
assert config.to_dict() == {"mode": "dynamic", "fields": None}

roundtrip = SchemaConfig.from_dict({"mode": "dynamic", "fields": None})
assert roundtrip.is_dynamic is True

config = SchemaConfig.from_dict({"mode": "strict", "fields": ["id: int", "score: float?"]})
assert config.to_dict()["fields"] == [
    {"name": "id", "type": "int", "required": True},
    {"name": "score", "type": "float", "required": False},
]
```
- Priority justification: these assertions protect the audit-log schema snapshot format and its ability to be reloaded, which is critical to ELSPETH’s auditability guarantees.
