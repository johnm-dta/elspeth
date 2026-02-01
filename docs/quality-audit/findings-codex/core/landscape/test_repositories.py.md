# Test Defect Report

## Summary

- RunRepository tests never assert crash on invalid `export_status` (including falsy invalid values), leaving Tier 1 corruption behavior unverified.

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Tier 1 Corruption Tests

## Evidence

- `tests/core/landscape/test_repositories.py:82-149` only covers valid and NULL `export_status`, with no invalid-value test.
```python
def test_load_converts_export_status_to_enum(self) -> None:
    ...
    export_status="completed",  # String from DB
...
def test_load_handles_null_export_status(self) -> None:
    ...
    export_status=None,  # NULL in DB
```
- `tests/core/landscape/test_repositories.py:151-182` has an invalid test for `status` but none for `export_status`.
```python
def test_load_crashes_on_invalid_status(self) -> None:
    ...
    status="invalid_garbage",  # Invalid!
```
- `src/elspeth/core/landscape/repositories.py:52-55` shows a falsy branch that would treat `""` as None and skip validation.
```python
export_status=ExportStatus(row.export_status) if row.export_status else None,
```

## Impact

- Invalid `export_status` values (including empty strings) could be silently treated as NULL, violating the Data Manifesto’s crash-on-corruption rule.
- Test suite provides false confidence that all enum fields are validated on read.

## Root Cause Hypothesis

- Tests focused on required `status` validation and overlooked the optional enum field, likely via copy-paste omissions.

## Recommended Fix

- Add a test in `tests/core/landscape/test_repositories.py` for invalid `export_status` (e.g., `"invalid_status"`) expecting `ValueError`.
- Add a test for falsy invalid values (e.g., `""`) to ensure they crash rather than defaulting to `None`.
- If these tests fail, update repository logic to treat empty strings as invalid (e.g., `if row.export_status is not None`).
---
# Test Defect Report

## Summary

- NodeRepository tests do not cover schema metadata fields (`schema_mode`, `schema_fields`), so schema audit data can be dropped without detection.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/core/landscape/test_repositories.py:191-223` defines `NodeRow` without `schema_mode`/`schema_fields` and asserts only enum conversions.
```python
@dataclass
class NodeRow:
    ...
    schema_hash: str | None = None
    sequence_in_pipeline: int | None = None

...
assert node.node_type == NodeType.SOURCE
assert node.determinism == Determinism.IO_READ
```
- `src/elspeth/contracts/audit.py:65-70` shows `Node` contract includes schema metadata fields that should be preserved.
```python
schema_mode: str | None = None
schema_fields: list[dict[str, object]] | None = None
```
- `src/elspeth/core/landscape/repositories.py:74-86` does not pass schema metadata into `Node`.
```python
return Node(
    ...
    schema_hash=row.schema_hash,
    sequence_in_pipeline=row.sequence_in_pipeline,
)
```

## Impact

- Schema audit metadata can be lost when loading nodes through the repository layer, weakening auditability and schema validation traceability.
- Tests won’t detect regressions or omissions involving schema metadata.

## Root Cause Hypothesis

- Schema metadata fields were added after the original repository tests and never incorporated.

## Recommended Fix

- Extend `tests/core/landscape/test_repositories.py` to include `schema_mode` and `schema_fields` in a `NodeRow` and assert they are preserved on `load()`.
- Add a negative test for malformed schema metadata if conversion is expected.
- If these tests fail, update `NodeRepository.load` to include `schema_mode` and `schema_fields` from the row (and parse JSON if necessary).
