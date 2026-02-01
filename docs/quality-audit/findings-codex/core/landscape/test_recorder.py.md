# Test Defect Report

## Summary

- Hash integrity for rows/node states/artifacts is only checked for non-NULL, not canonical correctness, so stable_hash or content_hash regressions would pass.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/core/landscape/test_recorder.py:400` only checks presence of the row hash:
```python
assert row.source_data_hash is not None
```
- `tests/core/landscape/test_recorder.py:666` only checks presence of the input hash:
```python
assert state.input_hash is not None
```
- `tests/core/landscape/test_recorder.py:706` only checks presence of the output hash:
```python
assert completed.output_hash is not None
```
- `tests/core/landscape/test_recorder.py:1317` omits artifact content/hash/type assertions:
```python
assert artifact.artifact_id is not None
assert artifact.path_or_uri == "/output/result.csv"
```
- `src/elspeth/core/landscape/recorder.py:749`, `src/elspeth/core/landscape/recorder.py:1050`, `src/elspeth/core/landscape/recorder.py:1163`, `src/elspeth/core/landscape/recorder.py:1701` show the recorder computes and stores these hashes, but tests never validate correctness:
```python
data_hash = stable_hash(data)
```

## Impact

- Hash correctness is the audit integrity anchor; a regression in canonicalization or hash computation could silently corrupt the audit trail while tests still pass.
- False confidence in determinism and provenance checks, especially for explain/export workflows that rely on hash stability.

## Root Cause Hypothesis

- Tests were written to confirm object creation rather than validate canonical hash determinism and artifact metadata integrity.

## Recommended Fix

- Add explicit stable_hash equality assertions for row, input, and output hashes in the relevant tests, and assert artifact metadata fields.
```python
from elspeth.core.canonical import stable_hash

expected_row_hash = stable_hash({"value": 42})
assert row.source_data_hash == expected_row_hash

expected_input_hash = stable_hash({"x": 1})
assert state.input_hash == expected_input_hash

expected_output_hash = stable_hash({"x": 1, "y": 2})
assert completed.output_hash == expected_output_hash

assert artifact.content_hash == "abc123"
assert artifact.artifact_type == "csv"
assert artifact.size_bytes == 1024
```
- Priority justification: audit hash correctness is core to ELSPETH’s legal traceability requirements.
---
# Test Defect Report

## Summary

- Fork lineage is not verified; `test_fork_token` asserts branch names and fork_group_id but never checks parent_token_id/ordinal linkage.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/core/landscape/test_recorder.py:430` only asserts branch names and fork_group_id; no parent relationship checks:
```python
assert child_tokens[0].branch_name == "stats"
assert child_tokens[1].branch_name == "classifier"
assert child_tokens[0].fork_group_id == child_tokens[1].fork_group_id
```
- `src/elspeth/core/landscape/recorder.py:875` records parent relationships for forks that are unverified by tests:
```python
token_parents_table.insert().values(
    token_id=child_id,
    parent_token_id=parent_token_id,
    ordinal=ordinal,
)
```

## Impact

- A regression that drops or corrupts fork lineage would break explainability and audit lineage for forks, yet current tests would still pass.
- Lineage gaps undermine the “no silent drops” audit standard.

## Root Cause Hypothesis

- Fork tests focus on fork_group_id/branch names and assume lineage is covered elsewhere, leaving fork-specific parent linkage unverified.

## Recommended Fix

- Extend `test_fork_token` (or add a new test) to assert token_parents entries for each child.
```python
parents_0 = recorder.get_token_parents(child_tokens[0].token_id)
assert len(parents_0) == 1
assert parents_0[0].parent_token_id == parent_token.token_id
assert parents_0[0].ordinal == 0

parents_1 = recorder.get_token_parents(child_tokens[1].token_id)
assert len(parents_1) == 1
assert parents_1[0].parent_token_id == parent_token.token_id
assert parents_1[0].ordinal == 1
```
- Priority justification: fork lineage is foundational to audit explainability and must be verified in recorder tests.
