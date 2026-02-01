# Test Defect Report
## Summary
- CLI row-plugin tests configure a Landscape DB but never verify audit records (node_states, token_outcomes, artifacts).

## Severity
- Severity: major
- Priority: P1

## Category
- Missing Audit Trail Verification

## Evidence
- `tests/cli/test_run_with_row_plugins.py:60` sets the Landscape audit DB URL in settings.
- `tests/cli/test_run_with_row_plugins.py:148` reads the output CSV and asserts substrings, with no audit DB queries anywhere in the file.

```python
"landscape": {"url": f"sqlite:///{tmp_path}/audit.db"},
...
output_content = output_csv.read_text()
assert "alice" in output_content
```

## Impact
- Audit trail regressions (missing node_states, token_outcomes, artifacts, or wrong hashes/lineage) can ship unnoticed while these tests still pass, undermining the auditability standard.

## Root Cause Hypothesis
- Tests were scoped to data-flow output only, and audit-trail verification was deferred or assumed to be covered elsewhere.

## Recommended Fix
- After each `runner.invoke`, open the audit DB and assert audit tables directly using the Landscape APIs:
  - Use `LandscapeDB.from_url` + `LandscapeRecorder.list_runs()` to find the run and assert status.
  - Use `get_rows`, `get_tokens`, and `get_node_states_for_token` to verify `status`, `input_hash`, `output_hash`, and `error_json`.
  - Use `get_token_outcomes_for_row` to verify terminal outcome (e.g., `RowOutcome.COMPLETED`) and sink name.
  - Use `get_artifacts` to verify `artifact_type`, `content_hash`, and `path_or_uri`, and compare `content_hash` to the CSV file hash.
```python
db = LandscapeDB.from_url(f"sqlite:///{audit_db}")
recorder = LandscapeRecorder(db)
runs = recorder.list_runs()
assert len(runs) == 1
run_id = runs[0].run_id
rows = recorder.get_rows(run_id)
assert len(rows) == 3
artifacts = recorder.get_artifacts(run_id)
assert artifacts[0].content_hash == hashlib.sha256(output_csv.read_bytes()).hexdigest()
```
---
# Test Defect Report
## Summary
- Output validation relies on substring checks against raw CSV text, which can pass even when headers/rows are wrong or incomplete.

## Severity
- Severity: minor
- Priority: P2

## Category
- Weak Assertions

## Evidence
- `tests/cli/test_run_with_row_plugins.py:148` uses `read_text()` and substring assertions instead of structured CSV checks.
- `tests/cli/test_run_with_row_plugins.py:165` and `tests/cli/test_run_with_row_plugins.py:178` only check for header substrings, not exact column sets or per-row mappings.

```python
output_content = output_csv.read_text()
assert "full_name" in output_content
assert "test_score" in output_content
assert "alice" in output_content
```

## Impact
- Tests can pass if output has extra/duplicate columns, missing rows, or incorrect field mapping (e.g., `score` still under the old column), allowing transform regressions to slip.

## Root Cause Hypothesis
- Convenience assertions were used instead of parsing CSV, and there is no shared helper for structured output verification.

## Recommended Fix
- Parse the CSV and assert exact headers and rows; verify row count and absence of old column names.
```python
with output_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))
assert rows == [
    {"id": "1", "full_name": "alice", "test_score": "75"},
    {"id": "2", "full_name": "bob", "test_score": "45"},
    {"id": "3", "full_name": "carol", "test_score": "90"},
]
```
