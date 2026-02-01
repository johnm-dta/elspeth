# Test Defect Report

## Summary

- Contract tests only validate hash length/positivity and same-data determinism; they never verify `content_hash`/`size_bytes` match the actual file, so incorrect metadata can pass.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/contracts/sink_contracts/test_csv_sink_contract.py:203` shows only shape checks for hash/size:
  ```python
  result = sink.write(rows, ctx)
  assert len(result.content_hash) == 64
  assert result.size_bytes > 0
  ```
- `tests/contracts/sink_contracts/test_csv_sink_contract.py:68` only compares identical data between files:
  ```python
  result1 = sink1.write(rows, ctx)
  result2 = sink2.write(rows, ctx)
  assert result1.content_hash == result2.content_hash
  ```
- The implementation claims file-derived metadata (`src/elspeth/plugins/sinks/csv_sink.py:140`), but no test validates that contract against the on-disk file.

## Impact

- A regression where `CSVSink` hashes the wrong file, returns a constant hash, or reports the wrong size would still pass.
- Audit integrity guarantees would be untested, giving false confidence in critical metadata.

## Root Cause Hypothesis

- Tests were written as interface checks (shape/determinism) without validating correctness against the produced artifact.

## Recommended Fix

- Add assertions that recompute the hash from the written file and compare `size_bytes` to the file size, plus a negative case with different data.
- Example pattern:
  ```python
  result = sink.write(rows, ctx)
  sink.close()
  expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
  assert result.content_hash == expected_hash
  assert result.size_bytes == csv_path.stat().st_size
  ```
- Priority is P1 because incorrect hash/size undermines audit integrity.
---
# Test Defect Report

## Summary

- Property-based generators explicitly filter out commas, quotes, and newlines, so core CSV quoting/escaping behavior is untested.

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- `tests/contracts/sink_contracts/test_csv_sink_contract.py:176` filters out CSV-critical characters:
  ```python
  st.text(min_size=1, max_size=20).filter(lambda s: "\n" not in s and "," not in s and '"' not in s)
  ```
- `tests/contracts/sink_contracts/test_csv_sink_contract.py:215` repeats the same exclusion; no other tests introduce values requiring CSV quoting.

## Impact

- Bugs in CSV quoting/escaping or delimiter handling could slip through, corrupting outputs for real-world data that includes commas/quotes/newlines.

## Root Cause Hypothesis

- The generators were simplified to avoid CSV parsing complexity, leaving important cases untested.

## Recommended Fix

- Add a targeted test with values containing commas, quotes, and newlines; read back with `csv.DictReader` and assert round-trip integrity.
- Alternatively, extend the property-based generator to include these characters and assert CSV read-back equivalence.
- Priority is P2 because this is a common real-world data shape for CSVs.
