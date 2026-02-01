# Test Quality Review: test_csv_sink.py

## Summary
The test suite has good coverage of basic CSVSink functionality and includes important audit integrity tests (content hashing, cumulative hash tracking). However, critical gaps exist: no integration tests verifying artifact registration in the Landscape database, no tests for file I/O failure scenarios, and no tests validating the sink's contract with the engine regarding when artifacts should be recorded.

## Poorly Constructed Tests

### Test: test_custom_delimiter (line 80)
**Issue**: Assertion logic is broken and doesn't actually test delimiter usage
**Evidence**:
```python
assert ";" in content
assert "," not in content.replace(",", "")  # Logic error: always true!
```
The second assertion removes all commas then checks if commas exist - this is always True regardless of delimiter. The test passes even if delimiter is ignored.

**Fix**: Replace with proper assertion:
```python
assert ";" in content
assert "," not in content.split("\n")[0]  # Check header row only
# OR use csv.reader to verify delimiter
```
**Priority**: P2

### Test: test_explicit_schema_creates_all_headers_including_optional (line 254)
**Issue**: Test doesn't verify the actual bug it claims to test - missing assertion on first row header presence
**Evidence**: The test verifies 'score' is in final headers and writes succeed, but doesn't verify that the CSV was created with correct headers from the start. A buggy implementation could recreate the file with new headers on the second write and this test would still pass.

**Fix**: Add assertion after first write:
```python
sink.write([{"id": 1}], ctx)
sink.flush()  # Ensure written

# Verify headers include optional field even though first row didn't have it
with open(output_file) as f:
    reader = csv.DictReader(f)
    assert reader.fieldnames is not None
    assert "score" in reader.fieldnames  # Header should exist from schema

sink.write([{"id": 2, "score": 1.5}], ctx)  # Should not fail
```
**Priority**: P1

## Misclassified Tests

### Test: test_implements_protocol (line 24)
**Issue**: This is a type-checking concern, not a unit test concern
**Evidence**: `isinstance(sink, SinkProtocol)` verifies runtime type conformance, but protocols are structural (duck typing). This test adds no value - if the sink didn't implement the protocol, other tests would fail.

**Fix**: Delete this test. Protocol conformance is better verified by mypy type checking, not runtime tests.
**Priority**: P3

### Test: test_has_required_attributes (line 31)
**Issue**: Testing implementation details rather than behavior
**Evidence**: `hasattr(sink, "input_schema")` checks for attribute existence, not correct behavior. If `input_schema` exists but returns wrong schema, test passes.

**Fix**: Delete or replace with behavioral test:
```python
def test_schema_validation_when_enabled():
    """When validate_input=True, sink rejects rows violating schema."""
    sink = CSVSink({
        "path": str(output_file),
        "schema": {"mode": "strict", "fields": ["id: int"]},
        "validate_input": True,
    })

    # Wrong type should crash (upstream bug per CLAUDE.md)
    with pytest.raises(ValidationError):
        sink.write([{"id": "not_an_int"}], ctx)
```
**Priority**: P2

## Infrastructure Gaps

### Gap: No integration tests for artifact recording
**Issue**: Tests never verify that artifacts are recorded in the Landscape database
**Evidence**: CLAUDE.md states "Sink output - Final artifacts with content hashes" is non-negotiable for auditability. No test verifies `recorder.register_artifact()` is called or that artifacts table contains correct entries.

**Fix**: Add integration test:
```python
def test_artifacts_recorded_in_landscape(tmp_path, test_db):
    """Sink writes must be recorded in artifacts table with content hash."""
    from elspeth.core.landscape.recorder import LandscapeRecorder

    output_file = tmp_path / "output.csv"
    recorder = LandscapeRecorder(test_db)

    # Create run, node, token, node_state (minimal fixture)
    run_id = "test-run"
    recorder.register_run(run_id, ...)
    node_id = recorder.register_node(run_id, "csv", "sink", ...)
    token_id = recorder.register_token(...)
    state_id = recorder.open_node_state(token_id, node_id, ...)

    sink = CSVSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})
    ctx = PluginContext(run_id=run_id, config={})

    artifact_desc = sink.write([{"id": "1"}], ctx)

    # THIS IS WHAT'S MISSING: Integration with recorder
    artifact = recorder.register_artifact(
        run_id=run_id,
        state_id=state_id,
        sink_node_id=node_id,
        artifact_type=artifact_desc.artifact_type,
        path=artifact_desc.path_or_uri,
        content_hash=artifact_desc.content_hash,
        size_bytes=artifact_desc.size_bytes,
    )

    # Verify artifact in database
    artifacts = test_db.get_artifacts_for_run(run_id)
    assert len(artifacts) == 1
    assert artifacts[0].content_hash == artifact_desc.content_hash
    assert artifacts[0].sink_node_id == node_id

    sink.close()
```
**Priority**: P0 - This is core audit integrity

### Gap: No tests for file I/O failure scenarios
**Issue**: No tests verify behavior when file operations fail (disk full, permission denied, I/O error)
**Evidence**: CLAUDE.md Plugin Ownership section: "Plugin method throws exception → CRASH - bug in our code". Need to verify that I/O errors propagate correctly and don't get silently swallowed.

**Fix**: Add failure scenario tests:
```python
def test_write_fails_on_permission_denied(tmp_path):
    """File permission errors propagate (don't hide system issues)."""
    output_file = tmp_path / "readonly.csv"
    output_file.touch()
    output_file.chmod(0o444)  # Read-only

    sink = CSVSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

    # Should raise PermissionError, not return success
    with pytest.raises(PermissionError):
        sink.write([{"id": "1"}], ctx)

def test_hash_computation_fails_on_corrupted_file(tmp_path, monkeypatch):
    """Hash computation errors propagate (audit integrity)."""
    output_file = tmp_path / "output.csv"
    sink = CSVSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

    # Simulate file corruption between write and hash
    def mock_open(*args, **kwargs):
        raise OSError("I/O error")

    monkeypatch.setattr("builtins.open", mock_open)

    # Should crash on hash computation failure
    with pytest.raises(OSError):
        sink.write([{"id": "1"}], ctx)
```
**Priority**: P1

### Gap: No test for fsync durability guarantee
**Issue**: No test verifies that `flush()` calls `os.fsync()` for crash durability
**Evidence**: CSVSink.flush() docstring states "CRITICAL: Ensures data survives process crash and power loss" but no test verifies fsync is called. This is vital for checkpoint integrity (Bug #2 context).

**Fix**: Add verification test:
```python
def test_flush_calls_fsync_for_durability(tmp_path, monkeypatch):
    """flush() must call os.fsync() to guarantee durability."""
    output_file = tmp_path / "output.csv"
    sink = CSVSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

    sink.write([{"id": "1"}], ctx)

    fsync_called = False
    original_fsync = os.fsync

    def mock_fsync(fd):
        nonlocal fsync_called
        fsync_called = True
        return original_fsync(fd)

    monkeypatch.setattr("os.fsync", mock_fsync)
    sink.flush()

    assert fsync_called, "flush() must call os.fsync() for crash durability"
    sink.close()
```
**Priority**: P0 - Critical for checkpoint safety

### Gap: No test for empty batch with existing headers
**Issue**: Test `test_batch_write_empty_list` only tests empty file case, not empty batch appended to existing file
**Evidence**: What happens if first write has data, second write is empty list? Does it return correct cumulative hash?

**Fix**: Add test:
```python
def test_empty_batch_after_data_returns_cumulative_hash(tmp_path, ctx):
    """Empty batch after data returns hash of existing file, not empty hash."""
    output_file = tmp_path / "output.csv"
    sink = CSVSink({"path": str(output_file), "schema": DYNAMIC_SCHEMA})

    # First write with data
    artifact1 = sink.write([{"id": "1"}], ctx)
    hash_after_data = artifact1.content_hash

    # Second write with empty batch
    artifact2 = sink.write([], ctx)

    # Should return hash of existing file, not empty hash
    assert artifact2.content_hash == hash_after_data
    assert artifact2.size_bytes == artifact1.size_bytes

    sink.close()
```
**Priority**: P2

### Gap: Missing fixture for PluginContext
**Issue**: Every test creates `PluginContext(run_id="test-run", config={})` manually - no shared fixture
**Evidence**: Lines 22, 47, 67, 87, 113, etc. - repeated context creation is maintenance burden

**Fix**: Already exists (line 20-22), but tests don't use it consistently. Audit all tests to use `ctx` fixture.
**Priority**: P3

### Gap: No test for Determinism enum value
**Issue**: Test `test_has_determinism` only checks equality, not enum type
**Evidence**: Line 208 asserts `sink.determinism == Determinism.IO_WRITE` but doesn't verify it's the actual enum, not string

**Fix**: Add type assertion:
```python
def test_has_determinism():
    """CSVSink has determinism attribute set to IO_WRITE enum."""
    from elspeth.contracts import Determinism
    from elspeth.plugins.sinks.csv_sink import CSVSink

    sink = CSVSink({"path": "/tmp/test.csv", "schema": DYNAMIC_SCHEMA})
    assert isinstance(sink.determinism, Determinism)
    assert sink.determinism == Determinism.IO_WRITE
```
**Priority**: P3

### Gap: No test for schema validation contract (allow_coercion=False)
**Issue**: CSVSink.__init__ sets `allow_coercion=False` per CLAUDE.md "Sinks reject wrong types (upstream bug)" but no test verifies this
**Evidence**: Line 86 in csv_sink.py, no corresponding test

**Fix**: Add contract test:
```python
def test_schema_rejects_wrong_types_no_coercion(tmp_path, ctx):
    """Sink schema uses allow_coercion=False (wrong types = upstream bug)."""
    from pydantic import ValidationError

    output_file = tmp_path / "output.csv"
    sink = CSVSink({
        "path": str(output_file),
        "schema": {"mode": "strict", "fields": ["id: int", "score: float"]},
        "validate_input": True,  # Enable validation to test schema
    })

    # String instead of int - should crash (no coercion)
    with pytest.raises(ValidationError, match="id"):
        sink.write([{"id": "not_an_int", "score": 1.5}], ctx)

    # String instead of float - should crash (no coercion)
    with pytest.raises(ValidationError, match="score"):
        sink.write([{"id": 1, "score": "not_a_float"}], ctx)

    sink.close()
```
**Priority**: P1 - Core Three-Tier Trust Model contract

## Poorly Isolated Tests

### Test: test_explicit_schema_creates_all_headers_including_optional (line 254)
**Issue**: Test references specific bug ticket in comment but doesn't use pytest.mark.xfail or parametrize pattern
**Evidence**: Comment says "Bug: P2-2026-01-19-csvsink-fieldnames-inferred-from-first-row" but test doesn't mark the bug status

**Fix**: If bug is still open, mark with xfail. If closed, remove bug reference and make it a regression test:
```python
@pytest.mark.regression("P2-2026-01-19-csvsink-fieldnames-inferred-from-first-row")
def test_explicit_schema_creates_all_headers_including_optional(...):
    """Headers should include all schema fields, not just first row keys."""
```
**Priority**: P3

## Positive Observations

### Content Hash Verification
Tests properly verify SHA-256 content hashing (test_batch_write_content_hash_is_sha256) - critical for audit integrity.

### Cumulative Hash Tracking
Excellent test coverage for cumulative hash behavior (test_cumulative_hash_after_multiple_writes) - this is non-obvious behavior that could easily regress.

### Schema Evolution
Good coverage of dynamic vs explicit schema handling (test_dynamic_schema_still_infers_from_row, test_explicit_schema_creates_all_headers_including_optional).

### Idempotency
Proper test for close() idempotency (test_close_is_idempotent) - important for cleanup reliability.

### Empty Batch Handling
Explicit test for empty batch edge case with correct hash expectation (test_batch_write_empty_list).
