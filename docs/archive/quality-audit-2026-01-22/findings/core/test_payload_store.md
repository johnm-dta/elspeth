# Test Quality Review: test_payload_store.py

## Summary
The test suite for `PayloadStore` covers basic happy-path operations but has **critical gaps** in the auditability contract. The "hashes survive payload deletion" principle is not tested, and there are significant missing edge cases around file operations, concurrency, and the relationship with the Landscape audit trail.

## Poorly Constructed Tests

### Test: test_protocol_has_required_methods (line 12)
**Issue**: Tests implementation detail (hasattr) instead of Protocol conformance
**Evidence**:
```python
assert hasattr(PayloadStore, "store")
```
**Fix**: Use `isinstance()` or structural subtyping check:
```python
from typing import runtime_checkable
assert runtime_checkable(PayloadStore)
# Or: assert isinstance(FilesystemPayloadStore(...), PayloadStore)
```
**Priority**: P2

### Test: test_store_returns_content_hash (line 25)
**Issue**: Validates hex format but not actual SHA-256 correctness
**Evidence**: Only checks length and character set, doesn't verify `hashlib.sha256(content).hexdigest()` equality
**Fix**: Assert `content_hash == hashlib.sha256(content).hexdigest()`
**Priority**: P1

### Test: test_retrieve_nonexistent_raises (line 56)
**Issue**: Creates malformed 64-char string instead of valid hash format
**Evidence**: `"nonexistent" * 4` = 44 chars, not 64
**Fix**: Use proper 64-char hex string: `"0" * 64`
**Priority**: P3

### Test: test_delete_nonexistent_returns_false (line 101)
**Issue**: Same malformed hash string as above
**Evidence**: `"nonexistent" * 4` = 44 chars, not 64
**Fix**: Use proper 64-char hex string: `"0" * 64`
**Priority**: P3

### Test: test_exists_returns_true_for_stored (line 46)
**Issue**: Tests two separate behaviors in one test (exists=True, exists=False)
**Evidence**:
```python
assert store.exists(content_hash) is True
assert store.exists("nonexistent" * 4) is False
```
**Fix**: Split into `test_exists_returns_true_for_stored` and `test_exists_returns_false_for_nonexistent`
**Priority**: P2

## Critical Missing Tests

### Missing: Hash Survives Deletion Test
**Issue**: Core auditability requirement not verified - "Hashes survive payload deletion - integrity is always verifiable" (CLAUDE.md line 18)
**Evidence**: No test verifies that after `delete()`, the Landscape still has the hash and can report integrity status
**Fix**: Add integration test:
```python
def test_landscape_hash_survives_payload_deletion(self, tmp_path: Path) -> None:
    """After payload deletion, Landscape hash remains and retrieve reports missing payload."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    db = LandscapeDB.in_memory()

    # Store payload, record in Landscape
    content = b"audit payload"
    content_hash = store.store(content)
    # ... create row with source_data_hash=content_hash, source_data_ref=content_hash

    # Delete payload
    store.delete(content_hash)

    # Hash still in Landscape
    with db.connection() as conn:
        result = conn.execute(select(rows_table.c.source_data_hash).where(...))
        assert result.scalar() == content_hash

    # Retrieve correctly raises KeyError (payload gone, hash proves it existed)
    with pytest.raises(KeyError, match=content_hash):
        store.retrieve(content_hash)
```
**Priority**: P0 (This is the entire point of separating payload store from audit trail)

### Missing: Idempotency Under Concurrent Access
**Issue**: `test_store_is_idempotent` doesn't test concurrent writes
**Evidence**: Sequential writes only - filesystem race conditions untested
**Fix**: Add property test or threading test:
```python
def test_store_idempotent_concurrent(self, tmp_path: Path) -> None:
    """Multiple threads storing same content return same hash without corruption."""
    from concurrent.futures import ThreadPoolExecutor

    store = FilesystemPayloadStore(base_path=tmp_path)
    content = b"shared content"

    with ThreadPoolExecutor(max_workers=4) as executor:
        hashes = list(executor.map(lambda _: store.store(content), range(10)))

    # All hashes identical
    assert len(set(hashes)) == 1
    # Content uncorrupted
    assert store.retrieve(hashes[0]) == content
```
**Priority**: P1 (Production will have concurrent access)

### Missing: File Permissions Errors
**Issue**: No tests for unreadable/unwritable directories
**Evidence**: Implementation uses `path.write_bytes()` with no permission checking
**Fix**: Add:
```python
def test_store_raises_on_unwritable_directory(self, tmp_path: Path) -> None:
    """Store raises PermissionError when directory is unwritable."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    tmp_path.chmod(0o444)  # Read-only

    with pytest.raises(PermissionError):
        store.store(b"content")

def test_retrieve_raises_on_unreadable_file(self, tmp_path: Path) -> None:
    """Retrieve raises PermissionError when file is unreadable."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    content_hash = store.store(b"content")

    file_path = tmp_path / content_hash[:2] / content_hash
    file_path.chmod(0o000)  # No permissions

    with pytest.raises(PermissionError):
        store.retrieve(content_hash)
```
**Priority**: P2

### Missing: Large Payload Handling
**Issue**: No tests for multi-GB payloads or streaming
**Evidence**: All test payloads < 100 bytes
**Fix**: Add:
```python
def test_store_handles_large_payload(self, tmp_path: Path) -> None:
    """Store can handle payloads larger than memory buffers (1 GB+)."""
    store = FilesystemPayloadStore(base_path=tmp_path)

    # 100 MB payload (adjust based on environment)
    large_content = b"x" * (100 * 1024 * 1024)
    content_hash = store.store(large_content)

    retrieved = store.retrieve(content_hash)
    assert len(retrieved) == len(large_content)
    assert hashlib.sha256(retrieved).hexdigest() == content_hash
```
**Priority**: P1 (LLM responses, CSV files can be large)

### Missing: Empty Payload Handling
**Issue**: No test for zero-length content
**Evidence**: All test payloads have non-zero length
**Fix**: Add:
```python
def test_store_empty_content(self, tmp_path: Path) -> None:
    """Store handles empty payload correctly."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    content_hash = store.store(b"")

    # SHA-256 of empty string
    assert content_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert store.retrieve(content_hash) == b""
```
**Priority**: P2

### Missing: Path Traversal Attack
**Issue**: No test for malicious hash values attempting directory traversal
**Evidence**: Implementation uses `content_hash[:2]` without validation
**Fix**: Add:
```python
def test_retrieve_rejects_path_traversal(self, tmp_path: Path) -> None:
    """Retrieve rejects malicious hash with path traversal characters."""
    store = FilesystemPayloadStore(base_path=tmp_path)

    # These should raise ValueError or KeyError, not access parent dirs
    with pytest.raises((ValueError, KeyError)):
        store.retrieve("../" + "0" * 62)

    with pytest.raises((ValueError, KeyError)):
        store.retrieve("../../../etc/passwd" + "0" * 40)
```
**Priority**: P0 (Security vulnerability)

### Missing: Hash Collision Handling
**Issue**: No test documenting behavior when hash collision occurs (astronomically unlikely but critical for audit trail)
**Evidence**: Implementation assumes SHA-256 collisions impossible
**Fix**: Add documentation test:
```python
def test_hash_collision_behavior_documented(self, tmp_path: Path) -> None:
    """Document behavior if hash collision occurs (should be impossible with SHA-256).

    If a collision ever occurred, the second store() would be skipped (idempotent)
    and retrieve() would return the first content. This would be detected by
    integrity check failing, raising IntegrityError.

    This test documents the behavior - NOT a real collision.
    """
    store = FilesystemPayloadStore(base_path=tmp_path)

    content1 = b"first content"
    content2 = b"different content"

    # Artificially create collision by writing both to same hash location
    hash1 = hashlib.sha256(content1).hexdigest()
    path = tmp_path / hash1[:2] / hash1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content1)

    # Now try to store content2 with same hash (simulate collision)
    # In reality, store() computes hash from content so this can't happen
    # But if filesystem was tampered with:
    retrieved = store.retrieve(hash1)
    assert retrieved == content1  # First content wins (filesystem exists check)
```
**Priority**: P3 (Documentational)

### Missing: Partial Write Handling
**Issue**: No test for interrupted writes (disk full, crash during write)
**Evidence**: Implementation uses `write_bytes()` which is not atomic
**Fix**: Add:
```python
def test_retrieve_detects_partial_write(self, tmp_path: Path) -> None:
    """Retrieve detects partial write via integrity check."""
    store = FilesystemPayloadStore(base_path=tmp_path)

    content = b"x" * 1000
    content_hash = hashlib.sha256(content).hexdigest()

    # Simulate partial write
    path = tmp_path / content_hash[:2] / content_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content[:500])  # Only half written

    with pytest.raises(IntegrityError) as exc_info:
        store.retrieve(content_hash)

    assert "integrity check failed" in str(exc_info.value)
```
**Priority**: P1

### Missing: Subdirectory Distribution Validation
**Issue**: `test_creates_directory_structure` doesn't validate distribution purpose
**Evidence**: Tests that subdirectory exists, but not that it prevents filesystem bottlenecks
**Fix**: Add property test:
```python
def test_subdirectory_distribution_spreads_files(self, tmp_path: Path) -> None:
    """First 2 chars of hash distribute files across 256 subdirectories."""
    store = FilesystemPayloadStore(base_path=tmp_path)

    # Store 1000 different payloads
    for i in range(1000):
        store.store(f"content_{i}".encode())

    # Count subdirectories created
    subdirs = [d for d in tmp_path.iterdir() if d.is_dir()]

    # Should have created multiple subdirs (exact number depends on hash distribution)
    # With 1000 payloads and 256 possible prefixes, expect > 100 unique prefixes
    assert len(subdirs) > 100, "Distribution should spread files across many subdirs"
```
**Priority**: P2

## Infrastructure Gaps

### Gap: No Fixture for Pre-Populated Store
**Issue**: Every test creates fresh store and manually stores content
**Evidence**: Lines 25-44 repeat store creation and content storage
**Fix**: Add fixture:
```python
@pytest.fixture
def populated_store(tmp_path: Path) -> tuple[FilesystemPayloadStore, dict[str, bytes]]:
    """PayloadStore with several pre-stored payloads."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    payloads = {
        "small": b"small content",
        "medium": b"x" * 1024,
        "large": b"y" * (10 * 1024),
    }
    refs = {name: store.store(content) for name, content in payloads.items()}
    return store, refs
```
**Priority**: P2

### Gap: No Property-Based Tests
**Issue**: Only example-based tests - edge cases likely missed
**Evidence**: No use of Hypothesis despite CLAUDE.md recommending it (line 310)
**Fix**: Add Hypothesis tests:
```python
from hypothesis import given, strategies as st

@given(content=st.binary(min_size=0, max_size=10000))
def test_store_retrieve_roundtrip_property(self, tmp_path: Path, content: bytes) -> None:
    """Any content can be stored and retrieved exactly."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    content_hash = store.store(content)
    assert store.retrieve(content_hash) == content

@given(content=st.binary(min_size=1, max_size=1000))
def test_store_idempotent_property(self, tmp_path: Path, content: bytes) -> None:
    """Storing same content N times returns same hash."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    hashes = [store.store(content) for _ in range(5)]
    assert len(set(hashes)) == 1
```
**Priority**: P1

### Gap: No Mock/Integration Test Separation
**Issue**: Tests use real filesystem - slow, order-dependent cleanup risks
**Evidence**: All tests use `tmp_path` fixture but no in-memory alternative
**Fix**: Consider abstract backend tests:
```python
class PayloadStoreContractTests:
    """Reusable test suite for any PayloadStore implementation."""

    @pytest.fixture
    def store(self) -> PayloadStore:
        raise NotImplementedError("Subclass must provide store fixture")

    def test_store_retrieve_roundtrip(self, store: PayloadStore) -> None:
        content = b"test"
        content_hash = store.store(content)
        assert store.retrieve(content_hash) == content

class TestFilesystemPayloadStore(PayloadStoreContractTests):
    @pytest.fixture
    def store(self, tmp_path: Path) -> PayloadStore:
        return FilesystemPayloadStore(base_path=tmp_path)

class TestS3PayloadStore(PayloadStoreContractTests):
    @pytest.fixture
    def store(self) -> PayloadStore:
        return S3PayloadStore(bucket="test-bucket", mock=True)
```
**Priority**: P2 (Enables testing future backends)

### Gap: No Cleanup Validation
**Issue**: Tests don't verify subdirectories are cleaned up when empty
**Evidence**: `delete()` removes file but not parent subdirectory
**Fix**: Add test:
```python
def test_delete_cleans_up_empty_subdirectories(self, tmp_path: Path) -> None:
    """Delete removes empty subdirectories after last file deleted."""
    store = FilesystemPayloadStore(base_path=tmp_path)
    content_hash = store.store(b"content")
    subdir = tmp_path / content_hash[:2]

    assert subdir.exists()
    store.delete(content_hash)

    # If this was the only file in subdir, subdir should be removed
    # (Implementation may need to add this behavior)
    # assert not subdir.exists()
```
**Priority**: P3 (Optimization, prevents directory clutter)

## Misclassified Tests

### Test: TestPayloadStoreProtocol (class)
**Issue**: Protocol tests should be in separate module for contract verification
**Evidence**: Mixed with implementation-specific tests in same file
**Fix**: Move to `tests/core/contracts/test_payload_store_protocol.py` or use abstract base class pattern
**Priority**: P2

### Test: Integrity tests (lines 108-164)
**Issue**: These are critical integration tests masquerading as unit tests
**Evidence**: Tests simulate data corruption, which requires understanding filesystem behavior
**Fix**: Move to `tests/integration/test_payload_store_integrity.py` or add explicit markers:
```python
@pytest.mark.integration
class TestPayloadStoreIntegrity:
    ...
```
**Priority**: P2

## Positive Observations

1. **Integrity verification tests are excellent**: Lines 108-164 verify corruption detection with expected/actual hash reporting (critical for auditability)
2. **Edge case coverage for corruption**: Tests both content replacement and truncation
3. **Clear test names**: Test names clearly state expected behavior
4. **Good use of tmp_path fixture**: Tests are isolated and don't pollute global filesystem
5. **Error message validation**: Lines 120-124, 161-163 verify error messages contain debugging information

## Risk Assessment

**Confidence**: High - I read both implementation and tests thoroughly

**Information Gaps**:
- Actual production usage patterns (payload size distribution, access patterns)
- Whether S3 or other backends are planned (affects contract testing priority)
- Retention policy integration details (PurgeManager tests show integration exists)

**Caveats**:
- Path traversal vulnerability is **inferred** from lack of validation in implementation - needs code review
- Concurrency issues are **theoretical** - actual risk depends on deployment architecture
- "Hash survives deletion" test is marked P0 but requires Landscape integration - may belong elsewhere

**Blockers**:
- P0 "Hash survives deletion" test requires understanding Landscape schema (found in purge tests, lines 523-573)
- P0 Path traversal test requires security review of hash format validation

## Recommended Next Steps

1. **Immediate (P0)**:
   - Add path traversal security test
   - Add "hash survives deletion" integration test (coordinate with Landscape tests)

2. **Before RC-2 (P1)**:
   - Add Hypothesis property tests for store/retrieve/delete operations
   - Add large payload test (100 MB+)
   - Add concurrent access test
   - Add partial write detection test

3. **Technical Debt (P2/P3)**:
   - Refactor to contract tests for future backends
   - Split protocol tests into separate module
   - Add cleanup validation test
   - Fix malformed hash strings in existing tests
