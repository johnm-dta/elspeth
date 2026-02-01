# Test Defect Report

## Summary

- Weak hash assertions: tests only check hash shape/idempotence and never verify the SHA-256 digest or that different content yields different hashes.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/test_payload_store.py:25` only asserts length/hex characters, not the actual SHA-256 digest.
- `tests/core/test_payload_store.py:64` only asserts same content yields same hash; there is no test that different content yields different hashes.
- `src/elspeth/core/payload_store.py:32` specifies SHA-256 hashes are required.

```python
def test_store_returns_content_hash(self, tmp_path: Path) -> None:
    ...
    # Should be SHA-256 hex
    assert len(content_hash) == 64
    assert all(c in "0123456789abcdef" for c in content_hash)

def test_store_is_idempotent(self, tmp_path: Path) -> None:
    ...
    hash1 = store.store(content)
    hash2 = store.store(content)
    assert hash1 == hash2
```

## Impact

- A regression to a non-SHA-256 algorithm (or a constant/low-entropy hash) could still pass tests, causing collisions and payload corruption.
- Weak verification undermines audit integrity guarantees tied to content-addressable storage.

## Root Cause Hypothesis

- Tests emphasize superficial format checks and idempotence, not correctness against the required SHA-256 contract.

## Recommended Fix

- Add a direct digest check using `hashlib.sha256(content).hexdigest()`.
- Add a test that stores two distinct payloads and asserts their hashes differ (optionally parameterized or Hypothesis-based).
- This is core integrity behavior, so prioritize to P1.
---
# Test Defect Report

## Summary

- Protocol test uses `hasattr`, which is a prohibited defensive pattern and does not enforce callable method presence.

## Severity

- Severity: minor
- Priority: P2

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/core/test_payload_store.py:16` uses `hasattr` on system code.
- `CLAUDE.md:494` prohibits `hasattr` usage.

```python
assert hasattr(PayloadStore, "store")
assert hasattr(PayloadStore, "retrieve")
assert hasattr(PayloadStore, "exists")
assert hasattr(PayloadStore, "delete")
```

## Impact

- Signals that defensive introspection is acceptable and allows non-callable or wrong-signature attributes to pass, weakening protocol enforcement.
- Normalizes a pattern explicitly disallowed by repository standards.

## Root Cause Hypothesis

- Test was written as a quick existence check without aligning to the no-defensive-patterns rule.

## Recommended Fix

- Replace `hasattr` with direct attribute access plus `callable` checks, e.g., `assert callable(PayloadStore.store)`, so missing attributes fail without defensive introspection.
- Keep the test focused on protocol contract, not defensive behavior.
