# Test Defect Report

## Summary

- Env-var path test only checks digest length, not that the environment key is actually used to compute the fingerprint.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/security/test_fingerprint.py:43` sets `ELSPETH_FINGERPRINT_KEY` but does not validate that key selection affects the digest.
- `tests/core/security/test_fingerprint.py:50` shows the only assertion is `len(result) == 64`, which would pass even if the wrong key were used.

```
def test_fingerprint_without_key_uses_env_var(...):
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "env-key-value")
    result = secret_fingerprint("my-secret")
    assert len(result) == 64
```

## Impact

- Key-selection regressions in the `key=None` path can slip through (e.g., wrong env var, wrong key source), producing incorrect fingerprints.
- Creates false confidence that production env-var configuration is correctly honored.

## Root Cause Hypothesis

- Test was written as a "no exception" smoke test rather than a correctness check.

## Recommended Fix

- Assert the exact expected HMAC for the env key, or compare against a known digest so the test fails if the env key is ignored.
- Example:

```
expected = hmac.new(b"env-key-value", b"my-secret", hashlib.sha256).hexdigest()
assert result == expected
```

- Priority justification: this is a security-critical configuration path; weak assertions allow silent mis-fingerprinting.
---
# Test Defect Report

## Summary

- No golden HMAC test vector; tests only validate shape/determinism/differences, so a non-HMAC algorithm could pass.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/security/test_fingerprint.py:12` and `tests/core/security/test_fingerprint.py:38` only assert type/hex/length; there is no fixed expected digest.
- `tests/core/security/test_fingerprint.py:18` and `tests/core/security/test_fingerprint.py:32` only assert equality/inequality between samples, not correctness of HMAC output.
- `src/elspeth/core/security/fingerprint.py:103` specifies HMAC-SHA256 in the contract.

```
def test_fingerprint_returns_hex_string(...):
    result = secret_fingerprint("my-api-key", key=b"test-key")
    assert isinstance(result, str)
    assert all(c in "0123456789abcdef" for c in result)

def test_fingerprint_length_is_64_chars(...):
    result = secret_fingerprint("test", key=b"key")
    assert len(result) == 64
```

## Impact

- An implementation that uses plain SHA256 (or other incorrect algorithm with 64-hex output) could still pass all tests, weakening secret fingerprint guarantees.
- Risks audit integrity because the fingerprint may no longer be a proper HMAC.

## Root Cause Hypothesis

- Tests focus on general properties instead of verifying a known-good digest.

## Recommended Fix

- Add a test with a fixed key/secret and a precomputed HMAC-SHA256 digest to lock the algorithm.
- Example:

```
assert secret_fingerprint("my-secret", key=b"key") == "expected_hex_digest_here"
```

- Priority justification: cryptographic correctness is a core security contract; tests should pin a known vector.
