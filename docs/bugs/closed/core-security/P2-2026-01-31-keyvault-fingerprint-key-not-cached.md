# Bug Report: Key Vault fingerprint key retrieved on every call (no cache)

## Summary

- `get_fingerprint_key()` makes a Key Vault API call every time when `ELSPETH_FINGERPRINT_KEY` is unset but `ELSPETH_KEYVAULT_URL` is set.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/core/security/fingerprint.py:58-99` - `get_fingerprint_key()` has no caching.
- Every call when env var unset makes Key Vault API call

## Impact

- User-facing impact: Performance degradation, potential rate limiting
- Data integrity: None
- Cost: Increased Key Vault API costs

## Proposed Fix

- Add module-level `_CACHED_FINGERPRINT_KEY: bytes | None = None` and return if populated

## Acceptance Criteria

- Key Vault called at most once per process

## Verification (2026-02-01)

**Status: STILL VALID**

- `get_fingerprint_key()` still performs a Key Vault lookup on every call when env var is unset. (`src/elspeth/core/security/fingerprint.py:58-95`)

## Resolution (2026-02-02)

**Status: RESOLVED**

### Implementation

1. Created `SecretLoader` abstraction in `src/elspeth/core/security/secret_loader.py`:
   - `KeyVaultSecretLoader` with built-in per-secret caching
   - `CachedSecretLoader` wrapper for any loader
   - `CompositeSecretLoader` for fallback chains
   - `EnvSecretLoader` for environment variables

2. Updated `get_fingerprint_key()` in `fingerprint.py`:
   - Added module-level `_cached_fingerprint_key` variable
   - Uses `KeyVaultSecretLoader` with caching
   - Added `clear_fingerprint_key_cache()` for testing

### Verification

- Test `test_get_fingerprint_key_uses_secret_loader_with_caching` verifies Key Vault is called at most once
- All 120 security tests pass

### Files Changed

- `src/elspeth/core/security/secret_loader.py` (new)
- `src/elspeth/core/security/fingerprint.py` (modified)
- `src/elspeth/core/security/__init__.py` (modified)
- `tests/core/security/test_secret_loader.py` (new)
- `tests/core/security/test_fingerprint.py` (modified)
- `tests/core/security/test_fingerprint_keyvault.py` (modified)
