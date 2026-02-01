# Test Defect Report

## Summary

- `test_basic_auth_username_and_password_both_fingerprinted` only asserts fingerprints differ, so it doesn’t prove both username and password are included or that a fingerprint is produced at all.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- `tests/core/security/test_url.py:191`
- Snippet shows the test only checks inequality:
```python
result_userpass = SanitizedWebhookUrl.from_raw_url(url_userpass)
result_user_only = SanitizedWebhookUrl.from_raw_url(url_user_only)

# Different fingerprints because one has password, one doesn't
assert result_userpass.fingerprint != result_user_only.fingerprint
```
- This passes even if `result_userpass.fingerprint` is `None` or only hashes the password, contradicting the test’s stated intent (“Both username and password are included”).

## Impact

- A regression where Basic Auth fingerprints omit the username or are not computed at all would still pass the test, weakening audit traceability for credentials embedded in URLs.

## Root Cause Hypothesis

- The test conflates “different fingerprint” with “fingerprint includes both values,” leading to an under-specified assertion.

## Recommended Fix

- Add explicit assertions that both credentials are included and that the fingerprint is present. Example:
```python
from elspeth.core.security.fingerprint import secret_fingerprint

expected = secret_fingerprint("|".join(sorted(["user", "pass"])))
assert result_userpass.fingerprint == expected
assert result_userpass.fingerprint is not None
```
- Optionally assert that changing either username or password changes the fingerprint, to verify both fields contribute.
