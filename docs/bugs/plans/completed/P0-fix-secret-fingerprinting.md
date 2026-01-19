# Implementation Plan: Fix Secret Fingerprinting (Deep + DSN + Fail-Closed)

**Bug:** P0-2026-01-19-secret-fingerprinting-shallow-and-dsn-password-leak.md
**Estimated Time:** 4-6 hours
**Complexity:** Medium
**Risk:** Medium (behavior change from fail-open to fail-closed)

## Summary

Three security issues with secret fingerprinting:
1. **Shallow** - Only top-level keys in `options` are fingerprinted, nested secrets pass through
2. **DSN gaps** - `landscape.url` can contain passwords that aren't sanitized
3. **Fail open** - Missing `ELSPETH_FINGERPRINT_KEY` silently preserves raw secrets

## Current Behavior

```python
# Only checks top-level keys:
options = {"api_key": "sk-123", "auth": {"api_key": "sk-456"}}
_fingerprint_secrets(options)
# Result: {"api_key_fingerprint": "...", "auth": {"api_key": "sk-456"}}  # LEAK!

# DSN password untouched:
landscape.url = "postgresql://user:password@localhost/db"  # LEAK!

# Missing key = silent pass-through:
# ELSPETH_FINGERPRINT_KEY not set → secrets preserved as-is  # LEAK!
```

## Implementation Steps

### Step 1: Add recursive secret fingerprinting

**File:** `src/elspeth/core/config.py`

**Replace `_fingerprint_secrets()` with recursive version:**

```python
def _fingerprint_secrets(
    options: dict[str, Any],
    *,
    fail_if_no_key: bool = True,
) -> dict[str, Any]:
    """Recursively replace secret fields with their fingerprints.

    Walks nested dicts and lists to find and fingerprint all secret fields,
    not just top-level ones.

    Args:
        options: Plugin options dict (may contain nested structures)
        fail_if_no_key: If True, raise if ELSPETH_FINGERPRINT_KEY not set
                        and secrets are found. If False, redact secrets
                        without fingerprinting (for dev mode).

    Returns:
        New dict with secrets replaced by fingerprints (or redacted)

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key available
                                and fail_if_no_key is True
    """
    from elspeth.core.security import secret_fingerprint, get_fingerprint_key

    # Check if we have a fingerprint key available
    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    def _process_value(key: str, value: Any) -> tuple[str, Any, bool]:
        """Process a single value, returning (new_key, new_value, was_secret)."""
        if isinstance(value, dict):
            return key, _recurse(value), False
        elif isinstance(value, list):
            return key, [_process_value("", item)[1] for item in value], False
        elif isinstance(value, str) and _is_secret_field(key):
            # This is a secret field
            if have_key:
                fp = secret_fingerprint(value)
                return f"{key}_fingerprint", fp, True
            elif fail_if_no_key:
                raise SecretFingerprintError(
                    f"Secret field '{key}' found but ELSPETH_FINGERPRINT_KEY is not set. "
                    f"Either set the environment variable or use ELSPETH_ALLOW_RAW_SECRETS=true "
                    f"for development (not recommended for production)."
                )
            else:
                # Dev mode: redact without fingerprint
                return f"{key}_redacted", "[REDACTED]", True
        else:
            return key, value, False

    def _recurse(d: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for key, value in d.items():
            new_key, new_value, was_secret = _process_value(key, value)
            result[new_key] = new_value
            # If we renamed the key (added _fingerprint or _redacted), don't keep original
            # (handled by using new_key)
        return result

    return _recurse(options)
```

### Step 2: Add SecretFingerprintError exception

**File:** `src/elspeth/core/config.py`

**Add near top of file (after imports):**

```python
class SecretFingerprintError(Exception):
    """Raised when secrets are found but cannot be fingerprinted.

    This occurs when:
    - Secret-like field names are found in config
    - ELSPETH_FINGERPRINT_KEY is not set
    - ELSPETH_ALLOW_RAW_SECRETS is not set to 'true'
    """
    pass
```

### Step 3: Add DSN password sanitization

**File:** `src/elspeth/core/config.py`

**Add new function:**

```python
def _sanitize_dsn(
    url: str,
    *,
    fail_if_no_key: bool = True,
) -> tuple[str, str | None, bool]:
    """Sanitize a database connection URL by removing/fingerprinting the password.

    Args:
        url: Database connection URL (SQLAlchemy format)
        fail_if_no_key: If True, raise if password found but no fingerprint key.
                        If False (dev mode), just remove password without fingerprint.

    Returns:
        Tuple of (sanitized_url, password_fingerprint or None, had_password)
        The third element indicates whether the original URL had a password.

    Raises:
        SecretFingerprintError: If password found, no key available, and fail_if_no_key=True

    Example:
        >>> _sanitize_dsn("postgresql://user:secret@host/db")
        ("postgresql://user@host/db", "abc123...", True)
    """
    from sqlalchemy.engine import URL
    from sqlalchemy.engine.url import make_url

    try:
        parsed = make_url(url)
    except Exception:
        # Not a valid URL - return as-is (might be a path or other format)
        return url, None, False

    if parsed.password is None:
        # No password in URL
        return url, None, False

    # Check if we have a fingerprint key
    from elspeth.core.security import get_fingerprint_key
    try:
        get_fingerprint_key()
        have_key = True
    except ValueError:
        have_key = False

    # Compute fingerprint if we have a key
    password_fingerprint = None
    if have_key:
        from elspeth.core.security import secret_fingerprint
        password_fingerprint = secret_fingerprint(parsed.password)
    elif fail_if_no_key:
        raise SecretFingerprintError(
            f"Database URL contains a password but ELSPETH_FINGERPRINT_KEY is not set. "
            f"Either set the environment variable or use ELSPETH_ALLOW_RAW_SECRETS=true "
            f"for development (not recommended for production)."
        )
    # else: dev mode - just remove password without fingerprint

    # Reconstruct URL without password using URL.create()
    # NOTE: Do NOT use parsed.set(password=None) - it replaces with '***' not removal
    sanitized = URL.create(
        drivername=parsed.drivername,
        username=parsed.username,
        password=None,
        host=parsed.host,
        port=parsed.port,
        database=parsed.database,
        query=parsed.query,
    )

    return str(sanitized), password_fingerprint, True
```

### Step 4: Apply DSN sanitization to landscape.url

**File:** `src/elspeth/core/config.py`

**Modify `_fingerprint_config_options()` to also handle landscape URL:**

```python
def _fingerprint_config_options(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Walk config and fingerprint secrets in all plugin options.

    Processes:
    - datasource.options
    - sinks.*.options
    - row_plugins[*].options
    - aggregations[*].options
    - landscape.url (DSN password)

    Args:
        raw_config: Raw config dict from Dynaconf

    Returns:
        Config with secrets fingerprinted

    Raises:
        SecretFingerprintError: If secrets found but no fingerprint key
                                and ELSPETH_ALLOW_RAW_SECRETS is not set
    """
    import os

    # Check dev mode override
    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
    fail_if_no_key = not allow_raw

    config = dict(raw_config)

    # === Landscape URL (DSN password) ===
    if "landscape" in config and isinstance(config["landscape"], dict):
        landscape = dict(config["landscape"])
        if "url" in landscape and isinstance(landscape["url"], str):
            # _sanitize_dsn returns (sanitized_url, fingerprint, had_password)
            sanitized_url, password_fp, had_password = _sanitize_dsn(
                landscape["url"],
                fail_if_no_key=fail_if_no_key,
            )
            landscape["url"] = sanitized_url
            if password_fp:
                landscape["url_password_fingerprint"] = password_fp
            elif had_password and not fail_if_no_key:
                # Dev mode: password was removed but not fingerprinted
                landscape["url_password_redacted"] = True
        config["landscape"] = landscape

    # === Datasource options ===
    if "datasource" in config and isinstance(config["datasource"], dict):
        ds = dict(config["datasource"])
        if "options" in ds and isinstance(ds["options"], dict):
            ds["options"] = _fingerprint_secrets(ds["options"], fail_if_no_key=fail_if_no_key)
        config["datasource"] = ds

    # === Sink options ===
    if "sinks" in config and isinstance(config["sinks"], dict):
        sinks = {}
        for name, sink_config in config["sinks"].items():
            if isinstance(sink_config, dict):
                sink = dict(sink_config)
                if "options" in sink and isinstance(sink["options"], dict):
                    sink["options"] = _fingerprint_secrets(sink["options"], fail_if_no_key=fail_if_no_key)
                sinks[name] = sink
            else:
                sinks[name] = sink_config
        config["sinks"] = sinks

    # === Row plugin options ===
    if "row_plugins" in config and isinstance(config["row_plugins"], list):
        plugins = []
        for plugin_config in config["row_plugins"]:
            if isinstance(plugin_config, dict):
                plugin = dict(plugin_config)
                if "options" in plugin and isinstance(plugin["options"], dict):
                    plugin["options"] = _fingerprint_secrets(plugin["options"], fail_if_no_key=fail_if_no_key)
                plugins.append(plugin)
            else:
                plugins.append(plugin_config)
        config["row_plugins"] = plugins

    # === Aggregation options ===
    if "aggregations" in config and isinstance(config["aggregations"], list):
        aggs = []
        for agg_config in config["aggregations"]:
            if isinstance(agg_config, dict):
                agg = dict(agg_config)
                if "options" in agg and isinstance(agg["options"], dict):
                    agg["options"] = _fingerprint_secrets(agg["options"], fail_if_no_key=fail_if_no_key)
                aggs.append(agg)
            else:
                aggs.append(agg_config)
        config["aggregations"] = aggs

    return config
```

### Step 5: Export the exception

**File:** `src/elspeth/core/__init__.py`

Add `SecretFingerprintError` to public exports so tests and user code can catch it:

```python
# In src/elspeth/core/__init__.py, add to imports and __all__:
from elspeth.core.config import SecretFingerprintError

__all__ = [
    # ... existing exports ...
    "SecretFingerprintError",
]
```

Also ensure `config.py` module-level `__all__` includes the exception:

```python
# Near the top of src/elspeth/core/config.py, after the class definition:
__all__ = [
    "ElspethSettings",
    "SecretFingerprintError",
    "load_settings",
    "resolve_config",
    # ... other public symbols ...
]
```

### Step 6: Update tests

**File:** `tests/core/test_config.py`

**Add these new tests to the existing `TestSecretFingerprinting` class, and update the existing `test_fingerprinting_skipped_when_no_key` test:**

```python
from elspeth.core.config import (
    _fingerprint_secrets,
    _sanitize_dsn,
    _fingerprint_config_options,
    SecretFingerprintError,
    load_settings,
)

class TestSecretFingerprinting:
    """Tests for secret fingerprinting."""

    # === NEW: Tests for recursive/nested fingerprinting ===

    def test_nested_secrets_are_fingerprinted(self, monkeypatch):
        """Secrets in nested dicts should be fingerprinted."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        options = {
            "api_key": "sk-top-level",
            "auth": {
                "api_key": "sk-nested",
                "nested": {
                    "token": "nested-token"
                }
            }
        }

        result = _fingerprint_secrets(options)

        # Top-level secret fingerprinted
        assert "api_key" not in result
        assert "api_key_fingerprint" in result

        # Nested secrets fingerprinted
        assert "api_key" not in result["auth"]
        assert "api_key_fingerprint" in result["auth"]
        assert "token" not in result["auth"]["nested"]
        assert "token_fingerprint" in result["auth"]["nested"]

    def test_secrets_in_lists_are_fingerprinted(self, monkeypatch):
        """Secrets inside list elements should be fingerprinted."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        options = {
            "providers": [
                {"name": "openai", "api_key": "sk-openai"},
                {"name": "anthropic", "api_key": "sk-anthropic"},
            ]
        }

        result = _fingerprint_secrets(options)

        for provider in result["providers"]:
            assert "api_key" not in provider
            assert "api_key_fingerprint" in provider

    # === NEW: Tests for fail-closed behavior ===

    def test_missing_key_raises_error_on_fingerprint(self, monkeypatch):
        """Missing fingerprint key should raise SecretFingerprintError."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        options = {"api_key": "sk-secret"}

        with pytest.raises(SecretFingerprintError) as exc_info:
            _fingerprint_secrets(options, fail_if_no_key=True)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)
        assert "api_key" in str(exc_info.value)

    def test_missing_key_raises_error_on_load_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """load_settings should raise SecretFingerprintError when key missing.

        This is a BREAKING CHANGE from previous behavior where secrets were
        silently preserved. Now we fail-closed to prevent accidental leakage.
        """
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        with pytest.raises(SecretFingerprintError) as exc_info:
            load_settings(config_file)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)

    def test_dev_mode_redacts_without_fingerprint(self, monkeypatch):
        """ELSPETH_ALLOW_RAW_SECRETS=true should redact without crashing."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        options = {"api_key": "sk-secret"}

        result = _fingerprint_secrets(options, fail_if_no_key=False)

        assert "api_key" not in result
        assert result.get("api_key_redacted") == "[REDACTED]"

    def test_dev_mode_allows_load_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """ELSPETH_ALLOW_RAW_SECRETS=true should allow load without fingerprint key."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        config_file = tmp_path / "settings.yaml"
        config_file.write_text("""
datasource:
  plugin: http_source
  options:
    api_key: sk-secret-key
sinks:
  output:
    plugin: csv_sink
output_sink: output
""")

        # Should NOT raise - dev mode allows redaction without fingerprinting
        settings = load_settings(config_file)

        # Secret should be redacted, not preserved raw
        assert "api_key" not in settings.datasource.options
        assert settings.datasource.options.get("api_key_redacted") == "[REDACTED]"

    # === NEW: Tests for DSN password handling ===

    def test_dsn_password_sanitized(self, monkeypatch):
        """DSN passwords should be removed and fingerprinted."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        url = "postgresql://user:secret_password@localhost:5432/mydb"
        sanitized, fingerprint, had_password = _sanitize_dsn(url)

        assert "secret_password" not in sanitized
        assert "user@localhost" in sanitized
        # Should NOT have *** placeholder - password fully removed
        assert "***" not in sanitized
        assert fingerprint is not None
        assert len(fingerprint) == 64  # SHA256 hex
        assert had_password is True

    def test_dsn_without_password_unchanged(self, monkeypatch):
        """DSN without password should pass through."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        url = "sqlite:///path/to/db.sqlite"
        sanitized, fingerprint, had_password = _sanitize_dsn(url)

        assert sanitized == url
        assert fingerprint is None
        assert had_password is False

    def test_dsn_password_raises_without_key(self, monkeypatch):
        """DSN with password should raise when no fingerprint key."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

        url = "postgresql://user:secret@localhost/db"

        with pytest.raises(SecretFingerprintError) as exc_info:
            _sanitize_dsn(url, fail_if_no_key=True)

        assert "ELSPETH_FINGERPRINT_KEY" in str(exc_info.value)

    def test_dsn_password_redacted_in_dev_mode(self, monkeypatch):
        """DSN password should be removed (not fingerprinted) in dev mode."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)

        url = "postgresql://user:secret@localhost/db"
        sanitized, fingerprint, had_password = _sanitize_dsn(url, fail_if_no_key=False)

        assert "secret" not in sanitized
        assert fingerprint is None  # No fingerprint in dev mode
        assert had_password is True

    def test_landscape_url_password_fingerprinted(self, monkeypatch):
        """landscape.url password should be fingerprinted."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")

        raw_config = {
            "landscape": {
                "url": "postgresql://user:mysecret@host/db"
            },
            "datasource": {"plugin": "csv", "options": {}},
            "sinks": {"output": {"plugin": "csv_sink"}},
            "output_sink": "output",
        }

        result = _fingerprint_config_options(raw_config)

        assert "mysecret" not in result["landscape"]["url"]
        assert "***" not in result["landscape"]["url"]  # No placeholder
        assert "url_password_fingerprint" in result["landscape"]
        assert len(result["landscape"]["url_password_fingerprint"]) == 64

    def test_landscape_url_password_redacted_in_dev_mode(self, monkeypatch):
        """landscape.url password should be redacted (with flag) in dev mode."""
        monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

        raw_config = {
            "landscape": {
                "url": "postgresql://user:mysecret@host/db"
            },
            "datasource": {"plugin": "csv", "options": {}},
            "sinks": {"output": {"plugin": "csv_sink"}},
            "output_sink": "output",
        }

        result = _fingerprint_config_options(raw_config)

        assert "mysecret" not in result["landscape"]["url"]
        assert "url_password_fingerprint" not in result["landscape"]
        assert result["landscape"]["url_password_redacted"] is True
```

**IMPORTANT: Remove/Replace the existing `test_fingerprinting_skipped_when_no_key` test** (around line 1745-1768 in the existing test file). This test expected secrets to be preserved - that's the old fail-open behavior we're removing. Replace it with `test_missing_key_raises_error_on_load_settings` above.

### Step 7: Document behavior change in README

**File:** `README.md`

**Location:** Add new section under "Configuration" or "Environment Variables"

Add documentation about secret fingerprinting requirements:

```markdown
## Secret Fingerprinting

ELSPETH fingerprints sensitive configuration values (API keys, tokens, passwords) before storing them in the audit trail. This ensures secrets are never written to the database in plain text.

### Required Environment Variable

```bash
# Production: Set a stable secret key for fingerprinting
export ELSPETH_FINGERPRINT_KEY="your-secret-key-here"
```

**IMPORTANT:** If `ELSPETH_FINGERPRINT_KEY` is not set and your configuration contains secrets, ELSPETH will raise a `SecretFingerprintError` at startup. This is intentional - silent secret leakage to the audit database is a security risk.

### Development Mode

For local development where fingerprint stability isn't required:

```bash
# Development only: Allow secrets without fingerprinting
export ELSPETH_ALLOW_RAW_SECRETS=true
```

This will redact secrets (replacing them with `[REDACTED]`) instead of fingerprinting them. **Do not use in production.**

### What Gets Fingerprinted

- Plugin options with secret-like field names (`api_key`, `token`, `password`, `secret`, etc.)
- Nested secrets in configuration objects
- Database passwords in `landscape.url` DSN strings

### Behavior Change Notice

Prior versions silently preserved raw secrets when `ELSPETH_FINGERPRINT_KEY` was unset. Current versions fail-closed by default - you must either:
1. Set `ELSPETH_FINGERPRINT_KEY` (recommended), or
2. Explicitly opt-in to dev mode with `ELSPETH_ALLOW_RAW_SECRETS=true`
```

## Testing Checklist

- [ ] Nested secrets in dicts are fingerprinted
- [ ] Secrets in lists are fingerprinted
- [ ] DSN passwords are removed from URLs (using `URL.create()`, NOT `set(password=None)`)
- [ ] DSN password fingerprints are stored separately
- [ ] Missing `ELSPETH_FINGERPRINT_KEY` raises `SecretFingerprintError` on `_fingerprint_secrets()`
- [ ] Missing `ELSPETH_FINGERPRINT_KEY` raises `SecretFingerprintError` on `load_settings()`
- [ ] Missing `ELSPETH_FINGERPRINT_KEY` raises `SecretFingerprintError` on `_sanitize_dsn()`
- [ ] `ELSPETH_ALLOW_RAW_SECRETS=true` redacts without crashing
- [ ] `ELSPETH_ALLOW_RAW_SECRETS=true` allows `load_settings()` with redaction
- [ ] Old `test_fingerprinting_skipped_when_no_key` test deleted/replaced
- [ ] SQLite URLs (no password) pass through unchanged
- [ ] README updated with secret fingerprinting documentation and behavior change notice

## Run Tests

```bash
# Set fingerprint key for tests that need it
export ELSPETH_FINGERPRINT_KEY="test-key-for-ci"

# Run config tests
.venv/bin/python -m pytest tests/core/test_config.py -v -k fingerprint

# Run all tests (may need ELSPETH_ALLOW_RAW_SECRETS=true for some)
ELSPETH_ALLOW_RAW_SECRETS=true .venv/bin/python -m pytest tests/ -v
```

## Migration Notes

### Breaking Change: Fail-Closed Default

**Before:** Missing `ELSPETH_FINGERPRINT_KEY` → secrets preserved silently
**After:** Missing key → `SecretFingerprintError` raised

**Migration path:**
1. Production: Set `ELSPETH_FINGERPRINT_KEY` (required)
2. Development: Set `ELSPETH_ALLOW_RAW_SECRETS=true` (explicit opt-in to unsafe behavior)
3. Tests: Update fixtures to either set the key or the dev override

### Existing Tests

The existing `test_fingerprinting_skipped_when_no_key` test (around line 1745-1768 in test_config.py) relies on the old "fail open" behavior and **must be deleted/replaced** with the new `test_missing_key_raises_error_on_load_settings` test.

Other tests that don't use secrets don't need changes. Tests that DO use secrets need one of:
- `monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")` for fingerprint testing, or
- `monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")` for tests that don't care about secrets

## Acceptance Criteria

1. ✅ Nested secrets in `options.auth.api_key` are fingerprinted
2. ✅ DSN passwords in `landscape.url` are removed + fingerprinted
3. ✅ Missing fingerprint key raises clear error (not silent leak)
4. ✅ Dev mode (`ELSPETH_ALLOW_RAW_SECRETS=true`) allows explicit opt-out
5. ✅ All existing tests pass (with appropriate env vars)

## Security Notes

**Why fail-closed is correct:**
- Silent secret leakage to audit DB is worse than a startup crash
- Operators will immediately know they need to set the key
- Dev mode is explicit opt-in, not accidental

**Why fingerprint (not just redact):**
- Fingerprints allow verifying "same API key used across runs" for debugging
- Redaction loses this capability entirely
- HMAC with secret key prevents rainbow table attacks
