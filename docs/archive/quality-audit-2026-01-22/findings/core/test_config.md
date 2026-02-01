# Test Quality Review: test_config.py

## Summary
The test suite for configuration validation is comprehensive with 2769 lines covering Pydantic model validation, configuration precedence, and secret fingerprinting. However, it exhibits critical anti-patterns including missing test isolation (shared state via environment variables), no verification of configuration precedence rules, and weak assertions that fail to verify crash-on-invalid semantics mandated by CLAUDE.md.

## Poorly Constructed Tests

### Test: test_load_with_env_override (line 131)
**Issue**: State leakage via environment variables without proper cleanup
**Evidence**: Uses `monkeypatch.setenv("ELSPETH_DATASOURCE__PLUGIN", "json")` but relies on pytest's automatic cleanup. If monkeypatch cleanup fails or test is interrupted, subsequent tests may receive wrong values.
**Fix**: Add explicit verification after test that env var is cleaned up, or use a fixture that guarantees cleanup with try/finally. For critical config tests, verify isolation by reading env after cleanup.
**Priority**: P1

### Test: TestSecretFieldFingerprinting class (lines 1703-2256)
**Issue**: Massive class (553 lines) testing 3 different responsibilities (load-time preservation, resolve-time fingerprinting, DSN handling)
**Evidence**: Class contains 20+ test methods mixing unit tests for `_fingerprint_secrets()`, integration tests for `load_settings()`, and DSN parsing tests. Single class violates single-responsibility principle.
**Fix**: Split into 3 classes: `TestSecretPreservationAtLoad`, `TestSecretFingerprintingInResolve`, `TestDSNPasswordHandling`. Each class should test one transformation stage.
**Priority**: P2

### Test: test_gate_settings_invalid_condition_syntax (line 922)
**Issue**: Weak assertion on error message substring
**Evidence**: `assert "Invalid condition syntax" in str(exc_info.value)` - this passes if error message says "Invalid condition syntax found in unrelated field". Doesn't verify WHICH field failed.
**Fix**: Use structured error access via `exc_info.value.errors()` to verify `loc` field matches `["condition"]` and `type` matches expected validation error type.
**Priority**: P2

### Test: test_export_sink_must_exist_when_enabled (line 471)
**Issue**: Assertion checks error message string instead of validation structure
**Evidence**: `assert "export.sink 'nonexistent_sink' not found in sinks" in str(exc_info.value)` - fragile to error message wording changes.
**Fix**: Access Pydantic validation errors via `exc_info.value.errors()` and assert on error structure: `{"loc": ("landscape", "export", "sink"), "type": "value_error"}`.
**Priority**: P3

### Test: test_load_missing_file_raises_file_not_found (line 178)
**Issue**: Assertion only checks exception type and partial message match
**Evidence**: `with pytest.raises(FileNotFoundError, match="Config file not found")` - doesn't verify WHICH file path was not found.
**Fix**: Capture exception and assert that `str(exc_info.value)` contains the actual missing path: `assert "nonexistent.yaml" in str(exc_info.value)`.
**Priority**: P3

### Test: test_dsn_password_sanitized (line 2160)
**Issue**: Assertion `assert "***" not in sanitized` is testing for absence of a placeholder that shouldn't exist
**Evidence**: Comment says "Should NOT have placeholder" but this is a negative assertion about an implementation detail, not verification of correct behavior.
**Fix**: Replace with positive assertion about what SHOULD be present: `assert sanitized == "postgresql://user@localhost:5432/mydb"` (exact expected output).
**Priority**: P2

### Test: test_api_key_is_fingerprinted_in_resolve_config (line 1738)
**Issue**: Asserts fingerprint length and character set but doesn't verify determinism
**Evidence**: `assert len(fingerprint) == 64` and `assert all(c in "0123456789abcdef" for c in fingerprint)` - doesn't verify that same input produces same fingerprint.
**Fix**: Add second call with same secret and assert fingerprints match: `fingerprint2 = resolve_config(load_settings(config_file))["datasource"]["options"]["api_key_fingerprint"]` then `assert fingerprint == fingerprint2`.
**Priority**: P1

### Test: test_checkpoint_settings_validation (line 574)
**Issue**: Assertion-free test masquerading as validation
**Evidence**: Comment says "every_n requires checkpoint_interval" but only tests that ValidationError is raised. Doesn't verify error message mentions the missing field.
**Fix**: Capture exception and assert error message contains "checkpoint_interval" and "required".
**Priority**: P2

### Test: test_coalesce_settings_quorum_requires_count (line 1376)
**Issue**: Weak regex match on error message
**Evidence**: `with pytest.raises(ValidationError, match="quorum_count")` - this passes if "quorum_count" appears ANYWHERE in error, even in unrelated validation.
**Fix**: Access `exc_info.value.errors()` and verify specific error has `loc=("quorum_count",)` and appropriate error type.
**Priority**: P3

### Test: test_nested_secrets_are_fingerprinted (line 1998)
**Issue**: Direct testing of private function `_fingerprint_secrets` instead of public API
**Evidence**: `from elspeth.core.config import _fingerprint_secrets` - testing implementation detail, not contract.
**Fix**: Test via `resolve_config()` with nested config structure. Remove direct `_fingerprint_secrets` import.
**Priority**: P2

## Misclassified Tests

### Test: TestSecretFieldFingerprinting.test_nested_secrets_are_fingerprinted (line 1998)
**Issue**: Unit test for private function mixed into integration test class
**Evidence**: Directly imports and tests `_fingerprint_secrets()` private function, while other tests in class use `load_settings()` + `resolve_config()`.
**Fix**: Move to separate `TestFingerprintSecretsUnit` class or eliminate entirely and test behavior via public API only.
**Priority**: P2

### Test: TestExpandTemplateFiles class (lines 2418-2628)
**Issue**: Integration tests for file I/O mixed with config validation tests
**Evidence**: All tests use `tmp_path` fixture to create files, read them, and verify expansion. These are filesystem integration tests, not pure config validation.
**Fix**: Move to `tests/integration/test_config_template_expansion.py` or create `tests/core/test_config_template_files.py` (clearly separated).
**Priority**: P3

### Test: TestLoadSettings.test_load_with_env_override (line 131)
**Issue**: Configuration precedence test doesn't verify full precedence chain
**Evidence**: Only tests "env var overrides YAML" but CLAUDE.md specifies 5-layer precedence. This is a partial integration test masquerading as comprehensive.
**Fix**: Create dedicated `TestConfigurationPrecedence` class with property-based tests using all 5 layers (runtime overrides, suite config, profile, pack defaults, system defaults).
**Priority**: P0

### Test: test_resolve_config_json_serializable (line 836)
**Issue**: Testing JSON serialization, not config resolution logic
**Evidence**: `json_str = json.dumps(resolved)` - this tests Python's json module, not our config code.
**Fix**: Move to `TestResolveConfigOutput` class focused on output format contracts, or strengthen to verify specific canonicalization requirements (e.g., no NaN/Infinity in output per CLAUDE.md).
**Priority**: P3

## Infrastructure Gaps

### Gap: No fixture for clean config environment
**Issue**: Each test using `monkeypatch` manually manages environment isolation
**Evidence**: 15+ tests import `monkeypatch` and call `setenv`/`delenv` individually. Repeated setup code.
**Fix**: Create `@pytest.fixture` named `clean_config_env` that yields a dict-like object and guarantees cleanup:
```python
@pytest.fixture
def clean_config_env(monkeypatch):
    # Clear all ELSPETH_ env vars
    for key in list(os.environ.keys()):
        if key.startswith("ELSPETH_"):
            monkeypatch.delenv(key, raising=False)
    yield monkeypatch
```
**Priority**: P1

### Gap: No property-based tests for configuration precedence
**Issue**: CLAUDE.md specifies 5-layer precedence but tests only verify individual layers
**Evidence**: No test verifies "runtime overrides > suite > profile > pack defaults > system" ordering with all 5 layers present.
**Fix**: Add Hypothesis-based property test:
```python
@given(
    runtime=st.dictionaries(st.text(), st.text()),
    suite=st.dictionaries(st.text(), st.text()),
    # ... etc for all 5 layers
)
def test_precedence_ordering(runtime, suite, profile, pack, system):
    # Verify runtime value wins when all layers define same key
```
**Priority**: P0

### Gap: No mutation testing for frozen settings
**Issue**: Multiple tests verify settings are frozen (lines 35, 93, 204, 304, 598, 734, 910, 1520) but use same pattern without verifying ALL fields are frozen
**Evidence**: Each test picks one field arbitrarily. Doesn't verify that adding a new mutable field would be caught.
**Fix**: Create parametrized test that iterates all fields on each model:
```python
@pytest.mark.parametrize("field_name", ["url", "pool_size"])
def test_database_settings_all_fields_frozen(field_name):
    settings = DatabaseSettings(url="sqlite:///test.db")
    with pytest.raises(ValidationError):
        setattr(settings, field_name, "new_value")
```
**Priority**: P2

### Gap: No tests verifying crash-on-invalid-config semantics from CLAUDE.md
**Issue**: CLAUDE.md states "Bad data in the audit trail = crash immediately. No coercion, no defaults, no silent recovery." Tests verify ValidationError is raised but don't verify that error propagates (doesn't get caught/logged).
**Evidence**: All validation tests use `with pytest.raises(ValidationError)` which only verifies exception is raised, not that it crashes the process.
**Fix**: Add integration tests that verify `load_settings()` with invalid config exits process (not just raises):
```python
def test_invalid_config_crashes_process(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("invalid: config")
    # Use subprocess to verify process exits non-zero
    result = subprocess.run([...], capture_output=True)
    assert result.returncode != 0
```
**Priority**: P1

### Gap: No tests for config loaded from all 5 precedence sources simultaneously
**Issue**: Individual layer tests exist but no test loads config from runtime + suite + profile + pack + system and verifies correct resolution
**Evidence**: `test_load_readme_example` (line 376) loads from single YAML file. No test constructs full 5-layer config tree.
**Fix**: Create fixture that builds multi-layer config directory structure and test loading from all layers.
**Priority**: P0

### Gap: Repeated YAML string construction in test methods
**Issue**: 40+ tests manually write YAML strings with repeated boilerplate (datasource, sinks, output_sink)
**Evidence**: Lines 112-125, 135-142, 152-162, 170-173, etc. all duplicate minimal config structure.
**Fix**: Create fixture factory:
```python
@pytest.fixture
def yaml_config_factory(tmp_path):
    def _factory(overrides: dict[str, Any]) -> Path:
        config = {
            "datasource": {"plugin": "csv"},
            "sinks": {"output": {"plugin": "csv"}},
            "output_sink": "output",
            **overrides
        }
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.dump(config))
        return path
    return _factory
```
**Priority**: P2

### Gap: No tests for configuration validation order
**Issue**: Pydantic validates fields in order. Tests don't verify that validation fails fast on first error (not accumulating all errors).
**Evidence**: Tests using `exc_info.value.errors()` (line 626) check error count but don't verify validation stops at first structural error.
**Fix**: Create test with multiple validation errors and verify only the first/most critical is reported (fail-fast semantics).
**Priority**: P3

### Gap: Secret fingerprinting tests don't verify fingerprint key rotation
**Issue**: Tests use static `"test-key"` but don't verify that changing fingerprint key changes fingerprints (defense against key exposure)
**Evidence**: All fingerprint tests use same `monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key")`. No test verifies different keys produce different fingerprints.
**Fix**: Add test:
```python
def test_different_fingerprint_keys_produce_different_fingerprints():
    config1 = resolve_with_key("key1", secret="same_secret")
    config2 = resolve_with_key("key2", secret="same_secret")
    assert config1["fingerprint"] != config2["fingerprint"]
```
**Priority**: P1

## Missing Test Coverage

### Missing: No tests for configuration defaults matching CLAUDE.md specification
**Issue**: CLAUDE.md specifies configuration precedence with 5 layers but no test verifies system defaults exist and match documented values
**Evidence**: Test `test_concurrency_settings_default` (line 356) verifies default is 4, but no central test validates all system defaults.
**Fix**: Create `test_system_defaults_match_specification()` that loads minimal config and asserts all defaults match CLAUDE.md.
**Priority**: P1

### Missing: No tests for Dynaconf behavior (multi-source precedence)
**Issue**: CLAUDE.md says "Dynaconf + Pydantic with multi-source precedence" but no tests verify Dynaconf-specific features (e.g., environment-specific overrides, dotenv loading)
**Evidence**: Only one test mentions env var override (line 131). No tests for `.env` files, `DYNACONF_` prefixes, or environment switching.
**Fix**: Add tests for Dynaconf features:
- Loading from `.env` files
- `DYNACONF_ELSPETH__` prefix handling
- Environment-specific config (dev/staging/prod)
**Priority**: P2

### Missing: No tests verifying NaN/Infinity rejection in config values
**Issue**: CLAUDE.md states "NaN and Infinity are strictly rejected" for canonical JSON. No tests verify config validation rejects these values.
**Evidence**: Secret fingerprinting tests exist (using HMAC SHA256) but no tests verify that config fields containing NaN/Inf fail validation.
**Fix**: Add test:
```python
def test_config_rejects_nan_in_numeric_fields():
    with pytest.raises(ValidationError):
        RetrySettings(initial_delay_seconds=float('nan'))
```
**Priority**: P2

### Missing: No tests for configuration change detection (immutability after load)
**Issue**: Tests verify settings are frozen at creation but don't verify that reloading same file produces identical hash (for audit trail)
**Evidence**: Frozen tests (lines 35, 93, etc.) only test field assignment. No test verifies `resolve_config()` produces deterministic output for audit hashing.
**Fix**: Add test that loads same config twice and verifies resolved dicts are identical (for canonical JSON hashing).
**Priority**: P2

## Positive Observations

### Comprehensive Pydantic validation coverage
Tests thoroughly verify validation errors for positive/negative constraints (pool_size > 0, max_workers > 0, etc.). Good use of `pytest.raises(ValidationError)`.

### Good use of tmp_path fixture
File-based tests consistently use `tmp_path` for isolation. No pollution of system directories.

### Secret fingerprinting separation verified
Tests correctly verify that `load_settings()` preserves secrets for runtime while `resolve_config()` fingerprints for audit trail (lines 1711-1995).

### Gate validation thoroughness
Gate configuration tests (lines 879-1154) cover complex validation: condition syntax checking, forbidden constructs, reserved labels, boolean vs non-boolean condition routing.

### Coalesce validation completeness
Coalesce tests (lines 1355-1701) verify cross-field validation (quorum requires count, select requires branch, etc.). Good coverage of policy-specific requirements.
