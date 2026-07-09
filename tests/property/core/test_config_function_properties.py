# tests/property/core/test_config_function_properties.py
"""Property-based tests for targeted config.py functions.

These tests focus on the data-transformation functions within config.py
that operate at the trust boundary (user YAML → internal representation):

1. _expand_env_vars() - Environment variable expansion with ${VAR:-default}
2. _sanitize_dsn() - Database URL password removal with fingerprinting
3. is_secret_field() - Secret field name detection (lives in core.secrets)
4. _fingerprint_secrets() - Recursive secret replacement

Properties tested:
- Expansion idempotency and no double-expansion
- Structure preservation (keys unchanged, non-strings pass through)
- Password removal completeness in DSNs
- Secret detection completeness and no false positives
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.core.config import _expand_env_vars
from elspeth.core.secrets import (
    SECRET_FIELD_NAMES,
    SECRET_FIELD_SUFFIXES,
    STRUCTURAL_FIELD_EXEMPTIONS,
    collect_credential_field_violations,
    is_secret_field,
)

# =============================================================================
# Strategies
# =============================================================================

# Env var names (uppercase with underscores)
env_var_names = st.text(
    min_size=2,
    max_size=20,
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789",
).filter(lambda s: s[0].isalpha())

# Default values for env var syntax
default_values = st.text(
    min_size=0,
    max_size=50,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P"),
        blacklist_characters="}",
    ),
)

# Non-string primitive values
non_string_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000),
    st.booleans(),
    st.none(),
)

# Config dict keys (lowercase with underscores, like YAML keys)
config_keys = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
).filter(lambda s: s[0].isalpha())

# Simple string values with no env var references
plain_strings = st.text(
    min_size=0,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P"), blacklist_characters="${}"),
)

# Field names that should NOT be detected as secrets
non_secret_field_names = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
).filter(lambda s: s.lower() not in SECRET_FIELD_NAMES and not s.lower().endswith(SECRET_FIELD_SUFFIXES))


# =============================================================================
# _expand_env_vars Properties
# =============================================================================


class TestExpandEnvVarsStructurePreservation:
    """Config structure must be preserved during expansion."""

    @given(
        keys=st.lists(config_keys, min_size=1, max_size=5, unique=True),
        values=st.lists(non_string_values, min_size=1, max_size=5),
    )
    @settings(max_examples=200)
    def test_non_string_values_pass_through(self, keys: list[str], values: list[Any]) -> None:
        """Property: Non-string values (int, float, bool, None) are never modified."""
        # Pad values to match keys length
        padded = (values * ((len(keys) // len(values)) + 1))[: len(keys)]
        config = dict(zip(keys, padded, strict=False))
        result = _expand_env_vars(config)
        assert result == config

    @given(
        keys=st.lists(config_keys, min_size=1, max_size=5, unique=True),
        values=st.lists(plain_strings, min_size=1, max_size=5),
    )
    @settings(max_examples=200)
    def test_strings_without_env_refs_unchanged(self, keys: list[str], values: list[str]) -> None:
        """Property: Strings without ${...} patterns are not modified."""
        padded = (values * ((len(keys) // len(values)) + 1))[: len(keys)]
        config = dict(zip(keys, padded, strict=False))
        result = _expand_env_vars(config)
        assert result == config

    @given(
        key=config_keys,
        inner_keys=st.lists(config_keys, min_size=1, max_size=3, unique=True),
        inner_values=st.lists(non_string_values, min_size=1, max_size=3),
    )
    @settings(max_examples=100)
    def test_nested_dicts_preserved(self, key: str, inner_keys: list[str], inner_values: list[Any]) -> None:
        """Property: Nested dict structure is preserved during expansion."""
        padded = (inner_values * ((len(inner_keys) // len(inner_values)) + 1))[: len(inner_keys)]
        config = {key: dict(zip(inner_keys, padded, strict=False))}
        result = _expand_env_vars(config)
        assert result == config

    @given(
        key=config_keys,
        values=st.lists(non_string_values, min_size=0, max_size=5),
    )
    @settings(max_examples=100)
    def test_lists_preserved(self, key: str, values: list[Any]) -> None:
        """Property: Lists are preserved during expansion."""
        config = {key: values}
        result = _expand_env_vars(config)
        assert result == config

    @given(config=st.dictionaries(config_keys, non_string_values, max_size=5))
    @settings(max_examples=100)
    def test_keys_never_modified(self, config: dict[str, Any]) -> None:
        """Property: Dict keys are NEVER modified by expansion."""
        result = _expand_env_vars(config)
        assert set(result.keys()) == set(config.keys())


class TestExpandEnvVarsExpansion:
    """Environment variable expansion behavior."""

    @given(var_name=env_var_names, var_value=plain_strings)
    @settings(max_examples=200)
    def test_set_env_var_expands(self, var_name: str, var_value: str) -> None:
        """Property: ${VAR} expands to env var value when set."""
        config = {"field": f"${{{var_name}}}"}
        with patch.dict(os.environ, {var_name: var_value}, clear=False):
            result = _expand_env_vars(config)
        assert result["field"] == var_value

    @given(var_name=env_var_names, default=default_values)
    @settings(max_examples=200)
    def test_missing_var_uses_default(self, var_name: str, default: str) -> None:
        """Property: ${VAR:-default} uses default when var is unset."""
        config = {"field": f"${{{var_name}:-{default}}}"}
        env = dict(os.environ)
        env.pop(var_name, None)
        with patch.dict(os.environ, env, clear=True):
            result = _expand_env_vars(config)
        assert result["field"] == default

    @given(var_name=env_var_names, var_value=plain_strings, default=default_values)
    @settings(max_examples=100)
    def test_set_var_ignores_default(self, var_name: str, var_value: str, default: str) -> None:
        """Property: ${VAR:-default} uses var value (not default) when var is set."""
        config = {"field": f"${{{var_name}:-{default}}}"}
        with patch.dict(os.environ, {var_name: var_value}, clear=False):
            result = _expand_env_vars(config)
        assert result["field"] == var_value

    @given(var_name=env_var_names)
    @settings(max_examples=100)
    def test_missing_var_no_default_raises(self, var_name: str) -> None:
        """Property: ${VAR} without default raises ValueError when var is unset."""
        config = {"field": f"${{{var_name}}}"}
        env = dict(os.environ)
        env.pop(var_name, None)
        with patch.dict(os.environ, env, clear=True), pytest.raises(ValueError, match="not set"):
            _expand_env_vars(config)

    @given(var_name=env_var_names)
    @settings(max_examples=50)
    def test_no_double_expansion(self, var_name: str) -> None:
        """Property: Expansion is single-pass — expanded values are not re-expanded.

        If FOO expands to '${BAR}', BAR is NOT expanded in the result.
        """
        config = {"field": f"${{{var_name}}}"}
        # Set var to a value that looks like another env var reference
        with patch.dict(os.environ, {var_name: "${SHOULD_NOT_EXPAND:-nope}"}, clear=False):
            result = _expand_env_vars(config)
        # The literal string "${SHOULD_NOT_EXPAND:-nope}" should be in the result
        assert result["field"] == "${SHOULD_NOT_EXPAND:-nope}"

    @given(
        var_name=env_var_names,
        var_value=plain_strings,
        prefix=plain_strings,
        suffix=plain_strings,
    )
    @settings(max_examples=100)
    def test_embedded_expansion(self, var_name: str, var_value: str, prefix: str, suffix: str) -> None:
        """Property: ${VAR} embedded in larger string expands correctly."""
        config = {"field": f"{prefix}${{{var_name}}}{suffix}"}
        with patch.dict(os.environ, {var_name: var_value}, clear=False):
            result = _expand_env_vars(config)
        assert result["field"] == f"{prefix}{var_value}{suffix}"


class TestExpandEnvVarsIdempotency:
    """Expansion with only defaults should be idempotent."""

    @given(default=default_values)
    @settings(max_examples=100)
    def test_idempotent_when_all_defaults_used(self, default: str) -> None:
        """Property: If result has no ${...} patterns, re-expanding gives same result.

        After first expansion, the result is a plain string (defaults filled in).
        Expanding again should be a no-op since there are no more ${...} patterns.
        """
        # Use a var name that's definitely not in env
        config = {"field": f"${{__DEFINITELY_NOT_SET_12345:-{default}}}"}
        env = dict(os.environ)
        env.pop("__DEFINITELY_NOT_SET_12345", None)
        with patch.dict(os.environ, env, clear=True):
            result1 = _expand_env_vars(config)
            result2 = _expand_env_vars(result1)
        assert result1 == result2


# =============================================================================
# is_secret_field Properties
# =============================================================================


class TestIsSecretFieldExactMatches:
    """All names in SECRET_FIELD_NAMES must be detected."""

    @given(name=st.sampled_from(sorted(SECRET_FIELD_NAMES)))
    @settings(max_examples=50)
    def test_exact_names_detected(self, name: str) -> None:
        """Property: Every name in SECRET_FIELD_NAMES returns True."""
        assert is_secret_field(name) is True

    @given(
        name=st.sampled_from(sorted(SECRET_FIELD_NAMES)),
        case_fn=st.sampled_from([str.upper, str.lower, str.title, str.swapcase]),
    )
    @settings(max_examples=100)
    def test_exact_names_case_insensitive(self, name: str, case_fn) -> None:
        """Property: Exact name matches are case-insensitive."""
        assert is_secret_field(case_fn(name)) is True


class TestIsSecretFieldSuffixMatches:
    """All fields ending in secret suffixes must be detected."""

    @given(
        prefix=st.text(
            min_size=1,
            max_size=20,
            alphabet="abcdefghijklmnopqrstuvwxyz_",
        ).filter(lambda s: s[0].isalpha()),
        suffix=st.sampled_from(list(SECRET_FIELD_SUFFIXES)),
    )
    @settings(max_examples=200)
    def test_suffix_matches_detected(self, prefix: str, suffix: str) -> None:
        """Property: Any field ending in a secret suffix returns True.

        Exact-name structural exemptions (``STRUCTURAL_FIELD_EXEMPTIONS``)
        are the deliberate exception and are covered by their own class.
        """
        name = f"{prefix}{suffix}"
        assume(name.lower() not in STRUCTURAL_FIELD_EXEMPTIONS)
        assert is_secret_field(name) is True

    @given(
        prefix=st.text(
            min_size=1,
            max_size=20,
            alphabet="abcdefghijklmnopqrstuvwxyz_",
        ).filter(lambda s: s[0].isalpha()),
        suffix=st.sampled_from(list(SECRET_FIELD_SUFFIXES)),
        case_fn=st.sampled_from([str.upper, str.lower, str.title]),
    )
    @settings(max_examples=100)
    def test_suffix_matches_case_insensitive(self, prefix: str, suffix: str, case_fn) -> None:
        """Property: Suffix matches are case-insensitive."""
        name = f"{prefix}{suffix}"
        assume(name.lower() not in STRUCTURAL_FIELD_EXEMPTIONS)
        field_name = case_fn(name)
        assert is_secret_field(field_name) is True


class TestIsSecretFieldNoFalsePositives:
    """Non-secret field names must NOT be detected."""

    @given(name=non_secret_field_names)
    @settings(max_examples=300)
    def test_non_secret_fields_not_detected(self, name: str) -> None:
        """Property: Fields that don't match exact names or suffixes return False."""
        assert is_secret_field(name) is False

    def test_partial_suffix_not_detected(self) -> None:
        """Edge case: Partial suffix matches don't trigger detection.

        '_key' is a suffix, but '_keynote' should not match because
        'keynote' doesn't end with '_key' — it ends with '_keynote'.
        Wait — actually 'my_keynote' DOES end with 'e', not '_key'.
        Let's verify the actual field names that should NOT match.
        """
        # These should NOT match because they don't end with exact suffixes
        assert is_secret_field("keyboard") is False
        assert is_secret_field("token_count") is False
        assert is_secret_field("password_length") is False
        assert is_secret_field("secret_sauce") is False

    def test_empty_string_not_detected(self) -> None:
        """Edge case: Empty string is not a secret field."""
        assert is_secret_field("") is False


class TestStructuralFieldExemptions:
    """Exact-name structural fields are exempt from the suffix heuristic.

    Regression fixtures for elspeth-61f2c0732e: the bare ``_key`` suffix
    false-positived on structural plugin fields (``JSONSource.data_key``,
    ``DataverseSink.alternate_key``), blocking /validate and /execute for
    legitimate secret-free pipelines. The fix is a fail-closed exact-name
    exemption list, NOT a narrower suffix set — real credential fields such
    as Langfuse tracing's ``secret_key``/``public_key`` are matched *only*
    by the bare ``_key`` suffix, so weakening the suffix would silently stop
    fingerprinting them into the audit trail.
    """

    @given(name=st.sampled_from(sorted(STRUCTURAL_FIELD_EXEMPTIONS)))
    @settings(max_examples=50)
    def test_exempt_names_not_secret(self, name: str) -> None:
        """Property: Every exempted structural name returns False."""
        assert is_secret_field(name) is False

    @given(
        name=st.sampled_from(sorted(STRUCTURAL_FIELD_EXEMPTIONS)),
        case_fn=st.sampled_from([str.upper, str.lower, str.title, str.swapcase]),
    )
    @settings(max_examples=50)
    def test_exempt_names_case_insensitive(self, name: str, case_fn) -> None:
        """Property: Exemptions apply to any casing, matching the predicate."""
        assert is_secret_field(case_fn(name)) is False

    @given(name=st.sampled_from(sorted(STRUCTURAL_FIELD_EXEMPTIONS)))
    @settings(max_examples=50)
    def test_exemptions_stay_purposeful(self, name: str) -> None:
        """Property: Every exemption must be a name the suffix heuristic would
        otherwise catch, and must not collide with an exact secret name.

        An entry failing the first check is dead weight; one failing the
        second would silently un-protect a known credential field.
        """
        assert name == name.lower()
        assert name.endswith(SECRET_FIELD_SUFFIXES)
        assert name not in SECRET_FIELD_NAMES

    def test_known_structural_fields_exempt(self) -> None:
        """The two shipped structural fields that motivated the exemption."""
        # JSONSource.data_key / azure_blob_source data_key: JSON extraction key
        assert is_secret_field("data_key") is False
        # DataverseSink.alternate_key: upsert routing column name
        assert is_secret_field("alternate_key") is False

    def test_real_credential_key_fields_still_detected(self) -> None:
        """Leak-regression guard: credential fields that rely on the bare
        ``_key`` suffix (or other heuristics) must keep matching."""
        for field in (
            "secret_key",  # Langfuse tracing — bare _key only
            "public_key",  # Langfuse tracing — bare _key only
            "master_key",
            "signing_key",
            "sas_key",
            "account_key",
            "api_key",
            "access_token",
            "client_secret",
            "db_password",
            "authorization",
            "connection_string",
        ):
            assert is_secret_field(field) is True, field

    def test_structural_literal_is_not_a_credential_violation(self) -> None:
        """The reported repro: literal structural values must not be flagged
        as fabricated credentials by the shared collector."""
        assert collect_credential_field_violations({"data_key": "results"}) == []
        assert collect_credential_field_violations({"alternate_key": "crabc_code"}) == []

    def test_credential_literal_still_a_violation(self) -> None:
        """Positive control: a literal in a true credential field still trips."""
        assert collect_credential_field_violations({"api_key": "sk-live-1234"}) == ["api_key"]
        assert collect_credential_field_violations({"secret_key": "sk-lf-1234"}) == ["secret_key"]


# =============================================================================
# _sanitize_dsn Properties
# =============================================================================


class TestSanitizeDsnProperties:
    """DSN sanitization must remove passwords completely."""

    def test_password_removed_from_postgresql_dsn(self) -> None:
        """Property: Password is removed from PostgreSQL DSN."""
        url = "postgresql://user:mysecretpassword@localhost:5432/mydb"  # secret-scan: allow-this-line
        # Need fingerprint key for this to work, or use dev mode
        sanitized, _fingerprint, had_password = _sanitize_dsn_dev_mode(url)
        assert "mysecretpassword" not in sanitized
        assert had_password is True

    def test_password_removed_from_mysql_dsn(self) -> None:
        """Property: Password is removed from MySQL DSN."""
        url = "mysql://admin:p4ssw0rd@db.host.com/production"  # secret-scan: allow-this-line
        sanitized, _fingerprint, had_password = _sanitize_dsn_dev_mode(url)
        assert "p4ssw0rd" not in sanitized
        assert had_password is True

    @given(
        password=st.text(
            min_size=11,
            max_size=50,
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
            ),
        ),
    )
    @settings(max_examples=200)
    def test_password_never_in_sanitized_url(self, password: str) -> None:
        """Property: For ANY password string, sanitized URL does not contain it.

        Uses min_size=11 because shorter passwords may coincidentally appear in
        the URL structure (e.g., 'postgres' in 'postgresql://', 'localhost' in
        the host). Real passwords are always longer, so this is not a meaningful
        security gap.
        """
        url = f"postgresql://user:{password}@localhost/db"  # secret-scan: allow-this-line
        sanitized, _, had_password = _sanitize_dsn_dev_mode(url)
        assert had_password is True
        assert password not in sanitized

    @given(
        host=st.sampled_from(["localhost", "db.example.com", "10.0.0.1", "my-host"]),
        port=st.integers(min_value=1, max_value=65535),
        database=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
    )
    @settings(max_examples=100)
    def test_non_password_components_preserved(self, host: str, port: int, database: str) -> None:
        """Property: Host, port, and database survive sanitization."""
        url = f"postgresql://user:secret@{host}:{port}/{database}"  # secret-scan: allow-this-line
        sanitized, _, _ = _sanitize_dsn_dev_mode(url)
        assert host in sanitized
        assert str(port) in sanitized
        assert database in sanitized

    def test_no_password_url_unchanged(self) -> None:
        """Property: URLs without passwords are returned as-is."""
        url = "sqlite:///path/to/db.sqlite"
        sanitized, fingerprint, had_password = _sanitize_dsn_dev_mode(url)
        assert sanitized == url
        assert fingerprint is None
        assert had_password is False

    def test_idempotent_sanitization(self) -> None:
        """Property: Sanitizing an already-sanitized URL is a no-op."""
        url = "postgresql://user:secret@localhost/db"  # secret-scan: allow-this-line
        sanitized1, _, _ = _sanitize_dsn_dev_mode(url)
        sanitized2, _, had_password = _sanitize_dsn_dev_mode(sanitized1)
        assert sanitized1 == sanitized2
        assert had_password is False

    def test_invalid_url_returned_as_is(self) -> None:
        """Property: Non-SQLAlchemy URLs are returned unchanged."""
        url = "not-a-valid-dsn"
        sanitized, fingerprint, had_password = _sanitize_dsn_dev_mode(url)
        assert sanitized == url
        assert fingerprint is None
        assert had_password is False

    @given(
        driver=st.sampled_from(["postgresql", "mysql", "mssql+pyodbc"]),
        username=st.text(min_size=1, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz"),
    )
    @settings(max_examples=50)
    def test_username_preserved(self, driver: str, username: str) -> None:
        """Property: Username survives sanitization."""
        url = f"{driver}://{username}:secret@localhost/db"
        sanitized, _, _ = _sanitize_dsn_dev_mode(url)
        assert username in sanitized


# =============================================================================
# Helpers
# =============================================================================


def _sanitize_dsn_dev_mode(url: str) -> tuple[str, str | None, bool]:
    """Call _sanitize_dsn in dev mode (no fingerprint key required)."""
    from elspeth.core.config import _sanitize_dsn

    return _sanitize_dsn(url, fail_if_no_key=False)
