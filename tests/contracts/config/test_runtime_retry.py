# tests/contracts/config/test_runtime_retry.py
"""Tests for RuntimeRetryConfig.

Retry-specific tests only. Common tests (frozen, slots, protocol, orphan fields)
are in test_runtime_common.py.
"""

import pytest

# NOTE: Protocol compliance, orphan detection, frozen/slots tests are in test_runtime_common.py


class TestRetryFieldNameMapping:
    """Verify field name mappings are correct and complete."""

    def test_retry_field_name_mapping(self) -> None:
        """Explicit assertion of all field name mappings."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        # These are the expected mappings (documented in task)
        expected_mappings = {
            "initial_delay_seconds": "base_delay",
            "max_delay_seconds": "max_delay",
        }

        actual_mappings = FIELD_MAPPINGS.get("RetrySettings", {})

        assert actual_mappings == expected_mappings, (
            f"RetrySettings field mappings incorrect.\nExpected: {expected_mappings}\nActual: {actual_mappings}"
        )

    def test_same_name_fields_not_in_mapping(self) -> None:
        """Fields with same name in Settings and Runtime should NOT be in FIELD_MAPPINGS."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        mappings = FIELD_MAPPINGS.get("RetrySettings", {})

        # These fields have the SAME name in both - should not be in mapping
        same_name_fields = {"max_attempts", "exponential_base"}

        for field in same_name_fields:
            assert field not in mappings, (
                f"Field '{field}' has same name in Settings and Runtime - should not be in FIELD_MAPPINGS (only list renames)"
            )


class TestRuntimeRetryFromSettings:
    """Test from_settings() factory method."""

    def test_from_settings_maps_all_fields(self) -> None:
        """from_settings() must map all Settings fields correctly."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        # Use non-default values to catch forgotten mappings
        settings = RetrySettings(
            max_attempts=99,
            initial_delay_seconds=99.0,
            max_delay_seconds=999.0,
            exponential_base=9.9,
        )

        config = RuntimeRetryConfig.from_settings(settings)

        # Verify field mapping
        assert config.max_attempts == 99, "max_attempts not mapped correctly"
        assert config.base_delay == 99.0, "initial_delay_seconds -> base_delay not mapped"
        assert config.max_delay == 999.0, "max_delay_seconds -> max_delay not mapped"
        assert config.exponential_base == 9.9, "exponential_base not mapped correctly"

    def test_from_settings_uses_internal_jitter(self) -> None:
        """from_settings() must use internal default for jitter."""
        from elspeth.contracts.config import INTERNAL_DEFAULTS
        from elspeth.contracts.config.runtime import RuntimeRetryConfig
        from elspeth.core.config import RetrySettings

        settings = RetrySettings()
        config = RuntimeRetryConfig.from_settings(settings)

        expected_jitter = INTERNAL_DEFAULTS["retry"]["jitter"]
        assert config.jitter == expected_jitter, f"jitter should be {expected_jitter} (from INTERNAL_DEFAULTS), got {config.jitter}"


class TestRuntimeRetryFromPolicy:
    """Test from_policy() factory method."""

    def test_from_policy_none_returns_no_retry(self) -> None:
        """from_policy(None) should return no_retry config."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.from_policy(None)

        assert config.max_attempts == 1, "from_policy(None) should return no_retry (max_attempts=1)"

    def test_from_policy_uses_provided_values(self) -> None:
        """from_policy() should use values from RetryPolicy dict."""
        from elspeth.contracts import RetryPolicy
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        policy: RetryPolicy = {
            "max_attempts": 5,
            "base_delay": 2.0,
            "max_delay": 120.0,
            "jitter": 0.5,
            "exponential_base": 3.0,
        }

        config = RuntimeRetryConfig.from_policy(policy)

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.5
        assert config.exponential_base == 3.0

    def test_from_policy_partial_uses_defaults(self) -> None:
        """from_policy() with partial dict should use POLICY_DEFAULTS."""
        from elspeth.contracts import RetryPolicy
        from elspeth.contracts.config import POLICY_DEFAULTS
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        # Only provide max_attempts, rest should use defaults
        policy: RetryPolicy = {"max_attempts": 7}

        config = RuntimeRetryConfig.from_policy(policy)

        assert config.max_attempts == 7
        assert config.base_delay == POLICY_DEFAULTS["base_delay"]
        assert config.max_delay == POLICY_DEFAULTS["max_delay"]
        assert config.jitter == POLICY_DEFAULTS["jitter"]
        assert config.exponential_base == POLICY_DEFAULTS["exponential_base"]

    def test_from_policy_clamps_invalid_values(self) -> None:
        """from_policy() should clamp values to safe minimums."""
        from elspeth.contracts import RetryPolicy
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        # Invalid values that need clamping
        policy: RetryPolicy = {
            "max_attempts": 0,  # Must be >= 1
            "base_delay": 0.0,  # Must be > 0
            "max_delay": 0.0,  # Must be > 0
            "jitter": -1.0,  # Must be >= 0
            "exponential_base": 0.5,  # Must be > 1
        }

        config = RuntimeRetryConfig.from_policy(policy)

        assert config.max_attempts >= 1, "max_attempts must be clamped to >= 1"
        assert config.base_delay > 0, "base_delay must be clamped to > 0"
        assert config.max_delay > 0, "max_delay must be clamped to > 0"
        assert config.jitter >= 0, "jitter must be clamped to >= 0"
        assert config.exponential_base > 1, "exponential_base must be clamped to > 1"


class TestRuntimeRetryConvenienceFactories:
    """Test convenience factory methods."""

    def test_default_uses_policy_defaults(self) -> None:
        """default() should create config with POLICY_DEFAULTS values."""
        from elspeth.contracts.config import POLICY_DEFAULTS
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.default()

        assert config.max_attempts == POLICY_DEFAULTS["max_attempts"]
        assert config.base_delay == POLICY_DEFAULTS["base_delay"]
        assert config.max_delay == POLICY_DEFAULTS["max_delay"]
        assert config.jitter == POLICY_DEFAULTS["jitter"]
        assert config.exponential_base == POLICY_DEFAULTS["exponential_base"]

    def test_no_retry_single_attempt(self) -> None:
        """no_retry() should create config with max_attempts=1."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.no_retry()

        assert config.max_attempts == 1, "no_retry should have max_attempts=1"


class TestFromPolicyTypeValidation:
    """Test from_policy() rejects malformed types with clear errors.

    P2-2026-01-21: from_policy() must reject non-numeric types with actionable
    error messages, not raw TypeError/ValueError from int()/float() conversion.
    """

    def test_from_policy_rejects_none_max_attempts(self) -> None:
        """None value for max_attempts should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts.*numeric.*None"):
            RuntimeRetryConfig.from_policy({"max_attempts": None})

    def test_from_policy_rejects_none_base_delay(self) -> None:
        """None value for base_delay should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*numeric.*None"):
            RuntimeRetryConfig.from_policy({"base_delay": None})

    def test_from_policy_rejects_non_numeric_string(self) -> None:
        """Non-numeric string should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts.*numeric.*'abc'"):
            RuntimeRetryConfig.from_policy({"max_attempts": "abc"})

    def test_from_policy_rejects_list_value(self) -> None:
        """List value should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts.*numeric.*list"):
            RuntimeRetryConfig.from_policy({"max_attempts": [1, 2, 3]})

    def test_from_policy_rejects_dict_value(self) -> None:
        """Dict value should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*numeric.*dict"):
            RuntimeRetryConfig.from_policy({"base_delay": {"value": 1.0}})

    def test_from_policy_accepts_numeric_string(self) -> None:
        """Numeric string like '3' should be coerced to int."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.from_policy({"max_attempts": "3"})
        assert config.max_attempts == 3

    def test_from_policy_accepts_float_string(self) -> None:
        """Float string like '2.5' should be coerced to float."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.from_policy({"base_delay": "2.5"})
        assert config.base_delay == 2.5

    def test_from_policy_multiple_invalid_fields_reports_first(self) -> None:
        """Multiple invalid fields should report at least one clearly."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        # Either field error is acceptable - just ensure we get a clear message
        with pytest.raises(ValueError, match=r"(max_attempts|base_delay).*numeric"):
            RuntimeRetryConfig.from_policy(
                {
                    "max_attempts": None,
                    "base_delay": "not-a-number",
                }
            )


class TestRuntimeRetryValidation:
    """Test validation in RuntimeRetryConfig."""

    def test_max_attempts_must_be_positive(self) -> None:
        """max_attempts < 1 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match="max_attempts must be >= 1"):
            RuntimeRetryConfig(
                max_attempts=0,
                base_delay=1.0,
                max_delay=60.0,
                jitter=1.0,
                exponential_base=2.0,
            )
