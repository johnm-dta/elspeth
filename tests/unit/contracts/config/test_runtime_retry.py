# tests/unit/contracts/config/test_runtime_retry.py
"""Tests for RuntimeRetryConfig.

Retry-specific tests only. Common tests (frozen, slots, protocol, orphan fields)
are in test_runtime_common.py.
"""

from types import MappingProxyType

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

        actual_mappings: dict[str, str] | MappingProxyType[str, str] = FIELD_MAPPINGS.get("RetrySettings", {})

        assert actual_mappings == expected_mappings, (
            f"RetrySettings field mappings incorrect.\nExpected: {expected_mappings}\nActual: {actual_mappings}"
        )

    def test_same_name_fields_not_in_mapping(self) -> None:
        """Fields with same name in Settings and Runtime should NOT be in FIELD_MAPPINGS."""
        from elspeth.contracts.config import FIELD_MAPPINGS

        mappings: dict[str, str] | MappingProxyType[str, str] = FIELD_MAPPINGS.get("RetrySettings", {})

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

    def test_from_policy_rejects_zero_max_attempts(self) -> None:
        """from_policy() should reject max_attempts=0 instead of silently clamping."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts must be >= 1"):
            RuntimeRetryConfig.from_policy({"max_attempts": 0})

    def test_from_policy_rejects_zero_base_delay(self) -> None:
        """from_policy() should reject base_delay=0.0 instead of silently clamping."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay must be >= 0\.01"):
            RuntimeRetryConfig.from_policy({"base_delay": 0.0})

    def test_from_policy_rejects_zero_max_delay(self) -> None:
        """from_policy() should reject max_delay=0.0 instead of silently clamping."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_delay must be >= 0\.1"):
            RuntimeRetryConfig.from_policy({"max_delay": 0.0})

    def test_from_policy_rejects_negative_jitter(self) -> None:
        """from_policy() should reject jitter=-1.0 instead of silently clamping."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"jitter must be >= 0\.0"):
            RuntimeRetryConfig.from_policy({"jitter": -1.0})

    def test_from_policy_rejects_low_exponential_base(self) -> None:
        """from_policy() should reject exponential_base=0.5 instead of silently clamping."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"exponential_base must be > 1\.0"):
            RuntimeRetryConfig.from_policy({"exponential_base": 0.5})

    def test_from_policy_rejects_exponential_base_exactly_one(self) -> None:
        """from_policy() should reject exponential_base=1.0 — linear backoff is not exponential."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"exponential_base must be > 1\.0"):
            RuntimeRetryConfig.from_policy({"exponential_base": 1.0})


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
            RuntimeRetryConfig.from_policy({"max_attempts": None})  # type: ignore[arg-type]  # Intentional: testing validation

    def test_from_policy_rejects_none_base_delay(self) -> None:
        """None value for base_delay should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*numeric.*None"):
            RuntimeRetryConfig.from_policy({"base_delay": None})  # type: ignore[arg-type]  # Intentional: testing validation

    def test_from_policy_rejects_non_numeric_string(self) -> None:
        """Non-numeric string should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts.*numeric.*'abc'"):
            RuntimeRetryConfig.from_policy({"max_attempts": "abc"})  # type: ignore[arg-type]  # Intentional: testing validation

    def test_from_policy_rejects_list_value(self) -> None:
        """List value should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_attempts.*numeric.*list"):
            RuntimeRetryConfig.from_policy({"max_attempts": [1, 2, 3]})  # type: ignore[arg-type]  # Intentional: testing validation

    def test_from_policy_rejects_dict_value(self) -> None:
        """Dict value should raise ValueError with clear message."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*numeric.*dict"):
            RuntimeRetryConfig.from_policy({"base_delay": {"value": 1.0}})  # type: ignore[arg-type]  # Intentional: testing validation

    def test_from_policy_accepts_numeric_string(self) -> None:
        """Numeric string like '3' should be coerced to int."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.from_policy({"max_attempts": "3"})  # type: ignore[arg-type]  # Intentional: testing validation
        assert config.max_attempts == 3

    def test_from_policy_accepts_float_string(self) -> None:
        """Float string like '2.5' should be coerced to float."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig.from_policy({"base_delay": "2.5"})  # type: ignore[arg-type]  # Intentional: testing validation
        assert config.base_delay == 2.5

    def test_from_policy_multiple_invalid_fields_reports_first(self) -> None:
        """Multiple invalid fields should report at least one clearly."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        # Either field error is acceptable - just ensure we get a clear message
        with pytest.raises(ValueError, match=r"(max_attempts|base_delay).*numeric"):
            RuntimeRetryConfig.from_policy(
                {  # type: ignore[arg-type]  # Intentional: testing validation
                    "max_attempts": None,
                    "base_delay": "not-a-number",
                }
            )


class TestRuntimeRetryRejectsNonFiniteFloats:
    """P0-06: NaN and Infinity must be rejected by float validation.

    Non-finite floats pass through to runtime config and eventually reach
    the canonical JSON layer (RFC 8785), which crashes during audit recording.
    Catching them at validation time gives actionable error messages.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_from_policy_rejects_non_finite_base_delay(self, value: float) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*finite"):
            RuntimeRetryConfig.from_policy({"base_delay": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_from_policy_rejects_non_finite_max_delay(self, value: float) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_delay.*finite"):
            RuntimeRetryConfig.from_policy({"max_delay": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_from_policy_rejects_non_finite_jitter(self, value: float) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"jitter.*finite"):
            RuntimeRetryConfig.from_policy({"jitter": value})

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_from_policy_rejects_non_finite_exponential_base(self, value: float) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"exponential_base.*finite"):
            RuntimeRetryConfig.from_policy({"exponential_base": value})

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity"])
    def test_from_policy_rejects_non_finite_string_coercion(self, value: str) -> None:
        """Strings like 'nan' and 'inf' are valid for float() but must be rejected."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay.*finite"):
            RuntimeRetryConfig.from_policy({"base_delay": value})  # type: ignore[arg-type]


class TestRuntimeRetryValidation:
    """Test __post_init__ validation in RuntimeRetryConfig.

    All fields are validated at construction time with clear error messages.
    No silent clamping — invalid values are rejected immediately.
    """

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

    def test_base_delay_must_be_positive(self) -> None:
        """base_delay < 0.01 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"base_delay must be >= 0\.01"):
            RuntimeRetryConfig(
                max_attempts=3,
                base_delay=0.0,
                max_delay=60.0,
                jitter=1.0,
                exponential_base=2.0,
            )

    def test_max_delay_must_be_positive(self) -> None:
        """max_delay < 0.1 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"max_delay must be >= 0\.1"):
            RuntimeRetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=0.0,
                jitter=1.0,
                exponential_base=2.0,
            )

    def test_jitter_must_be_non_negative(self) -> None:
        """jitter < 0 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"jitter must be >= 0\.0"):
            RuntimeRetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=60.0,
                jitter=-0.5,
                exponential_base=2.0,
            )

    def test_exponential_base_must_exceed_one(self) -> None:
        """exponential_base <= 1.0 should raise ValueError."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match=r"exponential_base must be > 1\.0"):
            RuntimeRetryConfig(
                max_attempts=3,
                base_delay=1.0,
                max_delay=60.0,
                jitter=1.0,
                exponential_base=1.0,
            )

    def test_max_attempts_rejects_bool(self) -> None:
        """max_attempts=True should raise TypeError — bool is not int."""
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(TypeError, match="max_attempts must be int"):
            RuntimeRetryConfig(
                max_attempts=True,
                base_delay=1.0,
                max_delay=60.0,
                jitter=1.0,
                exponential_base=2.0,
            )


class TestRuntimeRetryNonFiniteRejection:
    """Non-finite float fields must be rejected at construction time."""

    @pytest.mark.parametrize("field", ["base_delay", "max_delay", "jitter", "exponential_base"])
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_rejects_non_finite_float(self, field: str, bad_value: float) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        valid: dict[str, int | float] = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "jitter": 1.0,
            "exponential_base": 2.0,
        }
        valid[field] = bad_value
        with pytest.raises(ValueError, match=f"{field} must be finite"):
            RuntimeRetryConfig(**valid)  # type: ignore[arg-type]

    def test_finite_floats_accepted(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        config = RuntimeRetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=60.0,
            jitter=1.0,
            exponential_base=2.0,
        )
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0


class TestValidateIntFieldFractionalFloats:
    """Regression: _validate_int_field must reject non-integral floats."""

    def test_fractional_float_rejected(self) -> None:
        from elspeth.contracts.config.runtime import _validate_int_field

        with pytest.raises(ValueError, match="must be an integer"):
            _validate_int_field("max_attempts", 2.9)

    def test_integral_float_accepted(self) -> None:
        from elspeth.contracts.config.runtime import _validate_int_field

        assert _validate_int_field("max_attempts", 3.0) == 3

    def test_fractional_float_in_from_policy_rejected(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRetryConfig

        with pytest.raises(ValueError, match="must be an integer"):
            RuntimeRetryConfig.from_policy({"max_attempts": 2.7})  # type: ignore[arg-type]  # deliberate: tests rejection of fractional float in int field
