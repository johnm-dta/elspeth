# tests/engine/test_retry_policy.py
"""Tests for RetryPolicy TypedDict."""

from typing import ClassVar

from elspeth.contracts import RetryPolicy
from elspeth.contracts.config import POLICY_DEFAULTS, RuntimeRetryConfig


class TestRetryPolicy:
    """Verify RetryPolicy TypedDict works correctly."""

    def test_retry_policy_schema(self) -> None:
        """RetryPolicy should be a partial TypedDict with expected fields."""
        # Verify total=False (all fields optional)
        assert RetryPolicy.__total__ is False

        # Verify exact schema - all 5 fields present
        assert set(RetryPolicy.__annotations__) == {
            "max_attempts",
            "base_delay",
            "max_delay",
            "jitter",
            "exponential_base",
        }

    def test_retry_policy_importable(self) -> None:
        """RetryPolicy should be importable from contracts."""
        policy: RetryPolicy = {
            "max_attempts": 3,
            "base_delay": 1.0,
        }
        assert policy["max_attempts"] == 3

    def test_retry_config_from_policy_with_typed_dict(self) -> None:
        """RuntimeRetryConfig.from_policy should accept RetryPolicy."""
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

    def test_retry_policy_partial(self) -> None:
        """RetryPolicy should allow partial specification with correct defaults."""
        # Only specify some fields
        policy: RetryPolicy = {"max_attempts": 10}
        config = RuntimeRetryConfig.from_policy(policy)
        assert config.max_attempts == 10
        # Defaults for unspecified fields - verify ALL optional fields have defaults
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.jitter == 1.0
        assert config.exponential_base == 2.0

    def test_retry_policy_exponential_base(self) -> None:
        """RetryPolicy should support exponential_base configuration."""
        policy: RetryPolicy = {
            "max_attempts": 3,
            "exponential_base": 3.0,
        }
        config = RuntimeRetryConfig.from_policy(policy)
        assert config.exponential_base == 3.0

    def test_retry_policy_exponential_base_clamped(self) -> None:
        """exponential_base should be clamped to minimum 1.01."""
        # Invalid base < 1 should be clamped
        policy: RetryPolicy = {"exponential_base": 0.5}
        config = RuntimeRetryConfig.from_policy(policy)
        assert config.exponential_base >= 1.01

    def test_retry_policy_exponential_base_exactly_one_clamped(self) -> None:
        """exponential_base=1.0 should be clamped (would cause no backoff growth).

        A base of exactly 1.0 would mean 1^n = 1 for all n, resulting in
        constant delay instead of exponential backoff. This must be rejected.
        """
        policy: RetryPolicy = {"exponential_base": 1.0}
        config = RuntimeRetryConfig.from_policy(policy)
        assert config.exponential_base > 1.0, "exponential_base=1.0 must be clamped"
        assert config.exponential_base >= 1.01

    def test_retry_policy_exponential_base_negative_clamped(self) -> None:
        """Negative exponential_base should be clamped to minimum.

        Negative bases would cause alternating positive/negative delays
        which is nonsensical for retry timing.
        """
        policy: RetryPolicy = {"exponential_base": -5.0}
        config = RuntimeRetryConfig.from_policy(policy)
        assert config.exponential_base >= 1.01


class TestRetrySchemaAlignment:
    """Verify RetrySettings, RuntimeRetryConfig, and RetryPolicy stay in sync.

    These tests catch "config field orphaning" - when a field is added to
    Settings but not wired through to the runtime config or tenacity call.

    P2-2026-01-21 bug: exponential_base was added to RetrySettings but never
    mapped to RetryConfig or passed to tenacity. These tests would have
    caught that bug at commit time.
    """

    # Known field name mappings between Settings and Config
    FIELD_MAPPINGS: ClassVar[dict[str, str]] = {
        "initial_delay_seconds": "base_delay",
        "max_delay_seconds": "max_delay",
    }

    # Fields that exist in Config but not Settings (internal defaults)
    CONFIG_INTERNAL_ONLY: ClassVar[set[str]] = {"jitter"}

    def test_policy_defaults_matches_config_fields(self) -> None:
        """POLICY_DEFAULTS must have same fields as RuntimeRetryConfig.

        This catches the case where someone adds a field to RuntimeRetryConfig but
        forgets to add it to POLICY_DEFAULTS. Without this, from_policy()
        would crash at runtime when accessing the missing field.
        """
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())
        defaults_fields = set(POLICY_DEFAULTS.keys())

        missing_from_defaults = config_fields - defaults_fields
        extra_in_defaults = defaults_fields - config_fields

        assert not missing_from_defaults, (
            f"POLICY_DEFAULTS missing fields: {missing_from_defaults}. Add these to POLICY_DEFAULTS in contracts/config/defaults.py."
        )
        assert not extra_in_defaults, (
            f"POLICY_DEFAULTS has extra fields: {extra_in_defaults}. Remove these or add to RuntimeRetryConfig dataclass."
        )

    def test_retry_settings_fields_exist_in_config(self) -> None:
        """Every RetrySettings field must have a corresponding RuntimeRetryConfig field.

        This catches the case where someone adds a field to Settings but
        forgets to add it to the runtime Config dataclass.
        """
        from elspeth.core.config import RetrySettings

        settings_fields = set(RetrySettings.model_fields.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        # Map Settings field names to expected Config field names
        expected_in_config = {self.FIELD_MAPPINGS.get(f, f) for f in settings_fields}

        # Check that all expected fields exist in Config
        missing_from_config = expected_in_config - config_fields
        assert not missing_from_config, (
            f"RetrySettings fields not in RuntimeRetryConfig: {missing_from_config}. "
            f"Add these fields to RuntimeRetryConfig dataclass and wire in from_settings()."
        )

    def test_retry_config_covers_settings(self) -> None:
        """RuntimeRetryConfig should not have unexpected fields beyond Settings + internals.

        This catches Config bloat - fields added to Config that don't come
        from Settings and aren't documented as internal-only.
        """
        from elspeth.core.config import RetrySettings

        settings_fields = set(RetrySettings.model_fields.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        # Expected Config fields = mapped Settings fields + internal-only
        expected_config = {self.FIELD_MAPPINGS.get(f, f) for f in settings_fields} | self.CONFIG_INTERNAL_ONLY

        unexpected_in_config = config_fields - expected_config
        assert not unexpected_in_config, (
            f"RuntimeRetryConfig has undocumented fields: {unexpected_in_config}. Either add to Settings, add to CONFIG_INTERNAL_ONLY, or remove."
        )

    def test_retry_policy_matches_config(self) -> None:
        """RetryPolicy TypedDict should have same fields as RuntimeRetryConfig.

        This ensures plugin-level config (RetryPolicy) can configure the
        same options as global config (RetrySettings -> RuntimeRetryConfig).
        """
        policy_fields = set(RetryPolicy.__annotations__.keys())
        config_fields = set(RuntimeRetryConfig.__dataclass_fields__.keys())

        # Policy and Config should match exactly
        missing_from_policy = config_fields - policy_fields
        missing_from_config = policy_fields - config_fields

        assert not missing_from_policy, (
            f"RuntimeRetryConfig fields not in RetryPolicy: {missing_from_policy}. Add these to RetryPolicy TypedDict."
        )
        assert not missing_from_config, (
            f"RetryPolicy fields not in RuntimeRetryConfig: {missing_from_config}. Add these to RuntimeRetryConfig dataclass."
        )

    def test_from_settings_maps_all_fields_with_sentinel_values(self) -> None:
        """Verify from_settings() actually maps every Settings field.

        Uses sentinel values to detect "forgot to map, used default instead".
        If a field uses default instead of the Settings value, test fails.
        """
        from elspeth.core.config import RetrySettings

        # Use distinctive non-default values
        settings = RetrySettings(
            max_attempts=99,
            initial_delay_seconds=99.0,
            max_delay_seconds=999.0,
            exponential_base=9.9,
        )

        config = RuntimeRetryConfig.from_settings(settings)

        # Verify all Settings fields reached Config with correct values
        assert config.max_attempts == 99, "max_attempts not mapped from Settings"
        assert config.base_delay == 99.0, "initial_delay_seconds not mapped to base_delay"
        assert config.max_delay == 999.0, "max_delay_seconds not mapped to max_delay"
        assert config.exponential_base == 9.9, "exponential_base not mapped from Settings"
        # jitter is internal, verify it has the expected default
        assert config.jitter == 1.0, "jitter should be internal default"
