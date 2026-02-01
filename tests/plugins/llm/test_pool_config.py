# tests/plugins/llm/test_pool_config.py
"""Tests for pool configuration in LLM transforms."""

import pytest

from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.pooling import PoolConfig


class TestPoolConfigDefaults:
    """Test pool configuration defaults."""

    def test_default_pool_size_is_sequential(self) -> None:
        """Default pool_size=1 means sequential processing."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
            }
        )

        assert config.pool_config is None or config.pool_config.pool_size == 1

    def test_pool_size_1_is_sequential_mode(self) -> None:
        """pool_size=1 should not create pool config."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
                "pool_size": 1,
            }
        )

        # pool_size=1 means sequential, no pooling needed
        assert config.pool_config is None


class TestPoolConfigExplicit:
    """Test explicit pool configuration."""

    def test_pool_size_greater_than_1_creates_config(self) -> None:
        """pool_size > 1 should create pool config with defaults."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
                "pool_size": 10,
            }
        )

        assert config.pool_config is not None
        assert config.pool_config.pool_size == 10
        # AIMD defaults
        assert config.pool_config.min_dispatch_delay_ms == 0
        assert config.pool_config.max_dispatch_delay_ms == 5000
        assert config.pool_config.backoff_multiplier == 2.0
        assert config.pool_config.recovery_step_ms == 50
        # Max retry timeout default (1 hour)
        assert config.pool_config.max_capacity_retry_seconds == 3600

    def test_custom_aimd_settings(self) -> None:
        """Custom AIMD settings should be applied."""
        config = LLMConfig.from_dict(
            {
                "model": "gpt-4",
                "template": "{{ row.text }}",
                "schema": {"fields": "dynamic"},
                "required_input_fields": [],  # Explicit opt-out for this test
                "pool_size": 5,
                "min_dispatch_delay_ms": 10,
                "max_dispatch_delay_ms": 1000,
                "backoff_multiplier": 3.0,
                "recovery_step_ms": 25,
                "max_capacity_retry_seconds": 1800,  # 30 minutes
            }
        )

        assert config.pool_config is not None
        assert config.pool_config.pool_size == 5
        assert config.pool_config.min_dispatch_delay_ms == 10
        assert config.pool_config.max_dispatch_delay_ms == 1000
        assert config.pool_config.backoff_multiplier == 3.0
        assert config.pool_config.recovery_step_ms == 25
        assert config.pool_config.max_capacity_retry_seconds == 1800


class TestPoolConfigValidation:
    """Test pool configuration validation."""

    def test_min_dispatch_delay_must_not_exceed_max(self) -> None:
        """min_dispatch_delay_ms must be <= max_dispatch_delay_ms."""
        from pydantic import ValidationError

        from elspeth.plugins.pooling import PoolConfig

        with pytest.raises(ValidationError) as exc_info:
            PoolConfig(
                pool_size=10,
                min_dispatch_delay_ms=1000,
                max_dispatch_delay_ms=100,
            )

        # Verify the error message mentions the invariant
        error_str = str(exc_info.value)
        assert "min_dispatch_delay_ms" in error_str or "cannot exceed" in error_str.lower()

    def test_min_equal_to_max_dispatch_delay_is_allowed(self) -> None:
        """min_dispatch_delay_ms == max_dispatch_delay_ms should be allowed (fixed delay)."""
        from elspeth.plugins.pooling import PoolConfig

        # This should NOT raise - equal values are valid (fixed delay)
        config = PoolConfig(
            pool_size=10,
            min_dispatch_delay_ms=500,
            max_dispatch_delay_ms=500,
        )

        assert config.min_dispatch_delay_ms == 500
        assert config.max_dispatch_delay_ms == 500

    def test_pool_size_must_be_positive(self) -> None:
        """pool_size must be >= 1."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ row.text }}",
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 0,
                }
            )

    def test_backoff_multiplier_must_be_greater_than_1(self) -> None:
        """backoff_multiplier must be > 1."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ row.text }}",
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 10,
                    "backoff_multiplier": 0.5,
                }
            )

    def test_max_capacity_retry_seconds_must_be_positive(self) -> None:
        """max_capacity_retry_seconds must be > 0."""
        with pytest.raises(PluginConfigError):
            LLMConfig.from_dict(
                {
                    "model": "gpt-4",
                    "template": "{{ row.text }}",
                    "schema": {"fields": "dynamic"},
                    "required_input_fields": [],  # Explicit opt-out for this test
                    "pool_size": 10,
                    "max_capacity_retry_seconds": 0,
                }
            )


class TestPoolConfigToThrottleConfig:
    """Test PoolConfig.to_throttle_config() method."""

    def test_to_throttle_config_creates_throttle_config(self) -> None:
        """to_throttle_config() should create ThrottleConfig with AIMD settings."""
        pool_config = PoolConfig(
            pool_size=10,
            min_dispatch_delay_ms=100,
            max_dispatch_delay_ms=2000,
            backoff_multiplier=1.5,
            recovery_step_ms=25,
            max_capacity_retry_seconds=1800,
        )

        throttle_config = pool_config.to_throttle_config()

        assert throttle_config.min_dispatch_delay_ms == 100
        assert throttle_config.max_dispatch_delay_ms == 2000
        assert throttle_config.backoff_multiplier == 1.5
        assert throttle_config.recovery_step_ms == 25

    def test_to_throttle_config_excludes_non_aimd_fields(self) -> None:
        """ThrottleConfig should not include pool_size or max_capacity_retry_seconds."""
        pool_config = PoolConfig(
            pool_size=10,
            max_capacity_retry_seconds=1800,
        )

        throttle_config = pool_config.to_throttle_config()

        # ThrottleConfig is a dataclass, check it doesn't have these attributes
        assert not hasattr(throttle_config, "pool_size")
        assert not hasattr(throttle_config, "max_capacity_retry_seconds")
