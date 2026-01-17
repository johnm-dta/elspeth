# tests/engine/test_retry_policy.py
"""Tests for RetryPolicy TypedDict."""


class TestRetryPolicy:
    """Verify RetryPolicy TypedDict works correctly."""

    def test_retry_policy_importable(self) -> None:
        """RetryPolicy should be importable from contracts."""
        from elspeth.contracts import RetryPolicy

        policy: RetryPolicy = {
            "max_attempts": 3,
            "base_delay": 1.0,
        }
        assert policy["max_attempts"] == 3

    def test_retry_config_from_policy_with_typed_dict(self) -> None:
        """RetryConfig.from_policy should accept RetryPolicy."""
        from elspeth.contracts import RetryPolicy
        from elspeth.engine.retry import RetryConfig

        policy: RetryPolicy = {
            "max_attempts": 5,
            "base_delay": 2.0,
            "max_delay": 120.0,
            "jitter": 0.5,
        }

        config = RetryConfig.from_policy(policy)
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.jitter == 0.5

    def test_retry_policy_partial(self) -> None:
        """RetryPolicy should allow partial specification."""
        from elspeth.contracts import RetryPolicy
        from elspeth.engine.retry import RetryConfig

        # Only specify some fields
        policy: RetryPolicy = {"max_attempts": 10}
        config = RetryConfig.from_policy(policy)
        assert config.max_attempts == 10
        # Defaults for unspecified fields
        assert config.base_delay == 1.0
