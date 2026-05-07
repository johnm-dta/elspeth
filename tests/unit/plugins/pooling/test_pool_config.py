"""Regression tests for Phase 0 fix #9: PoolConfig zero-delay.

Bug: PoolConfig accepted max_dispatch_delay_ms=0 and the combination of
min_dispatch_delay_ms=0 with recovery_step_ms=0. These configurations
created infinite-speed retry loops on capacity errors because AIMD backoff
could never produce a non-zero delay.

Fix: Added validators that reject:
  (a) max_dispatch_delay_ms=0
  (b) min_dispatch_delay_ms=0 AND recovery_step_ms=0
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from elspeth.plugins.infrastructure.pooling.config import PoolConfig


class TestPoolConfigZeroDelayRejected:
    """Verify PoolConfig rejects configurations that disable backoff."""

    def test_max_dispatch_delay_zero_rejected(self) -> None:
        """max_dispatch_delay_ms=0 is rejected (no room for AIMD backoff)."""
        with pytest.raises(ValidationError, match="max_dispatch_delay_ms must be > 0"):
            PoolConfig(max_dispatch_delay_ms=0)

    def test_min_zero_with_recovery_zero_rejected(self) -> None:
        """min_dispatch_delay_ms=0 with recovery_step_ms=0 is rejected.

        AIMD recovery subtracts recovery_step_ms from current delay on success.
        If both floor and step are 0, delay stays at 0 forever — no backoff.
        """
        with pytest.raises(ValidationError, match="cannot both be 0"):
            PoolConfig(min_dispatch_delay_ms=0, recovery_step_ms=0)

    def test_min_zero_with_nonzero_recovery_accepted(self) -> None:
        """min_dispatch_delay_ms=0 is fine if recovery_step_ms > 0."""
        config = PoolConfig(min_dispatch_delay_ms=0, recovery_step_ms=50)
        assert config.min_dispatch_delay_ms == 0
        assert config.recovery_step_ms == 50

    def test_max_positive_accepted(self) -> None:
        """Positive max_dispatch_delay_ms is accepted."""
        config = PoolConfig(max_dispatch_delay_ms=1000)
        assert config.max_dispatch_delay_ms == 1000

    def test_default_config_valid(self) -> None:
        """Default PoolConfig values pass validation."""
        config = PoolConfig()
        assert config.max_dispatch_delay_ms == 5000
        assert config.recovery_step_ms == 50


class TestPoolConfigInfiniteBackoffRejected:
    """Verify PoolConfig rejects non-finite backoff_multiplier.

    Bug: elspeth-fadff1511a. PoolConfig accepted float("inf") via Pydantic
    gt=1.0 constraint, but ThrottleConfig.__post_init__ rejects it at runtime.
    The validation gap means invalid config passes validation and crashes
    during executor construction.
    """

    def test_infinity_rejected(self) -> None:
        """backoff_multiplier=inf is rejected at config validation time."""
        with pytest.raises(ValidationError, match="backoff_multiplier"):
            PoolConfig(backoff_multiplier=float("inf"))

    def test_negative_infinity_rejected(self) -> None:
        """backoff_multiplier=-inf is rejected (also fails gt=1.0)."""
        with pytest.raises(ValidationError):
            PoolConfig(backoff_multiplier=float("-inf"))

    def test_nan_rejected(self) -> None:
        """backoff_multiplier=NaN is rejected."""
        with pytest.raises(ValidationError):
            PoolConfig(backoff_multiplier=float("nan"))

    def test_finite_value_accepted(self) -> None:
        """A normal finite multiplier passes validation."""
        config = PoolConfig(backoff_multiplier=1.5)
        assert config.backoff_multiplier == 1.5
