# tests/property/core/test_rate_limiter_properties.py
"""Property-based tests for rate limiter behavior and invariants.

These tests verify the fundamental properties of ELSPETH's rate limiting system:

Rate Limiting Properties:
- Validation rejects invalid names and rates
- Acquire respects rate limits (never over-acquires)
- NoOpLimiter always succeeds without delay
- Registry returns same limiter for same service

Thread Safety Properties:
- Concurrent try_acquire() is atomic
- Registry get_limiter() is thread-safe

Timing Properties:
- Timeout is respected (doesn't exceed specified duration)
- Weight affects capacity correctly
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.rate_limit import NoOpLimiter, RateLimiter, RateLimitRegistry

# =============================================================================
# Strategies for generating rate limiter configurations
# =============================================================================

# Valid limiter names (alphanumeric starting with letter)
valid_names = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
).filter(lambda s: s[0].isalpha())

# Names with underscore and digits (still valid)
valid_names_extended = st.text(
    min_size=1,
    max_size=20,
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
).filter(lambda s: len(s) > 0 and s[0].isalpha())

# Invalid names (various patterns)
invalid_names = st.one_of(
    # Starts with digit
    st.text(
        min_size=1,
        max_size=10,
        alphabet="0123456789abcdef",
    ).filter(lambda s: s[0].isdigit()),
    # Contains special characters
    st.text(
        min_size=1,
        max_size=10,
        alphabet="abc-./!@#$%",
    ).filter(lambda s: any(c in s for c in "-./!@#$%")),
    # Empty string
    st.just(""),
)

# Positive rate values (valid)
positive_rates = st.integers(min_value=1, max_value=1000)

# Non-positive rates (invalid)
non_positive_rates = st.integers(min_value=-100, max_value=0)

# Weight values
weights = st.integers(min_value=1, max_value=10)


# =============================================================================
# RateLimiter Validation Property Tests
# =============================================================================


class TestRateLimiterValidationProperties:
    """Property tests for RateLimiter input validation."""

    @given(name=valid_names_extended, rpm=positive_rates)
    @settings(max_examples=100)
    def test_valid_name_and_rate_accepted(self, name: str, rpm: int) -> None:
        """Property: Valid names and positive rates create a limiter."""
        with RateLimiter(name=name, requests_per_minute=rpm) as limiter:
            assert limiter.name == name

    @given(name=invalid_names, rpm=positive_rates)
    @settings(max_examples=50)
    def test_invalid_name_rejected(self, name: str, rpm: int) -> None:
        """Property: Invalid names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name=name, requests_per_minute=rpm)

    @given(name=valid_names, rpm=non_positive_rates)
    @settings(max_examples=50)
    def test_non_positive_rpm_rejected(self, name: str, rpm: int) -> None:
        """Property: Non-positive requests_per_minute raises ValueError."""
        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(name=name, requests_per_minute=rpm)


class TestRateLimiterAcquireProperties:
    """Property tests for acquire/try_acquire behavior."""

    @given(name=valid_names, rpm=st.integers(min_value=5, max_value=100))
    @settings(max_examples=50)
    def test_try_acquire_succeeds_under_limit(self, name: str, rpm: int) -> None:
        """Property: try_acquire succeeds when under rate limit."""
        with RateLimiter(name=name, requests_per_minute=rpm) as limiter:
            # First acquire should always succeed
            assert limiter.try_acquire() is True

    @given(name=valid_names, weight=weights)
    @settings(max_examples=30)
    def test_weight_affects_capacity(self, name: str, weight: int) -> None:
        """Property: Larger weights consume more capacity."""
        # Use high rate to avoid immediate rate limiting
        rpm = weight + 1  # Ensure we have capacity for one weighted acquire
        with RateLimiter(name=name, requests_per_minute=rpm) as limiter:
            # Acquire with weight should succeed initially
            assert limiter.try_acquire(weight=weight) is True

    @given(name=valid_names)
    @settings(max_examples=20)
    def test_over_limit_returns_false(self, name: str) -> None:
        """Property: try_acquire returns False when over limit."""
        # Very low rate limit
        with RateLimiter(name=name, requests_per_minute=1) as limiter:
            # First acquire succeeds
            assert limiter.try_acquire() is True
            # Second immediate acquire fails (over limit)
            assert limiter.try_acquire() is False

    @given(name=valid_names, rpm=st.integers(min_value=1, max_value=10))
    @settings(max_examples=30)
    def test_acquire_deterministic_success(self, name: str, rpm: int) -> None:
        """Property: acquire() always succeeds (eventually)."""
        with RateLimiter(name=name, requests_per_minute=rpm) as limiter:
            # acquire() should always succeed (blocking)
            # We don't wait long, just verify no exception
            limiter.acquire()  # Should not raise


class TestRateLimiterTimeoutProperties:
    """Property tests for timeout behavior."""

    @given(name=valid_names)
    @settings(max_examples=20)
    def test_timeout_raises_after_deadline(self, name: str) -> None:
        """Property: acquire() with timeout raises TimeoutError when exceeded."""
        with RateLimiter(name=name, requests_per_minute=1) as limiter:
            # Exhaust the limit
            assert limiter.try_acquire() is True

            # Now acquire with very short timeout should fail
            with pytest.raises(TimeoutError):
                limiter.acquire(timeout=0.05)

    @given(name=valid_names)
    @settings(max_examples=10)
    def test_timeout_respects_duration(self, name: str) -> None:
        """Property: Timeout doesn't significantly exceed specified duration."""
        with RateLimiter(name=name, requests_per_minute=1) as limiter:
            # Exhaust the limit
            limiter.try_acquire()

            # Measure actual timeout duration
            timeout = 0.1
            start = time.monotonic()

            with pytest.raises(TimeoutError):
                limiter.acquire(timeout=timeout)

            elapsed = time.monotonic() - start

            # Should timeout within 100ms of target (accounting for polling interval)
            assert elapsed < timeout + 0.1, f"Timeout took too long: {elapsed}s > {timeout + 0.1}s"


class TestRateLimiterPersistenceProperties:
    """Property tests for SQLite persistence."""

    @given(name=valid_names, rpm=positive_rates)
    @settings(max_examples=20)
    def test_persistence_creates_tables(self, name: str, rpm: int) -> None:
        """Property: Persistence path creates SQLite tables."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "rate_limits.db")

            with RateLimiter(
                name=name,
                requests_per_minute=rpm,
                persistence_path=db_path,
            ):
                # Just verify it doesn't crash
                pass

            # DB file should exist after close
            assert Path(db_path).exists()


# =============================================================================
# NoOpLimiter Property Tests
# =============================================================================


class TestNoOpLimiterProperties:
    """Property tests for NoOpLimiter behavior."""

    @given(weight=weights)
    @settings(max_examples=50)
    def test_try_acquire_always_true(self, weight: int) -> None:
        """Property: NoOpLimiter.try_acquire() always returns True."""
        limiter = NoOpLimiter()
        # Call many times with various weights - always succeeds
        for _ in range(10):
            assert limiter.try_acquire(weight) is True

    @given(weight=weights)
    @settings(max_examples=30)
    def test_acquire_never_blocks(self, weight: int) -> None:
        """Property: NoOpLimiter.acquire() never blocks."""
        limiter = NoOpLimiter()
        start = time.monotonic()
        for _ in range(100):
            limiter.acquire(weight)
        elapsed = time.monotonic() - start

        # 100 acquires should be nearly instant (< 50ms)
        assert elapsed < 0.05, f"NoOpLimiter blocked for {elapsed}s"

    def test_context_manager_works(self) -> None:
        """Property: NoOpLimiter works as context manager."""
        with NoOpLimiter() as limiter:
            assert limiter.try_acquire() is True
            limiter.acquire()


# =============================================================================
# RateLimitRegistry Property Tests
# =============================================================================


class TestRateLimitRegistryProperties:
    """Property tests for RateLimitRegistry behavior."""

    @given(service_name=valid_names)
    @settings(max_examples=30)
    def test_same_service_returns_same_limiter(self, service_name: str) -> None:
        """Property: get_limiter() returns same instance for same service."""
        # Create mock settings
        settings = MagicMock()
        settings.enabled = True
        settings.persistence_path = None
        service_config = MagicMock()
        service_config.requests_per_minute = 100
        settings.get_service_config.return_value = service_config

        registry = RateLimitRegistry(settings)
        try:
            limiter1 = registry.get_limiter(service_name)
            limiter2 = registry.get_limiter(service_name)

            assert limiter1 is limiter2, "Same service should return same limiter instance"
        finally:
            registry.close()

    @given(services=st.lists(valid_names, min_size=2, max_size=5, unique=True))
    @settings(max_examples=20)
    def test_different_services_different_limiters(self, services: list[str]) -> None:
        """Property: Different services get different limiter instances."""
        settings = MagicMock()
        settings.enabled = True
        settings.persistence_path = None
        service_config = MagicMock()
        service_config.requests_per_minute = 100
        settings.get_service_config.return_value = service_config

        registry = RateLimitRegistry(settings)
        try:
            limiters = [registry.get_limiter(svc) for svc in services]

            # All limiters should be distinct objects
            assert len({id(lim) for lim in limiters}) == len(services)
        finally:
            registry.close()

    @given(service_name=valid_names)
    @settings(max_examples=20)
    def test_disabled_returns_noop(self, service_name: str) -> None:
        """Property: Disabled registry returns NoOpLimiter."""
        settings = MagicMock()
        settings.enabled = False

        registry = RateLimitRegistry(settings)
        limiter = registry.get_limiter(service_name)

        assert isinstance(limiter, NoOpLimiter)

    def test_reset_clears_all_limiters(self) -> None:
        """Property: reset_all() clears all cached limiters."""
        settings = MagicMock()
        settings.enabled = True
        settings.persistence_path = None
        service_config = MagicMock()
        service_config.requests_per_minute = 100
        settings.get_service_config.return_value = service_config

        registry = RateLimitRegistry(settings)
        try:
            # Get some limiters
            limiter1 = registry.get_limiter("service_a")
            registry.get_limiter("service_b")

            # Reset
            registry.reset_all()

            # Get same service - should be NEW instance
            limiter1_new = registry.get_limiter("service_a")
            assert limiter1 is not limiter1_new
        finally:
            registry.close()


# =============================================================================
# Thread Safety Property Tests
# =============================================================================


class TestRateLimiterThreadSafetyProperties:
    """Property tests for thread safety."""

    @given(name=valid_names)
    @settings(max_examples=10)
    def test_concurrent_try_acquire_is_safe(self, name: str) -> None:
        """Property: Concurrent try_acquire() calls don't corrupt state."""
        # High limit so we don't hit rate limiting during test
        with RateLimiter(name=name, requests_per_minute=1000) as limiter:
            successes: list[bool] = []
            lock = threading.Lock()

            def worker() -> None:
                for _ in range(10):
                    result = limiter.try_acquire()
                    with lock:
                        successes.append(result)

            # Run 5 threads concurrently
            threads = [threading.Thread(target=worker) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All should have succeeded (high limit)
            assert all(successes), f"Some acquires failed: {successes.count(False)} failures"

    @given(services=st.lists(valid_names, min_size=2, max_size=3, unique=True))
    @settings(max_examples=10)
    def test_registry_concurrent_get_limiter(self, services: list[str]) -> None:
        """Property: Concurrent get_limiter() calls are thread-safe."""
        settings = MagicMock()
        settings.enabled = True
        settings.persistence_path = None
        service_config = MagicMock()
        service_config.requests_per_minute = 100
        settings.get_service_config.return_value = service_config

        registry = RateLimitRegistry(settings)
        results: dict[str, list[object]] = {svc: [] for svc in services}
        lock = threading.Lock()

        def worker(service: str) -> None:
            limiter = registry.get_limiter(service)
            with lock:
                results[service].append(limiter)

        try:
            # Multiple threads request same services concurrently
            threads = []
            for service in services:
                for _ in range(3):  # 3 threads per service
                    t = threading.Thread(target=worker, args=(service,))
                    threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads for same service should get same limiter
            for service in services:
                limiters = results[service]
                assert all(lim is limiters[0] for lim in limiters), f"Service {service} got different limiter instances"
        finally:
            registry.close()


# =============================================================================
# Edge Case Property Tests
# =============================================================================


class TestRateLimiterEdgeCaseProperties:
    """Property tests for edge cases."""

    @given(name=valid_names)
    @settings(max_examples=20)
    def test_close_is_idempotent(self, name: str) -> None:
        """Property: Calling close() multiple times is safe."""
        limiter = RateLimiter(name=name, requests_per_minute=10)
        limiter.close()
        limiter.close()  # Should not raise
        limiter.close()  # Should not raise

    @given(name=valid_names, rpm=positive_rates)
    @settings(max_examples=20)
    def test_context_manager_cleanup(self, name: str, rpm: int) -> None:
        """Property: Context manager properly cleans up resources."""
        with RateLimiter(name=name, requests_per_minute=rpm) as limiter:
            limiter.try_acquire()
        # Should be closed after exiting context
        # (No way to verify directly, but shouldn't raise on second close)
        limiter.close()

    @given(name=valid_names, weight=st.integers(min_value=1, max_value=5))
    @settings(max_examples=20)
    def test_exact_capacity_consumption(self, name: str, weight: int) -> None:
        """Property: Acquiring exactly the capacity works, +1 fails."""
        capacity = weight * 2
        with RateLimiter(name=name, requests_per_minute=capacity) as limiter:
            # First acquire of weight should succeed
            assert limiter.try_acquire(weight=weight) is True
            # Second acquire of same weight should succeed (exactly at capacity)
            assert limiter.try_acquire(weight=weight) is True
            # Third should fail (over capacity)
            assert limiter.try_acquire(weight=weight) is False
