"""Tests for rate limiter."""

from __future__ import annotations

import time
from pathlib import Path

import pytest


class TestRateLimiterValidation:
    """Tests for rate limiter input validation."""

    def test_rejects_empty_name(self) -> None:
        """Empty name is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="", requests_per_minute=60)

    def test_rejects_name_starting_with_number(self) -> None:
        """Name starting with number is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="123api", requests_per_minute=60)

    def test_rejects_name_with_special_characters(self) -> None:
        """Name with special characters is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my-api", requests_per_minute=60)

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my.api", requests_per_minute=60)

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="my api", requests_per_minute=60)

    def test_rejects_sql_injection_attempt(self) -> None:
        """SQL injection attempts in name are rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="Invalid rate limiter name"):
            RateLimiter(name="api; DROP TABLE users;--", requests_per_minute=60)

    def test_accepts_valid_names(self) -> None:
        """Valid names are accepted."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="api", requests_per_minute=60) as limiter:
            assert limiter.name == "api"

        with RateLimiter(name="my_api", requests_per_minute=60) as limiter:
            assert limiter.name == "my_api"

        with RateLimiter(name="MyApi123", requests_per_minute=60) as limiter:
            assert limiter.name == "MyApi123"

        with RateLimiter(name="API_v2_production", requests_per_minute=60) as limiter:
            assert limiter.name == "API_v2_production"

    def test_rejects_zero_requests_per_minute(self) -> None:
        """Zero requests_per_minute is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(name="test", requests_per_minute=0)

    def test_rejects_negative_requests_per_minute(self) -> None:
        """Negative requests_per_minute is rejected."""
        from elspeth.core.rate_limit import RateLimiter

        with pytest.raises(ValueError, match="requests_per_minute must be positive"):
            RateLimiter(name="test", requests_per_minute=-5)


class TestRateLimiter:
    """Tests for rate limiting wrapper."""

    def test_create_limiter(self) -> None:
        """Can create a rate limiter."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(
            name="test_api",
            requests_per_minute=60,
        ) as limiter:
            assert limiter.name == "test_api"

    def test_acquire_within_limit(self) -> None:
        """acquire() succeeds when under limit."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="test", requests_per_minute=600) as limiter:
            # Should not raise or block significantly
            start = time.monotonic()
            limiter.acquire()
            elapsed = time.monotonic() - start

            assert elapsed < 0.1  # Should be near-instant

    def test_try_acquire_returns_false_when_exceeded(self) -> None:
        """try_acquire() returns False instead of blocking."""
        from elspeth.core.rate_limit import RateLimiter

        # Only 1 request per minute
        with RateLimiter(name="test", requests_per_minute=1) as limiter:
            # First: succeeds
            assert limiter.try_acquire() is True

            # Second (immediate): should fail without blocking
            assert limiter.try_acquire() is False

    def test_limiter_with_sqlite_persistence(self, tmp_path: Path) -> None:
        """Rate limits persist across limiter instances."""
        from elspeth.core.rate_limit import RateLimiter

        db_path = tmp_path / "limits.db"

        # First limiter uses up the quota
        limiter1 = RateLimiter(
            name="persistent",
            requests_per_minute=1,
            persistence_path=str(db_path),
        )
        limiter1.acquire()
        limiter1.close()  # Clean up first limiter

        # Second limiter (same name, same db) should see used quota
        limiter2 = RateLimiter(
            name="persistent",
            requests_per_minute=1,
            persistence_path=str(db_path),
        )

        # Should fail because quota already used
        assert limiter2.try_acquire() is False
        limiter2.close()

    def test_limiter_context_manager(self) -> None:
        """RateLimiter can be used as context manager."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="ctx_test", requests_per_minute=60) as limiter:
            limiter.acquire()
            assert limiter.try_acquire() is True

    def test_weight_parameter(self) -> None:
        """acquire() respects weight parameter."""
        from elspeth.core.rate_limit import RateLimiter

        with RateLimiter(name="weighted", requests_per_minute=5) as limiter:
            # Use up all 5 tokens at once
            limiter.acquire(weight=5)

            # Should fail - all tokens used
            assert limiter.try_acquire(weight=1) is False

    def test_requests_per_minute_limit(self) -> None:
        """Per-minute rate limits are enforced."""
        from elspeth.core.rate_limit import RateLimiter

        # Only 3 requests per minute
        with RateLimiter(
            name="minute_limit",
            requests_per_minute=3,
        ) as limiter:
            # First three should work
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True
            assert limiter.try_acquire() is True

            # Fourth should fail (hit minute limit)
            assert limiter.try_acquire() is False


class TestRateLimitRegistry:
    """Tests for rate limiter registry."""

    def test_get_or_create_limiter(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("api_a")
        limiter2 = registry.get_limiter("api_a")

        # Same instance returned
        assert limiter1 is limiter2

    def test_different_services_different_limiters(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter_a = registry.get_limiter("api_a")
        limiter_b = registry.get_limiter("api_b")

        assert limiter_a is not limiter_b

    def test_registry_respects_service_config(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings, ServiceRateLimit
        from elspeth.core.rate_limit import RateLimiter, RateLimitRegistry

        settings = RateLimitSettings(
            default_requests_per_minute=60,
            services={
                "slow_api": ServiceRateLimit(requests_per_minute=10),
            },
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        default_limiter = registry.get_limiter("fast_api")
        slow_limiter = registry.get_limiter("slow_api")

        # Type narrowing: enabled registry returns RateLimiter, not NoOpLimiter
        assert isinstance(default_limiter, RateLimiter)
        assert isinstance(slow_limiter, RateLimiter)
        assert default_limiter._requests_per_minute == 60
        assert slow_limiter._requests_per_minute == 10

    def test_registry_disabled(self) -> None:
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import NoOpLimiter, RateLimitRegistry

        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter = registry.get_limiter("any_api")

        # Should return no-op limiter
        assert isinstance(limiter, NoOpLimiter)

    def test_registry_disabled_returns_same_noop_instance(self) -> None:
        """When disabled, registry returns the same NoOpLimiter instance."""
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import NoOpLimiter, RateLimitRegistry

        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("api_a")
        limiter2 = registry.get_limiter("api_b")

        # Should return the same cached instance
        assert isinstance(limiter1, NoOpLimiter)
        assert limiter1 is limiter2

    def test_reset_all_clears_registry(self) -> None:
        """reset_all() clears all limiters from the registry."""
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        # Create some limiters
        limiter_a = registry.get_limiter("api_a")
        limiter_b = registry.get_limiter("api_b")

        # Verify they exist
        assert limiter_a is registry.get_limiter("api_a")
        assert limiter_b is registry.get_limiter("api_b")

        # Reset all
        registry.reset_all()

        # New calls should create new instances
        new_limiter_a = registry.get_limiter("api_a")
        new_limiter_b = registry.get_limiter("api_b")

        assert new_limiter_a is not limiter_a
        assert new_limiter_b is not limiter_b

    def test_registry_close(self) -> None:
        """close() cleans up all limiters."""
        from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry

        settings = RateLimitSettings(default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        # Create some limiters
        registry.get_limiter("api_a")
        registry.get_limiter("api_b")

        # Close should not raise
        registry.close()

        # After close, registry should be empty (new calls create new instances)
        # This verifies close() cleared internal state
        assert len(registry._limiters) == 0


class TestNoOpLimiter:
    """Tests for NoOpLimiter."""

    def test_noop_limiter_acquire(self) -> None:
        """NoOpLimiter.acquire() always succeeds instantly and returns None."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()

        # Should not block or raise, and should return None
        limiter.acquire()
        limiter.acquire(weight=100)
        # No assertion needed - acquire() returns None by design

    def test_noop_limiter_try_acquire(self) -> None:
        """NoOpLimiter.try_acquire() always returns True."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()

        assert limiter.try_acquire() is True
        assert limiter.try_acquire(weight=100) is True

    def test_noop_limiter_context_manager(self) -> None:
        """NoOpLimiter can be used as a context manager."""
        from elspeth.core.rate_limit import NoOpLimiter

        with NoOpLimiter() as limiter:
            limiter.acquire()
            assert limiter.try_acquire() is True

    def test_noop_limiter_close(self) -> None:
        """NoOpLimiter.close() returns None and is idempotent."""
        from elspeth.core.rate_limit import NoOpLimiter

        limiter = NoOpLimiter()

        # Should not raise - close() returns None by design
        limiter.close()

        # Should be idempotent (calling twice doesn't raise)
        limiter.close()
        # No assertion needed - close() returns None by design


class TestExcepthookSuppression:
    """Tests for the narrowly-scoped thread exception suppression.

    The rate limiter installs a custom threading.excepthook to suppress
    benign AssertionError from pyrate-limiter's leaker thread cleanup.
    These tests verify suppression is narrowly scoped.
    """

    def test_suppression_only_for_registered_threads(self) -> None:
        """Unregistered threads should not be suppressed."""

        from elspeth.core.rate_limit.limiter import (
            _custom_excepthook,
            _suppressed_lock,
            _suppressed_thread_idents,
        )

        # Create a mock ExceptHookArgs for an unregistered thread
        class MockThread:
            ident = 99999  # Not in suppression set
            name = "unregistered_thread"

        class MockArgs:
            exc_type = AssertionError
            exc_value = AssertionError("test")
            exc_traceback = None
            thread = MockThread()

        # Ensure the thread is NOT in the suppression set
        with _suppressed_lock:
            _suppressed_thread_idents.discard(99999)

        # Track if original hook was called
        original_called = []
        import elspeth.core.rate_limit.limiter as limiter_module

        original_hook = limiter_module._original_excepthook
        limiter_module._original_excepthook = lambda args: original_called.append(True)

        try:
            _custom_excepthook(MockArgs())  # type: ignore[arg-type]
            # Original hook should have been called
            assert len(original_called) == 1
        finally:
            limiter_module._original_excepthook = original_hook

    def test_suppression_only_for_assertion_error(self) -> None:
        """Only AssertionError should be suppressed, not other exceptions."""

        from elspeth.core.rate_limit.limiter import (
            _custom_excepthook,
            _suppressed_lock,
            _suppressed_thread_idents,
        )

        class MockThread:
            ident = 88888
            name = "registered_thread"

        # Register the thread for suppression
        with _suppressed_lock:
            _suppressed_thread_idents.add(88888)

        try:
            # Test with ValueError - should NOT be suppressed
            class MockArgsValueError:
                exc_type = ValueError  # Not AssertionError
                exc_value = ValueError("test")
                exc_traceback = None
                thread = MockThread()

            original_called = []
            import elspeth.core.rate_limit.limiter as limiter_module

            original_hook = limiter_module._original_excepthook
            limiter_module._original_excepthook = lambda args: original_called.append(True)

            try:
                _custom_excepthook(MockArgsValueError())  # type: ignore[arg-type]
                # Original hook should have been called (not suppressed)
                assert len(original_called) == 1
            finally:
                limiter_module._original_excepthook = original_hook
        finally:
            with _suppressed_lock:
                _suppressed_thread_idents.discard(88888)

    def test_suppression_works_for_registered_assertion_error(self) -> None:
        """Registered thread + AssertionError should be suppressed."""

        from elspeth.core.rate_limit.limiter import (
            _custom_excepthook,
            _suppressed_lock,
            _suppressed_thread_idents,
        )

        class MockThread:
            ident = 77777
            name = "leaker_thread"

        # Register the thread for suppression
        with _suppressed_lock:
            _suppressed_thread_idents.add(77777)

        class MockArgs:
            exc_type = AssertionError
            exc_value = AssertionError("bucket disposed")
            exc_traceback = None
            thread = MockThread()

        original_called = []
        import elspeth.core.rate_limit.limiter as limiter_module

        original_hook = limiter_module._original_excepthook
        limiter_module._original_excepthook = lambda args: original_called.append(True)

        try:
            _custom_excepthook(MockArgs())  # type: ignore[arg-type]
            # Original hook should NOT have been called (suppressed)
            assert len(original_called) == 0

            # Thread should be removed from suppression set after suppression
            with _suppressed_lock:
                assert 77777 not in _suppressed_thread_idents
        finally:
            limiter_module._original_excepthook = original_hook
            # Cleanup just in case
            with _suppressed_lock:
                _suppressed_thread_idents.discard(77777)

    def test_suppression_cleanup_after_clean_exit(self) -> None:
        """Thread idents are cleaned up even when thread exits without exception.

        Verifies fix for stale thread ident accumulation. After close(),
        leaker thread idents should be removed from the suppression set
        regardless of whether AssertionError was raised.
        """
        from elspeth.core.rate_limit import RateLimiter
        from elspeth.core.rate_limit.limiter import (
            _suppressed_lock,
            _suppressed_thread_idents,
        )

        # Create rate limiter which spawns a leaker thread
        limiter = RateLimiter(name="cleanup_test", requests_per_minute=60)

        # Acquire to ensure leaker thread is active
        limiter.acquire()

        # Get leaker ident before close
        leaker = limiter._limiter.bucket_factory._leaker
        leaker_ident = leaker.ident if leaker is not None else None

        # Close the limiter - this should clean up thread idents
        limiter.close()

        # Verify leaker ident was removed from suppression set (if it was registered)
        if leaker_ident is not None:
            with _suppressed_lock:
                assert leaker_ident not in _suppressed_thread_idents, (
                    f"Stale thread ident {leaker_ident} found in suppression set after close()"
                )


class TestAcquireThreadSafety:
    """Tests for acquire() thread safety."""

    def test_acquire_concurrency_no_interleaving(self) -> None:
        """Multiple threads calling acquire() should not interleave incorrectly.

        Verifies that concurrent acquire() calls are thread-safe and that
        tokens are consumed correctly without race conditions.
        """
        import threading

        from elspeth.core.rate_limit import RateLimiter

        # Allow 10 requests per minute (enough for this test)
        with RateLimiter(name="concurrent_test", requests_per_minute=10) as limiter:
            results: list[bool] = []
            errors: list[Exception] = []

            def worker() -> None:
                try:
                    # Each thread tries to acquire 1 token
                    limiter.acquire(weight=1)
                    results.append(True)
                except Exception as e:
                    errors.append(e)

            # Launch 10 threads (exactly at limit)
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All 10 should succeed (no errors)
            assert len(errors) == 0, f"Unexpected errors: {errors}"
            assert len(results) == 10

            # The 11th acquire should fail (quota exhausted)
            assert limiter.try_acquire() is False

    def test_acquire_timeout_parameter(self) -> None:
        """acquire() should respect timeout parameter and raise TimeoutError.

        Verifies that acquire() doesn't block forever when rate limited.
        Critical for high-stakes audit systems that require bounded behavior.
        """
        import pytest

        from elspeth.core.rate_limit import RateLimiter

        # Very restrictive: 1 request per minute
        with RateLimiter(name="timeout_test", requests_per_minute=1) as limiter:
            # First request: succeeds
            limiter.acquire()

            # Second request with short timeout: should raise TimeoutError
            with pytest.raises(TimeoutError, match=r"Failed to acquire.*timeout"):
                limiter.acquire(weight=1, timeout=0.1)

    def test_acquire_with_timeout_none_accepts_parameter(self) -> None:
        """acquire() accepts timeout=None parameter (API compatibility test).

        This test verifies that timeout=None doesn't raise an exception.
        We don't test actual indefinite blocking behavior as that would
        make the test suite hang - the other tests verify timeout behavior.
        """
        from elspeth.core.rate_limit import RateLimiter

        # Create limiter with capacity
        with RateLimiter(name="timeout_none_test", requests_per_minute=60) as limiter:
            # Should not raise - verifies timeout=None is accepted
            limiter.acquire(weight=1, timeout=None)

            # Should also work without explicit timeout parameter
            limiter.acquire(weight=1)
