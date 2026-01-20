"""Tests for RateLimitRegistry and NoOpLimiter."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from elspeth.core.config import RateLimitSettings, ServiceRateLimit
from elspeth.core.rate_limit.limiter import RateLimiter
from elspeth.core.rate_limit.registry import NoOpLimiter, RateLimitRegistry


class TestNoOpLimiter:
    """Tests for NoOpLimiter class.

    NoOpLimiter provides the same interface as RateLimiter but does nothing.
    All operations succeed instantly without any rate limiting.
    """

    def test_acquire_does_nothing(self) -> None:
        """acquire() completes without error."""
        limiter = NoOpLimiter()
        # Should not raise, should not block
        limiter.acquire()
        limiter.acquire(weight=10)

    def test_try_acquire_always_succeeds(self) -> None:
        """try_acquire() always returns True."""
        limiter = NoOpLimiter()
        assert limiter.try_acquire() is True
        assert limiter.try_acquire(weight=100) is True

    def test_close_does_nothing(self) -> None:
        """close() completes without error."""
        limiter = NoOpLimiter()
        limiter.close()
        # Should be safe to call multiple times
        limiter.close()

    def test_context_manager_protocol(self) -> None:
        """NoOpLimiter works as context manager."""
        limiter = NoOpLimiter()
        with limiter as ctx:
            assert ctx is limiter

    def test_context_manager_calls_close_on_exit(self) -> None:
        """Context manager calls close() on exit."""
        limiter = NoOpLimiter()
        with patch.object(limiter, "close") as mock_close:
            with limiter:
                pass
            mock_close.assert_called_once()


class TestRateLimitRegistryDisabled:
    """Tests for RateLimitRegistry when rate limiting is disabled."""

    def test_returns_noop_limiter_when_disabled(self) -> None:
        """Registry returns NoOpLimiter when rate limiting is disabled."""
        settings = RateLimitSettings(enabled=False)
        registry = RateLimitRegistry(settings)

        limiter = registry.get_limiter("any_service")

        assert isinstance(limiter, NoOpLimiter)

    def test_same_noop_instance_for_all_services(self) -> None:
        """All services get the same NoOpLimiter instance when disabled."""
        settings = RateLimitSettings(enabled=False)
        registry = RateLimitRegistry(settings)

        limiter1 = registry.get_limiter("service_a")
        limiter2 = registry.get_limiter("service_b")
        limiter3 = registry.get_limiter("service_c")

        assert limiter1 is limiter2
        assert limiter2 is limiter3


class TestRateLimitRegistryEnabled:
    """Tests for RateLimitRegistry when rate limiting is enabled."""

    def test_creates_limiter_for_unknown_service(self) -> None:
        """Registry creates new RateLimiter for unknown service."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        limiter = registry.get_limiter("new_service")

        assert isinstance(limiter, RateLimiter)
        registry.close()

    def test_returns_same_limiter_for_same_service(self) -> None:
        """Registry returns cached limiter for repeated requests."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        limiter1 = registry.get_limiter("my_service")
        limiter2 = registry.get_limiter("my_service")

        assert limiter1 is limiter2
        registry.close()

    def test_different_limiters_for_different_services(self) -> None:
        """Registry creates separate limiters for different services."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        limiter_a = registry.get_limiter("service_a")
        limiter_b = registry.get_limiter("service_b")

        assert limiter_a is not limiter_b
        registry.close()

    def test_uses_service_specific_config(self) -> None:
        """Registry uses per-service config when available."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
            services={
                "openai": ServiceRateLimit(
                    requests_per_second=5,
                    requests_per_minute=100,
                ),
            },
        )
        registry = RateLimitRegistry(settings)

        limiter = registry.get_limiter("openai")

        # The limiter should have been created with service-specific config
        # We can verify by checking it's a RateLimiter (not NoOp) with the right name
        assert isinstance(limiter, RateLimiter)
        assert limiter.name == "openai"
        registry.close()

    def test_uses_default_config_for_unconfigured_service(self) -> None:
        """Registry uses default config for services not explicitly configured."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=15,
            services={
                "openai": ServiceRateLimit(requests_per_second=5),
            },
        )
        registry = RateLimitRegistry(settings)

        # This service is not in the services dict
        limiter = registry.get_limiter("unknown_api")

        assert isinstance(limiter, RateLimiter)
        assert limiter.name == "unknown_api"
        registry.close()


class TestRateLimitRegistryThreadSafety:
    """Tests for RateLimitRegistry thread safety."""

    def test_concurrent_get_limiter_same_service(self) -> None:
        """Concurrent get_limiter calls for same service return same instance."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        results: list[RateLimiter | NoOpLimiter] = []

        def get_limiter() -> RateLimiter | NoOpLimiter:
            return registry.get_limiter("shared_service")

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_limiter) for _ in range(100)]
            results = [f.result() for f in futures]

        # All results should be the same instance
        first = results[0]
        assert all(r is first for r in results)
        registry.close()

    def test_concurrent_get_limiter_different_services(self) -> None:
        """Concurrent get_limiter calls for different services work correctly."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        def get_limiter(service_name: str) -> RateLimiter | NoOpLimiter:
            return registry.get_limiter(service_name)

        services = [f"service_{i}" for i in range(20)]

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_limiter, svc) for svc in services]
            results = [f.result() for f in futures]

        # Should have 20 different limiters
        unique_limiters = {id(r) for r in results}
        assert len(unique_limiters) == 20
        registry.close()


class TestRateLimitRegistryCleanup:
    """Tests for RateLimitRegistry cleanup methods."""

    def test_reset_all_clears_limiters(self) -> None:
        """reset_all() closes all limiters and clears the registry."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        # Create some limiters
        limiter1 = registry.get_limiter("service_a")
        limiter2 = registry.get_limiter("service_b")

        # Mock close on the limiters
        with patch.object(limiter1, "close") as mock_close1, patch.object(limiter2, "close") as mock_close2:
            registry.reset_all()
            mock_close1.assert_called_once()
            mock_close2.assert_called_once()

        # New request should create new limiter (not cached)
        new_limiter = registry.get_limiter("service_a")
        assert new_limiter is not limiter1
        registry.close()

    def test_close_releases_resources(self) -> None:
        """close() closes all limiters."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        # Create some limiters
        limiter1 = registry.get_limiter("service_a")
        limiter2 = registry.get_limiter("service_b")

        # Mock close on the limiters
        with patch.object(limiter1, "close") as mock_close1, patch.object(limiter2, "close") as mock_close2:
            registry.close()
            mock_close1.assert_called_once()
            mock_close2.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        registry.get_limiter("test_service")

        # Should not raise on multiple calls
        registry.close()
        registry.close()

    def test_reset_all_allows_new_limiters(self) -> None:
        """After reset_all(), new limiters can be created."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_second=10,
        )
        registry = RateLimitRegistry(settings)

        original = registry.get_limiter("service")
        registry.reset_all()
        new = registry.get_limiter("service")

        # Should be a different instance after reset
        assert original is not new
        registry.close()
