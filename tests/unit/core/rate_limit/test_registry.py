"""Tests for RateLimitRegistry and NoOpLimiter."""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

import elspeth.core.rate_limit as rate_limit
from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
from elspeth.contracts.contexts import RateLimitRegistryProtocol
from elspeth.core.config import RateLimitSettings, ServiceRateLimit
from elspeth.core.rate_limit.limiter import RateLimiter
from elspeth.core.rate_limit.registry import NoOpLimiter, RateLimitRegistry
from elspeth.plugins.infrastructure.clients.base import AuditedClientBase
from elspeth.plugins.infrastructure.clients.dataverse import DataverseClient
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient
from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient
from elspeth.plugins.infrastructure.clients.retrieval.azure_search import AzureSearchProvider


def test_limiter_protocol_is_shared_public_surface() -> None:
    """Registry and audited clients should depend on the limiter behavior contract."""
    assert "LimiterProtocol" in rate_limit.__all__
    assert rate_limit.LimiterProtocol.__name__ == "LimiterProtocol"
    assert RateLimitRegistryProtocol.get_limiter.__annotations__["return"] == "LimiterProtocol"
    assert RateLimitRegistry.get_limiter.__annotations__["return"] == "LimiterProtocol"
    for client_type in (AuditedClientBase, AuditedHTTPClient, AuditedLLMClient, DataverseClient, AzureSearchProvider):
        assert client_type.__init__.__annotations__["limiter"] == "LimiterProtocol | None"


class TestNoOpLimiter:
    """Tests for NoOpLimiter class.

    NoOpLimiter provides the same interface as RateLimiter but does nothing.
    All operations succeed instantly without any rate limiting.
    """

    def test_acquire_does_nothing(self) -> None:
        """acquire() returns None without error."""
        limiter = NoOpLimiter()
        # Should not raise, should not block, and should return None
        limiter.acquire()
        limiter.acquire(weight=10)
        # No assertion needed - acquire() returns None by design

    def test_acquire_accepts_timeout_parameter(self) -> None:
        """acquire() accepts timeout parameter for API compatibility with RateLimiter.

        NoOpLimiter must have the same signature as RateLimiter so that
        callers can use either interchangeably via RateLimitRegistry.get_limiter().

        Regression test for P2-2026-01-31-noop-limiter-signature-mismatch.
        """
        limiter = NoOpLimiter()
        # Should not raise TypeError - timeout is accepted but ignored
        limiter.acquire(timeout=0.1)
        limiter.acquire(weight=5, timeout=1.0)
        limiter.acquire(timeout=None)  # Explicit None should also work

    def test_try_acquire_always_succeeds(self) -> None:
        """try_acquire() always returns True."""
        limiter = NoOpLimiter()
        assert limiter.try_acquire() is True
        assert limiter.try_acquire(weight=100) is True

    def test_close_does_nothing(self) -> None:
        """close() returns None and is idempotent."""
        limiter = NoOpLimiter()
        limiter.close()
        # Should be safe to call multiple times
        limiter.close()
        # No assertion needed - close() returns None by design

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
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter = registry.get_limiter("any_service")

        assert isinstance(limiter, NoOpLimiter)

    def test_same_noop_instance_for_all_services(self) -> None:
        """All services get the same NoOpLimiter instance when disabled."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("service_a")
        limiter2 = registry.get_limiter("service_b")
        limiter3 = registry.get_limiter("service_c")

        assert limiter1 is limiter2
        assert limiter2 is limiter3

    @pytest.mark.parametrize("weight", [0, -1])
    def test_disabled_limiter_acquire_rejects_non_positive_weight(self, weight: int) -> None:
        """Disabled registry preserves RateLimiter's positive weight contract."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)
        limiter = registry.get_limiter("disabled_service")

        with pytest.raises(ValueError, match="weight must be positive"):
            limiter.acquire(weight=weight)

    def test_disabled_limiter_acquire_rejects_non_int_weight(self) -> None:
        """Disabled registry rejects non-int acquire weights like RateLimiter."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)
        limiter = registry.get_limiter("disabled_service")

        with pytest.raises(TypeError, match="weight must be int"):
            limiter.acquire(weight="1")  # type: ignore[arg-type]

    @pytest.mark.parametrize("weight", [0, -1])
    def test_disabled_limiter_try_acquire_rejects_non_positive_weight(self, weight: int) -> None:
        """Disabled registry preserves try_acquire's positive weight contract."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)
        limiter = registry.get_limiter("disabled_service")

        with pytest.raises(ValueError, match="weight must be positive"):
            limiter.try_acquire(weight=weight)

    def test_disabled_limiter_try_acquire_rejects_non_int_weight(self) -> None:
        """Disabled registry rejects non-int try_acquire weights like RateLimiter."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)
        limiter = registry.get_limiter("disabled_service")

        with pytest.raises(TypeError, match="weight must be int"):
            limiter.try_acquire(weight="1")  # type: ignore[arg-type]

    @pytest.mark.parametrize("timeout", ["1.0", math.nan, math.inf, -1.0])
    def test_disabled_limiter_acquire_rejects_invalid_timeout(self, timeout: float | str) -> None:
        """Disabled registry preserves RateLimiter's timeout validation contract."""
        settings = RateLimitSettings(enabled=False)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)
        limiter = registry.get_limiter("disabled_service")

        with pytest.raises((TypeError, ValueError), match="timeout must be"):
            limiter.acquire(timeout=timeout)  # type: ignore[arg-type]


class TestRateLimitRegistryEnabled:
    """Tests for RateLimitRegistry when rate limiting is enabled."""

    def test_creates_limiter_for_unknown_service(self) -> None:
        """Registry creates new RateLimiter for unknown service."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter = registry.get_limiter("new_service")

        assert isinstance(limiter, RateLimiter)
        registry.close()

    def test_returns_same_limiter_for_same_service(self) -> None:
        """Registry returns cached limiter for repeated requests."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("my_service")
        limiter2 = registry.get_limiter("my_service")

        assert limiter1 is limiter2
        registry.close()

    def test_different_limiters_for_different_services(self) -> None:
        """Registry creates separate limiters for different services."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter_a = registry.get_limiter("service_a")
        limiter_b = registry.get_limiter("service_b")

        assert limiter_a is not limiter_b
        registry.close()

    def test_uses_service_specific_config(self) -> None:
        """Registry uses per-service config when available.

        Verifies that the service-specific rate limits are actually applied,
        not just that a limiter is created.
        """
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
            services={
                "openai": ServiceRateLimit(
                    requests_per_minute=100,
                ),
            },
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter = registry.get_limiter("openai")

        # Verify type and name
        assert isinstance(limiter, RateLimiter)
        assert limiter.name == "openai"

        # Verify the service-specific config is actually applied
        assert limiter._requests_per_minute == 100

        registry.close()

    def test_uses_default_config_for_unconfigured_service(self) -> None:
        """Registry uses default config for services not explicitly configured.

        Verifies that unconfigured services get the default rate limits,
        not the service-specific limits.
        """
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=15,
            services={
                "openai": ServiceRateLimit(requests_per_minute=100),
            },
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        # This service is not in the services dict
        limiter = registry.get_limiter("unknown_api")

        # Verify type and name
        assert isinstance(limiter, RateLimiter)
        assert limiter.name == "unknown_api"

        # Verify the default config is applied (not openai's config)
        assert limiter._requests_per_minute == 15

        registry.close()


class TestRateLimitRegistryThreadSafety:
    """Tests for RateLimitRegistry thread safety."""

    def test_concurrent_get_limiter_same_service(self) -> None:
        """Concurrent get_limiter calls for same service return same instance."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

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
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

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


class TestRateLimitRegistryHostnameServiceNames:
    """Verify registry handles hostname-style service names.

    Bug: elspeth-a485c2b464. RateLimitRegistry passes raw service names
    to RateLimiter which requires ^[A-Za-z][A-Za-z0-9_]*$. Hostname-style
    keys like 'api.example.com' crash at runtime.
    """

    def test_hostname_service_name_accepted(self) -> None:
        """Hostname-style service names should work (registry sanitizes)."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        # This should NOT crash — registry should sanitize the name
        limiter = registry.get_limiter("api.example.com")
        assert isinstance(limiter, RateLimiter)
        registry.close()

    def test_hostname_service_names_cached_correctly(self) -> None:
        """Same hostname returns same limiter instance."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("api.example.com")
        limiter2 = registry.get_limiter("api.example.com")
        assert limiter1 is limiter2
        registry.close()

    def test_different_hostnames_get_different_limiters(self) -> None:
        """Different hostnames should map to different limiters."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        limiter1 = registry.get_limiter("api.example.com")
        limiter2 = registry.get_limiter("api.other.com")
        assert limiter1 is not limiter2
        registry.close()


class TestRateLimitRegistryCleanup:
    """Tests for RateLimitRegistry cleanup methods."""

    def test_close_releases_resources(self) -> None:
        """close() closes all limiters."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

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
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        registry.get_limiter("test_service")

        # Should not raise on multiple calls
        registry.close()
        registry.close()

    def test_close_allows_new_limiters(self) -> None:
        """After close(), new limiters can be created."""
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        original = registry.get_limiter("service")
        registry.close()
        new = registry.get_limiter("service")

        # Should be a different instance after close
        assert original is not new
        registry.close()


class TestRateLimitRegistryNameCollision:
    """elspeth-2af4e98ee2: distinct service names must not collide onto one
    persistent SQLite bucket via a non-injective sanitizer."""

    def _config(self, rpm: int = 1) -> RuntimeRateLimitConfig:
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=rpm,
            persistence_path="rl.db",
        )
        return RuntimeRateLimitConfig.from_settings(settings)

    def _rate_limiter(self, registry: RateLimitRegistry, service_name: str) -> RateLimiter:
        limiter = registry.get_limiter(service_name)
        assert isinstance(limiter, RateLimiter)
        return limiter

    def test_colliding_names_get_independent_persistent_buckets(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        registry = RateLimitRegistry(self._config(rpm=1))
        try:
            a = self._rate_limiter(registry, "api.example")
            b = self._rate_limiter(registry, "api_example")
            assert a is not b
            # Distinct persistent bucket names (the SQLite table is f"ratelimit_{name}").
            assert a.name != b.name
            # Independent RPM=1 buckets: exhausting one must not exhaust the other.
            assert a.try_acquire() is True
            assert b.try_acquire() is True
        finally:
            registry.close()

    def test_already_valid_name_bucket_is_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stability guard: an already-valid service name keeps its byte-identical
        bucket name (edge-only migration — common case is untouched)."""
        monkeypatch.chdir(tmp_path)
        registry = RateLimitRegistry(self._config(rpm=10))
        try:
            assert self._rate_limiter(registry, "openai").name == "openai"
            assert self._rate_limiter(registry, "api_example").name == "api_example"
        finally:
            registry.close()

    def test_valid_service_name_matching_rewritten_bucket_gets_independent_bucket(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        registry = RateLimitRegistry(self._config(rpm=1))
        try:
            rewritten = self._rate_limiter(registry, "api.example")
            matching_raw = self._rate_limiter(registry, rewritten.name)

            assert rewritten.name != matching_raw.name
            assert rewritten.try_acquire() is True
            assert matching_raw.try_acquire() is True
        finally:
            registry.close()
