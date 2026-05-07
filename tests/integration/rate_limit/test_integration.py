# tests/integration/rate_limit/test_integration.py
"""Integration tests for rate limit registry wiring.

These tests verify that RateLimitRegistry is properly wired through
the CLI -> Orchestrator -> PluginContext pipeline.

Migrated from tests/integration/test_rate_limit_integration.py
"""

import time
from typing import Any

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.config.runtime import RuntimeRateLimitConfig
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.config import RateLimitSettings
from elspeth.core.rate_limit import RateLimitRegistry
from elspeth.engine import Orchestrator
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.landscape import make_landscape_db


class RateLimitAwareTransform(BaseTransform):
    """Test transform that uses rate limiting from context.

    Follows the designed architecture: captures rate_limit_registry in
    on_start() from LifecycleContext and uses it from self in process().
    """

    name = "rate_limit_test"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._service_name = config.get("service_name", "test_service")
        self._call_times: list[float] = []
        self._registry: RateLimitRegistry | None = None

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture rate_limit_registry from lifecycle context."""
        super().on_start(ctx)
        self._registry = ctx.rate_limit_registry

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Process row, using rate limiter if available."""
        if self._registry is not None:
            limiter = self._registry.get_limiter(self._service_name)
            limiter.acquire()

        # Record the time of this call
        self._call_times.append(time.perf_counter())

        return TransformResult.success(make_pipeline_row({"processed": True, **row}), success_reason={"action": "processed"})

    @property
    def call_times(self) -> list[float]:
        """Get the recorded call times."""
        return self._call_times


class TestRateLimitRegistryInOrchestrator:
    """Test that RateLimitRegistry is properly passed to Orchestrator."""

    def test_orchestrator_accepts_rate_limit_registry(self) -> None:
        """Orchestrator constructor accepts rate_limit_registry parameter."""
        db = make_landscape_db()
        settings = RateLimitSettings(enabled=True, default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            # Should not raise
            orchestrator = Orchestrator(db, rate_limit_registry=registry)
            assert orchestrator._rate_limit_registry is registry
        finally:
            registry.close()
            db.close()

    def test_orchestrator_accepts_none_registry(self) -> None:
        """Orchestrator works without rate limit registry."""
        db = make_landscape_db()

        try:
            orchestrator = Orchestrator(db, rate_limit_registry=None)
            assert orchestrator._rate_limit_registry is None
        finally:
            db.close()


class TestRateLimitRegistryInContext:
    """Test that RateLimitRegistry is available in PluginContext."""

    def test_context_has_rate_limit_registry_field(self) -> None:
        """PluginContext has rate_limit_registry field."""
        settings = RateLimitSettings(enabled=True, default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            ctx = PluginContext(
                run_id="test-run",
                config={},
                rate_limit_registry=registry,
            )
            assert ctx.rate_limit_registry is registry
        finally:
            registry.close()

    def test_context_without_registry(self) -> None:
        """PluginContext works without rate limit registry."""
        ctx = PluginContext(
            run_id="test-run",
            config={},
            rate_limit_registry=None,
        )
        assert ctx.rate_limit_registry is None


class TestRateLimitThrottling:
    """Test that rate limiting actually throttles requests.

    These tests use try_acquire() for deterministic bucket-full detection
    and acquire(timeout=...) for blocking verification. No wall-clock timing
    assertions — the rate limiter's logic is purely computational.
    """

    def test_rate_limiter_rejects_excess_requests(self) -> None:
        """Rate limiter rejects requests after bucket is exhausted.

        With 10 req/min bucket: first 10 try_acquire() succeed, 11th fails.
        Deterministic — no wall-clock dependency.
        """
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            limiter = registry.get_limiter("throttle_test")

            # First 10 acquisitions fill the bucket
            for i in range(10):
                assert limiter.try_acquire(), f"Call {i + 1} should succeed (bucket holds 10)"

            # 11th is rejected — bucket is full
            assert not limiter.try_acquire(), "11th call should be rejected (bucket full)"
        finally:
            registry.close()

    def test_rate_limiter_blocking_acquire_times_out(self) -> None:
        """acquire() blocks and raises TimeoutError when bucket is full.

        Verifies the blocking path through the registry-created limiter,
        exercising the same code path as production PluginContext usage.
        """
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            limiter = registry.get_limiter("blocking_test")

            # Exhaust the bucket
            for _ in range(10):
                limiter.acquire()

            # Blocking acquire should time out (bucket won't refill in 50ms)
            with pytest.raises(TimeoutError):
                limiter.acquire(timeout=0.05)
        finally:
            registry.close()

    def test_rate_limiter_wired_through_transform(self) -> None:
        """Rate limiter works end-to-end through PluginContext and transform.

        Exercises the full integration path: Registry -> PluginContext -> Transform.
        First 10 rows process successfully, then the bucket is exhausted.
        """
        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=10,
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            transform = RateLimitAwareTransform({"service_name": "wiring_test"})
            ctx = PluginContext(
                run_id="test-run",
                config={},
                rate_limit_registry=registry,
            )

            # Lifecycle: on_start captures rate_limit_registry from context
            transform.on_start(ctx)

            # Process 10 rows — all should succeed (fills bucket)
            for i in range(10):
                result = transform.process(make_pipeline_row({"id": i}), ctx)
                assert result.status == "success", f"Row {i} should process successfully"

            assert len(transform.call_times) == 10

            # Underlying limiter should now be exhausted
            limiter = registry.get_limiter("wiring_test")
            assert not limiter.try_acquire(), "Bucket should be full after 10 rows"
        finally:
            registry.close()

    def test_disabled_rate_limit_no_throttle(self) -> None:
        """Disabled rate limiting never rejects requests."""
        settings = RateLimitSettings(
            enabled=False,
            default_requests_per_minute=1,  # Would reject after 1 if enabled
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            limiter = registry.get_limiter("no_throttle_test")

            # All calls succeed even though requests_per_minute=1
            for i in range(20):
                assert limiter.try_acquire(), f"Disabled limiter should never reject (call {i + 1})"
        finally:
            registry.close()


class TestRateLimitServiceConfig:
    """Test per-service rate limit configuration."""

    def test_per_service_rate_limits(self) -> None:
        """Different services can have different rate limits."""
        from elspeth.core.config import ServiceRateLimit

        settings = RateLimitSettings(
            enabled=True,
            default_requests_per_minute=60,
            services={
                "slow_service": ServiceRateLimit(requests_per_minute=10),
                "fast_service": ServiceRateLimit(requests_per_minute=600),
            },
        )
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            # Get limiters for different services
            slow_limiter = registry.get_limiter("slow_service")
            fast_limiter = registry.get_limiter("fast_service")
            default_limiter = registry.get_limiter("unknown_service")

            # All should be real limiters (enabled=True)
            from elspeth.core.rate_limit import NoOpLimiter

            assert not isinstance(slow_limiter, NoOpLimiter)
            assert not isinstance(fast_limiter, NoOpLimiter)
            assert not isinstance(default_limiter, NoOpLimiter)
        finally:
            registry.close()


class TestAuditedClientRateLimiting:
    """Test that audited clients actually use rate limiters."""

    def test_audited_llm_client_acquires_rate_limit(self) -> None:
        """AuditedLLMClient calls limiter.acquire() before making calls.

        This verifies that P2-2026-02-01 (rate limit registry not consumed)
        is fixed - audited clients now actually use rate limiters.
        """
        from unittest.mock import MagicMock

        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry
        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        settings = RateLimitSettings(enabled=True, default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            # Get a limiter
            limiter = registry.get_limiter("test_llm_service")

            # Spy on the limiter's acquire method
            original_acquire = limiter.acquire
            acquire_call_count = 0

            def counting_acquire(weight: int = 1, timeout: float | None = None) -> None:
                nonlocal acquire_call_count
                acquire_call_count += 1
                return original_acquire(weight, timeout)

            limiter.acquire = counting_acquire  # type: ignore[method-assign]

            # Mock the recorder (we don't need full landscape infrastructure for this test)
            mock_recorder = MagicMock()
            mock_recorder.allocate_call_index.return_value = 0

            # Mock the underlying OpenAI client
            mock_openai = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
            mock_response.model = "gpt-4"
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump.return_value = {}
            mock_openai.chat.completions.create.return_value = mock_response

            # Create audited client WITH limiter
            client = AuditedLLMClient(
                execution=mock_recorder,
                state_id="test-state-001",
                run_id="test-run-001",
                telemetry_emit=lambda event: None,
                underlying_client=mock_openai,
                provider="test",
                limiter=limiter,
            )

            # Make a call
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
            )

            # Verify limiter.acquire() was called
            assert acquire_call_count == 1, "limiter.acquire() should be called before LLM call"

            # Make another call
            mock_recorder.allocate_call_index.return_value = 1
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "World"}],
            )

            # Verify acquire called again
            assert acquire_call_count == 2, "limiter.acquire() should be called for each LLM call"
        finally:
            registry.close()

    def test_audited_http_client_acquires_rate_limit(self) -> None:
        """AuditedHTTPClient calls limiter.acquire() before making calls."""
        from unittest.mock import MagicMock, patch

        from elspeth.core.config import RateLimitSettings
        from elspeth.core.rate_limit import RateLimitRegistry
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        settings = RateLimitSettings(enabled=True, default_requests_per_minute=60)
        config = RuntimeRateLimitConfig.from_settings(settings)
        registry = RateLimitRegistry(config)

        try:
            # Get a limiter
            limiter = registry.get_limiter("test_http_service")

            # Spy on the limiter's acquire method
            original_acquire = limiter.acquire
            acquire_call_count = 0

            def counting_acquire(weight: int = 1, timeout: float | None = None) -> None:
                nonlocal acquire_call_count
                acquire_call_count += 1
                return original_acquire(weight, timeout)

            limiter.acquire = counting_acquire  # type: ignore[method-assign]

            # Mock the recorder
            mock_recorder = MagicMock()
            mock_recorder.allocate_call_index.return_value = 0

            # Mock httpx.Client to avoid actual network calls
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"result": "ok"}
            mock_response.content = b'{"result": "ok"}'
            mock_response.text = '{"result": "ok"}'

            with patch("httpx.Client") as mock_client_class:
                # Create audited client WITH limiter (inside patch so __init__ gets the mock)
                client = AuditedHTTPClient(
                    execution=mock_recorder,
                    state_id="test-state-001",
                    run_id="test-run-001",
                    telemetry_emit=lambda event: None,
                    timeout=30.0,
                    limiter=limiter,
                )
                mock_client_instance = mock_client_class.return_value
                mock_client_instance.post.return_value = mock_response

                # Make a call
                client.post("https://api.example.com/v1/test", json={"data": "value"})

            # Verify limiter.acquire() was called
            assert acquire_call_count == 1, "limiter.acquire() should be called before HTTP call"
        finally:
            registry.close()

    def test_audited_client_without_limiter_no_throttle(self) -> None:
        """AuditedLLMClient works without limiter."""
        from unittest.mock import MagicMock

        from elspeth.plugins.infrastructure.clients.llm import AuditedLLMClient

        # Mock the recorder
        mock_recorder = MagicMock()
        mock_recorder.allocate_call_index.return_value = 0

        # Mock the underlying OpenAI client
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_response.model = "gpt-4"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.model_dump.return_value = {}
        mock_openai.chat.completions.create.return_value = mock_response

        # Create audited client WITHOUT limiter (limiter=None is default)
        client = AuditedLLMClient(
            execution=mock_recorder,
            state_id="test-state-001",
            run_id="test-run-001",
            telemetry_emit=lambda event: None,
            underlying_client=mock_openai,
            provider="test",
            # limiter not passed - should default to None
        )

        # Make a call - should not raise
        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.content == "Hello!"
