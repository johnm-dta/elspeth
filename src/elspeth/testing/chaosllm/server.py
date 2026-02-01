# src/elspeth/testing/chaosllm/server.py
"""Starlette ASGI application for ChaosLLM fake LLM server.

Provides OpenAI and Azure OpenAI compatible endpoints with configurable
error injection, latency simulation, and response generation.

Usage:
    from elspeth.testing.chaosllm.server import create_app, ChaosLLMServer
    from elspeth.testing.chaosllm.config import ChaosLLMConfig

    config = ChaosLLMConfig()
    app = create_app(config)

    # Or use the server class for more control
    server = ChaosLLMServer(config)
    app = server.app
"""

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from elspeth.testing.chaosllm.config import (
    ChaosLLMConfig,
    ErrorInjectionConfig,
    LatencyConfig,
    ResponseConfig,
)
from elspeth.testing.chaosllm.error_injector import ErrorDecision, ErrorInjector
from elspeth.testing.chaosllm.latency_simulator import LatencySimulator
from elspeth.testing.chaosllm.metrics import MetricsRecorder
from elspeth.testing.chaosllm.response_generator import ResponseGenerator

# Error type to OpenAI error type mapping
_ERROR_TYPE_MAPPING = {
    "rate_limit": "rate_limit_error",
    "capacity_529": "capacity_error",
    "service_unavailable": "server_error",
    "bad_gateway": "server_error",
    "gateway_timeout": "server_error",
    "internal_error": "server_error",
    "forbidden": "permission_error",
    "not_found": "not_found_error",
}

# Error type to message mapping
_ERROR_MESSAGE_MAPPING = {
    "rate_limit": "Rate limit exceeded. Please retry after the specified time.",
    "capacity_529": "The model is currently overloaded. Please retry later.",
    "service_unavailable": "The service is temporarily unavailable.",
    "bad_gateway": "Bad gateway error.",
    "gateway_timeout": "Gateway timeout.",
    "internal_error": "Internal server error.",
    "forbidden": "You do not have permission to access this resource.",
    "not_found": "The requested resource was not found.",
}


class ChaosLLMServer:
    """Main ChaosLLM server class.

    Encapsulates all server state and provides methods for runtime
    configuration updates and metrics access.

    Attributes:
        app: The Starlette ASGI application
        run_id: Current run identifier

    Usage:
        server = ChaosLLMServer(config)
        # Use server.app with uvicorn or test client

        # Runtime updates
        server.update_config({"error_injection": {"rate_limit_pct": 50.0}})
        stats = server.get_stats()
        server.reset()
    """

    def __init__(self, config: ChaosLLMConfig) -> None:
        """Initialize the ChaosLLM server.

        Args:
            config: Server configuration
        """
        self._config = config
        self._error_injector = ErrorInjector(config.error_injection)
        self._response_generator = ResponseGenerator(config.response)
        self._latency_simulator = LatencySimulator(config.latency)
        self._metrics_recorder = MetricsRecorder(config.metrics)
        self._record_run_info()

        # Build the Starlette app
        self._app = self._create_app()

    def _create_app(self) -> Starlette:
        """Create the Starlette application with all routes."""
        routes = [
            # Health endpoint
            Route("/health", self._health_endpoint, methods=["GET"]),
            # LLM endpoints
            Route("/v1/chat/completions", self._chat_completions_endpoint, methods=["POST"]),
            Route(
                "/openai/deployments/{deployment}/chat/completions",
                self._azure_chat_completions_endpoint,
                methods=["POST"],
            ),
            # Admin endpoints
            Route("/admin/config", self._admin_config_endpoint, methods=["GET", "POST"]),
            Route("/admin/stats", self._admin_stats_endpoint, methods=["GET"]),
            Route("/admin/reset", self._admin_reset_endpoint, methods=["POST"]),
            Route("/admin/export", self._admin_export_endpoint, methods=["GET"]),
        ]
        return Starlette(debug=False, routes=routes)

    @property
    def app(self) -> Starlette:
        """Get the Starlette ASGI application."""
        return self._app

    @property
    def run_id(self) -> str:
        """Get the current run ID."""
        return self._metrics_recorder.run_id

    def get_stats(self) -> dict[str, Any]:
        """Get current metrics statistics."""
        return self._metrics_recorder.get_stats()

    def reset(self) -> str:
        """Reset metrics and start a new run.

        Returns:
            The new run_id
        """
        self._error_injector.reset()
        self._response_generator.reset()
        self._metrics_recorder.reset()
        self._record_run_info()
        return self._metrics_recorder.run_id

    def export_metrics(self) -> dict[str, Any]:
        """Export raw metrics data for external pushback."""
        data = self._metrics_recorder.export_data()
        data["config"] = {
            "server": self._config.server.model_dump(),
            "metrics": self._config.metrics.model_dump(),
            **self._get_current_config(),
        }
        return data

    def update_config(self, updates: dict[str, Any]) -> None:
        """Update server configuration at runtime.

        Args:
            updates: Dict with sections to update (error_injection, response, latency)
        """
        if "error_injection" in updates:
            # Merge with existing config
            current_error = self._error_injector._config.model_dump()
            current_error.update(updates["error_injection"])
            error_config = ErrorInjectionConfig(**current_error)
            self._error_injector = ErrorInjector(error_config)

        if "response" in updates:
            current_response = self._response_generator._config.model_dump()
            current_response.update(updates["response"])
            response_config = ResponseConfig(**current_response)
            self._response_generator = ResponseGenerator(response_config)

        if "latency" in updates:
            current_latency = self._latency_simulator._config.model_dump()
            current_latency.update(updates["latency"])
            latency_config = LatencyConfig(**current_latency)
            self._latency_simulator = LatencySimulator(latency_config)

    def _get_current_config(self) -> dict[str, Any]:
        """Get current configuration as dict."""
        return {
            "error_injection": self._error_injector._config.model_dump(),
            "response": self._response_generator._config.model_dump(),
            "latency": self._latency_simulator._config.model_dump(),
        }

    def _record_run_info(self) -> None:
        """Persist run info for the current metrics run."""
        config_json = json.dumps(
            {
                "server": self._config.server.model_dump(),
                "metrics": self._config.metrics.model_dump(),
                **self._get_current_config(),
            },
            sort_keys=True,
        )
        self._metrics_recorder.save_run_info(
            config_json=config_json,
            preset_name=self._config.preset_name,
        )

    # === Endpoint handlers ===

    async def _health_endpoint(self, request: Request) -> JSONResponse:
        """Handle GET /health."""
        return JSONResponse(
            {
                "status": "healthy",
                "run_id": self._metrics_recorder.run_id,
                "started_utc": self._metrics_recorder.started_utc,
                "in_burst": self._error_injector.is_in_burst(),
            }
        )

    async def _chat_completions_endpoint(self, request: Request) -> Response:
        """Handle POST /v1/chat/completions (OpenAI format)."""
        return await self._handle_completion_request(request, endpoint="/v1/chat/completions")

    async def _azure_chat_completions_endpoint(self, request: Request) -> Response:
        """Handle POST /openai/deployments/{deployment}/chat/completions (Azure format)."""
        deployment = request.path_params["deployment"]
        return await self._handle_completion_request(
            request,
            endpoint=f"/openai/deployments/{deployment}/chat/completions",
            deployment=deployment,
        )

    async def _admin_config_endpoint(self, request: Request) -> JSONResponse:
        """Handle GET/POST /admin/config."""
        if request.method == "GET":
            return JSONResponse(self._get_current_config())

        # POST - update config
        body = await request.json()
        self.update_config(body)
        return JSONResponse({"status": "updated", "config": self._get_current_config()})

    async def _admin_stats_endpoint(self, request: Request) -> JSONResponse:
        """Handle GET /admin/stats."""
        return JSONResponse(self._metrics_recorder.get_stats())

    async def _admin_reset_endpoint(self, request: Request) -> JSONResponse:
        """Handle POST /admin/reset."""
        new_run_id = self.reset()
        return JSONResponse({"status": "reset", "new_run_id": new_run_id})

    async def _admin_export_endpoint(self, request: Request) -> JSONResponse:
        """Handle GET /admin/export."""
        return JSONResponse(self.export_metrics())

    # === Request handling ===

    async def _handle_completion_request(
        self,
        request: Request,
        endpoint: str,
        deployment: str | None = None,
    ) -> Response:
        """Handle a chat completion request with error injection and metrics.

        Request flow:
        1. Record request start
        2. Check error injection
        3. If error: return error response
        4. If success: generate response, add latency, return
        """
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()
        timestamp_utc = datetime.now(UTC).isoformat()

        # Parse request body
        body = await request.json()
        model = body.get("model", "gpt-4")
        messages = body.get("messages", [])

        # Extract override headers
        mode_override = request.headers.get("X-Fake-Response-Mode")
        template_override = request.headers.get("X-Fake-Template")

        # Check for error injection
        decision = self._error_injector.decide()

        if decision.should_inject:
            if decision.error_type == "slow_response":
                return await self._handle_slow_response(
                    decision=decision,
                    request_id=request_id,
                    timestamp_utc=timestamp_utc,
                    endpoint=endpoint,
                    deployment=deployment,
                    body=body,
                    mode_override=mode_override,
                    template_override=template_override,
                    start_time=start_time,
                )
            return await self._handle_error_injection(
                decision=decision,
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                deployment=deployment,
                model=model,
                message_count=len(messages),
                start_time=start_time,
            )

        # No error - generate successful response
        return await self._handle_success_response(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            deployment=deployment,
            body=body,
            mode_override=mode_override,
            template_override=template_override,
            start_time=start_time,
        )

    async def _handle_slow_response(
        self,
        decision: ErrorDecision,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        body: dict[str, Any],
        mode_override: str | None,
        template_override: str | None,
        start_time: float,
    ) -> JSONResponse:
        """Handle a slow response that eventually succeeds."""
        delay = decision.delay_sec if decision.delay_sec is not None else 15.0
        return await self._handle_success_response(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            deployment=deployment,
            body=body,
            mode_override=mode_override,
            template_override=template_override,
            start_time=start_time,
            extra_delay_sec=delay,
            injection_type="slow_response",
        )

    async def _handle_error_injection(
        self,
        decision: ErrorDecision,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        model: str,
        message_count: int,
        start_time: float,
    ) -> Response:
        """Handle an injected error response."""
        # Connection-level errors
        if decision.is_connection_level:
            return await self._handle_connection_error(
                decision=decision,
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                deployment=deployment,
                model=model,
                message_count=message_count,
                start_time=start_time,
            )

        # Malformed responses
        if decision.is_malformed:
            return await self._handle_malformed_response(
                decision=decision,
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                deployment=deployment,
                model=model,
                message_count=message_count,
                start_time=start_time,
            )

        # HTTP-level errors
        return await self._handle_http_error(
            decision=decision,
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            deployment=deployment,
            model=model,
            message_count=message_count,
            start_time=start_time,
        )

    async def _handle_connection_error(
        self,
        decision: ErrorDecision,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        model: str,
        message_count: int,
        start_time: float,
    ) -> Response:
        """Handle a connection-level error (timeout, reset, stall)."""
        error_type = decision.error_type

        if error_type == "connection_failed":
            lead_delay = decision.start_delay_sec if decision.start_delay_sec is not None else 0.0
            if lead_delay > 0:
                await asyncio.sleep(lead_delay)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_request(
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                outcome="error_injected",
                deployment=deployment,
                model=model,
                status_code=None,
                error_type="connection_failed",
                injection_type="connection_failed",
                latency_ms=elapsed_ms,
                injected_delay_ms=lead_delay * 1000 if lead_delay > 0 else None,
                message_count=message_count,
            )
            raise ConnectionResetError("Connection failed after lead time")

        if error_type == "connection_stall":
            start_delay = decision.start_delay_sec if decision.start_delay_sec is not None else 0.0
            stall_delay = decision.delay_sec if decision.delay_sec is not None else 60.0
            if start_delay > 0:
                await asyncio.sleep(start_delay)
            await asyncio.sleep(stall_delay)

            elapsed_ms = (time.monotonic() - start_time) * 1000
            injected_delay_ms = (start_delay + stall_delay) * 1000
            self._record_request(
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                outcome="error_injected",
                deployment=deployment,
                model=model,
                status_code=None,
                error_type="connection_stall",
                injection_type="connection_stall",
                latency_ms=elapsed_ms,
                injected_delay_ms=injected_delay_ms if injected_delay_ms > 0 else None,
                message_count=message_count,
            )
            raise ConnectionResetError("Connection stalled and was closed by server")

        if error_type == "timeout":
            # Delay, then either return 504 or drop the connection
            delay = decision.delay_sec if decision.delay_sec is not None else 60.0
            await asyncio.sleep(delay)

            # After delay, record and then either respond or disconnect
            elapsed_ms = (time.monotonic() - start_time) * 1000
            status_code = decision.status_code
            self._record_request(
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                outcome="error_injected",
                deployment=deployment,
                model=model,
                status_code=status_code,
                error_type="timeout",
                injection_type="timeout",
                latency_ms=elapsed_ms,
                injected_delay_ms=delay * 1000,
                message_count=message_count,
            )
            if status_code == 504:
                return JSONResponse(
                    {"error": {"type": "timeout", "message": "Request timed out"}},
                    status_code=504,
                )
            raise ConnectionResetError("Request timed out")

        elif error_type == "connection_reset":
            # Record the attempt
            elapsed_ms = (time.monotonic() - start_time) * 1000
            self._record_request(
                request_id=request_id,
                timestamp_utc=timestamp_utc,
                endpoint=endpoint,
                outcome="error_injected",
                deployment=deployment,
                model=model,
                status_code=None,
                error_type="connection_reset",
                injection_type="connection_reset",
                latency_ms=elapsed_ms,
                message_count=message_count,
            )
            # Close the connection abruptly by raising an exception
            # In test client, this manifests as a disconnection
            raise ConnectionResetError("Connection reset by server")

        elif error_type == "slow_response":
            raise ValueError("slow_response should be handled by _handle_slow_response")

        # Should not reach here
        raise ValueError(f"Unknown connection error type: {error_type}")

    async def _handle_http_error(
        self,
        decision: ErrorDecision,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        model: str,
        message_count: int,
        start_time: float,
    ) -> JSONResponse:
        """Handle an HTTP-level error response."""
        # HTTP errors always have status_code and error_type set
        status_code = decision.status_code
        error_type = decision.error_type

        if status_code is None or error_type is None:
            raise ValueError("HTTP error decision must have status_code and error_type")

        # Build headers
        headers: dict[str, str] = {}
        if decision.retry_after_sec is not None:
            headers["Retry-After"] = str(decision.retry_after_sec)

        # Build error body
        openai_error_type = _ERROR_TYPE_MAPPING[error_type]
        error_message = _ERROR_MESSAGE_MAPPING[error_type]

        body = {
            "error": {
                "type": openai_error_type,
                "message": error_message,
                "code": error_type,
            }
        }

        # Record metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._record_request(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            outcome="error_injected",
            deployment=deployment,
            model=model,
            status_code=status_code,
            error_type=error_type,
            injection_type=error_type,
            latency_ms=elapsed_ms,
            message_count=message_count,
        )

        return JSONResponse(body, status_code=status_code, headers=headers)

    async def _handle_malformed_response(
        self,
        decision: ErrorDecision,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        model: str,
        message_count: int,
        start_time: float,
    ) -> Response:
        """Handle a malformed response (200 with bad content)."""
        malformed_type = decision.malformed_type

        # Record metrics first
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._record_request(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            outcome="error_malformed",
            deployment=deployment,
            model=model,
            status_code=200,
            error_type=f"malformed_{malformed_type}",
            injection_type=f"malformed_{malformed_type}",
            latency_ms=elapsed_ms,
            message_count=message_count,
        )

        if malformed_type == "invalid_json":
            # Return invalid JSON
            return Response(
                content=b'{malformed json... "unclosed',
                status_code=200,
                media_type="application/json",
            )

        elif malformed_type == "truncated":
            # Return truncated JSON
            valid_json = '{"id": "fake-123", "object": "chat.completion", "choices": [{"message": {"content": '
            return Response(
                content=valid_json.encode(),
                status_code=200,
                media_type="application/json",
            )

        elif malformed_type == "empty_body":
            # Return empty response body
            return Response(
                content=b"",
                status_code=200,
                media_type="application/json",
            )

        elif malformed_type == "missing_fields":
            # Return valid JSON but missing required fields
            return JSONResponse(
                {"id": "fake-123", "object": "chat.completion"},
                status_code=200,
            )

        elif malformed_type == "wrong_content_type":
            # Return with wrong Content-Type
            return Response(
                content=b"<html><body>Not JSON</body></html>",
                status_code=200,
                media_type="text/html",
            )

        # Should not reach here
        raise ValueError(f"Unknown malformed type: {malformed_type}")

    async def _handle_success_response(
        self,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        deployment: str | None,
        body: dict[str, Any],
        mode_override: str | None,
        template_override: str | None,
        start_time: float,
        extra_delay_sec: float | None = None,
        injection_type: str | None = None,
    ) -> JSONResponse:
        """Handle a successful response with latency simulation."""
        # Add latency
        delay = self._latency_simulator.simulate()
        total_delay = delay + (extra_delay_sec or 0.0)
        if total_delay > 0:
            await asyncio.sleep(total_delay)

        # Generate response
        response = self._response_generator.generate(
            body,
            mode_override=mode_override,
            template_override=template_override,
        )

        # Record metrics
        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._record_request(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            outcome="success",
            deployment=deployment,
            model=body.get("model", "gpt-4"),
            status_code=200,
            latency_ms=elapsed_ms,
            injected_delay_ms=total_delay * 1000 if total_delay > 0 else None,
            message_count=len(body.get("messages", [])),
            prompt_tokens_approx=response.prompt_tokens,
            response_tokens=response.completion_tokens,
            response_mode=mode_override or self._response_generator._config.mode,
            injection_type=injection_type,
        )

        return JSONResponse(response.to_dict())

    def _record_request(
        self,
        *,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        outcome: str,
        deployment: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        injection_type: str | None = None,
        latency_ms: float | None = None,
        injected_delay_ms: float | None = None,
        message_count: int | None = None,
        prompt_tokens_approx: int | None = None,
        response_tokens: int | None = None,
        response_mode: str | None = None,
    ) -> None:
        """Record a request to metrics."""
        self._metrics_recorder.record_request(
            request_id=request_id,
            timestamp_utc=timestamp_utc,
            endpoint=endpoint,
            outcome=outcome,
            deployment=deployment,
            model=model,
            status_code=status_code,
            error_type=error_type,
            injection_type=injection_type,
            latency_ms=latency_ms,
            injected_delay_ms=injected_delay_ms,
            message_count=message_count,
            prompt_tokens_approx=prompt_tokens_approx,
            response_tokens=response_tokens,
            response_mode=response_mode,
        )


def create_app(config: ChaosLLMConfig) -> Starlette:
    """Create a Starlette ASGI application from config.

    This is a convenience function for simple use cases. For more control
    over the server (runtime config updates, metrics access), use the
    ChaosLLMServer class directly.

    Args:
        config: ChaosLLM configuration

    Returns:
        Starlette ASGI application
    """
    server = ChaosLLMServer(config)
    # Attach server to app.state for admin endpoints to access
    server.app.state.server = server
    return server.app
