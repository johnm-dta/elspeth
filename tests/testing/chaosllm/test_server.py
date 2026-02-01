# tests/testing/chaosllm/test_server.py
"""Tests for ChaosLLM HTTP server."""

import json
import sqlite3
import time

import pytest
from starlette.testclient import TestClient

from elspeth.testing.chaosllm.config import (
    ChaosLLMConfig,
    ErrorInjectionConfig,
    LatencyConfig,
    MetricsConfig,
    ResponseConfig,
)
from elspeth.testing.chaosllm.server import (
    ChaosLLMServer,
    create_app,
)


@pytest.fixture
def tmp_metrics_db(tmp_path):
    """Create a temporary metrics database path."""
    return str(tmp_path / "test-metrics.db")


@pytest.fixture
def config(tmp_metrics_db):
    """Create a basic ChaosLLM config for testing."""
    return ChaosLLMConfig(
        metrics=MetricsConfig(database=tmp_metrics_db),
        latency=LatencyConfig(base_ms=0, jitter_ms=0),  # No latency for tests
    )


@pytest.fixture
def client(config):
    """Create a test client for the ChaosLLM server."""
    app = create_app(config)
    return TestClient(app)


@pytest.fixture
def server(config):
    """Create a ChaosLLMServer instance for testing."""
    return ChaosLLMServer(config)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "run_id" in data

    def test_health_check_includes_burst_status(self, tmp_metrics_db):
        """Health endpoint includes burst mode status."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            error_injection=ErrorInjectionConfig(burst={"enabled": True, "interval_sec": 30, "duration_sec": 5}),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "in_burst" in data


class TestOpenAICompletionsEndpoint:
    """Tests for POST /v1/chat/completions (OpenAI format)."""

    def test_basic_completion(self, client):
        """Basic completion request returns valid response."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI response format
        assert "id" in data
        assert data["id"].startswith("fake-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert data["model"] == "gpt-4"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "usage" in data

    def test_completion_with_temperature(self, client):
        """Completion request with temperature parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
            },
        )
        assert response.status_code == 200

    def test_completion_with_max_tokens(self, client):
        """Completion request with max_tokens parameter."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
            },
        )
        assert response.status_code == 200


class TestAzureCompletionsEndpoint:
    """Tests for POST /openai/deployments/{deployment}/chat/completions (Azure format)."""

    def test_azure_completion(self, client):
        """Azure completion request returns valid response."""
        response = client.post(
            "/openai/deployments/my-gpt4-deployment/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI response format
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert "usage" in data

    def test_azure_completion_with_api_version(self, client):
        """Azure endpoint accepts api-version query parameter."""
        response = client.post(
            "/openai/deployments/my-deployment/chat/completions?api-version=2024-02-01",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200

    def test_azure_completion_extracts_deployment(self, client):
        """Azure endpoint extracts deployment name from path."""
        response = client.post(
            "/openai/deployments/custom-deployment-name/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 200


class TestErrorInjection:
    """Tests for error injection behavior."""

    def test_rate_limit_error(self, tmp_metrics_db):
        """100% rate limit returns 429."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(rate_limit_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 429
        assert "Retry-After" in response.headers

    def test_capacity_529_error(self, tmp_metrics_db):
        """100% capacity error returns 529."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(capacity_529_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 529
        assert "Retry-After" in response.headers

    def test_internal_error(self, tmp_metrics_db):
        """100% internal error returns 500."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(internal_error_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 500

    def test_service_unavailable_error(self, tmp_metrics_db):
        """100% service unavailable returns 503."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(service_unavailable_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 503

    def test_slow_response_returns_success(self, tmp_metrics_db):
        """Slow response delays but still returns a successful response."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(
                slow_response_pct=100.0,
                slow_response_sec=(0, 0),
            ),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"


class TestMalformedResponses:
    """Tests for malformed response injection."""

    def test_invalid_json_response(self, tmp_metrics_db):
        """100% invalid JSON returns malformed JSON body."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(invalid_json_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200

        # Should NOT be valid JSON
        with pytest.raises(json.JSONDecodeError):
            response.json()

    def test_empty_body_response(self, tmp_metrics_db):
        """100% empty body returns empty response."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(empty_body_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200
        assert response.content == b""

    def test_missing_fields_response(self, tmp_metrics_db):
        """100% missing fields returns JSON without choices/usage."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(missing_fields_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200
        data = response.json()

        # Should be missing choices or usage
        assert "choices" not in data or "usage" not in data

    def test_wrong_content_type_response(self, tmp_metrics_db):
        """100% wrong content type returns text/html."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(wrong_content_type_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_truncated_response(self, tmp_metrics_db):
        """100% truncated returns cut-off JSON."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(truncated_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert response.status_code == 200

        # Should NOT be valid JSON due to truncation
        with pytest.raises(json.JSONDecodeError):
            response.json()


class TestResponseModeOverrides:
    """Tests for per-request response mode overrides via headers."""

    def test_mode_override_header(self, tmp_metrics_db):
        """X-Fake-Response-Mode header overrides configured mode."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            response=ResponseConfig(mode="random"),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Test message"}],
            },
            headers={"X-Fake-Response-Mode": "echo"},
        )
        assert response.status_code == 200
        data = response.json()

        # Echo mode should return the last user message
        content = data["choices"][0]["message"]["content"]
        assert content == "Echo: Test message"

    def test_template_override_header(self, tmp_metrics_db):
        """X-Fake-Template header overrides configured template."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            response=ResponseConfig(mode="template"),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
            headers={
                "X-Fake-Response-Mode": "template",
                "X-Fake-Template": "Custom template: {{ model }}",
            },
        )
        assert response.status_code == 200
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Custom template: gpt-4"


class TestAdminConfigEndpoint:
    """Tests for /admin/config endpoint."""

    def test_get_config(self, client):
        """GET /admin/config returns current configuration."""
        response = client.get("/admin/config")
        assert response.status_code == 200
        data = response.json()

        # Should have all config sections
        assert "error_injection" in data
        assert "response" in data
        assert "latency" in data

    def test_post_config_updates_error_injection(self, client):
        """POST /admin/config updates error injection settings."""
        # First verify current state
        response = client.get("/admin/config")
        original = response.json()
        assert original["error_injection"]["rate_limit_pct"] == 0.0

        # Update config
        response = client.post(
            "/admin/config",
            json={"error_injection": {"rate_limit_pct": 50.0}},
        )
        assert response.status_code == 200

        # Verify update
        response = client.get("/admin/config")
        updated = response.json()
        assert updated["error_injection"]["rate_limit_pct"] == 50.0


class TestAdminStatsEndpoint:
    """Tests for /admin/stats endpoint."""

    def test_get_stats_empty(self, client):
        """GET /admin/stats returns stats even when empty."""
        response = client.get("/admin/stats")
        assert response.status_code == 200
        data = response.json()

        assert "run_id" in data
        assert "started_utc" in data
        assert "total_requests" in data
        assert data["total_requests"] == 0

    def test_stats_increment_after_request(self, client):
        """Stats increment after successful request."""
        # Make a request
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )

        # Check stats
        response = client.get("/admin/stats")
        data = response.json()
        assert data["total_requests"] == 1


class TestAdminResetEndpoint:
    """Tests for /admin/reset endpoint."""

    def test_reset_clears_stats(self, client):
        """POST /admin/reset clears metrics and starts new run."""
        # Make some requests
        for _ in range(3):
            client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4", "messages": []},
            )

        # Verify we have stats
        response = client.get("/admin/stats")
        assert response.json()["total_requests"] == 3

        # Get original run_id
        original_run_id = response.json()["run_id"]

        # Reset
        response = client.post("/admin/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reset"
        assert "new_run_id" in data
        assert data["new_run_id"] != original_run_id

        # Verify stats are cleared
        response = client.get("/admin/stats")
        assert response.json()["total_requests"] == 0

    def test_reset_records_run_info(self, tmp_metrics_db):
        """POST /admin/reset persists run_info for new run."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            preset_name="gentle",
        )
        app = create_app(config)
        client = TestClient(app)

        # Initial run info should be recorded on startup
        stats = client.get("/admin/stats").json()
        run_id = stats["run_id"]
        with sqlite3.connect(tmp_metrics_db) as conn:
            row = conn.execute("SELECT run_id, preset_name, config_json FROM run_info").fetchone()
        assert row is not None
        assert row[0] == run_id
        assert row[1] == "gentle"
        assert row[2]

        # Reset should replace run_info with new run
        reset = client.post("/admin/reset").json()
        new_run_id = reset["new_run_id"]
        assert new_run_id != run_id
        with sqlite3.connect(tmp_metrics_db) as conn:
            rows = conn.execute("SELECT run_id, preset_name FROM run_info").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == new_run_id
        assert rows[0][1] == "gentle"


class TestMetricsRecording:
    """Tests for metrics recording behavior."""

    def test_successful_request_recorded(self, client):
        """Successful requests are recorded in metrics."""
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )

        response = client.get("/admin/stats")
        data = response.json()
        assert data["total_requests"] == 1
        assert data["requests_by_outcome"].get("success", 0) == 1

    def test_error_request_recorded(self, tmp_metrics_db):
        """Error responses are recorded in metrics."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(rate_limit_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )

        response = client.get("/admin/stats")
        data = response.json()
        assert data["total_requests"] == 1
        # Should be recorded as error
        assert data["requests_by_outcome"].get("error_injected", 0) == 1


class TestLatencySimulation:
    """Tests for latency simulation."""

    def test_latency_applied_to_requests(self, tmp_metrics_db):
        """Latency is applied to successful requests."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=100, jitter_ms=0),  # Fixed 100ms latency
        )
        app = create_app(config)
        client = TestClient(app)

        start = time.monotonic()
        client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        # Should take at least 100ms (allowing some margin)
        assert elapsed_ms >= 90


class TestChaosLLMServer:
    """Tests for the ChaosLLMServer class."""

    def test_server_creation(self, config):
        """ChaosLLMServer can be created from config."""
        server = ChaosLLMServer(config)
        assert server.app is not None
        assert server.run_id is not None

    def test_server_reset(self, server):
        """Server reset creates new run_id."""
        original_run_id = server.run_id
        server.reset()
        assert server.run_id != original_run_id

    def test_get_stats(self, server):
        """Server get_stats returns metrics."""
        stats = server.get_stats()
        assert "run_id" in stats
        assert "total_requests" in stats

    def test_update_config(self, server):
        """Server can update error injection config."""
        original_rate = server._error_injector._config.rate_limit_pct
        assert original_rate == 0.0

        server.update_config({"error_injection": {"rate_limit_pct": 25.0}})
        assert server._error_injector._config.rate_limit_pct == 25.0


class TestErrorResponseBodies:
    """Tests for error response body format."""

    def test_rate_limit_error_body(self, tmp_metrics_db):
        """429 error has OpenAI-compatible error body."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(rate_limit_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        data = response.json()

        assert "error" in data
        assert data["error"]["type"] == "rate_limit_error"
        assert "message" in data["error"]

    def test_server_error_body(self, tmp_metrics_db):
        """500 error has OpenAI-compatible error body."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(internal_error_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        data = response.json()

        assert "error" in data
        assert data["error"]["type"] == "server_error"


class TestContentTypeHeaders:
    """Tests for Content-Type headers."""

    def test_success_response_content_type(self, client):
        """Successful response has application/json content type."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert "application/json" in response.headers["content-type"]

    def test_error_response_content_type(self, tmp_metrics_db):
        """Error response has application/json content type."""
        config = ChaosLLMConfig(
            metrics=MetricsConfig(database=tmp_metrics_db),
            latency=LatencyConfig(base_ms=0, jitter_ms=0),
            error_injection=ErrorInjectionConfig(rate_limit_pct=100.0),
        )
        app = create_app(config)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
        assert "application/json" in response.headers["content-type"]
