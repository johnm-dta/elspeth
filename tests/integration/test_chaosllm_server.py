# tests/integration/test_chaosllm_server.py
"""Integration tests for ChaosLLM server using the pytest fixture.

These tests exercise the ChaosLLM server through the chaosllm_server fixture,
verifying end-to-end behavior of error injection, response modes, admin
endpoints, metrics recording, and burst patterns.
"""

import json
import sqlite3

import pytest


class TestBasicFunctionality:
    """Basic ChaosLLM server functionality tests."""

    def test_health_check_responds(self, chaosllm_server):
        """Server responds to health check with healthy status."""
        response = chaosllm_server.client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "run_id" in data
        assert "started_utc" in data
        assert "in_burst" in data

    def test_openai_endpoint_valid_response_format(self, chaosllm_server):
        """OpenAI endpoint returns valid response format."""
        response = chaosllm_server.post_completion(
            messages=[{"role": "user", "content": "Test message"}],
            model="gpt-4",
        )
        assert response.status_code == 200

        data = response.json()

        # Verify OpenAI-compatible structure
        assert "id" in data
        assert data["id"].startswith("fake-")
        assert data["object"] == "chat.completion"
        assert "created" in data
        assert isinstance(data["created"], int)
        assert data["model"] == "gpt-4"

        # Verify choices structure
        assert "choices" in data
        assert len(data["choices"]) == 1
        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert "content" in choice["message"]
        assert isinstance(choice["message"]["content"], str)
        assert choice["finish_reason"] == "stop"

        # Verify usage structure
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

    def test_azure_endpoint_valid_response_format(self, chaosllm_server):
        """Azure endpoint returns valid response format."""
        response = chaosllm_server.post_azure_completion(
            deployment="test-deployment",
            messages=[{"role": "user", "content": "Test message"}],
        )
        assert response.status_code == 200

        data = response.json()

        # Verify OpenAI-compatible structure
        assert "id" in data
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert "usage" in data

    def test_azure_endpoint_with_api_version(self, chaosllm_server):
        """Azure endpoint accepts api-version query parameter."""
        response = chaosllm_server.client.post(
            "/openai/deployments/test-deployment/chat/completions?api-version=2024-08-01-preview",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200


class TestErrorInjection:
    """Error injection tests via the fixture."""

    @pytest.mark.chaosllm(rate_limit_pct=100.0)
    def test_rate_limit_error_429_injected(self, chaosllm_server):
        """Rate limit error (429) injected at configured percentage."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

        # Verify Retry-After header is present
        assert "Retry-After" in response.headers
        retry_after = int(response.headers["Retry-After"])
        assert 1 <= retry_after <= 5  # Default retry_after_sec range

        # Verify error body format
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "rate_limit_error"
        assert "message" in data["error"]
        assert data["error"]["code"] == "rate_limit"

    @pytest.mark.chaosllm(capacity_529_pct=100.0)
    def test_capacity_error_529_injected(self, chaosllm_server):
        """Capacity error (529) injected at configured percentage."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 529

        # Verify Retry-After header is present
        assert "Retry-After" in response.headers

        # Verify error body format
        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "capacity_error"
        assert data["error"]["code"] == "capacity_529"

    @pytest.mark.chaosllm(service_unavailable_pct=100.0)
    def test_service_unavailable_503_injected(self, chaosllm_server):
        """Service unavailable error (503) injected."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 503

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"

    @pytest.mark.chaosllm(bad_gateway_pct=100.0)
    def test_bad_gateway_502_injected(self, chaosllm_server):
        """Bad gateway error (502) injected."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 502

    @pytest.mark.chaosllm(gateway_timeout_pct=100.0)
    def test_gateway_timeout_504_injected(self, chaosllm_server):
        """Gateway timeout error (504) injected."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 504

    @pytest.mark.chaosllm(internal_error_pct=100.0)
    def test_internal_error_500_injected(self, chaosllm_server):
        """Internal server error (500) injected."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert data["error"]["type"] == "server_error"
        assert data["error"]["code"] == "internal_error"


class TestMalformedResponses:
    """Tests for malformed response injection."""

    @pytest.mark.chaosllm(invalid_json_pct=100.0)
    def test_invalid_json_response_configured(self, chaosllm_server):
        """Invalid JSON response when configured at 100%."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        # Should NOT be parseable as JSON
        with pytest.raises(json.JSONDecodeError):
            response.json()

        # Verify content is malformed
        assert b"malformed" in response.content or b"unclosed" in response.content

    @pytest.mark.chaosllm(truncated_pct=100.0)
    def test_truncated_response_configured(self, chaosllm_server):
        """Truncated response when configured at 100%."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        # Should NOT be parseable as JSON due to truncation
        with pytest.raises(json.JSONDecodeError):
            response.json()

    @pytest.mark.chaosllm(empty_body_pct=100.0)
    def test_empty_body_response_configured(self, chaosllm_server):
        """Empty body response when configured at 100%."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200
        assert response.content == b""

    @pytest.mark.chaosllm(missing_fields_pct=100.0)
    def test_missing_fields_response_configured(self, chaosllm_server):
        """Response missing required fields when configured at 100%."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        data = response.json()
        # Should be missing choices and/or usage
        assert "choices" not in data or "usage" not in data

    @pytest.mark.chaosllm(wrong_content_type_pct=100.0)
    def test_wrong_content_type_configured(self, chaosllm_server):
        """Wrong Content-Type header when configured at 100%."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestResponseModes:
    """Response mode tests via the fixture."""

    def test_random_mode_generates_text(self, chaosllm_server):
        """Random mode (default) generates readable text content."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Random mode should generate non-empty content
        assert isinstance(content, str)
        assert len(content) > 0
        # Content should end with a period (random generator adds punctuation)
        assert content.endswith(".")

    @pytest.mark.chaosllm(mode="template")
    def test_template_mode_evaluates_jinja2(self, chaosllm_server):
        """Template mode evaluates Jinja2 templates."""
        # Use X-Fake-Template header to provide custom template
        response = chaosllm_server.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={
                "X-Fake-Response-Mode": "template",
                "X-Fake-Template": "Model: {{ model }}, Messages: {{ messages|length }}",
            },
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Model: gpt-4-turbo, Messages: 1"

    @pytest.mark.chaosllm(mode="echo")
    def test_echo_mode_reflects_input(self, chaosllm_server):
        """Echo mode reflects the input message."""
        test_message = "This is a test message for echo mode"
        response = chaosllm_server.post_completion(messages=[{"role": "user", "content": test_message}])
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == f"Echo: {test_message}"

    @pytest.mark.chaosllm(mode="echo")
    def test_echo_mode_handles_empty_messages(self, chaosllm_server):
        """Echo mode handles empty messages gracefully."""
        response = chaosllm_server.post_completion(messages=[])
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert "no messages" in content.lower()

    def test_per_request_mode_override(self, chaosllm_server):
        """X-Fake-Response-Mode header overrides configured mode."""
        # Default is random mode, override with echo
        response = chaosllm_server.client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Override test"}],
            },
            headers={"X-Fake-Response-Mode": "echo"},
        )
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Echo: Override test"


class TestAdminEndpoints:
    """Admin endpoint tests."""

    def test_admin_stats_returns_valid_json(self, chaosllm_server):
        """GET /admin/stats returns valid JSON structure."""
        response = chaosllm_server.client.get("/admin/stats")
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "started_utc" in data
        assert "total_requests" in data
        assert "requests_by_outcome" in data
        assert "requests_by_status_code" in data
        assert "latency_stats" in data
        assert "error_rate" in data

        # Verify latency_stats structure
        latency = data["latency_stats"]
        assert "avg_ms" in latency
        assert "p50_ms" in latency
        assert "p95_ms" in latency
        assert "p99_ms" in latency
        assert "max_ms" in latency

    def test_admin_config_get_returns_config(self, chaosllm_server):
        """GET /admin/config returns current configuration."""
        response = chaosllm_server.client.get("/admin/config")
        assert response.status_code == 200

        data = response.json()
        assert "error_injection" in data
        assert "response" in data
        assert "latency" in data

        # Verify error_injection structure
        error = data["error_injection"]
        assert "rate_limit_pct" in error
        assert "capacity_529_pct" in error

    def test_admin_config_post_updates_config(self, chaosllm_server):
        """POST /admin/config updates server configuration."""
        # Verify initial state
        response = chaosllm_server.client.get("/admin/config")
        initial = response.json()
        assert initial["error_injection"]["rate_limit_pct"] == 0.0

        # Update config
        response = chaosllm_server.client.post(
            "/admin/config",
            json={"error_injection": {"rate_limit_pct": 75.0}},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "updated"
        assert data["config"]["error_injection"]["rate_limit_pct"] == 75.0

        # Verify update persisted
        response = chaosllm_server.client.get("/admin/config")
        assert response.json()["error_injection"]["rate_limit_pct"] == 75.0

    def test_admin_config_post_partial_update(self, chaosllm_server):
        """POST /admin/config only updates specified fields."""
        # Get initial config
        response = chaosllm_server.client.get("/admin/config")
        initial = response.json()
        original_mode = initial["response"]["mode"]

        # Update only error injection
        chaosllm_server.client.post(
            "/admin/config",
            json={"error_injection": {"internal_error_pct": 10.0}},
        )

        # Verify response mode unchanged, only error_injection updated
        response = chaosllm_server.client.get("/admin/config")
        updated = response.json()
        assert updated["response"]["mode"] == original_mode
        assert updated["error_injection"]["internal_error_pct"] == 10.0

    def test_admin_export_returns_raw_data(self, chaosllm_server):
        """GET /admin/export returns raw requests and timeseries."""
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()

        response = chaosllm_server.client.get("/admin/export")
        assert response.status_code == 200

        data = response.json()
        assert "run_id" in data
        assert "started_utc" in data
        assert "requests" in data
        assert "timeseries" in data
        assert isinstance(data["requests"], list)
        assert isinstance(data["timeseries"], list)
        assert "config" in data

    def test_admin_reset_clears_metrics(self, chaosllm_server):
        """POST /admin/reset clears metrics and generates new run_id."""
        # Make some requests
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()

        # Verify requests recorded
        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 2
        original_run_id = stats["run_id"]

        # Reset
        response = chaosllm_server.client.post("/admin/reset")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "reset"
        assert "new_run_id" in data
        assert data["new_run_id"] != original_run_id

        # Verify metrics cleared
        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 0
        assert stats["run_id"] == data["new_run_id"]


class TestMetricsRecording:
    """Tests for metrics recording behavior."""

    def test_requests_recorded_to_database(self, chaosllm_server):
        """Requests are recorded to the metrics SQLite database."""
        # Make a request
        chaosllm_server.post_completion()

        # Verify in stats
        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 1
        assert stats["requests_by_outcome"]["success"] == 1

        # Verify in database directly
        db_path = chaosllm_server.metrics_db
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("SELECT COUNT(*) FROM requests")
            count = cursor.fetchone()[0]
            assert count == 1

            cursor = conn.execute("SELECT * FROM requests LIMIT 1")
            row = dict(cursor.fetchone())
            assert row["outcome"] == "success"
            assert row["status_code"] == 200
            assert row["endpoint"] == "/v1/chat/completions"

    def test_stats_reflect_actual_requests(self, chaosllm_server):
        """Stats accurately reflect actual requests made."""
        # Make multiple requests
        for _ in range(5):
            chaosllm_server.post_completion()

        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 5
        assert stats["requests_by_outcome"]["success"] == 5
        assert stats["requests_by_status_code"][200] == 5

    @pytest.mark.chaosllm(rate_limit_pct=100.0)
    def test_error_requests_recorded(self, chaosllm_server):
        """Error responses are recorded with correct outcome."""
        chaosllm_server.post_completion()

        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 1
        assert stats["requests_by_outcome"]["error_injected"] == 1
        assert stats["requests_by_status_code"][429] == 1

    def test_azure_requests_recorded_with_deployment(self, chaosllm_server):
        """Azure requests record the deployment name."""
        chaosllm_server.post_azure_completion("my-test-deployment")

        db_path = chaosllm_server.metrics_db
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("SELECT deployment FROM requests LIMIT 1")
            row = cursor.fetchone()
            assert row["deployment"] == "my-test-deployment"

    def test_latency_stats_populated(self, chaosllm_server):
        """Latency statistics are populated after requests."""
        # Make several requests
        for _ in range(10):
            chaosllm_server.post_completion()

        stats = chaosllm_server.get_stats()
        latency = stats["latency_stats"]

        # All latency values should be populated
        assert latency["avg_ms"] is not None
        assert latency["p50_ms"] is not None
        assert latency["p95_ms"] is not None
        assert latency["p99_ms"] is not None
        assert latency["max_ms"] is not None

        # Values should be non-negative
        assert latency["avg_ms"] >= 0
        assert latency["max_ms"] >= latency["avg_ms"]

    @pytest.mark.chaosllm(invalid_json_pct=100.0)
    def test_malformed_response_recorded_as_error_malformed(self, chaosllm_server):
        """Malformed responses are recorded with error_malformed outcome."""
        chaosllm_server.post_completion()

        stats = chaosllm_server.get_stats()
        assert stats["requests_by_outcome"]["error_malformed"] == 1


class TestBurstPatterns:
    """Tests for burst pattern behavior.

    Note: These tests verify burst configuration through the error injector's
    is_in_burst() method, since actually testing time-based burst windows
    would require long waits.
    """

    def test_burst_not_active_by_default(self, chaosllm_server):
        """Burst mode is not active by default."""
        response = chaosllm_server.client.get("/health")
        data = response.json()
        assert data["in_burst"] is False

    @pytest.mark.chaosllm(rate_limit_pct=0.0)
    def test_no_errors_when_not_in_burst(self, chaosllm_server):
        """No errors occur when burst is disabled and rate_limit_pct is 0."""
        # Make multiple requests, all should succeed
        for _ in range(10):
            response = chaosllm_server.post_completion()
            assert response.status_code == 200


class TestRuntimeConfigurationUpdates:
    """Tests for runtime configuration updates via the fixture."""

    def test_update_config_changes_error_rate(self, chaosllm_server):
        """update_config can change error rate at runtime."""
        # Initial request should succeed
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        # Update to 100% rate limiting
        chaosllm_server.update_config(rate_limit_pct=100.0)

        # Now request should fail with 429
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

    def test_update_config_changes_response_mode(self, chaosllm_server):
        """update_config can change response mode at runtime."""
        # Update to echo mode
        chaosllm_server.update_config(mode="echo")

        response = chaosllm_server.post_completion(messages=[{"role": "user", "content": "Runtime echo test"}])
        assert response.status_code == 200

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "Echo: Runtime echo test"

    def test_reset_clears_metrics_and_returns_new_run_id(self, chaosllm_server):
        """reset() clears metrics and returns new run_id."""
        # Make requests
        chaosllm_server.post_completion()
        original_run_id = chaosllm_server.run_id
        assert chaosllm_server.get_stats()["total_requests"] == 1

        # Reset
        new_run_id = chaosllm_server.reset()

        assert new_run_id != original_run_id
        assert chaosllm_server.run_id == new_run_id
        assert chaosllm_server.get_stats()["total_requests"] == 0


class TestFixtureIsolation:
    """Tests verifying fixture isolation between tests."""

    def test_isolation_part_1_make_requests(self, chaosllm_server):
        """Make requests in first test."""
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()
        assert chaosllm_server.get_stats()["total_requests"] == 2

    def test_isolation_part_2_verify_clean_state(self, chaosllm_server):
        """Verify second test has clean state."""
        # Should be 0, not 2 from previous test
        assert chaosllm_server.get_stats()["total_requests"] == 0

    @pytest.mark.chaosllm(rate_limit_pct=100.0)
    def test_isolation_part_3_with_marker(self, chaosllm_server):
        """Test with marker affecting config."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

    def test_isolation_part_4_marker_not_inherited(self, chaosllm_server):
        """Verify marker config not inherited from previous test."""
        response = chaosllm_server.post_completion()
        # Should be 200, not 429 from previous test's marker
        assert response.status_code == 200


class TestWaitForRequests:
    """Tests for the wait_for_requests helper."""

    def test_wait_for_requests_returns_true_when_count_reached(self, chaosllm_server):
        """wait_for_requests returns True when count is reached."""
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()

        result = chaosllm_server.wait_for_requests(3, timeout=1.0)
        assert result is True

    def test_wait_for_requests_returns_false_on_timeout(self, chaosllm_server):
        """wait_for_requests returns False when timeout expires."""
        result = chaosllm_server.wait_for_requests(100, timeout=0.1)
        assert result is False


class TestMultipleErrorTypes:
    """Tests for multiple error type interactions."""

    def test_runtime_switch_between_error_types(self, chaosllm_server):
        """Can switch between different error types at runtime."""
        # Start with rate limiting
        chaosllm_server.update_config(rate_limit_pct=100.0)
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

        # Switch to internal error
        chaosllm_server.update_config(rate_limit_pct=0.0, internal_error_pct=100.0)
        response = chaosllm_server.post_completion()
        assert response.status_code == 500

        # Switch to success
        chaosllm_server.update_config(internal_error_pct=0.0)
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

    def test_runtime_switch_to_malformed(self, chaosllm_server):
        """Can switch to malformed response type at runtime."""
        chaosllm_server.update_config(invalid_json_pct=100.0)
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        with pytest.raises(json.JSONDecodeError):
            response.json()
