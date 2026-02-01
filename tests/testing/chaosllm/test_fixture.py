# tests/testing/chaosllm/test_fixture.py
"""Tests for the ChaosLLM pytest fixture."""

import pytest


class TestChaosLLMFixtureBasics:
    """Basic fixture functionality tests."""

    def test_fixture_provides_client(self, chaosllm_server):
        """Fixture provides a working test client."""
        response = chaosllm_server.client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_fixture_provides_server(self, chaosllm_server):
        """Fixture provides server access."""
        assert chaosllm_server.server is not None
        assert chaosllm_server.server.run_id is not None

    def test_url_property(self, chaosllm_server):
        """URL property returns testserver URL."""
        assert chaosllm_server.url == "http://testserver"

    def test_admin_url_property(self, chaosllm_server):
        """Admin URL property returns correct path."""
        assert chaosllm_server.admin_url == "http://testserver/admin"

    def test_metrics_db_property(self, chaosllm_server):
        """Metrics DB property returns path."""
        assert chaosllm_server.metrics_db.exists()

    def test_run_id_property(self, chaosllm_server):
        """Run ID property returns server run_id."""
        assert chaosllm_server.run_id == chaosllm_server.server.run_id


class TestChaosLLMFixtureConvenienceMethods:
    """Tests for convenience methods on the fixture."""

    def test_post_completion(self, chaosllm_server):
        """post_completion sends a chat completion request."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert data["model"] == "gpt-4"

    def test_post_completion_with_messages(self, chaosllm_server):
        """post_completion accepts custom messages."""
        messages = [
            {"role": "system", "content": "You are a test"},
            {"role": "user", "content": "Hello"},
        ]
        response = chaosllm_server.post_completion(messages=messages)
        assert response.status_code == 200

    def test_post_completion_with_model(self, chaosllm_server):
        """post_completion accepts custom model."""
        response = chaosllm_server.post_completion(model="gpt-3.5-turbo")
        assert response.status_code == 200
        assert response.json()["model"] == "gpt-3.5-turbo"

    def test_post_azure_completion(self, chaosllm_server):
        """post_azure_completion sends Azure format request."""
        response = chaosllm_server.post_azure_completion("my-deployment")
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data

    def test_get_stats(self, chaosllm_server):
        """get_stats returns metrics."""
        stats = chaosllm_server.get_stats()
        assert "run_id" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] == 0

    def test_get_stats_after_request(self, chaosllm_server):
        """get_stats reflects requests made."""
        chaosllm_server.post_completion()
        stats = chaosllm_server.get_stats()
        assert stats["total_requests"] == 1

    def test_reset(self, chaosllm_server):
        """reset clears metrics and returns new run_id."""
        chaosllm_server.post_completion()
        old_run_id = chaosllm_server.run_id

        new_run_id = chaosllm_server.reset()

        assert new_run_id != old_run_id
        assert chaosllm_server.run_id == new_run_id
        assert chaosllm_server.get_stats()["total_requests"] == 0

    def test_wait_for_requests(self, chaosllm_server):
        """wait_for_requests returns True when count reached."""
        chaosllm_server.post_completion()
        chaosllm_server.post_completion()

        result = chaosllm_server.wait_for_requests(2, timeout=1.0)
        assert result is True

    def test_wait_for_requests_timeout(self, chaosllm_server):
        """wait_for_requests returns False on timeout."""
        result = chaosllm_server.wait_for_requests(100, timeout=0.1)
        assert result is False


class TestChaosLLMFixtureUpdateConfig:
    """Tests for runtime configuration updates."""

    def test_update_config_error_rate(self, chaosllm_server):
        """update_config can change error rates."""
        # Verify default has no rate limiting
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

        # Enable 100% rate limiting
        chaosllm_server.update_config(rate_limit_pct=100.0)

        # Now should get 429
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

    def test_update_config_multiple_fields(self, chaosllm_server):
        """update_config can change multiple fields."""
        chaosllm_server.update_config(
            rate_limit_pct=50.0,
            base_ms=100,
        )

        # Verify configuration was applied
        # We can't easily test percentages, but we can verify no exception

    def test_update_config_response_mode(self, chaosllm_server):
        """update_config can change response mode."""
        chaosllm_server.update_config(mode="echo")

        response = chaosllm_server.post_completion(messages=[{"role": "user", "content": "Test message"}])
        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content == "Echo: Test message"


class TestChaosLLMMarkerIntegration:
    """Tests for the @pytest.mark.chaosllm marker."""

    def test_default_config_no_marker(self, chaosllm_server):
        """Without marker, uses default config (no errors)."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200

    @pytest.mark.chaosllm(rate_limit_pct=100.0)
    def test_marker_error_rate(self, chaosllm_server):
        """Marker can set error rate."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

    @pytest.mark.chaosllm(internal_error_pct=100.0)
    def test_marker_internal_error(self, chaosllm_server):
        """Marker can set different error types."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 500

    @pytest.mark.chaosllm(mode="echo")
    def test_marker_response_mode(self, chaosllm_server):
        """Marker can set response mode."""
        response = chaosllm_server.post_completion(messages=[{"role": "user", "content": "Hello marker"}])
        assert response.status_code == 200
        content = response.json()["choices"][0]["message"]["content"]
        assert content == "Echo: Hello marker"


class TestChaosLLMFixtureIsolation:
    """Tests for fixture isolation between tests."""

    def test_first_request(self, chaosllm_server):
        """First test makes a request."""
        chaosllm_server.post_completion()
        assert chaosllm_server.get_stats()["total_requests"] == 1

    def test_second_request_isolated(self, chaosllm_server):
        """Second test should have fresh state."""
        # This should be 0, not 1, proving isolation
        assert chaosllm_server.get_stats()["total_requests"] == 0

    @pytest.mark.chaosllm(rate_limit_pct=100.0)
    def test_marker_in_first(self, chaosllm_server):
        """Test with marker affecting config."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 429

    def test_no_marker_after_marker(self, chaosllm_server):
        """Test without marker should not inherit previous config."""
        response = chaosllm_server.post_completion()
        # Should be 200, not 429 from previous test's marker
        assert response.status_code == 200


class TestChaosLLMErrorTypes:
    """Test various error injection types via the fixture."""

    @pytest.mark.chaosllm(capacity_529_pct=100.0)
    def test_capacity_error(self, chaosllm_server):
        """529 capacity error injection."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 529

    @pytest.mark.chaosllm(service_unavailable_pct=100.0)
    def test_service_unavailable(self, chaosllm_server):
        """503 service unavailable injection."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 503

    @pytest.mark.chaosllm(invalid_json_pct=100.0)
    def test_malformed_response(self, chaosllm_server):
        """Invalid JSON response injection."""
        import json

        response = chaosllm_server.post_completion()
        assert response.status_code == 200
        with pytest.raises(json.JSONDecodeError):
            response.json()

    @pytest.mark.chaosllm(empty_body_pct=100.0)
    def test_empty_body(self, chaosllm_server):
        """Empty body response injection."""
        response = chaosllm_server.post_completion()
        assert response.status_code == 200
        assert response.content == b""
