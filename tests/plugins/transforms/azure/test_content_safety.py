"""Tests for AzureContentSafety transform."""

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from elspeth.plugins.config_base import PluginConfigError

if TYPE_CHECKING:
    pass


def make_mock_context(http_response: dict[str, Any] | None = None) -> Mock:
    """Create mock PluginContext with HTTP client."""
    from elspeth.plugins.context import PluginContext

    ctx = Mock(spec=PluginContext, run_id="test-run")

    if http_response is not None:
        response_mock = Mock()
        response_mock.status_code = 200
        response_mock.json.return_value = http_response
        response_mock.raise_for_status = Mock()
        ctx.http_client.post.return_value = response_mock

    return ctx


class TestAzureContentSafetyConfig:
    """Tests for AzureContentSafetyConfig validation."""

    def test_config_requires_endpoint(self) -> None:
        """Config must specify endpoint."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError) as exc_info:
            AzureContentSafetyConfig.from_dict(
                {
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {
                        "hate": 2,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "endpoint" in str(exc_info.value).lower()

    def test_config_requires_api_key(self) -> None:
        """Config must specify api_key."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError) as exc_info:
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "fields": ["content"],
                    "thresholds": {
                        "hate": 2,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "api_key" in str(exc_info.value).lower()

    def test_config_requires_fields(self) -> None:
        """Config must specify fields."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError) as exc_info:
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "thresholds": {
                        "hate": 2,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    "schema": {"fields": "dynamic"},
                }
            )
        assert "fields" in str(exc_info.value).lower()

    def test_config_requires_all_thresholds(self) -> None:
        """Config must specify all four category thresholds."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError) as exc_info:
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {"hate": 2},  # Missing violence, sexual, self_harm
                    "schema": {"fields": "dynamic"},
                }
            )
        # Should mention one of the missing fields or thresholds
        err_str = str(exc_info.value).lower()
        assert (
            "violence" in err_str
            or "sexual" in err_str
            or "self_harm" in err_str
            or "thresholds" in err_str
        )

    def test_config_validates_threshold_range(self) -> None:
        """Thresholds must be 0-6."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError):
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {
                        "hate": 10,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_config_validates_threshold_range_negative(self) -> None:
        """Thresholds cannot be negative."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError):
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {
                        "hate": -1,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    "schema": {"fields": "dynamic"},
                }
            )

    def test_valid_config(self) -> None:
        """Valid config is accepted."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 4, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.endpoint == "https://test.cognitiveservices.azure.com"
        assert cfg.api_key == "test-key"
        assert cfg.fields == ["content"]
        assert cfg.thresholds.hate == 2
        assert cfg.thresholds.violence == 4
        assert cfg.thresholds.sexual == 2
        assert cfg.thresholds.self_harm == 0

    def test_valid_config_with_single_field(self) -> None:
        """Config accepts single field as string."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": "content",
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.fields == "content"

    def test_valid_config_with_all_fields(self) -> None:
        """Config accepts 'all' to analyze all string fields."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": "all",
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.fields == "all"

    def test_config_requires_schema(self) -> None:
        """Config must specify schema - inherited from TransformDataConfig."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        with pytest.raises(PluginConfigError) as exc_info:
            AzureContentSafetyConfig.from_dict(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {
                        "hate": 2,
                        "violence": 2,
                        "sexual": 2,
                        "self_harm": 0,
                    },
                    # Missing schema
                }
            )
        assert "schema" in str(exc_info.value).lower()

    def test_config_boundary_threshold_zero(self) -> None:
        """Threshold value 0 is valid (minimum)."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 0, "violence": 0, "sexual": 0, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.thresholds.hate == 0

    def test_config_boundary_threshold_six(self) -> None:
        """Threshold value 6 is valid (maximum)."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 6, "violence": 6, "sexual": 6, "self_harm": 6},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.thresholds.hate == 6


class TestAzureContentSafetyTransform:
    """Tests for AzureContentSafety transform."""

    def test_transform_has_required_attributes(self) -> None:
        """Transform has all protocol-required attributes."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        assert transform.name == "azure_content_safety"
        assert transform.determinism.value == "external_call"
        assert transform.plugin_version == "1.0.0"
        assert transform.is_batch_aware is False
        assert transform.creates_tokens is False

    def test_content_below_threshold_passes(self) -> None:
        """Content with severity below thresholds passes through."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )

        row = {"content": "Hello world", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row

    def test_content_exceeding_threshold_returns_error(self) -> None:
        """Content exceeding any threshold returns error."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 4},  # Exceeds threshold of 2
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )

        row = {"content": "Some hateful content", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "content_safety_violation"
        assert result.reason["categories"]["hate"]["exceeded"] is True
        assert result.reason["categories"]["hate"]["severity"] == 4

    def test_api_error_returns_retryable_error(self) -> None:
        """API rate limit errors return retryable error result."""
        import httpx

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context()
        ctx.http_client.post.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=Mock(status_code=429),
        )

        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.retryable is True

    def test_network_error_returns_retryable_error(self) -> None:
        """Network errors return retryable error result."""
        import httpx

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context()
        ctx.http_client.post.side_effect = httpx.RequestError("Connection failed")

        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "network_error"
        assert result.retryable is True

    def test_skips_missing_configured_field(self) -> None:
        """Transform skips fields not present in the row."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content", "optional_field"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )

        # Row is missing "optional_field"
        row = {"content": "safe data", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "success"

    def test_malformed_api_response_returns_error(self) -> None:
        """Malformed API responses return retryable error result.

        Azure API responses are external data (Tier 3: Zero Trust) and may
        return unexpected structures. This should be handled gracefully.
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        # Mock a malformed response (missing categoriesAnalysis)
        ctx = make_mock_context({"unexpectedField": "value"})

        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "network_error"
        assert "malformed" in result.reason["message"].lower()
        assert result.retryable is True

    def test_malformed_category_item_returns_error(self) -> None:
        """Malformed category items in API response return error.

        Each category item must have 'category' and 'severity' fields.
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        # Mock a response with malformed category items (missing severity)
        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate"},  # Missing "severity"
                ]
            }
        )

        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.retryable is True

    def test_missing_categories_treated_as_safe(self) -> None:
        """Missing categories in API response default to severity 0 (safe).

        If Azure returns fewer categories than expected, missing ones
        are treated as having severity 0 to avoid false positives.
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        # Mock a response with only some categories
        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    # Missing Violence, Sexual, SelfHarm
                ]
            }
        )

        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        # Should pass since missing categories default to 0, which is below threshold 2
        assert result.status == "success"

    def test_missing_http_client_raises_error(self) -> None:
        """Missing http_client in context raises RuntimeError."""
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = Mock(spec=PluginContext, run_id="test-run")
        ctx.http_client = None

        row = {"content": "test", "id": 1}

        with pytest.raises(RuntimeError, match="http_client"):
            transform.process(row, ctx)

    def test_close_is_noop(self) -> None:
        """close() method is a no-op but should not raise."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        # Should not raise
        transform.close()

        # Can be called multiple times (idempotent)
        transform.close()

    def test_api_called_with_correct_endpoint_and_headers(self) -> None:
        """API is called with correct endpoint, version, and headers."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com/",  # With trailing slash
                "api_key": "my-secret-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

        ctx = make_mock_context(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )

        row = {"content": "test text", "id": 1}
        transform.process(row, ctx)

        # Verify API was called correctly
        ctx.http_client.post.assert_called_once()
        call_args = ctx.http_client.post.call_args

        # Check URL (trailing slash should be stripped)
        expected_url = (
            "https://test.cognitiveservices.azure.com/contentsafety/text:analyze"
            "?api-version=2024-09-01"
        )
        assert call_args[0][0] == expected_url

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "my-secret-key"
        assert headers["Content-Type"] == "application/json"

        # Check request body
        json_body = call_args[1]["json"]
        assert json_body == {"text": "test text"}
