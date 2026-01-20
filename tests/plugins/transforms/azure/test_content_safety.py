"""Tests for AzureContentSafety transform."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import pytest

from elspeth.plugins.config_base import PluginConfigError

if TYPE_CHECKING:
    from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety


def make_mock_context() -> Mock:
    """Create mock PluginContext."""
    from elspeth.plugins.context import PluginContext

    ctx = Mock(spec=PluginContext, run_id="test-run")
    ctx.state_id = "test-state-id"
    ctx.landscape = Mock()
    return ctx


def make_content_safety_with_mock_response(
    config: dict[str, Any],
    response_data: dict[str, Any],
) -> tuple["AzureContentSafety", MagicMock]:
    """Create Content Safety transform with mocked HTTP client.

    Returns the transform and the mock client for assertions.
    """
    from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

    transform = AzureContentSafety(config)

    # Create mock response
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.json.return_value = response_data
    response_mock.raise_for_status = MagicMock()

    # Create mock client
    mock_client = MagicMock()
    mock_client.post.return_value = response_mock

    # Inject mock client directly (bypassing _get_http_client)
    transform._http_client = mock_client

    return transform, mock_client


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
        assert "violence" in err_str or "sexual" in err_str or "self_harm" in err_str or "thresholds" in err_str

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
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            },
        )

        ctx = make_mock_context()
        row = {"content": "Hello world", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "success"
        assert result.row == row

    def test_content_exceeding_threshold_returns_error(self) -> None:
        """Content exceeding any threshold returns error."""
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 4},  # Exceeds threshold of 2
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            },
        )

        ctx = make_mock_context()
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

        # Create mock client that raises HTTPStatusError
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=Mock(status_code=429),
        )
        transform._http_client = mock_client

        ctx = make_mock_context()
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

        # Create mock client that raises RequestError
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.RequestError("Connection failed")
        transform._http_client = mock_client

        ctx = make_mock_context()
        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "network_error"
        assert result.retryable is True

    def test_skips_missing_configured_field(self) -> None:
        """Transform skips fields not present in the row."""
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content", "optional_field"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            },
        )

        ctx = make_mock_context()
        # Row is missing "optional_field"
        row = {"content": "safe data", "id": 1}
        result = transform.process(row, ctx)

        assert result.status == "success"

    def test_malformed_api_response_returns_error(self) -> None:
        """Malformed API responses return retryable error result.

        Azure API responses are external data (Tier 3: Zero Trust) and may
        return unexpected structures. This should be handled gracefully.
        """
        # Mock a malformed response (missing categoriesAnalysis)
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            },
            {"unexpectedField": "value"},
        )

        ctx = make_mock_context()
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
        # Mock a response with malformed category items (missing severity)
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate"},  # Missing "severity"
                ]
            },
        )

        ctx = make_mock_context()
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
        # Mock a response with only some categories
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    # Missing Violence, Sexual, SelfHarm
                ]
            },
        )

        ctx = make_mock_context()
        row = {"content": "test", "id": 1}
        result = transform.process(row, ctx)

        # Should pass since missing categories default to 0, which is below threshold 2
        assert result.status == "success"

    def test_threshold_zero_with_severity_zero_passes(self) -> None:
        """Threshold=0 with severity=0 should pass (not block safe content).

        Per design doc: threshold=0 means "block severity > 0" not "block all".
        This is the edge case where >= would incorrectly block safe content.
        """
        # Azure returns all safe (severity 0)
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            },
        )

        ctx = make_mock_context()
        row = {"content": "completely safe content", "id": 1}
        result = transform.process(row, ctx)

        # Should pass - severity 0 is NOT > threshold 0
        assert result.status == "success"
        assert result.row == row

    def test_threshold_zero_blocks_severity_one(self) -> None:
        """Threshold=0 should block content with severity >= 1."""
        # Azure returns self_harm severity 1 (above threshold 0)
        transform, _mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 1},
                ]
            },
        )

        ctx = make_mock_context()
        row = {"content": "content with mild self-harm", "id": 1}
        result = transform.process(row, ctx)

        # Should block - severity 1 IS > threshold 0
        assert result.status == "error"
        assert result.reason["reason"] == "content_safety_violation"
        assert result.reason["categories"]["self_harm"]["exceeded"] is True

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
        transform, mock_client = make_content_safety_with_mock_response(
            {
                "endpoint": "https://test.cognitiveservices.azure.com/",  # With trailing slash
                "api_key": "my-secret-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            },
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            },
        )

        ctx = make_mock_context()
        row = {"content": "test text", "id": 1}
        transform.process(row, ctx)

        # Verify API was called correctly
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check URL (trailing slash should be stripped)
        expected_url = "https://test.cognitiveservices.azure.com/contentsafety/text:analyze?api-version=2024-09-01"
        assert call_args[0][0] == expected_url

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Ocp-Apim-Subscription-Key"] == "my-secret-key"
        assert headers["Content-Type"] == "application/json"

        # Check request body
        json_body = call_args[1]["json"]
        assert json_body == {"text": "test text"}


class TestContentSafetyPoolConfig:
    """Tests for Content Safety pool configuration."""

    def test_pool_size_default_is_one(self) -> None:
        """Default pool_size is 1 (sequential)."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
            }
        )

        assert cfg.pool_size == 1

    def test_pool_size_configurable(self) -> None:
        """pool_size can be configured."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 5,
            }
        )

        assert cfg.pool_size == 5

    def test_pool_config_property_returns_none_when_sequential(self) -> None:
        """pool_config returns None when pool_size=1."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert cfg.pool_config is None

    def test_pool_config_property_returns_config_when_pooled(self) -> None:
        """pool_config returns PoolConfig when pool_size>1."""
        from elspeth.plugins.transforms.azure.content_safety import (
            AzureContentSafetyConfig,
        )

        cfg = AzureContentSafetyConfig.from_dict(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert cfg.pool_config is not None
        assert cfg.pool_config.pool_size == 3


class TestContentSafetyPooledExecution:
    """Tests for Content Safety pooled execution."""

    def test_batch_aware_is_true_when_pooled(self) -> None:
        """Transform is batch_aware when pool_size > 1."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        assert transform.is_batch_aware is True

    def test_batch_aware_is_false_when_sequential(self) -> None:
        """Transform is not batch_aware when pool_size=1."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 1,
            }
        )

        assert transform.is_batch_aware is False

    def test_pooled_execution_processes_batch_concurrently(self) -> None:
        """Pooled transform processes batch rows concurrently."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        # Create mock response for safe content
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 0},
                {"category": "Sexual", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
            ]
        }
        response_mock.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = response_mock
        transform._http_client = mock_client

        ctx = make_mock_context()
        rows = [
            {"content": "Hello", "id": 1},
            {"content": "World", "id": 2},
            {"content": "Test", "id": 3},
        ]
        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

    def test_pooled_execution_handles_threshold_violations(self) -> None:
        """Pooled execution correctly handles content violations."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        def make_response(*args, **kwargs):
            """Return response based on content, not call order (avoids race condition)."""
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status = MagicMock()

            # Check the actual content being analyzed
            request_body = kwargs.get("json", {})
            text = request_body.get("text", "")

            if "violent" in text.lower():  # Match on content, not call order
                response.json.return_value = {
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": 0},
                        {"category": "Violence", "severity": 5},  # Exceeds threshold 2
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                    ]
                }
            else:
                response.json.return_value = {
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": 0},
                        {"category": "Violence", "severity": 0},
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                    ]
                }
            return response

        mock_client = MagicMock()
        mock_client.post.side_effect = make_response
        transform._http_client = mock_client

        ctx = make_mock_context()
        rows = [
            {"content": "Safe content 1", "id": 1},
            {"content": "Violent content here", "id": 2},  # This one triggers violation
            {"content": "Safe content 3", "id": 3},
        ]
        result = transform.process(rows, ctx)

        # Result should be success since not ALL rows failed
        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 3

        # Row 1: success (no error field)
        assert "_content_safety_error" not in result.rows[0]
        # Row 2: has content safety error
        assert "_content_safety_error" in result.rows[1]
        assert result.rows[1]["_content_safety_error"]["reason"] == "content_safety_violation"
        # Row 3: success (no error field)
        assert "_content_safety_error" not in result.rows[2]

    def test_pooled_rate_limit_triggers_capacity_error(self) -> None:
        """Rate limit (429) triggers CapacityError for AIMD retry."""
        import httpx

        from elspeth.plugins.pooling import CapacityError
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"fields": "dynamic"},
                "pool_size": 3,
            }
        )

        # Create mock client that returns 429
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=Mock(status_code=429),
        )
        transform._http_client = mock_client

        row = {"content": "test", "id": 1}

        # _process_single_with_state should raise CapacityError for 429
        with pytest.raises(CapacityError) as exc_info:
            transform._process_single_with_state(row, "test-state-id")

        assert exc_info.value.status_code == 429
