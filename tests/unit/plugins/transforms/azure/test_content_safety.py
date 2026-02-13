"""Tests for AzureContentSafety transform with BatchTransformMixin."""

import itertools
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.batching.ports import CollectorOutputPort
from elspeth.plugins.config_base import PluginConfigError
from elspeth.testing import make_pipeline_row

if TYPE_CHECKING:
    pass


def make_token(row_id: str = "row-1", token_id: str | None = None) -> TokenInfo:
    """Create a TokenInfo for testing."""
    return TokenInfo(
        row_id=row_id,
        token_id=token_id or f"token-{row_id}",
        row_data=make_pipeline_row({}),
    )


def make_mock_context(
    state_id: str = "test-state-001",
    token: TokenInfo | None = None,
) -> Mock:
    """Create mock PluginContext for testing with recorder.

    The context includes a mock landscape/recorder with allocate_call_index
    configured to return sequential indices.
    """
    counter = itertools.count()
    ctx = Mock(spec=PluginContext)
    ctx.run_id = "test-run"
    ctx.state_id = state_id
    ctx.landscape = Mock()
    ctx.landscape.record_call = Mock()
    ctx.landscape.allocate_call_index = Mock(side_effect=lambda _: next(counter))
    ctx.token = token if token is not None else make_token("row-1")
    return ctx


def _create_mock_http_response(response_data: dict[str, Any]) -> Mock:
    """Create a mock HTTP response with the given JSON data."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = response_data
    response.raise_for_status = Mock()
    response.headers = {"content-type": "application/json"}
    response.content = b"{}"
    response.text = "{}"
    return response


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
                    "schema": {"mode": "observed"},
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
                    "schema": {"mode": "observed"},
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
                    "schema": {"mode": "observed"},
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
                    "schema": {"mode": "observed"},
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
                    "schema": {"mode": "observed"},
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
                    "schema": {"mode": "observed"},
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
                "schema": {"mode": "observed"},
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
                "schema": {"mode": "observed"},
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
                "schema": {"mode": "observed"},
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
                "schema": {"mode": "observed"},
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
                "schema": {"mode": "observed"},
            }
        )

        assert cfg.thresholds.hate == 6


class TestAzureContentSafetyTransform:
    """Tests for AzureContentSafety transform attributes."""

    def test_transform_has_required_attributes(self) -> None:
        """Transform has all protocol-required attributes."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        assert transform.name == "azure_content_safety"
        assert transform.determinism.value == "external_call"
        assert transform.plugin_version == "1.0.0"
        assert transform.creates_tokens is False

    def test_process_raises_not_implemented(self) -> None:
        """process() raises NotImplementedError directing to accept()."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        ctx = make_mock_context()
        row = make_pipeline_row({"content": "test"})

        with pytest.raises(NotImplementedError, match="accept"):
            transform.process(row, ctx)


class TestContentSafetyBatchProcessing:
    """Tests for Content Safety with BatchTransformMixin."""

    @pytest.fixture(autouse=True)
    def mock_httpx_client(self):
        """Patch httpx.Client to prevent real HTTP calls."""
        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            yield mock_instance

    def test_connect_output_required_before_accept(self) -> None:
        """accept() raises RuntimeError if connect_output() not called."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        ctx = make_mock_context()
        row = make_pipeline_row({"content": "test"})

        with pytest.raises(RuntimeError, match="connect_output"):
            transform.accept(row, ctx)

    def test_connect_output_cannot_be_called_twice(self) -> None:
        """connect_output() raises if called more than once."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            with pytest.raises(RuntimeError, match="already called"):
                transform.connect_output(collector, max_pending=10)
        finally:
            transform.close()

    def test_content_below_threshold_passes(self, mock_httpx_client: MagicMock) -> None:
        """Content with severity below thresholds passes through."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "Hello world", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert result.row.to_dict() == row_data
        finally:
            transform.close()

    def test_content_exceeding_threshold_returns_error(self, mock_httpx_client: MagicMock) -> None:
        """Content exceeding any threshold returns error."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 4},  # Exceeds threshold of 2
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "Some hateful content", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "content_safety_violation"
            categories = result.reason["categories"]
            assert isinstance(categories, dict)  # dict variant, not list
            assert categories["hate"]["exceeded"] is True
            assert categories["hate"]["severity"] == 4
        finally:
            transform.close()

    def test_missing_configured_field_fails_closed(self, mock_httpx_client: MagicMock) -> None:
        """Missing value in explicitly-configured field fails CLOSED."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["optional_field", "content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            # Row is missing explicitly-configured "optional_field"
            row_data = {"content": "safe data", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "missing_field"
            assert result.reason["field"] == "optional_field"
            assert result.retryable is False
            assert mock_httpx_client.post.call_count == 0
        finally:
            transform.close()

    def test_non_string_configured_field_fails_closed(self, mock_httpx_client: MagicMock) -> None:
        """Non-string value in explicitly-configured field fails CLOSED.

        Security transform cannot analyze non-string content. Silently skipping
        would be a fail-OPEN vulnerability — the field goes unscanned.
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        # Set up mock response for the string field that gets processed
        # before the non-string field is encountered
        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content", "score"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            # score is an int — configured field with non-string value
            row_data = {"content": "safe text", "score": 42, "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "non_string_field"
            assert result.reason["field"] == "score"
            assert result.reason["actual_type"] == "int"
            assert result.retryable is False
        finally:
            transform.close()

    def test_all_mode_ignores_non_string_fields(self, mock_httpx_client: MagicMock) -> None:
        """When fields='all', non-string fields are correctly ignored.

        In 'all' mode, _get_fields_to_scan pre-filters to string-valued fields,
        so non-string fields never reach the type check. This is by design.
        """
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": "all",
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            # Mix of string and non-string fields
            row_data = {"content": "safe", "count": 42, "flag": True, "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            # Only "content" is a string — one API call
            assert mock_httpx_client.post.call_count == 1
        finally:
            transform.close()

    def test_malformed_api_response_returns_error(self, mock_httpx_client: MagicMock) -> None:
        """Malformed API responses return error result."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        # Missing categoriesAnalysis
        mock_response = _create_mock_http_response({"unexpectedField": "value"})
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "test", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "api_error"
            assert result.reason["error_type"] == "malformed_response"
            assert "malformed" in result.reason["message"].lower()
            assert result.retryable is False
        finally:
            transform.close()

    def test_malformed_category_item_returns_error(self, mock_httpx_client: MagicMock) -> None:
        """Malformed category items in API response return error."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        # Missing severity in category item
        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate"},  # Missing "severity"
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "test", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "api_error"
            assert result.reason["error_type"] == "malformed_response"
            assert result.retryable is False
        finally:
            transform.close()

    def test_missing_categories_rejected_fail_closed(self, mock_httpx_client: MagicMock) -> None:
        """Missing expected categories in API response are rejected (fail-closed)."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        # Only returns Hate category — missing Violence, Sexual, SelfHarm
        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    # Missing Violence, Sexual, SelfHarm
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "test", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Fail CLOSED: missing categories are rejected, not silently treated as safe
            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["error_type"] == "malformed_response"
            assert "missing expected categories" in result.reason["message"].lower()
            assert result.retryable is False
        finally:
            transform.close()

    def test_threshold_zero_with_severity_zero_passes(self, mock_httpx_client: MagicMock) -> None:
        """Threshold=0 with severity=0 should pass (not block safe content)."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "completely safe content", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Should pass - severity 0 is NOT > threshold 0
            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
            assert result.row is not None
            assert result.row.to_dict() == row_data
        finally:
            transform.close()

    def test_threshold_zero_blocks_severity_one(self, mock_httpx_client: MagicMock) -> None:
        """Threshold=0 should block content with severity >= 1."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 1},  # Above threshold 0
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "content with mild self-harm", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Should block - severity 1 IS > threshold 0
            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "error"
            assert result.reason is not None
            assert result.reason["reason"] == "content_safety_violation"
            categories = result.reason["categories"]
            assert isinstance(categories, dict)  # dict variant, not list
            assert categories["self_harm"]["exceeded"] is True
        finally:
            transform.close()

    def test_api_called_with_correct_endpoint_and_headers(self, mock_httpx_client: MagicMock) -> None:
        """API is called with correct endpoint, version, and headers."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com/",  # With trailing slash
                "api_key": "my-secret-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row_data = {"content": "test text", "id": 1}
            row = make_pipeline_row(row_data)
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            # Verify API was called correctly
            mock_httpx_client.post.assert_called_once()
            call_args = mock_httpx_client.post.call_args

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
        finally:
            transform.close()

    def test_multiple_rows_fifo_order(self, mock_httpx_client: MagicMock) -> None:
        """Multiple rows are processed and returned in FIFO order."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx_init = make_mock_context()
        transform.on_start(ctx_init)
        transform.connect_output(collector, max_pending=10)

        try:
            # Submit multiple rows with different markers
            rows_data = [
                {"content": "row 1", "marker": "first"},
                {"content": "row 2", "marker": "second"},
                {"content": "row 3", "marker": "third"},
            ]

            for i, row_data in enumerate(rows_data):
                token = make_token(f"row-{i}", f"token-{i}")
                ctx = make_mock_context(state_id=f"state-{i}", token=token)
                row = make_pipeline_row(row_data)
                transform.accept(row, ctx)

            transform.flush_batch_processing(timeout=10.0)

            # Results should be in FIFO order
            assert len(collector.results) == 3
            for i, (_, result, _) in enumerate(collector.results):
                assert isinstance(result, TransformResult)
                assert result.status == "success"
                assert result.row is not None
                assert result.row["marker"] == rows_data[i]["marker"]
        finally:
            transform.close()


class TestContentSafetyFailsClosed:
    """P0-03: Security transforms must fail CLOSED, not open.

    When Azure returns an unknown category (e.g., after a taxonomy update),
    the transform must reject the content — not silently pass it through.
    """

    @pytest.fixture(autouse=True)
    def mock_httpx_client(self):
        """Patch httpx.Client to prevent real HTTP calls."""
        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            yield mock_instance

    def test_unknown_category_fails_closed(self, mock_httpx_client: MagicMock) -> None:
        """Unknown category in API response must cause error, not pass through."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                    {"category": "FutureCategory", "severity": 4},  # Unknown category
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row = make_pipeline_row({"content": "test content", "id": 1})
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            # Must fail closed — unknown category = reject content
            assert result.status == "error", "Unknown category must fail CLOSED (error), not pass through as success"
            assert result.reason is not None
            assert "unknown_category" in result.reason["reason"]
        finally:
            transform.close()

    def test_known_categories_still_work_with_explicit_mapping(self, mock_httpx_client: MagicMock) -> None:
        """Known Azure categories map correctly with explicit lookup."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_response = _create_mock_http_response(
            {
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": 0},
                    {"category": "Violence", "severity": 0},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                ]
            }
        )
        mock_httpx_client.post.return_value = mock_response

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )

        collector = CollectorOutputPort()
        ctx = make_mock_context()
        transform.on_start(ctx)
        transform.connect_output(collector, max_pending=10)

        try:
            row = make_pipeline_row({"content": "safe content", "id": 1})
            transform.accept(row, ctx)
            transform.flush_batch_processing(timeout=10.0)

            assert len(collector.results) == 1
            _, result, _ = collector.results[0]
            assert isinstance(result, TransformResult)
            assert result.status == "success"
        finally:
            transform.close()


class TestContentSafetyInternalProcessing:
    """Tests for internal processing methods (used by BatchTransformMixin)."""

    @pytest.fixture(autouse=True)
    def mock_httpx_client(self):
        """Patch httpx.Client to prevent real HTTP calls."""
        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            yield mock_instance

    def test_process_single_with_state_raises_capacity_error_on_rate_limit(self, mock_httpx_client: MagicMock) -> None:
        """Rate limit (429) triggers CapacityError for AIMD retry."""
        import httpx

        from elspeth.plugins.pooling import CapacityError
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_httpx_client.post.side_effect = httpx.HTTPStatusError(
            "Rate limited",
            request=Mock(),
            response=Mock(status_code=429),
        )

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        ctx = make_mock_context()
        transform.on_start(ctx)

        row_data = {"content": "test", "id": 1}
        row = make_pipeline_row(row_data)

        with pytest.raises(CapacityError) as exc_info:
            transform._process_single_with_state(row, "test-state-id")

        assert exc_info.value.status_code == 429

    def test_process_single_with_state_returns_error_on_non_rate_limit_http_error(self, mock_httpx_client: MagicMock) -> None:
        """Non-429 HTTP errors return TransformResult.error (not retryable)."""
        import httpx

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_httpx_client.post.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=Mock(),
            response=Mock(status_code=400),
        )

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        ctx = make_mock_context()
        transform.on_start(ctx)

        row_data = {"content": "test", "id": 1}
        row = make_pipeline_row(row_data)
        result = transform._process_single_with_state(row, "test-state-id")

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.retryable is False

    def test_process_single_with_state_returns_error_on_network_error(self, mock_httpx_client: MagicMock) -> None:
        """Network errors return TransformResult.error (retryable)."""
        import httpx

        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        mock_httpx_client.post.side_effect = httpx.RequestError("Connection failed")

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        ctx = make_mock_context()
        transform.on_start(ctx)

        row_data = {"content": "test", "id": 1}
        row = make_pipeline_row(row_data)
        result = transform._process_single_with_state(row, "test-state-id")

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "network_error"
        assert result.retryable is True


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_close_shuts_down_batch_processing(self) -> None:
        """close() properly shuts down batch processing."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        with patch("httpx.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance

            transform = AzureContentSafety(
                {
                    "endpoint": "https://test.cognitiveservices.azure.com",
                    "api_key": "test-key",
                    "fields": ["content"],
                    "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                    "schema": {"mode": "observed"},
                }
            )

            collector = CollectorOutputPort()
            ctx = make_mock_context()
            transform.on_start(ctx)
            transform.connect_output(collector, max_pending=10)

            # Verify batch is initialized
            assert transform._batch_initialized is True

            # Close should shutdown cleanly
            transform.close()

            # After close, recorder should be cleared
            assert transform._recorder is None

    def test_close_without_batch_init_is_safe(self) -> None:
        """close() is safe to call without connect_output()."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        # Should not raise
        transform.close()

        # Can be called multiple times (idempotent)
        transform.close()

    def test_on_start_captures_recorder(self) -> None:
        """on_start captures recorder reference."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

        transform = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 0},
                "schema": {"mode": "observed"},
            }
        )

        mock_recorder = MagicMock()
        ctx = make_mock_context()
        ctx.landscape = mock_recorder

        # Before on_start, recorder should be None
        assert transform._recorder is None

        # After on_start, recorder should be captured
        transform.on_start(ctx)
        assert transform._recorder is mock_recorder
