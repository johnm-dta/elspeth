# tests/contracts/transform_contracts/test_azure_content_safety_contract.py
"""Contract tests for AzureContentSafety transform.

Verifies AzureContentSafety honors the TransformProtocol contract.
These tests mock the HTTP client since contract tests verify interface
compliance, not API integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

from .test_transform_protocol import (
    TransformContractPropertyTestBase,
    TransformErrorContractTestBase,
)

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


def _make_safe_response() -> dict[str, Any]:
    """Return a safe Azure Content Safety API response."""
    return {
        "categoriesAnalysis": [
            {"category": "Hate", "severity": 0},
            {"category": "Violence", "severity": 0},
            {"category": "Sexual", "severity": 0},
            {"category": "SelfHarm", "severity": 0},
        ]
    }


def _make_violation_response() -> dict[str, Any]:
    """Return an Azure Content Safety API response with hate violation."""
    return {
        "categoriesAnalysis": [
            {"category": "Hate", "severity": 4},  # Exceeds threshold of 2
            {"category": "Violence", "severity": 0},
            {"category": "Sexual", "severity": 0},
            {"category": "SelfHarm", "severity": 0},
        ]
    }


def _make_mock_context(http_response: dict[str, Any]) -> Mock:
    """Create mock PluginContext with HTTP client returning given response."""
    ctx = Mock(spec=PluginContext)
    ctx.run_id = "test-run-001"

    response_mock = Mock()
    response_mock.status_code = 200
    response_mock.json.return_value = http_response
    response_mock.raise_for_status = Mock()
    ctx.http_client.post.return_value = response_mock

    return ctx


class TestAzureContentSafetyContract(TransformContractPropertyTestBase):
    """Contract tests for AzureContentSafety plugin."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"content": "Hello world", "id": 1}

    @pytest.fixture
    def ctx(self) -> Mock:
        """Override context to provide mocked HTTP client with safe response."""
        return _make_mock_context(_make_safe_response())


class TestAzureContentSafetyErrorContract(TransformErrorContractTestBase):
    """Error contract tests for AzureContentSafety plugin."""

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """Return a configured transform instance."""
        return AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"fields": "dynamic"},
            }
        )

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully."""
        return {"content": "Hello world", "id": 1}

    @pytest.fixture
    def error_input(self) -> dict[str, Any]:
        """Return input that should trigger an error (content safety violation)."""
        return {"content": "Hateful content", "id": 2}

    @pytest.fixture
    def ctx(self) -> Mock:
        """Override context - returns violation response for error testing."""
        return _make_mock_context(_make_violation_response())
