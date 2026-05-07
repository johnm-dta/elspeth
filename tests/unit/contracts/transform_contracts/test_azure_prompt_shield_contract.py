# tests/unit/contracts/transform_contracts/test_azure_prompt_shield_contract.py
"""Contract tests for Azure Prompt Shield transform.

Note: Row-based contract tests (TransformContractPropertyTestBase) were removed because
AzurePromptShield uses BatchTransformMixin and doesn't support process(). The attribute
tests (name, schema, determinism) are now included in BatchTransformContractTestBase.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

from .test_batch_transform_protocol import BatchTransformContractTestBase

_HTTPX_CLIENT_CLASS = httpx.Client


def _make_clean_response() -> dict[str, Any]:
    return {
        "userPromptAnalysis": {"attackDetected": False},
        "documentsAnalysis": [{"attackDetected": False}],
    }


def _create_http_response(response_data: dict[str, Any], *, url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json=response_data,
        request=httpx.Request("POST", url),
    )


class TestAzurePromptShieldBatchContract(BatchTransformContractTestBase):
    """Verify Azure Prompt Shield transform honors BatchTransformMixin contract.

    These tests are critical for production use as they verify:
    - accept() returns immediately (non-blocking pipeline throughput)
    - Results arrive via OutputPort in FIFO order (audit trail integrity)
    - Token/state_id tracking is correct (lineage preservation)
    - Lifecycle methods are idempotent (crash recovery safety)
    """

    @pytest.fixture(autouse=True)
    def mock_httpx_for_batch(self):
        """Mock httpx.Client for all batch contract tests."""
        with patch("httpx.Client", autospec=True) as mock_client_class:
            mock_client_instance = MagicMock(spec_set=_HTTPX_CLIENT_CLASS)

            def _mocked_post(url: str, **_: object) -> httpx.Response:
                return _create_http_response(_make_clean_response(), url=url)

            mock_client_instance.post.side_effect = _mocked_post
            mock_client_class.return_value = mock_client_instance
            yield mock_client_class

    @pytest.fixture
    def batch_transform(self) -> BatchTransformMixin:
        """Provide unconfigured transform (no connect_output yet)."""
        t = AzurePromptShield(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["prompt"],
                "schema": {"mode": "observed"},
            }
        )
        t.on_error = "quarantine_sink"
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"prompt": "What is the weather?", "id": 1}
