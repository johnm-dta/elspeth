# tests/unit/contracts/transform_contracts/test_azure_content_safety_contract.py
"""Contract tests for Azure Content Safety transform.

Note: Row-based contract tests (TransformContractPropertyTestBase) were removed because
AzureContentSafety uses BatchTransformMixin and doesn't support process(). The attribute
tests (name, schema, determinism) are now included in BatchTransformContractTestBase.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety

from ._azure_batch_helpers import (
    patch_httpx_client_with_default,
    set_httpx_response,
    submit_and_collect_single_result,
)
from .test_batch_transform_protocol import BatchTransformContractTestBase, CollectingOutputPort


def _make_content_safety_response(
    *,
    hate: int = 0,
    violence: int = 0,
    sexual: int = 0,
    self_harm: int = 0,
) -> dict[str, Any]:
    """Per-service response factory: encodes Content Safety severity surface."""
    return {
        "categoriesAnalysis": [
            {"category": "Hate", "severity": hate},
            {"category": "Violence", "severity": violence},
            {"category": "Sexual", "severity": sexual},
            {"category": "SelfHarm", "severity": self_harm},
        ]
    }


class TestAzureContentSafetyBatchContract(BatchTransformContractTestBase):
    """Verify Azure Content Safety transform honors BatchTransformMixin contract.

    These tests are critical for production use as they verify:
    - accept() returns immediately (non-blocking pipeline throughput)
    - Results arrive via OutputPort in FIFO order (audit trail integrity)
    - Token/state_id tracking is correct (lineage preservation)
    - Lifecycle methods are idempotent (crash recovery safety)
    """

    @pytest.fixture(autouse=True)
    def mock_httpx_for_batch(self):
        """Mock httpx.Client for all batch contract tests."""
        with patch_httpx_client_with_default(_make_content_safety_response) as mock_client_class:
            yield mock_client_class

    @pytest.fixture
    def batch_transform(self) -> BatchTransformMixin:
        """Provide unconfigured transform (no connect_output yet)."""
        t = AzureContentSafety(
            {
                "endpoint": "https://test.cognitiveservices.azure.com",
                "api_key": "test-key",
                "fields": ["content"],
                "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
                "schema": {"mode": "observed"},
            }
        )
        t.on_error = "quarantine_sink"
        return t

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        return {"content": "Hello world", "id": 1}

    def test_severity_equal_to_threshold_emits_success(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: threshold comparison is strictly greater-than, not greater-or-equal."""
        set_httpx_response(
            mock_httpx_for_batch,
            _make_content_safety_response(hate=2, violence=2, sexual=2, self_harm=2),
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row.to_dict() == valid_input

    def test_severity_above_threshold_emits_content_safety_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: harmful content produces a structured error for on_error routing."""
        set_httpx_response(
            mock_httpx_for_batch,
            _make_content_safety_response(hate=3, violence=1, sexual=0, self_harm=0),
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "content_safety_violation"
        assert result.reason["field"] == "content"

        categories = result.reason["categories"]
        assert isinstance(categories, dict)
        assert categories["hate"] == {"severity": 3, "threshold": 2, "exceeded": True}
        assert categories["violence"] == {"severity": 1, "threshold": 2, "exceeded": False}
