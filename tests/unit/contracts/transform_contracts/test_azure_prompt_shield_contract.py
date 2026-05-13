# tests/unit/contracts/transform_contracts/test_azure_prompt_shield_contract.py
"""Contract tests for Azure Prompt Shield transform.

Note: Row-based contract tests (TransformContractPropertyTestBase) were removed because
AzurePromptShield uses BatchTransformMixin and doesn't support process(). The attribute
tests (name, schema, determinism) are now included in BatchTransformContractTestBase.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

from ._azure_batch_helpers import (
    patch_httpx_client_with_default,
    set_httpx_response,
    submit_and_collect_single_result,
)
from .test_batch_transform_protocol import BatchTransformContractTestBase, CollectingOutputPort


def _make_prompt_shield_response(
    *,
    user_attack: bool = False,
    document_attack: bool = False,
) -> dict[str, Any]:
    """Per-service response factory: encodes Prompt Shield dual-attack surface."""
    return {
        "userPromptAnalysis": {"attackDetected": user_attack},
        "documentsAnalysis": [{"attackDetected": document_attack}],
    }


def _make_clean_response() -> dict[str, Any]:
    return _make_prompt_shield_response()


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
        with patch_httpx_client_with_default(_make_clean_response) as mock_client_class:
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

    def test_user_prompt_attack_emits_prompt_injection_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: user-prompt attacks produce structured on_error routing data."""
        set_httpx_response(
            mock_httpx_for_batch,
            _make_prompt_shield_response(user_attack=True),
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "prompt_injection_detected"
        assert result.reason["field"] == "prompt"
        assert result.reason["attacks"] == {
            "user_prompt_attack": True,
            "document_attack": False,
        }

    def test_document_attack_emits_prompt_injection_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: document attacks also fail closed through the batch output path."""
        set_httpx_response(
            mock_httpx_for_batch,
            _make_prompt_shield_response(document_attack=True),
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "prompt_injection_detected"
        assert result.reason["field"] == "prompt"
        assert result.reason["attacks"] == {
            "user_prompt_attack": False,
            "document_attack": True,
        }

    def test_user_and_document_attacks_emit_combined_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: mixed attack signals are preserved in the error payload."""
        set_httpx_response(
            mock_httpx_for_batch,
            _make_prompt_shield_response(user_attack=True, document_attack=True),
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "prompt_injection_detected"
        assert result.reason["field"] == "prompt"
        assert result.reason["attacks"] == {
            "user_prompt_attack": True,
            "document_attack": True,
        }

    def test_http_error_emits_nonretryable_api_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: Prompt Shield HTTP failures route to on_error, not success."""
        set_httpx_response(
            mock_httpx_for_batch,
            {"error": {"message": "azure rejected request"}},
            status_code=400,
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "http_error"
        assert result.reason["status_code"] == 400

    def test_multiple_document_results_emit_malformed_response_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: document result count must match the submitted document count."""
        set_httpx_response(
            mock_httpx_for_batch,
            {
                "userPromptAnalysis": {"attackDetected": False},
                "documentsAnalysis": [
                    {"attackDetected": False},
                    {"attackDetected": True},
                ],
            },
        )

        result = submit_and_collect_single_result(
            started_transform,
            valid_input,
            mock_ctx_factory(),
            output_port,
        )

        assert result.status == "error"
        assert result.retryable is False
        assert result.reason is not None
        assert result.reason["reason"] == "api_error"
        assert result.reason["error_type"] == "malformed_response"
        assert "documentsAnalysis must have exactly 1 entry" in result.reason["message"]
