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

from elspeth.contracts import TransformResult
from elspeth.plugins.infrastructure.batching.mixin import BatchTransformMixin
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.testing import make_pipeline_row

from .test_batch_transform_protocol import BatchTransformContractTestBase, CollectingOutputPort

_HTTPX_CLIENT_CLASS = httpx.Client


def _make_prompt_shield_response(
    *,
    user_attack: bool = False,
    document_attack: bool = False,
) -> dict[str, Any]:
    return {
        "userPromptAnalysis": {"attackDetected": user_attack},
        "documentsAnalysis": [{"attackDetected": document_attack}],
    }


def _make_clean_response() -> dict[str, Any]:
    return _make_prompt_shield_response()


def _create_http_response(response_data: dict[str, Any], *, url: str) -> httpx.Response:
    return httpx.Response(
        200,
        json=response_data,
        request=httpx.Request("POST", url),
    )


def _set_prompt_shield_response(
    mock_client_class: MagicMock,
    response_data: dict[str, Any],
) -> None:
    mock_client_instance = mock_client_class.return_value

    def _mocked_post(url: str, **_: object) -> httpx.Response:
        return _create_http_response(response_data, url=url)

    mock_client_instance.post.side_effect = _mocked_post


def _submit_and_collect_single_result(
    started_transform: BatchTransformMixin,
    row_data: dict[str, Any],
    ctx: Any,
    output_port: CollectingOutputPort,
) -> TransformResult:
    started_transform.accept(make_pipeline_row(row_data), ctx)

    arrived = output_port.wait_for_results(1, timeout=10.0)
    assert arrived, "Result did not arrive via OutputPort within timeout"

    results = output_port.get_results()
    assert len(results) == 1
    _token, result, _state_id = results[0]
    assert isinstance(result, TransformResult)
    return result


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

    def test_user_prompt_attack_emits_prompt_injection_error(
        self,
        started_transform: BatchTransformMixin,
        valid_input: dict[str, Any],
        mock_ctx_factory: Any,
        output_port: CollectingOutputPort,
        mock_httpx_for_batch: MagicMock,
    ) -> None:
        """Contract: user-prompt attacks produce structured on_error routing data."""
        _set_prompt_shield_response(
            mock_httpx_for_batch,
            _make_prompt_shield_response(user_attack=True),
        )

        result = _submit_and_collect_single_result(
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
        _set_prompt_shield_response(
            mock_httpx_for_batch,
            _make_prompt_shield_response(document_attack=True),
        )

        result = _submit_and_collect_single_result(
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
