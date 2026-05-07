"""Tests for RAGRetrievalTransform lifecycle and process flow."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.errors import RetrievalNotReadyError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalError
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform


@pytest.fixture(autouse=True)
def _set_fingerprint_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure ELSPETH_FINGERPRINT_KEY is set for all tests."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fingerprint-key-for-rag-tests")


def _make_transform(**overrides: Any) -> RAGRetrievalTransform:
    """Create a transform with valid config."""
    config = {
        "output_prefix": "policy",
        "query_field": "question",
        "provider": "azure_search",
        "provider_config": {
            "endpoint": "https://test.search.windows.net",
            "index": "test-index",
            "api_key": "test-key",
        },
        "schema_config": {"mode": "observed"},
    }
    config.update(overrides)
    return RAGRetrievalTransform(config)


def _mock_ctx(state_id: str = "state-1", token_id: str = "token-1") -> MagicMock:
    """Create a mock TransformContext."""
    ctx = MagicMock()
    ctx.state_id = state_id
    ctx.run_id = "run-1"
    token = MagicMock()
    token.token_id = token_id
    ctx.token = token
    ctx.contract = MagicMock()
    return ctx


def _mock_lifecycle_ctx() -> MagicMock:
    """Create a mock LifecycleContext."""
    ctx = MagicMock()
    ctx.run_id = "run-1"
    ctx.landscape = MagicMock()
    ctx.telemetry_emit = MagicMock()
    ctx.rate_limit_registry = None
    return ctx


def _make_row(data: dict[str, Any]) -> PipelineRow:
    """Create a real PipelineRow."""
    contract = SchemaContract(mode="OBSERVED", fields=())
    return PipelineRow(data, contract)


class TestTransformLifecycle:
    def test_close_before_on_start_does_not_raise(self):
        transform = _make_transform()
        transform.close()

    def test_declares_truthful_pass_through(self) -> None:
        assert RAGRetrievalTransform.passes_through_input is True

    def test_declared_output_fields(self):
        transform = _make_transform()
        expected = frozenset(
            [
                "policy__rag_context",
                "policy__rag_score",
                "policy__rag_count",
                "policy__rag_sources",
            ]
        )
        assert transform.declared_output_fields == expected

    def test_output_schema_config_guaranteed_fields(self):
        transform = _make_transform()
        assert transform._output_schema_config is not None
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset(
            {
                "policy__rag_context",
                "policy__rag_score",
                "policy__rag_count",
                "policy__rag_sources",
            }
        )

    def test_state_id_guard(self):
        transform, _ = _setup_transform_with_mock_provider()

        ctx = _mock_ctx(state_id=None)
        row = _make_row({"question": "test"})

        with pytest.raises(RuntimeError, match="state_id"):
            transform.process(row, ctx)

    def test_forward_probe_preserves_query_field_and_close_remains_safe(self) -> None:
        transform = RAGRetrievalTransform(RAGRetrievalTransform.probe_config())
        original_provider = MagicMock()
        transform._provider = original_provider

        result = transform.execute_forward_invariant_probe(
            transform.forward_invariant_probe_rows(
                _make_row({"baseline": "kept"}),
            ),
            _mock_ctx(),
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline"] == "kept"
        assert result.row["rag_probe_query"] == "What is the policy?"
        assert result.row["policy__rag_context"] == "1. Probe context"
        assert result.row["policy__rag_count"] == 1
        assert result.row["policy__rag_score"] == 0.95
        assert transform._provider is original_provider
        assert transform._on_start_called is False
        original_provider.search.assert_not_called()

        transform.close()
        original_provider.close.assert_called_once()


def _ready_provider_result():
    """Default CollectionReadinessResult for tests that don't care about readiness."""
    from elspeth.contracts.probes import CollectionReadinessResult

    return CollectionReadinessResult(
        collection="test-index",
        reachable=True,
        count=10,
        message="Collection 'test-index' has 10 documents",
    )


def _setup_transform_with_mock_provider(chunks=None, **config_overrides):
    """Create a transform with a mock provider via PROVIDERS registry patch.

    Patches the PROVIDERS registry so on_start() constructs our mock provider
    (which passes the readiness check) instead of a real Azure provider.
    """
    mock_provider = MagicMock()
    mock_provider.search.return_value = chunks or []
    mock_provider.check_readiness.return_value = _ready_provider_result()
    mock_provider.last_skipped_count = 0
    mock_provider.last_skipped_reasons = []

    mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
    mock_factory = MagicMock(return_value=mock_provider)

    transform = _make_transform(**config_overrides)
    lifecycle_ctx = _mock_lifecycle_ctx()

    with patch.dict(
        "elspeth.plugins.transforms.rag.transform.PROVIDERS",
        {"azure_search": (mock_config_cls, mock_factory)},
    ):
        transform.on_start(lifecycle_ctx)

    return transform, mock_provider


class TestProcessFlow:
    def test_successful_retrieval(self):
        chunks = [
            RetrievalChunk(content="Result 1", score=0.9, source_id="doc1", metadata={}),
            RetrievalChunk(content="Result 2", score=0.7, source_id="doc2", metadata={}),
        ]
        transform, _ = _setup_transform_with_mock_provider(chunks)
        row = _make_row({"question": "What is RAG?"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        output = result.row.to_dict()
        assert "policy__rag_context" in output
        assert "1. Result 1" in output["policy__rag_context"]
        assert output["policy__rag_count"] == 2
        assert output["policy__rag_score"] == 0.9
        assert "policy__rag_sources" in output
        sources = json.loads(output["policy__rag_sources"])
        assert len(sources["sources"]) == 2

    def test_successful_retrieval_thaws_nested_source_metadata(self):
        chunks = [
            RetrievalChunk(
                content="Result 1",
                score=0.9,
                source_id="doc1",
                metadata={"outer": {"inner": 1}, "items": [{"k": "v"}]},
            ),
        ]
        transform, _ = _setup_transform_with_mock_provider(chunks)
        row = _make_row({"question": "What is RAG?"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        sources = json.loads(result.row["policy__rag_sources"])
        assert sources["sources"][0]["metadata"] == {"outer": {"inner": 1}, "items": [{"k": "v"}]}

    def test_zero_results_quarantine(self):
        transform, _ = _setup_transform_with_mock_provider(
            chunks=[],
            on_no_results="quarantine",
        )
        row = _make_row({"question": "obscure query"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason["reason"] == "no_results"

    def test_zero_results_quarantine_preserves_skipped_metadata(self):
        transform, mock_provider = _setup_transform_with_mock_provider(
            chunks=[],
            on_no_results="quarantine",
        )
        mock_provider.last_skipped_count = 2
        mock_provider.last_skipped_reasons = [{"reason": "bad_payload"}, {"reason": "score_nan"}]
        row = _make_row({"question": "obscure query"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason["skipped_count"] == 2
        assert result.reason["skipped_reasons"] == [{"reason": "bad_payload"}, {"reason": "score_nan"}]

    def test_zero_results_continue(self):
        transform, _ = _setup_transform_with_mock_provider(
            chunks=[],
            on_no_results="continue",
        )
        row = _make_row({"question": "obscure query"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        output = result.row.to_dict()
        assert output["policy__rag_context"] is None
        assert output["policy__rag_count"] == 0
        assert output["policy__rag_score"] is None

    def test_zero_results_continue_preserves_skipped_metadata(self):
        transform, mock_provider = _setup_transform_with_mock_provider(
            chunks=[],
            on_no_results="continue",
        )
        mock_provider.last_skipped_count = 2
        mock_provider.last_skipped_reasons = [{"reason": "bad_payload"}, {"reason": "score_nan"}]
        row = _make_row({"question": "obscure query"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        metadata = result.success_reason["metadata"]
        assert metadata["skipped_count"] == 2
        assert metadata["skipped_reasons"] == [{"reason": "bad_payload"}, {"reason": "score_nan"}]

    def test_retryable_error_propagates(self):
        transform, mock_provider = _setup_transform_with_mock_provider()
        mock_provider.search.side_effect = RetrievalError(
            "server error",
            retryable=True,
            status_code=500,
        )
        row = _make_row({"question": "test"})
        ctx = _mock_ctx()

        with pytest.raises(RetrievalError) as exc_info:
            transform.process(row, ctx)
        assert exc_info.value.retryable is True

    def test_non_retryable_error_returns_error_result(self):
        transform, mock_provider = _setup_transform_with_mock_provider()
        mock_provider.search.side_effect = RetrievalError(
            "bad request",
            retryable=False,
            status_code=400,
        )
        row = _make_row({"question": "test"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason["reason"] == "retrieval_failed"


class TestOnComplete:
    def test_emits_telemetry(self):
        transform, _ = _setup_transform_with_mock_provider()
        # on_complete uses the telemetry_emit captured during on_start
        # (stored as self._telemetry_emit), so we call on_complete with
        # any lifecycle_ctx — but assert on the transform's stored callback.
        lifecycle_ctx = _mock_lifecycle_ctx()
        transform.on_complete(lifecycle_ctx)
        # The telemetry_emit was set during on_start from the _mock_lifecycle_ctx
        # used inside _setup_transform_with_mock_provider. We need to check the
        # transform's internal reference.
        transform._telemetry_emit.assert_called_once()
        payload = transform._telemetry_emit.call_args[0][0]
        assert payload["event"] == "rag_retrieval_complete"
        assert "run_id" in payload
        assert payload["total_queries"] == 0
        assert payload["quarantine_count"] == 0

    def test_zero_rows_no_statistics_error(self):
        """Welford accumulators with zero rows should not raise."""
        transform, _ = _setup_transform_with_mock_provider()
        lifecycle_ctx = _mock_lifecycle_ctx()
        transform.on_complete(lifecycle_ctx)


class TestProcessGuards:
    def test_process_before_on_start_raises(self):
        transform = _make_transform()
        row = _make_row({"question": "test"})
        ctx = _mock_ctx()
        with pytest.raises(RuntimeError, match="before on_start"):
            transform.process(row, ctx)


class TestNoResultsQuarantineContext:
    """Verify the no_results quarantine error includes full audit context."""

    def test_no_results_error_includes_query_and_provider(self):
        """The no_results error reason must include query and provider for audit traceability."""
        transform, _ = _setup_transform_with_mock_provider(on_no_results="quarantine")

        row = _make_row({"question": "obscure query"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason["reason"] == "no_results"
        assert "query" in result.reason
        assert "provider" in result.reason


class TestRAGTransformReadinessGuard:
    """Tests for the readiness check in on_start()."""

    def _make_mock_provider(self, *, reachable: bool = True, count: int | None = 10, collection: str = "test-index") -> MagicMock:
        """Build a mock provider with check_readiness pre-configured."""
        from elspeth.contracts.probes import CollectionReadinessResult

        mock_provider = MagicMock()
        mock_provider.last_skipped_count = 0
        if not reachable:
            message = f"Collection '{collection}' unreachable"
            count = None  # Unreachable → count unknown
        elif count is not None and count > 0:
            message = f"Collection '{collection}' has {count} documents"
        elif count == 0:
            message = f"Collection '{collection}' is empty"
        else:
            message = f"Collection '{collection}' count unknown"

        mock_provider.check_readiness.return_value = CollectionReadinessResult(
            collection=collection,
            reachable=reachable,
            count=count,
            message=message,
        )
        return mock_provider

    def _run_on_start_with_mock(self, mock_provider: MagicMock) -> RAGRetrievalTransform:
        """Patch PROVIDERS registry and call on_start()."""
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with patch.dict(
            "elspeth.plugins.transforms.rag.transform.PROVIDERS",
            {"azure_search": (mock_config_cls, mock_factory)},
        ):
            transform.on_start(lifecycle_ctx)

        return transform

    def test_populated_collection_passes(self) -> None:
        """on_start() succeeds when collection has documents."""
        mock_provider = self._make_mock_provider(count=10)
        transform = self._run_on_start_with_mock(mock_provider)

        assert transform._provider is mock_provider
        mock_provider.check_readiness.assert_called_once()

    def test_readiness_recorded_in_landscape(self) -> None:
        """on_start() records the readiness check outcome in the audit trail."""
        mock_provider = self._make_mock_provider(count=42, collection="my-index")
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with patch.dict(
            "elspeth.plugins.transforms.rag.transform.PROVIDERS",
            {"azure_search": (mock_config_cls, mock_factory)},
        ):
            transform.on_start(lifecycle_ctx)

        lifecycle_ctx.landscape.record_readiness_check.assert_called_once_with(
            run_id="run-1",
            name="rag_retrieval",
            collection="my-index",
            reachable=True,
            count=42,
            message="Collection 'my-index' has 42 documents",
        )

    def test_empty_collection_raises(self) -> None:
        """on_start() raises RetrievalNotReadyError for empty collection."""
        from elspeth.contracts.errors import RetrievalNotReadyError

        mock_provider = self._make_mock_provider(count=0, reachable=True)
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with (
            patch.dict(
                "elspeth.plugins.transforms.rag.transform.PROVIDERS",
                {"azure_search": (mock_config_cls, mock_factory)},
            ),
            pytest.raises(RetrievalNotReadyError) as exc_info,
        ):
            transform.on_start(lifecycle_ctx)

        assert exc_info.value.collection == "test-index"

    def test_unreachable_collection_raises(self) -> None:
        """on_start() raises RetrievalNotReadyError for unreachable collection."""
        from elspeth.contracts.errors import RetrievalNotReadyError

        mock_provider = self._make_mock_provider(count=0, reachable=False)
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with (
            patch.dict(
                "elspeth.plugins.transforms.rag.transform.PROVIDERS",
                {"azure_search": (mock_config_cls, mock_factory)},
            ),
            pytest.raises(RetrievalNotReadyError) as exc_info,
        ):
            transform.on_start(lifecycle_ctx)

        assert exc_info.value.collection == "test-index"
        assert "unreachable" in exc_info.value.reason.lower()

    def test_error_includes_structured_fields(self) -> None:
        """RetrievalNotReadyError carries collection name and reason."""
        from elspeth.contracts.errors import RetrievalNotReadyError

        mock_provider = self._make_mock_provider(count=0, collection="my-vectors")
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with (
            patch.dict(
                "elspeth.plugins.transforms.rag.transform.PROVIDERS",
                {"azure_search": (mock_config_cls, mock_factory)},
            ),
            pytest.raises(RetrievalNotReadyError) as exc_info,
        ):
            transform.on_start(lifecycle_ctx)

        assert exc_info.value.collection == "my-vectors"
        assert "empty" in exc_info.value.reason.lower()

    def test_failed_readiness_still_recorded_in_landscape(self) -> None:
        """record_readiness_check is called even when the check fails (audit before raise)."""
        mock_provider = self._make_mock_provider(count=0, reachable=True, collection="empty-col")
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(return_value=mock_provider)

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with (
            patch.dict(
                "elspeth.plugins.transforms.rag.transform.PROVIDERS",
                {"azure_search": (mock_config_cls, mock_factory)},
            ),
            pytest.raises(RetrievalNotReadyError),
        ):
            transform.on_start(lifecycle_ctx)

        # Even though RetrievalNotReadyError was raised, the readiness check
        # must have been recorded BEFORE the raise — audit gap fix.
        lifecycle_ctx.landscape.record_readiness_check.assert_called_once_with(
            run_id="run-1",
            name="rag_retrieval",
            collection="empty-col",
            reachable=True,
            count=0,
            message="Collection 'empty-col' is empty",
        )

    def test_provider_construction_failure_is_recorded_before_raise(self) -> None:
        """Constructor-time provider failures still emit a failed readiness audit row."""
        mock_config_cls = MagicMock(return_value=MagicMock(index="test-index"))
        mock_factory = MagicMock(side_effect=RetrievalError("missing collection", retryable=False))

        transform = _make_transform()
        lifecycle_ctx = _mock_lifecycle_ctx()

        with (
            patch.dict(
                "elspeth.plugins.transforms.rag.transform.PROVIDERS",
                {"azure_search": (mock_config_cls, mock_factory)},
            ),
            pytest.raises(RetrievalNotReadyError, match="missing collection"),
        ):
            transform.on_start(lifecycle_ctx)

        lifecycle_ctx.landscape.record_readiness_check.assert_called_once_with(
            run_id="run-1",
            name="rag_retrieval",
            collection="test-index",
            reachable=False,
            count=None,
            message="missing collection",
        )

    def test_count_one_passes(self) -> None:
        """count=1 is the minimum passing value — boundary test."""
        mock_provider = self._make_mock_provider(count=1)
        transform = self._run_on_start_with_mock(mock_provider)

        assert transform._provider is mock_provider

    def test_negative_count_raises(self) -> None:
        """count=-1 (corrupted response) is rejected at CollectionReadinessResult construction."""
        from elspeth.contracts.probes import CollectionReadinessResult

        with pytest.raises(ValueError, match="count must be >= 0"):
            CollectionReadinessResult(
                collection="test-index",
                reachable=True,
                count=-1,
                message="corrupted",
            )


def test_plugin_discoverable():
    """rag_retrieval is found by the plugin scanner."""
    from elspeth.plugins.infrastructure.discovery import PLUGIN_SCAN_CONFIG

    assert "transforms/rag" in PLUGIN_SCAN_CONFIG["transforms"]
