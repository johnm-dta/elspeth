"""Integration tests for RAG retrieval transform pipeline.

These tests exercise the full transform lifecycle with real PipelineRow
instances. Azure provider tests use mocked HTTP (no real Azure calls),
but ChromaDB tests use real ephemeral Chroma — no mocks at all.
All data-carrying types are real to avoid BUG-LINEAGE-01 pattern.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.probes import CollectionReadinessResult
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform


def _ready_result(collection="test-index", count=10):
    """Default readiness result for tests that don't care about readiness."""
    return CollectionReadinessResult(
        collection=collection,
        reachable=True,
        count=count,
        message=f"Collection '{collection}' has {count} documents",
    )


@pytest.fixture(autouse=True)
def _set_fingerprint_key(monkeypatch):
    """Set ELSPETH_FINGERPRINT_KEY via env var."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fingerprint-key-for-integration")


def _make_row(data):
    contract = SchemaContract(mode="OBSERVED", fields=())
    return PipelineRow(data, contract)


def _mock_ctx(state_id="state-1"):
    ctx = MagicMock()
    ctx.state_id = state_id
    ctx.run_id = "run-1"
    token = MagicMock()
    token.token_id = "token-1"
    ctx.token = token
    return ctx


def _mock_lifecycle_ctx():
    ctx = MagicMock()
    ctx.run_id = "run-1"
    ctx.landscape = MagicMock()
    ctx.telemetry_emit = MagicMock()
    ctx.rate_limit_registry = None
    return ctx


def _create_transform_with_lifecycle(**config_overrides):
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
    config.update(config_overrides)
    transform = RAGRetrievalTransform(config)
    # Mock readiness I/O; no real Azure endpoint or DNS is available in tests.
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "10"
    mock_resp.raise_for_status = MagicMock()
    with (
        patch("elspeth.core.security.web.validate_url_for_ssrf"),
        patch("httpx.get", return_value=mock_resp),
    ):
        transform.on_start(_mock_lifecycle_ctx())
    return transform


class TestRAGPipelineIntegration:
    """End-to-end transform process with mock HTTP transport."""

    def test_full_retrieval_pipeline(self):
        """Query -> retrieve -> format -> output fields attached."""
        transform = _create_transform_with_lifecycle()

        chunks = [
            RetrievalChunk(content="Policy section 1", score=0.95, source_id="doc1", metadata={"page": 1}),
            RetrievalChunk(content="Policy section 2", score=0.82, source_id="doc2", metadata={"page": 3}),
        ]

        with patch.object(transform._provider, "search", return_value=chunks):
            row = _make_row({"question": "What is the refund policy?"})
            ctx = _mock_ctx()
            result = transform.process(row, ctx)

        assert result.status == "success"
        output = result.row.to_dict()

        assert "policy__rag_context" in output
        assert "1. Policy section 1" in output["policy__rag_context"]
        assert "2. Policy section 2" in output["policy__rag_context"]
        assert output["policy__rag_count"] == 2
        assert output["policy__rag_score"] == 0.95
        sources = json.loads(output["policy__rag_sources"])
        assert len(sources["sources"]) == 2
        assert sources["sources"][0]["source_id"] == "doc1"

        # Input field preserved
        assert output["question"] == "What is the refund policy?"

        # success_reason shape matches transform implementation
        assert result.success_reason["action"] == "rag_retrieval"
        assert result.success_reason["metadata"]["chunk_count"] == 2
        assert result.success_reason["metadata"]["best_score"] == 0.95

    def test_zero_results_quarantine(self):
        transform = _create_transform_with_lifecycle(on_no_results="quarantine")

        with patch.object(transform._provider, "search", return_value=[]):
            row = _make_row({"question": "obscure query"})
            ctx = _mock_ctx()
            result = transform.process(row, ctx)

        assert result.status == "error"
        assert result.reason["reason"] == "no_results"

    def test_zero_results_continue_with_sentinels(self):
        transform = _create_transform_with_lifecycle(on_no_results="continue")

        with patch.object(transform._provider, "search", return_value=[]):
            row = _make_row({"question": "obscure query"})
            ctx = _mock_ctx()
            result = transform.process(row, ctx)

        assert result.status == "success"
        output = result.row.to_dict()
        assert output["policy__rag_context"] is None
        assert output["policy__rag_count"] == 0
        assert output["policy__rag_score"] is None
        assert result.success_reason["metadata"]["no_results"] is True

    def test_on_complete_with_zero_rows(self):
        # Use same lifecycle_ctx for on_start and on_complete: transform stores
        # telemetry_emit from on_start and calls it in on_complete.
        lifecycle_ctx = _mock_lifecycle_ctx()
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
        transform = RAGRetrievalTransform(config)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "10"
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("elspeth.core.security.web.validate_url_for_ssrf"),
            patch("httpx.get", return_value=mock_resp),
        ):
            transform.on_start(lifecycle_ctx)
        transform.on_complete(lifecycle_ctx)
        lifecycle_ctx.telemetry_emit.assert_called_once()

    def test_plugin_discovery(self):
        from elspeth.plugins.infrastructure.discovery import PLUGIN_SCAN_CONFIG

        assert "transforms/rag" in PLUGIN_SCAN_CONFIG["transforms"]


class TestRAGPipelineWithChromaProvider:
    """End-to-end with real ChromaDB — no mocks at all."""

    chromadb = pytest.importorskip("chromadb")

    def _create_chroma_transform(self, documents, collection_suffix="default", **config_overrides):
        import chromadb

        collection_name = f"test-kb-{collection_suffix}"
        config = {
            "output_prefix": "kb",
            "query_field": "question",
            "provider": "chroma",
            "provider_config": {
                "collection": collection_name,
                "mode": "ephemeral",
                "distance_function": "cosine",
            },
            "schema_config": {"mode": "observed"},
        }
        config.update(config_overrides)

        # Pre-populate the collection BEFORE on_start() so ChromaSearchProvider's
        # __init__ finds it via get_collection(). Chroma ephemeral mode shares a
        # global in-memory backend — the collection must exist before provider init.
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            documents=[d["content"] for d in documents],
            ids=[d["id"] for d in documents],
            metadatas=[d.get("metadata") for d in documents],
        )

        transform = RAGRetrievalTransform(config)
        transform.on_start(_mock_lifecycle_ctx())

        return transform

    def test_full_retrieval_with_real_chroma(self):
        transform = self._create_chroma_transform(
            collection_suffix="retrieval",
            documents=[
                {"id": "policy-1", "content": "Refunds are processed within 30 days of purchase."},
                {"id": "policy-2", "content": "Returns must include original packaging."},
                {"id": "faq-1", "content": "Our office hours are 9am to 5pm Monday through Friday."},
            ],
        )
        row = _make_row({"question": "What is the refund policy?"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        output = result.row.to_dict()
        assert output["kb__rag_count"] >= 1
        assert output["kb__rag_score"] > 0.0
        assert "refund" in output["kb__rag_context"].lower() or "return" in output["kb__rag_context"].lower()

    def test_zero_results_with_high_min_score(self):
        transform = self._create_chroma_transform(
            collection_suffix="no-results",
            documents=[
                {"id": "doc1", "content": "Completely unrelated content about quantum mechanics."},
            ],
            min_score=0.99,
        )
        row = _make_row({"question": "refund policy"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)
        assert result.status == "error"
        assert result.reason["reason"] == "no_results"

    def test_metadata_flows_through_to_sources_json(self):
        transform = self._create_chroma_transform(
            collection_suffix="metadata",
            documents=[
                {"id": "doc1", "content": "Test document", "metadata": {"section": "intro", "page": 1}},
            ],
        )
        row = _make_row({"question": "test document"})
        ctx = _mock_ctx()

        result = transform.process(row, ctx)

        assert result.status == "success"
        sources = json.loads(result.row.to_dict()["kb__rag_sources"])
        assert sources["sources"][0]["metadata"]["section"] == "intro"


class TestRAGExecutionGraphAssembly:
    """Exercises ExecutionGraph.from_plugin_instances() with the RAG transform.

    CLAUDE.md mandates integration tests use from_plugin_instances().
    Uses build_linear_pipeline() from tests/fixtures/pipeline.py — the
    production-path assembly helper that calls from_plugin_instances() internally.
    """

    def test_rag_transform_in_execution_graph(self):
        from elspeth.core.config import SourceSettings
        from elspeth.core.dag import ExecutionGraph
        from tests.fixtures.factories import wire_transforms
        from tests.fixtures.plugins import CollectSink, ListSource

        rag_transform = RAGRetrievalTransform(
            {
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
        )

        source_connection = "list_source_out"
        sink_name = "default"
        source = ListSource([], name="list_source", on_success=source_connection)
        source_settings = SourceSettings(
            plugin=source.name,
            on_success=source_connection,
            options={},
        )
        sink = CollectSink(sink_name)
        wired_transforms = wire_transforms(
            [rag_transform],
            source_connection=source_connection,
            final_sink=sink_name,
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            source_settings=source_settings,
            transforms=wired_transforms,
            sinks={sink_name: sink},
            aggregations={},
            gates=[],
        )

        assert graph is not None
        node_plugin_names = [node.plugin_name for node in graph.get_nodes()]
        assert "rag_retrieval" in node_plugin_names
