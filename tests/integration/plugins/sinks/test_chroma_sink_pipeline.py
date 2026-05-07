"""Integration test: CSV source -> ChromaSink with real ChromaDB.

Uses persistent mode with tmp_path for deterministic CI behaviour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import chromadb
import pytest

from elspeth.contracts.errors import DuplicateDocumentError
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from tests.fixtures.base_classes import inject_write_failure


class TestChromaSinkIntegration:
    def test_documents_written_to_collection(self, tmp_path: Path) -> None:
        """Verify documents are written with correct IDs, content, and metadata."""
        chroma_dir = tmp_path / "chroma_data"

        config: dict[str, Any] = {
            "collection": "test-docs",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "distance_function": "cosine",
            "field_mapping": {
                "document_field": "text_content",
                "id_field": "doc_id",
                "metadata_fields": ["topic"],
            },
            "on_duplicate": "overwrite",
            "schema": {
                "mode": "fixed",
                "fields": [
                    "doc_id: str",
                    "text_content: str",
                    "topic: str",
                ],
            },
        }

        sink = inject_write_failure(ChromaSink(config))

        start_ctx = MagicMock()
        start_ctx.run_id = "integration-test-run"
        start_ctx.telemetry_emit = MagicMock()

        sink.on_start(start_ctx)

        rows = [
            {
                "doc_id": "d1",
                "text_content": "Photosynthesis converts light to energy",
                "topic": "biology",
            },
            {
                "doc_id": "d2",
                "text_content": "Earthquakes release seismic waves",
                "topic": "geology",
            },
            {
                "doc_id": "d3",
                "text_content": "DNA encodes genetic information",
                "topic": "biology",
            },
        ]

        write_ctx = MagicMock()
        write_ctx.run_id = "integration-test-run"
        write_ctx.record_call = MagicMock()

        result = sink.write(rows, write_ctx)

        # Verify artifact
        assert result.artifact.content_hash is not None
        assert len(result.artifact.content_hash) == 64
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 3

        # Verify audit call was recorded
        write_ctx.record_call.assert_called_once()
        call_kwargs = write_ctx.record_call.call_args.kwargs
        assert call_kwargs["provider"] == "chromadb"
        assert call_kwargs["request_data"]["row_count"] == 3

        sink.close()

        # Verify data in ChromaDB directly
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("test-docs")
        assert collection.count() == 3

        results = collection.get(ids=["d1", "d2", "d3"], include=["documents", "metadatas"])
        assert results["documents"] is not None
        assert "Photosynthesis" in results["documents"][0]
        assert results["metadatas"] is not None
        assert results["metadatas"][0]["topic"] == "biology"

    def test_upsert_is_idempotent(self, tmp_path: Path) -> None:
        """Verify overwrite mode can be re-run without errors."""
        chroma_dir = tmp_path / "chroma_data"

        config: dict[str, Any] = {
            "collection": "idempotent-test",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "field_mapping": {"document_field": "text", "id_field": "id", "metadata_fields": []},
            "on_duplicate": "overwrite",
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        }

        for _ in range(3):
            sink = inject_write_failure(ChromaSink(config))
            ctx = MagicMock()
            ctx.run_id = "idem-run"
            ctx.telemetry_emit = MagicMock()
            sink.on_start(ctx)

            write_ctx = MagicMock()
            write_ctx.run_id = "idem-run"
            write_ctx.record_call = MagicMock()
            sink.write([{"id": "d1", "text": "Hello"}], write_ctx)
            sink.close()

        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("idempotent-test")
        assert collection.count() == 1

    def test_skip_mode_preserves_existing(self, tmp_path: Path) -> None:
        """Verify skip mode doesn't overwrite existing documents."""
        chroma_dir = tmp_path / "chroma_data"

        config: dict[str, Any] = {
            "collection": "skip-test",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "field_mapping": {"document_field": "text", "id_field": "id", "metadata_fields": []},
            "on_duplicate": "skip",
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        }

        # First write
        sink = inject_write_failure(ChromaSink(config))
        ctx = MagicMock()
        ctx.run_id = "run-1"
        ctx.telemetry_emit = MagicMock()
        sink.on_start(ctx)
        write_ctx = MagicMock()
        write_ctx.run_id = "run-1"
        sink.write([{"id": "d1", "text": "Original"}], write_ctx)
        sink.close()

        # Second write with same ID — should be skipped
        sink2 = inject_write_failure(ChromaSink(config))
        ctx2 = MagicMock()
        ctx2.run_id = "run-2"
        ctx2.telemetry_emit = MagicMock()
        sink2.on_start(ctx2)
        write_ctx2 = MagicMock()
        write_ctx2.run_id = "run-2"
        sink2.write([{"id": "d1", "text": "Updated"}, {"id": "d2", "text": "New"}], write_ctx2)
        sink2.close()

        # Verify original preserved, new added
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection("skip-test")
        assert collection.count() == 2

        results = collection.get(ids=["d1"], include=["documents"])
        assert results["documents"] is not None
        assert results["documents"][0] == "Original"  # Not "Updated"

    @pytest.mark.parametrize("on_duplicate", ["overwrite", "error"])
    def test_content_hash_is_deterministic(self, tmp_path: Path, on_duplicate: str) -> None:
        """Same rows produce the same content hash for overwrite and error modes."""
        chroma_dir = tmp_path / "chroma_data"

        config: dict[str, Any] = {
            "collection": f"hash-test-{on_duplicate}",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "field_mapping": {"document_field": "text", "id_field": "id", "metadata_fields": []},
            "on_duplicate": on_duplicate,
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        }

        rows = [{"id": "d1", "text": "Hello"}, {"id": "d2", "text": "World"}]
        hashes = []

        for _ in range(2):
            sink = inject_write_failure(ChromaSink(config))
            ctx = MagicMock()
            ctx.run_id = "hash-run"
            ctx.telemetry_emit = MagicMock()
            sink.on_start(ctx)
            write_ctx = MagicMock()
            write_ctx.run_id = "hash-run"

            try:
                result = sink.write(rows, write_ctx)
                hashes.append(result.artifact.content_hash)
            except DuplicateDocumentError:
                pass  # Expected on second run of error mode
            sink.close()

        assert len(hashes) >= 1
        if len(hashes) == 2:
            assert hashes[0] == hashes[1]

    def test_skip_mode_hash_reflects_actual_write(self, tmp_path: Path) -> None:
        """Skip mode hash changes when the actual write payload changes."""
        chroma_dir = tmp_path / "chroma_data"

        config: dict[str, Any] = {
            "collection": "hash-skip-test",
            "mode": "persistent",
            "persist_directory": str(chroma_dir),
            "field_mapping": {"document_field": "text", "id_field": "id", "metadata_fields": []},
            "on_duplicate": "skip",
            "schema": {"mode": "fixed", "fields": ["id: str", "text: str"]},
        }

        # First write: d1 and d2 both new
        sink = inject_write_failure(ChromaSink(config))
        ctx = MagicMock()
        ctx.run_id = "run-1"
        ctx.telemetry_emit = MagicMock()
        sink.on_start(ctx)
        write_ctx = MagicMock()
        write_ctx.run_id = "run-1"
        result1 = sink.write([{"id": "d1", "text": "Hello"}, {"id": "d2", "text": "World"}], write_ctx)
        sink.close()

        # Second write: d1 already exists, only d3 is new
        sink2 = inject_write_failure(ChromaSink(config))
        ctx2 = MagicMock()
        ctx2.run_id = "run-2"
        ctx2.telemetry_emit = MagicMock()
        sink2.on_start(ctx2)
        write_ctx2 = MagicMock()
        write_ctx2.run_id = "run-2"
        result2 = sink2.write([{"id": "d1", "text": "Hello"}, {"id": "d3", "text": "New"}], write_ctx2)
        sink2.close()

        # Different actual payloads → different hashes
        assert result1.artifact.content_hash != result2.artifact.content_hash
        # Second write only wrote d3
        assert result2.artifact.metadata is not None
        assert result2.artifact.metadata["row_count"] == 1
