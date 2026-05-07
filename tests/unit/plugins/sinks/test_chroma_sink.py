"""Tests for ChromaSink plugin lifecycle and write operations."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.enums import CallStatus, CallType
from elspeth.contracts.errors import AuditIntegrityError, DuplicateDocumentError
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context, make_operation_context


def _make_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "collection": "test-collection",
        "mode": "persistent",
        "persist_directory": "./test_chroma",
        "distance_function": "cosine",
        "field_mapping": {
            "document_field": "text",
            "id_field": "doc_id",
            "metadata_fields": ["topic"],
        },
        "on_duplicate": "overwrite",
        "schema": {
            "mode": "fixed",
            "fields": ["doc_id: str", "text: str", "topic: str"],
        },
    }
    config.update(overrides)
    return config


def _make_sink_with_collection(mock_collection: MagicMock, **config_overrides: Any) -> ChromaSink:
    """Create a ChromaSink with a pre-set mock collection (skips on_start)."""
    sink = inject_write_failure(ChromaSink(_make_config(**config_overrides)))
    sink._collection = mock_collection
    return sink


def _make_lifecycle_ctx() -> Any:
    """Build a context suitable for on_start() / on_complete() lifecycle hooks."""
    return make_context()


def _make_sink_ctx() -> Any:
    """Build a PluginContext suitable for sink.write() calls with real audit trail."""
    return make_operation_context(
        operation_type="sink_write",
        node_id="sink",
        node_type="SINK",
        plugin_name="chroma_sink",
    )


def _make_mock_audit_ctx() -> Any:
    """Build a PluginContext with mock landscape for tests that inspect record_call args."""
    return make_context()


class TestChromaSinkOnStart:
    def test_constructs_persistent_client(self) -> None:
        sink = inject_write_failure(ChromaSink(_make_config()))
        ctx = _make_lifecycle_ctx()

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = MagicMock()

            sink.on_start(ctx)

            mock_chromadb.PersistentClient.assert_called_once()

    def test_constructs_http_client_with_heartbeat(self) -> None:
        config = {
            "collection": "test-collection",
            "mode": "client",
            "host": "localhost",
            "port": 8000,
            "ssl": False,
            "field_mapping": {
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": [],
            },
            "schema": {
                "mode": "fixed",
                "fields": ["doc_id: str", "text: str"],
            },
        }
        sink = inject_write_failure(ChromaSink(config))
        ctx = _make_lifecycle_ctx()

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_client = MagicMock()
            mock_chromadb.HttpClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = MagicMock()

            sink.on_start(ctx)

            mock_chromadb.HttpClient.assert_called_once_with(
                host="localhost",
                port=8000,
                ssl=False,
            )
            mock_client.heartbeat.assert_called_once()

    def test_on_start_failure_raises(self) -> None:
        sink = inject_write_failure(ChromaSink(_make_config()))
        ctx = _make_lifecycle_ctx()

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.side_effect = RuntimeError("Connection refused")

            with pytest.raises(RuntimeError, match="Connection refused"):
                sink.on_start(ctx)


class TestChromaSinkWriteOverwrite:
    def test_upserts_rows_with_correct_mapping(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()

        rows = [
            {"doc_id": "d1", "text": "Hello world", "topic": "greeting"},
            {"doc_id": "d2", "text": "Goodbye", "topic": "farewell"},
        ]

        result = sink.write(rows, ctx)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert call_kwargs.kwargs["ids"] == ["d1", "d2"]
        assert call_kwargs.kwargs["documents"] == ["Hello world", "Goodbye"]
        assert call_kwargs.kwargs["metadatas"] == [
            {"topic": "greeting"},
            {"topic": "farewell"},
        ]
        assert result is not None

    def test_records_audit_call(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_mock_audit_ctx()

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]
        sink.write(rows, ctx)

        ctx.landscape.record_call.assert_called_once()
        call_kwargs = ctx.landscape.record_call.call_args.kwargs
        assert call_kwargs["call_type"] is CallType.VECTOR

    def test_returns_artifact_with_content_hash(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]
        result = sink.write(rows, ctx)

        assert result.artifact.content_hash is not None
        assert len(result.artifact.content_hash) == 64  # SHA-256 hex

    def test_empty_rows_returns_empty_artifact(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()

        result = sink.write([], ctx)

        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 0
        mock_collection.upsert.assert_not_called()


class TestChromaSinkWriteSkip:
    def test_skip_filters_existing_ids(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="skip")

        ctx = _make_sink_ctx()

        rows = [
            {"doc_id": "d1", "text": "Existing", "topic": "t"},
            {"doc_id": "d2", "text": "New", "topic": "t"},
        ]
        sink.write(rows, ctx)

        mock_collection.add.assert_called_once()
        add_kwargs = mock_collection.add.call_args.kwargs
        assert add_kwargs["ids"] == ["d2"]
        assert add_kwargs["documents"] == ["New"]

    def test_skip_all_existing_does_not_call_add(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1", "d2"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="skip")

        ctx = _make_sink_ctx()

        rows = [
            {"doc_id": "d1", "text": "A", "topic": "t"},
            {"doc_id": "d2", "text": "B", "topic": "t"},
        ]
        sink.write(rows, ctx)

        mock_collection.add.assert_not_called()

    def test_skip_audit_records_actual_rows_written(self) -> None:
        """Skip mode must record only the rows actually sent to ChromaDB."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="skip")

        ctx = _make_mock_audit_ctx()

        rows = [
            {"doc_id": "d1", "text": "Existing", "topic": "t"},
            {"doc_id": "d2", "text": "New", "topic": "t"},
        ]
        artifact = sink.write(rows, ctx)

        # Artifact must reflect actual write, not full batch
        assert artifact.artifact.metadata is not None
        assert artifact.artifact.metadata["row_count"] == 1

        # Audit call must include skip info — check request/response data
        # passed through to landscape.record_call via RawCallPayload
        ctx.landscape.record_call.assert_called_once()
        call_kwargs = ctx.landscape.record_call.call_args.kwargs
        request_data = call_kwargs["request_data"].to_dict()
        response_data = call_kwargs["response_data"].to_dict()
        assert request_data["row_count"] == 1  # Actual write, not full batch
        assert request_data["batch_size"] == 2  # Full batch size
        assert request_data["document_ids"] == ["d2"]  # Only written IDs
        assert response_data["rows_written"] == 1
        assert response_data["rows_skipped"] == 1
        assert response_data["skipped_ids"] == ["d1"]

    def test_skip_content_hash_reflects_actual_payload(self) -> None:
        """Hash must be over the subset actually sent, not the full batch."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="skip")

        ctx = _make_sink_ctx()
        rows = [
            {"doc_id": "d1", "text": "Existing", "topic": "t"},
            {"doc_id": "d2", "text": "New", "topic": "t"},
        ]
        skip_artifact = sink.write(rows, ctx)

        # Write only d2 in overwrite mode — should produce the same hash
        mock_collection2 = MagicMock()
        sink2 = _make_sink_with_collection(mock_collection2)
        ctx2 = _make_sink_ctx()
        overwrite_artifact = sink2.write([{"doc_id": "d2", "text": "New", "topic": "t"}], ctx2)

        assert skip_artifact.artifact.content_hash == overwrite_artifact.artifact.content_hash


class TestChromaSinkWriteError:
    def test_error_mode_raises_on_duplicates(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1", "d3"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="error")

        ctx = _make_sink_ctx()

        rows = [
            {"doc_id": "d1", "text": "A", "topic": "t"},
            {"doc_id": "d2", "text": "B", "topic": "t"},
            {"doc_id": "d3", "text": "C", "topic": "t"},
        ]

        with pytest.raises(DuplicateDocumentError) as exc_info:
            sink.write(rows, ctx)

        assert "d1" in exc_info.value.duplicate_ids
        assert "d3" in exc_info.value.duplicate_ids
        mock_collection.add.assert_not_called()

    def test_error_mode_succeeds_when_no_duplicates(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="error")

        ctx = _make_sink_ctx()

        rows = [{"doc_id": "d1", "text": "A", "topic": "t"}]
        result = sink.write(rows, ctx)

        mock_collection.add.assert_called_once()
        assert result is not None

    def test_error_mode_records_audit_before_raising(self) -> None:
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="error")

        ctx = _make_mock_audit_ctx()

        with pytest.raises(DuplicateDocumentError):
            sink.write([{"doc_id": "d1", "text": "A", "topic": "t"}], ctx)

        ctx.landscape.record_call.assert_called_once()
        call_kwargs = ctx.landscape.record_call.call_args.kwargs
        assert call_kwargs["status"] is CallStatus.ERROR
        # Error details must include exception info (#4)
        error_data = call_kwargs["error"].to_dict()
        assert error_data["type"] == "DuplicateDocumentError"
        assert "d1" in error_data["message"]

    def test_duplicate_ids_stored_as_tuple(self) -> None:
        """DuplicateDocumentError.duplicate_ids must be immutable."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": ["d1"]}
        sink = _make_sink_with_collection(mock_collection, on_duplicate="error")

        ctx = _make_sink_ctx()
        with pytest.raises(DuplicateDocumentError) as exc_info:
            sink.write([{"doc_id": "d1", "text": "A", "topic": "t"}], ctx)

        assert isinstance(exc_info.value.duplicate_ids, tuple)


class TestChromaSinkAuditIntegrity:
    def test_audit_recording_failure_raises_audit_integrity_error(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()

        def failing_record_call(**kwargs: Any) -> Any:
            raise RuntimeError("DB write failed")

        ctx.record_call = failing_record_call

        rows = [{"doc_id": "d1", "text": "Hello", "topic": "t"}]

        with pytest.raises(AuditIntegrityError, match="audit") as exc_info:
            sink.write(rows, ctx)

        # The ChromaDB write still happened
        mock_collection.upsert.assert_called_once()
        # Exception chain preserved (#13)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_value_error_from_chroma_gets_audit_record(self) -> None:
        """Chroma raises ValueError for invalid payloads (e.g. nested dict in metadata).

        ValueError from Chroma API calls is wrapped as _ChromaPayloadRejection
        so it can be distinguished from ValueError in our own code (which should
        crash as a framework bug). The audit record captures the original error.
        """
        from elspeth.plugins.sinks.chroma_sink import _ChromaPayloadRejection

        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = ValueError("Expected str, int, float or bool, got dict")
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_mock_audit_ctx()

        with pytest.raises(_ChromaPayloadRejection, match="Expected str"):
            sink.write([{"doc_id": "d1", "text": "A", "topic": "t"}], ctx)

        # Audit record must have been written before re-raising
        ctx.landscape.record_call.assert_called_once()
        call_kwargs = ctx.landscape.record_call.call_args.kwargs
        assert call_kwargs["status"] is CallStatus.ERROR
        error_data = call_kwargs["error"].to_dict()
        assert error_data["type"] == "_ChromaPayloadRejection"

    def test_error_path_audit_failure_raises_audit_integrity_error(self) -> None:
        """If error-path record_call also fails, AuditIntegrityError is raised."""
        import chromadb.errors

        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = chromadb.errors.ChromaError("write failed")  # type: ignore[abstract]
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()

        def failing_record_call(**kwargs: Any) -> Any:
            raise RuntimeError("audit also broken")

        ctx.record_call = failing_record_call

        with pytest.raises(AuditIntegrityError, match="audit") as exc_info:
            sink.write([{"doc_id": "d1", "text": "A", "topic": "t"}], ctx)

        # Exception chain preserved — audit_exc is the cause
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestChromaSinkFlush:
    def test_flush_is_noop(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        sink.flush()

        mock_collection.assert_not_called()


class TestChromaSinkClose:
    def test_close_releases_resources(self) -> None:
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        mock_client = MagicMock()
        sink._client = mock_client

        sink.close()

        assert sink._client is None
        assert sink._collection is None  # type: ignore[unreachable]
        mock_client.clear_system_cache.assert_called_once()


class TestChromaSinkMetadataTypeValidation:
    """Metadata values must be str|int|float|bool|None — ChromaDB's constraint."""

    def test_valid_scalar_types_pass(self) -> None:
        """str, int, float, bool metadata values are accepted."""
        mock_collection = MagicMock()
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["s", "i", "f", "b"],
            },
            schema={
                "mode": "flexible",
                "fields": [
                    "doc_id: str",
                    "text: str",
                    "s: str",
                    "i: int",
                    "f: float",
                    "b: bool",
                ],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hi", "s": "val", "i": 42, "f": 3.14, "b": True}]
        sink.write(rows, ctx)

        mock_collection.upsert.assert_called_once()

    def test_none_metadata_value_passes(self) -> None:
        """None is a valid ChromaDB metadata value."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hi", "topic": None}]
        sink.write(rows, ctx)

        mock_collection.upsert.assert_called_once()

    def test_invalid_type_filtered_and_diverted(self) -> None:
        """Rows with non-scalar metadata are diverted, not sent to ChromaDB."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "good", "topic": "science"},
            {"doc_id": "d2", "text": "bad", "topic": {"nested": "dict"}},
            {"doc_id": "d3", "text": "good", "topic": "math"},
        ]
        result = sink.write(rows, ctx)

        # Only valid rows sent to ChromaDB
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["d1", "d3"]
        assert call_kwargs["documents"] == ["good", "good"]

        # Diversion records the rejection
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 1
        assert "topic" in result.diversions[0].reason
        assert result.diversions[0].row_data["doc_id"] == "d2"

    def test_all_rows_rejected_returns_zero_write(self) -> None:
        """When ALL rows have bad metadata, return zero-write artifact with diversions."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "bad", "topic": ["a", "list"]},
            {"doc_id": "d2", "text": "bad", "topic": {"nested": "dict"}},
        ]
        result = sink.write(rows, ctx)

        # Nothing sent to ChromaDB
        mock_collection.upsert.assert_not_called()
        mock_collection.add.assert_not_called()

        # Diversions capture all rejections
        assert len(result.diversions) == 2
        assert result.diversions[0].row_index == 0
        assert result.diversions[1].row_index == 1

        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 0

    @pytest.mark.parametrize(
        "bad_value, expected_type_name",
        [
            ({"nested": "dict"}, "dict"),
            (["a", "list"], "list"),
            ((1, 2), "tuple"),
        ],
        ids=["dict", "list", "tuple"],
    )
    def test_diversion_reason_includes_type_name(self, bad_value: Any, expected_type_name: str) -> None:
        """Diversion reason records the actual type name for diagnosis."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hi", "topic": bad_value}]
        result = sink.write(rows, ctx)

        assert len(result.diversions) == 1
        assert expected_type_name in result.diversions[0].reason
        assert "topic" in result.diversions[0].reason

    def test_no_metadata_fields_skips_validation(self) -> None:
        """When metadata_fields is empty, no validation needed (metadatas=None)."""
        mock_collection = MagicMock()
        config = _make_config(
            field_mapping={"document_field": "text", "id_field": "doc_id", "metadata_fields": []},
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hi"}]
        sink.write(rows, ctx)

        mock_collection.upsert.assert_called_once()


class TestChromaSinkDivertRow:
    """Tests for _divert_row() integration with metadata validation."""

    def test_invalid_metadata_diverted(self) -> None:
        """Rows with non-primitive metadata types are diverted via _divert_row()."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "good", "topic": "science"},
            {"doc_id": "d2", "text": "bad", "topic": {"nested": "dict"}},
            {"doc_id": "d3", "text": "also bad", "topic": [1, 2, 3]},
        ]
        result = sink.write(rows, ctx)

        # Only the valid row is written
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["d1"]

        # Two rows diverted
        assert len(result.diversions) == 2
        assert result.diversions[0].row_index == 1
        assert result.diversions[0].row_data["doc_id"] == "d2"
        assert "Invalid ChromaDB metadata types" in result.diversions[0].reason
        assert result.diversions[1].row_index == 2
        assert result.diversions[1].row_data["doc_id"] == "d3"

    def test_all_valid_no_diversions(self) -> None:
        """When all rows have valid metadata, no diversions are recorded."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows = [
            {"doc_id": "d1", "text": "hello", "topic": "greeting"},
            {"doc_id": "d2", "text": "goodbye", "topic": "farewell"},
        ]
        result = sink.write(rows, ctx)

        assert len(result.diversions) == 0
        mock_collection.upsert.assert_called_once()

    def test_all_invalid_all_diverted(self) -> None:
        """When all rows have bad metadata, all are diverted and nothing written."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "bad1", "topic": {"nested": True}},
            {"doc_id": "d2", "text": "bad2", "topic": [1, 2]},
            {"doc_id": "d3", "text": "bad3", "topic": (1, 2)},
        ]
        result = sink.write(rows, ctx)

        # Nothing written to ChromaDB
        mock_collection.upsert.assert_not_called()
        mock_collection.add.assert_not_called()

        # All rows diverted
        assert len(result.diversions) == 3
        assert result.diversions[0].row_index == 0
        assert result.diversions[1].row_index == 1
        assert result.diversions[2].row_index == 2
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 0


class TestChromaSinkNonFiniteMetadata:
    """Non-finite float metadata must be diverted before reaching ChromaDB or canonical hashing."""

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_non_finite_metadata_diverted(self, bad_value: float) -> None:
        """Row with non-finite float metadata is diverted, valid rows still written."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "good", "topic": "science"},
            {"doc_id": "d2", "text": "bad", "topic": bad_value},
        ]
        result = sink.write(rows, ctx)

        # Only valid row sent to ChromaDB
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["d1"]
        assert call_kwargs["documents"] == ["good"]

        # Bad row diverted with descriptive reason
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 1
        assert "non-finite" in result.diversions[0].reason
        assert "topic" in result.diversions[0].reason

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "neg_inf"])
    def test_all_non_finite_metadata_returns_zero_write(self, bad_value: float) -> None:
        """When all rows have non-finite metadata, nothing is sent to ChromaDB."""
        mock_collection = MagicMock()
        sink = _make_sink_with_collection(mock_collection)
        sink._on_write_failure = "csv_failsink"

        ctx = _make_sink_ctx()
        rows: list[dict[str, Any]] = [
            {"doc_id": "d1", "text": "bad", "topic": bad_value},
        ]
        result = sink.write(rows, ctx)

        mock_collection.upsert.assert_not_called()
        assert len(result.diversions) == 1


class TestChromaSinkEmptyMetadata:
    """Empty metadata dicts {} crash ChromaDB — rows with all metadata fields absent
    must be sent with metadatas=None, not metadatas=[{}]."""

    def test_all_metadata_fields_absent_single_row(self) -> None:
        """A row missing ALL configured metadata fields must not produce {}."""
        mock_collection = MagicMock()
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["topic", "category"],
            },
            schema={
                "mode": "flexible",
                "fields": ["doc_id: str", "text: str", "topic: str", "category: str"],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        # Row has id and document but none of the configured metadata fields
        rows = [{"doc_id": "d1", "text": "hello"}]
        result = sink.write(rows, ctx)

        # Row must be written (not diverted — missing metadata is not an error)
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["ids"] == ["d1"]
        assert call_kwargs["documents"] == ["hello"]
        # metadatas must be None, NOT [{}]
        assert call_kwargs["metadatas"] is None
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 1
        assert len(result.diversions) == 0

    def test_mixed_batch_metadata_present_and_absent(self) -> None:
        """Batch with some rows having metadata and others not — both are written."""
        mock_collection = MagicMock()
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["topic"],
            },
            schema={
                "mode": "flexible",
                "fields": ["doc_id: str", "text: str", "topic: str"],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        rows = [
            {"doc_id": "d1", "text": "with meta", "topic": "science"},
            {"doc_id": "d2", "text": "no meta"},  # topic absent
        ]
        result = sink.write(rows, ctx)

        # Both rows must be written — two sub-batch calls
        assert mock_collection.upsert.call_count == 2

        # First call: row with metadata
        first_call = mock_collection.upsert.call_args_list[0].kwargs
        assert first_call["ids"] == ["d1"]
        assert first_call["metadatas"] == [{"topic": "science"}]

        # Second call: row without metadata
        second_call = mock_collection.upsert.call_args_list[1].kwargs
        assert second_call["ids"] == ["d2"]
        assert second_call["metadatas"] is None

        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 2
        assert len(result.diversions) == 0

    def test_all_metadata_fields_absent_skip_mode(self) -> None:
        """Skip mode works correctly when metadata fields are absent."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["topic"],
            },
            on_duplicate="skip",
            schema={
                "mode": "flexible",
                "fields": ["doc_id: str", "text: str", "topic: str"],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hello"}]
        result = sink.write(rows, ctx)

        # Row is new (not in existing), so it's added
        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args.kwargs
        assert call_kwargs["metadatas"] is None
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 1

    def test_all_metadata_fields_absent_error_mode(self) -> None:
        """Error mode works correctly when metadata fields are absent."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"ids": []}
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["topic"],
            },
            on_duplicate="error",
            schema={
                "mode": "flexible",
                "fields": ["doc_id: str", "text: str", "topic: str"],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        rows = [{"doc_id": "d1", "text": "hello"}]
        result = sink.write(rows, ctx)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args.kwargs
        assert call_kwargs["metadatas"] is None
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 1

    def test_partial_metadata_fields_not_empty(self) -> None:
        """Row with SOME metadata fields present produces a non-empty dict (not affected)."""
        mock_collection = MagicMock()
        config = _make_config(
            field_mapping={
                "document_field": "text",
                "id_field": "doc_id",
                "metadata_fields": ["topic", "category"],
            },
            schema={
                "mode": "flexible",
                "fields": ["doc_id: str", "text: str", "topic: str", "category: str"],
            },
        )
        sink = inject_write_failure(ChromaSink(config))
        sink._collection = mock_collection

        ctx = _make_sink_ctx()
        # Row has topic but not category — meta is {"topic": "science"}, not empty
        rows = [{"doc_id": "d1", "text": "hello", "topic": "science"}]
        result = sink.write(rows, ctx)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args.kwargs
        assert call_kwargs["metadatas"] == [{"topic": "science"}]
        assert len(result.diversions) == 0
