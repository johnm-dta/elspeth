"""Tests for ChromaSink plugin lifecycle and write operations."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from unittest.mock import create_autospec, patch

import chromadb
import pytest
from chromadb.api import ClientAPI

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.engine.orchestrator.preflight import (
    SinkEffectCapabilityError,
    validate_sink_effect_capability,
    validate_sink_effect_type_capability,
)
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


def _telemetry_emit(event: Any) -> None:
    """Function spec for telemetry emit interactions."""


def _make_chroma_collection_double() -> Any:
    """Create a Chroma collection double specced to the SDK collection API."""
    return create_autospec(chromadb.Collection, instance=True, spec_set=True)


def _make_chroma_client_double() -> Any:
    """Create a Chroma client double specced to the SDK client API."""
    return create_autospec(ClientAPI, instance=True, spec_set=True)


def _make_sink_with_collection(mock_collection: Any, **config_overrides: Any) -> ChromaSink:
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


class TestChromaSinkCompletionTelemetry:
    """elspeth-ee69831e4c: on_complete telemetry emit must be best-effort.

    Telemetry fires AFTER successful writes and their audit record; an emit
    failure must not fail completion. Tier-1/audit-integrity errors still
    propagate (audit corruption outranks).
    """

    def test_telemetry_failure_does_not_fail_completion(self) -> None:
        sink = ChromaSink(_make_config())
        sink._telemetry_emit = create_autospec(
            _telemetry_emit,
            side_effect=RuntimeError("telemetry transport down"),
            spec_set=True,
        )
        sink.on_complete(_make_lifecycle_ctx())  # must not raise
        sink._telemetry_emit.assert_called_once()

    def test_tier1_error_during_telemetry_propagates(self) -> None:
        sink = ChromaSink(_make_config())
        sink._telemetry_emit = create_autospec(
            _telemetry_emit,
            side_effect=AuditIntegrityError("audit corruption during emit"),
            spec_set=True,
        )
        with pytest.raises(AuditIntegrityError):
            sink.on_complete(_make_lifecycle_ctx())


class TestChromaSinkOnStart:
    def test_constructs_persistent_client(self) -> None:
        sink = inject_write_failure(ChromaSink(_make_config()))
        ctx = _make_lifecycle_ctx()

        with patch("elspeth.plugins.sinks.chroma_sink.chromadb") as mock_chromadb:
            mock_client = _make_chroma_client_double()
            mock_chromadb.PersistentClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = _make_chroma_collection_double()

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
            mock_client = _make_chroma_client_double()
            mock_chromadb.HttpClient.return_value = mock_client
            mock_client.get_or_create_collection.return_value = _make_chroma_collection_double()

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


class TestChromaSinkFlush:
    def test_flush_is_noop(self) -> None:
        mock_collection = _make_chroma_collection_double()
        sink = _make_sink_with_collection(mock_collection)

        sink.flush()

        mock_collection.assert_not_called()


class TestChromaSinkClose:
    def test_close_releases_resources(self) -> None:
        mock_collection = _make_chroma_collection_double()
        sink = _make_sink_with_collection(mock_collection)
        mock_client = _make_chroma_client_double()
        sink._client = mock_client

        sink.close()

        assert sink._client is None
        assert sink._collection is None  # type: ignore[unreachable]
        mock_client.clear_system_cache.assert_called_once()


def _chroma_effect_member(ordinal: int, row: dict[str, object]) -> SinkEffectMember:
    row_json = canonical_json(row)
    lineage_json = "[]"
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json=lineage_json,
        lineage_hash=hashlib.sha256(lineage_json.encode()).hexdigest(),
        payload_hash=hashlib.sha256(row_json.encode()).hexdigest(),
        row=row,
        member_effect_id=stable_hash({"effect": "b" * 64, "ordinal": ordinal}),
    )


def _chroma_effect_context() -> RestrictedSinkEffectContext:
    return RestrictedSinkEffectContext(
        run_id="run-1",
        run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
        operation_id="operation-1",
        sink_node_id="sink-1",
    )


class _RecoverableChromaCollection:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, dict[str, object] | None]] = {}
        self.upsert_count_by_id: dict[str, int] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]] | None,
    ) -> None:
        for index, (document_id, document) in enumerate(zip(ids, documents, strict=True)):
            metadata = None if metadatas is None else dict(metadatas[index])
            self.documents[document_id] = (document, metadata)
            self.upsert_count_by_id[document_id] = self.upsert_count_by_id.get(document_id, 0) + 1

    def get(self, *, ids: list[str], include: list[str] | None = None) -> dict[str, object]:
        del include
        found = [document_id for document_id in ids if document_id in self.documents]
        return {
            "ids": found,
            "documents": [self.documents[document_id][0] for document_id in found],
            "metadatas": [self.documents[document_id][1] for document_id in found],
        }


class TestChromaMemberEffects:
    def test_overwrite_declares_complete_member_effect_capability(self) -> None:
        validate_sink_effect_type_capability(
            ChromaSink,
            "overwrite",
            SinkEffectInputKind.PIPELINE_MEMBERS,
        )

    @pytest.mark.parametrize("mode", ["skip", "error"])
    def test_non_reconcilable_modes_fail_preflight_before_construction(self, mode: str) -> None:
        with pytest.raises(SinkEffectCapabilityError, match=r"on_duplicate=overwrite|overwrite"):
            validate_sink_effect_type_capability(
                ChromaSink,
                mode,
                SinkEffectInputKind.PIPELINE_MEMBERS,
            )

    def test_instance_shadowed_member_method_is_rejected_before_io(self) -> None:
        sink = ChromaSink(_make_config())
        sink.commit_member_effect = None  # type: ignore[method-assign,assignment]
        with pytest.raises(SinkEffectCapabilityError, match="commit_member_effect"):
            validate_sink_effect_capability(
                sink,
                "overwrite",
                SinkEffectInputKind.PIPELINE_MEMBERS,
            )

    def test_overwrite_member_recovery_upserts_each_id_once(self) -> None:
        collection = _RecoverableChromaCollection()
        sink = ChromaSink(_make_config())
        sink._collection = collection  # type: ignore[assignment]
        members = tuple(
            _chroma_effect_member(index, row)
            for index, row in enumerate(
                (
                    {"doc_id": "d1", "text": "one", "topic": "alpha"},
                    {"doc_id": "d2", "text": "two", "topic": "beta"},
                    {"doc_id": "d3", "text": "three", "topic": "gamma"},
                )
            )
        )
        effect_input = SinkEffectPipelineMembersInput(members, members)
        ctx = _chroma_effect_context()
        inspection = sink.inspect_effect(SinkEffectInspectionRequest(effect_id="b" * 64, target="{}", predecessor_descriptor=None), ctx)
        plan = sink.prepare_effect(SinkEffectPrepareRequest(effect_id="b" * 64, effect_input=effect_input, inspection=inspection), ctx)

        sink.commit_member_effect(plan, members[0], effect_input, ctx)

        recovered = ChromaSink(_make_config())
        recovered._collection = collection  # type: ignore[assignment]
        states = [recovered.reconcile_member_effect(plan, member, effect_input, ctx).kind for member in members]
        assert states == [
            SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR,
            SinkEffectReconcileKind.NOT_APPLIED,
            SinkEffectReconcileKind.NOT_APPLIED,
        ]
        for member, state in zip(members, states, strict=True):
            if state is SinkEffectReconcileKind.NOT_APPLIED:
                recovered.commit_member_effect(plan, member, effect_input, ctx)

        assert collection.upsert_count_by_id == {"d1": 1, "d2": 1, "d3": 1}

    @pytest.mark.parametrize(
        ("row", "stored_metadata", "metadata_fields"),
        [
            ({"doc_id": "d1", "text": "one", "topic": None}, None, ["topic"]),
            (
                {"doc_id": "d1", "text": "one", "topic": None, "category": "science"},
                {"category": "science"},
                ["topic", "category"],
            ),
        ],
    )
    def test_reconcile_accepts_chroma_normalized_null_metadata(
        self,
        row: dict[str, object],
        stored_metadata: dict[str, object] | None,
        metadata_fields: list[str],
    ) -> None:
        collection = _RecoverableChromaCollection()
        sink = ChromaSink(
            _make_config(
                field_mapping={
                    "document_field": "text",
                    "id_field": "doc_id",
                    "metadata_fields": metadata_fields,
                },
                schema={"mode": "observed"},
            )
        )
        sink._collection = collection  # type: ignore[assignment]
        member = _chroma_effect_member(0, row)
        effect_input = SinkEffectPipelineMembersInput((member,), (member,))
        ctx = _chroma_effect_context()
        inspection = sink.inspect_effect(SinkEffectInspectionRequest(effect_id="b" * 64, target="{}", predecessor_descriptor=None), ctx)
        plan = sink.prepare_effect(SinkEffectPrepareRequest(effect_id="b" * 64, effect_input=effect_input, inspection=inspection), ctx)
        collection.documents["d1"] = ("one", stored_metadata)

        result = sink.reconcile_member_effect(plan, member, effect_input, ctx)

        assert result.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR

    def test_reconcile_returns_unknown_for_divergent_document_or_metadata(self) -> None:
        collection = _RecoverableChromaCollection()
        sink = ChromaSink(_make_config())
        sink._collection = collection  # type: ignore[assignment]
        member = _chroma_effect_member(0, {"doc_id": "d1", "text": "one", "topic": "alpha"})
        effect_input = SinkEffectPipelineMembersInput((member,), (member,))
        ctx = _chroma_effect_context()
        inspection = sink.inspect_effect(SinkEffectInspectionRequest(effect_id="b" * 64, target="{}", predecessor_descriptor=None), ctx)
        plan = sink.prepare_effect(SinkEffectPrepareRequest(effect_id="b" * 64, effect_input=effect_input, inspection=inspection), ctx)
        collection.documents["d1"] = ("tampered", {"topic": "alpha"})

        result = sink.reconcile_member_effect(plan, member, effect_input, ctx)

        assert result.kind is SinkEffectReconcileKind.UNKNOWN

    def test_effect_mode_resolver_reads_on_duplicate_without_io(self) -> None:
        resolved = ChromaSink._resolve_sink_effect_mode(
            _make_config(on_duplicate="overwrite"),
            purpose=SinkEffectExecutionPurpose.FRESH,
        )
        assert resolved is not None and resolved.value == "overwrite"
