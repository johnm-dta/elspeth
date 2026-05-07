# tests/unit/contracts/sink_contracts/test_sink_protocol.py
"""Contract tests for Sink plugins.

These tests verify that sink implementations honor the SinkProtocol contract.
They test interface guarantees, not implementation details.

Contract guarantees verified:
1. write() MUST return ArtifactDescriptor
2. ArtifactDescriptor MUST have content_hash (SHA-256, 64 hex chars)
3. ArtifactDescriptor MUST have size_bytes
4. flush() MUST be idempotent
5. close() MUST be idempotent
6. Same data MUST produce same content_hash (determinism for audit)

Usage:
    Create a subclass with fixtures providing:
    - sink_factory: A callable that returns a fresh sink instance
    - sample_rows: A list of row dicts to write
    - ctx: A PluginContext for the test

    class TestMySinkContract(SinkContractTestBase):
        @pytest.fixture
        def sink_factory(self, tmp_path):
            def factory():
                return MySink({"path": str(tmp_path / "output.csv"), ...})
            return factory

        @pytest.fixture
        def sample_rows(self):
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import (
    ArtifactDescriptor,
    Determinism,
    OutputValidationResult,
    PluginSchema,
    RowDiversion,
    SinkWriteResult,
)
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.base import BaseSink
from tests.fixtures.factories import make_context

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol


class _SinkContractRowSchema(PluginSchema):
    id: int
    name: str


class _SinkContractExemplarSink(BaseSink):
    """Concrete sink proving this contract base is executable in isolation."""

    name = "sink_contract_exemplar"
    input_schema: type[PluginSchema] = _SinkContractRowSchema
    idempotent = True
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:sink-contract-exemplar"
    supports_resume = True
    declared_required_fields = frozenset({"id", "name"})

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._on_write_failure = "discard"
        self._written_batches: list[list[dict[str, Any]]] = []
        self._resume_configured = False
        self._closed = False

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        _ = ctx
        self._written_batches.append([dict(row) for row in rows])
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path="/virtual/sink-contract-exemplar",
                content_hash=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            ),
            diversions=self._get_diversions(),
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def configure_for_resume(self) -> None:
        self._resume_configured = True


def _require_sample_rows(sample_rows: list[dict[str, Any]]) -> None:
    assert sample_rows, (
        "Sink contract fixtures must include at least one row; otherwise write() "
        "and hash contract tests can pass without exercising a meaningful payload."
    )
    for row in sample_rows:
        assert isinstance(row, dict)
        assert row, "Sink contract fixture rows must be non-empty so hash tests can mutate payload content"


def _write_rows(
    sink: SinkProtocol,
    sample_rows: list[dict[str, Any]],
    ctx: PluginContext,
) -> SinkWriteResult:
    _require_sample_rows(sample_rows)
    result = sink.write(sample_rows, ctx)
    assert isinstance(result, SinkWriteResult), f"write() returned {type(result).__name__}, expected SinkWriteResult"
    assert isinstance(result.artifact, ArtifactDescriptor), "SinkWriteResult.artifact must be an ArtifactDescriptor"
    assert isinstance(result.diversions, tuple), "SinkWriteResult.diversions must be an immutable tuple"
    assert all(isinstance(diversion, RowDiversion) for diversion in result.diversions)
    return result


def _sink_boundary_snapshot(sink: SinkProtocol) -> dict[str, Any]:
    return {
        "config": dict(sink.config),
        "declared_required_fields": sink.declared_required_fields,
        "input_schema": sink.input_schema,
        "name": sink.name,
        "needs_resume_field_resolution": sink.needs_resume_field_resolution,
        "node_id": sink.node_id,
        "plugin_version": sink.plugin_version,
        "source_file_hash": sink.source_file_hash,
        "supports_resume": sink.supports_resume,
        "write_failure_route": sink._on_write_failure,
    }


def _local_file_artifact_snapshot(result: SinkWriteResult) -> tuple[int, str] | None:
    artifact = result.artifact
    if artifact.artifact_type != "file" or artifact.path_or_uri is None:
        return None
    path = Path(artifact.path_or_uri)
    if not path.exists():
        return None
    data = path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()


class SinkContractTestBase(ABC):
    """Abstract base class for sink contract verification.

    Subclasses must provide fixtures for:
    - sink_factory: A callable that returns a fresh sink instance
    - sample_rows: A list of row dicts to write
    - ctx: A PluginContext for the test
    """

    @pytest.fixture
    @abstractmethod
    def sink_factory(self) -> Callable[[], SinkProtocol]:
        """Provide a factory that creates fresh sink instances."""
        raise NotImplementedError

    @pytest.fixture
    def sink(self, sink_factory: Callable[[], SinkProtocol]) -> SinkProtocol:
        """Provide a configured sink instance."""
        return sink_factory()

    @pytest.fixture
    @abstractmethod
    def sample_rows(self) -> list[dict[str, Any]]:
        """Provide sample rows to write."""
        raise NotImplementedError

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Provide a PluginContext for testing."""
        return make_context(run_id="test-run-001", node_id="test-sink")

    # =========================================================================
    # Protocol Attribute Contracts
    # =========================================================================

    def test_sink_engine_identity_surface_is_coherent(self, sink: SinkProtocol) -> None:
        """Contract: engine-facing identity and audit metadata MUST be well formed."""
        assert isinstance(sink.name, str)
        assert len(sink.name) > 0
        assert isinstance(sink.input_schema, type)
        assert issubclass(sink.input_schema, PluginSchema)
        assert isinstance(sink.determinism, Determinism)
        assert isinstance(sink.plugin_version, str)
        assert isinstance(sink.config, dict)
        assert sink.node_id is None or isinstance(sink.node_id, str)
        assert sink.source_file_hash is None or isinstance(sink.source_file_hash, str)

    def test_sink_durability_and_routing_surface_is_coherent(self, sink: SinkProtocol) -> None:
        """Contract: durability, resume, and failsink surfaces MUST be normalized."""
        assert isinstance(sink.idempotent, bool)
        assert isinstance(sink.supports_resume, bool)
        assert isinstance(sink.declared_required_fields, frozenset)
        assert all(isinstance(field, str) for field in sink.declared_required_fields)
        assert sink._on_write_failure is None or isinstance(sink._on_write_failure, str)

    def test_sink_can_reset_diversion_log(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST expose diversion-log reset for SinkExecutor."""
        sink._reset_diversion_log()

    def test_sink_exposes_config_model(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST expose its optional config validation model."""
        config_model = sink.get_config_model(sink.config)
        if config_model is None:
            return
        assert isinstance(config_model.model_json_schema(), dict)

    def test_sink_exposes_config_schema(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST expose complete JSON Schema for plugin discovery."""
        assert isinstance(sink.get_config_schema(), dict)

    # =========================================================================
    # write() Method Contracts
    # =========================================================================

    def test_write_returns_artifact_descriptor(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: write() MUST return SinkWriteResult with ArtifactDescriptor."""
        _write_rows(sink, sample_rows, ctx)

    def test_artifact_has_content_hash(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have content_hash (audit integrity!)."""
        result = _write_rows(sink, sample_rows, ctx)

        assert result.artifact.content_hash is not None, "ArtifactDescriptor.content_hash is None - REQUIRED for audit integrity"
        assert isinstance(result.artifact.content_hash, str)

    def test_content_hash_is_sha256_hex(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: content_hash MUST be a valid SHA-256 hex string (64 chars)."""
        result = _write_rows(sink, sample_rows, ctx)

        assert len(result.artifact.content_hash) == 64, (
            f"content_hash has {len(result.artifact.content_hash)} chars, expected 64 for SHA-256"
        )
        assert all(c in "0123456789abcdef" for c in result.artifact.content_hash), (
            f"content_hash contains invalid hex chars: {result.artifact.content_hash}"
        )

    def test_artifact_has_size_bytes(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have size_bytes."""
        result = _write_rows(sink, sample_rows, ctx)

        assert result.artifact.size_bytes is not None, "ArtifactDescriptor.size_bytes is None - REQUIRED for verification"
        assert isinstance(result.artifact.size_bytes, int)
        assert result.artifact.size_bytes >= 0

    def test_artifact_has_artifact_type(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have artifact_type."""
        result = _write_rows(sink, sample_rows, ctx)

        assert result.artifact.artifact_type is not None
        assert result.artifact.artifact_type in ("file", "database", "webhook")

    def test_artifact_has_path_or_uri(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have path_or_uri."""
        result = _write_rows(sink, sample_rows, ctx)

        assert result.artifact.path_or_uri is not None
        assert isinstance(result.artifact.path_or_uri, str)
        assert len(result.artifact.path_or_uri) > 0

    def test_write_result_has_diversion_tuple(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: write() MUST return immutable diversion records."""
        result = _write_rows(sink, sample_rows, ctx)
        assert isinstance(result.diversions, tuple)
        assert all(isinstance(diversion, RowDiversion) for diversion in result.diversions)

    # =========================================================================
    # Empty Batch Contracts
    # =========================================================================

    def test_write_empty_batch_returns_descriptor(
        self,
        sink: SinkProtocol,
        ctx: PluginContext,
    ) -> None:
        """Contract: write([]) MUST return a valid ArtifactDescriptor."""
        result = sink.write([], ctx)

        assert isinstance(result, SinkWriteResult)
        assert isinstance(result.artifact, ArtifactDescriptor)
        assert result.artifact.content_hash is not None
        assert result.artifact.size_bytes is not None
        assert result.artifact.size_bytes >= 0
        assert isinstance(result.diversions, tuple)

    # =========================================================================
    # Resume Contracts
    # =========================================================================

    def test_validate_output_target_returns_result(self, sink: SinkProtocol) -> None:
        """Contract: validate_output_target() MUST return OutputValidationResult."""
        assert isinstance(sink.validate_output_target(), OutputValidationResult)

    def test_resume_field_resolution_surface(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST expose resume field-resolution hooks."""
        assert isinstance(sink.needs_resume_field_resolution, bool)
        sink.set_resume_field_resolution({"Original ID": "id"})

    def test_resume_configuration_for_resumable_sinks(self, sink: SinkProtocol) -> None:
        """Contract: Sinks that claim resume support MUST configure without raising."""
        if sink.supports_resume:
            sink.configure_for_resume()

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_flush_is_idempotent(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: repeated flush() preserves sink metadata and written artifacts."""
        result = _write_rows(sink, sample_rows, ctx)
        boundary_snapshot = _sink_boundary_snapshot(sink)
        artifact_snapshot = _local_file_artifact_snapshot(result)

        assert sink.flush() is None
        assert sink.flush() is None
        assert sink.flush() is None

        assert _sink_boundary_snapshot(sink) == boundary_snapshot
        if artifact_snapshot is not None:
            assert _local_file_artifact_snapshot(result) == artifact_snapshot

    def test_close_is_idempotent(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: repeated close() preserves sink metadata and written artifacts."""
        result = _write_rows(sink, sample_rows, ctx)
        assert sink.flush() is None
        boundary_snapshot = _sink_boundary_snapshot(sink)
        artifact_snapshot = _local_file_artifact_snapshot(result)

        assert sink.close() is None
        assert sink.close() is None
        assert sink.close() is None

        assert _sink_boundary_snapshot(sink) == boundary_snapshot
        if artifact_snapshot is not None:
            assert _local_file_artifact_snapshot(result) == artifact_snapshot

    def test_on_start_preserves_sink_contract_metadata(
        self,
        sink: SinkProtocol,
        ctx: PluginContext,
    ) -> None:
        """Contract: on_start() MUST preserve sink contract metadata."""
        boundary_snapshot = _sink_boundary_snapshot(sink)

        assert sink.on_start(ctx) is None

        assert _sink_boundary_snapshot(sink) == boundary_snapshot

    def test_on_complete_preserves_sink_contract_metadata(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: on_complete() MUST preserve sink metadata and written artifacts."""
        result = _write_rows(sink, sample_rows, ctx)
        boundary_snapshot = _sink_boundary_snapshot(sink)
        artifact_snapshot = _local_file_artifact_snapshot(result)

        assert sink.on_complete(ctx) is None

        assert _sink_boundary_snapshot(sink) == boundary_snapshot
        if artifact_snapshot is not None:
            assert _local_file_artifact_snapshot(result) == artifact_snapshot


class SinkDeterminismContractTestBase(SinkContractTestBase):
    """Extended base for testing sink content hash determinism.

    Critical for audit integrity: same data MUST produce same content_hash.

    Subclasses must provide a sink_factory fixture that returns fresh instances.
    """

    def test_same_data_same_hash(
        self,
        sink_factory: Callable[[], SinkProtocol],
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: Same data MUST produce same content_hash (audit integrity!).

        This is THE critical property for sinks. If this fails, the audit
        trail cannot be verified because hashes won't match.
        """
        first_sink = sink_factory()
        first_result = _write_rows(first_sink, sample_rows, ctx)
        first_hash = first_result.artifact.content_hash
        first_sink.close()

        second_sink = sink_factory()
        second_result = _write_rows(second_sink, sample_rows, ctx)
        second_hash = second_result.artifact.content_hash
        second_sink.close()

        assert first_hash == second_hash, (
            f"Same data produced different hashes - audit integrity compromised! first={first_hash}, second={second_hash}"
        )

    def test_content_hash_changes_with_data(
        self,
        sink_factory: Callable[[], SinkProtocol],
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: Different data SHOULD produce different content_hash.

        Note: Collisions are theoretically possible but astronomically unlikely.
        This test verifies the hash is computed from actual content.
        """
        first_sink = sink_factory()
        first_result = _write_rows(first_sink, sample_rows, ctx)
        first_hash = first_result.artifact.content_hash
        first_sink.close()

        _require_sample_rows(sample_rows)
        modified_rows = [row.copy() for row in sample_rows]
        first_key = next(iter(modified_rows[0].keys()))
        modified_rows[0][first_key] = "MODIFIED_VALUE_FOR_HASH_TEST"

        second_sink = sink_factory()
        second_result = _write_rows(second_sink, modified_rows, ctx)
        second_hash = second_result.artifact.content_hash
        second_sink.close()

        assert first_hash != second_hash, f"Different data produced same hash - hash not computed from content! hash={first_hash}"


class TestSinkProtocolContractBase(SinkDeterminismContractTestBase):
    """Self-test for the reusable sink contract base."""

    @pytest.fixture
    def sink_factory(self) -> Callable[[], SinkProtocol]:
        return lambda: _SinkContractExemplarSink({})

    @pytest.fixture
    def sample_rows(self) -> list[dict[str, Any]]:
        return [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
