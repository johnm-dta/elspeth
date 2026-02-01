# tests/contracts/sink_contracts/test_sink_protocol.py
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

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.plugins.context import PluginContext

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SinkProtocol


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
        """Provide a configured sink instance (for backwards compatibility)."""
        return sink_factory()

    @pytest.fixture
    @abstractmethod
    def sample_rows(self) -> list[dict[str, Any]]:
        """Provide sample rows to write."""
        raise NotImplementedError

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Provide a PluginContext for testing."""
        return PluginContext(
            run_id="test-run-001",
            config={},
            node_id="test-sink",
            plugin_name="test",
        )

    # =========================================================================
    # Protocol Attribute Contracts
    # =========================================================================

    def test_sink_has_name(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST have a 'name' attribute."""
        assert isinstance(sink.name, str)
        assert len(sink.name) > 0

    def test_sink_has_input_schema(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST have an 'input_schema' that is a PluginSchema subclass."""
        assert isinstance(sink.input_schema, type)
        assert issubclass(sink.input_schema, PluginSchema)

    def test_sink_has_determinism(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST have a 'determinism' attribute."""
        assert isinstance(sink.determinism, Determinism)

    def test_sink_has_plugin_version(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST have a 'plugin_version' attribute."""
        assert isinstance(sink.plugin_version, str)

    def test_sink_has_idempotent_flag(self, sink: SinkProtocol) -> None:
        """Contract: Sink MUST have an 'idempotent' attribute."""
        assert isinstance(sink.idempotent, bool)

    # =========================================================================
    # write() Method Contracts
    # =========================================================================

    def test_write_returns_artifact_descriptor(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: write() MUST return ArtifactDescriptor."""
        result = sink.write(sample_rows, ctx)
        assert isinstance(result, ArtifactDescriptor), f"write() returned {type(result).__name__}, expected ArtifactDescriptor"

    def test_artifact_has_content_hash(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have content_hash (audit integrity!)."""
        result = sink.write(sample_rows, ctx)

        assert result.content_hash is not None, "ArtifactDescriptor.content_hash is None - REQUIRED for audit integrity"
        assert isinstance(result.content_hash, str)

    def test_content_hash_is_sha256_hex(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: content_hash MUST be a valid SHA-256 hex string (64 chars)."""
        result = sink.write(sample_rows, ctx)

        assert len(result.content_hash) == 64, f"content_hash has {len(result.content_hash)} chars, expected 64 for SHA-256"
        assert all(c in "0123456789abcdef" for c in result.content_hash), f"content_hash contains invalid hex chars: {result.content_hash}"

    def test_artifact_has_size_bytes(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have size_bytes."""
        result = sink.write(sample_rows, ctx)

        assert result.size_bytes is not None, "ArtifactDescriptor.size_bytes is None - REQUIRED for verification"
        assert isinstance(result.size_bytes, int)
        assert result.size_bytes >= 0

    def test_artifact_has_artifact_type(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have artifact_type."""
        result = sink.write(sample_rows, ctx)

        assert result.artifact_type is not None
        assert result.artifact_type in ("file", "database", "webhook")

    def test_artifact_has_path_or_uri(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: ArtifactDescriptor MUST have path_or_uri."""
        result = sink.write(sample_rows, ctx)

        assert result.path_or_uri is not None
        assert isinstance(result.path_or_uri, str)
        assert len(result.path_or_uri) > 0

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

        assert isinstance(result, ArtifactDescriptor)
        assert result.content_hash is not None
        assert result.size_bytes is not None
        assert result.size_bytes >= 0

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_flush_is_idempotent(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: flush() MUST be safe to call multiple times."""
        sink.write(sample_rows, ctx)

        sink.flush()
        sink.flush()
        sink.flush()

    def test_close_is_idempotent(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: close() MUST be safe to call multiple times."""
        sink.write(sample_rows, ctx)
        sink.flush()

        sink.close()
        sink.close()
        sink.close()

    def test_on_start_does_not_raise(
        self,
        sink: SinkProtocol,
        ctx: PluginContext,
    ) -> None:
        """Contract: on_start() lifecycle hook MUST not raise."""
        sink.on_start(ctx)

    def test_on_complete_does_not_raise(
        self,
        sink: SinkProtocol,
        sample_rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> None:
        """Contract: on_complete() lifecycle hook MUST not raise."""
        sink.write(sample_rows, ctx)
        sink.on_complete(ctx)


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
        first_result = first_sink.write(sample_rows, ctx)
        first_hash = first_result.content_hash
        first_sink.close()

        second_sink = sink_factory()
        second_result = second_sink.write(sample_rows, ctx)
        second_hash = second_result.content_hash
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
        first_result = first_sink.write(sample_rows, ctx)
        first_hash = first_result.content_hash
        first_sink.close()

        modified_rows = [dict(row) for row in sample_rows]
        if modified_rows:
            first_key = next(iter(modified_rows[0].keys()))
            modified_rows[0][first_key] = "MODIFIED_VALUE_FOR_HASH_TEST"

        second_sink = sink_factory()
        second_result = second_sink.write(modified_rows, ctx)
        second_hash = second_result.content_hash
        second_sink.close()

        assert first_hash != second_hash, f"Different data produced same hash - hash not computed from content! hash={first_hash}"


# =============================================================================
# Protocol Attribute Tests (standalone - don't need sink instance)
# =============================================================================


def test_sink_protocol_requires_supports_resume() -> None:
    """SinkProtocol must declare supports_resume attribute."""
    from elspeth.plugins.protocols import SinkProtocol

    # Verify protocol has supports_resume in annotations
    assert "supports_resume" in SinkProtocol.__annotations__, (
        "SinkProtocol must declare 'supports_resume: bool' attribute for resume capability"
    )


def test_sink_protocol_requires_configure_for_resume() -> None:
    """SinkProtocol must declare configure_for_resume method."""
    from elspeth.plugins.protocols import SinkProtocol

    # Verify protocol has configure_for_resume method
    assert hasattr(SinkProtocol, "configure_for_resume"), "SinkProtocol must declare 'configure_for_resume()' method for resume capability"
    assert callable(SinkProtocol.configure_for_resume), "SinkProtocol.configure_for_resume must be a callable method"
