# tests/plugins/test_base_sink_contract.py
"""Tests for BaseSink output contract tracking."""

from typing import Any

from elspeth.contracts import ArtifactDescriptor
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.testing import make_field


class StubSink(BaseSink):
    """Stub sink implementation for testing."""

    name = "stub"
    input_schema = None  # type: ignore
    _on_write_failure: str | None = "discard"

    def write(
        self,
        rows: list[dict[str, Any]],
        ctx: SinkContext,
    ) -> SinkWriteResult:
        return SinkWriteResult(
            artifact=ArtifactDescriptor(
                artifact_type="file",
                path_or_uri="/test/output.csv",
                content_hash="abc123",
                size_bytes=0,
            )
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TestBaseSinkContract:
    """Test contract tracking on BaseSink."""

    def test_get_output_contract_returns_none_initially(self) -> None:
        """get_output_contract() returns None when not set."""
        sink = StubSink({})
        assert sink.get_output_contract() is None

    def test_set_output_contract(self) -> None:
        """set_output_contract() stores contract for retrieval."""
        sink = StubSink({})
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="id", required=True, source="declared"),),
            locked=True,
        )

        sink.set_output_contract(contract)

        assert sink.get_output_contract() is contract

    def test_update_output_contract(self) -> None:
        """Contract can be updated."""
        sink = StubSink({})

        # Initial contract
        initial = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="id", required=True, source="declared"),),
            locked=True,
        )
        sink.set_output_contract(initial)

        # Update with different contract
        updated = SchemaContract(
            mode="FIXED",
            fields=(
                make_field("id", int, original_name="id", required=True, source="declared"),
                make_field("name", str, original_name="name", required=True, source="declared"),
            ),
            locked=True,
        )
        sink.set_output_contract(updated)

        retrieved = sink.get_output_contract()
        assert retrieved is updated
        assert retrieved is not initial

    def test_class_attribute_default_is_none(self) -> None:
        """Class attribute _output_contract defaults to None."""
        assert BaseSink._output_contract is None
