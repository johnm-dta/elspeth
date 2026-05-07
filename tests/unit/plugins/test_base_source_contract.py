# tests/plugins/test_base_source_contract.py
"""Tests for BaseSource schema contract tracking."""

from collections.abc import Iterator
from typing import Any

from elspeth.contracts import SourceRow
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.testing import make_contract, make_field


class StubSource(BaseSource):
    """Stub source implementation for testing."""

    name = "stub"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._on_validation_failure = "quarantine"
        # Manually set output_schema for protocol compliance
        from elspeth.contracts.data import PluginSchema

        self.output_schema = PluginSchema

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        yield SourceRow.valid({"id": 1}, contract=make_contract({"id": 1}))

    def close(self) -> None:
        pass


class TestBaseSourceContract:
    """Test contract tracking on BaseSource."""

    def test_declared_guaranteed_fields_defaults_empty(self) -> None:
        """BaseSource starts with an inert empty declaration surface."""
        source = StubSource({})
        assert source.declared_guaranteed_fields == frozenset()

    def test_get_schema_contract_returns_none_before_load(self) -> None:
        """get_schema_contract() returns None before load()."""
        source = StubSource({})
        assert source.get_schema_contract() is None

    def test_set_schema_contract(self) -> None:
        """set_schema_contract() stores contract for retrieval."""
        source = StubSource({})
        contract = SchemaContract(
            mode="FIXED",
            fields=(make_field("id", int, original_name="id", required=True, source="declared"),),
            locked=True,
        )

        source.set_schema_contract(contract)

        assert source.get_schema_contract() is contract

    def test_update_schema_contract(self) -> None:
        """Contract can be updated (for first-row locking)."""
        source = StubSource({})

        # Initial unlocked contract
        initial = SchemaContract(mode="OBSERVED", fields=(), locked=False)
        source.set_schema_contract(initial)

        # Lock it
        locked = initial.with_field("id", "id", 1).with_locked()
        source.set_schema_contract(locked)

        retrieved = source.get_schema_contract()
        assert retrieved is locked
        assert retrieved is not None  # for type checker
        assert retrieved.locked is True
