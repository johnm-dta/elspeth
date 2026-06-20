# tests/unit/contracts/source_contracts/test_source_protocol.py
"""Contract tests for Source plugins.

These tests verify that source implementations honor the SourceProtocol contract.
They test interface guarantees, not implementation details.

Contract guarantees verified:
1. load() MUST return an iterator and yield SourceRow objects (not raw dicts)
2. Contract fixtures MUST yield at least one row and one valid row
3. Valid rows MUST have data dicts and SchemaContract instances
4. Quarantined rows MUST have error and destination, and no SchemaContract
5. close() MUST be idempotent (safe to call multiple times)
6. Lifecycle hooks on_start/on_complete MUST not raise
7. output_schema MUST be a PluginSchema subclass
8. Current SourceProtocol audit/config/schema surfaces MUST be exposed

Usage:
    Create a subclass with fixtures providing:
    - source: The source plugin instance
    - ctx: A PluginContext for the test

    class TestMySourceContract(SourceContractTestBase):
        @pytest.fixture
        def source(self, tmp_path):
            return MySource({"path": str(tmp_path / "data.csv"), ...})

        @pytest.fixture
        def source_data(self, tmp_path):
            # Create test data file
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import pytest

from elspeth.contracts import Determinism, PluginSchema, SchemaContract, SourceRow, create_contract_from_config
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.infrastructure.base import BaseSource
from tests.fixtures.factories import make_context

if TYPE_CHECKING:
    from elspeth.contracts import SourceProtocol


class _SourceContractRowSchema(PluginSchema):
    id: int
    name: str


class _SourceContractExemplarSource(BaseSource):
    """Concrete source proving this contract base is executable in isolation."""

    name = "source_contract_exemplar"
    output_schema: type[PluginSchema] = _SourceContractRowSchema
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:source-contract-exemplar"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._on_validation_failure = "discard"
        self.on_success = "output"
        self._rows: tuple[dict[str, Any], ...] = (
            {"id": 1, "name": "alpha"},
            {"id": 2, "name": "beta"},
        )
        schema_config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        )
        self._initialize_declared_guaranteed_fields(schema_config)
        self.set_schema_contract(create_contract_from_config(schema_config))

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        contract = self.require_schema_contract()
        for source_row_index, row in enumerate(self._rows):
            yield SourceRow.valid(dict(row), contract=contract, source_row_index=source_row_index)

    def close(self) -> None:
        pass


def _collect_rows(source: SourceProtocol, ctx: PluginContext) -> list[SourceRow]:
    rows = list(source.load(ctx))
    assert rows, (
        "Source contract fixtures must yield at least one row; otherwise row-shape "
        "contract tests can pass without exercising SourceRow emission."
    )
    for row in rows:
        assert isinstance(row, SourceRow), (
            f"load() yielded {type(row).__name__}, expected SourceRow. "
            "Sources must wrap rows with SourceRow.valid() or SourceRow.quarantined()."
        )
    return rows


def _source_boundary_snapshot(
    source: SourceProtocol,
) -> tuple[
    frozenset[str],
    dict[str, Any] | None,
    tuple[dict[str, str], str | None] | None,
    str,
    str,
]:
    """Capture the source contract metadata that must survive lifecycle hooks."""
    contract = source.get_schema_contract()
    field_resolution = source.get_field_resolution()
    return (
        frozenset(source.declared_guaranteed_fields),
        contract.to_checkpoint_format() if contract is not None else None,
        (dict(field_resolution[0]), field_resolution[1]) if field_resolution is not None else None,
        source._on_validation_failure,
        source.on_success,
    )


class SourceContractTestBase(ABC):
    """Abstract base class for source contract verification.

    Subclasses must provide fixtures for:
    - source: The source plugin instance to test
    - ctx: A PluginContext for the test
    """

    @pytest.fixture
    @abstractmethod
    def source(self) -> SourceProtocol:
        """Provide a configured source instance."""
        raise NotImplementedError

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Provide a PluginContext for testing."""
        return make_context(run_id="test-run-001", node_id="test-source")

    # =========================================================================
    # Protocol Attribute Contracts
    # =========================================================================

    def test_source_engine_identity_surface_is_coherent(self, source: SourceProtocol) -> None:
        """Contract: engine-facing identity and audit metadata MUST be well formed."""
        assert isinstance(source.name, str)
        assert len(source.name) > 0
        assert isinstance(source.output_schema, type)
        assert issubclass(source.output_schema, PluginSchema)
        assert isinstance(source.determinism, Determinism)
        assert isinstance(source.plugin_version, str)
        assert isinstance(source.config, dict)
        assert source.node_id is None or isinstance(source.node_id, str)
        assert source.source_file_hash is None or isinstance(source.source_file_hash, str)

    def test_source_routing_and_declaration_surface_is_coherent(self, source: SourceProtocol) -> None:
        """Contract: source routing and guaranteed-field surfaces MUST be normalized."""
        assert isinstance(source._on_validation_failure, str)
        assert source._on_validation_failure
        assert isinstance(source.on_success, str)
        assert source.on_success
        assert isinstance(source.declared_guaranteed_fields, frozenset)
        assert all(isinstance(field, str) for field in source.declared_guaranteed_fields)

    def test_source_exposes_field_resolution(self, source: SourceProtocol) -> None:
        """Contract: Source MUST expose optional field-normalization audit metadata."""
        resolution = source.get_field_resolution()
        if resolution is None:
            return
        mapping, version = resolution
        assert isinstance(mapping, Mapping)
        assert all(isinstance(original, str) for original in mapping)
        assert all(isinstance(normalized, str) for normalized in mapping.values())
        assert version is None or isinstance(version, str)

    def test_source_exposes_schema_contract(self, source: SourceProtocol) -> None:
        """Contract: Source MUST expose its current schema contract, when available."""
        contract = source.get_schema_contract()
        assert contract is None or isinstance(contract, SchemaContract)

    def test_source_exposes_config_model(self, source: SourceProtocol) -> None:
        """Contract: Source MUST expose its optional config validation model."""
        config_model = source.get_config_model(source.config)
        if config_model is None:
            return
        assert isinstance(config_model.model_json_schema(), dict)

    def test_source_exposes_config_schema(self, source: SourceProtocol) -> None:
        """Contract: Source MUST expose complete JSON Schema for plugin discovery."""
        assert isinstance(source.get_config_schema(), dict)

    # =========================================================================
    # load() Method Contracts
    # =========================================================================

    def test_load_returns_iterator(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: load() MUST return an iterator."""
        result = source.load(ctx)
        assert isinstance(result, Iterator)
        assert iter(result) is result

    def test_load_yields_source_rows(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: load() MUST yield SourceRow objects, not raw dicts."""
        _collect_rows(source, ctx)

    def test_valid_rows_have_data_and_contract(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Valid SourceRows MUST have data dict and schema contract."""
        valid_rows = [row for row in _collect_rows(source, ctx) if not row.is_quarantined]
        assert valid_rows, "Source contract fixtures must include at least one valid row"
        for row in valid_rows:
            assert row.row is not None, "Valid SourceRow has None data"
            assert isinstance(row.row, dict), f"Valid SourceRow.row is {type(row.row).__name__}, expected dict"
            assert isinstance(row.contract, SchemaContract), "Valid SourceRow must carry its schema contract"

    def test_quarantined_rows_have_error(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Quarantined SourceRows MUST have error message."""
        for row in _collect_rows(source, ctx):
            if row.is_quarantined:
                assert row.quarantine_error is not None, "Quarantined SourceRow has None error"
                assert isinstance(row.quarantine_error, str), f"quarantine_error is {type(row.quarantine_error).__name__}, expected str"

    def test_quarantined_rows_have_destination(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Quarantined SourceRows MUST have destination."""
        for row in _collect_rows(source, ctx):
            if row.is_quarantined:
                assert row.quarantine_destination is not None, "Quarantined SourceRow has None destination"
                assert isinstance(row.quarantine_destination, str), (
                    f"quarantine_destination is {type(row.quarantine_destination).__name__}, expected str"
                )

    def test_quarantined_rows_do_not_carry_schema_contract(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Quarantined SourceRows MUST be contractless."""
        for row in _collect_rows(source, ctx):
            if row.is_quarantined:
                assert row.contract is None, "Quarantined SourceRow must not carry a schema contract"

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_close_is_idempotent(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: close() MUST be repeatable without erasing source contract metadata."""
        _collect_rows(source, ctx)
        before_close = _source_boundary_snapshot(source)

        assert source.close() is None
        assert _source_boundary_snapshot(source) == before_close

        assert source.close() is None
        assert source.close() is None
        assert _source_boundary_snapshot(source) == before_close

    def test_on_start_preserves_source_contract_metadata(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: on_start() MUST not erase source boundary contract metadata."""
        before_start = _source_boundary_snapshot(source)

        assert source.on_start(ctx) is None
        assert _source_boundary_snapshot(source) == before_start

    def test_on_complete_preserves_source_contract_metadata(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: on_complete() MUST not erase the completed source boundary contract."""
        _collect_rows(source, ctx)
        before_complete = _source_boundary_snapshot(source)

        assert source.on_complete(ctx) is None
        assert _source_boundary_snapshot(source) == before_complete


# =============================================================================
# Property-based contract verification using Hypothesis
# =============================================================================


class SourceContractPropertyTestBase(SourceContractTestBase):
    """Extended base with property-based contract verification.

    Adds property tests that verify contracts hold for multiple loads.
    """

    def test_multiple_loads_yield_consistent_count(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Property: Multiple loads should yield same row count (determinism check).

        Note: This only applies to DETERMINISTIC sources. Non-deterministic
        sources (e.g., live API feeds) may return different counts.
        """
        if source.determinism == Determinism.DETERMINISTIC:
            count1 = sum(1 for _ in source.load(ctx))
            count2 = sum(1 for _ in source.load(ctx))
            assert count1 == count2, f"Deterministic source returned different counts: {count1} vs {count2}"

    def test_load_exhaust_does_not_raise(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Property: Fully exhausting load() iterator should not raise."""
        # This catches iterator issues like premature StopIteration
        rows = list(source.load(ctx))
        assert isinstance(rows, list)


class TestSourceProtocolContractBase(SourceContractPropertyTestBase):
    """Self-test for the reusable source contract base."""

    @pytest.fixture
    def source(self) -> SourceProtocol:
        return _SourceContractExemplarSource({})
