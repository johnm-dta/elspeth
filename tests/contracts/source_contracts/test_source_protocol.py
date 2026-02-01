# tests/contracts/source_contracts/test_source_protocol.py
"""Contract tests for Source plugins.

These tests verify that source implementations honor the SourceProtocol contract.
They test interface guarantees, not implementation details.

Contract guarantees verified:
1. load() MUST yield SourceRow objects (not raw dicts)
2. Valid rows MUST have non-None data with dict type
3. Quarantined rows MUST have error and destination
4. close() MUST be idempotent (safe to call multiple times)
5. Lifecycle hooks on_start/on_complete MUST not raise
6. output_schema MUST be a PluginSchema subclass

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
from typing import TYPE_CHECKING

import pytest

from elspeth.contracts import Determinism, PluginSchema, SourceRow
from elspeth.plugins.context import PluginContext

if TYPE_CHECKING:
    from elspeth.plugins.protocols import SourceProtocol


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
        return PluginContext(
            run_id="test-run-001",
            config={},
            node_id="test-source",
            plugin_name="test",
        )

    # =========================================================================
    # Protocol Attribute Contracts
    # =========================================================================

    def test_source_has_name(self, source: SourceProtocol) -> None:
        """Contract: Source MUST have a 'name' attribute."""
        assert isinstance(source.name, str)
        assert len(source.name) > 0

    def test_source_has_output_schema(self, source: SourceProtocol) -> None:
        """Contract: Source MUST have an 'output_schema' that is a PluginSchema subclass."""
        assert isinstance(source.output_schema, type)
        assert issubclass(source.output_schema, PluginSchema)

    def test_source_has_determinism(self, source: SourceProtocol) -> None:
        """Contract: Source MUST have a 'determinism' attribute."""
        assert isinstance(source.determinism, Determinism)

    def test_source_has_plugin_version(self, source: SourceProtocol) -> None:
        """Contract: Source MUST have a 'plugin_version' attribute."""
        assert isinstance(source.plugin_version, str)

    # =========================================================================
    # load() Method Contracts
    # =========================================================================

    def test_load_returns_iterator(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: load() MUST return an iterator."""
        result = source.load(ctx)
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_load_yields_source_rows(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: load() MUST yield SourceRow objects, not raw dicts."""
        for row in source.load(ctx):
            assert isinstance(row, SourceRow), (
                f"load() yielded {type(row).__name__}, expected SourceRow. "
                "Sources must wrap rows with SourceRow.valid() or SourceRow.quarantined()."
            )

    def test_valid_rows_have_data(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Valid SourceRows MUST have non-None data dict."""
        for row in source.load(ctx):
            if not row.is_quarantined:
                assert row.row is not None, "Valid SourceRow has None data"
                assert isinstance(row.row, dict), f"Valid SourceRow.row is {type(row.row).__name__}, expected dict"

    def test_quarantined_rows_have_error(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Quarantined SourceRows MUST have error message."""
        for row in source.load(ctx):
            if row.is_quarantined:
                assert row.quarantine_error is not None, "Quarantined SourceRow has None error"
                assert isinstance(row.quarantine_error, str), f"quarantine_error is {type(row.quarantine_error).__name__}, expected str"

    def test_quarantined_rows_have_destination(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: Quarantined SourceRows MUST have destination."""
        for row in source.load(ctx):
            if row.is_quarantined:
                assert row.quarantine_destination is not None, "Quarantined SourceRow has None destination"
                assert isinstance(row.quarantine_destination, str), (
                    f"quarantine_destination is {type(row.quarantine_destination).__name__}, expected str"
                )

    # =========================================================================
    # Lifecycle Contracts
    # =========================================================================

    def test_close_is_idempotent(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: close() MUST be safe to call multiple times."""
        # Exhaust the iterator
        list(source.load(ctx))

        # close() should not raise on first call
        source.close()

        # close() should not raise on subsequent calls (idempotent)
        source.close()
        source.close()

    def test_on_start_does_not_raise(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: on_start() lifecycle hook MUST not raise."""
        source.on_start(ctx)

    def test_on_complete_does_not_raise(self, source: SourceProtocol, ctx: PluginContext) -> None:
        """Contract: on_complete() lifecycle hook MUST not raise."""
        list(source.load(ctx))
        source.on_complete(ctx)


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
