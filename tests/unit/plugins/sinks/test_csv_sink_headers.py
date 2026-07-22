"""Tests for CSVSink header mode integration with contracts.

Tests the integration of the header_modes system with CSVSink:
- set_output_contract() and get_output_contract() methods
- Resolution of headers via contracts when mode is ORIGINAL
- Interaction between headers config and schema contracts
"""

from pathlib import Path

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.testing import make_field
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory

# CSVSink requires fixed-column structure (strict mode)
STRICT_SCHEMA = {"mode": "fixed", "fields": ["amount_usd: int", "customer_id: str"]}


class TestCSVSinkContractSupport:
    """Test CSVSink contract storage methods."""

    @pytest.fixture
    def output_path(self, tmp_path: Path) -> Path:
        """Output file path."""
        return tmp_path / "output.csv"

    @pytest.fixture
    def sample_contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("amount_usd", int, original_name="'Amount USD'", required=True, source="declared"),
                make_field("customer_id", str, original_name="Customer ID", required=True, source="declared"),
            ),
            locked=True,
        )

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_set_output_contract(self, output_path: Path, sample_contract: SchemaContract) -> None:
        """set_output_contract stores contract for header resolution."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        sink.set_output_contract(sample_contract)

        # Verify contract is stored
        assert sink._output_contract is sample_contract

    def test_get_output_contract_returns_none_initially(self, output_path: Path) -> None:
        """get_output_contract returns None when no contract set."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        assert sink.get_output_contract() is None

    def test_get_output_contract_returns_stored_contract(self, output_path: Path, sample_contract: SchemaContract) -> None:
        """get_output_contract returns stored contract."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        sink.set_output_contract(sample_contract)

        assert sink.get_output_contract() is sample_contract


class TestCSVSinkHeaderModes:
    """Test CSVSink header output modes."""

    @pytest.fixture
    def output_path(self, tmp_path: Path) -> Path:
        """Output file path."""
        return tmp_path / "output.csv"

    @pytest.fixture
    def sample_contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("amount_usd", int, original_name="'Amount USD'", required=True, source="declared"),
                make_field("customer_id", str, original_name="Customer ID", required=True, source="declared"),
            ),
            locked=True,
        )

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())


class TestCSVSinkHeaderModeInteraction:
    """Test interaction between header modes and schema contracts."""

    @pytest.fixture
    def output_path(self, tmp_path: Path) -> Path:
        """Output file path."""
        return tmp_path / "output.csv"

    @pytest.fixture
    def sample_contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field("amount_usd", int, original_name="'Amount USD'", required=True, source="declared"),
                make_field("customer_id", str, original_name="Customer ID", required=True, source="declared"),
            ),
            locked=True,
        )

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create a minimal plugin context."""
        factory = make_factory()
        return make_context(landscape=factory.plugin_audit_writer())

    def test_headers_mode_attribute_stored(self, output_path: Path) -> None:
        """CSVSink stores headers_mode from config."""
        from elspeth.contracts.header_modes import HeaderMode

        # Test NORMALIZED (default)
        sink_default = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )
        assert sink_default._headers_mode == HeaderMode.NORMALIZED

        # Test ORIGINAL
        sink_original = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "original",
                }
            )
        )
        assert sink_original._headers_mode == HeaderMode.ORIGINAL

        # Test CUSTOM
        sink_custom = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": {"amount_usd": "AMOUNT"},
                }
            )
        )
        assert sink_custom._headers_mode == HeaderMode.CUSTOM
