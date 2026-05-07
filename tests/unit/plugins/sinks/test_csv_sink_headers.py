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

    def test_normalized_headers_default(self, output_path: Path, ctx: PluginContext) -> None:
        """Default mode uses normalized (Python identifier) headers."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                }
            )
        )

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        content = output_path.read_text()
        assert "amount_usd" in content
        assert "customer_id" in content
        # Should NOT have original headers
        assert "'Amount USD'" not in content
        assert "Customer ID" not in content

    def test_original_headers_from_contract(self, output_path: Path, sample_contract: SchemaContract, ctx: PluginContext) -> None:
        """headers: original restores source headers from contract."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "original",
                }
            )
        )

        # Provide contract to sink via explicit set_output_contract
        sink.set_output_contract(sample_contract)

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        content = output_path.read_text()
        # Should have original headers from contract
        assert "'Amount USD'" in content
        assert "Customer ID" in content
        # Should NOT have normalized headers in header row
        # Note: normalized names will be in data row (as values), but header should be original
        lines = content.strip().split("\n")
        header_line = lines[0]
        assert "'Amount USD'" in header_line
        assert "Customer ID" in header_line

    def test_original_headers_from_ctx_contract(self, output_path: Path, sample_contract: SchemaContract, ctx: PluginContext) -> None:
        """headers: original resolves from ctx.contract if _output_contract not set.

        This tests the production path where orchestrator sets ctx.contract
        but doesn't explicitly call sink.set_output_contract(). The sink
        should lazily capture the contract on first write().
        """
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "original",
                }
            )
        )

        # Simulate orchestrator behavior: set contract on context, not sink
        ctx.contract = sample_contract
        # Use == instead of is to avoid mypy type narrowing that makes later code unreachable
        assert sink._output_contract == None  # Not explicitly set  # noqa: E711

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        # Contract should have been captured from context
        assert sink._output_contract is sample_contract

        content = output_path.read_text()
        # Should have original headers from contract
        lines = content.strip().split("\n")
        header_line = lines[0]
        assert "'Amount USD'" in header_line
        assert "Customer ID" in header_line

    def test_custom_headers_mapping(self, output_path: Path, ctx: PluginContext) -> None:
        """headers: {mapping} uses custom header names."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": {
                        "amount_usd": "AMOUNT",
                        "customer_id": "CUST_ID",
                    },
                }
            )
        )

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        content = output_path.read_text()
        lines = content.strip().split("\n")
        header_line = lines[0]
        assert "AMOUNT" in header_line
        assert "CUST_ID" in header_line

    def test_original_headers_without_contract_or_landscape_errors(self, output_path: Path) -> None:
        """headers: original without contract or Landscape raises ValueError.

        When the user explicitly requests original headers but no source of
        original names is available (no contract, no Landscape), write()
        raises an error rather than silently degrading to normalized names.
        """
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "original",
                }
            )
        )
        # Deliberately NOT setting a contract or landscape
        ctx_no_landscape = PluginContext(run_id="test-run", config={}, landscape=None)

        with pytest.raises(ValueError, match="requires Landscape"):
            sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx_no_landscape)


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

    def test_contract_used_with_original_mode(self, output_path: Path, sample_contract: SchemaContract, ctx: PluginContext) -> None:
        """Contract is used for header resolution when mode is ORIGINAL."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "original",
                }
            )
        )
        sink.set_output_contract(sample_contract)

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        # Verify header row has original names
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert "'Amount USD'" in lines[0]
        assert "Customer ID" in lines[0]

    def test_contract_ignored_with_normalized_mode(self, output_path: Path, sample_contract: SchemaContract, ctx: PluginContext) -> None:
        """Contract is ignored when mode is NORMALIZED (explicit or default)."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": "normalized",  # Explicit normalized
                }
            )
        )
        sink.set_output_contract(sample_contract)

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        # Verify header row has normalized names (contract ignored)
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert "amount_usd" in lines[0]
        assert "customer_id" in lines[0]
        assert "'Amount USD'" not in lines[0]

    def test_custom_mapping_takes_precedence_over_contract(
        self, output_path: Path, sample_contract: SchemaContract, ctx: PluginContext
    ) -> None:
        """Custom mapping takes precedence over contract original names."""
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_path),
                    "schema": STRICT_SCHEMA,
                    "headers": {
                        "amount_usd": "MY_AMOUNT",
                        "customer_id": "MY_CUSTOMER",
                    },
                }
            )
        )
        sink.set_output_contract(sample_contract)  # Set contract but should be ignored

        sink.write([{"amount_usd": 100, "customer_id": "C001"}], ctx)
        sink.close()

        # Verify header row has custom mapping names
        content = output_path.read_text()
        lines = content.strip().split("\n")
        assert "MY_AMOUNT" in lines[0]
        assert "MY_CUSTOMER" in lines[0]
        assert "'Amount USD'" not in lines[0]
