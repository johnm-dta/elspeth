# tests/core/test_template_extraction_dual.py
"""Tests for extracting fields with original name annotation."""

import types

import pytest

from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.templates import extract_jinja2_fields_with_names


class TestExtractWithNames:
    """Test field extraction with original name resolution."""

    @pytest.fixture
    def contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                FieldContract("amount_usd", "'Amount USD'", int, True, "declared"),
                FieldContract("customer_id", "Customer ID", str, True, "declared"),
                FieldContract("simple", "simple", str, True, "declared"),
            ),
            locked=True,
        )

    def test_extract_returns_normalized_and_original(self, contract: SchemaContract) -> None:
        """Extraction with contract returns both name forms."""
        template = "{{ row.amount_usd }} and {{ row[\"'Amount USD'\"] }}"

        result = extract_jinja2_fields_with_names(template, contract)

        # Should report the field once with both names
        assert "amount_usd" in result
        assert result["amount_usd"]["original"] == "'Amount USD'"
        assert result["amount_usd"]["normalized"] == "amount_usd"

    def test_extract_resolves_original_to_normalized(self, contract: SchemaContract) -> None:
        """Original name references resolve to normalized."""
        template = '{{ row["Customer ID"] }}'

        result = extract_jinja2_fields_with_names(template, contract)

        assert "customer_id" in result
        assert result["customer_id"]["original"] == "Customer ID"

    def test_extract_deduplicates_same_field(self, contract: SchemaContract) -> None:
        """Same field accessed both ways is deduplicated."""
        template = "{{ row.amount_usd }} {{ row[\"'Amount USD'\"] }}"

        result = extract_jinja2_fields_with_names(template, contract)

        # Only one entry for the field
        assert len(result) == 1
        assert "amount_usd" in result

    def test_extract_reports_unknown_field(self, contract: SchemaContract) -> None:
        """Unknown field is reported with normalized as original."""
        template = "{{ row.unknown_field }}"

        result = extract_jinja2_fields_with_names(template, contract)

        # Unknown field - original equals what was written
        assert "unknown_field" in result
        assert result["unknown_field"]["original"] == "unknown_field"
        assert result["unknown_field"]["resolved"] is False

    def test_extract_without_contract_returns_as_written(self) -> None:
        """Without contract, returns field names as written."""
        template = "{{ row.field_name }}"

        result = extract_jinja2_fields_with_names(template)  # No contract

        assert "field_name" in result
        assert result["field_name"]["original"] == "field_name"
        assert result["field_name"]["resolved"] is False

    def test_extract_mixed_known_unknown(self, contract: SchemaContract) -> None:
        """Mix of known and unknown fields works."""
        template = "{{ row.amount_usd }} {{ row.unknown }}"

        result = extract_jinja2_fields_with_names(template, contract)

        assert "amount_usd" in result
        assert result["amount_usd"]["resolved"] is True

        assert "unknown" in result
        assert result["unknown"]["resolved"] is False

    def test_extract_original_name_only(self, contract: SchemaContract) -> None:
        """Template using only original names."""
        template = '{{ row["\'Amount USD\'"] }} {{ row["Customer ID"] }}'

        result = extract_jinja2_fields_with_names(template, contract)

        assert "amount_usd" in result  # Resolved to normalized
        assert "customer_id" in result
        assert len(result) == 2

    def test_extract_raises_on_contract_corruption_after_name_resolution(self, contract: SchemaContract) -> None:
        """Resolved names should fail fast if contract field index is corrupted."""
        corrupted_index = {k: v for k, v in contract._by_normalized.items() if k != "customer_id"}
        object.__setattr__(contract, "_by_normalized", types.MappingProxyType(corrupted_index))

        with pytest.raises(KeyError, match="customer_id"):
            extract_jinja2_fields_with_names('{{ row["Customer ID"] }}', contract)
