"""Tests for sink header mode resolution.

Task 4 of Phase 3: Sink Header Modes implementation.
"""

from __future__ import annotations

import types

import pytest

from elspeth.contracts.header_modes import (
    HeaderMode,
    parse_header_mode,
    resolve_headers,
)
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.testing import make_field


class TestParseHeaderMode:
    """Test parsing header mode from config."""

    def test_parse_normalized(self) -> None:
        """String 'normalized' parses to NORMALIZED mode."""
        mode = parse_header_mode("normalized")
        assert mode == HeaderMode.NORMALIZED

    def test_parse_original(self) -> None:
        """String 'original' parses to ORIGINAL mode."""
        mode = parse_header_mode("original")
        assert mode == HeaderMode.ORIGINAL

    def test_parse_dict_is_custom(self) -> None:
        """Dict config is CUSTOM mode."""
        mode = parse_header_mode({"amount_usd": "AMOUNT_USD"})
        assert mode == HeaderMode.CUSTOM

    def test_parse_none_defaults_to_normalized(self) -> None:
        """None defaults to NORMALIZED."""
        mode = parse_header_mode(None)
        assert mode == HeaderMode.NORMALIZED

    def test_parse_invalid_raises(self) -> None:
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid header mode"):
            parse_header_mode("invalid_mode")


class TestResolveHeaders:
    """Test header resolution for different modes."""

    @pytest.fixture
    def contract(self) -> SchemaContract:
        """Contract with original name mappings."""
        return SchemaContract(
            mode="FLEXIBLE",
            fields=(
                make_field(
                    "amount_usd",
                    int,
                    original_name="'Amount USD'",
                    required=True,
                    source="declared",
                ),
                make_field(
                    "customer_id",
                    str,
                    original_name="Customer ID",
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )

    def test_normalized_mode(self, contract: SchemaContract) -> None:
        """NORMALIZED mode uses normalized names."""
        headers = resolve_headers(
            contract=contract,
            mode=HeaderMode.NORMALIZED,
            custom_mapping=None,
        )

        assert headers == {"amount_usd": "amount_usd", "customer_id": "customer_id"}

    def test_original_mode(self, contract: SchemaContract) -> None:
        """ORIGINAL mode uses original names from contract."""
        headers = resolve_headers(
            contract=contract,
            mode=HeaderMode.ORIGINAL,
            custom_mapping=None,
        )

        assert headers == {"amount_usd": "'Amount USD'", "customer_id": "Customer ID"}

    def test_original_mode_raises_on_contract_lookup_miss(self, contract: SchemaContract) -> None:
        """ORIGINAL mode should crash on contract corruption (Tier 1 data)."""
        corrupted_index = {k: v for k, v in contract._by_normalized.items() if k != "customer_id"}
        object.__setattr__(contract, "_by_normalized", types.MappingProxyType(corrupted_index))

        with pytest.raises(KeyError, match="customer_id"):
            resolve_headers(
                contract=contract,
                mode=HeaderMode.ORIGINAL,
                custom_mapping=None,
            )

    def test_custom_mode(self, contract: SchemaContract) -> None:
        """CUSTOM mode uses provided mapping."""
        custom = {"amount_usd": "AMOUNT", "customer_id": "CUSTOMER"}

        headers = resolve_headers(
            contract=contract,
            mode=HeaderMode.CUSTOM,
            custom_mapping=custom,
        )

        assert headers == {"amount_usd": "AMOUNT", "customer_id": "CUSTOMER"}

    def test_custom_partial_mapping(self, contract: SchemaContract) -> None:
        """CUSTOM mode with partial mapping falls back to normalized."""
        custom = {"amount_usd": "AMOUNT"}  # customer_id not mapped

        headers = resolve_headers(
            contract=contract,
            mode=HeaderMode.CUSTOM,
            custom_mapping=custom,
        )

        assert headers["amount_usd"] == "AMOUNT"
        assert headers["customer_id"] == "customer_id"  # Fallback

    def test_no_contract_returns_identity(self) -> None:
        """Without contract, returns identity mapping for known fields."""
        headers = resolve_headers(
            contract=None,
            mode=HeaderMode.ORIGINAL,
            custom_mapping=None,
            field_names=["a", "b"],
        )

        assert headers == {"a": "a", "b": "b"}

    def test_no_contract_no_fields_returns_empty(self) -> None:
        """Without contract and without field_names, returns empty dict."""
        headers = resolve_headers(
            contract=None,
            mode=HeaderMode.NORMALIZED,
            custom_mapping=None,
        )

        assert headers == {}

    def test_no_contract_custom_mode(self) -> None:
        """CUSTOM mode without contract uses mapping with fallback."""
        custom = {"a": "A_HEADER"}

        headers = resolve_headers(
            contract=None,
            mode=HeaderMode.CUSTOM,
            custom_mapping=custom,
            field_names=["a", "b"],
        )

        assert headers["a"] == "A_HEADER"
        assert headers["b"] == "b"  # Fallback to identity


class TestHeaderModeEnum:
    """Test HeaderMode enum properties."""

    def test_enum_values_exist(self) -> None:
        """All three modes exist."""
        assert HeaderMode.NORMALIZED is not None
        assert HeaderMode.ORIGINAL is not None
        assert HeaderMode.CUSTOM is not None

    def test_enum_values_distinct(self) -> None:
        """All modes are distinct."""
        modes = [HeaderMode.NORMALIZED, HeaderMode.ORIGINAL, HeaderMode.CUSTOM]
        assert len(set(modes)) == 3
