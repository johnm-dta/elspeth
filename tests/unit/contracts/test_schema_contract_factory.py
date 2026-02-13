"""Tests for SchemaContract factory from SchemaConfig."""

from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract_factory import (
    create_contract_from_config,
    map_schema_mode,
)


class TestMapSchemaMode:
    """Test mapping SchemaConfig modes to SchemaContract modes."""

    def test_fixed_maps_to_fixed(self) -> None:
        """fixed mode maps to FIXED."""
        assert map_schema_mode("fixed") == "FIXED"

    def test_flexible_maps_to_flexible(self) -> None:
        """flexible mode maps to FLEXIBLE."""
        assert map_schema_mode("flexible") == "FLEXIBLE"

    def test_observed_maps_to_observed(self) -> None:
        """observed mode maps to OBSERVED."""
        assert map_schema_mode("observed") == "OBSERVED"


class TestCreateContractFromConfig:
    """Test creating SchemaContract from SchemaConfig."""

    def test_dynamic_schema_creates_observed_contract(self) -> None:
        """Dynamic schema creates unlocked OBSERVED contract."""
        config = SchemaConfig.from_dict({"mode": "observed"})
        contract = create_contract_from_config(config)

        assert contract.mode == "OBSERVED"
        assert contract.locked is False
        assert len(contract.fields) == 0

    def test_strict_schema_creates_fixed_contract(self) -> None:
        """Strict schema creates FIXED contract with declared fields."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        )
        contract = create_contract_from_config(config)

        assert contract.mode == "FIXED"
        assert len(contract.fields) == 2
        # Fields are declared and required
        id_field = next(f for f in contract.fields if f.normalized_name == "id")
        assert id_field.python_type is int
        assert id_field.required is True
        assert id_field.source == "declared"

    def test_free_schema_creates_flexible_contract(self) -> None:
        """Free schema creates FLEXIBLE contract."""
        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
                "fields": ["id: int"],
            }
        )
        contract = create_contract_from_config(config)

        assert contract.mode == "FLEXIBLE"
        assert contract.locked is False
        assert len(contract.fields) == 1
        assert contract.fields[0].normalized_name == "id"

    def test_optional_field_not_required(self) -> None:
        """Optional field (?) has required=False."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "note: str?"],
            }
        )
        contract = create_contract_from_config(config)

        note_field = next(f for f in contract.fields if f.normalized_name == "note")
        assert note_field.required is False

    def test_field_type_mapping(self) -> None:
        """Field types map correctly to Python types."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": [
                    "a: int",
                    "b: str",
                    "c: float",
                    "d: bool",
                    "e: any",
                ],
            }
        )
        contract = create_contract_from_config(config)

        type_map = {f.normalized_name: f.python_type for f in contract.fields}
        assert type_map["a"] is int
        assert type_map["b"] is str
        assert type_map["c"] is float
        assert type_map["d"] is bool
        # 'any' type maps to object (base type)
        assert type_map["e"] is object

    def test_fixed_contract_is_locked(self) -> None:
        """FIXED schemas start locked (types fully declared)."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int"],
            }
        )
        contract = create_contract_from_config(config)

        assert contract.locked is True

    def test_flexible_contract_starts_unlocked(self) -> None:
        """FLEXIBLE schemas start unlocked for first-row extra-field inference."""
        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
                "fields": ["id: int"],
            }
        )
        contract = create_contract_from_config(config)

        assert contract.locked is False

    def test_dynamic_contract_is_unlocked(self) -> None:
        """Dynamic schemas start unlocked (types inferred from first row)."""
        config = SchemaConfig.from_dict({"mode": "observed"})
        contract = create_contract_from_config(config)

        assert contract.locked is False


class TestContractWithFieldResolution:
    """Test creating contracts with field resolution (original names)."""

    def test_field_resolution_sets_original_names(self) -> None:
        """Field resolution mapping populates original_name."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["amount_usd: int", "customer_id: str"],
            }
        )
        resolution = {
            "'Amount USD'": "amount_usd",
            "Customer ID": "customer_id",
        }
        contract = create_contract_from_config(config, field_resolution=resolution)

        amount_field = next(f for f in contract.fields if f.normalized_name == "amount_usd")
        assert amount_field.original_name == "'Amount USD'"

        customer_field = next(f for f in contract.fields if f.normalized_name == "customer_id")
        assert customer_field.original_name == "Customer ID"

    def test_no_resolution_uses_normalized_as_original(self) -> None:
        """Without resolution, original_name equals normalized_name."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int"],
            }
        )
        contract = create_contract_from_config(config)  # No resolution

        id_field = contract.fields[0]
        assert id_field.original_name == id_field.normalized_name == "id"

    def test_partial_resolution(self) -> None:
        """Resolution mapping can be partial (not all fields mapped)."""
        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["mapped_field: int", "unmapped_field: str"],
            }
        )
        resolution = {"Original Header": "mapped_field"}  # Only one field
        contract = create_contract_from_config(config, field_resolution=resolution)

        mapped = next(f for f in contract.fields if f.normalized_name == "mapped_field")
        unmapped = next(f for f in contract.fields if f.normalized_name == "unmapped_field")

        assert mapped.original_name == "Original Header"
        assert unmapped.original_name == "unmapped_field"  # Falls back to normalized
