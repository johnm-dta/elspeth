"""Tests for schema configuration types."""

import pytest


class TestFieldDefinition:
    """Tests for FieldDefinition parsing."""

    def test_field_definition_exists(self) -> None:
        """FieldDefinition can be imported."""
        from elspeth.contracts.schema import FieldDefinition

        assert FieldDefinition is not None

    def test_parse_required_field(self) -> None:
        """Parse 'name: str' as required string field."""
        from elspeth.contracts.schema import FieldDefinition

        field = FieldDefinition.parse("name: str")
        assert field.name == "name"
        assert field.field_type == "str"
        assert field.required is True

    def test_parse_optional_field(self) -> None:
        """Parse 'score: float?' as optional float field."""
        from elspeth.contracts.schema import FieldDefinition

        field = FieldDefinition.parse("score: float?")
        assert field.name == "score"
        assert field.field_type == "float"
        assert field.required is False

    def test_parse_all_types(self) -> None:
        """All supported types parse correctly."""
        from elspeth.contracts.schema import FieldDefinition

        for type_name in ["str", "int", "float", "bool", "any"]:
            field = FieldDefinition.parse(f"field: {type_name}")
            assert field.field_type == type_name

    def test_parse_invalid_type_raises(self) -> None:
        """Invalid type raises ValueError."""
        from elspeth.contracts.schema import FieldDefinition

        with pytest.raises(ValueError, match="Unknown type"):
            FieldDefinition.parse("field: invalid")

    def test_parse_malformed_raises(self) -> None:
        """Malformed field spec raises ValueError."""
        from elspeth.contracts.schema import FieldDefinition

        with pytest.raises(ValueError, match="Invalid field"):
            FieldDefinition.parse("no_colon_here")

    def test_parse_hyphenated_field_name_raises(self) -> None:
        """Field names with hyphens raise helpful error."""
        from elspeth.contracts.schema import FieldDefinition

        with pytest.raises(ValueError, match=r"user_id.*instead"):
            FieldDefinition.parse("user-id: int")

    def test_parse_dotted_field_name_raises(self) -> None:
        """Field names with dots raise helpful error."""
        from elspeth.contracts.schema import FieldDefinition

        with pytest.raises(ValueError, match=r"data_field.*instead"):
            FieldDefinition.parse("data.field: str")

    def test_parse_numeric_prefix_raises(self) -> None:
        """Field names starting with digits raise error."""
        from elspeth.contracts.schema import FieldDefinition

        with pytest.raises(ValueError, match="cannot start with a digit"):
            FieldDefinition.parse("123field: int")


class TestSchemaConfig:
    """Tests for SchemaConfig parsing."""

    def test_schema_config_exists(self) -> None:
        """SchemaConfig can be imported."""
        from elspeth.contracts.schema import SchemaConfig

        assert SchemaConfig is not None

    def test_dynamic_schema(self) -> None:
        """Parse dynamic schema config."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"fields": "dynamic"})
        assert config.is_dynamic is True
        assert config.mode is None
        assert config.fields is None

    def test_strict_schema(self) -> None:
        """Parse strict schema with explicit fields."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["id: int", "name: str"],
            }
        )
        assert config.is_dynamic is False
        assert config.mode == "strict"
        assert config.fields is not None  # strict mode always has fields
        assert len(config.fields) == 2
        assert config.fields[0].name == "id"
        assert config.fields[1].name == "name"

    def test_free_schema(self) -> None:
        """Parse free schema with explicit fields."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "free",
                "fields": ["id: int", "name: str", "score: float?"],
            }
        )
        assert config.is_dynamic is False
        assert config.mode == "free"
        assert config.fields is not None  # free mode with explicit fields
        assert len(config.fields) == 3
        assert config.fields[2].required is False

    def test_explicit_fields_require_mode(self) -> None:
        """Explicit fields without mode raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"mode.*required"):
            SchemaConfig.from_dict({"fields": ["id: int"]})

    def test_dynamic_ignores_mode(self) -> None:
        """Dynamic fields ignores mode if provided."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"fields": "dynamic", "mode": "strict"})
        assert config.is_dynamic is True
        assert config.mode is None  # Ignored

    def test_missing_fields_raises(self) -> None:
        """Missing fields key raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"fields.*required"):
            SchemaConfig.from_dict({})

    def test_empty_fields_raises(self) -> None:
        """Empty fields list raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="at least one field"):
            SchemaConfig.from_dict({"mode": "strict", "fields": []})

    def test_duplicate_field_names_raises(self) -> None:
        """Duplicate field names raise error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="Duplicate field names"):
            SchemaConfig.from_dict(
                {
                    "mode": "strict",
                    "fields": ["id: int", "id: str"],
                }
            )

    def test_dict_form_field_specs(self) -> None:
        """Dict-form field specs (YAML: - id: int) parse correctly.

        Bug: P1-2026-01-20-schema-config-yaml-example-crashes-parsing
        """
        from elspeth.contracts.schema import SchemaConfig

        # YAML `- id: int` parses as [{"id": "int"}]
        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": [{"id": "int"}, {"name": "str"}],
            }
        )
        assert config.is_dynamic is False
        assert config.mode == "strict"
        assert len(config.fields) == 2
        assert config.fields[0].name == "id"
        assert config.fields[0].field_type == "int"
        assert config.fields[1].name == "name"
        assert config.fields[1].field_type == "str"

    def test_dict_form_optional_field(self) -> None:
        """Dict-form optional field specs parse correctly.

        Bug: P1-2026-01-20-schema-config-yaml-example-crashes-parsing
        """
        from elspeth.contracts.schema import SchemaConfig

        # YAML `- score: float?` parses as [{"score": "float?"}]
        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": [{"score": "float?"}],
            }
        )
        assert config.fields[0].name == "score"
        assert config.fields[0].field_type == "float"
        assert config.fields[0].required is False

    def test_mixed_string_and_dict_form(self) -> None:
        """Mixed string and dict field specs work together.

        Bug: P1-2026-01-20-schema-config-yaml-example-crashes-parsing
        """
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "strict",
                "fields": ["id: int", {"name": "str"}, "score: float?"],
            }
        )
        assert len(config.fields) == 3
        assert config.fields[0].name == "id"
        assert config.fields[1].name == "name"
        assert config.fields[2].name == "score"

    def test_multi_key_dict_raises_valueerror(self) -> None:
        """Multi-key dict in fields raises ValueError, not AttributeError.

        Bug: P1-2026-01-20-schema-config-yaml-example-crashes-parsing
        """
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"dict with 2 keys"):
            SchemaConfig.from_dict(
                {
                    "mode": "strict",
                    "fields": [{"id": "int", "name": "str"}],
                }
            )

    def test_invalid_field_type_raises_valueerror(self) -> None:
        """Invalid field type (not str/dict) raises ValueError, not AttributeError.

        Bug: P1-2026-01-20-schema-config-yaml-example-crashes-parsing
        """
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"must be a string.*or a dict"):
            SchemaConfig.from_dict(
                {
                    "mode": "strict",
                    "fields": [[1, 2, 3]],
                }
            )
