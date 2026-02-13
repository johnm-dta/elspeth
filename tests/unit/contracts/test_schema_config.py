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

    def test_observed_schema(self) -> None:
        """Parse observed schema config."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"mode": "observed"})
        assert config.is_observed is True
        assert config.mode == "observed"
        assert config.fields is None

    def test_fixed_schema(self) -> None:
        """Parse fixed schema with explicit fields."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        )
        assert config.is_observed is False
        assert config.mode == "fixed"
        assert config.fields is not None  # fixed mode always has fields
        assert len(config.fields) == 2
        assert config.fields[0].name == "id"
        assert config.fields[1].name == "name"

    def test_flexible_schema(self) -> None:
        """Parse flexible schema with explicit fields."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "flexible",
                "fields": ["id: int", "name: str", "score: float?"],
            }
        )
        assert config.is_observed is False
        assert config.mode == "flexible"
        assert config.fields is not None  # flexible mode with explicit fields
        assert len(config.fields) == 3
        assert config.fields[2].required is False

    def test_explicit_fields_require_mode(self) -> None:
        """Explicit fields without mode raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"mode.*required"):
            SchemaConfig.from_dict({"fields": ["id: int"]})

    def test_observed_with_fields_raises(self) -> None:
        """Observed mode with explicit fields raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"Observed schemas.*cannot have explicit field definitions"):
            SchemaConfig.from_dict({"mode": "observed", "fields": ["id: int"]})

    def test_observed_with_string_fields_raises(self) -> None:
        """Observed mode rejects non-list fields values (string)."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"Observed schemas.*cannot have explicit field definitions"):
            SchemaConfig.from_dict({"mode": "observed", "fields": "id: int"})

    def test_observed_with_dict_fields_raises(self) -> None:
        """Observed mode rejects non-list fields values (dict)."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"Observed schemas.*cannot have explicit field definitions"):
            SchemaConfig.from_dict({"mode": "observed", "fields": {"id": "int"}})

    def test_observed_with_empty_fields_list_is_allowed(self) -> None:
        """Observed mode allows explicit empty list as equivalent to no fields."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"mode": "observed", "fields": []})
        assert config.mode == "observed"
        assert config.fields is None

    def test_missing_mode_raises(self) -> None:
        """Missing mode key raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"mode.*required"):
            SchemaConfig.from_dict({})

    def test_empty_fields_raises(self) -> None:
        """Empty fields list raises error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="at least one field"):
            SchemaConfig.from_dict({"mode": "fixed", "fields": []})

    def test_duplicate_field_names_raises(self) -> None:
        """Duplicate field names raise error."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="Duplicate field names"):
            SchemaConfig.from_dict(
                {
                    "mode": "fixed",
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
                "mode": "fixed",
                "fields": [{"id": "int"}, {"name": "str"}],
            }
        )
        assert config.is_observed is False
        assert config.mode == "fixed"
        assert config.fields is not None
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
                "mode": "fixed",
                "fields": [{"score": "float?"}],
            }
        )
        assert config.fields is not None
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
                "mode": "fixed",
                "fields": ["id: int", {"name": "str"}, "score: float?"],
            }
        )
        assert config.fields is not None
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
                    "mode": "fixed",
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
                    "mode": "fixed",
                    "fields": [[1, 2, 3]],
                }
            )


class TestSchemaConfigSerialization:
    """Tests for SchemaConfig serialization and round-trip."""

    def test_observed_schema_to_dict(self) -> None:
        """Observed schema serializes with mode='observed'."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"mode": "observed"})
        serialized = config.to_dict()
        assert serialized == {"mode": "observed", "fields": None}

    def test_observed_schema_roundtrip(self) -> None:
        """Observed schema survives serialization round-trip."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict({"mode": "observed"})
        serialized = config.to_dict()
        roundtrip = SchemaConfig.from_dict(serialized)
        assert roundtrip.is_observed is True
        assert roundtrip.mode == "observed"
        assert roundtrip.fields is None

    def test_fixed_schema_to_dict(self) -> None:
        """Fixed schema with fields serializes correctly."""
        from elspeth.contracts.schema import SchemaConfig

        config = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str", "score: float?"],
            }
        )
        serialized = config.to_dict()
        assert serialized["mode"] == "fixed"
        assert len(serialized["fields"]) == 3
        assert serialized["fields"][0] == {"name": "id", "type": "int", "required": True}
        assert serialized["fields"][1] == {"name": "name", "type": "str", "required": True}
        assert serialized["fields"][2] == {"name": "score", "type": "float", "required": False}

    def test_fixed_schema_roundtrip(self) -> None:
        """Fixed schema survives serialization round-trip via dict-form fields."""
        from elspeth.contracts.schema import SchemaConfig

        original = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        )
        serialized = original.to_dict()
        roundtrip = SchemaConfig.from_dict(
            {
                "mode": serialized["mode"],
                "fields": [{f["name"]: f["type"] + ("?" if not f["required"] else "")} for f in serialized["fields"]],
            }
        )
        assert roundtrip.is_observed is False
        assert roundtrip.mode == "fixed"
        assert roundtrip.fields is not None
        assert len(roundtrip.fields) == 2
        assert roundtrip.fields[0].name == "id"
        assert roundtrip.fields[0].field_type == "int"
        assert roundtrip.fields[1].name == "name"
        assert roundtrip.fields[1].field_type == "str"


class TestAuditFields:
    """Tests for audit_fields schema attribute.

    BUG-AZURE-03: LLM transforms emit metadata fields that aren't declared
    in their output schema. The audit_fields attribute distinguishes between:
    - guaranteed_fields: Stable API contract fields (downstream can depend on)
    - audit_fields: Provenance metadata that exists but isn't stability-contracted
    """

    def test_audit_fields_parsing(self) -> None:
        """Verify audit_fields is parsed from config dict."""
        from elspeth.contracts.schema import SchemaConfig

        config = {
            "mode": "observed",
            "guaranteed_fields": ["response", "response_usage"],
            "audit_fields": ["response_template_hash", "response_lookup_source"],
        }
        schema = SchemaConfig.from_dict(config)

        assert schema.audit_fields == ("response_template_hash", "response_lookup_source")
        assert schema.guaranteed_fields == ("response", "response_usage")

    def test_audit_fields_not_in_effective_guaranteed(self) -> None:
        """Verify audit_fields are excluded from effective guaranteed fields."""
        from elspeth.contracts.schema import SchemaConfig

        schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("response", "response_usage"),
            audit_fields=("response_template_hash",),
        )

        effective = schema.get_effective_guaranteed_fields()
        assert "response" in effective
        assert "response_usage" in effective
        assert "response_template_hash" not in effective

    def test_audit_fields_serialization_roundtrip(self) -> None:
        """Verify audit_fields survive to_dict/from_dict round-trip."""
        from elspeth.contracts.schema import SchemaConfig

        schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("response",),
            audit_fields=("response_template_hash", "response_lookup_source"),
        )

        serialized = schema.to_dict()
        assert "audit_fields" in serialized
        assert serialized["audit_fields"] == ["response_template_hash", "response_lookup_source"]

        roundtrip = SchemaConfig.from_dict(serialized)
        assert roundtrip.audit_fields == ("response_template_hash", "response_lookup_source")

    def test_audit_fields_rejects_non_list(self) -> None:
        """audit_fields must be a list."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="must be a list"):
            SchemaConfig.from_dict({"mode": "observed", "audit_fields": "not_a_list"})

    def test_audit_fields_rejects_duplicates(self) -> None:
        """audit_fields must not contain duplicates."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="Duplicate field names"):
            SchemaConfig.from_dict({"mode": "observed", "audit_fields": ["hash", "hash"]})

    def test_audit_fields_rejects_invalid_identifiers(self) -> None:
        """audit_fields must be valid Python identifiers."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match="valid Python identifier"):
            SchemaConfig.from_dict({"mode": "observed", "audit_fields": ["valid", "invalid-field"]})


class TestContractFieldSubsetValidation:
    """Tests for validating contract fields are subsets of declared fields.

    Bug: P2-2026-01-31-schema-config-undefined-contract-fields

    For explicit schemas (mode=fixed/flexible), guaranteed_fields, required_fields,
    and audit_fields MUST be subsets of declared field names. Typos in these
    lists would otherwise create false audit claims.

    For observed schemas, there are no declared fields to validate against,
    so arbitrary field names are allowed (this is the only way to express
    contracts for observed schemas).
    """

    def test_guaranteed_fields_typo_in_explicit_schema_raises(self) -> None:
        """Typo in guaranteed_fields for explicit schema raises ValueError."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"guaranteed_fields.*not declared.*custmer_id"):
            SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": ["customer_id: str", "amount: float"],
                    "guaranteed_fields": ["custmer_id"],  # Typo!
                }
            )

    def test_required_fields_typo_in_explicit_schema_raises(self) -> None:
        """Typo in required_fields for explicit schema raises ValueError."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"required_fields.*not declared.*custmer_id"):
            SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": ["customer_id: str"],
                    "required_fields": ["custmer_id"],  # Typo!
                }
            )

    def test_audit_fields_typo_in_explicit_schema_raises(self) -> None:
        """Typo in audit_fields for explicit schema raises ValueError."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"audit_fields.*not declared.*custmer_id"):
            SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": ["customer_id: str"],
                    "audit_fields": ["custmer_id"],  # Typo!
                }
            )

    def test_multiple_undefined_fields_all_reported(self) -> None:
        """Multiple undefined fields are all reported in error message."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"(typo1.*typo2|typo2.*typo1)"):
            SchemaConfig.from_dict(
                {
                    "mode": "fixed",
                    "fields": ["customer_id: str"],
                    "guaranteed_fields": ["typo1", "typo2"],
                }
            )

    def test_valid_contract_fields_accepted(self) -> None:
        """Valid contract fields (subset of declared) are accepted."""
        from elspeth.contracts.schema import SchemaConfig

        schema = SchemaConfig.from_dict(
            {
                "mode": "fixed",
                "fields": ["customer_id: str", "amount: float", "timestamp: str"],
                "guaranteed_fields": ["customer_id", "amount"],
                "required_fields": ["customer_id"],
                "audit_fields": ["timestamp"],
            }
        )
        assert schema.guaranteed_fields == ("customer_id", "amount")
        assert schema.required_fields == ("customer_id",)
        assert schema.audit_fields == ("timestamp",)

    def test_observed_schema_allows_arbitrary_contract_fields(self) -> None:
        """Observed schemas allow arbitrary field names in contracts.

        For observed schemas, there are no declared fields, so contract fields
        are the ONLY way to express guarantees. We can't validate them.
        """
        from elspeth.contracts.schema import SchemaConfig

        schema = SchemaConfig.from_dict(
            {
                "mode": "observed",
                "guaranteed_fields": ["any_field_name", "another_field"],
                "required_fields": ["completely_arbitrary"],
            }
        )
        assert schema.guaranteed_fields == ("any_field_name", "another_field")
        assert schema.required_fields == ("completely_arbitrary",)

    def test_flexible_mode_also_validates_contract_fields(self) -> None:
        """Flexible mode schemas also validate contract field subsets."""
        from elspeth.contracts.schema import SchemaConfig

        with pytest.raises(ValueError, match=r"guaranteed_fields.*not declared"):
            SchemaConfig.from_dict(
                {
                    "mode": "flexible",
                    "fields": ["customer_id: str"],
                    "guaranteed_fields": ["typo_field"],
                }
            )
