"""Tests for split LLM metadata functions."""

from elspeth.contracts.token_usage import TokenUsage


class TestPopulateLlmOperationalFields:
    def test_sets_usage_and_model_in_output(self):
        from elspeth.plugins.transforms.llm import populate_llm_operational_fields

        output: dict[str, object] = {}
        usage = TokenUsage.known(prompt_tokens=10, completion_tokens=20)
        populate_llm_operational_fields(output, "llm_response", usage=usage, model="gpt-4")
        assert output["llm_response_usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 20,
        }
        assert output["llm_response_model"] == "gpt-4"

    def test_usage_none_sets_none(self):
        from elspeth.plugins.transforms.llm import populate_llm_operational_fields

        output: dict[str, object] = {}
        populate_llm_operational_fields(output, "resp", usage=None, model="claude-3")
        assert output["resp_usage"] is None
        assert output["resp_model"] == "claude-3"

    def test_does_not_set_audit_fields(self):
        from elspeth.plugins.transforms.llm import LLM_AUDIT_SUFFIXES, populate_llm_operational_fields

        output: dict[str, object] = {}
        populate_llm_operational_fields(output, "r", usage=None, model=None)
        for suffix in LLM_AUDIT_SUFFIXES:
            assert f"r{suffix}" not in output


class TestBuildLlmAuditMetadata:
    def test_returns_dict_with_all_six_audit_fields(self):
        from elspeth.plugins.transforms.llm import build_llm_audit_metadata

        result = build_llm_audit_metadata(
            "llm_response",
            template_hash="abc123",
            variables_hash="def456",
            template_source="/path/to/template.j2",
            lookup_hash="ghi789",
            lookup_source="/path/to/lookup.yaml",
            system_prompt_source="/path/to/system.txt",
        )
        assert result == {
            "llm_response_template_hash": "abc123",
            "llm_response_variables_hash": "def456",
            "llm_response_template_source": "/path/to/template.j2",
            "llm_response_lookup_hash": "ghi789",
            "llm_response_lookup_source": "/path/to/lookup.yaml",
            "llm_response_system_prompt_source": "/path/to/system.txt",
        }

    def test_handles_none_values(self):
        from elspeth.plugins.transforms.llm import build_llm_audit_metadata

        result = build_llm_audit_metadata(
            "r",
            template_hash="hash",
            variables_hash="hash2",
            template_source=None,
            lookup_hash=None,
            lookup_source=None,
            system_prompt_source=None,
        )
        assert result["r_template_source"] is None
        assert result["r_lookup_hash"] is None
        assert result["r_lookup_source"] is None
        assert result["r_system_prompt_source"] is None

    def test_variables_hash_none_for_batch_level_metadata(self):
        """Batch-level audit metadata must use None for variables_hash, not a sentinel.

        Per-row variable hashes are recorded in the calls table via record_call().
        The batch-level summary should honestly represent absence (None), not
        fabricate a string like 'batch-varies-per-row'.
        """
        from elspeth.plugins.transforms.llm import build_llm_audit_metadata

        result = build_llm_audit_metadata(
            "llm_response",
            template_hash="abc123",
            variables_hash=None,
            template_source="/path/to/template.j2",
            lookup_hash=None,
            lookup_source=None,
            system_prompt_source=None,
        )
        assert result["llm_response_variables_hash"] is None

    def test_no_fabricated_sentinel_in_variables_hash(self):
        """variables_hash must never contain a sentinel string — only real hashes or None."""
        from elspeth.plugins.transforms.llm import build_llm_audit_metadata

        result = build_llm_audit_metadata(
            "llm_response",
            template_hash="abc123",
            variables_hash=None,
            template_source=None,
            lookup_hash=None,
            lookup_source=None,
            system_prompt_source=None,
        )
        value = result["llm_response_variables_hash"]
        assert value is None or (isinstance(value, str) and value != "batch-varies-per-row")

    def test_uses_correct_prefix(self):
        from elspeth.plugins.transforms.llm import build_llm_audit_metadata

        result = build_llm_audit_metadata(
            "category_llm_response",
            template_hash="h",
            variables_hash="v",
            template_source=None,
            lookup_hash=None,
            lookup_source=None,
            system_prompt_source=None,
        )
        assert all(k.startswith("category_llm_response_") for k in result)
        assert len(result) == 6


class TestAugmentedSchemaExcludesAuditFields:
    def test_single_query_schema_excludes_audit_fields(self):
        """_build_augmented_output_schema output schema must NOT include audit field names."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import LLM_AUDIT_SUFFIXES, _build_augmented_output_schema

        flexible_config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_augmented_output_schema(
            base_schema_config=flexible_config,
            response_field="llm_response",
            schema_name="TestOutputSchema",
        )
        field_names = set(schema_cls.model_fields.keys())
        for suffix in LLM_AUDIT_SUFFIXES:
            assert f"llm_response{suffix}" not in field_names, (
                f"Audit field 'llm_response{suffix}' should NOT be in output schema — audit fields belong in success_reason['metadata']"
            )

    def test_multi_query_schema_excludes_audit_fields(self):
        """_build_multi_query_output_schema output schema must NOT include audit field names."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import LLM_AUDIT_SUFFIXES, _build_multi_query_output_schema

        flexible_config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_multi_query_output_schema(
            base_schema_config=flexible_config,
            response_field="llm_response",
            query_names=("sentiment", "topic"),
            schema_name="TestMultiOutputSchema",
        )
        field_names = set(schema_cls.model_fields.keys())
        for query_name in ("sentiment", "topic"):
            prefix = f"{query_name}_llm_response"
            for suffix in LLM_AUDIT_SUFFIXES:
                assert f"{prefix}{suffix}" not in field_names, (
                    f"Audit field '{prefix}{suffix}' should NOT be in output schema — audit fields belong in success_reason['metadata']"
                )


class TestSchemaFieldTypesAreReal:
    """Bug fix for elspeth-8c9f3a41fe and elspeth-5c063fd55d.

    LLM output schema builders must advertise real types, not Any | None.
    """

    def test_single_query_response_field_is_str(self) -> None:
        """The response field must be typed as str, not any."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import _build_augmented_output_schema

        config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_augmented_output_schema(
            base_schema_config=config,
            response_field="llm_response",
            schema_name="TestSchema",
        )
        fields = schema_cls.model_fields
        # Response field itself → str
        assert "llm_response" in fields
        annotation = fields["llm_response"].annotation
        assert annotation is not None
        # Optional str is str | None
        assert str in (annotation.__args__ if hasattr(annotation, "__args__") else (annotation,))

    def test_single_query_model_field_is_str(self) -> None:
        """The _model field must be typed as str, not any."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import _build_augmented_output_schema

        config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_augmented_output_schema(
            base_schema_config=config,
            response_field="llm_response",
            schema_name="TestSchema",
        )
        fields = schema_cls.model_fields
        assert "llm_response_model" in fields
        annotation = fields["llm_response_model"].annotation
        assert annotation is not None
        assert str in (annotation.__args__ if hasattr(annotation, "__args__") else (annotation,))

    def test_multi_query_response_fields_are_str(self) -> None:
        """Multi-query prefixed response and model fields must be str."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import _build_multi_query_output_schema

        config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_multi_query_output_schema(
            base_schema_config=config,
            response_field="llm_response",
            query_names=("quality",),
            schema_name="TestSchema",
        )
        fields = schema_cls.model_fields
        for name in ("quality_llm_response", "quality_llm_response_model"):
            assert name in fields, f"Missing field {name}"
            annotation = fields[name].annotation
            assert annotation is not None
            assert str in (annotation.__args__ if hasattr(annotation, "__args__") else (annotation,)), (
                f"Field {name} should be str-typed, got {annotation}"
            )

    def test_multi_query_extracted_fields_preserve_types(self) -> None:
        """Extracted output_fields must use their declared types, not any."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.transforms.llm import _build_multi_query_output_schema

        config = SchemaConfig(mode="flexible", fields=())
        schema_cls = _build_multi_query_output_schema(
            base_schema_config=config,
            response_field="llm_response",
            query_names=("quality",),
            schema_name="TestSchema",
            extracted_fields={"quality": (("quality_score", "int"), ("quality_label", "str"))},
        )
        fields = schema_cls.model_fields
        # quality_score should be int-typed
        assert "quality_score" in fields
        score_annotation = fields["quality_score"].annotation
        assert score_annotation is not None
        assert int in (score_annotation.__args__ if hasattr(score_annotation, "__args__") else (score_annotation,)), (
            f"quality_score should be int-typed, got {score_annotation}"
        )
        # quality_label should be str-typed
        assert "quality_label" in fields
        label_annotation = fields["quality_label"].annotation
        assert label_annotation is not None
        assert str in (label_annotation.__args__ if hasattr(label_annotation, "__args__") else (label_annotation,)), (
            f"quality_label should be str-typed, got {label_annotation}"
        )
