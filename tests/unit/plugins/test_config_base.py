# tests/plugins/test_config_base.py
"""Tests for plugin configuration base classes."""

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.infrastructure.config_base import (
    DataPluginConfig,
    PathConfig,
    PluginConfig,
    PluginConfigError,
    SinkPathConfig,
)

# Helper to create a dynamic schema for tests
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestPluginConfig:
    """Tests for PluginConfig base class."""

    def test_rejects_extra_fields(self) -> None:
        """Extra fields should raise validation error."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(ValidationError) as exc_info:
            MyConfig(name="test", unknown_field="value")  # type: ignore[call-arg]

        assert "Extra inputs are not permitted" in str(exc_info.value)

    def test_from_dict_wraps_validation_error(self) -> None:
        """from_dict should wrap ValidationError in PluginConfigError."""

        class MyConfig(PluginConfig):
            required_field: str

        with pytest.raises(PluginConfigError) as exc_info:
            MyConfig.from_dict({})  # Missing required_field

        assert "Invalid configuration for MyConfig" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_from_dict_exposes_sanitized_cause_without_model_banner(self) -> None:
        """cause omits Pydantic's model banner and docs URL."""

        class MyConfig(PluginConfig):
            required_field: str

        with pytest.raises(PluginConfigError) as exc_info:
            MyConfig.from_dict({})

        cause = exc_info.value.cause
        assert cause is not None
        assert "required_field" in cause
        assert "Field required" in cause
        assert "validation error for MyConfig" not in cause
        assert "pydantic.dev" not in cause

    def test_from_dict_success(self) -> None:
        """from_dict should return valid config on success."""

        class MyConfig(PluginConfig):
            name: str
            count: int = 10

        cfg = MyConfig.from_dict({"name": "test"})

        assert cfg.name == "test"
        assert cfg.count == 10

    def test_from_dict_with_defaults(self) -> None:
        """from_dict should use default values when not provided."""

        class MyConfig(PluginConfig):
            required: str
            optional: str = "default_value"

        cfg = MyConfig.from_dict({"required": "provided"})

        assert cfg.required == "provided"
        assert cfg.optional == "default_value"

    def test_from_dict_rejects_extra_fields(self) -> None:
        """from_dict should reject extra fields via PluginConfigError."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(PluginConfigError) as exc_info:
            MyConfig.from_dict({"name": "test", "typo_field": "value"})

        assert "Invalid configuration for MyConfig" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("payload", "type_name"),
        [
            (None, "NoneType"),
            (123, "int"),
            ("not a dict", "str"),
            (["not", "a", "dict"], "list"),
        ],
    )
    def test_from_dict_rejects_non_dict_inputs(self, payload: object, type_name: str) -> None:
        """from_dict should raise PluginConfigError for non-dict input types."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(PluginConfigError, match=rf"config must be a dict, got {type_name}"):
            MyConfig.from_dict(payload)


class TestPathConfig:
    """Tests for PathConfig base class."""

    def test_rejects_empty_path(self) -> None:
        """Empty path should raise validation error."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        with pytest.raises(ValidationError) as exc_info:
            FileConfig(path="", schema_config=DYNAMIC_SCHEMA)

        assert "path cannot be empty" in str(exc_info.value)

    def test_rejects_whitespace_only_path(self) -> None:
        """Whitespace-only path should raise validation error."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        with pytest.raises(ValidationError) as exc_info:
            FileConfig(path="   ", schema_config=DYNAMIC_SCHEMA)

        assert "path cannot be empty" in str(exc_info.value)

    def test_accepts_valid_path(self) -> None:
        """Valid path should be accepted."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        cfg = FileConfig(path="/path/to/file.csv", schema_config=DYNAMIC_SCHEMA)

        assert cfg.path == "/path/to/file.csv"

    def test_resolved_path_absolute(self) -> None:
        """Absolute path should not change when resolved."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        cfg = FileConfig(path="/absolute/path.csv", schema_config=DYNAMIC_SCHEMA)
        result = cfg.resolved_path()

        assert result == Path("/absolute/path.csv")

    def test_resolved_path_absolute_ignores_base_dir(self) -> None:
        """Absolute path should ignore base_dir when provided."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        cfg = FileConfig(path="/absolute/path.csv", schema_config=DYNAMIC_SCHEMA)
        result = cfg.resolved_path(base_dir=Path("/other/base"))

        assert result == Path("/absolute/path.csv")

    def test_resolved_path_relative_without_base(self) -> None:
        """Relative path without base_dir should return as-is."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        cfg = FileConfig(path="relative/path.csv", schema_config=DYNAMIC_SCHEMA)
        result = cfg.resolved_path()

        assert result == Path("relative/path.csv")

    def test_resolved_path_relative_with_base(self) -> None:
        """Relative path should be resolved against base_dir."""

        class FileConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"

        cfg = FileConfig(path="relative/path.csv", schema_config=DYNAMIC_SCHEMA)
        result = cfg.resolved_path(base_dir=Path("/base"))

        assert result == Path("/base/relative/path.csv")

    def test_path_config_with_additional_fields(self) -> None:
        """PathConfig subclass can have additional validated fields."""

        class CSVConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "sink"
            delimiter: str = ","
            encoding: str = "utf-8"

        cfg = CSVConfig(
            path="data.csv",
            delimiter=";",
            encoding="latin-1",
            schema_config=DYNAMIC_SCHEMA,
        )

        assert cfg.path == "data.csv"
        assert cfg.delimiter == ";"
        assert cfg.encoding == "latin-1"

    def test_path_config_rejects_extra_fields(self) -> None:
        """PathConfig subclass should still reject extra fields."""

        class CSVConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "sink"
            delimiter: str = ","

        with pytest.raises(ValidationError):
            CSVConfig(path="data.csv", schema_config=DYNAMIC_SCHEMA, unknown="value")  # type: ignore[call-arg]


class TestPluginConfigInheritance:
    """Tests for config inheritance patterns."""

    def test_deep_inheritance(self) -> None:
        """Multiple levels of inheritance should work correctly."""

        class MiddleConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"
            compression: str = "none"

        class SpecificConfig(MiddleConfig):
            format_version: int = 1

        cfg = SpecificConfig(
            path="data.bin",
            compression="gzip",
            format_version=2,
            schema_config=DYNAMIC_SCHEMA,
        )

        assert cfg.path == "data.bin"
        assert cfg.compression == "gzip"
        assert cfg.format_version == 2
        assert cfg.resolved_path() == Path("data.bin")

    def test_from_dict_returns_correct_subclass_type(self) -> None:
        """from_dict should return instance of the subclass, not base class."""

        class SpecificConfig(PathConfig):
            _plugin_component_type: ClassVar[str | None] = "source"
            custom_field: str = "default"

        cfg = SpecificConfig.from_dict(
            {
                "path": "test.csv",
                "custom_field": "custom",
                "schema": {"mode": "observed"},
            }
        )

        assert isinstance(cfg, SpecificConfig)
        assert cfg.custom_field == "custom"


class TestPluginConfigFromDictSchemaGuard:
    """Tests for PluginConfig.from_dict schema type guard."""

    @pytest.mark.parametrize(
        ("schema_value", "type_name"),
        [
            (None, "NoneType"),
            ("observed", "str"),
            (["mode", "observed"], "list"),
            (42, "int"),
        ],
    )
    def test_from_dict_rejects_non_dict_schema(self, schema_value: object, type_name: str) -> None:
        """from_dict raises PluginConfigError when schema is not a dict."""

        class MyConfig(PluginConfig):
            name: str

        with pytest.raises(PluginConfigError, match=rf"'schema' must be a dict, got {type_name}"):
            MyConfig.from_dict({"name": "test", "schema": schema_value})


class TestPluginConfigWithSchema:
    """Tests for schema in plugin config."""

    def test_plugin_config_accepts_schema(self) -> None:
        """PluginConfig can have schema section."""
        from elspeth.plugins.infrastructure.config_base import PluginConfig

        class TestConfig(PluginConfig):
            name: str

        config = TestConfig.from_dict(
            {
                "name": "test",
                "schema": {"mode": "observed"},
            }
        )
        assert config.name == "test"
        assert config.schema_config is not None
        assert config.schema_config.is_observed is True

    def test_plugin_config_schema_optional_by_default(self) -> None:
        """Schema is optional in base PluginConfig."""
        from elspeth.plugins.infrastructure.config_base import PluginConfig

        class TestConfig(PluginConfig):
            name: str

        config = TestConfig.from_dict({"name": "test"})
        assert config.name == "test"
        assert config.schema_config is None

    def test_data_plugin_config_requires_schema(self) -> None:
        """DataPluginConfig (for sources/sinks) requires schema."""

        class SourceConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "source"
            path: str

        # Should fail without schema - from_dict wraps in PluginConfigError
        # Error message uses alias "schema" not field name "schema_config"
        with pytest.raises(PluginConfigError, match=r"schema[\s\S]*Field required"):
            SourceConfig.from_dict({"path": "data.csv"})

        # Should succeed with schema
        config = SourceConfig.from_dict(
            {
                "path": "data.csv",
                "schema": {"mode": "observed"},
            }
        )
        assert config.schema_config is not None

    def test_data_plugin_config_with_explicit_schema(self) -> None:
        """DataPluginConfig accepts explicit schema definition."""

        class SourceConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "source"
            path: str

        config = SourceConfig.from_dict(
            {
                "path": "data.csv",
                "schema": {
                    "mode": "fixed",
                    "fields": ["id: int", "name: str"],
                },
            }
        )
        assert config.schema_config is not None
        assert config.schema_config.mode == "fixed"
        assert config.schema_config.fields is not None  # fixed mode has fields
        assert len(config.schema_config.fields) == 2


class TestSourceDataConfig:
    """Tests for SourceDataConfig with on_validation_failure."""

    def test_on_validation_failure_required(self) -> None:
        """on_validation_failure must be explicitly specified."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        with pytest.raises(ValidationError) as exc_info:
            SourceDataConfig(  # type: ignore[call-arg]  # testing missing required arg
                path="data.csv",
                schema_config=SchemaConfig.from_dict({"mode": "observed"}),
                # on_validation_failure is missing - should fail
            )

        # Should fail because on_validation_failure is required
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors if e["loc"]]
        assert "on_validation_failure" in field_names

    def test_on_validation_failure_accepts_sink_name(self) -> None:
        """on_validation_failure accepts a sink name."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        config = SourceDataConfig(
            path="data.csv",
            schema_config=SchemaConfig.from_dict({"fields": ["id: int", "name: str"], "mode": "fixed"}),
            on_validation_failure="quarantine_sink",
        )

        assert config.on_validation_failure == "quarantine_sink"

    def test_on_validation_failure_accepts_discard(self) -> None:
        """on_validation_failure accepts 'discard' for explicit /dev/null."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        config = SourceDataConfig(
            path="data.csv",
            schema_config=SchemaConfig.from_dict({"fields": ["id: int"], "mode": "flexible"}),
            on_validation_failure="discard",
        )

        assert config.on_validation_failure == "discard"

    def test_on_validation_failure_rejects_empty_string(self) -> None:
        """on_validation_failure rejects empty string."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        with pytest.raises(ValidationError):
            SourceDataConfig(
                path="data.csv",
                schema_config=SchemaConfig.from_dict({"fields": ["id: int"], "mode": "fixed"}),
                on_validation_failure="",
            )

    def test_source_config_inherits_path_and_schema(self) -> None:
        """SourceDataConfig inherits path and schema_config requirements."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        config = SourceDataConfig(
            path="data/input.csv",
            schema_config=SchemaConfig.from_dict({"fields": ["name: str"], "mode": "fixed"}),
            on_validation_failure="bad_rows",
        )

        assert config.path == "data/input.csv"
        assert config.schema_config is not None
        assert config.schema_config.mode == "fixed"


class TestSinkPathConfigHeaders:
    """Tests for SinkPathConfig header output validation."""

    def test_custom_headers_reject_non_normalized_mapping_keys(self) -> None:
        """CUSTOM header mappings must use normalized field names as keys."""
        with pytest.raises(
            PluginConfigError,
            match=r"headers custom mapping keys\[0\].*not a valid Python identifier",
        ):
            SinkPathConfig.from_dict(
                {
                    "path": "out.csv",
                    "schema": {"mode": "observed"},
                    "headers": {"Customer ID": "CID"},
                }
            )

    def test_custom_headers_reject_blank_output_names(self) -> None:
        """CUSTOM header mappings must not emit blank or whitespace-only headers."""
        with pytest.raises(
            PluginConfigError,
            match=r"headers custom mapping value for 'customer_id' must not be empty or whitespace-only",
        ):
            SinkPathConfig.from_dict(
                {
                    "path": "out.csv",
                    "schema": {"mode": "observed"},
                    "headers": {"customer_id": "   "},
                }
            )


class TestAzureAuthConfigValidator:
    """Tests for AzureAuthConfig model validator — mutual exclusivity and missing fields."""

    def test_no_auth_method_raises(self) -> None:
        """No auth method configured raises ValueError."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match="No authentication method configured"):
            AzureAuthConfig()

    def test_multiple_auth_methods_raises(self) -> None:
        """Multiple auth methods configured raises ValueError."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match="Multiple authentication methods configured"):
            AzureAuthConfig(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test",
                sas_token="sv=2020-08-04&ss=b",
                account_url="https://test.blob.core.windows.net",
            )

    def test_sas_token_without_account_url_raises(self) -> None:
        """SAS token without account_url raises ValueError."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match="SAS token auth requires account_url"):
            AzureAuthConfig(sas_token="sv=2020-08-04&ss=b")

    def test_managed_identity_without_account_url_raises(self) -> None:
        """Managed Identity without account_url raises ValueError."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match="Managed Identity auth requires account_url"):
            AzureAuthConfig(use_managed_identity=True)

    def test_partial_service_principal_raises(self) -> None:
        """Incomplete service principal config raises ValueError listing missing fields."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match=r"Service Principal auth requires all fields.*Missing.*client_secret"):
            AzureAuthConfig(
                tenant_id="tenant-123",
                client_id="client-456",
                account_url="https://test.blob.core.windows.net",
            )

    def test_service_principal_without_account_url_raises(self) -> None:
        """Service principal with all 3 SP fields but no account_url raises ValueError."""
        from pydantic import ValidationError

        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        with pytest.raises(ValidationError, match="Service Principal auth requires account_url"):
            AzureAuthConfig(
                tenant_id="tenant-123",
                client_id="client-456",
                client_secret="secret-789",
            )

    def test_valid_connection_string_auth(self) -> None:
        """Valid connection string auth succeeds."""
        from elspeth.plugins.infrastructure.azure_auth import AzureAuthConfig

        config = AzureAuthConfig(connection_string="DefaultEndpointsProtocol=https;AccountName=test")
        assert config.auth_method == "connection_string"


class TestTransformDataConfig:
    """Tests for TransformDataConfig.

    Note: on_error/on_success routing is now at the settings level
    (TransformSettings in core/config.py), not in plugin config.
    Routing properties are on BaseTransform, injected by the
    instantiation bridge in cli_helpers.py.
    """

    def test_transform_config_accepts_schema(self) -> None:
        """TransformDataConfig accepts schema_config."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        config = TransformDataConfig(
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )

        assert config.schema_config is not None
        assert config.schema_config.is_observed is True

    def test_transform_config_rejects_routing_fields(self) -> None:
        """on_error/on_success are not config-level fields (extra=forbid)."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        with pytest.raises(ValidationError):
            TransformDataConfig(
                schema_config=SchemaConfig.from_dict({"mode": "observed"}),
                on_error="sink",  # type: ignore[call-arg]
            )

    def test_transform_config_accepts_required_input_fields(self) -> None:
        """TransformDataConfig accepts required_input_fields declaration."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        config = TransformDataConfig(
            schema_config=SchemaConfig.from_dict({"mode": "fixed", "fields": ["id: int"]}),
            required_input_fields=["id", "name"],
        )

        assert config.required_input_fields == ["id", "name"]

    def test_transform_config_required_input_fields_optional(self) -> None:
        """required_input_fields defaults to None when not specified."""
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        config = TransformDataConfig(
            schema_config=SchemaConfig.from_dict({"mode": "fixed", "fields": ["id: int"]}),
        )

        assert config.required_input_fields is None

    def test_transform_config_inherits_schema_requirement(self) -> None:
        """TransformDataConfig inherits schema requirement from DataPluginConfig."""
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        with pytest.raises(ValidationError):
            TransformDataConfig()  # type: ignore[call-arg]  # Missing schema_config - testing runtime validation


class TestComponentTypeEnforcement:
    """Tests for __init_subclass__ enforcement of _plugin_component_type.

    DataPluginConfig subclasses must set _plugin_component_type to a non-None
    value (either directly or via an intermediate base). Only explicitly exempt
    intermediate bases (PathConfig, DataPluginConfig) may leave it unset.
    """

    def test_direct_subclass_without_component_type_raises(self) -> None:
        """Direct DataPluginConfig subclass missing _plugin_component_type raises TypeError."""
        with pytest.raises(TypeError, match="does not set _plugin_component_type"):

            class BadConfig(DataPluginConfig):
                pass

    def test_direct_subclass_with_component_type_succeeds(self) -> None:
        """Direct DataPluginConfig subclass with _plugin_component_type succeeds."""

        class GoodConfig(DataPluginConfig):
            _plugin_component_type: ClassVar[str | None] = "transform"

        assert GoodConfig._plugin_component_type == "transform"

    def test_path_config_subclass_without_component_type_raises(self) -> None:
        """PathConfig subclass missing _plugin_component_type raises TypeError.

        PathConfig is exempt, but its exemption does NOT inherit — subclasses
        must set _plugin_component_type themselves.
        """
        with pytest.raises(TypeError, match="does not set _plugin_component_type"):

            class BadPathChild(PathConfig):
                pass

    def test_inherited_component_type_from_intermediate_base(self) -> None:
        """Subclass inheriting _plugin_component_type from SourceDataConfig succeeds."""
        from elspeth.plugins.infrastructure.config_base import SourceDataConfig

        class GoodSource(SourceDataConfig):
            extra_field: str = "default"

        assert GoodSource._plugin_component_type == "source"

    def test_inherited_component_type_through_chain(self) -> None:
        """Deeply nested subclass inheriting through chain succeeds."""
        from elspeth.plugins.infrastructure.config_base import TransformDataConfig

        class MiddleTransform(TransformDataConfig):
            buffer_size: int = 100

        class LeafTransform(MiddleTransform):
            mode: str = "fast"

        assert LeafTransform._plugin_component_type == "transform"

    def test_exempt_base_does_not_propagate_exemption(self) -> None:
        """_component_type_exempt on PathConfig does not propagate to its children.

        This is the critical invariant: exemption is per-class (checked via
        cls.__dict__), not inherited through the MRO.
        """
        # PathConfig itself is exempt — defining it didn't raise
        assert "_component_type_exempt" in PathConfig.__dict__

        # But a subclass of PathConfig is NOT exempt unless it sets the field
        with pytest.raises(TypeError, match="does not set _plugin_component_type"):

            class NotExempt(PathConfig):
                pass

    def test_error_message_includes_class_name(self) -> None:
        """TypeError message includes the offending class name for debuggability."""
        with pytest.raises(TypeError, match="MissingTypeConfig") as exc_info:

            class MissingTypeConfig(DataPluginConfig):
                pass

        assert "source" in str(exc_info.value) or "sink" in str(exc_info.value) or "transform" in str(exc_info.value)

    def test_all_real_config_classes_have_component_type(self) -> None:
        """Smoke test: all production DataPluginConfig subclasses resolve component_type.

        This catches any class that somehow bypassed the __init_subclass__ hook
        (e.g. defined before the hook was added and cached in .pyc).
        """
        from elspeth.plugins.infrastructure.config_base import (
            SinkPathConfig,
            SourceDataConfig,
            TransformDataConfig,
        )

        for cls in (SourceDataConfig, SinkPathConfig, TransformDataConfig):
            assert cls._plugin_component_type is not None, f"{cls.__name__}._plugin_component_type is None"
