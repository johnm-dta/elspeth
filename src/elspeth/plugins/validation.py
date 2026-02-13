"""Plugin configuration validation subsystem.

Validates plugin configurations BEFORE instantiation, providing clear error
messages and enabling test fixtures to bypass validation when needed.

Design:
- Validation is separate from plugin construction
- Returns structured errors (not exceptions) for better error messages
- Validates against Pydantic config models (CSVSourceConfig, etc.)
- Does NOT instantiate plugins (just validates config)

Usage:
    validator = PluginConfigValidator()
    errors = validator.validate_source_config("csv", config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    source = CSVSource(config)  # Assumes config is valid
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as PydanticValidationError

from elspeth.plugins.config_base import PluginConfigError

if TYPE_CHECKING:
    from elspeth.plugins.config_base import PluginConfig


@dataclass
class ValidationError:
    """Structured validation error.

    Attributes:
        field: Field name that failed validation
        message: Human-readable error message
        value: The invalid value (for debugging)
    """

    field: str
    message: str
    value: Any


class PluginConfigValidator:
    """Validates plugin configurations before instantiation.

    Validates configs against Pydantic models (CSVSourceConfig, etc.)
    without actually instantiating the plugin.
    """

    def validate_source_config(
        self,
        source_type: str,
        config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate source plugin configuration.

        Args:
            source_type: Plugin type name (e.g., "csv", "json")
            config: Plugin configuration dict

        Returns:
            List of validation errors (empty if valid)
        """
        # Get config model for source type
        config_model = self._get_source_config_model(source_type)

        # Handle special case: null_source has no config class
        if config_model is None:
            return []  # No validation needed

        # Validate using Pydantic
        try:
            config_model.from_dict(config)
            return []  # Valid
        except PydanticValidationError as e:
            return self._extract_errors(e)
        except PluginConfigError as e:
            return self._extract_wrapped_plugin_config_error(e, config)

    def _get_source_config_model(self, source_type: str) -> type["PluginConfig"] | None:
        """Get Pydantic config model for source type.

        Returns:
            Config model class, or None for sources with no config (e.g., null_source)
        """
        # Import here to avoid circular dependencies
        if source_type == "csv":
            from elspeth.plugins.sources.csv_source import CSVSourceConfig

            return CSVSourceConfig
        elif source_type == "json":
            from elspeth.plugins.sources.json_source import JSONSourceConfig

            return JSONSourceConfig
        elif source_type == "azure_blob":
            from elspeth.plugins.azure.blob_source import AzureBlobSourceConfig

            return AzureBlobSourceConfig
        elif source_type == "null":
            # NullSource has no config class (resume-only source)
            # Return None to signal "no validation needed"
            return None
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def validate_transform_config(
        self,
        transform_type: str,
        config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate transform plugin configuration.

        Args:
            transform_type: Plugin type name (e.g., "passthrough", "field_mapper")
            config: Plugin configuration dict

        Returns:
            List of validation errors (empty if valid)
        """
        # Get config model for transform type
        config_model = self._get_transform_config_model(transform_type)

        # Validate using Pydantic
        try:
            config_model.from_dict(config)
            return []  # Valid
        except PydanticValidationError as e:
            return self._extract_errors(e)
        except PluginConfigError as e:
            return self._extract_wrapped_plugin_config_error(e, config)

    def validate_sink_config(
        self,
        sink_type: str,
        config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate sink plugin configuration.

        Args:
            sink_type: Plugin type name (e.g., "csv", "json")
            config: Plugin configuration dict

        Returns:
            List of validation errors (empty if valid)
        """
        # Get config model for sink type
        config_model = self._get_sink_config_model(sink_type)

        # Validate using Pydantic
        try:
            config_model.from_dict(config)
            return []  # Valid
        except PydanticValidationError as e:
            return self._extract_errors(e)
        except PluginConfigError as e:
            return self._extract_wrapped_plugin_config_error(e, config)

    def _extract_wrapped_plugin_config_error(
        self,
        error: PluginConfigError,
        config: dict[str, object],
    ) -> list[ValidationError]:
        """Convert wrapped PluginConfigError causes into structured errors.

        PluginConfig.from_dict() wraps:
        - PydanticValidationError for model-level validation failures
        - ValueError for schema parsing failures before model validation
        """
        cause = error.__cause__

        if type(cause) is PydanticValidationError:
            return self._extract_errors(cause)

        if type(cause) is ValueError:
            if "schema" in config:
                return [ValidationError(field="schema", message=str(cause), value=config["schema"])]
            return [ValidationError(field="config", message=str(cause), value=config)]

        raise error

    def validate_schema_config(
        self,
        schema_config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate schema configuration independently of plugin.

        Args:
            schema_config: Schema configuration dict (contents of 'schema' key)

        Returns:
            List of validation errors (empty if valid)
        """
        # Import here to avoid circular dependencies
        from elspeth.contracts.schema import SchemaConfig

        try:
            SchemaConfig.from_dict(schema_config)
            return []  # Valid
        except ValueError as e:
            # SchemaConfig.from_dict raises ValueError for invalid configs
            # Convert to structured error
            return [
                ValidationError(
                    field="schema",
                    message=str(e),
                    value=schema_config,
                )
            ]
        # NOTE: No catch-all Exception handler here.
        # SchemaConfig.from_dict() documents it raises ValueError for invalid config.
        # Any other exception (TypeError, AttributeError, etc.) is a bug in our code
        # that should crash immediately per CLAUDE.md - not be silently converted
        # to a "validation error" that hides the real problem.

    def _get_transform_config_model(self, transform_type: str) -> type["PluginConfig"]:
        """Get Pydantic config model for transform type.

        Returns:
            Config model class for the transform type
        """
        # Import here to avoid circular dependencies
        if transform_type == "passthrough":
            from elspeth.plugins.transforms.passthrough import PassThroughConfig

            return PassThroughConfig
        elif transform_type == "field_mapper":
            from elspeth.plugins.transforms.field_mapper import FieldMapperConfig

            return FieldMapperConfig
        elif transform_type == "json_explode":
            from elspeth.plugins.transforms.json_explode import JSONExplodeConfig

            return JSONExplodeConfig
        elif transform_type == "keyword_filter":
            from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

            return KeywordFilterConfig
        elif transform_type == "truncate":
            from elspeth.plugins.transforms.truncate import TruncateConfig

            return TruncateConfig
        elif transform_type == "batch_replicate":
            from elspeth.plugins.transforms.batch_replicate import BatchReplicateConfig

            return BatchReplicateConfig
        elif transform_type == "batch_stats":
            from elspeth.plugins.transforms.batch_stats import BatchStatsConfig

            return BatchStatsConfig
        elif transform_type == "azure_content_safety":
            from elspeth.plugins.transforms.azure.content_safety import AzureContentSafetyConfig

            return AzureContentSafetyConfig
        elif transform_type == "azure_prompt_shield":
            from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShieldConfig

            return AzurePromptShieldConfig
        elif transform_type == "azure_llm":
            from elspeth.plugins.llm.azure import AzureOpenAIConfig

            return AzureOpenAIConfig
        elif transform_type == "azure_batch_llm":
            from elspeth.plugins.llm.azure_batch import AzureBatchConfig

            return AzureBatchConfig
        elif transform_type == "azure_multi_query_llm":
            from elspeth.plugins.llm.multi_query import MultiQueryConfig

            return MultiQueryConfig
        elif transform_type == "openrouter_llm":
            from elspeth.plugins.llm.openrouter import OpenRouterConfig

            return OpenRouterConfig
        elif transform_type == "openrouter_batch_llm":
            from elspeth.plugins.llm.openrouter_batch import OpenRouterBatchConfig

            return OpenRouterBatchConfig
        elif transform_type == "openrouter_multi_query_llm":
            from elspeth.plugins.llm.openrouter_multi_query import OpenRouterMultiQueryConfig

            return OpenRouterMultiQueryConfig
        elif transform_type == "web_scrape":
            from elspeth.plugins.transforms.web_scrape import WebScrapeConfig

            return WebScrapeConfig
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")

    def _get_sink_config_model(self, sink_type: str) -> type["PluginConfig"]:
        """Get Pydantic config model for sink type.

        Returns:
            Config model class for the sink type
        """
        # Import here to avoid circular dependencies
        if sink_type == "csv":
            from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

            return CSVSinkConfig
        elif sink_type == "json":
            from elspeth.plugins.sinks.json_sink import JSONSinkConfig

            return JSONSinkConfig
        elif sink_type == "database":
            from elspeth.plugins.sinks.database_sink import DatabaseSinkConfig

            return DatabaseSinkConfig
        elif sink_type == "azure_blob":
            from elspeth.plugins.azure.blob_sink import AzureBlobSinkConfig

            return AzureBlobSinkConfig
        else:
            raise ValueError(f"Unknown sink type: {sink_type}")

    def _extract_errors(
        self,
        pydantic_error: PydanticValidationError,
    ) -> list[ValidationError]:
        """Convert Pydantic errors to structured ValidationError list."""
        errors: list[ValidationError] = []

        for err in pydantic_error.errors():
            # Pydantic error dict has: loc, msg, type, ctx
            field_path = ".".join(str(loc) for loc in err["loc"])
            message = err["msg"]

            # Pydantic error dict includes failing input value.
            value = err["input"]

            errors.append(
                ValidationError(
                    field=field_path,
                    message=message,
                    value=value,
                )
            )

        return errors
