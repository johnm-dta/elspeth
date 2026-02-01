"""Audit tests for plugin schema contracts.

Verifies plugins follow schema initialization contract.
Critical for new architecture - plugins MUST set schemas in __init__().

IMPORTANT: This test uses direct attribute access (not hasattr) per CLAUDE.md
prohibition on defensive patterns. If a plugin is missing an expected attribute,
that's a bug to fix, not an error to silently skip.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.plugins.manager import PluginManager

# =============================================================================
# MINIMAL VALID CONFIGS PER PLUGIN
# =============================================================================
# Each plugin requires specific config. We provide the minimum valid config
# to instantiate the plugin and verify its schema contracts.
#
# Plugins requiring external credentials use pytest.skip with explanation.
# =============================================================================


# Sources: require path (or equivalent) + schema + on_validation_failure
SOURCE_CONFIGS: dict[str, dict[str, Any] | None] = {
    "csv": {
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    },
    "json": {
        "path": "/tmp/test.json",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    },
    "null": {
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    },
    # Azure Blob Source requires valid Azure credentials - cannot test without mocking
    "azure_blob": None,
}


# Transforms: require schema at minimum, some require additional config
TRANSFORM_CONFIGS: dict[str, dict[str, Any] | None] = {
    "passthrough": {
        "schema": {"fields": "dynamic"},
    },
    "batch_replicate": {
        # Uses copies_field (default "copies"), default_copies, include_copy_index
        "schema": {"fields": "dynamic"},
    },
    "batch_stats": {
        # Requires value_field (no default)
        "schema": {"fields": "dynamic"},
        "value_field": "amount",
    },
    "keyword_filter": {
        # Requires fields (str or list) and blocked_patterns (list of regex)
        "schema": {"fields": "dynamic"},
        "fields": "text",
        "blocked_patterns": ["test_pattern"],
    },
    "truncate": {
        # Uses fields dict of field_name -> max_length
        "schema": {"fields": "dynamic"},
        "fields": {"text": 100},
    },
    "json_explode": {
        # Requires array_field (the field to explode)
        "schema": {"fields": "dynamic"},
        "array_field": "items",
    },
    "field_mapper": {
        # Uses mapping dict of source_field -> target_field
        "schema": {"fields": "dynamic"},
        "mapping": {"old_name": "new_name"},
    },
    # Azure transforms require valid Azure credentials
    "azure_content_safety": None,
    "azure_prompt_shield": None,
    # LLM transforms require valid API credentials
    "azure_llm": None,
    "azure_batch_llm": None,
    "azure_multi_query_llm": None,
    "openrouter_llm": None,
    "openrouter_multi_query_llm": None,
    "openrouter_batch_llm": None,
}


# Sinks: require path (or equivalent) + schema
# Note: CSVSink requires strict schema (fixed columns), JSONSink allows dynamic
# Each config includes a test_row that matches the schema for validation testing
SINK_CONFIGS: dict[str, dict[str, Any] | None] = {
    "csv": {
        "path": "/tmp/output.csv",
        "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
        "test_row": {"id": 1, "name": "test"},  # Matches strict schema
    },
    "json": {
        "path": "/tmp/output.json",
        "schema": {"fields": "dynamic"},
        "test_row": {"test_field": "test_value"},  # Dynamic accepts any fields
    },
    # Database sink requires valid database URL
    "database": None,
    # Azure Blob Sink requires valid Azure credentials
    "azure_blob": None,
}


# =============================================================================
# SOURCE SCHEMA CONTRACT TESTS
# =============================================================================


class TestSourceSchemaContracts:
    """Verify all source plugins set output_schema in __init__()."""

    @pytest.fixture
    def manager(self) -> PluginManager:
        """Create plugin manager with built-in plugins registered."""
        mgr = PluginManager()
        mgr.register_builtin_plugins()
        return mgr

    def test_all_sources_have_explicit_config(self, manager: PluginManager) -> None:
        """Every registered source must have an entry in SOURCE_CONFIGS.

        This prevents new sources from being silently untested.
        """
        for plugin_cls in manager.get_sources():
            assert plugin_cls.name in SOURCE_CONFIGS, (
                f"Source '{plugin_cls.name}' is not in SOURCE_CONFIGS. "
                f"Add explicit config or None (with skip reason) to test schema contracts."
            )

    @pytest.mark.parametrize(
        "source_name",
        [name for name, cfg in SOURCE_CONFIGS.items() if cfg is not None],
    )
    def test_source_sets_output_schema(self, manager: PluginManager, source_name: str) -> None:
        """Source plugins must set output_schema in __init__().

        Tests:
        1. Plugin can be instantiated with minimal config
        2. output_schema attribute is set (not using hasattr - direct access)
        3. output_schema is a PluginSchema subclass with model_validate method
        """
        plugin_cls = manager.get_source_by_name(source_name)
        config = SOURCE_CONFIGS[source_name]
        assert config is not None  # Guaranteed by parametrize filter

        # Instantiate plugin - if this fails, that's a bug in our config
        instance = plugin_cls(config)

        # CRITICAL: output_schema must be set
        # Direct access - if attribute missing, test fails (no hasattr hiding)
        schema = instance.output_schema

        # Schema must be a PluginSchema subclass (Pydantic model)
        # Dynamic schemas are still PluginSchema subclasses, never None
        assert schema is not None, (
            f"Source '{source_name}' has output_schema=None. Dynamic schemas must still be PluginSchema subclasses, not None."
        )

        # Behavioral validation: schema must be usable
        # This catches cases where schema is set to wrong type
        assert hasattr(schema, "model_validate"), (
            f"Source '{source_name}' output_schema is not a Pydantic model. Got {type(schema).__name__} instead of PluginSchema subclass."
        )

        # Actually validate a row (proves schema works)
        schema.model_validate({"test_field": "test_value"})

    @pytest.mark.parametrize(
        "source_name,reason",
        [
            ("azure_blob", "Requires Azure Storage credentials"),
        ],
    )
    def test_source_requires_credentials_skip(self, manager: PluginManager, source_name: str, reason: str) -> None:
        """Document sources that require external credentials.

        These are intentionally skipped in unit tests. Integration tests
        with proper credentials should verify their schema contracts.
        """
        pytest.skip(f"Source '{source_name}' cannot be tested without credentials: {reason}")


# =============================================================================
# TRANSFORM SCHEMA CONTRACT TESTS
# =============================================================================


class TestTransformSchemaContracts:
    """Verify all transform plugins set input/output schemas in __init__()."""

    @pytest.fixture
    def manager(self) -> PluginManager:
        """Create plugin manager with built-in plugins registered."""
        mgr = PluginManager()
        mgr.register_builtin_plugins()
        return mgr

    def test_all_transforms_have_explicit_config(self, manager: PluginManager) -> None:
        """Every registered transform must have an entry in TRANSFORM_CONFIGS.

        This prevents new transforms from being silently untested.
        """
        for plugin_cls in manager.get_transforms():
            assert plugin_cls.name in TRANSFORM_CONFIGS, (
                f"Transform '{plugin_cls.name}' is not in TRANSFORM_CONFIGS. "
                f"Add explicit config or None (with skip reason) to test schema contracts."
            )

    @pytest.mark.parametrize(
        "transform_name",
        [name for name, cfg in TRANSFORM_CONFIGS.items() if cfg is not None],
    )
    def test_transform_sets_both_schemas(self, manager: PluginManager, transform_name: str) -> None:
        """Transform plugins must set input_schema and output_schema in __init__().

        Tests:
        1. Plugin can be instantiated with minimal config
        2. Both schema attributes are set (direct access, no hasattr)
        3. Both schemas are PluginSchema subclasses with model_validate method
        """
        plugin_cls = manager.get_transform_by_name(transform_name)
        config = TRANSFORM_CONFIGS[transform_name]
        assert config is not None  # Guaranteed by parametrize filter

        # Instantiate plugin
        instance = plugin_cls(config)

        # CRITICAL: Both schemas must be set
        # Direct access - no hasattr hiding
        input_schema = instance.input_schema
        output_schema = instance.output_schema

        # Schemas must be PluginSchema subclasses, not None
        assert input_schema is not None, (
            f"Transform '{transform_name}' has input_schema=None. Dynamic schemas must still be PluginSchema subclasses."
        )
        assert output_schema is not None, (
            f"Transform '{transform_name}' has output_schema=None. Dynamic schemas must still be PluginSchema subclasses."
        )

        # Behavioral validation
        assert hasattr(input_schema, "model_validate"), f"Transform '{transform_name}' input_schema is not a Pydantic model."
        assert hasattr(output_schema, "model_validate"), f"Transform '{transform_name}' output_schema is not a Pydantic model."

        # Actually validate rows
        input_schema.model_validate({"test_field": "test_value"})
        output_schema.model_validate({"test_field": "test_value"})

    @pytest.mark.parametrize(
        "transform_name,reason",
        [
            ("azure_content_safety", "Requires Azure Content Safety endpoint and API key"),
            ("azure_prompt_shield", "Requires Azure Prompt Shield endpoint and API key"),
            ("azure_llm", "Requires Azure OpenAI deployment and API key"),
            ("azure_batch_llm", "Requires Azure OpenAI deployment and API key"),
            ("azure_multi_query_llm", "Requires Azure OpenAI deployment and API key"),
            ("openrouter_llm", "Requires OpenRouter API key"),
            ("openrouter_multi_query_llm", "Requires OpenRouter API key"),
        ],
    )
    def test_transform_requires_credentials_skip(self, manager: PluginManager, transform_name: str, reason: str) -> None:
        """Document transforms that require external credentials.

        These are intentionally skipped in unit tests. Integration tests
        with proper credentials should verify their schema contracts.
        """
        pytest.skip(f"Transform '{transform_name}' cannot be tested without credentials: {reason}")


# =============================================================================
# SINK SCHEMA CONTRACT TESTS
# =============================================================================


class TestSinkSchemaContracts:
    """Verify all sink plugins set input_schema in __init__()."""

    @pytest.fixture
    def manager(self) -> PluginManager:
        """Create plugin manager with built-in plugins registered."""
        mgr = PluginManager()
        mgr.register_builtin_plugins()
        return mgr

    def test_all_sinks_have_explicit_config(self, manager: PluginManager) -> None:
        """Every registered sink must have an entry in SINK_CONFIGS.

        This prevents new sinks from being silently untested.
        """
        for plugin_cls in manager.get_sinks():
            assert plugin_cls.name in SINK_CONFIGS, (
                f"Sink '{plugin_cls.name}' is not in SINK_CONFIGS. Add explicit config or None (with skip reason) to test schema contracts."
            )

    @pytest.mark.parametrize(
        "sink_name",
        [name for name, cfg in SINK_CONFIGS.items() if cfg is not None],
    )
    def test_sink_sets_input_schema(self, manager: PluginManager, sink_name: str) -> None:
        """Sink plugins must set input_schema in __init__().

        Tests:
        1. Plugin can be instantiated with minimal config
        2. input_schema attribute is set (direct access, no hasattr)
        3. input_schema is a PluginSchema subclass with model_validate method
        """
        plugin_cls = manager.get_sink_by_name(sink_name)
        full_config = SINK_CONFIGS[sink_name]
        assert full_config is not None  # Guaranteed by parametrize filter

        # Extract test_row before passing config to plugin
        test_row = full_config.get("test_row", {"test_field": "test_value"})
        config = {k: v for k, v in full_config.items() if k != "test_row"}

        # Instantiate plugin
        instance = plugin_cls(config)

        # CRITICAL: input_schema must be set
        schema = instance.input_schema

        # Schema must be PluginSchema subclass, not None
        assert schema is not None, f"Sink '{sink_name}' has input_schema=None. Dynamic schemas must still be PluginSchema subclasses."

        # Behavioral validation
        assert hasattr(schema, "model_validate"), f"Sink '{sink_name}' input_schema is not a Pydantic model."

        # Actually validate a row (using test_row that matches the schema)
        schema.model_validate(test_row)

    @pytest.mark.parametrize(
        "sink_name,reason",
        [
            ("database", "Requires valid database URL"),
            ("azure_blob", "Requires Azure Storage credentials"),
        ],
    )
    def test_sink_requires_credentials_skip(self, manager: PluginManager, sink_name: str, reason: str) -> None:
        """Document sinks that require external credentials.

        These are intentionally skipped in unit tests. Integration tests
        with proper credentials should verify their schema contracts.
        """
        pytest.skip(f"Sink '{sink_name}' cannot be tested without credentials: {reason}")


# =============================================================================
# PLUGIN INIT I/O SAFETY TEST
# =============================================================================


class TestPluginInitSafety:
    """Verify plugins don't perform I/O in __init__() (brittle validation risk)."""

    def test_csv_source_does_not_perform_io_in_init(self) -> None:
        """CSVSource __init__() should not try to open the file.

        This addresses Systems Thinking concern about validation brittleness.
        Plugins should delay I/O until execute time, not __init__().
        """
        manager = PluginManager()
        manager.register_builtin_plugins()

        csv_source_cls = manager.get_source_by_name("csv")

        # Use nonexistent file - if __init__() tries to open it, this crashes
        instance = csv_source_cls(
            {
                "path": "/nonexistent/file/that/does/not/exist.csv",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
            }
        )

        # Schema should still be set even for nonexistent file
        # Direct access - no hasattr
        schema = instance.output_schema
        assert schema is not None
        schema.model_validate({"test": "value"})

    def test_json_source_does_not_perform_io_in_init(self) -> None:
        """JSONSource __init__() should not try to open the file."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        json_source_cls = manager.get_source_by_name("json")

        instance = json_source_cls(
            {
                "path": "/nonexistent/file/that/does/not/exist.json",
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "quarantine",
            }
        )

        schema = instance.output_schema
        assert schema is not None
        schema.model_validate({"test": "value"})

    def test_csv_sink_does_not_perform_io_in_init(self) -> None:
        """CSVSink __init__() should not try to create the file."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        csv_sink_cls = manager.get_sink_by_name("csv")

        # CSVSink requires strict schema (fixed columns)
        instance = csv_sink_cls(
            {
                "path": "/nonexistent/directory/that/does/not/exist/output.csv",
                "schema": {"mode": "strict", "fields": ["id: int", "name: str"]},
            }
        )

        schema = instance.input_schema
        assert schema is not None
        # Validate with row matching the strict schema
        schema.model_validate({"id": 1, "name": "test"})

    def test_json_sink_does_not_perform_io_in_init(self) -> None:
        """JSONSink __init__() should not try to create the file."""
        manager = PluginManager()
        manager.register_builtin_plugins()

        json_sink_cls = manager.get_sink_by_name("json")

        instance = json_sink_cls(
            {
                "path": "/nonexistent/directory/that/does/not/exist/output.json",
                "schema": {"fields": "dynamic"},
            }
        )

        schema = instance.input_schema
        assert schema is not None
        schema.model_validate({"test": "value"})
