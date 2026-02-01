"""Tests for plugin configuration validation subsystem."""

import pytest

from elspeth.plugins.validation import PluginConfigValidator


def test_validator_accepts_valid_source_config():
    """Valid source config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert errors == []


def test_validator_rejects_missing_required_field():
    """Missing required field returns error."""
    validator = PluginConfigValidator()

    config = {
        # Missing 'path' - required by CSVSourceConfig
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert len(errors) == 1
    assert "path" in errors[0].field
    assert "required" in errors[0].message.lower()


def test_validator_rejects_invalid_field_type():
    """Invalid field type returns error."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/test.csv",
        "skip_rows": "not_an_int",  # Should be int
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert len(errors) == 1
    assert "skip_rows" in errors[0].field


def test_validator_accepts_null_source_with_empty_config():
    """null_source has no config class, validation should pass with empty dict."""
    validator = PluginConfigValidator()

    config = {}

    errors = validator.validate_source_config("null", config)
    assert errors == []


def test_validator_accepts_null_source_with_arbitrary_config():
    """null_source ignores config, validation should pass with any dict."""
    validator = PluginConfigValidator()

    config = {
        "arbitrary_field": "ignored",
        "another_field": 42,
    }

    errors = validator.validate_source_config("null", config)
    assert errors == []


def test_validator_accepts_valid_transform_config():
    """Valid transform config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_transform_config("passthrough", config)
    assert errors == []


def test_validator_accepts_valid_sink_config():
    """Valid sink config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/output.csv",
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_sink_config("csv", config)
    assert errors == []


def test_validator_rejects_unknown_gate_type():
    """Gate validation raises error since no gate plugins exist yet."""
    validator = PluginConfigValidator()

    config = {
        "field": "score",
        "threshold": 0.5,
        "schema": {"fields": "dynamic"},
    }

    # No gate plugins exist in codebase yet, so this should raise
    with pytest.raises(ValueError) as exc_info:
        validator.validate_gate_config("threshold", config)

    assert "Unknown gate type" in str(exc_info.value)


def test_validator_rejects_invalid_transform_config():
    """Invalid transform config with missing required field returns error."""
    validator = PluginConfigValidator()

    config = {
        # Missing 'schema_config' - required by PassThroughConfig (DataPluginConfig)
    }

    errors = validator.validate_transform_config("passthrough", config)
    assert len(errors) == 1
    # DataPluginConfig validator error reports the field name that failed
    assert errors[0].field == "schema_config"


def test_validator_rejects_invalid_sink_config():
    """Invalid sink config with missing required field returns error."""
    validator = PluginConfigValidator()

    config = {
        # Missing 'url' - required by DatabaseSinkConfig
        "table": "test_table",
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_sink_config("database", config)
    assert len(errors) == 1
    assert "url" in errors[0].field
    assert "required" in errors[0].message.lower()


def test_validator_rejects_unknown_transform_type():
    """Unknown transform type raises ValueError."""
    validator = PluginConfigValidator()

    config = {
        "schema": {"fields": "dynamic"},
    }

    with pytest.raises(ValueError) as exc_info:
        validator.validate_transform_config("nonexistent_transform", config)

    assert "Unknown transform type" in str(exc_info.value)


def test_validator_rejects_unknown_sink_type():
    """Unknown sink type raises ValueError."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/output.txt",
        "schema": {"fields": "dynamic"},
    }

    with pytest.raises(ValueError) as exc_info:
        validator.validate_sink_config("nonexistent_sink", config)

    assert "Unknown sink type" in str(exc_info.value)


def test_validator_rejects_unknown_source_type():
    """Unknown source type raises ValueError."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/input.txt",
        "schema": {"fields": "dynamic"},
    }

    with pytest.raises(ValueError) as exc_info:
        validator.validate_source_config("nonexistent_source", config)

    assert "Unknown source type" in str(exc_info.value)


def test_validator_accepts_database_sink_config():
    """Valid database sink config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "url": "sqlite:///test.db",
        "table": "test_table",
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_sink_config("database", config)
    assert errors == []


def test_validator_accepts_azure_blob_source_config():
    """Valid Azure blob source config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "container": "test-container",
        "blob_path": "data/input.csv",
        "format": "csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net",
    }

    errors = validator.validate_source_config("azure_blob", config)
    assert errors == []


def test_validator_accepts_azure_blob_sink_config():
    """Valid Azure blob sink config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "container": "test-container",
        "blob_path": "output_{{ run_id }}.csv",
        "format": "csv",
        "schema": {"fields": "dynamic"},
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net",
    }

    errors = validator.validate_sink_config("azure_blob", config)
    assert errors == []


def test_validator_validates_schema_config():
    """Valid schema configs pass validation."""
    validator = PluginConfigValidator()

    # Valid dynamic schema
    schema_config = {"fields": "dynamic"}
    errors = validator.validate_schema_config(schema_config)
    assert errors == []

    # Valid explicit strict schema
    schema_config = {
        "mode": "strict",
        "fields": ["id: int", "name: str", "score: float?"],
    }
    errors = validator.validate_schema_config(schema_config)
    assert errors == []

    # Valid explicit free schema
    schema_config = {
        "mode": "free",
        "fields": ["id: int"],
    }
    errors = validator.validate_schema_config(schema_config)
    assert errors == []


def test_validator_rejects_invalid_schema_mode():
    """Invalid schema mode returns error."""
    validator = PluginConfigValidator()

    # Invalid mode value
    schema_config = {
        "mode": "invalid_mode",
        "fields": ["id: int"],
    }

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0
    # Error should mention the invalid mode
    assert any("invalid_mode" in err.message.lower() or "mode" in err.message.lower() for err in errors)


def test_validator_rejects_schema_missing_fields():
    """Schema missing required 'fields' key returns error."""
    validator = PluginConfigValidator()

    schema_config = {"mode": "strict"}  # Missing 'fields'

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0
    assert any("fields" in err.message.lower() for err in errors)


def test_validator_rejects_schema_empty_fields():
    """Schema with empty fields list returns error."""
    validator = PluginConfigValidator()

    schema_config = {
        "mode": "strict",
        "fields": [],  # Empty list
    }

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0


def test_validator_rejects_schema_invalid_field_type():
    """Schema with wrong type for 'fields' returns error."""
    validator = PluginConfigValidator()

    schema_config = {
        "mode": "strict",
        "fields": "not_a_list",  # Should be list
    }

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0


def test_validator_rejects_malformed_field_spec():
    """Schema with malformed field spec returns error."""
    validator = PluginConfigValidator()

    schema_config = {
        "mode": "strict",
        "fields": ["id-int"],  # Wrong format, should be "id: int"
    }

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0
