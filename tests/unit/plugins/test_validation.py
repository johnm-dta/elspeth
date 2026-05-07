"""Tests for plugin configuration validation subsystem."""

import pytest

from elspeth.plugins.infrastructure.validation import (
    validate_schema_config,
    validate_sink_config,
    validate_source_config,
    validate_transform_config,
)


def test_validator_accepts_valid_source_config():
    """Valid source config passes validation."""

    config = {
        "path": "/tmp/test.csv",
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
    }

    errors = validate_source_config("csv", config)
    assert errors == []


def test_validator_rejects_missing_required_field():
    """Missing required field returns error."""

    config = {
        # Missing 'path' - required by CSVSourceConfig
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
    }

    errors = validate_source_config("csv", config)
    assert len(errors) == 1
    assert "path" in errors[0].field
    assert "required" in errors[0].message.lower()


def test_validator_rejects_invalid_field_type():
    """Invalid field type returns error."""

    config = {
        "path": "/tmp/test.csv",
        "skip_rows": "not_an_int",  # Should be int
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
    }

    errors = validate_source_config("csv", config)
    assert len(errors) == 1
    assert "skip_rows" in errors[0].field


def test_validator_accepts_null_source_with_empty_config():
    """null_source has no config class, validation should pass with empty dict."""

    config = {}

    errors = validate_source_config("null", config)
    assert errors == []


def test_validator_accepts_null_source_with_arbitrary_config():
    """null_source ignores config, validation should pass with any dict."""

    config = {
        "arbitrary_field": "ignored",
        "another_field": 42,
    }

    errors = validate_source_config("null", config)
    assert errors == []


def test_validator_accepts_valid_transform_config():
    """Valid transform config passes validation."""

    config = {
        "schema": {"mode": "observed"},
    }

    errors = validate_transform_config("passthrough", config)
    assert errors == []


def test_validator_accepts_valid_sink_config():
    """Valid sink config passes validation."""

    config = {
        "path": "/tmp/output.csv",
        "schema": {"mode": "observed"},
    }

    errors = validate_sink_config("csv", config)
    assert errors == []


def test_validator_rejects_invalid_transform_config():
    """Invalid transform config with missing required field returns error."""

    config = {
        # Missing schema - required by PassThroughConfig (DataPluginConfig)
    }

    errors = validate_transform_config("passthrough", config)
    assert len(errors) == 1
    # DataPluginConfig validator error reports the alias "schema", not field name "schema_config"
    assert errors[0].field == "schema"


def test_validator_rejects_invalid_sink_config():
    """Invalid sink config with missing required field returns error."""

    config = {
        # Missing 'url' - required by DatabaseSinkConfig
        "table": "test_table",
        "schema": {"mode": "observed"},
    }

    errors = validate_sink_config("database", config)
    assert len(errors) == 1
    assert "url" in errors[0].field
    assert "required" in errors[0].message.lower()


def test_validator_rejects_unknown_transform_type():
    """Unknown transform type raises ValueError."""

    config = {
        "schema": {"mode": "observed"},
    }

    with pytest.raises(ValueError) as exc_info:
        validate_transform_config("nonexistent_transform", config)

    assert "Unknown transform type" in str(exc_info.value)


def test_validator_rejects_unknown_sink_type():
    """Unknown sink type raises ValueError."""

    config = {
        "path": "/tmp/output.txt",
        "schema": {"mode": "observed"},
    }

    with pytest.raises(ValueError) as exc_info:
        validate_sink_config("nonexistent_sink", config)

    assert "Unknown sink type" in str(exc_info.value)


def test_validator_rejects_unknown_source_type():
    """Unknown source type raises ValueError."""

    config = {
        "path": "/tmp/input.txt",
        "schema": {"mode": "observed"},
    }

    with pytest.raises(ValueError) as exc_info:
        validate_source_config("nonexistent_source", config)

    assert "Unknown source type" in str(exc_info.value)


def test_validator_accepts_database_sink_config():
    """Valid database sink config passes validation."""

    config = {
        "url": "sqlite:///test.db",
        "table": "test_table",
        "schema": {"mode": "observed"},
    }

    errors = validate_sink_config("database", config)
    assert errors == []


def test_validator_accepts_azure_blob_source_config():
    """Valid Azure blob source config passes validation."""

    config = {
        "container": "test-container",
        "blob_path": "data/input.csv",
        "format": "csv",
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net",
    }

    errors = validate_source_config("azure_blob", config)
    assert errors == []


def test_validator_accepts_azure_blob_sink_config():
    """Valid Azure blob sink config passes validation."""

    config = {
        "container": "test-container",
        "blob_path": "output_{{ run_id }}.csv",
        "format": "csv",
        "schema": {"mode": "observed"},
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test==;EndpointSuffix=core.windows.net",
    }

    errors = validate_sink_config("azure_blob", config)
    assert errors == []


def test_validator_validates_schema_config():
    """Valid schema configs pass validation."""

    # Valid dynamic schema
    schema_config = {"mode": "observed"}
    errors = validate_schema_config(schema_config)
    assert errors == []

    # Valid explicit strict schema
    schema_config = {
        "mode": "fixed",
        "fields": ["id: int", "name: str", "score: float?"],
    }
    errors = validate_schema_config(schema_config)
    assert errors == []

    # Valid explicit free schema
    schema_config = {
        "mode": "flexible",
        "fields": ["id: int"],
    }
    errors = validate_schema_config(schema_config)
    assert errors == []


def test_validator_rejects_invalid_schema_mode():
    """Invalid schema mode returns error."""

    # Invalid mode value
    schema_config = {
        "mode": "invalid_mode",
        "fields": ["id: int"],
    }

    errors = validate_schema_config(schema_config)
    assert len(errors) > 0
    # Error should mention the invalid mode
    assert any("invalid_mode" in err.message.lower() or "mode" in err.message.lower() for err in errors)


def test_validator_rejects_schema_missing_fields():
    """Schema missing required 'fields' key returns error."""

    schema_config = {"mode": "fixed"}  # Missing 'fields'

    errors = validate_schema_config(schema_config)
    assert len(errors) > 0
    assert any("fields" in err.message.lower() for err in errors)


def test_validator_rejects_schema_empty_fields():
    """Schema with empty fields list returns error."""

    schema_config = {
        "mode": "fixed",
        "fields": [],  # Empty list
    }

    errors = validate_schema_config(schema_config)
    assert len(errors) > 0


def test_validator_rejects_schema_invalid_field_type():
    """Schema with wrong type for 'fields' returns error."""

    schema_config = {
        "mode": "fixed",
        "fields": "not_a_list",  # Should be list
    }

    errors = validate_schema_config(schema_config)
    assert len(errors) > 0


def test_validator_rejects_malformed_field_spec():
    """Schema with malformed field spec returns error."""

    schema_config = {
        "mode": "fixed",
        "fields": ["id-int"],  # Wrong format, should be "id: int"
    }

    errors = validate_schema_config(schema_config)
    assert len(errors) > 0


@pytest.mark.parametrize(
    ("validator_method", "plugin_type", "config"),
    [
        (
            "validate_source_config",
            "csv",
            {
                "path": "/tmp/test.csv",
                "schema": {"mode": "bad"},
                "on_validation_failure": "quarantine",
            },
        ),
        (
            "validate_transform_config",
            "passthrough",
            {
                "schema": {"mode": "bad"},
            },
        ),
        (
            "validate_sink_config",
            "csv",
            {
                "path": "/tmp/out.csv",
                "schema": {"mode": "bad"},
            },
        ),
    ],
)
def test_validator_returns_structured_error_for_invalid_plugin_schema(
    validator_method: str,
    plugin_type: str,
    config: dict[str, object],
) -> None:
    """Invalid embedded schema config should return ValidationError list, not raise."""
    dispatch = {
        "validate_source_config": validate_source_config,
        "validate_transform_config": validate_transform_config,
        "validate_sink_config": validate_sink_config,
    }
    validate = dispatch[validator_method]

    errors = validate(plugin_type, config)

    assert len(errors) > 0
    assert errors[0].field == "schema"
    assert errors[0].value == {"mode": "bad"}
    assert "mode" in errors[0].message.lower()


def test_validator_returns_structured_error_for_unknown_field():
    """Unknown fields are rejected by Pydantic's extra='forbid' config.

    Since normalize_fields was removed, passing it should trigger a
    validation error for the unknown field.
    """

    config = {
        "path": "/tmp/test.csv",
        "normalize_fields": True,
        "on_validation_failure": "quarantine",
        "schema": {"mode": "observed"},
    }

    errors = validate_source_config("csv", config)
    assert len(errors) > 0
