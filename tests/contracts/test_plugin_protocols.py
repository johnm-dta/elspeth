"""Test plugin protocol contracts and schema validation."""

import pytest


def test_source_validates_output_schema_on_init() -> None:
    """Sources should validate output schema during construction."""
    from elspeth.plugins.config_base import PluginConfigError
    from elspeth.plugins.sources.csv_source import CSVSource

    # Valid schema - should succeed
    config = {
        "path": "test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "discard",
    }
    CSVSource(config)  # Should not raise

    # Invalid schema - should fail during __init__
    bad_config = {
        "path": "test.csv",
        "schema": {"mode": "strict", "fields": ["invalid syntax"]},
        "on_validation_failure": "discard",
    }

    with pytest.raises(PluginConfigError, match="Invalid field spec"):
        CSVSource(bad_config)  # Fails during construction
