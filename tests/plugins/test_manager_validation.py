# tests/plugins/test_manager_validation.py
"""Tests for plugin manager attribute validation."""

import pytest

from elspeth.contracts import Determinism, NodeType
from elspeth.plugins.manager import PluginSpec


class TestPluginSpecValidation:
    """Tests for PluginSpec.from_plugin() validation."""

    def test_missing_name_raises(self) -> None:
        """Plugin without name attribute should raise ValueError."""

        class BadPlugin:
            plugin_version = "1.0.0"
            # Missing: name

        with pytest.raises(ValueError, match="must define 'name' attribute"):
            PluginSpec.from_plugin(BadPlugin, NodeType.TRANSFORM)

    def test_missing_version_raises(self) -> None:
        """Plugin without plugin_version should raise ValueError."""

        class BadPlugin:
            name = "bad"
            # Missing: plugin_version

        with pytest.raises(ValueError, match="must define 'plugin_version' attribute"):
            PluginSpec.from_plugin(BadPlugin, NodeType.TRANSFORM)

    def test_valid_plugin_succeeds(self) -> None:
        """Plugin with required attributes should succeed."""

        class GoodPlugin:
            name = "good"
            plugin_version = "1.0.0"

        spec = PluginSpec.from_plugin(GoodPlugin, NodeType.TRANSFORM)
        assert spec.name == "good"
        assert spec.version == "1.0.0"

    def test_determinism_defaults_to_deterministic(self) -> None:
        """Plugins without determinism should default to DETERMINISTIC."""

        class SimplePlugin:
            name = "simple"
            plugin_version = "1.0.0"
            # Missing: determinism (should default)

        spec = PluginSpec.from_plugin(SimplePlugin, NodeType.TRANSFORM)
        assert spec.determinism == Determinism.DETERMINISTIC

    def test_schemas_default_to_none(self) -> None:
        """Plugins without schemas should have None schema hashes."""

        class MinimalPlugin:
            name = "minimal"
            plugin_version = "1.0.0"
            # Missing: input_schema, output_schema (should default to None)

        spec = PluginSpec.from_plugin(MinimalPlugin, NodeType.TRANSFORM)
        assert spec.input_schema_hash is None
        assert spec.output_schema_hash is None

    def test_error_message_includes_class_name(self) -> None:
        """Error message should identify which plugin is missing attributes."""

        class MyBadlyNamedPlugin:
            plugin_version = "1.0.0"

        with pytest.raises(ValueError, match="MyBadlyNamedPlugin"):
            PluginSpec.from_plugin(MyBadlyNamedPlugin, NodeType.TRANSFORM)
