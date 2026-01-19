# tests/plugins/test_manager_validation.py
"""Tests for plugin manager attribute validation.

Note: Tests for missing required attributes (name, plugin_version, determinism)
have been removed. With type[PluginProtocol], mypy enforces these at compile time.
Plugins are system-owned code, not user extensions, so runtime validation is
unnecessary - a missing attribute would be caught during development.
"""

from elspeth.contracts import Determinism, NodeType
from elspeth.plugins.manager import PluginSpec


class TestPluginSpecValidation:
    """Tests for PluginSpec.from_plugin() validation."""

    def test_valid_plugin_succeeds(self) -> None:
        """Plugin with required attributes should succeed."""

        class GoodPlugin:
            name = "good"
            plugin_version = "1.0.0"
            determinism = Determinism.DETERMINISTIC

        spec = PluginSpec.from_plugin(GoodPlugin, NodeType.TRANSFORM)
        assert spec.name == "good"
        assert spec.version == "1.0.0"
        assert spec.determinism == Determinism.DETERMINISTIC

    def test_schemas_default_to_none(self) -> None:
        """Plugins without schemas should have None schema hashes."""

        class MinimalPlugin:
            name = "minimal"
            plugin_version = "1.0.0"
            determinism = Determinism.DETERMINISTIC
            # input_schema and output_schema are optional, default to None

        spec = PluginSpec.from_plugin(MinimalPlugin, NodeType.TRANSFORM)
        assert spec.input_schema_hash is None
        assert spec.output_schema_hash is None
