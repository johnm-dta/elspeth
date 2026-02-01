# tests/plugins/test_manager.py
"""Tests for plugin manager."""

from typing import Any


class TestPluginManager:
    """Plugin discovery and registration."""

    def test_create_manager(self) -> None:
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        assert manager is not None

    def test_register_plugin(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            y: int

        class MyTransform:
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "y": row["x"] * 2}, success_reason={"action": "test"})

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list[type[MyTransform]]:
                return [MyTransform]

        manager = PluginManager()
        manager.register(MyPlugin())

        transforms = manager.get_transforms()
        assert len(transforms) == 1
        assert transforms[0].name == "my_transform"

    def test_get_plugin_by_name(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult

        class Schema(PluginSchema):
            x: int

        class TransformA:
            name = "transform_a"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class TransformB:
            name = "transform_b"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(
                self,
            ) -> list[type[TransformA] | type[TransformB]]:
                return [TransformA, TransformB]

        manager = PluginManager()
        manager.register(MyPlugin())

        transform = manager.get_transform_by_name("transform_b")
        assert transform.name == "transform_b"


class TestDuplicateNameValidation:
    """Prevent duplicate plugin names."""

    def test_duplicate_transform_raises(self) -> None:
        import pytest

        from elspeth.plugins import PluginManager, hookimpl

        class Plugin1:
            @hookimpl
            def elspeth_get_transforms(self) -> list[type]:
                class T1:
                    name = "duplicate_name"

                return [T1]

        class Plugin2:
            @hookimpl
            def elspeth_get_transforms(self) -> list[type]:
                class T2:
                    name = "duplicate_name"

                return [T2]

        manager = PluginManager()
        manager.register(Plugin1())

        with pytest.raises(ValueError, match="duplicate_name"):
            manager.register(Plugin2())

    def test_same_name_different_types_ok(self) -> None:
        """Same name in different plugin types is allowed."""
        from elspeth.plugins import PluginManager, hookimpl

        class Plugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list[type]:
                class T:
                    name = "processor"

                return [T]

            @hookimpl
            def elspeth_get_sinks(self) -> list[type]:
                class S:
                    name = "processor"  # Same name, different type

                return [S]

        manager = PluginManager()
        manager.register(Plugin())  # Should not raise


class TestMissingPluginRaises:
    """Verify PluginManager raises on unknown plugins."""

    def test_get_source_by_name_raises_on_unknown_plugin(self) -> None:
        """Verify PluginManager raises ValueError for unknown source plugins."""
        import pytest

        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()

        # Try to get plugin that doesn't exist
        with pytest.raises(ValueError, match="Unknown source plugin: nonexistent"):
            manager.get_source_by_name("nonexistent")

    def test_get_transform_by_name_raises_on_unknown_plugin(self) -> None:
        """Verify PluginManager raises ValueError for unknown transform plugins."""
        import pytest

        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()

        with pytest.raises(ValueError, match="Unknown transform plugin: nonexistent"):
            manager.get_transform_by_name("nonexistent")

    def test_get_sink_by_name_raises_on_unknown_plugin(self) -> None:
        """Verify PluginManager raises ValueError for unknown sink plugins."""
        import pytest

        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()

        with pytest.raises(ValueError, match="Unknown sink plugin: nonexistent"):
            manager.get_sink_by_name("nonexistent")


class TestDiscoveryBasedRegistration:
    """Test PluginManager with automatic discovery."""

    def test_register_builtin_discovers_csv_source(self) -> None:
        """Verify register_builtin_plugins finds CSVSource via discovery."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        source = manager.get_source_by_name("csv")
        assert source is not None
        assert source.name == "csv"

    def test_register_builtin_discovers_all_transforms(self) -> None:
        """Verify register_builtin_plugins finds all transforms."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        transforms = manager.get_transforms()
        names = [t.name for t in transforms]

        assert "passthrough" in names
        assert "field_mapper" in names

    def test_register_builtin_discovers_all_sinks(self) -> None:
        """Verify register_builtin_plugins finds all sinks."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        sinks = manager.get_sinks()
        names = [s.name for s in sinks]

        assert "csv" in names
        assert "json" in names


class TestManagerValidation:
    """PluginManager validates configs before instantiation."""

    def test_manager_validates_before_instantiation(self) -> None:
        """Invalid config raises ValueError with field name."""
        import pytest

        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        # Invalid config - missing required 'path'
        invalid_config = {
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(ValueError) as exc_info:
            manager.create_source("csv", invalid_config)

        # Error message should mention the field name
        assert "path" in str(exc_info.value)

    def test_manager_creates_plugin_with_valid_config(self) -> None:
        """Valid config creates working plugin."""
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        valid_config = {
            "path": "/tmp/test.csv",
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "quarantine",
        }

        source = manager.create_source("csv", valid_config)

        # Verify plugin is functional
        assert source.name == "csv"
        assert source.output_schema is not None
        assert hasattr(source, "load")
