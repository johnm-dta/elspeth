# tests/plugins/test_manager.py
"""Tests for plugin manager."""

from typing import Any

from elspeth.testing import make_pipeline_row


class TestPluginManager:
    """Plugin discovery and registration."""

    def test_create_manager(self) -> None:
        from elspeth.plugins.infrastructure.manager import PluginManager

        manager = PluginManager()
        assert manager is not None

    def test_register_plugin(self) -> None:
        from elspeth.contracts import PluginSchema
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.plugins.infrastructure.results import TransformResult

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
                return TransformResult.success(make_pipeline_row({**row, "y": row["x"] * 2}), success_reason={"action": "test"})

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
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.plugins.infrastructure.results import TransformResult

        class Schema(PluginSchema):
            x: int

        class TransformA:
            name = "transform_a"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict[str, Any]) -> None:
                pass

            def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
                return TransformResult.success(make_pipeline_row(row), success_reason={"action": "test"})

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
                return TransformResult.success(make_pipeline_row(row), success_reason={"action": "test"})

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

        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager

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

    def test_duplicate_registration_does_not_pollute_pluggy_state(self) -> None:
        """After a duplicate registration fails, the manager must be in a clean state.

        Regression: P1-2026-02-14 — register() added the plugin to pluggy before
        duplicate detection in _refresh_caches(). When _refresh_caches() raised
        ValueError, the plugin remained in pluggy, poisoning all future registrations.
        """
        import pytest

        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager

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

        class Plugin3:
            @hookimpl
            def elspeth_get_transforms(self) -> list[type]:
                class T3:
                    name = "unique_name"

                return [T3]

        manager = PluginManager()
        manager.register(Plugin1())

        # Duplicate registration should fail
        with pytest.raises(ValueError, match="duplicate_name"):
            manager.register(Plugin2())

        # After failure, manager must still be clean:
        # 1. Only the original Plugin1 transform should exist
        transforms = manager.get_transforms()
        assert len(transforms) == 1
        assert transforms[0].name == "duplicate_name"

        # 2. Registering a distinct plugin should succeed
        manager.register(Plugin3())
        transforms = manager.get_transforms()
        names = sorted(t.name for t in transforms)
        assert names == ["duplicate_name", "unique_name"]

    def test_same_name_different_types_ok(self) -> None:
        """Same name in different plugin types is allowed."""
        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager

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
        """Verify PluginManager raises PluginNotFoundError for unknown source plugins."""
        import pytest

        from elspeth.plugins.infrastructure.manager import (
            PluginManager,
            PluginNotFoundError,
        )

        manager = PluginManager()

        # Try to get plugin that doesn't exist
        with pytest.raises(PluginNotFoundError, match="Unknown source plugin: nonexistent"):
            manager.get_source_by_name("nonexistent")

    def test_get_transform_by_name_raises_on_unknown_plugin(self) -> None:
        """Verify PluginManager raises PluginNotFoundError for unknown transform plugins."""
        import pytest

        from elspeth.plugins.infrastructure.manager import (
            PluginManager,
            PluginNotFoundError,
        )

        manager = PluginManager()

        with pytest.raises(PluginNotFoundError, match="Unknown transform plugin: nonexistent"):
            manager.get_transform_by_name("nonexistent")

    def test_get_sink_by_name_raises_on_unknown_plugin(self) -> None:
        """Verify PluginManager raises PluginNotFoundError for unknown sink plugins."""
        import pytest

        from elspeth.plugins.infrastructure.manager import (
            PluginManager,
            PluginNotFoundError,
        )

        manager = PluginManager()

        with pytest.raises(PluginNotFoundError, match="Unknown sink plugin: nonexistent"):
            manager.get_sink_by_name("nonexistent")


class TestHookValidation:
    """Validate plugin hook implementations at registration time."""

    def test_unknown_hook_raises_and_does_not_pollute_caches(self) -> None:
        """Misspelled hook names must crash immediately and leave caches unchanged."""
        import pluggy
        import pytest

        from elspeth.plugins.infrastructure.hookspecs import hookimpl
        from elspeth.plugins.infrastructure.manager import PluginManager

        class TypoPlugin:
            @hookimpl
            def elspeth_get_tranforms(self) -> list[type]:  # pragma: no cover - typo under test
                return []

        manager = PluginManager()

        with pytest.raises(pluggy.PluginValidationError, match="unknown hook 'elspeth_get_tranforms'"):
            manager.register(TypoPlugin())

        assert manager.get_sources() == []
        assert manager.get_transforms() == []
        assert manager.get_sinks() == []


class TestDiscoveryBasedRegistration:
    """Test PluginManager with automatic discovery."""

    def test_register_builtin_discovers_csv_source(self) -> None:
        """Verify register_builtin_plugins finds CSVSource via discovery."""
        from elspeth.plugins.infrastructure.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        source = manager.get_source_by_name("csv")
        assert source is not None
        assert source.name == "csv"

    def test_register_builtin_discovers_all_transforms(self) -> None:
        """Verify register_builtin_plugins finds all transforms."""
        from elspeth.plugins.infrastructure.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        transforms = manager.get_transforms()
        names = [t.name for t in transforms]

        assert "passthrough" in names
        assert "field_mapper" in names

    def test_register_builtin_discovers_all_sinks(self) -> None:
        """Verify register_builtin_plugins finds all sinks."""
        from elspeth.plugins.infrastructure.manager import PluginManager

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

        from elspeth.plugins.infrastructure.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        # Invalid config - missing required 'path'
        invalid_config = {
            "schema": {"mode": "observed"},
            "on_validation_failure": "quarantine",
        }

        with pytest.raises(ValueError) as exc_info:
            manager.create_source("csv", invalid_config)

        # Error message should mention the field name
        assert "path" in str(exc_info.value)

    def test_manager_creates_plugin_with_valid_config(self) -> None:
        """Valid config creates working plugin."""
        from elspeth.plugins.infrastructure.manager import PluginManager

        manager = PluginManager()
        manager.register_builtin_plugins()

        valid_config = {
            "path": "/tmp/test.csv",
            "schema": {"mode": "observed"},
            "on_validation_failure": "quarantine",
        }

        source = manager.create_source("csv", valid_config)

        # Verify plugin is functional
        assert source.name == "csv"
        assert source.output_schema is not None
        assert hasattr(source, "load")
