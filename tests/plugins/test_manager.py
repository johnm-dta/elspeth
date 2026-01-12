# tests/plugins/test_manager.py
"""Tests for plugin manager."""


class TestPluginManager:
    """Plugin discovery and registration."""

    def test_create_manager(self) -> None:
        from elspeth.plugins.manager import PluginManager

        manager = PluginManager()
        assert manager is not None

    def test_register_plugin(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class InputSchema(PluginSchema):
            x: int

        class OutputSchema(PluginSchema):
            x: int
            y: int

        class MyTransform:
            name = "my_transform"
            input_schema = InputSchema
            output_schema = OutputSchema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success({**row, "y": row["x"] * 2})

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list:
                return [MyTransform]

        manager = PluginManager()
        manager.register(MyPlugin())

        transforms = manager.get_transforms()
        assert len(transforms) == 1
        assert transforms[0].name == "my_transform"

    def test_get_plugin_by_name(self) -> None:
        from elspeth.plugins.context import PluginContext
        from elspeth.plugins.hookspecs import hookimpl
        from elspeth.plugins.manager import PluginManager
        from elspeth.plugins.results import TransformResult
        from elspeth.plugins.schemas import PluginSchema

        class Schema(PluginSchema):
            x: int

        class TransformA:
            name = "transform_a"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class TransformB:
            name = "transform_b"
            input_schema = Schema
            output_schema = Schema

            def __init__(self, config: dict) -> None:
                pass

            def process(self, row: dict, ctx: PluginContext) -> TransformResult:
                return TransformResult.success(row)

            def on_register(self, ctx: PluginContext) -> None:
                pass

            def on_start(self, ctx: PluginContext) -> None:
                pass

            def on_complete(self, ctx: PluginContext) -> None:
                pass

        class MyPlugin:
            @hookimpl
            def elspeth_get_transforms(self) -> list:
                return [TransformA, TransformB]

        manager = PluginManager()
        manager.register(MyPlugin())

        transform = manager.get_transform_by_name("transform_b")
        assert transform is not None
        assert transform.name == "transform_b"

        missing = manager.get_transform_by_name("nonexistent")
        assert missing is None
