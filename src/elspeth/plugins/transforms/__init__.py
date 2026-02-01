"""Built-in transform plugins for ELSPETH.

Transforms process rows in the pipeline. Each transform receives a row
and returns a TransformResult indicating success/failure and output data.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    transform = manager.get_transform_by_name("field_mapper")
"""
