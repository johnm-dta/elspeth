"""Built-in sink plugins for ELSPETH.

Sinks output data to destinations. Multiple sinks per run.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    sink = manager.get_sink_by_name("csv_sink")
"""
