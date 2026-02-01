"""Built-in source plugins for ELSPETH.

Sources load data into the pipeline. Exactly one source per run.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    source = manager.get_source_by_name("csv_source")
"""
