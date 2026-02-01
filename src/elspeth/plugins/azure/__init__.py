"""Azure plugin pack for ELSPETH.

Provides sources and sinks for Azure Blob Storage integration.
Supports multiple authentication methods:
- Connection string
- Managed Identity (for Azure-hosted workloads)
- Service Principal (for automated/CI scenarios)

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    source = manager.get_source_by_name("azure_blob")
    sink = manager.get_sink_by_name("azure_blob")
"""
