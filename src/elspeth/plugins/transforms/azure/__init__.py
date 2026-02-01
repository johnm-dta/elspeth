"""Azure transform plugins.

Provides Azure-specific transforms for content safety and prompt shielding.

Plugins are accessed via PluginManager, not direct imports:
    manager = PluginManager()
    manager.register_builtin_plugins()
    transform = manager.get_transform_by_name("azure_content_safety")
"""
