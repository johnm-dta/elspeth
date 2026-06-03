from elspeth.plugins.infrastructure.config_base import DataPluginConfig


class GoodConfig(DataPluginConfig):
    _plugin_component_type = "transform"
    path: str
