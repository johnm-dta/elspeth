"""Universal, fail-closed policy for web-visible plugins."""

from elspeth.web.plugin_policy.compiler import REQUIRED_WEB_PLUGIN_IDS, compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId, WebPluginPolicy
from elspeth.web.plugin_policy.profiles import RuntimeWebPluginConfig

__all__ = [
    "REQUIRED_WEB_PLUGIN_IDS",
    "PluginAvailabilitySnapshot",
    "PluginId",
    "RuntimeWebPluginConfig",
    "WebPluginPolicy",
    "compile_web_plugin_policy",
]
