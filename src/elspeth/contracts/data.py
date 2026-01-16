"""Plugin data schema contracts.

PluginSchema is the base class for plugin input/output schemas.
Plugins declare their expected data shape by subclassing this.
"""

from pydantic import BaseModel


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Subclass to define the expected shape of data for a plugin:

        class MyInputSchema(PluginSchema):
            name: str
            value: int

    Uses Pydantic for validation - this is a trust boundary
    (user data entering the system).
    """

    model_config = {"frozen": True, "extra": "forbid"}
