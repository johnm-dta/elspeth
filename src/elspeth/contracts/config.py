"""Configuration contracts.

Configuration types use Pydantic (not dataclasses) because they validate
user-provided YAML - a legitimate trust boundary per Data Manifesto.

These are re-exports from core/config.py for import consistency.
The actual definitions stay in core/config.py where Pydantic validation logic lives.
"""

from elspeth.core.config import (
    CheckpointSettings,
    ConcurrencySettings,
    DatabaseSettings,
    DatasourceSettings,
    ElspethSettings,
    LandscapeExportSettings,
    LandscapeSettings,
    PayloadStoreSettings,
    RateLimitSettings,
    RetrySettings,
    RowPluginSettings,
    SinkSettings,
)

__all__ = [
    "CheckpointSettings",
    "ConcurrencySettings",
    "DatabaseSettings",
    "DatasourceSettings",
    "ElspethSettings",
    "LandscapeExportSettings",
    "LandscapeSettings",
    "PayloadStoreSettings",
    "RateLimitSettings",
    "RetrySettings",
    "RowPluginSettings",
    "SinkSettings",
]
