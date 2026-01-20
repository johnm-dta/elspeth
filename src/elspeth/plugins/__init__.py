# src/elspeth/plugins/__init__.py
"""Plugin system: Sources, Transforms, Sinks via pluggy.

This module provides the plugin infrastructure for Elspeth:

- Protocols: Type contracts for plugin implementations
- Base classes: Convenient base classes with lifecycle hooks
- Results: Return types for plugin operations
- Schemas: Pydantic-based input/output schemas
- Manager: Plugin discovery and registration
- Hookspecs: pluggy hook definitions

Phase 3 Integration:
- PluginContext carries Landscape, Tracer, PayloadStore
- Result types include audit fields (hashes, duration)
- Base classes have lifecycle hooks for engine integration
"""

# Results
# Base classes
# Enums (re-exported from contracts as part of public plugin API)
# Schemas (canonical location: elspeth.contracts)
from elspeth.contracts import (
    CompatibilityResult,
    Determinism,
    NodeType,
    PluginSchema,
    RoutingKind,
    RoutingMode,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)
from elspeth.plugins.base import (
    BaseGate,
    BaseSink,
    BaseSource,
    BaseTransform,
)

# Config base classes
from elspeth.plugins.config_base import (
    DataPluginConfig,
    PathConfig,
    PluginConfig,
    PluginConfigError,
    SourceDataConfig,
    TransformDataConfig,
)

# Context
from elspeth.plugins.context import PluginContext

# Hookspecs
from elspeth.plugins.hookspecs import hookimpl, hookspec

# Manager
from elspeth.plugins.manager import PluginManager, PluginSpec

# Protocols
from elspeth.plugins.protocols import (
    CoalescePolicy,
    CoalesceProtocol,
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)
from elspeth.plugins.results import (
    GateResult,
    RoutingAction,
    RowOutcome,
    SourceRow,
    TransformResult,
)

__all__ = [  # - grouped by category for readability
    # Results (NOTE: AcceptResult deleted in aggregation structural cleanup)
    "GateResult",
    "RoutingAction",
    "RowOutcome",
    "SourceRow",
    "TransformResult",
    # Context
    "PluginContext",
    # Schemas
    "CompatibilityResult",
    "PluginSchema",
    "SchemaValidationError",
    "check_compatibility",
    "validate_row",
    # Protocols
    "CoalescePolicy",
    "CoalesceProtocol",
    "GateProtocol",
    "SinkProtocol",
    "SourceProtocol",
    "TransformProtocol",
    # Base classes
    "BaseGate",
    "BaseSink",
    "BaseSource",
    "BaseTransform",
    # Config base classes
    "DataPluginConfig",
    "PathConfig",
    "PluginConfig",
    "PluginConfigError",
    "SourceDataConfig",
    "TransformDataConfig",
    # Manager
    "PluginManager",
    "PluginSpec",
    # Hookspecs
    "hookimpl",
    "hookspec",
    # Enums
    "Determinism",
    "NodeType",
    "RoutingKind",
    "RoutingMode",
]
