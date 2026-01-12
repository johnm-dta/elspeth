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
from elspeth.plugins.results import (
    AcceptResult,
    GateResult,
    RoutingAction,
    RowOutcome,
    TransformResult,
)

# Context
from elspeth.plugins.context import PluginContext

# Schemas
from elspeth.plugins.schemas import (
    CompatibilityResult,
    PluginSchema,
    SchemaValidationError,
    check_compatibility,
    validate_row,
)

# Protocols
from elspeth.plugins.protocols import (
    AggregationProtocol,
    CoalescePolicy,
    CoalesceProtocol,
    GateProtocol,
    SinkProtocol,
    SourceProtocol,
    TransformProtocol,
)

# Base classes
from elspeth.plugins.base import (
    BaseAggregation,
    BaseGate,
    BaseSink,
    BaseSource,
    BaseTransform,
)

# Manager
from elspeth.plugins.manager import PluginManager, PluginSpec

# Hookspecs
from elspeth.plugins.hookspecs import hookimpl, hookspec

# Enums
from elspeth.plugins.enums import (
    Determinism,
    NodeType,
    RoutingKind,
    RoutingMode,
)

__all__ = [
    # Results
    "AcceptResult",
    "GateResult",
    "RoutingAction",
    "RowOutcome",
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
    "AggregationProtocol",
    "CoalescePolicy",
    "CoalesceProtocol",
    "GateProtocol",
    "SinkProtocol",
    "SourceProtocol",
    "TransformProtocol",
    # Base classes
    "BaseAggregation",
    "BaseGate",
    "BaseSink",
    "BaseSource",
    "BaseTransform",
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
