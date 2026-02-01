# src/elspeth/core/canonical.py
"""
Canonical JSON serialization for deterministic hashing.

Two-phase approach:
1. Normalize: Convert pandas/numpy types to JSON-safe primitives (our code)
2. Serialize: Produce deterministic JSON per RFC 8785/JCS (rfc8785 package)

IMPORTANT: NaN and Infinity are strictly REJECTED, not silently converted.
This is defense-in-depth for audit integrity.

NOTE: For non-canonical data that cannot be serialized (malformed external
data at Tier-3 trust boundary), use repr_hash() as a fallback. This is NOT
deterministic across Python versions but is appropriate for quarantined data
where the content is already flagged as problematic.
"""

from __future__ import annotations

import base64
import hashlib
import math
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import rfc8785

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph

# Version string stored with every run for hash verification
CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states for float AND Decimal
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Check for NaN/Infinity FIRST (before type coercion)
    if isinstance(obj, float | np.floating):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Cannot canonicalize non-finite float: {obj}. Use None for missing values, not NaN.")
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Primitives pass through unchanged
    if obj is None or isinstance(obj, str | int | bool):
        return obj

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        # BUG-CANON-01 fix: Reject NaN/Infinity in arrays
        # Multi-dimensional arrays need element-wise validation
        if obj.size > 0:  # Only check non-empty arrays
            try:
                # np.any() works on all dtypes, returns False for non-numeric
                if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
                    raise ValueError(
                        "NaN/Infinity found in NumPy array. Audit trail requires finite values only. Use None for missing values, not NaN."
                    )
            except TypeError:
                # np.isnan/isinf raise TypeError for non-numeric dtypes (e.g., strings)
                # This is expected and safe - non-numeric arrays can't contain NaN/Inf
                pass
        return [_normalize_value(x) for x in obj.tolist()]

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        # Naive timestamps assumed UTC (explicit policy)
        if obj.tz is None:
            return obj.tz_localize("UTC").isoformat()
        return obj.tz_convert("UTC").isoformat()

    # Intentional missing values (NOT NaN - that's rejected above)
    if obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
        return None

    # Standard library types
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=UTC)
        return obj.astimezone(UTC).isoformat()

    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, Decimal):
        if not obj.is_finite():  # Rejects NaN, sNaN, Infinity, -Infinity
            raise ValueError(f"Cannot canonicalize non-finite Decimal: {obj}. Use None for missing values, not NaN/Infinity.")
        return str(obj)

    return obj


def _normalize_for_canonical(data: Any) -> Any:
    """Recursively normalize a data structure for canonical JSON.

    Converts pandas/numpy types to JSON-safe primitives.

    Args:
        data: Any data structure (dict, list, primitive)

    Returns:
        Normalized data structure with only JSON-safe types

    Raises:
        ValueError: If data contains NaN, Infinity, or other non-serializable values
    """
    if isinstance(data, dict):
        return {k: _normalize_for_canonical(v) for k, v in data.items()}
    if isinstance(data, list | tuple):
        return [_normalize_for_canonical(v) for v in data]
    return _normalize_value(data)


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON for hashing.

    Two-phase approach:
    1. Normalize pandas/numpy types to JSON-safe primitives (our code)
    2. Serialize per RFC 8785/JCS standard (rfc8785 package)

    Args:
        obj: Data structure to serialize

    Returns:
        Canonical JSON string (no whitespace, sorted keys)

    Raises:
        ValueError: If data contains NaN, Infinity, or other non-finite values
        TypeError: If data contains types that cannot be serialized
    """
    normalized = _normalize_for_canonical(obj)
    result: bytes = rfc8785.dumps(normalized)
    return result.decode("utf-8")


def stable_hash(obj: Any, version: str = CANONICAL_VERSION) -> str:
    """Compute stable hash of object.

    Args:
        obj: Data structure to hash
        version: Hash algorithm version (stored with runs for verification)

    Returns:
        SHA-256 hex digest of canonical JSON
    """
    canonical = canonical_json(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_full_topology_hash(graph: ExecutionGraph) -> str:
    """Compute hash of complete DAG topology for checkpoint validation.

    Unlike upstream-only hashing, this ensures ANY topology change
    (including sibling branches in multi-sink DAGs) invalidates checkpoint resume.

    This enforces the audit integrity invariant: one run_id = one configuration.
    A single run cannot contain outputs produced under different pipeline configs.

    Args:
        graph: Execution graph to hash

    Returns:
        SHA-256 hash of canonical full topology representation.
    """
    nx_graph = graph.get_nx_graph()

    topology_data = {
        "nodes": sorted(
            [
                {
                    "node_id": n,
                    "plugin_name": graph.get_node_info(n).plugin_name,
                    "config_hash": stable_hash(graph.get_node_info(n).config),
                }
                for n in nx_graph.nodes()
            ],
            key=lambda x: x["node_id"],
        ),
        "edges": sorted(
            [_edge_to_canonical_dict(nx_graph, u, v, k) for u, v, k in nx_graph.edges(keys=True)],
            key=lambda x: (x["from"], x["to"], x["key"]),
        ),
    }

    return stable_hash(topology_data)


def _edge_to_canonical_dict(
    graph: nx.MultiDiGraph[Any],
    u: str,
    v: str,
    k: str,
) -> dict[str, Any]:
    """Convert edge to canonical dict for hashing.

    Uses explicit defaults for missing attributes to ensure
    hash stability across graphs with inconsistent edge data.

    Args:
        graph: NetworkX graph containing the edge
        u: Source node ID
        v: Target node ID
        k: Edge key

    Returns:
        Canonical dict representation of edge attributes.
    """
    edge_data = graph.edges[u, v, k]
    # Edge data is Tier 1 (Our Data) - crash on missing/wrong attributes
    # If label or mode are missing/wrong, that's a bug in ExecutionGraph.add_edge()
    return {
        "from": u,
        "to": v,
        "key": k,
        "label": edge_data["label"],
        "mode": edge_data["mode"].value,
    }


def repr_hash(obj: Any) -> str:
    """Generate SHA-256 hash of repr() for non-canonical data.

    Used as fallback when canonical_json fails (NaN, Infinity, or other
    non-serializable types). This provides deterministic hashing within
    the same Python version, but is NOT guaranteed to be stable across
    different Python versions due to repr() implementation differences.

    This is appropriate for Tier-3 (external data) trust boundary where
    data is already malformed and being quarantined.

    Args:
        obj: Any Python object

    Returns:
        SHA-256 hex digest of repr(obj)

    Example:
        >>> repr_hash(42)
        '73475cb40a568e8da8a045ced110137e159f890ac4da883b6b17dc651b3a8049'
        >>> repr_hash({"value": float("nan")})  # Non-canonical data
        '49ce040dd7d56208...'
    """
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()
