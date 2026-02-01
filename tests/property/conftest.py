# tests/property/conftest.py
"""Shared Hypothesis strategies for property-based tests.

These strategies are extracted for reuse across property test modules.
They follow ELSPETH's data model and trust boundaries.

Strategy Categories:
- JSON-safe values (RFC 8785 compatible)
- Row-like data (transform input/output)
- External data simulation (messy headers, malformed input)
- Mutable structures (for isolation testing)

Usage:
    from tests.property.conftest import row_data, messy_headers

    @given(data=row_data)
    def test_transform_preserves_structure(data: dict) -> None:
        ...
"""

# =============================================================================
# Hypothesis Settings
# =============================================================================
#
# For standardized @settings decorators, import from tests.property.settings:
#   from tests.property.settings import STANDARD_SETTINGS, DETERMINISM_SETTINGS
#
# Tiers: DETERMINISM (500), STATE_MACHINE (200), STANDARD (100), SLOW (50), QUICK (20)
# =============================================================================

from __future__ import annotations

import keyword
from collections.abc import Iterator
from typing import Any

from hypothesis import strategies as st

from elspeth.contracts import ArtifactDescriptor, SourceRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.conftest import _TestSchema, _TestSinkBase, _TestSourceBase

# =============================================================================
# RFC 8785 / JSON Canonicalization Scheme Constraints
# =============================================================================

# RFC 8785 (JCS) uses JavaScript-safe integers: -(2^53-1) to (2^53-1)
# Values outside this range cause serialization precision issues
MAX_SAFE_INT = 2**53 - 1
MIN_SAFE_INT = -(2**53 - 1)


# =============================================================================
# Core JSON Strategies
# =============================================================================

# JSON-safe primitives (excluding NaN/Infinity which ELSPETH strictly rejects)
json_primitives = (
    st.none()
    | st.booleans()
    | st.integers(min_value=MIN_SAFE_INT, max_value=MAX_SAFE_INT)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=100)
)

# Recursive strategy for nested JSON structures (arrays and objects)
json_values = st.recursive(
    json_primitives,
    lambda children: (st.lists(children, max_size=10) | st.dictionaries(st.text(max_size=20), children, max_size=10)),
    max_leaves=50,
)

# Valid dict keys (strings only in JSON, non-empty for field names)
dict_keys = st.text(min_size=1, max_size=50)

# Row-like data (what transforms actually process)
# This matches the shape of data flowing through ELSPETH pipelines
row_data = st.dictionaries(
    keys=dict_keys,
    values=json_primitives,
    min_size=1,
    max_size=20,
)


# =============================================================================
# External Data Simulation (Tier 3 - Zero Trust)
# =============================================================================

# Messy headers that external systems might provide
# Includes unicode, special characters, whitespace, digits
# Filtered to ensure at least one alphanumeric (can normalize to something)
messy_headers = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),  # Letters, Numbers, Punctuation, Symbols, Spaces
        blacklist_categories=("Cc",),  # Exclude control characters
    ),
).filter(lambda s: any(c.isalnum() for c in s))

# Headers that will definitely normalize to valid identifiers
# More constrained than messy_headers for tests needing guaranteed success
normalizable_headers = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N")),  # Letters and numbers only
).filter(lambda s: s[0].isalpha() if s else False)  # Must start with letter

# Python keywords (for testing keyword collision handling)
python_keywords = st.sampled_from(list(keyword.kwlist))


# =============================================================================
# Mutable Nested Data (For Isolation Testing)
# =============================================================================

# Strategy that generates mutable nested structures
# Used to verify deepcopy isolation in fork_token and similar operations
mutable_nested_data = st.dictionaries(
    keys=st.text(
        min_size=1,
        max_size=10,
        alphabet=st.characters(whitelist_categories=("L",)),  # Letters only for clean keys
    ),
    values=st.one_of(
        st.integers(),
        st.lists(st.integers(), min_size=1, max_size=5),
        st.dictionaries(
            st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=("L",))),
            st.integers(),
            min_size=1,
            max_size=3,
        ),
    ),
    min_size=1,
    max_size=5,
)

# Deeply nested mutable data (stress test for deepcopy)
deeply_nested_data = st.recursive(
    st.integers(),
    lambda children: st.one_of(
        st.lists(children, min_size=1, max_size=3),
        st.dictionaries(
            st.text(min_size=1, max_size=5, alphabet=st.characters(whitelist_categories=("L",))),
            children,
            min_size=1,
            max_size=3,
        ),
    ),
    max_leaves=20,
)


# =============================================================================
# Binary Data (For Payload Store Testing)
# =============================================================================

# Arbitrary binary content for payload store tests
binary_content = st.binary(min_size=0, max_size=10_000)

# Non-empty binary content (most realistic for actual payloads)
nonempty_binary = st.binary(min_size=1, max_size=10_000)

# Small binary for fast tests
small_binary = st.binary(min_size=1, max_size=1000)


# =============================================================================
# Configuration Values (For Retry/Settings Testing)
# =============================================================================

# Valid retry attempt counts
valid_max_attempts = st.integers(min_value=1, max_value=100)

# Valid delay values (positive floats, reasonable bounds)
valid_delays = st.floats(min_value=0.001, max_value=3600.0, allow_nan=False, allow_infinity=False)

# Valid jitter values (non-negative)
valid_jitter = st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False)


# =============================================================================
# Branch Names (For Fork/Coalesce Testing)
# =============================================================================

# Valid branch names (non-empty strings, unique when used in lists)
branch_names = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
)

# List of unique branch names (for fork operations)
unique_branches = st.lists(branch_names, min_size=1, max_size=5, unique=True)

# Multiple branches (at least 2, for testing fork behavior)
multiple_branches = st.lists(branch_names, min_size=2, max_size=5, unique=True)


# =============================================================================
# ID and Name Strategies
# =============================================================================

# Valid ID strings (UUID-like hex strings)
id_strings = st.text(
    min_size=8,
    max_size=40,
    alphabet="0123456789abcdef",
)

# Sink/node names (lowercase with underscores)
sink_names = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_",
)

# Path/label names (for routing)
path_names = st.text(
    min_size=1,
    max_size=30,
    alphabet="abcdefghijklmnopqrstuvwxyz_0123456789",
).filter(lambda s: s[0].isalpha())


# =============================================================================
# Shared Test Fixtures (for integration/audit property tests)
# =============================================================================


class PropertyTestSchema(_TestSchema):
    """Schema for property tests - accepts any dict with dynamic fields."""

    pass


class ListSource(_TestSourceBase):
    """Source that emits rows from a provided list."""

    name = "property_list_source"
    output_schema = PropertyTestSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        for row in self._data:
            yield SourceRow.valid(row)

    def close(self) -> None:
        pass


class PassTransform(BaseTransform):
    """Transform that passes rows through unchanged."""

    name = "property_pass_transform"
    input_schema = PropertyTestSchema
    output_schema = PropertyTestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "passthrough"})


class ConditionalErrorTransform(BaseTransform):
    """Transform that errors on rows where 'fail' key is truthy.

    IMPORTANT: Uses direct key access per CLAUDE.md - if 'fail' key is
    missing, that's a test authoring bug that should crash.
    """

    name = "property_conditional_error"
    input_schema = PropertyTestSchema
    output_schema = PropertyTestSchema
    _on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"fields": "dynamic"}})

    def process(self, row: Any, ctx: Any) -> TransformResult:
        # Direct access - no defensive .get() per CLAUDE.md
        if row["fail"]:
            return TransformResult.error({"reason": "property_test_error"})
        return TransformResult.success(row, success_reason={"action": "test"})


class CollectSink(_TestSinkBase):
    """Sink that collects written rows in memory."""

    name = "property_collect_sink"

    def __init__(self, sink_name: str = "default") -> None:
        self.name = sink_name
        self.results: list[dict[str, Any]] = []

    def on_start(self, ctx: Any) -> None:
        self.results = []

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        self.results.extend(rows)
        return ArtifactDescriptor.for_file(
            path=f"memory://{self.name}",
            size_bytes=len(str(rows)),
            content_hash="test_hash",
        )

    def close(self) -> None:
        pass
