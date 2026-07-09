"""Coalesce schema materialization — SchemaConfig → PluginSchema factory.

Synthesizes the Pydantic model class for a union-merge coalesce node so
PHASE 2 edge validation can type-check downstream edges. Extracted from
graph.py (elspeth-b2c6ab6db8): schema materialization is contract policy,
not graph topology.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

from pydantic import ConfigDict, create_model

from elspeth.contracts import PluginSchema
from elspeth.contracts.schema import FIELD_TYPE_MAP, SchemaConfig
from elspeth.core.dag.models import GraphValidationError

_coalesce_schema_counter = itertools.count(1)


def build_coalesce_schema(
    schema_config: SchemaConfig,
    *,
    coalesce_id: str | None = None,
) -> type[PluginSchema]:
    """Build a PluginSchema class from a coalesce SchemaConfig.

    Used by get_effective_producer_schema() to enable PHASE 2 type validation
    on union-merge coalesce edges.  The SchemaConfig is set by the builder's
    ``_assign_schema()`` on the coalesce node.

    Args:
        schema_config: The schema configuration for the coalesce node.
        coalesce_id: The node ID of the coalesce node, for error attribution.

    Raises:
        GraphValidationError: If the schema config has no fields (indicates
            a builder bug — observed schemas should not reach this path).
    """
    counter = next(_coalesce_schema_counter)

    if schema_config.fields is None:
        raise GraphValidationError(
            f"Coalesce union schema has no fields (mode={schema_config.mode!r}). "
            "Observed schemas should be filtered before calling build_coalesce_schema.",
            component_id=coalesce_id,
            component_type="coalesce",
        )

    field_definitions: dict[str, Any] = {}
    for fd in schema_config.fields:
        python_type = FIELD_TYPE_MAP[fd.field_type]
        # Pydantic type must include None if the field can ever be None at runtime.
        # This happens when:
        #   - fd.nullable=True: field explicitly allows None values (e.g., from
        #     coalesce where a nullable branch can win via last_wins collision)
        #   - fd.required=False: field may be absent (Pydantic defaults absent to None)
        can_be_none = fd.nullable or not fd.required
        field_type = python_type | None if can_be_none else python_type
        if fd.required:
            field_definitions[fd.name] = (field_type, ...)
        else:
            field_definitions[fd.name] = (field_type, None)

    extra_mode: Literal["allow", "ignore", "forbid"] = "allow" if schema_config.mode == "flexible" else "forbid"

    return create_model(
        f"_CoalesceUnionSchema_{counter}",
        __base__=PluginSchema,
        __module__=__name__,
        __config__=ConfigDict(extra=extra_mode, strict=True),
        **field_definitions,
    )
