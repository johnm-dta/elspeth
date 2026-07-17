"""Public accessors for composer tool schema contracts.

The registered tool declarations are the wire authority.  Consumers must
select from :func:`get_tool_definitions` rather than reaching into the
dispatch registry's private lookup tables.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from elspeth.web.composer.tools._dispatch import get_tool_definitions


def canonical_set_pipeline_schema() -> dict[str, Any]:
    """Return an isolated copy of the registered ``set_pipeline`` schema."""
    for definition in get_tool_definitions():
        if definition.get("name") == "set_pipeline":
            parameters = definition.get("parameters")
            if not isinstance(parameters, dict):  # pragma: no cover - registry integrity guard
                raise RuntimeError("registered set_pipeline parameters must be a JSON-schema object")
            return deepcopy(parameters)
    raise RuntimeError("registered set_pipeline tool definition is missing")


__all__ = ["canonical_set_pipeline_schema"]
