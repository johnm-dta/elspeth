# src/elspeth/plugins/sources/null_source.py
"""NullSource - a source that yields nothing.

Used by resume operations where row data comes from the payload store,
not from the original source. Satisfies PipelineConfig.source typing
while actual row data is retrieved separately.
"""

from collections.abc import Iterator
from typing import Any

from elspeth.contracts import Determinism, PluginSchema, SourceRow
from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext


class NullSourceSchema(PluginSchema):
    """Dynamic schema for NullSource - accepts any row structure.

    Since NullSource yields no rows, the schema never validates anything.
    This exists only to satisfy the SourceProtocol.output_schema requirement.
    """

    pass


class NullSource(BaseSource):
    """A source that yields no rows.

    Used during resume when row data comes from the payload store.
    The source slot in PipelineConfig must be filled, but the source
    is never actually called during resume - rows are retrieved from
    stored payloads instead.

    This source is deterministic (always yields nothing) and requires
    no configuration.
    """

    name = "null"
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
    output_schema: type[PluginSchema] = NullSourceSchema

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize NullSource.

        Args:
            config: Configuration dict (ignored - NullSource needs no config).
        """
        super().__init__(config)

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Yield no rows.

        Resume operations retrieve row data from the payload store,
        not from this source.

        Args:
            ctx: Plugin context (unused).

        Returns:
            Empty iterator.
        """
        return iter([])

    def close(self) -> None:
        """No resources to close."""
        pass
