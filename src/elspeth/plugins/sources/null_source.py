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
    no configuration. The schema for type restoration comes from the
    original run's audit trail, not from NullSource.
    """

    name = "null"
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
    output_schema: type[PluginSchema] = NullSourceSchema
    # NullSource yields no rows, so it never quarantines - but set to satisfy protocol
    _on_validation_failure: str = "discard"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize NullSource.

        Args:
            config: Configuration dict. NullSource requires no specific config,
                but BaseSource expects schema config. If not provided, defaults
                to dynamic schema since NullSource never validates rows anyway.
        """
        config_copy = dict(config)
        if "schema" not in config_copy:
            config_copy["schema"] = {"fields": "dynamic"}
        super().__init__(config_copy)
        # Set _schema_class to satisfy protocol, but resume will use stored schema from audit trail
        self._schema_class = NullSourceSchema

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
