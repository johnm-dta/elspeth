"""PassThrough transform plugin.

Passes rows through unchanged. Useful for testing and debugging pipelines.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

import copy
from typing import Any

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class PassThroughConfig(TransformDataConfig):
    """Configuration for passthrough transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
    """


class PassThrough(BaseTransform):
    """Pass rows through unchanged.

    Use cases:
    - Testing pipeline wiring without modification
    - Debugging data flow (add logging in subclass)
    - Placeholder for future transform logic

    Config options:
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
    """

    name = "passthrough"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:2163447d28c7063d"
    config_model = PassThroughConfig

    # ADR-007: PassThrough emits a deep copy of the input row unchanged, so every
    # input field is present on every emitted row. Canonical pass-through exemplar.
    passes_through_input = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = PassThroughConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
        self.input_schema, self.output_schema = self._create_schemas(cfg.schema_config, "PassThrough")

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 §Clause 4 invariant harness."""
        return {"schema": {"mode": "observed"}}

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Return row unchanged (deep copy to prevent mutation).

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with unchanged row data

        Raises:
            PluginContractViolation: Raised by executor if row fails input schema
                validation. This indicates a bug in the upstream source/transform.
        """
        output_contract = self._align_output_contract(row.contract)
        return TransformResult.success(
            PipelineRow(copy.deepcopy(row.to_dict()), output_contract),
            success_reason={"action": "passthrough"},
        )

    def close(self) -> None:
        """No resources to release."""
        pass
