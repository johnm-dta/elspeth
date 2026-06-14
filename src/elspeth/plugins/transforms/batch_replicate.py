"""Batch replicate transform plugin.

Demonstrates output_mode: transform deaggregation by replicating rows.
Each input row is replicated based on a 'copies' field, producing more
output rows than input rows (N inputs -> M outputs where M >= N).

IMPORTANT: This transform uses is_batch_aware=True, meaning the engine
will buffer rows and call process() with a list when the trigger fires.

For output_mode: transform, the engine creates NEW tokens for each output
row, with parent linkage to track deaggregation lineage.
"""

import copy
from typing import Any

from pydantic import Field, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import PluginContractViolation, TransformSuccessReason
from elspeth.contracts.field_collision import detect_field_collisions
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import PluginConfigError, TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class BatchReplicateConfig(TransformDataConfig):
    """Configuration for batch replicate transform.

    Requires a field that specifies how many copies of each row to produce.
    """

    copies_field: str = Field(
        default="copies",
        description="Name of the field containing the number of copies to make",
        min_length=1,
    )
    default_copies: int = Field(
        default=1,
        ge=1,
        le=10000,
        description="Default number of copies if copies_field is missing or invalid",
    )
    max_copies: int = Field(
        default=10000,
        ge=1,
        le=10000,
        description="Upper bound on copies per row to prevent unbounded replication",
    )
    include_copy_index: bool = Field(
        default=True,
        description="Whether to add a 'copy_index' field (0-based) to each output row",
    )

    @model_validator(mode="after")
    def _default_within_max(self) -> "BatchReplicateConfig":
        if self.default_copies > self.max_copies:
            msg = f"default_copies ({self.default_copies}) exceeds max_copies ({self.max_copies})"
            raise ValueError(msg)
        return self


class BatchReplicate(BaseTransform):
    """Replicate rows based on a copies field.

    This is a batch-aware transform that demonstrates output_mode: transform
    deaggregation. It receives N input rows and produces M output rows where
    M is the sum of all copies values.

    Example: If input has 3 rows with copies=2,3,1 respectively:
    - Input: 3 rows
    - Output: 6 rows (2 + 3 + 1)
    - Each output row has copy_index showing which copy it is

    Config options:
        schema: Required. Schema for input validation
        copies_field: Field name for copy count (default: "copies")
        default_copies: Fallback if field missing (default: 1)
        include_copy_index: Add copy_index field (default: True)

    Example YAML:
        aggregations:
          - name: replicate_batch
            plugin: batch_replicate
            trigger:
              count: 5
            output_mode: transform  # CRITICAL: Creates new tokens for outputs
            options:
              schema:
                mode: observed
              copies_field: quantity
              default_copies: 1
    """

    name = "batch_replicate"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:3e9b08173c506ade"
    config_model = BatchReplicateConfig
    is_batch_aware = True  # CRITICAL: Engine buffers rows for batch processing

    # Mixed-validity batches can quarantine some input rows (e.g. copies < 1)
    # while still succeeding for the rest of the batch. That makes the
    # unconditional pass-through contract dishonest even though every emitted
    # row deep-copies the originating input before adding copy_index.
    passes_through_input = False

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Replicates rows within an aggregation batch based on a copy-count field.",
                composer_hints=(
                    "Use batch_replicate under aggregations with output_mode=transform so emitted copies become new tokens.",
                    "copies_field must be an int when present; missing values use default_copies and values outside 1..max_copies are quarantined.",
                    "include_copy_index=True adds copy_index, so avoid input fields with that name or disable it.",
                    "This expands row count and can drop invalid source rows from the successful output.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 §Clause 4 invariant harness."""
        return {
            "schema": {"mode": "observed"},
            "copies_field": "copies",
            "default_copies": 1,
            "max_copies": 10,
            "include_copy_index": True,
        }

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Drive the backward invariant with a mixed-validity batch.

        ``BatchReplicate`` is non-pass-through because one row can be
        quarantined while another still emits successfully. A singleton probe
        row only exercises the happy path. We therefore synthesize a two-row
        batch:

        - row 0: invalid copies + a unique field that exists nowhere else
        - row 1: valid copies so the transform still returns success

        The harness can then observe that the success rows do not preserve the
        unique invalid-row field, proving the class-level
        ``passes_through_input = False`` declaration is honest.
        """
        unique_invalid_field = "quarantined_only_marker"

        invalid_payload = probe.to_dict()
        invalid_payload[self._copies_field] = 0
        invalid_payload[unique_invalid_field] = "invalid-branch"
        invalid_contract = probe.contract.with_field(
            normalized=unique_invalid_field,
            original=unique_invalid_field,
            value=invalid_payload[unique_invalid_field],
        )

        valid_payload = probe.to_dict()
        valid_payload[self._copies_field] = 1

        return [
            PipelineRow(invalid_payload, invalid_contract),
            PipelineRow(valid_payload, probe.contract),
        ]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchReplicateConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        # Declare output fields for centralized collision detection.
        self.declared_output_fields = frozenset(["copy_index"] if cfg.include_copy_index else [])
        self._copies_field = cfg.copies_field
        self._default_copies = cfg.default_copies
        self._max_copies = cfg.max_copies
        self._include_copy_index = cfg.include_copy_index

        self._schema_config = cfg.schema_config
        self._reject_explicit_copy_index_collision(cfg)

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "BatchReplicate",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)

    def _reject_explicit_copy_index_collision(self, cfg: BatchReplicateConfig) -> None:
        """Reject explicit schemas that would always collide with copy_index emission."""
        if not cfg.include_copy_index or cfg.schema_config.fields is None:
            return

        declared_schema_fields = {field.name for field in cfg.schema_config.fields}
        collisions = detect_field_collisions(declared_schema_fields, self.declared_output_fields)
        if collisions is None:
            return

        cause = (
            "BatchReplicate schema declares field(s) "
            f"{collisions!r}, but include_copy_index=True would overwrite them. "
            "Remove the colliding field from the explicit schema or disable include_copy_index."
        )
        raise PluginConfigError(
            cause,
            cause=cause,
            plugin_class=self.config_model.__name__,
            plugin_name=self.name,
            component_type="transform",
        )

    def process(  # type: ignore[override] # Batch signature: list[PipelineRow] instead of PipelineRow
        self, rows: list[PipelineRow], ctx: TransformContext
    ) -> TransformResult:
        """Replicate each row based on its copies field.

        Args:
            rows: List of input rows (batch-aware receives list[PipelineRow])
            ctx: Plugin context

        Returns:
            TransformResult.success_multi() with replicated rows
        """
        if not rows:
            # Empty batch is an anomaly — return error, not fabricated data.
            # A synthetic PipelineRow({batch_empty: True}) would flow through
            # the pipeline as real data, corrupting the audit trail.
            return TransformResult.error(
                {"reason": "empty_batch"},
                retryable=False,
            )

        valid_rows: list[dict[str, Any]] = []
        emitted_contracts: list[SchemaContract] = []
        quarantined: list[dict[str, Any]] = []
        quarantined_indices: list[int] = []

        for row_index, row in enumerate(rows):
            # Get copies count - field is optional, type must be correct if present
            if self._copies_field not in row:
                # Field missing - use default, still bounded by max_copies
                copies = min(self._default_copies, self._max_copies)
            else:
                raw_copies = row[self._copies_field]

                # Contract enforcement: copies_field must be int if present
                # Tier 2 pipeline data - wrong types indicate upstream bug
                # Use `type(x) is int` instead of `isinstance(x, int)` because
                # bool is a subclass of int in Python, so isinstance(True, int)
                # returns True. We must reject bool values explicitly.
                if type(raw_copies) is not int:
                    raise TypeError(
                        f"Field '{self._copies_field}' must be int, got {type(raw_copies).__name__}. "
                        f"This indicates an upstream validation bug - check source schema or prior transforms."
                    )

                # Value-level validation: copies must be >= 1 and <= max_copies
                # Tier 2 operation safety - type is correct but value is unsafe
                if raw_copies < 1 or raw_copies > self._max_copies:
                    # Record the row INDEX for traceability, never the row body:
                    # row.to_dict() is Tier-2/3 content and must not leak into the
                    # audit success_reason (prior row-content-leak bug family).
                    quarantined.append(
                        {
                            "reason": "invalid_copies",
                            "field": self._copies_field,
                            "value": raw_copies,
                            "row_index": row_index,
                        }
                    )
                    quarantined_indices.append(row_index)
                    continue

                copies = raw_copies

            if self.declared_output_fields:
                collisions = detect_field_collisions(set(row.keys()), self.declared_output_fields)
                if collisions is not None:
                    raise PluginContractViolation(
                        f"Transform '{self.name}' would overwrite existing input fields "
                        f"{collisions}. This is a pipeline configuration error — the transform's "
                        f"output fields collide with fields already present in the row."
                    )

            # Create copies of this row
            emitted_contracts.append(row.contract)
            for copy_idx in range(copies):
                # Deep copy ensures each replica is fully independent —
                # shallow copy would share nested mutable values across copies
                output = copy.deepcopy(row.to_dict())
                if self._include_copy_index:
                    output["copy_index"] = copy_idx
                valid_rows.append(output)

        # If ALL rows were quarantined, return error — no valid output to expand
        if not valid_rows:
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "error": f"All {len(quarantined)} rows quarantined: invalid copies values",
                    "row_errors": [{"row_index": q["row_index"], "reason": q["reason"]} for q in quarantined],
                },
                retryable=False,
            )

        # Build the emitted contract from rows that actually produced output.
        # Quarantined-only fields must not leak into successful child tokens.
        first_contract = rows[0].contract
        for i, row in enumerate(rows[1:], start=1):
            if row.contract.mode != first_contract.mode:
                raise ValueError(
                    f"Heterogeneous contract modes in batch: row 0 has mode "
                    f"'{first_contract.mode}', row {i} has mode '{row.contract.mode}'. "
                    f"All rows in a batch must share the same contract mode."
                )
        merged_fields: dict[str, FieldContract] = {}
        for contract in emitted_contracts:
            for fc in contract.fields:
                if fc.normalized_name not in merged_fields:
                    merged_fields[fc.normalized_name] = fc

        # Add copy_index as a new inferred field if configured
        if self._include_copy_index:
            merged_fields["copy_index"] = FieldContract(
                normalized_name="copy_index",
                original_name="copy_index",
                python_type=int,
                required=False,
                source="inferred",
            )

        output_contract = SchemaContract(
            mode=first_contract.mode,
            fields=tuple(merged_fields.values()),
            locked=True,
        )
        output_contract = self._align_output_contract(output_contract)

        success_reason: TransformSuccessReason = {
            "action": "processed",
            "fields_added": ["copy_index"] if self._include_copy_index else [],
        }
        if quarantined:
            success_reason["metadata"] = {
                "quarantined_count": len(quarantined),
                "quarantined": quarantined,
                "quarantined_indices": quarantined_indices,
            }

        # Return only valid replicated rows — quarantined rows are in success_reason
        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in valid_rows],
            success_reason=success_reason,
        )

    def close(self) -> None:
        """No resources to release."""
        pass
