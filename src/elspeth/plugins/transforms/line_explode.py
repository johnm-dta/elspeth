"""LineExplode deaggregation transform.

Transforms one row containing a string field into multiple rows, one for each
line in that string. This is a deterministic 1-to-N expansion for scraped HTML,
plain text files, and other text payloads that need line-oriented downstream
processing.

THREE-TIER TRUST MODEL COMPLIANCE:

Per the plugin protocol, transforms trust that pipeline field types are correct:
- Sources and earlier transforms validate required fields and their declared types
- A non-string source field is an upstream validation bug and should crash
- Empty strings are valid string values but cannot be deaggregated into rows, so
  they are returned as non-retryable data errors
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.contract_propagation import narrow_contract_to_output
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult

if TYPE_CHECKING:
    from elspeth.contracts.plugin_assistance import PluginAssistance
    from elspeth.contracts.plugin_semantics import InputSemanticRequirements


class LineExplodeConfig(TransformDataConfig):
    """Configuration for line-oriented string deaggregation."""

    source_field: str = Field(..., description="Name of the string field to split into lines")
    output_field: str = Field(default="line", description="Name for each emitted line")
    include_index: bool = Field(default=True, description="Whether to include a line index field")
    index_field: str = Field(default="line_index", description="Name for the emitted line index")

    @field_validator("source_field", "output_field", "index_field")
    @classmethod
    def _reject_empty_field_names(cls, v: str, info: Any) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must be non-empty")
        return stripped

    @field_validator("output_field", "index_field")
    @classmethod
    def _validate_output_identifiers(cls, v: str, info: Any) -> str:
        if not v.isidentifier():
            raise ValueError(f"{info.field_name} must be a valid Python identifier, got {v!r}")
        return v

    @model_validator(mode="after")
    def _reject_field_collisions(self) -> LineExplodeConfig:
        if self.output_field == self.source_field:
            raise ValueError(f"output_field and source_field must differ, both are '{self.output_field}'")
        if self.include_index:
            if self.index_field == self.source_field:
                raise ValueError(f"index_field and source_field must differ, both are '{self.index_field}'")
            if self.index_field == self.output_field:
                raise ValueError(f"index_field and output_field must differ, both are '{self.index_field}'")
        return self

    @model_validator(mode="after")
    def _require_statically_resolvable_source_field_when_output_contract_depends_on_it(self) -> LineExplodeConfig:
        """Fail closed when static output-contract derivation cannot resolve aliases."""
        static_output_inputs: set[str] = set(self.schema_config.guaranteed_fields or ())
        if self.schema_config.fields is not None:
            static_output_inputs.update(field.name for field in self.schema_config.fields)

        if static_output_inputs and self.source_field not in static_output_inputs:
            raise ValueError(
                "source_field must use the normalized field name when schema declares "
                "guaranteed_fields or explicit fields for output-contract propagation. "
                f"Got {self.source_field!r}; known normalized fields are {sorted(static_output_inputs)!r}."
            )

        return self


def _line_explode_added_output_fields(
    *,
    output_field: str,
    include_index: bool,
    index_field: str,
) -> tuple[FieldDefinition, ...]:
    """Return the typed fields LineExplode guarantees on successful output rows."""
    fields = [
        FieldDefinition(name=output_field, field_type="str", required=True),
    ]
    if include_index:
        fields.append(FieldDefinition(name=index_field, field_type="int", required=True))
    return tuple(fields)


def _build_line_explode_output_schema_config(cfg: LineExplodeConfig) -> SchemaConfig:
    """Build output schema config excluding the consumed source field."""
    field_by_name: dict[str, FieldDefinition] = {}
    if cfg.schema_config.fields is not None:
        field_by_name.update((field.name, field) for field in cfg.schema_config.fields if field.name != cfg.source_field)

    added_fields = _line_explode_added_output_fields(
        output_field=cfg.output_field,
        include_index=cfg.include_index,
        index_field=cfg.index_field,
    )
    field_by_name.update((field.name, field) for field in added_fields)

    base_guaranteed = set(cfg.schema_config.guaranteed_fields or ())
    base_guaranteed.discard(cfg.source_field)
    output_guaranteed = base_guaranteed | {field.name for field in added_fields}

    return SchemaConfig(
        mode=cfg.schema_config.mode if cfg.schema_config.fields is not None else "flexible",
        fields=tuple(field_by_name.values()),
        guaranteed_fields=tuple(sorted(output_guaranteed)),
        audit_fields=cfg.schema_config.audit_fields,
        required_fields=cfg.schema_config.required_fields,
    )


def _build_line_explode_input_requirements(
    *,
    source_field: str,
) -> InputSemanticRequirements:
    from elspeth.contracts.plugin_semantics import (
        ContentKind,
        FieldSemanticRequirement,
        InputSemanticRequirements,
        TextFraming,
        UnknownSemanticPolicy,
    )

    return InputSemanticRequirements(
        fields=(
            FieldSemanticRequirement(
                field_name=source_field,
                accepted_content_kinds=frozenset(
                    {ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN},
                ),
                accepted_text_framings=frozenset(
                    {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE},
                ),
                requirement_code="line_explode.source_field.line_framed_text",
                severity="high",
                unknown_policy=UnknownSemanticPolicy.FAIL,
                configured_by=("source_field",),
            ),
        ),
    )


class LineExplode(BaseTransform):
    """Explode a string field into one output row per line."""

    name = "line_explode"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:e426cd6ca548e3e8"
    config_model = LineExplodeConfig
    creates_tokens = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "source_field": "line_explode_text",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = LineExplodeConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

        self._source_field = cfg.source_field
        self._output_field = cfg.output_field
        self._include_index = cfg.include_index
        self._index_field = cfg.index_field

        fields = [cfg.output_field]
        if cfg.include_index:
            fields.append(cfg.index_field)
        self.declared_output_fields = frozenset(fields)

        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "LineExplode",
            adds_fields=True,
        )
        self._output_schema_config = _build_line_explode_output_schema_config(cfg)

    def input_semantic_requirements(self) -> InputSemanticRequirements:
        return _build_line_explode_input_requirements(
            source_field=self._source_field,
        )

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> PluginAssistance | None:
        from elspeth.contracts.plugin_assistance import PluginAssistance

        if issue_code != "line_explode.source_field.line_framed_text":
            return None
        return PluginAssistance(
            plugin_name="line_explode",
            issue_code="line_explode.source_field.line_framed_text",
            summary=(
                "line_explode calls splitlines() on the configured source_field. "
                "Compact text (a single string with no newlines) emits one row "
                "containing the whole content — the opposite of line "
                "deaggregation. The producer must emit newline-framed or "
                "line-compatible text."
            ),
            suggested_fixes=(
                "Configure the upstream producer to emit newline-framed text "
                "(for web_scrape: text_separator: '\\n', or use format: markdown).",
                "Choose a producer whose output_semantics() declares a text_framing of newline_framed or line_compatible.",
            ),
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the real source-field consumption path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._source_field,
                value="only-line",
            )
        ]

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Explode a string field into multiple rows."""
        source_value = row[self._source_field]
        if type(source_value) is not str:
            raise TypeError(
                f"Field '{self._source_field}' must be a string, got {type(source_value).__name__}. "
                "This indicates an upstream validation bug - check source schema or prior transforms."
            )

        lines = source_value.splitlines()
        if len(lines) == 0:
            return TransformResult.error(
                {"reason": "invalid_input", "field": self._source_field, "error": "empty string"},
                retryable=False,
            )

        row_data = row.to_dict()
        if self._source_field in row_data:
            normalized_source_field = self._source_field
        else:
            normalized_source_field = row.contract.resolve_name(self._source_field)
        base = {k: v for k, v in row_data.items() if k != normalized_source_field}

        output_rows: list[dict[str, Any]] = []
        for i, line in enumerate(lines):
            output = copy.deepcopy(base)
            output[self._output_field] = line
            if self._include_index:
                output[self._index_field] = i
            output_rows.append(output)

        first_keys = set(output_rows[0].keys())
        for i, output_row in enumerate(output_rows[1:], start=1):
            row_keys = set(output_row.keys())
            if row_keys != first_keys:
                raise ValueError(
                    f"Multi-row output has heterogeneous schema: "
                    f"row 0 has fields {sorted(first_keys)}, "
                    f"row {i} has fields {sorted(row_keys)}"
                )

        output_contract = narrow_contract_to_output(
            input_contract=row.contract,
            output_row=output_rows[0],
        )
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)

        fields_added = [self._output_field]
        if self._include_index:
            fields_added.append(self._index_field)

        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in output_rows],
            success_reason={
                "action": "transformed",
                "fields_added": fields_added,
                "fields_removed": [self._source_field],
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
