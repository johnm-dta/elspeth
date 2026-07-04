"""Report assembly transform plugin.

Collate a buffer of text rows into a single report row with pagination
metadata sourced from the ``AggregationBatchContext`` exposed by
``AggregationExecutor``. Designed for narrative outputs (paginated
markdown, plain text logs, HTML fragments) where one report covers one
flush window and the next flush emits a sibling report.
"""

from __future__ import annotations

import html
import re
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult

type ReportAssembleRow = dict[str, object]

_REPORT_METADATA_FIELDS = frozenset(
    {
        "report_format",
        "report_index",
        "line_start",
        "line_end",
        "line_count",
        "lines_seen_total",
        "flush_trigger",
        "is_end_of_source_report",
    }
)

# ``title`` and ``join_with`` are rendered into user-visible report output. A
# ``${VAR}`` placeholder there would be expanded from the host environment (on
# the operator CLI loader path) and copied into a downloadable artifact, so
# these presentation fields reject env-var references outright.
_ENV_VAR_REFERENCE_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}")


class ReportAssembleConfig(TransformDataConfig):
    """Configuration for the report_assemble transform."""

    text_field: str = Field(description="Name of the string field to collate")
    output_field: str = Field(default="report_body", description="Output field for assembled text")
    format: Literal["plain_text", "markdown", "html_fragment"] = Field(
        default="plain_text",
        description="Output rendering format: 'plain_text' for newline-joined text, 'markdown' for # title prefix, or 'html_fragment' for <h1>/<p> block elements (escaped).",
    )
    join_with: str = Field(default="\n", description="Separator used between input rows")
    title: str | None = Field(default=None, description="Optional report title")

    @field_validator("text_field", "output_field")
    @classmethod
    def _reject_empty(cls, value: str, info: Any) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be empty")
        return stripped

    @field_validator("output_field")
    @classmethod
    def _validate_output_identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError(f"output_field must be a valid Python identifier, got {value!r}")
        return value

    @field_validator("join_with", "title")
    @classmethod
    def _reject_env_ref_placeholders(cls, value: str | None, info: Any) -> str | None:
        if value is not None and _ENV_VAR_REFERENCE_PATTERN.search(value):
            raise ValueError(
                f"{info.field_name} must not contain environment-variable placeholders; "
                "report_assemble emits this value in user-visible report output"
            )
        return value

    @model_validator(mode="after")
    def _reject_output_field_collision(self) -> ReportAssembleConfig:
        # Detect at config time: process() builds the output dict with the
        # custom output_field first, then overwrites with the metadata keys,
        # so a colliding output_field would silently drop the report body.
        if self.output_field in _REPORT_METADATA_FIELDS:
            raise ValueError(
                f"output_field {self.output_field!r} collides with a reserved "
                f"report metadata field. Reserved names: {sorted(_REPORT_METADATA_FIELDS)!r}"
            )
        return self


class ReportAssemble(BaseTransform):
    """Assemble a paginated report from a flushed batch of text rows."""

    name = "report_assemble"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:b426d7331ebaacce"
    config_model = ReportAssembleConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Assembles a flushed batch of text rows into one paginated report row.",
                composer_hints=(
                    "Use report_assemble under aggregations with a trigger; it requires aggregation_batch context.",
                    "text_field must be a string field present on every input row.",
                    "Choose format as plain_text, markdown, or html_fragment; html_fragment escapes text before wrapping paragraphs.",
                    "output_field must not collide with reserved report metadata fields such as report_index or line_count.",
                    "Output is one report row per flush window, not the original text rows.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "text_field": "report_assemble_probe_value",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = ReportAssembleConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._text_field = cfg.text_field
        self._output_field = cfg.output_field
        self._format = cfg.format
        self._join_with = cfg.join_with
        self._title = cfg.title

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.add(cfg.text_field)
        if base_required != set(cfg.schema_config.required_fields or ()):
            schema_config = SchemaConfig(
                mode=cfg.schema_config.mode,
                fields=cfg.schema_config.fields,
                guaranteed_fields=cfg.schema_config.guaranteed_fields,
                audit_fields=cfg.schema_config.audit_fields,
                required_fields=tuple(base_required),
            )
        else:
            schema_config = cfg.schema_config

        self._schema_config = schema_config
        self.declared_output_fields = frozenset({cfg.output_field, *_REPORT_METADATA_FIELDS})

        self.input_schema, self.output_schema = self._create_schemas(
            schema_config,
            "ReportAssemble",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Reductive override: one report row replaces N input rows.

        Mirrors ``BatchStats._build_output_schema_config`` — the user's input
        schema declares what upstream produces, but report_assemble consumes
        those rows and emits a single derived row carrying only the output
        field plus pagination metadata. Propagating input field declarations
        would cause ``SchemaConfigModeViolation`` at runtime.
        """
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def _render_plain_text(self, lines: list[str]) -> str:
        body = self._join_with.join(lines)
        if self._title:
            return f"{self._title}\n\n{body}" if body else self._title
        return body

    def _render_markdown(self, lines: list[str]) -> str:
        body = self._join_with.join(lines)
        if self._title:
            return f"# {self._title}\n\n{body}" if body else f"# {self._title}"
        return body

    def _render_html_fragment(self, lines: list[str]) -> str:
        # Rule (consistent across all three renderers): self._join_with separates
        # body elements; the title-to-body separator is a per-format constant
        # (plain_text/markdown use "\n\n"; html_fragment uses "\n" because
        # block-level HTML elements don't require blank-line separation).
        join_with = html.escape(self._join_with)
        paragraphs = join_with.join(f"<p>{html.escape(line)}</p>" for line in lines)
        if self._title:
            return f"<h1>{html.escape(self._title)}</h1>\n{paragraphs}" if paragraphs else f"<h1>{html.escape(self._title)}</h1>"
        return paragraphs

    def _render(self, lines: list[str]) -> str:
        if self._format == "plain_text":
            return self._render_plain_text(lines)
        if self._format == "markdown":
            return self._render_markdown(lines)
        # Literal type guarantees the only remaining value is "html_fragment".
        return self._render_html_fragment(lines)

    def _output_contract_for(self, output: ReportAssembleRow) -> SchemaContract:
        """Build one shared output contract for the assembled report row."""
        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=object,
                required=False,
                source="inferred",
            )
            for key in output
        )
        output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        return self._align_output_contract(output_contract)

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        # Hypothesis-generated probe rows arrive without ``text_field``, so the
        # aggregate output path can't run on them as-is. Mirror the
        # BatchStats override: graft the configured text_field with a sample
        # value so the real ``process()`` aggregate path executes during the
        # ADR-009 backward invariant sweep.
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._text_field,
                value="probe text",
            )
        ]

    def process(  # type: ignore[override] # Batch signature: list[PipelineRow]
        self, rows: list[PipelineRow], ctx: TransformContext
    ) -> TransformResult:
        """Assemble one report row from a flushed batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)
        if ctx.aggregation_batch is None:
            raise RuntimeError("report_assemble must run as an aggregation node with aggregation_batch context")

        lines: list[str] = []
        for index, row in enumerate(rows):
            value = row[self._text_field]
            if type(value) is not str:
                raise TypeError(
                    f"Field {self._text_field!r} must be a string, got {type(value).__name__} in batch row {index}. "
                    "This indicates an upstream validation bug — check source schema or prior transforms."
                )
            lines.append(value)

        report_text = self._render(lines)
        batch = ctx.aggregation_batch
        output: ReportAssembleRow = {
            self._output_field: report_text,
            "report_format": self._format,
            "report_index": batch.flush_index,
            "line_start": batch.row_start,
            "line_end": batch.row_end,
            "line_count": batch.batch_size,
            "lines_seen_total": batch.rows_seen_total,
            "flush_trigger": batch.trigger_type,
            "is_end_of_source_report": batch.is_end_of_source,
        }
        output_contract = self._output_contract_for(output)
        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "assembled_report",
                "fields_added": sorted(output),
                "metadata": {
                    "batch_size": batch.batch_size,
                    "flush_trigger": batch.trigger_type,
                },
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
